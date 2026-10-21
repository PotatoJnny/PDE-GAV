#!/usr/bin/env python
"""Bounded, explicit continuous-fit comparison; uses the original NS projector.

Only generator train/validation measurement files are read. Spatial holdout is
an offline diagnostic, not a new production reconstruction or fitting backend.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'legacy/ns/src'), str(ROOT / 'scripts')]
import numpy as np
import torch
from prepare_ns_reconstructed_generator import file_sha256, write_json, summary, relative_errors
from pde_agent_verifier_solver.plugins.ns.completed_fields import completed_grid, field_from_coefficients, numerical_symbols
from pde_agent_verifier_solver.plugins.ns.experimental_mms_project import _path, _coordinates, _window_clock
from agent_verifier.neural_generator.structured_ns_analytic import analytic_quantities


def build_plan(manifest_path, geometry_path, out, windows=24, candidate_configs=None):
    manifest = json.loads(manifest_path.read_text())
    geometry = json.loads(geometry_path.read_text())
    selections = {}
    for role in ('generator_training', 'generator_validation'):
        rows = manifest['splits'][role]
        indices = np.linspace(0, len(rows)-1, min(windows, len(rows)), dtype=int)
        chosen = [rows[i] for i in indices]
        for a, b in zip(chosen, chosen[1:]):
            if a['source_recording'] == b['source_recording'] and a['source_frames'][1] > b['source_frames'][0]:
                raise ValueError('diagnostic windows overlap; reduce --windows')
        selections[role] = chosen
    candidates = []
    for k, extension, ridge in itertools.product((16, 18), (.15, .25), (1e-6, 1e-7)):
        cfg = numerical_symbols().StructuredNSProjectionConfig(spatial_mode_x=k, spatial_mode_y=k,
            time_order=12, spatial_extension=extension, ridge_relative=ridge).to_dict()
        candidates.append({'id': f'k{k}_e{extension:g}_r{ridge:g}', 'projection_config': cfg})
    baseline_id = 'k16_e0.25_r1e-06'
    if candidate_configs is not None:
        candidates = json.loads(Path(candidate_configs).read_text())
        for candidate in candidates:
            cfg = numerical_symbols().StructuredNSProjectionConfig.from_dict(candidate['projection_config'])
            cfg.validate()
            if max(cfg.spatial_mode_x, cfg.spatial_mode_y) >= 20:
                raise ValueError('this study preserves spectral_max_mode=20; candidates must have K<20')
        baselines = [row for row in candidates if row['id'] == baseline_id]
        expected = numerical_symbols().StructuredNSProjectionConfig(spatial_mode_x=16, spatial_mode_y=16, time_order=12, spatial_extension=.25, ridge_relative=1e-6).to_dict()
        if len(baselines) != 1 or baselines[0]['projection_config'] != expected or len({c['id'] for c in candidates}) != len(candidates):
            raise ValueError('custom candidates require unique IDs and the unchanged baseline configuration')
    candidates.sort(key=lambda c: c['id'] != baseline_id)
    return {'schema': 'ns_measured_representation_study_v1', 'manifest': str(manifest_path),
        'manifest_sha256': file_sha256(manifest_path), 'geometry': geometry,
        'selections': selections, 'candidates': candidates, 'baseline_id': baseline_id,
        'viscosity': 1.0006053268765131e-6,
        'spatial_holdout': '(ix+2*iy)%5==0 on interior sites; edges retained; identical in every case',
        'selection': {'score': 'mean validation heldout velocity relative L2',
            'eligibility': 'validation full-grid and heldout u/v errors each <= baseline; derivative RMS <= 2x baseline; laplacian <= 3x baseline',
            'tie': 'within 1% of best score, prefer lower K then stronger ridge',
            'diagnostic_limits_are_not_physical_truth': True},
        'excluded_roles': ['development', 'verifier', 'final_test', 'reconstruction_validation']}


def read_windows(plan):
    manifest_path = Path(plan['manifest'])
    clock = _window_clock(json.loads(manifest_path.read_text()))
    coordinates, data = None, {}
    for role, rows in plan['selections'].items():
        arrays = []
        for row in rows:
            source = _path(row['measurement_reference_file'], manifest_path.parent)
            with np.load(source, allow_pickle=False) as f:
                x, y, times = _coordinates(f, clock=clock, source_frames=row['source_frames'])
                left, right, bottom, top = plan['geometry']['domain_bounds']
                keep = (x >= left) & (x <= right) & (y >= bottom) & (y <= top)
                actual = (x[keep], y[keep], times)
                if coordinates is None:
                    coordinates = actual
                elif any(a.shape != b.shape or not np.allclose(a, b, rtol=1e-8, atol=1e-10)
                         for a, b in zip(actual, coordinates)):
                    raise ValueError('measurement coordinates/clocks differ')
                velocity = np.asarray(f['velocity'], dtype=np.float64)[..., keep]
                if not np.isfinite(velocity).all():
                    raise ValueError('nonfinite measurement')
                arrays.append(velocity)
        data[role] = np.stack(arrays)
    return data, coordinates


def error_report(predicted, target):
    return {'velocity': summary(relative_errors(predicted, target)),
            'u': summary(relative_errors(predicted[:, :, 0], target[:, :, 0])),
            'v': summary(relative_errors(predicted[:, :, 1], target[:, :, 1]))}


def evaluate(projector, coefficients):
    values = torch.as_tensor(coefficients, dtype=torch.float64)
    psi = values[:, 0].reshape(len(values), projector.config.time_coefficient_count, -1)
    temporal = torch.einsum('tr,nrs->nts', projector.temporal_design, psi)
    return torch.einsum('fs,nts->ntf', projector.spatial_operators['velocity_design'], temporal).reshape(
        len(values), len(projector.times), 2, -1).detach().numpy()


def derivatives(coefficients, config, plan, times, grid):
    # Mid-cell queries test between measurements; fixed edge/corner queries also
    # detect boundary derivative growth when extension or regularization changes.
    xs = (grid.x[:-1] + grid.x[1:]) / 2
    ys = (grid.y[:-1] + grid.y[1:]) / 2
    ts = float(times[0]) + np.array([0., .17, .47, .83, 1.]) * float(times[-1]-times[0])
    interior = np.stack(np.meshgrid(xs[::4], ys[::3], indexing='ij'), -1).reshape(-1, 2)
    edge_x = np.unique(np.r_[grid.x[::4], grid.x[-1]])
    edge_y = np.unique(np.r_[grid.y[::3], grid.y[-1]])
    boundary = np.unique(np.concatenate((
        np.column_stack((edge_x, np.full(len(edge_x), grid.y[0]))),
        np.column_stack((edge_x, np.full(len(edge_x), grid.y[-1]))),
        np.column_stack((np.full(len(edge_y), grid.x[0]), edge_y)),
        np.column_stack((np.full(len(edge_y), grid.x[-1]), edge_y)),
    )), axis=0)
    spatial = np.concatenate((interior, boundary))
    queries = torch.tensor(np.column_stack((np.repeat(spatial, len(ts), axis=0),
                                           np.tile(ts, len(spatial)))), dtype=torch.float64)
    groups = {'gradient': ('d_x_u', 'd_y_u', 'd_x_v', 'd_y_v'),
              'time_derivative': ('d_t_u', 'd_t_v'), 'laplacian': ('laplacian_u', 'laplacian_v'),
              'forcing': ('forcing_u', 'forcing_v'), 'velocity': ('u', 'v')}
    collected = {name: [] for name in groups}
    divergence = []
    case_indices = np.unique(np.linspace(0, len(coefficients)-1, 4, dtype=int))
    for index in case_indices:
        field = field_from_coefficients(coefficients[index], config=config,
            geometry=grid.continuous_geometry, time_interval=(float(times[0]), float(times[-1])),
            viscosity=plan['viscosity'])
        q = analytic_quantities(field, queries, uniform_time=False)
        for name, keys in groups.items():
            collected[name].append(np.stack([q[k].detach().numpy() for k in keys]))
        divergence.append(q['divergence'].detach().numpy())
    report = {name: {'rms': float(np.sqrt(np.mean(np.asarray(rows)**2))),
                     'p99_absolute': float(np.quantile(np.abs(rows), .99)),
                     'maximum_absolute': float(np.max(np.abs(rows)))} for name, rows in collected.items()}
    report['divergence_maximum_absolute'] = float(np.max(np.abs(divergence)))
    report['query_points_per_case'] = len(queries)
    report['boundary_spatial_points'] = len(boundary)
    report['interior_spatial_points'] = len(interior)
    report['query_times'] = ts.tolist()
    report['case_indices'] = case_indices.tolist()
    return report


def run(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan = build_plan(Path(args.manifest).resolve(), Path(args.geometry_json).resolve(), out, args.windows, args.candidate_configs)
    write_json(out/'plan.json', plan)  # frozen before measurement arrays are opened
    data, (x, y, times) = read_windows(plan)
    grid, (ix, iy) = completed_grid(x, y, plan['geometry'])
    if not grid.fluid_mask.all():
        raise ValueError('this comparison expects a complete measured rectangle')
    # Convert native paired-point order to the original projector's sorted grid.
    sorted_data = {}
    for role, values in data.items():
        target = np.zeros((*values.shape[:3], len(grid.x), len(grid.y)))
        target[..., ix, iy] = values
        sorted_data[role] = target
    xx, yy = np.indices(grid.fluid_mask.shape)
    holdout = ((xx + 2*yy) % 5 == 0) & (xx > 0) & (xx < len(grid.x)-1) & (yy > 0) & (yy < len(grid.y)-1)
    fit_grid = replace(grid, fluid_mask=grid.fluid_mask & ~holdout)
    module = numerical_symbols()
    results = []
    for candidate in plan['candidates']:
        started = time.time()
        config = module.StructuredNSProjectionConfig.from_dict(candidate['projection_config'])
        projector = module.StructuredNSCoefficientProjector(geometry=grid, t=times, config=config)
        row = dict(candidate)
        for role, target in sorted_data.items():
            c, report = projector.project(velocity=target, pressure=np.zeros((len(target), len(times), *grid.fluid_mask.shape)))
            pred = evaluate(projector, c)
            row[role] = error_report(pred, target.reshape(*target.shape[:3], -1))
            if role == 'generator_validation':
                row['derivatives'] = derivatives(c.detach().numpy(), config.to_dict(), plan, times, grid)
        held_projector = module.StructuredNSCoefficientProjector(geometry=fit_grid, t=times, config=config)
        target = sorted_data['generator_validation']
        held_c, _ = held_projector.project(velocity=target, pressure=np.zeros((len(target), len(times), *grid.fluid_mask.shape)))
        held_pred = evaluate(projector, held_c)[..., holdout.ravel()]
        row['validation_spatial_holdout'] = error_report(held_pred, target[..., holdout])
        row['holdout_points'] = int(holdout.sum())
        row['operator_diagnostics'] = projector.operator_diagnostics
        row['elapsed_seconds'] = time.time()-started
        results.append(row)
        write_json(out/'results.partial.json', results)
        print(json.dumps({'candidate': row['id'], 'full': row['generator_validation']['velocity']['mean'],
            'heldout': row['validation_spatial_holdout']['velocity']['mean'],
            'v': row['generator_validation']['v']['mean'], 'derivatives': row['derivatives'], 'seconds': row['elapsed_seconds']}), flush=True)
    baseline = results[0]
    for row in results:
        acceptable = all(row[split][comp]['mean'] <= baseline[split][comp]['mean'] * (1+1e-10)
            for split in ('generator_validation', 'validation_spatial_holdout') for comp in ('u','v'))
        for key, limit in [('gradient',2),('time_derivative',2),('forcing',2),('laplacian',3)]:
            ratio = row['derivatives'][key]['rms']/baseline['derivatives'][key]['rms']
            row['derivatives'][key]['ratio_to_baseline'] = ratio
            acceptable &= ratio <= limit
        row['eligible'] = bool(acceptable)
    eligible = [row for row in results if row['eligible']]
    best = min(row['validation_spatial_holdout']['velocity']['mean'] for row in eligible)
    tied = [row for row in eligible if row['validation_spatial_holdout']['velocity']['mean'] <= best*1.01]
    selected = min(tied, key=lambda row: (row['projection_config']['spatial_mode_x'], -row['projection_config']['ridge_relative'], row['validation_spatial_holdout']['velocity']['mean']))
    write_json(out/'results.json', {'plan_sha256': file_sha256(out/'plan.json'), 'candidates': results,
        'selected_id': selected['id'], 'selected_projection_config': selected['projection_config'],
        'scope': 'configuration selection on generator validation; not independent final quality evidence'})
    print(json.dumps({'selected_id': selected['id'], 'projection_config': selected['projection_config']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--geometry-json', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--windows', type=int, default=24)
    p.add_argument('--candidate-configs', help='Explicit JSON candidate list, including unchanged baseline')
    run(p.parse_args())
