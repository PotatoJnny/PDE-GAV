#!/usr/bin/env python
"""Adaptive training-only capacity diagnostic; never selects a Generator.

Freeze the exact six configurations and inherited windows before opening arrays.
The original joint objective, solvers, field evaluator and query protocol are reused.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'legacy/ns/src'), str(ROOT/'scripts')]
import numpy as np
import torch
from study_ns_measured_representation import read_windows, error_report, derivatives
from prepare_ns_reconstructed_generator import write_json, file_sha256
from pde_agent_verifier_solver.plugins.ns.completed_fields import completed_grid
from agent_verifier.neural_generator.structured_ns_answer_first import StructuredNSProjectionConfig
from agent_verifier.neural_generator.structured_ns_spline_projection import JointSmoothNSCoefficientProjector


def make_masks(shape, count, frames):
    xx, yy = np.indices(shape)
    spatial = (((xx+2*yy) % 5 == 0) & (xx > 0) & (xx < shape[0]-1)
               & (yy > 0) & (yy < shape[1]-1)).ravel()
    temporal = np.zeros(count, dtype=bool)
    temporal[frames] = True
    if not spatial.any() or not temporal.any() or temporal[[0, -1]].any():
        raise ValueError('holdout must include interior sites and whole interior frames')
    return spatial, temporal


def cv_metrics(prediction, target, sh, th):
    spatial = error_report(prediction[:, ~th, :, :][:, :, :, sh], target[:, ~th, :, :][:, :, :, sh])
    temporal = error_report(prediction[:, th, :, :][:, :, :, ~sh], target[:, th, :, :][:, :, :, ~sh])
    return {'spatial_holdout': spatial, 'time_holdout': temporal,
            'score': .5*(spatial['velocity']['mean']+temporal['velocity']['mean'])}


def freeze(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source_plan = Path(args.source_plan).resolve()
    source_results = Path(args.source_results).resolve()
    inherited = json.loads(source_plan.read_text())
    previous = json.loads(source_results.read_text())
    if set(inherited['selections']) != {'generator_training'} or len(inherited['selections']['generator_training']) != 12:
        raise ValueError('capacity probe requires exactly the original 12 training windows')
    rows = inherited['selections']['generator_training']
    for a, b in itertools.combinations(rows, 2):
        if (a['source_recording'] == b['source_recording'] and a['source_frames'][0] < b['source_frames'][1]
                and b['source_frames'][0] < a['source_frames'][1]):
            raise ValueError('inherited source windows overlap')
    candidates = []
    for ns, nt, lambdas in ((27, 21, ((0., 0.),)), (39, 25, ((0., 0.), (1e-8, 1e-6)))):
        for spatial, temporal in (('cosine', 'chebyshev'), ('bspline', 'bspline')):
            for ls, lt in lambdas:
                cfg = StructuredNSProjectionConfig(spatial_mode_x=ns-1, spatial_mode_y=ns-1,
                    time_order=nt-1, spatial_basis=spatial, temporal_basis=temporal,
                    spatial_extension=.075 if spatial == 'cosine' else 0.,
                    fitting_method='joint_smoothing', smoothing_spatial=ls, smoothing_temporal=lt)
                cfg.validate()
                candidates.append({'id': f'{spatial}_{temporal}_s{ns}_t{nt}_ls{ls:g}_lt{lt:g}', 'config': cfg.to_dict()})
    plan = {key: inherited[key] for key in ('manifest', 'manifest_sha256', 'geometry', 'selections',
        'viscosity', 'time_holdout_frames', 'space_holdout', 'excluded_roles')}
    plan.update(schema='ns_adaptive_spline_capacity_probe_v1',
        status='adaptive diagnostic proposed after seeing the original 72-candidate results; not initial preregistration',
        source_plan=str(source_plan), source_plan_sha256=file_sha256(source_plan),
        source_results=str(source_results), source_results_sha256=file_sha256(source_results),
        candidates=candidates, solver={'tolerance': 1e-8, 'max_iterations': 500},
        interpretation='No automatic selection; full-fit improvements require heldout and derivative assessment. No Flow or final-test access.',
        later_generator_capacity_gate='39 requires at least 0.001 absolute CV improvement over the already selected eligible 27 configuration, plus unchanged derivative guards. Otherwise do not use 39 for Generator training. Fixed before any 39 results.',
        zero_smoothing='lambda_s=lambda_t=0 removes both field penalties; original preconditioner-only jitter and stopping tolerance remain.',
        cv='Same retained-time x retained-site fit and original mean of spatial/whole-frame holdout errors; recompute all six for audit.',
        derivative_protocol='Original derivatives() queries: fixed mid-cell and outer boundary/corners, five times, four evenly spaced selected windows; evaluate full and CV fields.',
        reference_control=next(r for r in previous['full_fits'] if r['id'] == 'current_v2'),
        earlier_zero_cv=[r for r in previous['rows'] if r['id'] in {c['id'] for c in candidates}],
        planned_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    write_json(out/'plan.json', plan)
    (out/'plan.md').write_text(
        '# Adaptive NS capacity diagnostic\n\n'
        'This is a new diagnostic after the 72-candidate study, not a replacement for its frozen selection protocol. '
        'Only its same 12 non-overlapping generator-training windows are read. Generator validation, development, '
        'verifier and final-test data are excluded.\n\n'
        'Six configurations: cosine/Chebyshev and quintic/cubic B-spline at 27×27×21 with zero penalties; '
        'both at 39×39×25 with (λs,λt)=(0,0) and (1e−8,1e−6). Cosine extension stays 0.075. '
        'All other objective, geometry, units, masks and solver settings are inherited.\n\n'
        'Report full-grid errors and spatial/whole-frame holdout errors for velocity/u/v, plus derivative RMS, p99 '
        'and maxima at the original fixed interior and boundary queries. Both full and heldout-fit fields are evaluated. '
        'CG failure is an explicit failed configuration, not a high-quality fit. Full-fit gains alone do not recommend '
        'a representation. A 39 configuration must improve CV by at least 0.1 percentage points over the already selected eligible 27 configuration, '
        'and pass the unchanged derivative guards, before it can even be considered for subsequent Generator training. '
        'No Generator training or automatic production change follows this script.\n')
    print(json.dumps({'frozen_plan': str(out/'plan.json'), 'candidates': len(candidates)}), flush=True)


def run(args):
    torch.set_num_threads(args.threads)
    out = Path(args.out).resolve()
    plan = json.loads((out/'plan.json').read_text())
    if (out/'results.json').exists() or (out/'progress.json').exists():
        raise FileExistsError('refusing to replace existing capacity results')
    if set(plan['selections']) != {'generator_training'}:
        raise ValueError('nontraining data is forbidden in this diagnostic')
    data, (x, y, times) = read_windows(plan)
    native = data['generator_training']
    grid, (ix, iy) = completed_grid(x, y, plan['geometry'])
    if not grid.fluid_mask.all():
        raise ValueError('capacity diagnostic requires the inherited complete ROI')
    target = np.zeros((*native.shape[:3], len(grid.x), len(grid.y)))
    target[..., ix, iy] = native
    flat = target.reshape(*target.shape[:3], -1)
    sh, th = make_masks(grid.fluid_mask.shape, len(times), plan['time_holdout_frames'])
    write_json(out/'input_audit.json', {'shape': list(flat.shape), 'read_roles': list(data),
        'spatial_holdout_count': int(sh.sum()), 'retained_sites': int((~sh).sum()),
        'time_holdout_frames': np.flatnonzero(th).tolist(), 'retained_times': int((~th).sum()),
        'plan_sha256': file_sha256(out/'plan.json')})
    results = []
    for candidate in plan['candidates']:
        start = time.time()
        row = dict(candidate)
        cfg = StructuredNSProjectionConfig.from_dict(candidate['config'])
        try:
            projector = JointSmoothNSCoefficientProjector(geometry=grid, t=times, config=cfg)
        except (RuntimeError, ValueError) as exc:
            row['construction_failure'] = str(exc)
            projector = None
        if projector is not None:
            for label, system in [('full', projector.prepare_system()),
                                  ('cv', projector.prepare_system(time_mask=~th, spatial_mask=~sh))]:
                tick = time.time()
                try:
                    coefficients, convergence = projector.solve(flat, system=system, **plan['solver'])
                    prediction = projector.predict(coefficients).detach().numpy()
                    block = {'convergence': convergence, 'all_points': error_report(prediction, flat)}
                    if label == 'cv':
                        block.update(cv_metrics(prediction, flat, sh, th))
                        earlier = next((r for r in plan['earlier_zero_cv'] if r['id'] == candidate['id']), None)
                        if earlier is not None and 'score' in earlier:
                            block['score_difference_from_original_run'] = block['score']-earlier['score']
                    cc = torch.stack((coefficients, torch.zeros_like(coefficients)), dim=1).reshape(
                        len(coefficients), 2, cfg.time_coefficient_count, cfg.spatial_mode_x+1, cfg.spatial_mode_y+1)
                    block['derivatives'] = derivatives(cc.detach().numpy(), cfg.to_dict(), plan, times, grid)
                    baseline = plan['reference_control']['derivatives']
                    block['derivative_ratios_to_v2_full_fit'] = {q: {
                        stat: block['derivatives'][q][stat]/max(baseline[q][stat], 1e-30)
                        for stat in ('rms', 'maximum_absolute')}
                        for q in ('gradient', 'time_derivative', 'laplacian', 'forcing')}
                    np.save(out/f"{candidate['id']}_{label}_coefficients.npy", cc.detach().numpy())
                except (RuntimeError, ValueError) as exc:
                    block = {'failure': str(exc)}
                block['seconds'] = time.time()-tick
                row[label] = block
                print(json.dumps({'event': 'case_stage', 'id': row['id'], 'stage': label,
                    'failure': block.get('failure'), 'velocity': block.get('all_points', {}).get('velocity'),
                    'score': block.get('score'), 'seconds': block['seconds']}), flush=True)
                write_json(out/'progress.json', {'completed': results, 'current': row})
        row['seconds'] = time.time()-start
        results.append(row)
        write_json(out/'progress.json', {'completed': results})
    report = {'plan_sha256': file_sha256(out/'plan.json'), 'reference_control': plan['reference_control'],
        'candidates': results, 'scope': plan['status'], 'recommendation': 'none: capacity and instability diagnostic only'}
    write_json(out/'results.json', report)
    lines = ['# Adaptive capacity results', '',
        'Same 12 training windows; no validation/final data. Percent errors are mean per-window relative L2. '
        'CV is the original equal mean of spatial and whole-frame holdout errors. '
        'Derivative ratios compare full-fit fields against the original v2 full-fit control.', '',
        '| Configuration | Full velocity % | Full u % | Full v % | CV % | Laplacian RMS / max ratio | Forcing RMS / max ratio |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in results:
        full, cv = row.get('full', {}), row.get('cv', {})
        errors, ratios = full.get('all_points'), full.get('derivative_ratios_to_v2_full_fit')
        if errors is None or ratios is None:
            lines.append(f"| {row['id']} | failed | | | | | |")
            continue
        ratio = lambda q: f"{ratios[q]['rms']:.3g} / {ratios[q]['maximum_absolute']:.3g}"
        score = f"{100*cv['score']:.4f}" if 'score' in cv else 'failed'
        lines.append(f"| {row['id']} | {100*errors['velocity']['mean']:.4f} | {100*errors['u']['mean']:.4f} | "
                     f"{100*errors['v']['mean']:.4f} | {score} | {ratio('laplacian')} | {ratio('forcing')} |")
    lines += ['', 'Zero penalties remove field smoothing, while preconditioner-only jitter and finite CG stopping remain. '
              'A lower training fit with worse holdout or unstable derivatives is not a recommended Generator representation. '
              'The JSON records separate spatial/time errors, p99/max derivatives for full and CV fields, and solver failures.']
    (out/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'event': 'complete', 'results': str(out/'results.json')}), flush=True)


def self_check():
    from types import SimpleNamespace
    from agent_verifier.neural_generator.continuous_geometry import ContinuousGeometrySpec
    torch.set_num_threads(1)
    cfg = StructuredNSProjectionConfig(spatial_mode_x=6, spatial_mode_y=6, time_order=3,
        spatial_basis='bspline', temporal_basis='bspline', fitting_method='joint_smoothing',
        smoothing_spatial=1e-8, smoothing_temporal=1e-6)
    x, y, t = np.linspace(0., 1., 13), np.linspace(0., 1., 12), np.linspace(0., 1., 9)
    geometry = ContinuousGeometrySpec((0., 1., 0., 1.), (), outer_boundary='manufactured_trace')
    grid = SimpleNamespace(x=x, y=y, fluid_mask=np.ones((len(x), len(y)), bool),
        config=SimpleNamespace(nx=len(x), ny=len(y)),
        to_json_dict=lambda: {'continuous_geometry': geometry.to_dict()})
    projector = JointSmoothNSCoefficientProjector(geometry=grid, t=t, config=cfg)
    sh, th = make_masks(grid.fluid_mask.shape, len(t), [3, 4])
    psi = torch.tensor(np.random.default_rng(24).normal(size=(2, 4, 49)), dtype=torch.float64)
    target = projector.predict(psi).numpy()
    poisoned = target.copy()
    poisoned[:, th] = np.nan
    poisoned[:, :, :, sh] = np.nan
    system = projector.prepare_system(time_mask=~th, spatial_mask=~sh)
    a, _ = projector.solve(target, system=system)
    b, _ = projector.solve(poisoned, system=system)
    torch.testing.assert_close(a, b, rtol=0., atol=0.)
    exact = cv_metrics(target, target, sh, th)
    corrupted = target.copy(); corrupted[:, th] += 1.; corrupted[:, :, :, sh] += 2.
    broken = cv_metrics(corrupted, target, sh, th)
    assert exact['score'] == 0. and broken['score'] > 0.
    print(json.dumps({'self_check': 'pass', 'excluded_observations': 'NaN poison leaves fitted coefficients bit-identical',
                      'holdout_metric': 'zero for identical fields; positive after heldout-only corruption'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-plan'); parser.add_argument('--source-results'); parser.add_argument('--out')
    parser.add_argument('--threads', type=int, default=8)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', action='store_true'); mode.add_argument('--run', action='store_true')
    mode.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif not args.out or (args.freeze and (not args.source_plan or not args.source_results)):
        parser.error('--out is required; --freeze also needs --source-plan and --source-results')
    elif args.freeze:
        freeze(args)
    else:
        run(args)
