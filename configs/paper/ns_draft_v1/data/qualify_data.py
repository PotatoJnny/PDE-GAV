#!/usr/bin/env python3
"""Bounded CPU checks on the 50 calibration cases; never score the test banks.

This checks the stored discrete MMS relation, not an independent forward solve.
Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python configs/paper/ns_draft_v1/data/qualify_data.py
"""
from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'legacy/ns/src')]

from agent_verifier.neural_generator.triplane_dataset import (  # noqa: E402
    GeometryContract, KiteBaffleGeometryConfig, case_metrics,
    mms_forcing_from_velocity_pressure,
)
from pde_agent_verifier_solver.plugins.ns.common_evaluation import load_numerical_split  # noqa: E402
from freeze_data import sha  # noqa: E402


def rel(error, truth):
    return float(np.linalg.norm(np.asarray(error).ravel()) /
                 max(float(np.linalg.norm(np.asarray(truth).ravel())), 1e-12))


def summary(values):
    return {'count': len(values), 'mean': float(np.mean(values)),
            'median': float(np.median(values)), 'min': float(np.min(values)),
            'max': float(np.max(values)), 'q90': float(np.quantile(values, .9))}


def main():
    manifest_path = HERE / 'calibration_manifest.json'
    rows = json.loads(manifest_path.read_text())['splits']['calibration']
    assert len(rows) == 50 and set(Counter(r['family'] for r in rows).values()) == {5}
    # Public adapter compatibility only: exactly one public record from each bank.
    adapter_checks = []
    for filename, names in {
        'development_manifest.json': ('development', 'validation'),
        'calibration_manifest.json': ('calibration',),
        'test_draft_manifest.json': ('test_draft',),
        'test_final_manifest.json': ('test_final',),
    }.items():
        path = HERE / filename
        manifest = json.loads(path.read_text())
        for split in names:
            key = manifest['splits'][split][0]['case_id']
            records = load_numerical_split(path, split=split, case_ids=[key])
            assert len(records) == 1 and records[0].case.case_id == key
            adapter_checks.append({'manifest': filename, 'split': split,
                                   'case_id': key, 'public_record_loaded': True,
                                   'hidden_truth_queried': False})
    stokes_path = REPO / 'legacy/ns/baselines/triplane/spectral_stokes_projection_submission.py'
    spec = importlib.util.spec_from_file_location('ns_calibration_stokes', stokes_path)
    stokes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stokes)
    results = []
    started = time.perf_counter()
    for row in rows:
        directory = Path(row['case_dir'])
        with np.load(directory / 'public_arrays.npz', allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        with np.load(directory / 'hidden_truth_cache.npz', allow_pickle=False) as data:
            velocity = np.asarray(data['u_true'], dtype=np.float64)
            pressure = np.asarray(data['pressure'], dtype=np.float64)
        question = json.loads((directory / 'public_question.json').read_text())
        assert question['forcing_derivation'] == 'mms_from_hidden_velocity_pressure'
        assert velocity.shape == (32, 2, 64, 64)
        assert question['viscosity'] == .01
        geometry = GeometryContract(
            config=KiteBaffleGeometryConfig(),
            x=arrays['x'], y=arrays['y'], fluid_mask=arrays['fluid_mask'],
            obstacle_mask=arrays['obstacle_mask'], outer_boundary_mask=arrays['outer_boundary_mask'],
            obstacle_signed_distance=arrays['obstacle_signed_distance'],
            brinkman_indicator=arrays['brinkman_indicator'], contract_hash=question['geometry']['contract_hash'],
        )
        fluid = np.asarray(arrays['fluid_mask'], dtype=bool)
        forcing = np.asarray(arrays['forcing'], dtype=np.float64)
        recomputed = mms_forcing_from_velocity_pressure(
            velocity=velocity, pressure=pressure, t=arrays['t'],
            viscosity=question['viscosity'], geometry=geometry,
        )
        metrics = case_metrics(velocity=velocity, pressure=pressure, forcing=forcing,
                               t=arrays['t'], geometry=geometry)
        forcing_rel = rel((recomputed - forcing)[:, :, fluid], forcing[:, :, fluid])
        initial_abs = float(np.max(np.abs(velocity[0] - arrays['initial_velocity'])))
        finite = all(np.isfinite(a).all() for a in (velocity, pressure, forcing, arrays['initial_velocity']))
        hold = np.broadcast_to(arrays['initial_velocity'], velocity.shape)
        hold_rel = float(np.mean([rel((hold[:, c] - velocity[:, c])[:, fluid], velocity[:, c, fluid]) for c in range(2)]))
        stokes_started = time.perf_counter()
        stokes_result = stokes.solve_case(str(directory))['velocity']
        stokes_seconds = time.perf_counter() - stokes_started
        stokes_rel = float(np.mean([rel((stokes_result[:, c] - velocity[:, c])[:, fluid], velocity[:, c, fluid]) for c in range(2)]))
        results.append({
            'case_id': row['case_id'], 'family': row['family'],
            'public_arrays_sha256': sha(directory / 'public_arrays.npz'),
            'hidden_truth_cache_sha256': sha(directory / 'hidden_truth_cache.npz'),
            'all_finite': bool(finite), 'public_initial_max_abs_difference': initial_abs,
            'discrete_mms_forcing_relative_l2': forcing_rel,
            'hold_initial_component_mean_relative_l2_native_grid': hold_rel,
            'periodic_masked_stokes_component_mean_relative_l2_native_grid': stokes_rel,
            'periodic_masked_stokes_wall_seconds': stokes_seconds,
            'release_style_metrics': metrics,
            'discrete_array_consistency_passed': bool(finite and initial_abs <= 1e-7 and forcing_rel <= 1e-5),
        })
    aggregate_keys = ('public_initial_max_abs_difference', 'discrete_mms_forcing_relative_l2',
                      'hold_initial_component_mean_relative_l2_native_grid',
                      'periodic_masked_stokes_component_mean_relative_l2_native_grid',
                      'periodic_masked_stokes_wall_seconds')
    output = {
        'schema': 'ns_paper_data_calibration_v1', 'manifest': str(manifest_path), 'manifest_sha256': sha(manifest_path),
        'case_count': len(results), 'family_counts': dict(Counter(r['family'] for r in results)),
        'adapter_public_record_checks': adapter_checks, 'cases': results,
        'summary': {key: summary([r[key] for r in results]) for key in aggregate_keys},
        'all_discrete_array_consistency_checks_passed': all(r['discrete_array_consistency_passed'] for r in results),
        'reference_construction': 'Numerical velocity/pressure trajectories; public forcing recomputed by the repository discrete MMS derivative operators.',
        'check_limits': [
            'This repeats the data-construction differential operators; it checks storage/contract consistency, not independent physical accuracy.',
            'The adapter interpolates cached finite-grid truth; this check does not establish exact continuous PDE consistency at arbitrary query coordinates.',
            'Stored velocity is only approximately divergence-free; no independent grid/time convergence result was established.',
            'The legacy Stokes routine uses periodic projection followed by a fluid mask; it is a diagnostic approximation, not a reliable obstacle-domain reference solver.',
            'Diagnostic native-grid errors are not final paper random-query scores; no test-bank prediction was evaluated.',
        ],
        'scientific_qualification': {
            'known_array_contract_inconsistency': any(not r['discrete_array_consistency_passed'] for r in results),
            'independent_forward_reference_qualified': False,
            'continuous_pde_error_floor_qualified': False,
            'launch_interpretation': 'Consistency-qualified for the declared discrete-MMS benchmark only if the array checks pass; independent solver/refinement accuracy claims remain pending.',
            'next_action': 'Implement and verify a public-input-only no-slip obstacle-domain NS reference, freeze the same interpolated initial/forcing fields across refinements, and evaluate these 50 calibration cases with at least two grid/time resolutions. No validated repository command for that independent qualification was found.',
        },
        'implementation': {'checker': str(Path(__file__).resolve()), 'checker_sha256': sha(Path(__file__).resolve()),
                           'mms_operators_sha256': sha(REPO / 'legacy/ns/src/agent_verifier/neural_generator/triplane_dataset.py'),
                           'stokes_source_sha256': sha(stokes_path)},
        'elapsed_seconds': time.perf_counter() - started,
    }
    path = REPO / 'docs/paper/evidence/ns_draft_v1_data_calibration.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f'refusing to overwrite existing qualification evidence: {path}')
    path.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'report': str(path), 'all_checks_passed': output['all_discrete_array_consistency_checks_passed'],
                      'summary': output['summary'], 'elapsed_seconds': output['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    main()
