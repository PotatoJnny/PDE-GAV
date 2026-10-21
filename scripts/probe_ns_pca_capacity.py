#!/usr/bin/env python
"""One adaptive PCA capacity diagnostic on frozen NS train/validation coefficients."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'legacy/ns/src'), str(ROOT/'scripts')]
import numpy as np
import torch
from prepare_ns_reconstructed_generator import file_sha256, write_json, relative_errors, summary, velocity_from_coefficients
from summarize_ns_measured_representation_change import load_reference, coefficient_batches
from pde_agent_verifier_solver.plugins.ns.completed_fields import completed_grid
from agent_verifier.neural_generator.structured_ns_answer_first import StructuredNSProjectionConfig, _projection_spatial_operators, chebyshev_basis
from agent_verifier.neural_generator.structured_ns_basis import axis_basis_derivatives
from agent_verifier.neural_generator.continuous_geometry import continuous_geometry_spec_from_contract
from agent_verifier.neural_generator.structured_ns_distribution import StructuredCoefficientPCA


class CachedVelocity:
    """Original physical velocity and temporal design; no fitting operator or SVD."""
    def __init__(self, reference, contract):
        self.config = StructuredNSProjectionConfig.from_dict(contract['projection_config'])
        self.config.validate()
        grid, (ix, iy) = completed_grid(reference['x'], reference['y'], contract['geometry'])
        if not grid.fluid_mask.all():
            raise ValueError('this frozen comparison requires its complete rectangular ROI')
        self.spatial = _projection_spatial_operators(torch, geometry=grid, geometry_spec=continuous_geometry_spec_from_contract(grid),
            config=self.config, device=torch.device('cpu'))['velocity_design']
        times = torch.as_tensor(reference['time'], dtype=torch.float64)
        tau = (times-times[0])/(times[-1]-times[0])
        self.temporal = (chebyshev_basis(tau, order=self.config.time_order)
            if self.config.temporal_basis == 'chebyshev' else axis_basis_derivatives(tau, config=self.config, axis='t')[0])
        flat = ix*len(grid.y)+iy
        self.source_order = np.searchsorted(np.sort(flat), flat)
        self.shape = (2, self.config.time_coefficient_count, self.config.spatial_mode_x+1, self.config.spatial_mode_y+1)

    def __call__(self, coefficients):
        values = torch.as_tensor(coefficients, dtype=torch.float64)
        if tuple(values.shape[1:]) != self.shape:
            raise ValueError('coefficient shape does not match cached continuous representation')
        psi = values[:, 0].reshape(len(values), self.config.time_coefficient_count, -1)
        projected = self.temporal @ psi @ self.spatial.T
        return projected.reshape(len(values), len(self.temporal), 2, -1).numpy()[..., self.source_order]


def array_sha256(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float32).tobytes(order='C')).hexdigest()


def freeze(args):
    prepared, old = Path(args.prepared).resolve(), Path(args.old_prepared).resolve()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    important = ('continuous_contract.json', 'preparation_config.json', 'validation_reference.npz',
        'pca_checkpoint.pt', 'pca_selection.json', 'coefficients/train/coefficient_manifest.json',
        'coefficients/validation/coefficient_manifest.json')
    assets = {}
    for name, directory in [('new', prepared), ('old', old)]:
        assets[name] = {'prepared': str(directory), 'files': {key: file_sha256(directory/key) for key in important}}
    plan = {'schema': 'ns_adaptive_pca_capacity_v1', 'assets': assets,
        'status': 'adaptive followup to the failed fixed-rank representation gate; not initial preregistration',
        'fit': {'rank': 256, 'oversample': 16, 'power_iterations': 4, 'seed': 20260911},
        'report_ranks': [16, 32, 64, 128, 256], 'eligible_ranks': [128, 256],
        'expected_training_cases': 893, 'expected_validation_cases': 193,
        'selection': {'maximum_mean_pca_to_measured': .09762428419077477,
            'old_v2_rank64_mean': .10062428419077477, 'required_absolute_improvement': .003,
            'rule': 'smallest rank in [128,256] at or below frozen maximum mean PCA-to-measured error; otherwise none'},
        'algorithm': 'unchanged StructuredCoefficientPCA.fit; training-only centering/scaling/randomized SVD',
        'basis_warning': 'all reported ranks truncate the same maximum-rank256 q272 randomized-SVD fit; its rank64 is not the old maximum-rank64 q80 fit',
        'read_roles': ['generator_training', 'generator_validation'],
        'excluded_roles': ['development', 'verifier', 'final_test', 'reconstruction_validation'],
        'validation_status': 'existing generator-validation has already been used for selection; not independent final-test evidence',
        'next_action': 'save checkpoint/corpus manifests and selection only; do not start Flow',
        'planned_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    write_json(out/'plan.json', plan)
    (out/'plan.md').write_text('# Adaptive PCA capacity diagnostic\n\n'
        'Freeze one original PCA fit: 893 training coefficient cases, rank 256, oversample 16, '
        '4 power iterations, seed 20260911. Report truncations 16/32/64/128/256 on the identical 193 existing '
        'generator-validation measurements, separately against measured and continuous velocity (whole/u/v). '
        'The new rank64 uses the larger q272 randomized-SVD subspace; it is not the earlier q80 fit.\n\n'
        'Select only the smallest rank from {128,256} with mean PCA-to-measured relative L2 <= 0.09762428419077477 '
        '(at least 0.003 better than old v2 rank64). Otherwise select none. No automatic Flow training. '
        'No solver evaluation data are loaded. Validation is reused selection data. '
        'Verify coefficient SHA/signature, split case ordering, and identical old/new measurement references.\n')
    print(json.dumps({'event': 'frozen', 'plan': str(out/'plan.json')}), flush=True)


def error_rows(actual, target):
    return {'velocity': relative_errors(actual, target).tolist(),
            'u': relative_errors(actual[:, :, 0], target[:, :, 0]).tolist(),
            'v': relative_errors(actual[:, :, 1], target[:, :, 1]).tolist()}


def evaluate(pca, reference, evaluator, continuous, *, batch_cases):
    collected = defaultdict(lambda: defaultdict(list))
    for start in range(0, len(reference['coefficients']), batch_cases):
        stop = start+batch_cases
        values = reference['coefficients'][start:stop]
        reconstructed = pca.reconstruct(values).cpu().numpy()
        predicted = evaluator(reconstructed)
        for name, target in [('pca_to_measured', reference['measured'][start:stop]),
                             ('pca_to_continuous', continuous[start:stop])]:
            for component, errors in error_rows(predicted, target).items():
                collected[name][component].extend(errors)
    return {'metrics': {stage: {component: summary(errors) for component, errors in columns.items()}
                         for stage, columns in collected.items()}, 'per_case': dict(collected)}


def verify_manifest_files(prepared, split):
    path = prepared/'coefficients'/split/'coefficient_manifest.json'
    manifest = json.loads(path.read_text())
    return manifest


def corpus_alias(prepared, split, output):
    original = prepared/'coefficients'/split/'coefficient_manifest.json'
    manifest = json.loads(original.read_text())
    manifest['shards'] = [{**row, 'path': str((original.parent/row['path']).resolve())} for row in manifest['shards']]
    target = output/'coefficients'/split/'coefficient_manifest.json'
    target.parent.mkdir(parents=True)
    write_json(target, manifest)
    return str(target)


def run(args):
    torch.set_num_threads(args.threads)
    out = Path(args.out).resolve(); plan = json.loads((out/'plan.json').read_text())
    if (out/'pca_checkpoint.pt').exists() or (out/'results.json').exists():
        raise FileExistsError('refusing to replace existing PCA results')
    started = time.time()
    for asset in plan['assets'].values():
        directory = Path(asset['prepared'])
    prepared = Path(plan['assets']['new']['prepared']); old = Path(plan['assets']['old']['prepared'])
    config, contract, reference = load_reference(prepared)
    old_config, old_contract, old_reference = load_reference(old)
    for key in ('x', 'y', 'time', 'measured'):
        if not np.array_equal(reference[key], old_reference[key]):
            raise ValueError(f'old/new {key} or measurement order differ')
    if (reference['case_ids'] != old_reference['case_ids'] or contract['geometry'] != old_contract['geometry']
            or config['coordinate_units'] != old_config['coordinate_units']):
        raise ValueError('old/new Case IDs, geometry or coordinate units differ')
    if len(reference['coefficients']) != plan['expected_validation_cases']:
        raise ValueError('unexpected generator-validation case count')
    parts, ids = [], []
    for values, batch_ids in coefficient_batches(prepared, 'train', args.batch_cases):
        parts.append(values); ids.extend(batch_ids)
    train = np.concatenate(parts)
    old_ids = []
    old_train_manifest = verify_manifest_files(old, 'train')
    for row in old_train_manifest['shards']:
        with np.load(old/'coefficients/train'/row['path'], allow_pickle=False) as arrays:
            old_ids.extend(arrays['case_ids'].astype(str).tolist())
    if ids != old_ids or len(train) != plan['expected_training_cases'] or len(set(ids)) != len(ids):
        raise ValueError('old/new training Case ID ordering or count differs')
    if set(ids) & set(reference['case_ids']):
        raise ValueError('training and validation case IDs overlap')
    original_new_pca = StructuredCoefficientPCA.load(prepared/'pca_checkpoint.pt')
    if original_new_pca.source_signature != contract:
        raise ValueError('original prepared PCA source signature differs from continuous contract')
    evaluator, old_evaluator = CachedVelocity(reference, contract), CachedVelocity(old_reference, old_contract)
    # One bounded direct check per representation against the prior public helper.
    for ref, con, ev in [(reference, contract, evaluator), (old_reference, old_contract, old_evaluator)]:
        direct = velocity_from_coefficients(ref['coefficients'][:1], x=ref['x'], y=ref['y'], times=ref['time'],
            geometry=con['geometry'], projection_config=con['projection_config'])
        np.testing.assert_allclose(ev(ref['coefficients'][:1]), direct, rtol=1e-12, atol=1e-12)
    continuous = np.concatenate([evaluator(reference['coefficients'][i:i+args.batch_cases])
                                for i in range(0, len(reference['coefficients']), args.batch_cases)])
    old_continuous = np.concatenate([old_evaluator(old_reference['coefficients'][i:i+args.batch_cases])
                                    for i in range(0, len(old_reference['coefficients']), args.batch_cases)])
    old_pca = StructuredCoefficientPCA.load(old/'pca_checkpoint.pt').with_rank(64)
    old_control = evaluate(old_pca, old_reference, old_evaluator, old_continuous, batch_cases=args.batch_cases)
    old_mean = old_control['metrics']['pca_to_measured']['velocity']['mean']
    if not np.isclose(old_mean, plan['selection']['old_v2_rank64_mean'], rtol=0., atol=1e-10):
        raise ValueError(f'old v2 reference mean differs from the frozen gate: {old_mean}')
    original_new_control = evaluate(original_new_pca.with_rank(64), reference, evaluator, continuous, batch_cases=args.batch_cases)
    audit = {'training_cases': len(train), 'validation_cases': len(reference['coefficients']),
        'training_case_ids': ids, 'validation_case_ids': reference['case_ids'],
        'source_signature': contract, 'paired_old_new_observations_identical': True,
        'old_v2_rank64_recomputed_mean': old_mean, 'cached_design_matches_original_helper': True,
        'old_v2_control': old_control, 'new_original_max64_control': original_new_control,
        'continuous_to_measured': {k: summary(v) for k, v in error_rows(continuous, reference['measured']).items()}}
    write_json(out/'input_audit.json', audit)
    print(json.dumps({'event': 'verified_before_fit', 'train': len(train), 'validation': len(reference['coefficients'])}), flush=True)
    pca = StructuredCoefficientPCA.fit(train, source_signature=contract, **plan['fit'])
    pca.save(out/'pca_checkpoint.pt')
    restored = StructuredCoefficientPCA.load(out/'pca_checkpoint.pt')
    if restored.rank != 256 or restored.source_signature != contract:
        raise ValueError('saved PCA failed identity verification')
    torch.testing.assert_close(restored.reconstruct(reference['coefficients'][:2]),
                               pca.reconstruct(reference['coefficients'][:2]), rtol=0., atol=0.)
    rows = []
    for rank in plan['report_ranks']:
        result = evaluate(pca.with_rank(rank), reference, evaluator, continuous, batch_cases=args.batch_cases)
        row = {'rank': rank, 'maximum_fitted_rank': pca.rank, 'explained_coefficient_variance': pca.explained_variance_fraction(rank), **result}
        rows.append(row); write_json(out/'progress.json', {'ranks': rows})
        print(json.dumps({'event': 'rank', 'rank': rank, 'metrics': row['metrics']}), flush=True)
    eligible = [r['rank'] for r in rows if r['rank'] in plan['eligible_ranks'] and
        r['metrics']['pca_to_measured']['velocity']['mean'] <= plan['selection']['maximum_mean_pca_to_measured']]
    selected = min(eligible) if eligible else None
    manifests = {split: corpus_alias(prepared, split, out) for split in ('train', 'validation')}
    selection = {'schema': 'ns_pca_capacity_selection_v1', 'selected_rank': selected, 'selection_split': 'generator_validation',
        'rule': plan['selection']['rule'], 'threshold': plan['selection']['maximum_mean_pca_to_measured'],
        'pca_checkpoint': str(out/'pca_checkpoint.pt'), 'pca_checkpoint_sha256': file_sha256(out/'pca_checkpoint.pt'),
        'maximum_fitted_rank': 256, 'source_signature': contract, 'corpus_manifests': manifests,
        'basis_warning': plan['basis_warning'], 'flow_training_started': False}
    write_json(out/'selection.json', selection)
    report = {'plan_sha256': file_sha256(out/'plan.json'), 'ranks': rows, 'selection': selection,
        'continuous_to_measured': audit['continuous_to_measured'], 'old_v2_rank64': old_control['metrics'],
        'new_original_max64_rank64': original_new_control['metrics'], 'elapsed_seconds': time.time()-started}
    write_json(out/'results.json', report)
    lines = ['# Adaptive PCA capacity results', '', plan['basis_warning']+'.', '',
        'Only 893 training coefficients fit PCA; the same 193 reused generator-validation measurements select rank. '
        'No solver evaluation data are used. Errors below are mean per-window relative L2 percentages.', '',
        '| Rank | PCA → measured | u | v | PCA → continuous | u | v |', '|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        a, b = [row['metrics'][k] for k in ('pca_to_measured', 'pca_to_continuous')]
        numbers = [100*g[c]['mean'] for g in (a, b) for c in ('velocity', 'u', 'v')]
        lines.append('| '+str(row['rank'])+' | '+' | '.join(f'{v:.4f}%' for v in numbers)+' |')
    lines += ['', f'Frozen threshold: {100*selection["threshold"]:.6f}%. Selected rank: {selected}. Flow was not started.', '',
        'Coefficient manifests refer to the unchanged original coefficient shard bytes; no new field fitting or corpus transformation occurred. '
        'A rank passing this gate establishes compression fidelity on reused validation only, not Generator distribution quality or independent final-test performance.']
    (out/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'event': 'complete', 'selected_rank': selected, 'seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared'); p.add_argument('--old-prepared'); p.add_argument('--out', required=True)
    p.add_argument('--threads', type=int, default=16); p.add_argument('--batch-cases', type=int, default=16)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', action='store_true'); mode.add_argument('--run', action='store_true')
    a = p.parse_args()
    if a.freeze and (not a.prepared or not a.old_prepared):
        p.error('--freeze requires --prepared and --old-prepared')
    freeze(a) if a.freeze else run(a)
