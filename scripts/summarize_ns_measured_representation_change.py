#!/usr/bin/env python
"""Compare two frozen measured-PIV representations without fitting or sampling.

Use existing raw, 256-sample Flow evaluations. Paired errors concern the same
validation measurements; generated draws receive distribution summaries only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

from prepare_ns_reconstructed_generator import (
    file_sha256, relative_errors, summary, velocity_from_coefficients, write_json,
)
from agent_verifier.neural_generator.structured_ns_distribution import StructuredCoefficientPCA


def read_json(path):
    return json.loads(Path(path).read_text())


def coefficient_batches(prepared, split, batch_cases):
    path = prepared / 'coefficients' / split / 'coefficient_manifest.json'
    manifest = read_json(path)
    count, seen = 0, set()
    for row in manifest['shards']:
        shard = path.parent / row['path']
        with np.load(shard, allow_pickle=False) as arrays:
            values = arrays['coefficients']
            ids = arrays['case_ids'].astype(str)
        if len(values) != row['count'] or len(ids) != len(values) or not np.isfinite(values).all():
            raise ValueError(f'invalid coefficient shard: {shard}')
        for cid in ids:
            if cid in seen:
                raise ValueError(f'duplicate coefficient Case ID: {cid}')
            seen.add(cid)
        for start in range(0, len(values), batch_cases):
            yield values[start:start+batch_cases], ids[start:start+batch_cases].tolist()
        count += len(values)
    if count != manifest['encoded_cases']:
        raise ValueError('coefficient manifest count differs from its shards')


def raw_evaluation(prepared):
    found = []
    for path in sorted(prepared.glob('evaluation_*/quality_report.json')):
        report = read_json(path)
        if report.get('flow_state') == 'raw' and report.get('generator_sample_count') == 256:
            found.append((path, report))
    if len(found) != 1:
        raise ValueError(f'expected exactly one existing raw256 evaluation under {prepared}, found {len(found)}')
    path, report = found[0]
    sample_path = path.parent / 'latent_audit/structured_flow_samples.npz'
    with np.load(sample_path, allow_pickle=False) as arrays:
        samples = arrays['coefficients']
    if len(samples) != 256 or not np.isfinite(samples).all():
        raise ValueError('raw256 sample coefficients are incomplete or nonfinite')
    command = read_json(path.parent / 'sampling_command.json')['command']
    def option(name):
        return command[command.index(name)+1] if name in command else None
    if option('--state') != 'raw' or option('--samples') != '256':
        raise ValueError('sampling command disagrees with raw256 evaluation')
    checkpoint = Path(report['flow_checkpoint'])
    if not checkpoint.is_file():
        raise ValueError(f'Flow checkpoint missing or changed: {checkpoint}')
    return samples, {
        'evaluation_directory': str(path.parent), 'flow_checkpoint': str(checkpoint),
        'flow_checkpoint_sha256': report['flow_checkpoint_sha256'],
        'sample_coefficients_sha256': file_sha256(sample_path), 'sample_count': len(samples),
        'flow_state': 'raw', 'integration_steps': int(option('--integration-steps')),
        'seed': int(option('--seed')), 'quality_report_sha256': file_sha256(path),
    }


def load_reference(prepared):
    config = read_json(prepared / 'preparation_config.json')
    if config.get('input_mode') != 'measured':
        raise ValueError('this comparison requires input_mode=measured in both assets')
    contract = read_json(prepared / 'continuous_contract.json')
    with np.load(prepared / 'validation_reference.npz', allow_pickle=False) as arrays:
        reference = {key: arrays[key] for key in ('x', 'y', 'time', 'coefficients', 'measured')}
    expected = (len(reference['coefficients']), len(reference['time']), 2, len(reference['x']))
    if reference['measured'].shape != expected or not all(np.isfinite(v).all() for v in reference.values()):
        raise ValueError('invalid measured validation reference')
    ids, start = [], 0
    for values, batch_ids in coefficient_batches(prepared, 'validation', 64):
        if not np.array_equal(values, reference['coefficients'][start:start+len(values)]):
            raise ValueError('validation reference coefficients differ from validation corpus/order')
        ids.extend(batch_ids)
        start += len(values)
    if start != len(reference['coefficients']):
        raise ValueError('validation reference count differs from its corpus')
    reference['case_ids'] = ids
    return config, contract, reference


def rms_per_window(values):
    # Each entry is sqrt(mean over time and spatial sites), component separately.
    return np.sqrt(np.mean(np.asarray(values, dtype=np.float64)**2, axis=(1, 3)))


def add_errors(storage, stage, actual, target):
    storage[stage]['overall'].extend(relative_errors(actual, target).tolist())
    for i, component in enumerate(('u', 'v')):
        storage[stage][component].extend(relative_errors(actual[:, :, i], target[:, :, i]).tolist())


def summarize_asset(prepared, config, contract, reference, *, batch_cases, device):
    rank = int(read_json(prepared / 'pca_selection.json')['selected_rank'])
    pca = StructuredCoefficientPCA.load(prepared / 'pca_checkpoint.pt').with_rank(rank)
    samples, generation = raw_evaluation(prepared)
    if tuple(samples.shape[1:]) != tuple(reference['coefficients'].shape[1:]):
        raise ValueError('generated coefficient shape differs from validation fields')
    kwargs = {'x': reference['x'], 'y': reference['y'], 'times': reference['time'],
              'geometry': contract['geometry'], 'projection_config': contract['projection_config'],
              'device': device}
    errors = defaultdict(lambda: defaultdict(list))
    windows = defaultdict(list)
    windows['validation_measured'].append(rms_per_window(reference['measured']))
    first_fields = {'measured': reference['measured'][0]}
    train_ids = []
    # Keep only small per-window statistics; never retain all training velocities.
    for values, ids in coefficient_batches(prepared, 'train', batch_cases):
        continuous = velocity_from_coefficients(values, **kwargs)
        compressed = velocity_from_coefficients(pca.reconstruct(values).cpu().numpy(), **kwargs)
        windows['training_continuous'].append(rms_per_window(continuous))
        windows['training_pca'].append(rms_per_window(compressed))
        add_errors(errors, 'training_pca_to_continuous', compressed, continuous)
        train_ids.extend(ids)
    for start in range(0, len(reference['coefficients']), batch_cases):
        values = reference['coefficients'][start:start+batch_cases]
        measured = reference['measured'][start:start+batch_cases]
        continuous = velocity_from_coefficients(values, **kwargs)
        compressed = velocity_from_coefficients(pca.reconstruct(values).cpu().numpy(), **kwargs)
        add_errors(errors, 'validation_continuous_to_measured', continuous, measured)
        add_errors(errors, 'validation_pca_to_continuous', compressed, continuous)
        add_errors(errors, 'validation_pca_to_measured', compressed, measured)
        windows['validation_continuous'].append(rms_per_window(continuous))
        windows['validation_pca'].append(rms_per_window(compressed))
        if start == 0:
            first_fields.update(continuous=continuous[0], pca=compressed[0])
    for start in range(0, len(samples), batch_cases):
        generated = velocity_from_coefficients(samples[start:start+batch_cases], **kwargs)
        windows['generated'].append(rms_per_window(generated))
        if start == 0:
            first_fields['generated'] = generated[0]
    rms = {}
    for stage, rows in windows.items():
        values = np.concatenate(rows)
        if not np.isfinite(values).all():
            raise ValueError(f'nonfinite velocity statistics in {stage}')
        rms[stage] = {component: {**summary(values[:, i]),
            'std_across_windows': float(np.std(values[:, i], ddof=0)),
            'count': len(values)} for i, component in enumerate(('u', 'v'))}
    report = {'prepared_directory': str(prepared), 'projection_config': contract['projection_config'],
        'selected_pca_rank': rank, 'pca_checkpoint_sha256': file_sha256(prepared / 'pca_checkpoint.pt'),
        'continuous_contract_sha256': file_sha256(prepared / 'continuous_contract.json'),
        'validation_reference_sha256': file_sha256(prepared / 'validation_reference.npz'),
        'training_case_count': len(train_ids), 'validation_case_count': len(reference['case_ids']),
        'paired_relative_l2': {stage: {component: summary(v) for component, v in groups.items()}
                               for stage, groups in errors.items()},
        'window_component_rms': rms, 'generation': generation,
        'coordinate_units': config.get('coordinate_units'), 'pressure_model': config.get('pressure_model')}
    return report, first_fields, train_ids


def paired_plot(out, reference, old, new):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = len(reference['time']) // 2
    x, y = reference['x'], reference['y']
    truth = reference['measured'][0, frame]
    old_velocity, new_velocity = old['continuous'][frame], new['continuous'][frame]
    figure, axes = plt.subplots(2, 5, figsize=(17, 7), constrained_layout=True)
    for row, component in enumerate(('speed', 'v')):
        convert = (lambda a: np.linalg.norm(a, axis=0)) if component == 'speed' else (lambda a: a[1])
        fields = [convert(a) for a in (truth, old_velocity, new_velocity)]
        difference = [np.abs(fields[i]-fields[0]) for i in (1, 2)]
        high = max(float(np.max(np.abs(a))) for a in fields)
        low = 0 if component == 'speed' else -high
        for column, (values, title) in enumerate(zip(fields, ('Measured reference', 'Old continuous', 'New continuous'))):
            artist = axes[row, column].scatter(x, y, c=values, marker='s', s=7,
                vmin=low, vmax=high, cmap='viridis' if row == 0 else 'RdBu_r')
            axes[row, column].set(title=f'{title}: {component}', aspect='equal', xlabel='x', ylabel='y')
        figure.colorbar(artist, ax=list(axes[row, :3]), shrink=.65)
        error_max = max(float(a.max()) for a in difference)
        for column, (values, title) in enumerate(zip(difference, ('Old absolute error', 'New absolute error')), 3):
            artist = axes[row, column].scatter(x, y, c=values, marker='s', s=7,
                                               vmin=0, vmax=error_max, cmap='magma')
            axes[row, column].set(title=f'{title}: {component}', aspect='equal', xlabel='x', ylabel='y')
        figure.colorbar(artist, ax=list(axes[row, 3:]), shrink=.65)
    figure.suptitle(f'Paired validation Case 0: {reference["case_ids"][0]}, t={reference["time"][frame]:.5g}\n'
                   'Speed error is |predicted speed - measured speed|; v error is |predicted v - measured v|')
    figure.savefig(out / 'paired_validation_case0.png', dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    for row, component in enumerate(('speed', 'v')):
        values = [np.linalg.norm(a['generated'][frame], axis=0) if row == 0 else a['generated'][frame, 1]
                  for a in (old, new)]
        high = max(float(np.max(np.abs(a))) for a in values)
        for column, (value, label) in enumerate(zip(values, ('Old', 'New'))):
            artist = axes[row, column].scatter(x, y, c=value, marker='s', s=7,
                vmin=0 if row == 0 else -high, vmax=high, cmap='viridis' if row == 0 else 'RdBu_r')
            axes[row, column].set(title=f'{label} unpaired draw 0: {component}', aspect='equal', xlabel='x', ylabel='y')
        figure.colorbar(artist, ax=list(axes[row]), shrink=.65)
    figure.suptitle(f'Independent Generator draws; no paired prediction error, t={reference["time"][frame]:.5g}')
    figure.savefig(out / 'unpaired_generator_samples.png', dpi=160)
    plt.close(figure)


def markdown(report):
    old, new = report['old'], report['new']
    lines = ['# Measured NS representation comparison', '',
        'The same validation observations and ROI are compared. Generator draws are unpaired; no generated-to-measured trajectory error is computed.', '',
        f'Old PCA rank: {old["selected_pca_rank"]}; new PCA rank: {new["selected_pca_rank"]}. Both use existing raw256 draws.', '',
        '| Paired stage | Component | Old mean L2 | New mean L2 | Old median L2 | New median L2 |',
        '|---|---|---:|---:|---:|---:|']
    for stage in ('validation_continuous_to_measured', 'validation_pca_to_continuous', 'validation_pca_to_measured'):
        for component in ('overall', 'u', 'v'):
            a, b = [r['paired_relative_l2'][stage][component] for r in (old, new)]
            lines.append(f'| {stage.replace("_", " ")} | {component} | {a["mean"]:.3%} | {b["mean"]:.3%} | {a["median"]:.3%} | {b["median"]:.3%} |')
    lines += ['', 'Each window RMS averages squared velocity over time and spatial sites. The reported standard deviation is across window RMS values (population std, ddof=0), not velocity fluctuation RMS or uncertainty in the mean.', '',
        '| Window population | Component | Old RMS mean | New RMS mean | Old RMS std | New RMS std |',
        '|---|---|---:|---:|---:|---:|']
    for stage in old['window_component_rms']:
        for component in ('u', 'v'):
            a, b = [r['window_component_rms'][stage][component] for r in (old, new)]
            lines.append(f'| {stage.replace("_", " ")} | {component} | {a["mean"]:.6g} | {b["mean"]:.6g} | {a["std_across_windows"]:.6g} | {b["std_across_windows"]:.6g} |')
    lines += ['', 'Read the stages separately: continuous vs measured diagnoses the representation; PCA vs continuous diagnoses compression; generated RMS statistics vs training PCA diagnose a limited aspect of Flow distribution matching. RMS summaries do not establish full distribution fidelity, and stage errors are not additive.', '',
        'Generator validation was available for configuration/rank selection, so these are diagnostic validation results, not untouched final-test evidence. Windows may overlap within a split; their count is not an independent sample count. Pressure and forcing belong to the manufactured NS task, not independently measured experimental truth.', '',
        '![Paired validation](paired_validation_case0.png)', '',
        '![Unpaired Generator draws](unpaired_generator_samples.png)', '']
    return '\n'.join(lines)


def run(args):
    old_path = Path(args.old_prepared).expanduser().resolve(strict=True)
    new_path = Path(args.new_prepared).expanduser().resolve(strict=True)
    out = Path(args.out).expanduser().resolve()
    if out.exists() or any(out == p or p in out.parents for p in (old_path, new_path)):
        raise ValueError('choose a new output directory outside both prepared asset directories')
    if args.batch_cases < 1 or args.threads < 1:
        raise ValueError('batch_cases and threads must be positive')
    import torch
    torch.set_num_threads(args.threads)
    old_config, old_contract, old_ref = load_reference(old_path)
    new_config, new_contract, new_ref = load_reference(new_path)
    for key in ('x', 'y', 'time', 'measured'):
        if not np.array_equal(old_ref[key], new_ref[key]):
            raise ValueError(f'old/new validation {key} differ; paired comparison requires identical observations and ordering')
    if old_ref['case_ids'] != new_ref['case_ids'] or old_contract['geometry'] != new_contract['geometry']:
        raise ValueError('old/new validation Case IDs or declared geometry differ')
    if old_config.get('coordinate_units') != new_config.get('coordinate_units'):
        raise ValueError('old/new coordinate units differ')
    old, old_fields, old_ids = summarize_asset(old_path, old_config, old_contract, old_ref,
                                             batch_cases=args.batch_cases, device=args.device)
    new, new_fields, new_ids = summarize_asset(new_path, new_config, new_contract, new_ref,
                                             batch_cases=args.batch_cases, device=args.device)
    if old_ids != new_ids:
        raise ValueError('old/new training Case IDs/order differ')
    for key in ('integration_steps', 'flow_state', 'sample_count'):
        if old['generation'][key] != new['generation'][key]:
            raise ValueError(f'old/new sampling protocol differs in {key}')
    report = {'schema': 'ns_measured_representation_change_v1', 'old': old, 'new': new,
        'paired_validation_case_id': old_ref['case_ids'][0], 'paired_frame_index': len(old_ref['time'])//2,
        'rms_definition': 'sqrt(mean(velocity_component**2 over time and spatial sites)), then mean/std across windows',
        'std_ddof': 0, 'generated_samples_are_unpaired': True,
        'scope': 'generator train/validation diagnostics only; no verifier/final data or new fitting/sampling',
        'plots': ['paired_validation_case0.png', 'unpaired_generator_samples.png']}
    out.mkdir(parents=True)
    paired_plot(out, old_ref, old_fields, new_fields)
    write_json(out / 'comparison.json', report)
    (out / 'comparison.md').write_text(markdown(report))
    print(json.dumps({'report': str(out/'comparison.json'), 'summary': str(out/'comparison.md')}, indent=2), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-prepared', required=True)
    p.add_argument('--new-prepared', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--batch-cases', type=int, default=32)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=1)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
