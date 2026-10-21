#!/usr/bin/env python3
"""Compare exact NS public queries against the unchanged autograd reference.

No Agent, sampling, training or bank writes. Run GPU/full workloads in a scheduled
allocation. Reports cold calls, alternating warm timings and full-value parity;
these timings are not whole-experiment speedups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--strata', default='base,guided,base_perturbed,guided_perturbed')
    parser.add_argument('--cases-per-stratum', type=int, default=1)
    parser.add_argument('--point-counts', default='32,512,4096')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--time-patterns', default='varying,uniform')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    counts = tuple(int(n) for n in args.point_counts.split(','))
    time_patterns = tuple(args.time_patterns.split(','))
    if not time_patterns or any(p not in {'varying', 'uniform'} for p in time_patterns):
        parser.error('time patterns must be varying and/or uniform')
    if min(*counts, args.cases_per_stratum, args.repeats, args.threads) < 1:
        parser.error('counts must be positive')
    if args.output.exists():
        raise FileExistsError('use a fresh output path')
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / 'src'), str(root / 'legacy/ns/src')]
    import numpy as np
    import torch
    from agent_verifier.neural_generator.structured_ns_answer_first import load_structured_continuous_case
    from pde_agent_verifier_solver.plugins.ns.continuous_backend import StructuredNSMMSAnswer

    class ReferenceAnswer(StructuredNSMMSAnswer):
        def _evaluate_validated_forcing(self, coordinates):
            with self._query_context():
                q = self._field.quantities(coordinates[:, (1, 2, 0)])
                return np.stack((q['forcing_u'].detach().cpu().numpy(),
                                 q['forcing_v'].detach().cpu().numpy()), axis=-1)

        def _evaluate_validated_velocity(self, coordinates):
            with self._query_context():
                u, v, _ = self._field.velocity_pressure(coordinates[:, (1, 2, 0)])
                return np.stack((u.detach().cpu().numpy(), v.detach().cpu().numpy()), axis=-1)

    torch.set_num_threads(args.threads)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('requested GPU measurement requires an allocation')
    bank = args.bank.resolve()
    data = json.loads(bank.read_text())
    strata = args.strata.split(',')
    rows = []
    for stratum in strata:
        chosen = [r for r in data['cases'] if r['stratum'] == stratum][:args.cases_per_stratum]
        if len(chosen) != args.cases_per_stratum:
            raise ValueError(f'bank has insufficient cases in {stratum}')
        rows.extend(chosen)
    source_paths = [
        'src/pde_agent_verifier_solver/plugins/ns/continuous_backend.py',
        'legacy/ns/src/agent_verifier/neural_generator/structured_ns_answer_first.py',
        'legacy/ns/src/agent_verifier/neural_generator/triplane_mms.py',
        'legacy/ns/src/agent_verifier/neural_generator/continuous_geometry.py',
        'legacy/ns/src/agent_verifier/neural_generator/structured_ns_analytic.py',
        'legacy/ns/src/agent_verifier/neural_generator/structured_ns_geometry_jets.py',
        'scripts/benchmark_ns_analytic_derivatives.py',
    ]
    report = {'scope': 'Public query call parity and latency; includes validation and CPU output copy; no RPC/Agent/score',
              'status': 'running', 'device': args.device, 'torch_version': torch.__version__,
              'device_name': torch.cuda.get_device_name() if args.device.startswith('cuda') else 'CPU',
              'threads': args.threads, 'job_id': os.environ.get('SLURM_JOB_ID'),
              'bank': str(bank), 'bank_sha256': digest(bank), 'seed': 101,
              'rtol': 2e-10, 'atol': 1e-9, 'repeats': args.repeats,
              'source_sha256': {p: digest(root / p) for p in source_paths}, 'rows': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    def sync():
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()

    for index, row in enumerate(rows):
        state = (bank.parent / row['state_path']).resolve()
        if not state.is_relative_to(bank.parent):
            raise ValueError('saved field identity changed')
        field = load_structured_continuous_case(state.parent, device=args.device)
        before, after = ReferenceAnswer(field), StructuredNSMMSAnswer(field)
        if not getattr(after, '_analytic_field', False):
            raise RuntimeError('optimized public answer did not select the analytic path')
        x0, x1, y0, y1 = after.domain_bounds
        for count in counts:
            rng = np.random.default_rng(101 + index * 997 + count)
            points = np.column_stack((rng.uniform(*after.time_interval, count),
                                      rng.uniform(x0, x1, count), rng.uniform(y0, y1, count)))
            special = [(after.time_interval[0], x0, y0), (after.time_interval[1], x1, y1)]
            special.extend((sum(after.time_interval) / 2, x, y)
                           for polygon in field.geometry_spec.obstacle_polygons for x, y in polygon)
            points[:min(count, len(special))] = special[:count]
            for kind, pattern in [('initial', 'uniform')] + [('forcing', p) for p in time_patterns]:
                query = points.copy()
                if kind == 'initial':
                    query[:, 0] = after.time_interval[0]
                elif pattern == 'uniform':
                    query[:, 0] = sum(after.time_interval) / 2
                callbacks = {name: (lambda answer=answer: answer.evaluate_velocity(query))
                             if kind == 'initial' else
                             (lambda answer=answer: answer.evaluate_forcing(query))
                             for name, answer in (('reference', before), ('analytic', after))}
                cold, outputs = {}, {}
                for name, callback in callbacks.items():
                    sync()
                    start = perf_counter()
                    outputs[name] = callback()
                    sync()
                    cold[name] = perf_counter() - start
                assert np.isfinite(outputs['reference']).all() and np.isfinite(outputs['analytic']).all()
                np.testing.assert_allclose(outputs['analytic'], outputs['reference'], rtol=2e-10, atol=1e-9)
                samples = {name: [] for name in callbacks}
                for repeat in range(args.repeats):
                    order = ('reference', 'analytic') if repeat % 2 == 0 else ('analytic', 'reference')
                    for name in order:
                        sync()
                        start = perf_counter()
                        callbacks[name]()
                        sync()
                        samples[name].append(perf_counter() - start)
                medians = {name: statistics.median(values) for name, values in samples.items()}
                result = {'case_id': row['case']['case_id'], 'stratum': row['stratum'], 'kind': kind,
                          'points': count, 'time_pattern': pattern, 'cold_seconds': cold, 'seconds': samples, 'median_seconds': medians,
                          'speedup': medians['reference'] / medians['analytic'],
                          'maximum_absolute_error': float(np.max(np.abs(outputs['analytic'] - outputs['reference']))),
                          'reference_output_sha256': hashlib.sha256(outputs['reference'].tobytes()).hexdigest(),
                          'analytic_output_sha256': hashlib.sha256(outputs['analytic'].tobytes()).hexdigest()}
                report['rows'].append(result)
                save()
                print(json.dumps({k: result[k] for k in ('stratum', 'kind', 'time_pattern', 'points', 'median_seconds', 'speedup', 'maximum_absolute_error')}), flush=True)
    report['status'] = 'passed'
    save()


if __name__ == '__main__':
    main()
