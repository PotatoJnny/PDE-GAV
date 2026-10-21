#!/usr/bin/env python3
"""Replay saved NS public queries to diagnose CUDA autograd starvation.

Use a scheduled allocation for the full workload. No Agent, generation, scoring,
training or source-bank writes occur. --source-root permits old/new replay of
identical fields and queries. --evaluation-pool uses the actual scoped public
Oracle adapter, including its optional analytic graph cache. This diagnostic
checks numerical outputs but does not produce paper scores.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import faulthandler
import hashlib
import json
from pathlib import Path
import sys
from threading import Lock
from time import perf_counter
import traceback


def _graph_cache_diagnostics(resources):
    result = []
    for resource in resources.values():
        cache = getattr(resource, '_analytic_cuda_cache', None)
        diagnostics = getattr(cache, 'diagnostics', None)
        result.append({'device': str(resource.device), 'available': cache is not None,
                       'diagnostics': None if diagnostics is None else dict(diagnostics)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--stratum', default='guided')
    parser.add_argument('--workers', type=int, default=28)
    parser.add_argument('--case-count', type=int, default=56)
    parser.add_argument('--grid', type=int, default=193)
    parser.add_argument('--forcing-times', type=int, default=7)
    parser.add_argument('--evaluation-pool', action='store_true',
                        help='Use EvaluationDevicePool and public_evaluation_answer for actual scoped GPU queries.')
    args = parser.parse_args()
    if min(args.workers, args.case_count, args.grid, args.forcing_times) < 1:
        parser.error('counts must be positive')
    source, bank_path = args.source_root.resolve(), args.bank.resolve()
    sys.path[:0] = [str(source / 'src'), str(source / 'legacy/ns/src')]
    import numpy as np
    import torch
    from agent_verifier.neural_generator.structured_ns_answer_first import load_structured_continuous_case
    from pde_agent_verifier_solver.plugins.ns.continuous_backend import StructuredNSMMSAnswer
    from pde_agent_verifier_solver.sdk._candidate_timing import bind_queue_wait_observer

    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    events_path = args.output.with_suffix('.calls.jsonl')
    if args.output.exists() or events_path.exists():
        raise FileExistsError('diagnostic output already exists; use a fresh path')
    data = json.loads(bank_path.read_text())
    rows = [row for row in data['cases'] if row['stratum'] == args.stratum][:args.case_count]
    if len(rows) != args.case_count:
        raise ValueError('bank has insufficient existing Cases for the requested diagnostic')
    report = {'scope': 'public numerical query replay; no solver score or formal run',
              'source_root': str(source), 'bank': str(bank_path),
              'bank_sha256': hashlib.sha256(bank_path.read_bytes()).hexdigest(),
              'device': args.device, 'torch_version': torch.__version__,
              'workers': args.workers, 'case_count': len(rows), 'grid': args.grid,
              'forcing_times': args.forcing_times, 'evaluation_pool': args.evaluation_pool,
              'calls_path': str(events_path), 'cases': [], 'status': 'preparing',
              'source_field_device': 'cpu' if args.evaluation_pool else args.device,
              'all_outputs_finite': False, 'cleanup_confirmed': None,
              'graph_cache_before_close': [], 'graph_cache_after_close': []}
    pool, resources = None, {}
    answers, queries = [], []
    answer = None
    lifecycle_started = perf_counter()
    started = None
    log_lock = Lock()
    # A long trace is progress evidence, not by itself an execution failure.
    faulthandler.dump_traceback_later(120, repeat=True)
    try:
        if args.evaluation_pool:
            from pde_agent_verifier_solver.plugins.ns.evaluation_devices import public_evaluation_answer
            from pde_agent_verifier_solver.sdk.evaluation_devices import (
                EvaluationDevicePool, bind_evaluation_device_pool,
            )
            pool = EvaluationDevicePool((args.device,))
            pool_context = bind_evaluation_device_pool(pool)
        else:
            pool_context = nullcontext()
        with pool_context:
            for row in rows:
                state = (bank_path.parent / row['state_path']).resolve()
                if not state.is_relative_to(bank_path.parent):
                    raise ValueError('saved field identity differs from the bank manifest')
                # The scoped adapter owns the borrowed GPU copy. Keeping its
                # frozen source on CPU also permits meaningful GPU cleanup.
                answer = StructuredNSMMSAnswer(load_structured_continuous_case(
                    state.parent, device=report['source_field_device']))
                if pool is not None:
                    answer, resource = public_evaluation_answer(row['case']['case_id'], answer)
                    if resource is None:
                        raise RuntimeError('evaluation-pool diagnostic did not acquire an NS GPU resource')
                    resources[id(resource)] = resource
                xmin, xmax, ymin, ymax = answer.domain_bounds
                xx, yy = np.meshgrid(np.linspace(xmin, xmax, args.grid),
                                     np.linspace(ymin, ymax, args.grid), indexing='xy')
                xy = np.column_stack((xx.ravel(), yy.ravel()))
                initial = np.column_stack((np.full(len(xy), answer.time_interval[0]), xy))
                plan = [('initial', initial[start:start + 4096]) for start in range(0, len(xy), 4096)]
                plan += [('forcing', np.column_stack((np.full(len(xy[start:start + 4096]), time),
                                                       xy[start:start + 4096])))
                         for time in np.linspace(*answer.time_interval, args.forcing_times)
                         for start in range(0, len(xy), 4096)]
                answers.append(answer)
                queries.append(plan)
        report.update(setup_seconds=perf_counter() - lifecycle_started, status='running',
                      calls_per_case=len(queries[0]), expected_calls=sum(map(len, queries)))
        started = perf_counter()
        with events_path.open('x', buffering=1) as stream:
            def run(index):
                answer = answers[index]
                totals = Counter()
                for call_index, (kind, points) in enumerate(queries[index]):
                    waiting = [0, 0., 0.]
                    def observe(active):
                        now = perf_counter()
                        if active:
                            if not waiting[0]:
                                waiting[1] = now
                            waiting[0] += 1
                        else:
                            waiting[0] -= 1
                            if not waiting[0]:
                                waiting[2] += now - waiting[1]
                    call_started = perf_counter()
                    with bind_queue_wait_observer(observe):
                        values = (answer.evaluate_velocity(points) if kind == 'initial'
                                  else answer.evaluate_forcing(points))
                    elapsed = perf_counter() - call_started
                    values = np.asarray(values, dtype=np.float64)
                    if values.shape != (len(points), 2) or not np.isfinite(values).all():
                        raise ValueError(f'Case {index} call {call_index} returned invalid/nonfinite NS values')
                    row = {'case_index': index, 'case_id': rows[index]['case']['case_id'],
                           'call_index': call_index, 'kind': kind, 'points': len(points),
                           'elapsed_seconds': elapsed, 'controller_wait_seconds': waiting[2],
                           'charged_service_seconds': max(0., elapsed - waiting[2]),
                           'finished_after_seconds': perf_counter() - started,
                           'all_values_finite': True,
                           'output_sha256': hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()}
                    with log_lock:
                        stream.write(json.dumps(row) + '\n')
                    totals.update(calls=1, points=len(points), elapsed_seconds=elapsed,
                                  controller_wait_seconds=waiting[2],
                                  charged_service_seconds=row['charged_service_seconds'])
                return {'case_index': index, 'case_id': rows[index]['case']['case_id'], **totals}
            # __exit__ joins every query thread even when map raises. The outer
            # finally therefore never frees graphs while a thread uses them.
            with ThreadPoolExecutor(max_workers=args.workers) as workers:
                report['cases'] = list(workers.map(run, range(len(rows))))
        report.update(query_wall_seconds=perf_counter() - started, status='passed',
                      all_outputs_finite=True)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        failure_in_progress = sys.exc_info()[0] is not None
        if started is not None:
            report.setdefault('query_wall_seconds', perf_counter() - started)
        else:
            report.setdefault('setup_seconds', perf_counter() - lifecycle_started)
        # Default mode keeps its historical query-stage wall measurement. Pool
        # setup/cleanup are additional explicit costs, never hidden in speedups.
        report['total_wall_seconds'] = report.get('query_wall_seconds', 0.0)
        try:
            if pool is not None:
                try:
                    report['graph_cache_before_close'] = _graph_cache_diagnostics(resources)
                finally:
                    # Diagnostic code must not retain borrowed field copies
                    # after the device owner releases its own answer cache.
                    answer = None
                    answers.clear()
                    cleanup_started = perf_counter()
                    try:
                        pool.close()
                    finally:
                        report['pool_cleanup_seconds'] = perf_counter() - cleanup_started
                        report['cleanup_confirmed'] = pool.cleanup_confirmed
                        report['graph_cache_after_close'] = _graph_cache_diagnostics(resources)
        except BaseException:
            report.update(status='failed', cleanup_error=traceback.format_exc())
            if not failure_in_progress:
                raise
        finally:
            faulthandler.cancel_dump_traceback_later()
            report['lifecycle_wall_seconds'] = perf_counter() - lifecycle_started
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'cases'}), flush=True)


if __name__ == '__main__':
    main()
