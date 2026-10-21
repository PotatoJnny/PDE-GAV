#!/usr/bin/env python3
"""Benchmark actual NS sampler calls with request-local CUDA graphs on/off.

Includes every graph construction, sampling, PCA decoding and field creation.
This makes no Agent/API calls and does not restart any formal experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--guidance-checkpoint', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / 'legacy/ns/src'), str(root / 'src')]
    started = time.perf_counter()
    import torch
    from agent_verifier.neural_generator.structured_ns_guidance import (
        load_structured_reward_guidance_checkpoint, StructuredRewardGradient,
    )
    from pde_agent_verifier_solver.plugins.ns.structured_flow_runtime import LegacyStructuredNSFlowSampler
    imports_seconds = time.perf_counter() - started
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('This comparison requires an allocated CUDA GPU')
    assets = root / 'assets/generators/ns/structured_flow_v1'
    started = time.perf_counter()
    samplers = {
        name: LegacyStructuredNSFlowSampler(
            flow_checkpoint=assets / 'flow_checkpoint.pt', pca_checkpoint=assets / 'pca_checkpoint.pt',
            time_interval=(0., .12), viscosity=.01, device='cuda', integration_steps=256,
            flow_state='raw', integration_method='heun', cuda_graph_sampling=(name == 'graph'),
        ) for name in ('eager', 'graph')
    }
    reward, payload = load_structured_reward_guidance_checkpoint(args.guidance_checkpoint, map_location='cuda')
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - started
    files = [
        'legacy/ns/src/agent_verifier/neural_generator/continuous_geometry.py',
        'legacy/ns/src/agent_verifier/neural_generator/structured_ns_answer_first.py',
        'legacy/ns/src/agent_verifier/neural_generator/structured_ns_distribution.py',
        'legacy/ns/src/agent_verifier/neural_generator/structured_ns_guidance.py',
        'src/pde_agent_verifier_solver/plugins/ns/structured_flow_runtime.py',
        'src/pde_agent_verifier_solver/plugins/ns/continuous_backend.py',
        'src/pde_agent_verifier_solver/plugins/ns/flow_cuda_graph.py',
        'src/pde_agent_verifier_solver/plugins/ns/canonical_backend.py',
        'scripts/run_canonical_ns_campaign.py',
    ]
    report = {
        'schema': 'ns_cuda_graph_sampler_benchmark_v1', 'status': 'running',
        'job_id': os.environ.get('SLURM_JOB_ID'), 'node': os.environ.get('SLURMD_NODENAME'),
        'gpu': torch.cuda.get_device_name(), 'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda, 'threads': torch.get_num_threads(),
        'seed': 101, 'repeats_small_batches': args.repeats,
        'settings': {'integration_steps': 256, 'integration_method': 'heun', 'flow_state': 'raw',
                     'flow_dtype': 'float32', 'field_dtype': 'float64',
                     'guidance_strength': float(payload['guidance_strength']),
                     'guidance_max_flow_ratio': float(payload['guidance_max_flow_ratio'])},
        'source_sha256': {name: digest(root / name) for name in files},
        'benchmark_source_sha256': digest(__file__),
        'checkpoint_sha256': {'flow': digest(assets / 'flow_checkpoint.pt'),
                              'pca': digest(assets / 'pca_checkpoint.pt'),
                              'guidance': digest(args.guidance_checkpoint)},
        'startup': {'imports_seconds': imports_seconds, 'two_samplers_and_reward_load_seconds': setup_seconds},
        'scope': 'Actual LegacyStructuredNSFlowSampler call wall time, synchronized, including fresh graph capture on every graph call, integration, PCA decoding and continuous field construction. Excludes checkpoint loading, Agent/RPC, forcing evaluation and training. Eager is the previously optimized implementation.',
        'rows': [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    counters = {'captures': 0, 'replays': 0}
    original_capture = torch.cuda.CUDAGraph.capture_begin
    original_replay = torch.cuda.CUDAGraph.replay

    def capture(graph, *values, **settings):
        counters['captures'] += 1
        return original_capture(graph, *values, **settings)

    def replay(graph, *values, **settings):
        counters['replays'] += 1
        return original_replay(graph, *values, **settings)

    torch.cuda.CUDAGraph.capture_begin = capture
    torch.cuda.CUDAGraph.replay = replay

    def rng_state():
        return torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()

    def sample(name, kind, count):
        gradient = StructuredRewardGradient(
            reward, strength=payload['guidance_strength'], max_flow_ratio=payload['guidance_max_flow_ratio'],
        ) if kind == 'guided' else None
        seeds = tuple(101 + index for index in range(count))
        before_rng = rng_state()
        before = dict(counters)
        torch.cuda.synchronize()
        started = time.perf_counter()
        if kind == 'guided':
            fields = samplers[name].sample_many_with_guidance(seeds, guidance=gradient, guidance_scale=1.)
        else:
            fields = samplers[name].sample_many(seeds)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        calls = {key: counters[key] - before[key] for key in counters}
        after_rng = rng_state()
        assert all(torch.equal(left, right) for left, right in zip(before_rng, after_rng)), 'Global RNG changed'
        if name == 'graph':
            assert calls['captures'] == 1, ('Expected one new capture on every call', calls)
            assert calls['replays'] == 511, ('Graph path not used for all Heun stages', calls)
        else:
            assert calls == {'captures': 0, 'replays': 0}, calls
        assert len(fields) == count
        return fields, gradient.diagnostics() if gradient else None, seconds, calls

    def equivalent(before, after):
        fields_before, diagnostics_before, _, _ = before
        fields_after, diagnostics_after, _, _ = after
        assert diagnostics_before == diagnostics_after, 'Guidance diagnostics differ'
        for left, right in zip(fields_before, fields_after, strict=True):
            assert torch.equal(left.coefficients.psi, right.coefficients.psi), 'Streamfunction coefficients differ'
            assert torch.equal(left.coefficients.pressure, right.coefficients.pressure), 'Pressure coefficients differ'
            assert left.config.to_dict() == right.config.to_dict()
            assert left.geometry_spec.to_dict() == right.geometry_spec.to_dict()
            assert left.domain_bounds == right.domain_bounds
            assert (left.viscosity, left.t0, left.t1) == (right.viscosity, right.t0, right.t1)

    try:
        for kind in ('base', 'guided'):
            for count in ((1, 16, 64, 512, 1024) if kind == 'base' else (1, 16, 64, 512)):
                repetitions = args.repeats if count <= 16 else 1
                timings = {'eager': [], 'graph': []}
                observed = {'eager': [], 'graph': []}
                first_calls = {}
                if count <= 16:
                    before = sample('eager', kind, count)
                    after = sample('graph', kind, count)
                    equivalent(before, after)
                    first_calls = {'eager': before[2], 'graph': after[2]}
                    del before, after
                for repeat in range(repetitions):
                    order = ('eager', 'graph') if repeat % 2 == 0 else ('graph', 'eager')
                    result = {}
                    for name in order:
                        result[name] = sample(name, kind, count)
                        timings[name].append(result[name][2])
                        observed[name].append(result[name][3])
                    equivalent(result['eager'], result['graph'])
                    del result
                medians = {name: statistics.median(values) for name, values in timings.items()}
                row = {'kind': kind, 'cases': count, 'repetitions': repetitions,
                       'first_untimed_calls_seconds': first_calls, 'seconds': timings,
                       'median_seconds': medians, 'speedup': medians['eager'] / medians['graph'],
                       'observed_cuda_graph_operations': observed,
                       'coefficients_bitwise_equal': True, 'diagnostics_equal': True,
                       'global_cpu_cuda_rng_unchanged': True,
                       'large_batch_note': 'Single equivalence/latency observation; no stable speedup estimate' if count > 16 else None}
                report['rows'].append(row)
                save()
                print(json.dumps({key: row[key] for key in ('kind', 'cases', 'median_seconds', 'speedup', 'repetitions')}), flush=True)
        report['status'] = 'passed'
        save()
    finally:
        torch.cuda.CUDAGraph.capture_begin = original_capture
        torch.cuda.CUDAGraph.replay = original_replay


if __name__ == '__main__':
    main()
