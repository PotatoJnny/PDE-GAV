#!/usr/bin/env python3
"""Compare frozen and current NS generator internals without running an Agent.

Provide the original scientific modules in --baseline-dir. This benchmark keeps
checkpoint weights, random inputs, precision, integration method and steps fixed.
It times actual forcing queries and frozen reward-guided flow sampling separately.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--guidance-checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1:
        parser.error('repeats and threads must be positive')
    sys.path[:0] = [str(root / 'legacy/ns/src'), str(root / 'src')]
    import numpy as np
    import torch
    from agent_verifier.neural_generator import structured_ns_guidance as current_guidance
    from pde_agent_verifier_solver.plugins.ns.structured_flow_runtime import LegacyStructuredNSFlowSampler

    torch.set_num_threads(args.threads)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA allocation required for requested GPU measurement')
    scientific = root / 'legacy/ns/src/agent_verifier/neural_generator'
    names = ['continuous_geometry.py', 'structured_ns_answer_first.py',
             'structured_ns_guidance.py', 'structured_ns_distribution.py']
    old_geometry = load_module('_latency_old_geometry', args.baseline_dir / names[0])
    old_mms = load_module('_latency_old_mms', args.baseline_dir / names[1])
    old_guidance = load_module('_latency_old_guidance', args.baseline_dir / names[2])
    old_flow = load_module('_latency_old_flow', args.baseline_dir / names[3])
    old_mms.continuous_no_slip_masks = old_geometry.continuous_no_slip_masks
    old_mms.legacy_c3_continuous_no_slip_masks = old_geometry.legacy_c3_continuous_no_slip_masks
    assets = root / 'assets/generators/ns/structured_flow_v1'
    sampler = LegacyStructuredNSFlowSampler(
        flow_checkpoint=assets / 'flow_checkpoint.pt',
        pca_checkpoint=assets / 'pca_checkpoint.pt',
        time_interval=(0.0, 0.12), viscosity=0.01, device=args.device,
        integration_steps=256, flow_state='raw', integration_method='heun',
    )
    flow_checkpoint = torch.load(assets / 'flow_checkpoint.pt', map_location='cpu', weights_only=False)
    before_flow = old_flow.StructuredCoefficientRectifiedFlow.build(
        old_flow.StructuredCoefficientFlowConfig.from_dict(flow_checkpoint['model_config'])
    ).to(args.device)
    before_flow.load_state_dict(flow_checkpoint['model_state_dict'])
    before_flow.eval()
    flow_models = {'before': before_flow, 'after': sampler._model}
    checkpoint = torch.load(args.guidance_checkpoint, map_location=args.device, weights_only=False)
    model = current_guidance.StructuredRewardModel.build(
        current_guidance.StructuredRewardModelConfig.from_dict(checkpoint['model_config'])
    ).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    strength = float(checkpoint['guidance_strength'])
    cap = float(checkpoint['guidance_max_flow_ratio'])

    def sync():
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()

    def measure(callbacks):
        for callback in callbacks.values():
            callback()
        sync()
        samples = {name: [] for name in callbacks}
        for repeat in range(args.repeats):
            order = ('before', 'after') if repeat % 2 == 0 else ('after', 'before')
            for name in order:
                sync()
                started = time.perf_counter()
                callbacks[name]()
                sync()
                samples[name].append(time.perf_counter() - started)
        medians = {name: statistics.median(values) for name, values in samples.items()}
        return {'seconds': samples, 'median_seconds': medians,
                'speedup': medians['before'] / medians['after']}

    report = {
        'schema': 'ns_generator_latency_v1', 'status': 'running',
        'device': args.device, 'device_name': (torch.cuda.get_device_name() if args.device.startswith('cuda') else 'CPU'),
        'torch_version': torch.__version__, 'threads': args.threads,
        'seed': 101, 'repeats': args.repeats, 'job_id': os.environ.get('SLURM_JOB_ID'),
        'source_sha256': {name: {'before': digest(args.baseline_dir / name),
                                'after': digest(scientific / name)} for name in names},
        'benchmark_source_sha256': digest(__file__),
        'checkpoint_sha256': {name: digest(path) for name, path in {
            'flow': assets / 'flow_checkpoint.pt', 'pca': assets / 'pca_checkpoint.pt',
            'guidance': args.guidance_checkpoint}.items()},
        'settings': {'integration_steps': 256, 'integration_method': 'heun',
                     'flow_state': 'raw', 'guidance_strength': strength,
                     'guidance_max_flow_ratio': cap, 'flow_dtype': 'float32',
                     'oracle_dtype': 'float64'},
        'scope': 'GPU kernel/internal call microbenchmark; excludes Agent, RPC, queueing, training, serialization and Official audit; no formal experiment or API calls',
        'forcing': [], 'base_sampling': [], 'guided_sampling': [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    # Use an actual Base-flow sample, with identical coefficients for both paths.
    field = sampler.sample_many((101,))[0]
    fields = {'before': old_mms.ContinuousStructuredNSMMS(
        field.coefficients, config=field.config, geometry_spec=field.geometry_spec,
        domain_bounds=field.domain_bounds, viscosity=field.viscosity,
        t0=field.t0, t1=field.t1, device=args.device), 'after': field}
    lower_x, upper_x, lower_y, upper_y = field.domain_bounds
    for count in (32, 128, 512, 4096):
        points = np.random.default_rng(101 + count).random((count, 3))
        points[:, 0] = lower_x + points[:, 0] * (upper_x - lower_x)
        points[:, 1] = lower_y + points[:, 1] * (upper_y - lower_y)
        points[:, 2] *= 0.12
        special = [(lower_x, .5, 0.), (upper_x, .5, .12),
                   (.5, lower_y, .06), (.5, upper_y, .06)]
        special += [(x, y, .06) for polygon in field.geometry_spec.obstacle_polygons for x, y in polygon]
        points[:len(special)] = special
        outputs = {name: value.quantities(points, include_laplacian_components=True)
                   for name, value in fields.items()}
        errors = {}
        for key, value in outputs['before'].items():
            before = value.detach().cpu().numpy()
            after = outputs['after'][key].detach().cpu().numpy()
            assert np.isfinite(before).all() and np.isfinite(after).all(), key
            np.testing.assert_allclose(after, before, rtol=2e-10, atol=1e-9, err_msg=key)
            errors[key] = float(np.max(np.abs(after - before)))

        def forcing(value):
            output = value.quantities(points, include_laplacian_components=False)
            return torch.stack((output['forcing_u'], output['forcing_v']), dim=-1).detach().cpu().numpy()

        np.testing.assert_allclose(forcing(fields['after']), forcing(fields['before']), rtol=2e-10, atol=1e-9)
        row = {'points': count, 'max_absolute_errors_by_quantity': errors,
               'equivalence_rtol': 2e-10, 'equivalence_atol': 1e-9,
               **measure({name: lambda value=value: forcing(value) for name, value in fields.items()})}
        report['forcing'].append(row)
        save()
        print(json.dumps({'stage': 'forcing', 'points': count, 'median_seconds': row['median_seconds'], 'speedup': row['speedup']}), flush=True)

    for kind in ('base', 'guided'):
        for count in (1, 16):
            random = torch.Generator(device=args.device).manual_seed(101 + count)
            initial = torch.randn((count, int(sampler.pca.rank)), device=args.device, generator=random)

            def sample(name):
                gradient_type = (old_guidance if name == 'before' else current_guidance).StructuredRewardGradient
                gradient = (gradient_type(model, strength=strength, max_flow_ratio=cap)
                            if kind == 'guided' else None)
                with torch.no_grad():
                    scores = flow_models[name].sample(count, steps=256, device=args.device,
                        initial_values=initial, guidance=gradient,
                        guidance_scale=1.0 if gradient is not None else 0.0, method='heun')
                return scores, gradient

            outputs = {name: sample(name) for name in flow_models}
            torch.testing.assert_close(outputs['after'][0], outputs['before'][0], rtol=0, atol=0)
            # Decoding is shared and unchanged; check the actual field coefficients too.
            with torch.no_grad():
                before_coefficients = sampler.pca.decode(outputs['before'][0])
                after_coefficients = sampler.pca.decode(outputs['after'][0])
            torch.testing.assert_close(after_coefficients, before_coefficients, rtol=0, atol=0)
            if kind == 'guided':
                assert outputs['after'][1].diagnostics() == outputs['before'][1].diagnostics()
            row = {'cases': count, 'integration_stages': 511,
                   'bitwise_identical_coefficients': True,
                   'identical_diagnostics': True if kind == 'guided' else None,
                   **measure({'before': lambda: sample('before'), 'after': lambda: sample('after')})}
            report[kind + '_sampling'].append(row)
            save()
            print(json.dumps({'stage': kind + '_sampling', 'cases': count,
                              'median_seconds': row['median_seconds'], 'speedup': row['speedup']}), flush=True)
    report['status'] = 'passed'
    save()


if __name__ == '__main__':
    main()
