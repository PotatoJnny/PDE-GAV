#!/usr/bin/env python3
"""Profile frozen NS Guided sampling; no Agent, training, or dataset mutation."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

BOOT = time.perf_counter()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path, source=None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if source is None:
        spec.loader.exec_module(module)
    else:
        exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def wrap_region(source, start, end, label):
    left = source.index(start)
    right = source.index(end, left)
    block = source[left:right]
    indent = len(block.splitlines()[0]) - len(block.splitlines()[0].lstrip())
    replacement = (' ' * indent + 'with _phase(' + repr(label) + '):\n'
                   + ''.join('    ' + line if line.strip() else line
                             for line in block.splitlines(keepends=True)))
    return source[:left] + replacement + source[right:]


def instrument_flow(source):
    for line in ('        velocity = velocity_at_time(values, time, index)\n',
                 '        next_velocity = velocity_at_time(provisional, next_time, index + 1)\n'):
        assert source.count(line) == 1
        source = source.replace(line, "        with _phase('flow_forward'):\n    " + line)
    return source


def instrument_guidance(source):
    for start, end, label in (
        ('            differentiated = values.detach().float().requires_grad_(True)\n',
         '            gradient = torch.autograd.grad(prediction.sum(), differentiated)[0]\n',
         'reward_forward'),
        ('            gradient = torch.autograd.grad(prediction.sum(), differentiated)[0]\n',
         '\n        base = base_velocity.detach().float()\n', 'reward_autograd'),
        ('        base = base_velocity.detach().float()\n',
         '        if not torch.isfinite(update).all():\n', 'guidance_normalization'),
        ('        if not torch.isfinite(update).all():\n',
         '        self._calls += 1\n', 'finite_check'),
        ('        self._calls += 1\n',
         '        return update.to(device=values.device, dtype=values.dtype)\n', 'diagnostics'),
    ):
        source = wrap_region(source, start, end, label)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--guidance-checkpoint', required=True, type=Path)
    parser.add_argument('--repeats', default=3, type=int)
    parser.add_argument('--profile-stages', default=16, type=int,
                        help='Profiler runs this many actual integration stages after warmup; steady timing always uses full256Heun.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / 'legacy/ns/src'), str(root / 'src')]
    started = time.perf_counter()
    import torch
    import numpy as np
    imports_seconds = time.perf_counter() - started
    torch.set_num_threads(1)
    started = time.perf_counter()
    torch.cuda.init()
    torch.cuda.synchronize()
    cuda_init_seconds = time.perf_counter() - started
    flow_path = args.snapshot / 'structured_ns_distribution.py'
    guidance_path = args.snapshot / 'structured_ns_guidance.py'
    flow = load_module('_guided_profile_frozen_flow', flow_path)
    guidance = load_module('_guided_profile_frozen_guidance', guidance_path)
    instrumented = args.output.parent / 'instrumented'
    instrumented.mkdir(parents=True, exist_ok=True)
    flow_source = instrument_flow(flow_path.read_text())
    guidance_source = instrument_guidance(guidance_path.read_text())
    (instrumented / flow_path.name).write_text(flow_source)
    (instrumented / guidance_path.name).write_text(guidance_source)
    timed_flow = load_module('_guided_profile_timed_flow', flow_path, flow_source)
    timed_guidance = load_module('_guided_profile_timed_guidance', guidance_path, guidance_source)
    assets = root / 'assets/generators/ns/structured_flow_v1'
    started = time.perf_counter()
    flow_payload = torch.load(assets / 'flow_checkpoint.pt', map_location='cpu', weights_only=False)
    reward_payload = torch.load(args.guidance_checkpoint, map_location='cpu', weights_only=False)
    checkpoint_read_seconds = time.perf_counter() - started
    started = time.perf_counter()

    def build(flow_module, guidance_module):
        model = flow_module.StructuredCoefficientRectifiedFlow.build(
            flow_module.StructuredCoefficientFlowConfig.from_dict(flow_payload['model_config'])
        )
        model.load_state_dict(flow_payload['model_state_dict'])
        model = model.cuda().eval()
        reward = guidance_module.StructuredRewardModel.build(
            guidance_module.StructuredRewardModelConfig.from_dict(reward_payload['model_config'])
        )
        reward.load_state_dict(reward_payload['model_state_dict'])
        reward = reward.cuda().eval()
        return model, reward

    model, reward = build(flow, guidance)
    torch.cuda.synchronize()
    model_build_seconds = time.perf_counter() - started
    settings = dict(strength=float(reward_payload['guidance_strength']),
                    max_flow_ratio=float(reward_payload['guidance_max_flow_ratio']))
    counts = (1, 16)
    values = {}
    for count in counts:
        generator = torch.Generator(device='cuda').manual_seed(101 + count)
        values[count] = torch.randn((count, model.config.feature_dim), device='cuda', generator=generator)
    report = {
        'schema': 'ns_guided_latency_profile_v1', 'status': 'running',
        'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name(),
        'threads': torch.get_num_threads(), 'seed': 101,
        'source_sha256': {path.name: digest(path) for path in args.snapshot.glob('*.py')},
        'script_sha256': digest(__file__),
        'checkpoint_sha256': {'flow': digest(assets / 'flow_checkpoint.pt'), 'reward': digest(args.guidance_checkpoint)},
        'settings': {'steps': 256, 'method': 'heun', 'flow_state': 'raw', 'dtype': 'float32', **settings},
        'startup': {'imports_seconds': imports_seconds, 'cuda_init_seconds': cuda_init_seconds,
                    'checkpoint_read_seconds': checkpoint_read_seconds, 'model_build_transfer_seconds': model_build_seconds},
        'interpretation_limits': 'Steady wall timings exclude profiler/event instrumentation. Phase CUDA-event intervals include device idle time while CPU launches work; kernel activity times come from a shorter instrumented profiler window. These are sampling-core timings, not RPC or whole Agent runs.',
        'rows': [],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    def sample(flow_model, reward_model, gradient_type, count):
        gradient = gradient_type(reward_model, **settings)
        with torch.no_grad():
            output = flow_model.sample(count, steps=256, device='cuda', initial_values=values[count],
                                      guidance=gradient, guidance_scale=1., method='heun')
        return output, gradient

    torch.cuda.synchronize()
    started = time.perf_counter()
    first_output = sample(model, reward, guidance.StructuredRewardGradient, 1)
    torch.cuda.synchronize()
    report['startup']['first_sample_1_seconds'] = time.perf_counter() - started
    report['startup']['process_to_first_sample_seconds'] = time.perf_counter() - BOOT
    print(json.dumps({'startup': report['startup']}), flush=True)
    save()

    # Instrumented modules preserve all sampling arithmetic; only phase scopes differ.
    timed_model, timed_reward = build(timed_flow, timed_guidance)

    class Recorder:
        def __init__(self):
            self.mode = 'off'
            self.records = defaultdict(list)
            self.profiler = None

        @contextmanager
        def phase(self, name):
            if self.mode == 'events':
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                start = time.perf_counter()
                try:
                    yield
                finally:
                    host = time.perf_counter() - start
                    end.record()
                    self.records[name].append((host, begin, end))
            elif self.mode == 'profile':
                with torch.profiler.record_function('ns::' + name):
                    yield
                if name == 'diagnostics' and self.profiler is not None:
                    self.profiler.step()
            else:
                yield

    recorder = Recorder()
    timed_flow._phase = recorder.phase
    timed_guidance._phase = recorder.phase
    references = {}
    for count in counts:
        reference, reference_gradient = sample(model, reward, guidance.StructuredRewardGradient, count)
        references[count] = reference
        instrumented_output, instrumented_gradient = sample(timed_model, timed_reward, timed_guidance.StructuredRewardGradient, count)
        assert torch.equal(reference, instrumented_output)
        assert reference_gradient.diagnostics() == instrumented_gradient.diagnostics()
        samples = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            started = time.perf_counter()
            sample(model, reward, guidance.StructuredRewardGradient, count)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - started)
        row = {'cases': count, 'steady_seconds': samples, 'steady_median_seconds': statistics.median(samples),
               'instrumented_outputs_bitwise_equal': True, 'instrumented_diagnostics_equal': True}
        recorder.mode = 'events'
        recorder.records.clear()
        torch.cuda.synchronize()
        started = time.perf_counter()
        event_output, _ = sample(timed_model, timed_reward, timed_guidance.StructuredRewardGradient, count)
        torch.cuda.synchronize()
        row['event_instrumented_wall_seconds'] = time.perf_counter() - started
        assert torch.equal(reference, event_output)
        row['phases'] = {}
        for name, entries in recorder.records.items():
            row['phases'][name] = {'calls': len(entries),
                'host_seconds': sum(value[0] for value in entries),
                'cuda_event_seconds': sum(value[1].elapsed_time(value[2]) for value in entries) / 1000}
        recorder.records.clear()
        save()
        print(json.dumps({'cases': count, 'steady_median_seconds': row['steady_median_seconds'],
                          'event_instrumented_wall_seconds': row['event_instrumented_wall_seconds'],
                          'phases': row['phases']}), flush=True)

        recorder.mode = 'off'
        report['rows'].append(row)
        save()

    for row in report['rows']:
        count = row['cases']
        # Capture a short window of the actual full Heun trajectory, including
        # its conditioning cache; stage-end profiler.step() is instrumentation.
        recorder.mode = 'profile'
        stages = min(args.profile_stages, 509)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA],
                                    schedule=torch.profiler.schedule(wait=1, warmup=1, active=stages, repeat=1),
                                    record_shapes=False, profile_memory=False, with_stack=False) as profiler:
            recorder.profiler = profiler
            profile_output, _ = sample(timed_model, timed_reward,
                                       timed_guidance.StructuredRewardGradient, count)
            torch.cuda.synchronize()
        recorder.profiler = None
        assert torch.equal(references[count], profile_output)
        events = profiler.events()
        gpu_events = [event for event in events
                      if str(event.device_type).endswith('CUDA')
                      and not getattr(event, 'is_user_annotation', False)
                      and not event.name.startswith(('ns::', 'ProfilerStep'))]
        kernels = defaultdict(lambda: {'calls': 0, 'device_seconds': 0.0})
        for event in gpu_events:
            kernels[event.name]['calls'] += 1
            kernels[event.name]['device_seconds'] += event.time_range.elapsed_us() / 1e6
        aggregates = profiler.key_averages()
        phases = {}
        for event in aggregates:
            if event.key.startswith('ns::') and str(event.device_type).endswith('CPU'):
                phases[event.key[4:]] = {'calls': event.count,
                    'cpu_total_seconds': event.cpu_time_total / 1e6,
                    'device_total_seconds': event.device_time_total / 1e6}
        row['profiler'] = {'stage_count': stages, 'phase_totals': phases,
            'device_activity_seconds': sum(event.time_range.elapsed_us() for event in gpu_events) / 1e6,
            'device_activity_count': len(gpu_events),
            'device_activity_filter': 'CUDA activities excluding explicit user annotations and ProfilerStep ranges; includes kernels and memory transfers',
            'top_device_activities': sorted([{'name': name, **data} for name, data in kernels.items()],
                                          key=lambda value: value['device_seconds'], reverse=True)[:20],
            'top_self_cpu_ops': [{'name': event.key, 'calls': event.count,
                                  'self_cpu_seconds': event.self_cpu_time_total / 1e6}
                                 for event in sorted(aggregates, key=lambda value: value.self_cpu_time_total, reverse=True)[:20]],
            'window_scope': 'Consecutive stages after one wait and one warm stage of actual full256Heun trajectory; conditioning cache remains enabled. Full uninstrumented steady timing and CUDA-event timing are reported separately.'}
        recorder.mode = 'off'
        save()
    report['status'] = 'passed'
    save()
    print(json.dumps({'status': 'passed', 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
