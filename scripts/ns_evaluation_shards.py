#!/usr/bin/env python3
"""Prepare, run and merge short NS terminal evaluations; prepare never submits.

A shard is an ordinary ns_common_evaluation_v1 manifest, evaluated by the
existing HiddenVerifier. Each lane owns disjoint shards and its own continuation
record. The original run and its Agent/Promotion state are never modified.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shlex
import sys
from threading import Event

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'legacy/ns/src'), str(ROOT / 'scripts')]

from pde_agent_verifier_solver.engine.checkpoint_store import (
    RunSegmentBudget, atomic_json, run_writer_lock, source_identity,
)
from pde_agent_verifier_solver.engine.common_evaluation import _CommonCheckpoint
from pde_agent_verifier_solver.plugins.ns.common_evaluation import run_ns_common_evaluation
from pde_agent_verifier_solver.sdk.evaluation_devices import EvaluationDevicePool, bind_evaluation_device_pool

SCHEMA = 'ns_evaluation_shards_v1'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def absolute(value, base):
    path = Path(value).expanduser()
    return str((base / path).resolve())


class PrepareOnly:
    """Materialize identities and references without invoking any solver."""
    def __init__(self):
        self.cancel = Event()

    def should_pause(self, stage, required_seconds=0):
        return True


class DrainBudget:
    """Stop dispatching early; only the real allocation deadline cancels work."""
    def __init__(self, budget, case_seconds):
        self.budget = budget
        self.cancel = budget.cancel
        self.reserve = case_seconds + 60  # Worker startup/cleanup; scoring unchanged.

    def should_pause(self, stage, required_seconds=0):
        return self.budget.should_pause(stage, max(required_seconds, self.reserve))


def checkpoint(root):
    """Read and hash-check a closed, complete-per-Case common checkpoint."""
    root = Path(root)
    saved = read(root / 'checkpoint.json')
    store = _CommonCheckpoint(root, saved['identity'], resume=True)
    plans = read(root / 'plans.json')['cases']
    expected = len(plans) * len(store.identity['programs'])
    if expected != store.identity['n_program_cases']:
        raise ValueError('checkpoint program/Case coverage differs')
    from pde_agent_verifier_solver.engine.common_evaluation import content_sha256
    if content_sha256(plans) != store.identity['plans_sha256']:
        raise ValueError('checkpoint plan hash differs')
    for index, rows in store.rows.items():
        validate_rows(rows, store.identity, plans, index)
    return store, plans, saved


def validate_rows(rows, identity, plans, index):
    program = identity['programs'][index // len(plans)]
    plan = plans[index % len(plans)]
    expected = {key: program[key] for key in
                ('method', 'run_id', 'seed', 'artifact_id', 'artifact_sha256')}
    expected.update({key: plan[key] for key in
                     ('cohort', 'dataset_id', 'case_id', 'plan_sha256',
                      'reference_sha256', 'public_case_sha256')})
    expected['task_id'] = identity['task_id']
    if [row.get('metric') for row in rows] != identity['metrics']:
        raise ValueError('cached Case has incomplete metrics')
    for row in rows:
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError('cached program, Case, plan or reference differs')
        if row.get('status') == 'ok':
            if row.get('error_code') is not None or not isinstance(row.get('error'), (int, float)) \
                    or not math.isfinite(row['error']):
                raise ValueError('cached successful Case has an invalid error')
        elif row.get('status') != 'failed' or row.get('error') is not None or not row.get('error_code'):
            raise ValueError('cached failure is not a complete solver outcome')


def prepare(manifest, output, *, reuse=None, shard_size=96, lanes=3, workers=4, max_segments=12,
            resource_profile='evaluation_3h_compact'):
    """Freeze exact inputs, import checked rows and partition only missing work."""
    for value, name in ((shard_size, 'shard_size'), (lanes, 'lanes'),
                        (workers, 'workers'), (max_segments, 'max_segments')):
        positive(value, name)
    from ns_slurm_continuation import _submission_spec, evaluation_worker_limit
    resources = _submission_spec(ROOT / 'scripts/slurm_ns_evaluate_short.sh', [], 1, resource_profile)
    limit = evaluation_worker_limit(resource_profile, resources['cpus'])
    if lanes > 3 or workers > limit:
        raise ValueError(f'short evaluation supports at most 3 lanes and {limit} workers for {resource_profile}')
    manifest = Path(manifest).resolve()
    spec = read(manifest)
    if spec.get('schema') != 'ns_common_evaluation_v1':
        raise ValueError('prepare requires an NS common terminal evaluation manifest')
    for dataset in spec['datasets']:
        dataset['manifest'] = absolute(dataset['manifest'], manifest.parent)
        if dataset.get('device', 'cpu') != 'cpu':
            raise ValueError('load frozen banks on CPU; the worker GPU serves public Oracles')
    for program in spec['programs']:
        program['bundle'] = absolute(program['bundle'], manifest.parent)
    if spec.get('query', {}).get('plans_file'):
        spec['query']['plans_file'] = absolute(spec['query']['plans_file'], manifest.parent)
    # A new evaluation output has its own execution concurrency. We never edit
    # max_workers or identity inside an existing checkpoint.
    spec['budget']['max_workers'] = workers
    old = old_plans = None
    reuse_info = None
    if reuse is not None:
        reuse = Path(reuse).resolve()
        original_bytes = (reuse / 'checkpoint.json').read_bytes()
        old, old_plans, saved = checkpoint(reuse)
        reuse_info = {'output': str(reuse), 'checkpoint_sha256': sha256(original_bytes).hexdigest(),
                      'n_reused_program_cases': len(old.rows),
                      'original_segment_wall_seconds': saved['elapsed_wall_seconds']}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    atomic_json(output / 'manifest.json', spec)
    aggregate = output / 'aggregate'
    run_ns_common_evaluation(output / 'manifest.json', aggregate, segment_budget=PrepareOnly())
    target, plans, _ = checkpoint(aggregate)
    resolved = read(aggregate / 'resolved_manifest.json')
    if old is not None:
        # Same evaluator code, task settings, frozen programs and full ordered
        # plan/reference list. Only execution worker count is allowed to differ.
        for key, value in target.identity.items():
            if key != 'max_workers' and old.identity.get(key) != value:
                raise ValueError(f'reuse scientific identity differs: {key}')
        old_resolved = read(reuse / 'resolved_manifest.json')
        if old_plans != plans or old_resolved.get('evaluation_source_sha256') != resolved['evaluation_source_sha256']:
            raise ValueError('reuse plans or Task implementation differ')
        if old_resolved.get('datasets') != resolved['datasets']:
            raise ValueError('reuse dataset identity or provenance differs')
        for index, rows in old.rows.items():
            validate_rows(rows, target.identity, plans, index)
            target.add(index, rows)
        target.save('paused', 0.0)
        if (reuse / 'checkpoint.json').read_bytes() != original_bytes:
            raise ValueError('original evaluation changed during preparation')
    # New shards use the aggregate's frozen bundle, not mutable author files.
    by_cohort = {row['cohort']: row for row in resolved['datasets']}
    shards = []
    ncases = len(plans)
    for program_index, program in enumerate(resolved['programs']):
        case_index = 0
        while case_index < ncases:
            index = program_index * ncases + case_index
            if index in target.rows:
                case_index += 1
                continue
            start = case_index
            cohort = plans[start]['cohort']
            selected = []
            while case_index < ncases and len(selected) < shard_size:
                index = program_index * ncases + case_index
                if index in target.rows or plans[case_index]['cohort'] != cohort:
                    break
                selected.append(index)
                case_index += 1
            shard_id = f'{len(shards):04d}'
            directory = output / 'shards' / shard_id
            directory.mkdir(parents=True)
            selected_plans = plans[start:case_index]
            atomic_json(directory / 'plans.json', {'schema': 'common_evaluation_plans_v1', 'cases': selected_plans})
            shard = deepcopy(spec)
            dataset = {key: value for key, value in by_cohort[cohort].items()
                       if key in ('dataset_id', 'manifest', 'cohort', 'split', 'device')}
            dataset['case_ids'] = [row['case_id'] for row in selected_plans]
            shard['datasets'] = [dataset]
            entry = deepcopy(spec['programs'][program_index])
            if program['frozen_bundle'] is not None:
                entry.update(bundle=program['frozen_bundle'], expected_tree_sha256=program['artifact_sha256'])
            shard['programs'] = [entry]
            shard['require_initial'] = False  # Full manifest already checked S0 if required.
            shard['query']['plans_file'] = str(directory / 'plans.json')
            # Numerical plans use query_seed + global Case index. Preserve the
            # original offset when taking a contiguous subset; no resampling.
            if shard['query'].get('seed') is not None:
                shard['query']['seed'] += start
            atomic_json(directory / 'manifest.json', shard)
            shards.append({'id': shard_id, 'lane': len(shards) % lanes,
                           'indices': selected, 'manifest_sha256': digest(directory / 'manifest.json'),
                           'plans_sha256': digest(directory / 'plans.json')})
    active_lanes = min(lanes, len(shards))
    for lane in range(active_lanes):
        (output / 'lanes' / str(lane) / 'slurm_logs').mkdir(parents=True)
    plan = {'schema': SCHEMA, 'status': 'prepared', 'source_root': str(ROOT),
            'source_sha256': source_identity(ROOT), 'original_manifest': str(manifest),
            'original_manifest_sha256': digest(manifest), 'manifest_sha256': digest(output / 'manifest.json'),
            'identity': target.identity, 'n_program_cases': target.identity['n_program_cases'],
            'n_reused_program_cases': len(target.rows), 'reuse': reuse_info,
            'lanes': active_lanes, 'workers': workers, 'shard_size': shard_size,
            'max_segments': max_segments, 'shards': shards,
            'resources': {'profile': resource_profile, 'gpus': 1, 'flush_seconds': 600,
                          **{key: resources[key] for key in ('cpus', 'memory', 'partition', 'wall_seconds')}}}
    atomic_json(output / 'plan.json', plan)
    atomic_json(output / 'status.json', {'status': 'prepared', 'n_program_cases': plan['n_program_cases'],
                                        'n_reused_program_cases': len(target.rows), 'n_shards': len(shards),
                                        'submitted': False})
    return plan


def load_plan(path):
    path = Path(path).resolve()
    plan = read(path)
    if plan.get('schema') != SCHEMA or plan['source_root'] != str(ROOT):
        raise ValueError('evaluation plan source differs; use its frozen source')
    if plan['reuse'] and digest(Path(plan['reuse']['output']) / 'checkpoint.json') != plan['reuse']['checkpoint_sha256']:
        raise ValueError('original evaluation has advanced; prepare again before running a duplicate')
    assigned = [i for shard in plan['shards'] for i in shard['indices']]
    if len(assigned) != len(set(assigned)) or len(assigned) + plan['n_reused_program_cases'] != plan['n_program_cases']:
        raise ValueError('evaluation shards overlap or have incomplete coverage')
    if any(not isinstance(i, int) or not 0 <= i < plan['n_program_cases'] for i in assigned):
        raise ValueError('evaluation shard index outside the full plan')
    return path.parent, plan


def shard_paths(root, shard):
    if shard['id'] != f'{int(shard["id"]):04d}':
        raise ValueError('invalid shard id')
    directory = root / 'shards' / shard['id']
    return directory / 'manifest.json', directory / 'result'


def run_lane(plan_path, lane, *, budget, evaluation_devices=()):
    """Run a disjoint lane; caller owns Slurm submission and allocation cleanup."""
    root, plan = load_plan(plan_path)
    if isinstance(lane, bool) or not isinstance(lane, int) or not 0 <= lane < plan['lanes']:
        raise ValueError('lane is outside this evaluation plan')
    lane_root = root / 'lanes' / str(lane)
    drain = DrainBudget(budget, plan['identity']['case_hard_wall_seconds'])
    completed = 0
    pending = False
    with run_writer_lock(lane_root):
        previous_path = lane_root / 'checkpoint.json'
        if previous_path.exists() and read(previous_path).get('plan_sha256') != digest(plan_path):
            raise ValueError('lane resume plan changed')
        with EvaluationDevicePool(evaluation_devices) as pool, bind_evaluation_device_pool(pool):
            for shard in plan['shards']:
                if shard['lane'] != lane:
                    continue
                manifest, output = shard_paths(root, shard)
                saved = read(output / 'checkpoint.json') if (output / 'checkpoint.json').is_file() else None
                if saved and saved['status'] == 'completed':
                    stored, _, _ = checkpoint(output)
                    if len(stored.rows) != len(shard['indices']):
                        raise ValueError('completed shard has missing results')
                    completed += len(stored.rows)
                    continue
                if drain.should_pause('between_shards'):
                    completed += len(saved['completed']) if saved else 0
                    pending = True
                    break
                result = run_ns_common_evaluation(manifest, output, resume=saved is not None,
                                                  segment_budget=drain)
                saved = read(output / 'checkpoint.json')
                completed += len(saved['completed'])
                if result['status'] == 'paused':
                    pending = True
                    break
        # Device/session cleanup must finish before publishing a resumable lane.
        result = {'status': 'paused' if pending else 'completed', 'lane': lane,
                  'plan_sha256': digest(plan_path), 'n_completed_program_cases': completed,
                  'n_program_cases': sum(len(s['indices']) for s in plan['shards'] if s['lane'] == lane)}
        atomic_json(lane_root / 'checkpoint.json', result)
        atomic_json(lane_root / 'status.json', result)
        return result


def merge(plan_path):
    """Publish the existing common-evaluation tables only after exact coverage."""
    root, plan = load_plan(plan_path)
    with run_writer_lock(root):
        target, plans, _ = checkpoint(root / 'aggregate')
        if target.identity != plan['identity']:
            raise ValueError('aggregate checkpoint identity changed')
        incoming = {}
        shard_wall = 0.0
        for shard in plan['shards']:
            _, output = shard_paths(root, shard)
            store, _, saved = checkpoint(output)
            if saved['status'] != 'completed' or len(store.rows) != len(shard['indices']):
                raise ValueError('cannot publish comparison before every shard is completed')
            for key in ('task_id', 'metrics', 'settings_sha256', 'evaluation_program_id',
                        'evaluator_source_sha256', 'case_hard_wall_seconds', 'max_workers'):
                if store.identity[key] != target.identity[key]:
                    raise ValueError(f'shard evaluation contract changed: {key}')
            for local_index, rows in store.rows.items():
                index = shard['indices'][local_index]
                validate_rows(rows, target.identity, plans, index)
                if index in incoming or (index in target.rows and target.rows[index] != rows):
                    raise ValueError('duplicate or conflicting program/Case results')
                incoming[index] = rows
            shard_wall += saved.get('pipeline_wall_seconds', saved['elapsed_wall_seconds'])
        if set(target.rows) | set(incoming) != set(range(plan['n_program_cases'])):
            raise ValueError('incomplete program/Case coverage')
        for index, rows in incoming.items():
            if index not in target.rows:
                target.add(index, rows)
        target.save('paused', 0.0)
        # All rows are committed: the normal evaluator verifies inputs and
        # computes its standard summaries without executing any solver again.
        summary = run_ns_common_evaluation(root / 'manifest.json', root / 'aggregate', resume=True)
        summary['dispatch'] = {
            'plan': str(Path(plan_path).resolve()), 'n_shards': len(plan['shards']),
            'n_reused_program_cases': plan['n_reused_program_cases'],
            'new_shard_segment_wall_seconds': shard_wall,
            'reuse': plan['reuse'],
            'timing_note': 'Shard time sums allocation execution, not parallel makespan or queue wait. '
                           'Common total/pipeline wall fields below measure prepare/merge only; '
                           'reused work is recorded separately and is not charged to new shards.',
        }
        summary['timing_definition']['total_wall_seconds'] = 'Prepare/merge verifier wall time only; see dispatch for shard execution cost.'
        summary['timing_definition']['pipeline_wall_seconds'] = 'Prepare/merge pipeline wall time only; see dispatch for shard execution cost.'
        atomic_json(root / 'aggregate' / 'summary.json', summary)
        atomic_json(root / 'status.json', {'status': 'completed', 'n_program_cases': plan['n_program_cases'],
                                          'n_failed_program_cases': summary['n_failed_program_cases']})
        return summary


def submit_commands(plan_path, python):
    """Return reviewable argv lists; never invoke sbatch here."""
    from ns_slurm_continuation import _submission_spec, submission_resource_flags
    root, plan = load_plan(plan_path)
    resources = _submission_spec(ROOT / 'scripts/slurm_ns_evaluate_short.sh', [], 1,
                                 plan['resources']['profile'])
    # Preserve a virtualenv entry point: resolving its symlink loses site-packages.
    commands = []
    for lane in range(plan['lanes']):
        lane_root = root / 'lanes' / str(lane)
        initial_env = ('--export=ALL,' + f'NS_CONTINUATION_RUN_ROOT={lane_root},'
                       'NS_CONTINUATION_SEGMENT_INDEX=1,NS_CONTINUATION_INTENT_ID=,'
                       f'NS_CONTINUATION_MAX_SEGMENTS={plan["max_segments"]}')
        commands.append(['sbatch', '--parsable', initial_env, *submission_resource_flags(resources),
                         f'--output={lane_root / "slurm_logs/segment-%j.out"}',
                         f'--error={lane_root / "slurm_logs/segment-%j.err"}',
                         str(ROOT / 'scripts/slurm_ns_evaluate_short.sh'), str(Path(plan_path).resolve()),
                         str(lane), os.path.abspath(os.path.expanduser(os.fspath(python)))])
    return commands


def run_slurm(plan_path, lane, *, auto_continue=False):
    from ns_slurm_continuation import runner_lease, publish_outcome, submit_next, SubmissionUncertainError
    from run_ns_draft_job import allocation_remaining_seconds
    root, plan = load_plan(plan_path)
    if isinstance(lane, bool) or not isinstance(lane, int) or not 0 <= lane < plan['lanes']:
        raise ValueError('lane is outside this evaluation plan')
    job = os.environ['SLURM_JOB_ID']
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(devices) != 1 or not devices[0]:
        raise ValueError('short evaluation requires exactly one allocated GPU')
    if int(os.environ['SLURM_CPUS_PER_TASK']) < plan['resources']['cpus']:
        raise ValueError(f"short evaluation requires {plan['resources']['cpus']} allocated CPUs")
    import torch
    if torch.cuda.device_count() != 1:
        raise ValueError('expected one visible CUDA device')
    torch.set_num_threads(1)
    lane_root = root / 'lanes' / str(lane)
    continuation_root = os.environ.get('NS_CONTINUATION_RUN_ROOT')
    if continuation_root and Path(continuation_root).resolve() != lane_root:
        raise ValueError('continuation lane differs')
    index = int(os.environ.get('NS_CONTINUATION_SEGMENT_INDEX', '1'))
    intent = os.environ.get('NS_CONTINUATION_INTENT_ID')
    with runner_lease(lane_root, job, index, max_segments=plan['max_segments'], intent_id=intent):
        previous = read(lane_root / 'checkpoint.json') if (lane_root / 'checkpoint.json').exists() else {}
        budget = RunSegmentBudget(min(plan['resources']['wall_seconds'], allocation_remaining_seconds(job)),
                                  flush_seconds=plan['resources']['flush_seconds'])
        try:
            with budget.signals():
                outcome = run_lane(plan_path, lane, budget=budget, evaluation_devices=('cuda:0',))
            if outcome['status'] == 'paused' and outcome['n_completed_program_cases'] <= previous.get('n_completed_program_cases', 0):
                raise RuntimeError('no completed Case this segment; automatic continuation stopped')
            publish_outcome(lane_root, status=outcome['status'], checkpoint_path=lane_root / 'checkpoint.json')
        finally:
            budget.close()
    if auto_continue and outcome['status'] == 'paused':
        try:
            state = submit_next(lane_root, slurm_script=ROOT / 'scripts/slurm_ns_evaluate_short.sh',
                                script_args=[str(Path(plan_path).resolve()), str(lane), sys.executable],
                                gpu_count=1, resource_profile=plan['resources']['profile'],
                                max_segments=plan['max_segments'])
            outcome['continuation_status'] = state['status']
        except SubmissionUncertainError as exc:
            # An accepted afterok successor must not be blocked by an ambiguous
            # receipt. The existing durable intent prevents a duplicate retry.
            outcome.update(continuation_status='submission_unknown', continuation_error=str(exc))
        except Exception as exc:
            outcome.update(continuation_status='stopped', continuation_error=str(exc))
            atomic_json(lane_root / 'status.json', outcome)
            raise
        atomic_json(lane_root / 'status.json', outcome)
    return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='CPU-only freeze/verify/partition; no submission or solver calls')
    prep.add_argument('--manifest', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--reuse', type=Path)
    prep.add_argument('--shard-size', type=int, default=96)
    prep.add_argument('--lanes', type=int, default=3)
    prep.add_argument('--workers', type=int, default=4)
    prep.add_argument('--resource-profile', default='evaluation_3h_compact',
                      choices=('evaluation_3h_compact', 'evaluation_3h', 'evaluation_3h_parallel'))
    prep.add_argument('--max-segments', type=int, default=12)
    command = sub.add_parser('commands', help='Print initial sbatch commands; do not execute them')
    command.add_argument('--plan', type=Path, required=True)
    command.add_argument('--python', type=Path, default=sys.executable)
    run = sub.add_parser('run', help='Execute one lane inside an existing single-GPU allocation')
    run.add_argument('--plan', type=Path, required=True)
    run.add_argument('--lane', type=int, required=True)
    run.add_argument('--auto-continue', action='store_true')
    combine = sub.add_parser('merge', help='CPU-only complete coverage verification and standard tables')
    combine.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        result = prepare(args.manifest, args.output, reuse=args.reuse, shard_size=args.shard_size,
                         lanes=args.lanes, workers=args.workers, max_segments=args.max_segments,
                         resource_profile=args.resource_profile)
        print(json.dumps({key: result[key] for key in ('status', 'n_program_cases', 'n_reused_program_cases', 'lanes')}))
    elif args.command == 'commands':
        for command in submit_commands(args.plan, args.python):
            print(shlex.join(command))
    elif args.command == 'run':
        print(json.dumps(run_slurm(args.plan, args.lane, auto_continue=args.auto_continue)))
    else:
        result = merge(args.plan)
        print(json.dumps({'status': result['status'], 'n_rows': result['n_rows'],
                          'n_failed_program_cases': result['n_failed_program_cases']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
