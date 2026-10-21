#!/usr/bin/env python3
"""Resume only the evaluation tail of an existing Campaign on one GPU.

Uses the original Controller, saved Cases, Promotion and final-test manifest.
No Agent execution is possible. Resource-only checkpoint transfers are explicit,
leave their input untouched, retain Controller RNG stream 0 and all Case caches.
"""
from __future__ import annotations

import argparse
import copyreg
from contextlib import contextmanager
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time
import traceback
from types import MappingProxyType

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'ns_campaign_evaluation_only_v1'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


class InactiveAgent:
    gpu_devices = ()

    def run(self, *_args, **_kwargs):
        raise RuntimeError('evaluation-only continuation cannot run an Agent')


class EvaluationOnlyAgents:
    audits = ()
    quarantined_gpu_devices = ()
    evaluation_device_map = None

    def execute(self, *_args, **_kwargs):
        raise RuntimeError('evaluation-only continuation cannot run an Agent')

    execute_batch = execute

    @contextmanager
    def reserve_evaluation_devices(self, programs):
        from pde_agent_verifier_solver.sdk.evaluation_devices import EvaluationDevicePool
        if any(not isinstance(program, InactiveAgent) for program in programs):
            raise ValueError('evaluation-only service received an active Agent program')
        with EvaluationDevicePool(('cuda:0',)) as pool:
            yield pool

    def close(self):
        pass


def require_evaluation_tail(value, round_count, agent_ids):
    state = value.state
    rounds = state.get('round_results', ())
    if [r.round_index for r in rounds] != list(range(1, len(rounds) + 1)):
        raise ValueError('nonconsecutive completed rounds')
    if len(rounds) == round_count and state.get('phase') in {'terminal_ready', 'round_prepare', 'terminal_prepare', 'terminal_done', 'complete'}:
        return
    data = state.get('round_data', {})
    if (len(rounds) == round_count - 1 and state.get('phase') in {'official_ready', 'official_done'}
            and len(data.get('outcomes', ())) == len(agent_ids)
            and len(data.get('agent_logs', ())) == len(agent_ids)
            and {log.agent_id for log in data['agent_logs']} == set(agent_ids)
            and all(outcome.execution_status == 'completed' for outcome in data['outcomes'])):
        return
    raise ValueError('Campaign still needs Agent work; not an evaluation-only continuation')


class _Reference:
    def __init__(self, token):
        self.token = token


class _ReferenceReader(pickle.Unpickler):
    def persistent_load(self, token):
        return _Reference(token)


class _ReferenceWriter(pickle.Pickler):
    def __init__(self, stream):
        from pde_agent_verifier_solver.engine.checkpoint_store import _Pickler
        super().__init__(stream, protocol=5)
        self.dispatch_table = _Pickler(io.BytesIO(), None).dispatch_table

    def persistent_id(self, value):
        return value.token if isinstance(value, _Reference) else None


def encode_references(value):
    stream = io.BytesIO()
    _ReferenceWriter(stream).dump(value)
    return stream.getvalue()


def decode_references(data):
    return _ReferenceReader(io.BytesIO(data)).load()


def scientific_payload_digest(value):
    """Compare checkpoint content, including every immutable Case reference.

    Pickle memo numbers/object aliasing can change when a Candidate codec
    revalidates and reconstructs its bundle; they are not scientific content.
    """
    from collections.abc import Mapping
    from dataclasses import fields, is_dataclass
    from enum import Enum
    from pde_agent_verifier_solver.engine.python_candidate import PythonCandidateProgram
    def normalize(item):
        if isinstance(item, _Reference):
            return ['reference', normalize(item.token)]
        if isinstance(item, PythonCandidateProgram):
            return ['candidate', str(item.bundle_locator), item.tree_sha256]
        if isinstance(item, Enum):
            return ['enum', type(item).__module__, type(item).__name__, normalize(item.value)]
        if is_dataclass(item):
            return ['dataclass', type(item).__module__, type(item).__name__,
                    [[field.name, normalize(getattr(item, field.name))] for field in fields(item)]]
        if isinstance(item, Mapping):
            pairs = [[normalize(k), normalize(v)] for k, v in item.items()]
            return ['mapping', sorted(pairs, key=lambda pair: json.dumps(pair[0], sort_keys=True))]
        if isinstance(item, (tuple, list, set, frozenset)):
            contents = [normalize(v) for v in item]
            if isinstance(item, (set, frozenset)):
                contents.sort(key=lambda v: json.dumps(v, sort_keys=True))
            return [type(item).__name__, contents]
        if isinstance(item, Path):
            return ['path', str(item)]
        if isinstance(item, float):
            return ['float', item.hex()]
        if isinstance(item, (str, int, bool)) or item is None:
            return item
        if isinstance(item, bytes):
            return ['bytes', item.hex()]
        if type(item).__module__.startswith('numpy') and hasattr(item, 'dtype'):
            return ['array', str(item.dtype), list(item.shape), item.tobytes().hex()]
        if type(item).__module__ == 'types' and type(item).__name__ == 'SimpleNamespace':
            return ['namespace', normalize(vars(item))]
        raise TypeError(f'unhandled scientific checkpoint value: {type(item)}')
    return sha256(json.dumps(normalize(value), sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def transfer_checkpoint(original, target, *, old_source, config, controller_stream=0):
    """Offline resource migration; never instantiate a Generator or initialize CUDA."""
    from pde_agent_verifier_solver.engine.checkpoint_store import atomic_json, atomic_bytes, source_identity, scientific_progress
    original, target, old_source = Path(original).resolve(), Path(target).resolve(), Path(old_source).resolve()
    if target.exists() or target == original or target.is_relative_to(original):
        raise ValueError('transfer needs a new destination outside the original checkpoint')
    manifest = read(original / 'checkpoint.json')
    before = digest(original / 'checkpoint.json')
    identity = manifest['identity']
    config_value = {**read(config), "_config_path": str(Path(config).resolve())}
    if identity['source_sha256'] != source_identity(old_source):
        raise ValueError('original source differs from the saved Campaign')
    if identity['config_sha256'] != sha256(json.dumps(config_value, sort_keys=True).encode()).hexdigest():
        raise ValueError('original config differs from the saved Campaign')
    reference = manifest['payload']
    if Path(reference['name']).name != reference['name']:
        raise ValueError('invalid checkpoint payload path')
    payload_path = original / 'objects' / reference['name']
    payload = decode_references(payload_path.read_bytes())
    require_evaluation_tail(payload['value'], config_value['round_count'],
                            [agent['agent_id'] for agent in config_value['agents']])
    value_before = scientific_payload_digest(payload['value'])
    if scientific_progress(payload['value']) != manifest['progress']:
        raise ValueError('saved scientific progress differs')
    cuda_states = payload['rng'].get('cuda')
    if not cuda_states or controller_stream != 0 or len(cuda_states) not in (1, 3):
        raise ValueError('expected the canonical Controller RNG at cuda:0 (one or three original devices)')
    old_rng = sha256(encode_references(payload['rng'])).hexdigest()
    payload['rng']['cuda'] = [cuda_states[controller_stream]]
    encoded = encode_references(payload)
    restored = decode_references(encoded)
    if scientific_payload_digest(restored['value']) != value_before:
        raise ValueError('resource migration changed scientific payload')
    # Content-addressed sidecars are immutable and may be shared using hard links.
    # The mutable checkpoint manifest is always an independent atomic file.
    target.mkdir(parents=True, mode=0o700)
    objects = target / 'objects'
    objects.mkdir(mode=0o700)
    for path in (original / 'objects').iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError('checkpoint sidecars must be regular files')
        os.link(path, objects / path.name)
    new_payload_hash = sha256(encoded).hexdigest()
    atomic_bytes(objects / (new_payload_hash + '.pkl'), encoded)
    updated = {**manifest, 'identity': {**identity, 'source_sha256': source_identity(ROOT)},
               'payload': {'name': new_payload_hash + '.pkl', 'sha256': new_payload_hash}}
    atomic_json(target / 'checkpoint.json', updated)
    if digest(original / 'checkpoint.json') != before:
        raise ValueError('original checkpoint advanced during preparation')
    receipt = {'original': str(original), 'original_checkpoint_sha256': before,
               'original_source': str(old_source), 'original_identity': identity,
               'new_identity': updated['identity'], 'scientific_progress': manifest['progress'],
               'scientific_payload_preserved': True, 'scientific_payload_sha256': value_before, 'original_rng_sha256': old_rng,
               'cuda_streams_before': len(cuda_states), 'cuda_streams_after': 1,
               'retained_controller_stream': controller_stream,
               'controller_rng_sha256': sha256(cuda_states[0].tobytes()).hexdigest(),
               'retired_device_rngs_retained_in_original_checkpoint': True,
               'completed_rounds': len(payload['value'].state['round_results']),
               'phase': payload['value'].state['phase']}
    atomic_json(target.parent / 'resource_transfer.json', receipt)
    return receipt


class DrainBudget:
    def __init__(self, budget, case_seconds, *, guidance_training_seconds=0, guidance_sampling_seconds=0):
        self.budget, self.cancel = budget, budget.cancel
        self.reserve = case_seconds + 120
        self.stages = {'before_terminal_guidance': guidance_training_seconds + 120,
                       'terminal_guidance_ready': guidance_sampling_seconds + 120}

    def should_pause(self, stage, required_seconds=0):
        return self.budget.should_pause(stage, max(required_seconds, self.reserve, self.stages.get(stage, 0)))


def campaign(plan, output, budget):
    from pde_agent_verifier_solver.engine import CampaignController, EventStream
    from pde_agent_verifier_solver.engine.checkpoint_store import CheckpointStore, run_writer_lock
    from pde_agent_verifier_solver.engine.campaign_continuation import CampaignPauseControl
    from pde_agent_verifier_solver.reporting.costs import MeasuredGenerator
    from pde_agent_verifier_solver.engine.audit_journal import AuditJournal
    import run_canonical_ns_campaign as canonical
    config_path = Path(plan['config'])
    config, config_path = canonical._load_config(config_path)
    canonical._validate_config(config)
    composition = None
    scientific = output / 'scientific'
    writer = canonical._RunWriter(scientific)
    segment = Path('segments') / f'eval-{os.environ.get("SLURM_JOB_ID", "test")}'
    journal = None
    with run_writer_lock(scientific):
        try:
            writer.write_json('status.json', {'status': 'restoring_evaluation', 'updated_at_unix': time.time()})
            composition = canonical._compose(config=config, config_path=config_path,
                    run_root=scientific, codex_binary=Path('/unused-in-evaluation-only'),
                    auth_file=Path('/unused-in-evaluation-only'), worker_gpus=(), controller_gpu='0',
                    evaluation_workers=plan['workers'])
            audit = canonical._bind_hidden_evaluation_audit(config, config_path, composition, scientific / segment)
            if audit is not None:
                composition.generator = audit
            composition.generator = MeasuredGenerator(composition.generator, scientific / 'generator_operations.jsonl', resume=True)
            store = CheckpointStore(scientific / 'checkpoints', identity=plan['checkpoint_identity'], generator=composition.generator)
            # Check runtime inputs, not only the offline migration manifest.
            actual_identity = {'config_sha256': sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                               'source_sha256': plan['source_sha256'],
                               'generator_inputs': composition.provenance.get('generator_artifacts', {})}
            if actual_identity != plan['checkpoint_identity']:
                raise ValueError('evaluation composition differs from the original scientific inputs')
            continuation = store.load()
            require_evaluation_tail(continuation, config['round_count'], [a['agent_id'] for a in config['agents']])
            writer.restore_events(continuation.controller_state['events'], segment)
            writer.write_json(str(segment / 'input_provenance.json'), composition.provenance)
            journal = AuditJournal(scientific / 'candidate_execution_audits.jsonl', encode=canonical._jsonable)
            def flush():
                composition.generator.flush_operations()
                journal.append(composition.candidate_execution.audits)
            def save(value):
                flush()
                store.save(value)
            events = EventStream()
            events.subscribe(writer.record_event)
            controller = CampaignController(campaign=composition.campaign, dataset=composition.dataset,
                problem=composition.problem, task=composition.task, generator=composition.generator,
                agents=composition.agents, events=events, agent_execution=composition.execution,
                public_problem_bundle=composition.public_problem_bundle, candidate_execution=composition.candidate_execution)
            composition.candidate_execution.allocation_cancellation_event = budget.cancel
            pause = CampaignPauseControl(save, budget.should_pause, hidden_batch_size=plan['workers'])
            writer.write_json('status.json', {'status': 'running', 'evaluation_only': True, 'updated_at_unix': time.time()})
            result = controller.run(continuation=continuation, pause_control=pause)
            if audit is not None:
                audit.assert_complete()
            writer.write_json('result.json', canonical._jsonable(result))
            writer.write_json('candidate_history.json', canonical._jsonable(result.candidate_history))
            writer.write_text('CAMPAIGN_MEMORY.md', result.memory.markdown + '\n')
            if not (scientific / 'selected_candidate').exists():
                canonical._publish_selected_candidate(result.selected.artifact, scientific / 'selected_candidate')
            writer.write_json('status.json', {'status': 'completed', 'rounds_completed': len(result.rounds),
                'selected_artifact_id': result.selected.artifact.artifact_id, 'best_terminal_round': result.best_terminal_round,
                'evaluation_only': True, 'finished_at_unix': time.time()})
        finally:
            if composition is not None:
                for service in (composition.candidate_execution, composition.execution):
                    close = getattr(service, 'close', None)
                    if callable(close):
                        close()
                composition.generator.flush_operations()
                if journal is not None:
                    journal.append(composition.candidate_execution.audits)
                    writer.write_json('candidate_execution_audits.json', journal.rows)


def final_evaluation(plan, output, budget):
    from pde_agent_verifier_solver.engine.checkpoint_store import atomic_json
    from pde_agent_verifier_solver.plugins.ns.common_evaluation import run_ns_common_evaluation
    from pde_agent_verifier_solver.sdk.evaluation_devices import EvaluationDevicePool, bind_evaluation_device_pool
    spec = read(plan['final_manifest_template'])
    spec['programs'][0]['bundle'] = str(output / 'scientific/selected_candidate')
    spec['budget']['max_workers'] = plan['workers']
    path = output / 'common_evaluation_manifest.json'
    if path.exists() and read(path) != spec:
        raise ValueError('frozen common evaluation specification differs')
    atomic_json(path, spec)
    directory = output / 'common_evaluation'
    with EvaluationDevicePool(('cuda:0',)) as pool, bind_evaluation_device_pool(pool):
        return run_ns_common_evaluation(path, directory, resume=(directory / 'checkpoint.json').exists(), segment_budget=budget)


def validate_resources(plan):
    from ns_slurm_continuation import _submission_spec, evaluation_worker_limit
    resources = plan['resources']
    profile = resources['profile']
    if profile not in {'evaluation_3h_compact', 'evaluation_3h_parallel'}:
        raise ValueError('unexpected evaluation resource profile')
    expected = _submission_spec(ROOT / 'scripts/slurm_ns_campaign_evaluation_short.sh', [], 1, profile)
    for key in ('cpus', 'memory', 'wall_seconds'):
        if resources[key] != expected[key]:
            raise ValueError(f'evaluation resource differs from profile: {key}')
    if resources['gpus'] != 1 or resources['flush_seconds'] != 600:
        raise ValueError('evaluation GPU or flush budget differs')
    workers = plan['workers']
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= evaluation_worker_limit(profile, expected['cpus']):
        raise ValueError('evaluation workers exceed the allocated CPU budget')


def load_plan(path):
    from pde_agent_verifier_solver.engine.checkpoint_store import source_identity
    plan = read(path)
    if plan['schema'] != SCHEMA or Path(plan['source_root']).resolve() != ROOT:
        raise ValueError('use the prepared frozen evaluation source')
    validate_resources(plan)
    transfer = read(Path(plan['output']) / 'scientific/resource_transfer.json')
    if digest(Path(transfer['original']) / 'checkpoint.json') != transfer['original_checkpoint_sha256']:
        raise ValueError('original run advanced; refusing duplicate evaluation')
    return plan


def evaluation_progress(checkpoint, workflow_phase):
    """Also count completed final Guidance training between two allocations."""
    manifest = read(checkpoint)
    if manifest['schema'] != 'campaign_checkpoint_v1':
        from run_ns_draft_job import _checkpoint_progress
        return _checkpoint_progress(checkpoint, workflow_phase)
    reference = manifest['payload']
    if Path(reference['name']).name != reference['name']:
        raise ValueError('invalid checkpoint payload')
    payload_path = Path(checkpoint).parent / 'objects' / reference['name']
    state = decode_references(payload_path.read_bytes())['value'].state
    return sha256(json.dumps({'workflow_phase': workflow_phase, 'progress': manifest['progress'],
                              'final_guidance_ready': bool(state.get('final_guidance_ready'))}, sort_keys=True).encode()).hexdigest()


def run(path):
    from pde_agent_verifier_solver.engine.checkpoint_store import RunSegmentBudget, atomic_json
    from pde_agent_verifier_solver.engine.campaign_continuation import CampaignPaused
    from ns_slurm_continuation import runner_lease, publish_outcome, submit_next, SubmissionUncertainError
    from run_ns_draft_job import allocation_remaining_seconds
    plan = load_plan(path)
    output = Path(plan['output'])
    job = os.environ['SLURM_JOB_ID']
    if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) != 1 or int(os.environ['SLURM_CPUS_PER_TASK']) < plan['resources']['cpus']:
        raise ValueError(f"evaluation needs one GPU and {plan['resources']['cpus']} allocated CPUs")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('one usable CUDA GPU required')
    torch.set_num_threads(1)
    expected_root = os.environ.get('NS_CONTINUATION_RUN_ROOT')
    if expected_root and Path(expected_root).resolve() != output:
        raise ValueError('continuation run root differs')
    index = int(os.environ.get('NS_CONTINUATION_SEGMENT_INDEX', '1'))
    record = {'job_id': job, 'segment_index': index, 'evaluation_only': True, 'status': 'running', 'started_at_unix': time.time()}
    outcome = 'failed'
    with runner_lease(output, job, index, max_segments=plan['max_segments']):
        budget = RunSegmentBudget(min(10800, allocation_remaining_seconds(job)), flush_seconds=600)
        protocol = read(plan['config'])['generator'].get('guidance', {}).get('protocol', {})
        drain = DrainBudget(budget, plan['case_hard_wall_seconds'],
                            guidance_training_seconds=protocol.get('training_budget', {}).get('maximum_wall_seconds', 0),
                            guidance_sampling_seconds=protocol.get('inference_wall_budget_seconds', 0))
        workflow_path = output / 'workflow.json'
        workflow = read(workflow_path) if workflow_path.exists() else {'phase': 'scientific'}
        original_progress = None
        try:
            atomic_json(output / 'status.json', record)
            with budget.signals():
                checkpoint = output / ('scientific/checkpoints/checkpoint.json' if workflow['phase'] == 'scientific' else 'common_evaluation/checkpoint.json')
                if checkpoint.exists():
                    original_progress = evaluation_progress(checkpoint, workflow['phase'])
                if workflow['phase'] == 'scientific':
                    campaign(plan, output, drain)
                    workflow['phase'] = 'final_evaluation'
                    atomic_json(workflow_path, workflow)
                if workflow['phase'] == 'final_evaluation':
                    checkpoint = output / 'common_evaluation/checkpoint.json'
                    report = final_evaluation(plan, output, drain)
                    if report['status'] == 'paused':
                        raise CampaignPaused(stage='final_evaluation')
                    if report['status'] != 'completed':
                        raise RuntimeError('final evaluation did not complete')
                    workflow['phase'] = 'completed'
                outcome = 'completed'
                publish_outcome(output, status=outcome)
        except CampaignPaused as pause:
            checkpoint = output / ('scientific/checkpoints/checkpoint.json' if workflow['phase'] == 'scientific' else 'common_evaluation/checkpoint.json')
            progress = evaluation_progress(checkpoint, workflow['phase'])
            if progress == original_progress:
                raise RuntimeError('no scientific progress this allocation; continuation stopped') from pause
            workflow['last_pause_progress'] = progress
            outcome = 'paused'
            record['stage'] = pause.stage
            publish_outcome(output, status=outcome, checkpoint_path=checkpoint)
        except BaseException as error:
            record.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
            atomic_json(output / 'status.json', record)
            raise
        finally:
            budget.close()
        record.update(status=outcome, phase=workflow['phase'], finished_at_unix=time.time())
        atomic_json(workflow_path, workflow)
        atomic_json(output / 'status.json', record)
    if outcome == 'paused':
        try:
            state = submit_next(output, slurm_script=ROOT / 'scripts/slurm_ns_campaign_evaluation_short.sh',
                    script_args=[str(Path(path).resolve()), sys.executable], gpu_count=1,
                    resource_profile=plan['resources']['profile'], max_segments=plan['max_segments'])
            record['continuation_status'] = state['status']
        except SubmissionUncertainError as error:
            record.update(continuation_status='submission_unknown', continuation_error=str(error))
        except BaseException as error:
            record.update(continuation_status='stopped', continuation_error=str(error))
            atomic_json(output / 'status.json', record)
            raise
        atomic_json(output / 'status.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.plan), allow_nan=False), flush=True)
