"""Persistent continuation decisions without any real scheduler submission."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
from threading import Event, Thread
from types import SimpleNamespace

import pytest


@pytest.fixture
def continuation():
    spec = importlib.util.spec_from_file_location(
        'ns_slurm_continuation_test', Path(__file__).resolve().parents[1] / 'scripts/ns_slurm_continuation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def run(tmp_path):
    root = tmp_path / 'run'
    root.mkdir()
    (root / 'checkpoint.json').write_text('{"round": 2}\n')
    wrapper = tmp_path / 'frozen wrapper.sh'
    wrapper.write_text('#!/bin/bash\nexit 0\n')
    return root, wrapper


def pause(module, root, *, max_segments=12):
    with module.runner_lease(root, '100', 1, max_segments=max_segments):
        return module.publish_outcome(root, status='paused', checkpoint_path='checkpoint.json')


def submit(module, run, **kwargs):
    root, wrapper = run
    return module.submit_next(root, slurm_script=wrapper, script_args=['full', 'a path with spaces'],
                              gpu_count=kwargs.pop('gpu_count', 3), **kwargs)


def fake_sbatch(monkeypatch, module, root, *, stdout='200;cluster\n', returncode=0):
    calls = []

    def invoke(command, **options):
        state = json.loads((root / 'slurm_continuation.json').read_text())
        assert state['status'] == 'submitting'  # Intent committed before external side effect.
        assert state['submissions'][-1]['job_id'] is None
        calls.append((command, options))
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr='scheduler diagnostic')

    monkeypatch.setattr(module.subprocess, 'run', invoke)
    return calls


def test_lease_publishes_atomic_pause_with_hash(continuation, run):
    root, _ = run
    state = pause(continuation, root)
    assert state['status'] == 'paused'
    assert state['segment_index'] == 1
    assert state['checkpoint'] == {'path': str(root / 'checkpoint.json'),
                                   'sha256': sha256((root / 'checkpoint.json').read_bytes()).hexdigest()}
    assert continuation.read_state(root) == state
    assert not list(root.glob('.slurm-continuation-*.tmp'))
    assert not (root / 'status.json').exists()


def test_requires_lease_and_real_checkpoint(continuation, run):
    root, _ = run
    with pytest.raises(continuation.ContinuationError, match='lease'):
        continuation.publish_outcome(root, status='paused', checkpoint_path='checkpoint.json')
    with continuation.runner_lease(root, '100', 1):
        with pytest.raises(continuation.ContinuationError, match='requires a committed'):
            continuation.publish_outcome(root, status='paused')
        continuation.publish_outcome(root, status='paused', checkpoint_path='checkpoint.json',
                                     checkpoint_sha256='wrong')
        assert continuation.read_state(root)['status'] == 'paused'



@pytest.mark.parametrize('outcome', ['completed', 'failed'])
def test_terminal_outcomes_never_submit(continuation, run, monkeypatch, outcome):
    root, _ = run
    with continuation.runner_lease(root, '100', 1):
        continuation.publish_outcome(root, status=outcome)
    calls = fake_sbatch(monkeypatch, continuation, root)
    assert submit(continuation, run)['status'] == outcome
    assert calls == []
    with pytest.raises(continuation.ContinuationError, match='recorded next'):
        with continuation.runner_lease(root, '100', 1):
            pass


def test_missing_outcome_is_terminal_failure(continuation, run):
    root, _ = run
    with pytest.raises(continuation.ContinuationError, match='without publishing'):
        with continuation.runner_lease(root, '100', 1):
            pass
    assert continuation.read_state(root)['status'] == 'failed'


@pytest.mark.parametrize('after_pause', [False, True])
def test_body_or_late_cleanup_error_stops_continuation(continuation, run, after_pause):
    root, _ = run
    with pytest.raises(RuntimeError, match='cleanup failed'):
        with continuation.runner_lease(root, '100', 1):
            if after_pause:
                continuation.publish_outcome(root, status='paused', checkpoint_path='checkpoint.json')
            raise RuntimeError('cleanup failed')
    state = continuation.read_state(root)
    assert state['status'] == 'failed'
    assert state['segments'][-1]['status'] == 'failed'
    assert 'cleanup failed' in state['error']


def test_active_lease_blocks_other_worker_and_submit(continuation, run, monkeypatch):
    root, _ = run
    calls = fake_sbatch(monkeypatch, continuation, root)
    with continuation.runner_lease(root, '100', 1):
        with ThreadPoolExecutor(max_workers=1) as executor:
            def attempt():
                with continuation.runner_lease(root, '101', 1):
                    pass
            with pytest.raises(continuation.RunnerActiveError):
                executor.submit(attempt).result(timeout=2)
        continuation.publish_outcome(root, status='paused', checkpoint_path='checkpoint.json')
        with pytest.raises(continuation.RunnerActiveError):
            submit(continuation, run)
    assert calls == []


@pytest.mark.parametrize('gpu_count', [2, 3])
def test_exact_resources_dependency_identity_and_duplicate_submit(continuation, run, monkeypatch, gpu_count):
    root, wrapper = run
    monkeypatch.setenv('PDE_GAV_SLURM_ACCOUNT', 'test-account')
    paused = pause(continuation, root)
    calls = fake_sbatch(monkeypatch, continuation, root)
    state = submit(continuation, run, gpu_count=gpu_count)
    assert state['status'] == 'submitted'
    assert state['run_id'] == paused['run_id']
    assert submit(continuation, run, gpu_count=gpu_count) == state
    assert len(calls) == 1
    command, options = calls[0]
    for value in ['--account=test-account', '--partition=gpubase_l40s_b2', '--qos=normal',
                  '--time=12:00:00', '--cpus-per-task=32', '--mem=160G',
                  f'--gres=gpu:l40s:{gpu_count}', '--dependency=afterok:100']:
        assert value in command
    assert command[-3:] == [str(wrapper), 'full', 'a path with spaces']
    environment = options['env']
    assert environment['NS_CONTINUATION_RUN_ROOT'] == str(root)
    assert environment['NS_CONTINUATION_SEGMENT_INDEX'] == '2'
    assert environment['NS_CONTINUATION_MAX_SEGMENTS'] == '12'
    assert environment['NS_CONTINUATION_INTENT_ID'] == state['submissions'][-1]['intent_id']
    assert environment['NS_CONTINUATION_CHECKPOINT_SHA256'] == paused['checkpoint']['sha256']
    assert options['timeout'] == 60
    assert state['submissions'][-1]['job_id'] == '200'


def test_concurrent_submitters_make_one_request(continuation, run, monkeypatch):
    root, _ = run
    pause(continuation, root)
    calls = fake_sbatch(monkeypatch, continuation, root)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(submit, continuation, run) for _ in range(2)]
        results = [future.result(timeout=3) for future in futures]
    assert len(calls) == 1
    assert results[0] == results[1]


def test_checkpoint_checksum_does_not_block_submission(continuation, run, monkeypatch):
    root, _ = run
    pause(continuation, root)
    calls = fake_sbatch(monkeypatch, continuation, root)
    (root / 'checkpoint.json').write_text('{"round": 999}')
    assert submit(continuation, run)['status'] == 'submitted'
    assert len(calls) == 1


@pytest.mark.parametrize('kind', ['timeout', 'nonzero', 'malformed'])
def test_uncertain_submission_is_durable_and_never_retried(continuation, run, monkeypatch, kind):
    root, _ = run
    pause(continuation, root)
    calls = []

    def invoke(*args, **kwargs):
        calls.append(args)
        if kind == 'timeout':
            raise subprocess.TimeoutExpired('sbatch', 60)
        return SimpleNamespace(returncode=1 if kind == 'nonzero' else 0,
                               stdout='200' if kind == 'nonzero' else 'Submitted batch job 200', stderr='err')

    monkeypatch.setattr(continuation.subprocess, 'run', invoke)
    with pytest.raises(continuation.SubmissionUncertainError):
        submit(continuation, run)
    state = continuation.read_state(root)
    assert state['status'] == 'submission_unknown'
    assert state['submissions'][-1]['intent_id']
    with pytest.raises(continuation.SubmissionUncertainError):
        submit(continuation, run)
    assert len(calls) == 1


def uncertain(continuation, run, monkeypatch, *, max_segments=12):
    root, _ = run
    pause(continuation, root, max_segments=max_segments)
    fake_sbatch(monkeypatch, continuation, root, stdout='lost receipt')
    with pytest.raises(continuation.SubmissionUncertainError):
        submit(continuation, run, max_segments=max_segments)
    return continuation.read_state(root)['submissions'][-1]


def test_reconcile_requires_intent_comment_and_evidence(continuation, run, monkeypatch):
    root, _ = run
    record = uncertain(continuation, run, monkeypatch)
    args = dict(intent_id=record['intent_id'], job_id='200', evidence='scontrol matched saved comment')
    with pytest.raises(continuation.ContinuationError, match='comment'):
        continuation.reconcile_submission(root, **args, scheduler_comment='wrong')
    with pytest.raises(ValueError, match='evidence'):
        continuation.reconcile_submission(root, **{**args, 'evidence': ''}, scheduler_comment=record['comment'])
    with pytest.raises(continuation.ContinuationError, match='intent'):
        continuation.reconcile_submission(root, **{**args, 'intent_id': 'wrong'}, scheduler_comment=record['comment'])
    state = continuation.reconcile_submission(root, **args, scheduler_comment=record['comment'])
    assert state['status'] == 'submitted'
    assert state['submissions'][-1]['job_id'] == '200'


def test_reconciled_absent_attempts_still_bounded(continuation, run, monkeypatch):
    root, _ = run
    record = uncertain(continuation, run, monkeypatch, max_segments=2)
    with pytest.raises(ValueError, match='confirmed absence'):
        continuation.reconcile_submission(root, intent_id=record['intent_id'], evidence='not enough')
    state = continuation.reconcile_submission(root, intent_id=record['intent_id'], confirmed_absent=True,
                                              evidence='scheduler and accounting confirmed no matching job')
    assert state['status'] == 'paused'
    with pytest.raises(continuation.SegmentLimitError):
        submit(continuation, run, max_segments=2)


def test_maximum_segment_stops(continuation, run, monkeypatch):
    root, _ = run
    pause(continuation, root, max_segments=1)
    calls = fake_sbatch(monkeypatch, continuation, root)
    with pytest.raises(continuation.SegmentLimitError):
        submit(continuation, run, max_segments=1)
    assert calls == []


def test_resume_identity_and_checkpoint_are_checked(continuation, run, monkeypatch):
    root, _ = run
    first = pause(continuation, root)
    fake_sbatch(monkeypatch, continuation, root)
    state = submit(continuation, run)
    intent = state['submissions'][-1]['intent_id']
    for job, index, supplied_intent in [('201', 2, intent), ('200', 3, intent), ('200', 2, 'wrong')]:
        with pytest.raises(continuation.ContinuationError, match='differs'):
            with continuation.runner_lease(root, job, index, intent_id=supplied_intent):
                pass
    with continuation.runner_lease(root, '200', 2, intent_id=intent) as resumed:
        assert resumed['run_id'] == first['run_id']
        assert len(resumed['segments']) == 2
        (root / 'checkpoint.json').write_text('{"round": 3}')
        continuation.publish_outcome(root, status='paused', checkpoint_path='checkpoint.json')
    next_state = submit(continuation, run)
    assert '--dependency=afterok:200' in next_state['submissions'][-1]['command']
    assert next_state['submissions'][-1]['segment_index'] == 3


def test_worker_can_confirm_matching_uncertain_intent(continuation, run, monkeypatch):
    root, _ = run
    record = uncertain(continuation, run, monkeypatch)
    with continuation.runner_lease(root, '200', 2, intent_id=record['intent_id']):
        continuation.publish_outcome(root, status='completed')
    state = continuation.read_state(root)
    assert state['status'] == 'completed'
    assert state['submissions'][-1]['worker_acknowledged'] is True


def test_worker_starting_before_receipt_cannot_race_state(continuation, run, monkeypatch):
    root, _ = run
    pause(continuation, root)
    started = Event()
    errors = []
    worker_threads = []

    def invoke(command, **kwargs):
        intent = kwargs['env']['NS_CONTINUATION_INTENT_ID']
        def worker():
            started.set()
            try:
                with continuation.runner_lease(root, '200', 2, intent_id=intent):
                    continuation.publish_outcome(root, status='completed')
            except BaseException as error:
                errors.append(error)
        thread = Thread(target=worker)
        worker_threads.append(thread)
        thread.start()
        assert started.wait(2)
        return SimpleNamespace(returncode=0, stdout='200', stderr='')

    monkeypatch.setattr(continuation.subprocess, 'run', invoke)
    submit(continuation, run)
    for thread in worker_threads:
        thread.join(timeout=3)
        assert not thread.is_alive()
    assert errors == []
    assert continuation.read_state(root)['status'] == 'completed'


def test_wrapper_and_resource_drift_rejected(continuation, run, monkeypatch):
    root, wrapper = run
    pause(continuation, root)
    calls = fake_sbatch(monkeypatch, continuation, root)
    submit(continuation, run)
    with pytest.raises(continuation.ContinuationError, match='resources changed'):
        submit(continuation, run, gpu_count=2)
    with pytest.raises(continuation.ContinuationError, match='persisted run limit'):
        submit(continuation, run, max_segments=13)
    wrapper.write_text('#!/bin/bash\nexit 1\n')
    with pytest.raises(continuation.ContinuationError, match='resources changed'):
        submit(continuation, run)
    assert len(calls) == 1


@pytest.mark.parametrize('gpu_count', [1, 4, True, 2.0, '2'])
def test_gpu_count_must_be_explicit_integer_mode(continuation, run, gpu_count):
    root, _ = run
    pause(continuation, root)
    with pytest.raises(ValueError, match='2 or 3 GPUs'):
        submit(continuation, run, gpu_count=gpu_count)


def test_checkpoint_cannot_be_outside_persistent_run(continuation, run):
    root, wrapper = run
    with continuation.runner_lease(root, '100', 1):
        with pytest.raises(continuation.ContinuationError, match='inside the persistent'):
            continuation.publish_outcome(root, status='paused', checkpoint_path=wrapper)
        continuation.publish_outcome(root, status='failed')


def test_node_local_run_directory_rejected(continuation, run, monkeypatch):
    root, _ = run
    monkeypatch.setenv('SLURM_TMPDIR', str(root.parent))
    with pytest.raises(ValueError, match='SLURM_TMPDIR'):
        with continuation.runner_lease(root, '100', 1):
            pass


@pytest.mark.parametrize('profile,gpus,cpus,memory', [
    ('evaluation_3h', 1, 16, '96G'), ('development_3h', 2, 32, '160G'),
    ('development_3h', 3, 32, '160G'), ('evaluation_3h_compact', 1, 8, '64G'),
    ('evaluation_3h_parallel', 1, 16, '64G'),
])
def test_short_profile_retained_by_idempotent_successor(continuation, run, monkeypatch, profile, gpus, cpus, memory):
    root, _ = run
    pause(continuation, root)
    calls = fake_sbatch(monkeypatch, continuation, root)
    state = submit(continuation, run, gpu_count=gpus, resource_profile=profile)
    assert submit(continuation, run, gpu_count=gpus, resource_profile=profile) == state
    assert len(calls) == 1
    partitions = (','.join(f'gpubase_l40s_b{i}' for i in range(1, 6))
                  if profile in {'evaluation_3h_compact', 'evaluation_3h_parallel'} else 'gpubase_l40s_b1')
    for value in [f'--partition={partitions}', '--time=03:00:00',
                  f'--gres=gpu:l40s:{gpus}', f'--cpus-per-task={cpus}', f'--mem={memory}',
                  '--dependency=afterok:100']:
        assert value in calls[0][0]
    intent = state['submissions'][-1]['intent_id']
    with continuation.runner_lease(root, '200', 2, intent_id=intent):
        continuation.publish_outcome(root, status='paused', checkpoint_path='checkpoint.json')
    successor = submit(continuation, run, gpu_count=gpus, resource_profile=profile)
    assert '--time=03:00:00' in successor['submissions'][-1]['command']
    assert '--dependency=afterok:200' in successor['submissions'][-1]['command']


def test_short_profile_cannot_silently_shrink_an_existing_run(continuation, run, monkeypatch):
    root, _ = run
    pause(continuation, root)
    calls = fake_sbatch(monkeypatch, continuation, root)
    submit(continuation, run)
    with pytest.raises(continuation.ContinuationError, match='resources changed'):
        submit(continuation, run, gpu_count=1, resource_profile='evaluation_3h')
    assert len(calls) == 1


@pytest.mark.parametrize('profile,hours,memory', [
    ('plasma_gpu_12h_b2_b3', 12, '128G'), ('plasma_gpu_6h_b2_b3', 6, '64G'),
])
def test_plasma_combined_partitions_persist_across_successors(
        continuation, run, monkeypatch, profile, hours, memory):
    root, _ = run
    pause(continuation, root)
    calls = fake_sbatch(monkeypatch, continuation, root)
    state = submit(continuation, run, gpu_count=4, resource_profile=profile)
    assert submit(continuation, run, gpu_count=4, resource_profile=profile) == state
    assert len(calls) == 1
    for flag in ['--partition=gpubase_l40s_b2,gpubase_l40s_b3',
                 f'--time={hours:02d}:00:00', '--cpus-per-task=24',
                 f'--mem={memory}', '--gres=gpu:l40s:4', '--dependency=afterok:100']:
        assert flag in calls[0][0]
    with pytest.raises(continuation.ContinuationError, match='resources changed'):
        submit(continuation, run, gpu_count=4, resource_profile=profile.removesuffix('_b2_b3'))
    intent = state['submissions'][-1]['intent_id']
    with continuation.runner_lease(root, '200', 2, intent_id=intent):
        continuation.publish_outcome(root, status='paused', checkpoint_path='checkpoint.json')
    successor = submit(continuation, run, gpu_count=4, resource_profile=profile)
    assert '--partition=gpubase_l40s_b2,gpubase_l40s_b3' in successor['submissions'][-1]['command']
    assert '--dependency=afterok:200' in successor['submissions'][-1]['command']
