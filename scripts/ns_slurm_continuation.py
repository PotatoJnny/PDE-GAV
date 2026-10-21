#!/usr/bin/env python3
"""Bounded Slurm continuation for one persistent NS run.

The scientific runner owns checkpoint contents and pauses only at supported
boundaries. This module owns the separate ``slurm_continuation.json`` record,
its atomic writes, a process-held runner lease, and at most one next submission.
It never cancels jobs, changes accounts, retries uncertain submissions, or
interprets scientific checkpoint sidecars. Run/checkpoint paths must be on the
caller's persistent shared storage, not node-local scratch.

Runner integration::

    with runner_lease(run_root, job_id, segment_index):
        ...  # run and close all scientific execution services
        publish_outcome(run_root, status='paused', checkpoint_path=manifest)
    submit_next(run_root, slurm_script=wrapper, script_args=args, gpu_count=2)

A paused segment exits successfully. The next job has an ``afterok`` dependency
and receives NS_CONTINUATION_{RUN_ROOT,SEGMENT_INDEX,INTENT_ID,MAX_SEGMENTS,
CHECKPOINT_PATH,CHECKPOINT_SHA256} in its submission environment. A wrapper must
pass these to the same runner; each Agent still gets its original hard budget.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import time
from uuid import uuid4


_SCHEMA = 'ns_slurm_continuation_v1'
_STATE = 'slurm_continuation.json'
_CURRENT = ContextVar('ns_slurm_runner_lease', default=None)
_JOB_ID = re.compile(r'[1-9][0-9]*\Z')
_RECEIPT = re.compile(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?\Z')
_OUTCOMES = {'paused', 'completed', 'failed'}
_SUBMISSION_STATES = {'submitting', 'submission_unknown', 'submitted'}
PLASMA_L40S_PARTITIONS = 'gpubase_l40s_b2,gpubase_l40s_b3'


class ContinuationError(RuntimeError):
    pass


class RunnerActiveError(ContinuationError):
    pass


class SubmissionUncertainError(ContinuationError):
    pass


class SegmentLimitError(ContinuationError):
    pass


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a finite positive integer')
    return value


def _job_id(value):
    if not isinstance(value, str) or not _JOB_ID.fullmatch(value):
        raise ValueError('job_id must be a positive numeric Slurm job id')
    return value


def _root(value, *, create=False):
    root = Path(value).expanduser().resolve()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ContinuationError('persistent run directory does not exist')
    scratch = os.environ.get('SLURM_TMPDIR')
    if scratch and root.is_relative_to(Path(scratch).resolve()):
        raise ValueError('continuation run directory must not be under SLURM_TMPDIR')
    return root


@contextmanager
def _lock(root, name, *, blocking=True):
    flags = os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor = os.open(root / name, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ContinuationError('continuation lock must be a regular file')
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise RunnerActiveError('this NS run already has an active runner') from error
        yield
    finally:
        os.close(descriptor)


def _write(root, state):
    descriptor, temporary_name = tempfile.mkstemp(prefix='.slurm-continuation-', suffix='.tmp', dir=root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(state, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, root / _STATE)
        directory = os.open(root, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _read(root):
    try:
        state = json.loads((root / _STATE).read_text())
    except (OSError, ValueError) as error:
        raise ContinuationError('continuation state is unavailable or invalid') from error
    if (not isinstance(state, dict) or state.get('schema') != _SCHEMA
            or state.get('run_root') != str(root)
            or state.get('status') not in _OUTCOMES | _SUBMISSION_STATES | {'running'}
            or not isinstance(state.get('segments'), list)
            or not isinstance(state.get('submissions'), list)):
        raise ContinuationError('continuation state has the wrong schema or run identity')
    _positive_integer(state.get('segment_index'), 'saved segment_index')
    _positive_integer(state.get('max_segments'), 'saved max_segments')
    _job_id(state.get('job_id'))
    if state['segment_index'] > state['max_segments']:
        raise ContinuationError('saved segment index exceeds its limit')
    return state


def read_state(run_root):
    root = _root(run_root)
    with _lock(root, '.ns-continuation-state.lock'):
        return _read(root)


def _checkpoint(root, path, expected=None):
    supplied = Path(path)
    resolved = (supplied if supplied.is_absolute() else root / supplied).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ContinuationError('checkpoint must be a committed file inside the persistent run directory')
    digest = sha256()
    with resolved.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    actual = digest.hexdigest()
    return {'path': str(resolved), 'sha256': actual}


def _saved_checkpoint(root, state):
    saved = state.get('checkpoint')
    if not isinstance(saved, dict) or set(saved) != {'path', 'sha256'}:
        raise ContinuationError('paused run has no committed checkpoint descriptor')
    return _checkpoint(root, saved['path'], saved['sha256'])


@contextmanager
def runner_lease(run_root, job_id, segment_index, *, max_segments=12, intent_id=None):
    """Hold one process lease until all execution/cleanup and outcome writes end.

    An initial segment is registered only once. Later segments must match the
    persisted submission identity. Neither a crashed running segment nor a
    completed/failed segment can silently acquire a fresh Agent budget.
    """
    root = _root(run_root, create=True)
    _job_id(job_id)
    _positive_integer(segment_index, 'segment_index')
    _positive_integer(max_segments, 'max_segments')
    if segment_index > max_segments:
        raise SegmentLimitError('segment_index exceeds max_segments')
    if _CURRENT.get() is not None:
        raise RunnerActiveError('nested NS runner leases are not supported')
    intent_id = intent_id or os.environ.get('NS_CONTINUATION_INTENT_ID')
    with _lock(root, '.ns-continuation-runner.lock', blocking=False):
        with _lock(root, '.ns-continuation-state.lock'):
            if not (root / _STATE).exists():
                if segment_index != 1:
                    raise ContinuationError('the first registered segment must have index 1')
                state = {'schema': _SCHEMA, 'run_root': str(root), 'run_id': uuid4().hex,
                         'status': 'running', 'job_id': job_id, 'segment_index': 1,
                         'max_segments': max_segments, 'checkpoint': None,
                         'segments': [], 'submissions': [], 'submission_spec': None}
            else:
                state = _read(root)
                if max_segments != state['max_segments']:
                    raise ContinuationError('max_segments differs from the persisted run limit')
                if state['status'] not in _SUBMISSION_STATES or not state['submissions']:
                    raise ContinuationError('only the recorded next submission may start another segment')
                submission = state['submissions'][-1]
                if (submission['segment_index'] != segment_index
                        or segment_index != state['segment_index'] + 1
                        or submission['intent_id'] != intent_id
                        or submission.get('job_id') not in {None, job_id}):
                    raise ContinuationError('runner job/segment/intent differs from the submitted continuation')
                _saved_checkpoint(root, state)
                # A real worker can acknowledge a matching intent even if its
                # submitter lost sbatch's receipt after the job was accepted.
                submission.update(job_id=job_id, status='running', worker_acknowledged=True)
                state.update(status='running', segment_index=segment_index, job_id=job_id)
            state['segments'].append({'segment_index': segment_index, 'job_id': job_id,
                                      'status': 'running', 'started_at_unix': time.time()})
            _write(root, state)
        token = _CURRENT.set((root, job_id, segment_index))
        try:
            yield json.loads(json.dumps(state))
        except BaseException as error:
            _publish_failure(root, job_id, segment_index, f'{type(error).__name__}: {error}')
            raise
        else:
            with _lock(root, '.ns-continuation-state.lock'):
                ended = _read(root)
            if ended['status'] not in _OUTCOMES:
                _publish_failure(root, job_id, segment_index, 'runner exited without publishing an outcome')
                raise ContinuationError('runner exited without publishing paused/completed/failed')
        finally:
            _CURRENT.reset(token)


def _publish_failure(root, job_id, segment_index, message):
    with _lock(root, '.ns-continuation-state.lock'):
        state = _read(root)
        if state['job_id'] != job_id or state['segment_index'] != segment_index:
            raise ContinuationError('runner identity changed while recording failure')
        state.update(status='failed', error=message[:4000])
        state['segments'][-1].update(status='failed', finished_at_unix=time.time(), error=message[:4000])
        _write(root, state)


def publish_outcome(run_root, *, status, checkpoint_path=None, checkpoint_sha256=None):
    """Publish a safe scientific outcome while the caller still holds its lease.

    A paused outcome certifies that the engine has durably committed the whole
    checkpoint generation and closed scientific services. Only the manifest's
    file hash is checked here; the engine validates its referenced sidecars.
    """
    root = _root(run_root)
    lease = _CURRENT.get()
    if lease is None or lease[0] != root:
        raise ContinuationError('publishing an outcome requires this process to hold the runner lease')
    if status not in _OUTCOMES:
        raise ValueError('outcome status must be paused, completed, or failed')
    if status == 'paused' and checkpoint_path is None:
        raise ContinuationError('paused outcome requires a committed checkpoint file')
    checkpoint = (_checkpoint(root, checkpoint_path, checkpoint_sha256)
                  if checkpoint_path is not None else None)
    with _lock(root, '.ns-continuation-state.lock'):
        state = _read(root)
        if state['job_id'] != lease[1] or state['segment_index'] != lease[2] or state['status'] != 'running':
            raise ContinuationError('runner outcome was already published or its identity changed')
        state['status'] = status
        if checkpoint is not None:
            state['checkpoint'] = checkpoint
        state['segments'][-1].update(status=status, finished_at_unix=time.time(),
                                     checkpoint=state.get('checkpoint'))
        _write(root, state)
        return state


def _submission_spec(slurm_script, script_args, gpu_count, resource_profile='development_12h'):
    script = Path(slurm_script).expanduser().resolve()
    if not script.is_file():
        raise ValueError('slurm_script must be an existing frozen wrapper')
    if isinstance(script_args, (str, bytes)):
        raise TypeError('script_args must be a sequence of separate arguments')
    arguments = tuple(script_args)
    if any(not isinstance(value, str) or '\0' in value for value in arguments):
        raise ValueError('script_args must contain strings without NUL bytes')
    profiles = {
        'development_12h': ({2, 3}, 'gpubase_l40s_b2', 43200, 32, '160G'),
        'evaluation_3h': ({1}, 'gpubase_l40s_b1', 10800, 16, '96G'),
        'evaluation_3h_compact': (
            {1}, ','.join(f'gpubase_l40s_b{i}' for i in range(1, 6)), 10800, 8, '64G'),
        'evaluation_3h_parallel': (
            {1}, ','.join(f'gpubase_l40s_b{i}' for i in range(1, 6)), 10800, 16, '64G'),
        'development_3h': ({2, 3}, 'gpubase_l40s_b1', 10800, 32, '160G'),
        'literature_3h': ({1}, 'gpubase_bygpu_b1', 10800, 16, '64G'),
        'plasma_gpu_12h': ({4}, 'gpubase_l40s_b2', 43200, 24, '128G'),
        'plasma_gpu_6h': ({4}, 'gpubase_l40s_b2', 21600, 24, '64G'),
        # Preserve the legacy profiles for already-frozen b2 deployments.
        'plasma_gpu_12h_b2_b3': ({4}, PLASMA_L40S_PARTITIONS, 43200, 24, '128G'),
        'plasma_gpu_6h_b2_b3': ({4}, PLASMA_L40S_PARTITIONS, 21600, 24, '64G'),
    }
    if resource_profile not in profiles:
        raise ValueError('unknown NS resource profile')
    allowed, partition, seconds, cpus, memory = profiles[resource_profile]
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count not in allowed:
        requirement = ('1 GPU for evaluation only' if allowed == {1} else
                       '4 GPUs' if allowed == {4} else '2 or 3 GPUs')
        raise ValueError(f'gpu_count must explicitly retain {requirement}')
    return {'slurm_script': str(script), 'script_sha256': sha256(script.read_bytes()).hexdigest(),
            'script_args': list(arguments), 'gpu_count': gpu_count,
            'account': os.environ.get('PDE_GAV_SLURM_ACCOUNT', ''), 'qos': 'normal', 'partition': partition,
            'wall_seconds': seconds, 'cpus': cpus, 'memory': memory}


def evaluation_worker_limit(resource_profile, cpus):
    """Leave CPUs for the Controller/Oracle; preserve older profile limits."""
    return cpus - 2 if resource_profile == 'evaluation_3h_parallel' else cpus // 2


def submission_resource_flags(spec):
    """Use identical explicit requests on the initial allocation and successor."""
    account = [f'--account={spec["account"]}'] if spec.get('account') else []
    return account + [f'--qos={spec["qos"]}',
            f'--partition={spec["partition"]}',
            f'--time={spec["wall_seconds"] // 3600:02d}:00:00',
            f'--cpus-per-task={spec["cpus"]}', f'--mem={spec["memory"]}',
            f'--gres=gpu:l40s:{spec["gpu_count"]}']


def submit_next(run_root, *, slurm_script, script_args, gpu_count, max_segments=12,
                resource_profile='development_12h'):
    """Submit once from paused; duplicate calls return the recorded active job.

    Intent is fsynced before sbatch. No ambiguous failure is automatically
    retried. At most max_segments-1 submission attempts are made after the
    original allocation, including attempts explicitly reconciled as absent.
    """
    root = _root(run_root)
    _positive_integer(max_segments, 'max_segments')
    spec = _submission_spec(slurm_script, script_args, gpu_count, resource_profile)
    with _lock(root, '.ns-continuation-state.lock'):
        state = _read(root)
        if state['max_segments'] != max_segments:
            raise ContinuationError('max_segments differs from the persisted run limit')
        if state['submission_spec'] is not None and state['submission_spec'] != spec:
            raise ContinuationError('continuation wrapper, arguments, or resources changed')
        if state['status'] in {'completed', 'failed', 'running', 'submitted'}:
            return state
        if state['status'] in {'submitting', 'submission_unknown'}:
            raise SubmissionUncertainError('submission intent requires reconciliation; sbatch will not be repeated')
        # The state lock is held while probing the long runner lock, but this
        # probe never blocks: a publishing runner can always finish its write.
        with _lock(root, '.ns-continuation-runner.lock', blocking=False):
            checkpoint = _saved_checkpoint(root, state)
            if state['segment_index'] >= max_segments or len(state['submissions']) >= max_segments - 1:
                raise SegmentLimitError('NS continuation segment/submission limit reached')
            index = state['segment_index'] + 1
            intent_id = uuid4().hex
            comment = f'ns-continuation:{state["run_id"]}:{intent_id}'
            (root / 'slurm_logs').mkdir(exist_ok=True)
            command = ['sbatch', '--parsable', '--export=ALL',
                       *submission_resource_flags(spec),
                       f'--dependency=afterok:{state["job_id"]}',
                       f'--job-name=ns-segment-{state["run_id"][:12]}-{index}',
                       f'--comment={comment}', f'--output={root / "slurm_logs" / "segment-%j.out"}',
                       f'--error={root / "slurm_logs" / "segment-%j.err"}',
                       spec['slurm_script'], *spec['script_args']]
            submission = {'segment_index': index, 'intent_id': intent_id, 'comment': comment,
                          'status': 'submitting', 'job_id': None, 'command': command,
                          'checkpoint': checkpoint, 'created_at_unix': time.time()}
            state['submissions'].append(submission)
            state.update(status='submitting', submission_spec=spec)
            _write(root, state)
        # A rapidly starting worker may acquire the runner lease now; its short
        # state-lock wait lasts until the receipt or uncertainty is committed.
        environment = dict(os.environ)
        environment.update(NS_CONTINUATION_RUN_ROOT=str(root), NS_CONTINUATION_SEGMENT_INDEX=str(index),
                           NS_CONTINUATION_INTENT_ID=intent_id, NS_CONTINUATION_MAX_SEGMENTS=str(max_segments),
                           NS_CONTINUATION_CHECKPOINT_PATH=checkpoint['path'],
                           NS_CONTINUATION_CHECKPOINT_SHA256=checkpoint['sha256'])
        try:
            completed = subprocess.run(command, env=environment, text=True, capture_output=True,
                                       timeout=60, check=False)
            receipt = _RECEIPT.fullmatch(completed.stdout.strip())
            if completed.returncode != 0 or receipt is None:
                raise ContinuationError('sbatch returned no unambiguous successful job id: '
                                        f'exit={completed.returncode}; stdout={completed.stdout[:1000]!r}; '
                                        f'stderr={completed.stderr[:1000]!r}')
            submission.update(status='submitted', job_id=receipt.group(1), accepted_at_unix=time.time())
            state['status'] = 'submitted'
            _write(root, state)
            return state
        except BaseException as error:
            submission.update(status='submission_unknown', error=f'{type(error).__name__}: {error}'[:4000])
            state['status'] = 'submission_unknown'
            _write(root, state)
            raise SubmissionUncertainError(
                f'sbatch outcome is uncertain for intent {intent_id}; inspect its scheduler comment before retrying'
            ) from error


def reconcile_submission(run_root, *, intent_id, evidence, job_id=None, scheduler_comment=None,
                         confirmed_absent=False):
    """Record an explicitly checked scheduler result; never query or resubmit.

    The caller must check the scheduler/accounting using the saved comment,
    submission time and wrapper. A known accepted job requires that exact
    comment; confirming absence requires explicit evidence and consumes the
    original bounded submission attempt. A still-active runner blocks changes.
    """
    root = _root(run_root)
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError('scheduler reconciliation requires explicit evidence')
    if job_id is not None:
        _job_id(job_id)
        if confirmed_absent:
            raise ValueError('a submission cannot be both accepted and absent')
    elif confirmed_absent is not True:
        raise ValueError('provide a verified job_id or explicitly confirmed absence')
    with _lock(root, '.ns-continuation-state.lock'):
        state = _read(root)
        if state['status'] not in {'submitting', 'submission_unknown'} or not state['submissions']:
            raise ContinuationError('only an uncertain submission can be reconciled')
        submission = state['submissions'][-1]
        if submission['intent_id'] != intent_id:
            raise ContinuationError('submission intent changed after scheduler inspection')
        with _lock(root, '.ns-continuation-runner.lock', blocking=False):
            _saved_checkpoint(root, state)
            if job_id is not None:
                if scheduler_comment != submission['comment']:
                    raise ContinuationError('observed scheduler comment does not match the submission intent')
                submission.update(status='submitted', job_id=job_id, reconciliation_evidence=evidence)
                state['status'] = 'submitted'
            else:
                submission.update(status='confirmed_absent', reconciliation_evidence=evidence)
                state['status'] = 'paused'
            _write(root, state)
            return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--slurm-script', required=True, type=Path)
    parser.add_argument('--gpu-count', required=True, type=int, choices=(1, 2, 3, 4))
    parser.add_argument('--resource-profile', default='development_12h',
                        choices=('development_12h', 'evaluation_3h', 'evaluation_3h_compact', 'evaluation_3h_parallel', 'development_3h', 'literature_3h', 'plasma_gpu_12h', 'plasma_gpu_6h', 'plasma_gpu_12h_b2_b3', 'plasma_gpu_6h_b2_b3'))
    parser.add_argument('--max-segments', type=int, default=12)
    parser.add_argument('script_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    script_args = args.script_args[1:] if args.script_args[:1] == ['--'] else args.script_args
    state = submit_next(args.run_root, slurm_script=args.slurm_script, script_args=script_args,
                        gpu_count=args.gpu_count, max_segments=args.max_segments,
                        resource_profile=args.resource_profile)
    print(json.dumps(state, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
