"""Formal Slurm dispatch resumes one run across scientific and final stages."""
from contextvars import Context
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from tests.test_ns_draft_dispatch import ROOT, driver


@pytest.fixture
def dispatch(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    import ns_slurm_continuation as continuation
    module = driver()
    batch = tmp_path / 'batch'
    source = batch / 'source'
    (source / 'scripts').mkdir(parents=True)
    (source / 'scripts/slurm_ns_draft_first_batch.sh').write_text('#!/bin/bash\nexit 0\n')
    for name in module.AUTHORIZED_METHODS:
        destination = source / module.CONFIG_PATHS[name]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / module.CONFIG_PATHS[name]).read_bytes())
    control = batch / 'control'
    control.mkdir()
    (control / 'preflight_passed.json').write_text(json.dumps({
        'status': 'passed', 'protocol_revision': module.PAPER_PROTOCOL}))
    monkeypatch.setattr(module, 'ROOT', source)
    monkeypatch.setattr(module, 'verify_snapshot', lambda: None)
    monkeypatch.setattr(module, 'allocation_remaining_seconds', lambda job: 43200)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    for key in tuple(__import__('os').environ):
        if key.startswith('NS_CONTINUATION_'):
            monkeypatch.delenv(key)
    monkeypatch.setenv('NS_JOB_GPU_TOKENS', 'GPU-a,GPU-b,GPU-c')
    submitted = []
    def sbatch(command, **kwargs):
        assert command[0] == 'sbatch'
        receipt = str(100002 + len(submitted))
        submitted.append((command, dict(kwargs['env']), receipt))
        return SimpleNamespace(returncode=0, stdout=receipt + '\n', stderr='')
    monkeypatch.setattr(continuation.subprocess, 'run', sbatch)
    def run(mode='base', job='100001'):
        monkeypatch.setenv('SLURM_JOB_ID', job)
        return module.main(['--mode', mode, '--batch-root', str(batch)])
    def next_segment():
        command, environment, receipt = submitted[-1]
        for key, value in environment.items():
            if key.startswith('NS_CONTINUATION_'):
                monkeypatch.setenv(key, value)
        return receipt
    return SimpleNamespace(module=module, batch=batch, submitted=submitted, run=run,
                           next_segment=next_segment, continuation=continuation)


def _save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def _scientific_checkpoint(scientific, *, progress='round-1:28-cases', sequence=1, baseline=False):
    name = 'baseline_checkpoints' if baseline else 'checkpoints'
    return _save(scientific / name / 'checkpoint.json', {
        'schema': 'campaign_checkpoint_v1', 'sequence': sequence,
        'payload': {'sha256': f'payload-with-timing-and-rng-{sequence}'}, 'progress': progress})


def _common_checkpoint(output, *, elapsed=1, completed=None):
    return _save(output / 'common_evaluation/checkpoint.json', {
        'schema': 'common_evaluation_checkpoint_v1', 'elapsed_wall_seconds': elapsed,
        'completed': {'0': {'sha256': 'result-0'}} if completed is None else completed})


def _read(path):
    return json.loads(path.read_text())


def test_scientific_then_common_final_pause_resume_keeps_same_run(dispatch, monkeypatch):
    calls, final_calls = [], []
    def canonical(argv, *, segment_budget):
        calls.append(tuple(argv))
        assert argv[argv.index('--worker-gpus') + 1] == 'GPU-a,GPU-b'
        assert argv[argv.index('--controller-gpu') + 1] == 'GPU-c'
        assert not segment_budget.should_pause('development', 14400 + 600)
        execution = _read(Path(argv[argv.index('--config') + 1]))['execution']
        assert execution['agent_count'] == 2
        assert execution['agent_hard_wall_seconds'] == 14400
        assert execution['candidate_case_hard_wall_seconds'] == 900
        scientific = Path(argv[argv.index('--run-root') + 1])
        if len(calls) == 1:
            assert '--resume' not in argv
            _scientific_checkpoint(scientific)
            return 75
        assert '--resume' in argv
        return 0
    monkeypatch.setitem(sys.modules, 'run_canonical_ns_campaign', SimpleNamespace(main=canonical))
    def final(mode, bundle, output, batch, **options):
        final_calls.append((Path(bundle), Path(output), options['resume']))
        if len(final_calls) == 1:
            assert not options['resume']
            checkpoint = _common_checkpoint(output)
            return {'status': 'paused', 'checkpoint_path': str(checkpoint)}
        assert options['resume']
        return {'status': 'completed'}
    monkeypatch.setattr(dispatch.module, 'evaluate_final_bundle', final)
    assert dispatch.run() == 0  # afterok depends on this successful pause exit.
    output = dispatch.batch / 'runs/base-s101/attempt-100001'
    assert _read(output / 'workflow.json')['phase'] == 'scientific'
    assert '--dependency=afterok:100001' in dispatch.submitted[0][0]
    assert '--gres=gpu:l40s:3' in dispatch.submitted[0][0]
    assert '--time=12:00:00' in dispatch.submitted[0][0]
    assert dispatch.run(job=dispatch.next_segment()) == 0
    assert _read(output / 'workflow.json')['phase'] == 'final_evaluation'
    assert '--dependency=afterok:100002' in dispatch.submitted[1][0]
    assert dispatch.run(job=dispatch.next_segment()) == 0
    assert len(calls) == 2 and len(final_calls) == 2 and len(dispatch.submitted) == 2
    assert final_calls[0][0] == final_calls[1][0] == output / 'scientific/selected_candidate'
    assert final_calls[0][1] == final_calls[1][1] == output
    assert _read(output / 'workflow.json')['phase'] == 'completed'
    state = dispatch.continuation.read_state(output)
    assert state['status'] == 'completed'
    assert [row['status'] for row in state['segments']] == ['paused', 'paused', 'completed']
    assert _read(output / 'status.json')['returncode'] == 0


def test_fixed_competition_resumes_without_changing_gpu_or_agent_budget(dispatch, monkeypatch):
    from pde_agent_verifier_solver.engine.campaign_continuation import CampaignPaused
    from pde_agent_verifier_solver.plugins.ns.baselines import cli
    calls = []
    monkeypatch.setenv('NS_JOB_GPU_TOKENS', 'GPU-left,GPU-right')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-right,GPU-left')
    def baseline(config, scientific, *, resume, segment_budget):
        payload = _read(config)
        calls.append((payload, scientific, resume))
        assert payload['agent']['hard_wall_seconds'] == 14400
        assert payload['search'] == _read(ROOT / dispatch.module.CONFIG_PATHS['fixed'])['search']
        assert [row['gpu_devices'] for row in payload['agent']['roles']] == [['GPU-left'], ['GPU-right']]
        if not resume:
            _scientific_checkpoint(scientific, baseline=True)
            raise CampaignPaused(stage='baseline_selection')
        return SimpleNamespace(report={'selected_bundle': str(scientific / 'selected_candidate')})
    monkeypatch.setattr(cli, 'run_from_config', baseline)
    def final(*args, **kwargs):
        assert kwargs['evaluation_devices'] == ('cuda:1', 'cuda:0')
        return {'status': 'completed'}
    monkeypatch.setattr(dispatch.module, 'evaluate_final_bundle', final)
    assert dispatch.run(mode='fixed') == 0
    assert dispatch.run(mode='fixed', job=dispatch.next_segment()) == 0
    assert [row[2] for row in calls] == [False, True]
    assert calls[0][1] == calls[1][1]
    assert '--gres=gpu:l40s:2' in dispatch.submitted[0][0]


@pytest.mark.parametrize('phase', ['scientific', 'final_evaluation'])
def test_no_progress_stops_second_segment_despite_rewritten_checkpoint(dispatch, monkeypatch, phase):
    calls = []
    def canonical(argv, **kwargs):
        if phase == 'scientific':
            calls.append(1)
            _scientific_checkpoint(Path(argv[argv.index('--run-root') + 1]), sequence=len(calls))
            return 75
        return 0
    monkeypatch.setitem(sys.modules, 'run_canonical_ns_campaign', SimpleNamespace(main=canonical))
    def final(mode, bundle, output, batch, **options):
        calls.append(1)
        checkpoint = _common_checkpoint(output, elapsed=len(calls))
        return {'status': 'paused', 'checkpoint_path': str(checkpoint)}
    monkeypatch.setattr(dispatch.module, 'evaluate_final_bundle', final)
    assert dispatch.run() == 0
    job = dispatch.next_segment()
    with pytest.raises(RuntimeError, match='no scientific progress'):
        dispatch.run(job=job)
    output = dispatch.batch / 'runs/base-s101/attempt-100001'
    assert len(calls) == 2 and len(dispatch.submitted) == 1
    assert dispatch.continuation.read_state(output)['status'] == 'failed'
    assert _read(output / 'status.json')['returncode'] == 1
    assert _read(dispatch.batch / f'job_status/base-{job}.json')['status'] == 'failed'


def test_allocation_probe_failure_is_recorded_and_cannot_submit(dispatch, monkeypatch):
    def unavailable(job):
        raise RuntimeError('scontrol EndTime unavailable')
    monkeypatch.setattr(dispatch.module, 'allocation_remaining_seconds', unavailable)
    with pytest.raises(RuntimeError, match='EndTime unavailable'):
        dispatch.run()
    output = dispatch.batch / 'runs/base-s101/attempt-100001'
    state = _read(dispatch.batch / 'job_status/base-100001.json')
    assert state['status'] == 'failed' and state['returncode'] == 1
    assert dispatch.continuation.read_state(output)['status'] == 'failed'
    assert _read(output / 'status.json')['error'] == 'scontrol EndTime unavailable'
    assert not dispatch.submitted


def test_rejected_second_writer_does_not_overwrite_active_run_status(dispatch, monkeypatch):
    output = dispatch.batch / 'runs/base-s101/attempt-100001'
    monkeypatch.setenv('NS_CONTINUATION_RUN_ROOT', str(output))
    with dispatch.continuation.runner_lease(output, '100001', 1):
        expected = {'status': 'running', 'job_id': '100001', 'owner': True}
        _save(output / 'status.json', expected)
        # A fresh execution context reaches the actual file lock, rather than
        # the same-context nested-lease guard.
        with pytest.raises(dispatch.continuation.RunnerActiveError, match='active runner'):
            Context().run(dispatch.run, job='100099')
        assert _read(output / 'status.json') == expected
        assert dispatch.continuation.read_state(output)['job_id'] == '100001'
        assert _read(dispatch.batch / 'job_status/base-100099.json')['status'] == 'failed'
        dispatch.continuation.publish_outcome(output, status='completed')
    assert not dispatch.submitted


@pytest.mark.parametrize('failure', [RuntimeError('scientific runner error'), 13])
def test_scientific_failure_stops_without_resubmission(dispatch, monkeypatch, failure):
    def canonical(argv, **kwargs):
        if isinstance(failure, Exception):
            raise failure
        return failure
    monkeypatch.setitem(sys.modules, 'run_canonical_ns_campaign', SimpleNamespace(main=canonical))
    with pytest.raises(RuntimeError):
        dispatch.run()
    output = dispatch.batch / 'runs/base-s101/attempt-100001'
    assert dispatch.continuation.read_state(output)['status'] == 'failed'
    assert _read(output / 'status.json')['status'] == 'failed'
    assert not dispatch.submitted


def test_uncertain_sbatch_receipt_keeps_parent_success_for_possible_afterok(dispatch, monkeypatch):
    def canonical(argv, **kwargs):
        _scientific_checkpoint(Path(argv[argv.index('--run-root') + 1]))
        return 75
    monkeypatch.setitem(sys.modules, 'run_canonical_ns_campaign', SimpleNamespace(main=canonical))
    def lost_receipt(command, **kwargs):
        raise TimeoutError('receipt lost after possible acceptance')
    monkeypatch.setattr(dispatch.continuation.subprocess, 'run', lost_receipt)
    assert dispatch.run() == 0
    output = dispatch.batch / 'runs/base-s101/attempt-100001'
    state = dispatch.continuation.read_state(output)
    assert state['status'] == 'submission_unknown' and len(state['submissions']) == 1
    assert _read(output / 'status.json')['continuation_status'] == 'submission_unknown'
    assert _read(output / 'status.json')['returncode'] == 0


def test_progress_fingerprint_ignores_only_bookkeeping(dispatch, tmp_path):
    checkpoint = _scientific_checkpoint(tmp_path, sequence=1)
    first = dispatch.module._checkpoint_progress(checkpoint, 'scientific')
    _scientific_checkpoint(tmp_path, sequence=88)
    assert dispatch.module._checkpoint_progress(checkpoint, 'scientific') == first
    _scientific_checkpoint(tmp_path, progress='round-1:56-cases', sequence=89)
    assert dispatch.module._checkpoint_progress(checkpoint, 'scientific') != first
    with pytest.raises(ValueError, match='progress descriptor'):
        dispatch.module._checkpoint_progress(_save(tmp_path / 'unknown.json', {}), 'scientific')


def test_unknown_saved_workflow_phase_cannot_be_called_completed(dispatch):
    output = dispatch.batch / 'runs/base-s101/attempt-100001'
    _save(output / 'workflow.json', {'mode': 'base', 'phase': 'typo-phase',
                                     'source_snapshot': str(dispatch.module.ROOT)})
    with pytest.raises(ValueError, match='identity or phase'):
        dispatch.run()
    assert dispatch.continuation.read_state(output)['status'] == 'failed'
    assert _read(output / 'status.json')['returncode'] == 1
    assert not dispatch.submitted


def test_final_pause_rejects_checkpoint_from_wrong_stage(dispatch, monkeypatch):
    monkeypatch.setitem(sys.modules, 'run_canonical_ns_campaign', SimpleNamespace(main=lambda *args, **kwargs: 0))
    def wrong_stage(mode, bundle, output, batch, **options):
        checkpoint = _scientific_checkpoint(output / 'scientific')
        return {'status': 'paused', 'checkpoint_path': str(checkpoint)}
    monkeypatch.setattr(dispatch.module, 'evaluate_final_bundle', wrong_stage)
    with pytest.raises(ValueError, match='unexpected checkpoint path'):
        dispatch.run()
    output = dispatch.batch / 'runs/base-s101/attempt-100001'
    assert dispatch.continuation.read_state(output)['status'] == 'failed'
    assert not dispatch.submitted
