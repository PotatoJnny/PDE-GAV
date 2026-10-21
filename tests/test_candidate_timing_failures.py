"""Execution regressions for disclosed budgets and timing/cleanup failure boundaries."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Event
import time

import pytest

from pde_agent_verifier_solver.engine import (
    CandidateExecutionError, CandidateExecutionTimeout, PythonCandidateCodec,
    PythonTaskWorkerSpec, SubprocessCandidateExecutionService,
)
from pde_agent_verifier_solver.engine import candidate_execution as execution
from pde_agent_verifier_solver.plugins.ns import NSSolverTask
from pde_agent_verifier_solver.sdk import EvaluationPhase, EvaluationPlan, TaskKind, write_candidate_bundle_manifest
from tests.test_candidate_execution import _artifact, _case, _candidate_bundle, _evaluation_settings, _query


def _service(root, *, budget=5.0, workers=1, drain=0.6):
    settings = _evaluation_settings(_query())
    evaluation = NSSolverTask().create_evaluation(settings)
    service = SubprocessCandidateExecutionService(
        root, task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),
        evaluation_settings=settings, max_workers=workers,
        case_hard_wall_seconds=budget, oracle_drain_timeout_seconds=drain,
    )
    plan = EvaluationPlan(phase=EvaluationPhase.OFFICIAL, query=_query(), settings=settings)
    return service, evaluation, plan


def test_real_worker_receives_remaining_configured_budget(tmp_path):
    source = _candidate_bundle(tmp_path / 'source')
    entry = source / 'submission.py'
    entry.write_text(entry.read_text().replace(
        '    del solve_budget',
        '    assert solve_budget is not None\n'
        '    assert 0.0 < solve_budget < 5.0\n'
        '    return Handle(solve_budget)',
    ))
    write_candidate_bundle_manifest(source, files=('submission.py',), overwrite=True)
    artifact = PythonCandidateCodec(task_kind=TaskKind.SOLVER).bind_existing(
        source, storage_root=tmp_path / 'sealed', artifact_id='budget-aware',
        name='Budget-aware', summary='Budget propagation probe', usage='prepare then query',
    )
    service, evaluation, plan = _service(tmp_path / 'execution')
    outputs = service.execute_cases(artifact, (_case('budget', value=1.0),), (plan,), evaluation)
    budget_seen = outputs[0].prediction.fields['velocity_x'][0]
    assert 0 < budget_seen < 5
    assert service.audits[0].timeout_seconds == 5


def test_controller_response_decode_is_not_charged_to_candidate(tmp_path, monkeypatch):
    artifact = _artifact(tmp_path)
    service, evaluation, plan = _service(tmp_path / 'execution')
    original = execution._worker_output_from_payload

    def slow_decode(*args, **kwargs):
        time.sleep(0.45)
        return original(*args, **kwargs)

    monkeypatch.setattr(execution, '_worker_output_from_payload', slow_decode)
    service.execute_cases(artifact, (_case('decode', value=1.0),), (plan,), evaluation)
    audit = service.audits[0]
    assert audit.controller_postprocess_wall_seconds >= 0.45
    assert audit.wall_seconds - audit.charged_wall_seconds >= 0.45
    assert audit.worker_wall_seconds == pytest.approx(audit.charged_wall_seconds + audit.oracle_queue_wait_seconds)


def test_timeout_attribution_stops_before_slow_process_cleanup(tmp_path, monkeypatch):
    artifact = _artifact(tmp_path)
    service, evaluation, plan = _service(tmp_path / 'execution', budget=1.5)
    original = execution._terminate_process_group

    def slow_termination(process):
        time.sleep(0.4)
        return original(process)

    monkeypatch.setattr(execution, '_terminate_process_group', slow_termination)
    with pytest.raises(CandidateExecutionTimeout):
        service.execute_cases(artifact, (_case('slow-solver', value=1.0, sleep_seconds=15),), (plan,), evaluation)
    audit = service.audits[0]
    assert 1.5 <= audit.charged_wall_seconds < 1.8
    assert audit.wall_seconds - audit.charged_wall_seconds >= 0.4


def test_public_oracle_original_error_is_retained_privately(tmp_path):
    artifact = _artifact(tmp_path)
    service, evaluation, plan = _service(tmp_path / 'execution')

    def broken_oracle(*_args):
        raise RuntimeError('private-backend-root-cause-20260910')

    with pytest.raises(CandidateExecutionError):
        service.execute_cases(
            artifact, (_case('oracle-error', value=1.0, use_oracle=True),), (plan,), evaluation,
            public_session_methods=('marker',), invoke_public_session=broken_oracle,
            close_public_session=lambda _case_id: None,
        )
    audit = service.audits[0]
    diagnostic = json.loads((Path(audit.workspace) / 'execution_failure.json').read_text())
    failure = diagnostic['oracle_failures'][0]
    assert failure['error_type'] == 'RuntimeError'
    assert failure['method'] == 'marker'
    assert 'private-backend-root-cause-20260910' in failure['traceback']
    # The worker-facing error stays sanitized; the Controller keeps the cause.
    response = json.loads(Path(audit.response_path).read_text())
    assert 'private-backend-root-cause-20260910' not in response['error_message']


def test_failed_candidate_shutdown_allows_idle_thread_scheduling_grace(tmp_path, monkeypatch):
    artifact = _artifact(tmp_path)
    service, evaluation, plan = _service(tmp_path / 'execution')
    original = execution._CandidateOracleState._serve_work

    def delayed_thread_exit(state):
        original(state)
        # Model a completed scientific worker briefly awaiting CPU scheduling
        # during teardown. This is not a hung scientific call.
        time.sleep(0.2)

    monkeypatch.setattr(execution._CandidateOracleState, '_serve_work', delayed_thread_exit)
    kwargs = dict(public_session_methods=('marker',),
                  invoke_public_session=lambda _case, _method, args, _kwargs: args[0],
                  close_public_session=lambda _case: None)
    with pytest.raises(CandidateExecutionError):
        service.execute_cases(
            artifact, (_case('ordinary-bad-solver', value=1.0, use_oracle=True, fail_after_oracle=True),),
            (plan,), evaluation, **kwargs,
        )
    result = service.execute_cases(
        artifact, (_case('next-good-solver', value=2.0, use_oracle=True),), (plan,), evaluation, **kwargs,
    )
    assert result[0].prediction.fields['velocity_x'] == (2.0, 2.0)
    assert not list(service.root.glob('service-failure-*.json'))


def test_finishing_slow_case_does_not_block_other_case_admission(tmp_path, monkeypatch):
    host = execution._CandidateOracleHost(
        tmp_path / 'oracle-arrays', case_ids=('slow', 'other'), methods=(), invoke=None, close=None,
    )
    entered, release = Event(), Event()
    original = host.state.wait_case_drained

    def drain(case_id, timeout):
        if case_id == 'slow':
            entered.set()
            assert release.wait(3)
        return original(case_id, timeout)

    monkeypatch.setattr(host.state, 'wait_case_drained', drain)
    with ThreadPoolExecutor(max_workers=2) as pool:
        finished = pool.submit(host.finish_case, 'slow', drain_timeout_seconds=1.0)
        try:
            assert entered.wait(2)
            admitted = pool.submit(host.manifest_for_case, 'other')
            assert admitted.result(timeout=0.4)['case_id'] == 'other'
        finally:
            release.set()
            finished.result(timeout=2)
            host.close(drain_timeout_seconds=0.5)


def test_concurrent_case_finish_and_host_close_close_session_once(tmp_path):
    entered, release = Event(), Event()
    calls = []

    def close_session(case_id):
        calls.append(case_id)
        entered.set()
        assert release.wait(3)

    host = execution._CandidateOracleHost(
        tmp_path / 'oracle-arrays', case_ids=('one',), methods=(), invoke=None, close=close_session,
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        finish = pool.submit(host.finish_case, 'one', drain_timeout_seconds=1.0)
        assert entered.wait(2)
        close = pool.submit(host.close, drain_timeout_seconds=1.0)
        try:
            time.sleep(0.1)
            assert calls == ['one']
        finally:
            release.set()
            finish.result(timeout=2)
            close.result(timeout=2)
    assert calls == ['one']


def test_worker_helper_thread_oracle_wait_credits_main_task_clock(tmp_path):
    source = _candidate_bundle(tmp_path / 'source')
    entry = source / 'submission.py'
    entry.write_text(entry.read_text().replace(
        '        value = open_public_case_session(case).marker(value)',
        '        from concurrent.futures import ThreadPoolExecutor\n'
        '        session = open_public_case_session(case)\n'
        '        with ThreadPoolExecutor(max_workers=1) as pool:\n'
        '            value = pool.submit(session.marker, value).result()',
    ))
    write_candidate_bundle_manifest(source, files=('submission.py',), overwrite=True)
    artifact = PythonCandidateCodec(task_kind=TaskKind.SOLVER).bind_existing(
        source, storage_root=tmp_path / 'sealed', artifact_id='threaded-oracle',
        name='Threaded public query', summary='Main Task sees Controller refunds', usage='prepare then query',
    )
    service, evaluation, plan = _service(tmp_path / 'execution', workers=2)

    def invoke(_case, _method, args, _kwargs):
        time.sleep(1.0)
        return args[0]

    outputs = service.execute_cases(
        artifact, tuple(_case(f'threaded-{i}', value=i, use_oracle=True) for i in range(2)),
        (plan, plan), evaluation, public_session_methods=('marker',),
        invoke_public_session=invoke, close_public_session=lambda _case: None,
    )
    assert max(audit.oracle_queue_wait_seconds for audit in service.audits) >= 0.6
    assert max(output.metrics['candidate_solve_queue_wait_seconds'] for output in outputs) >= 0.6
    assert all(output.metrics['candidate_solve_wall_seconds'] < 1.6 for output in outputs)


@pytest.mark.parametrize('exception_name', ['CandidateExecutionInfrastructureError', 'CandidateExecutionTimeout'])
def test_worker_preserves_typed_failure_across_process_boundary(tmp_path, exception_name):
    source = _candidate_bundle(tmp_path / 'source')
    entry = source / 'submission.py'
    entry.write_text(entry.read_text().replace(
        '    del solve_budget',
        f'    from pde_agent_verifier_solver.engine.candidate_execution import {exception_name}\n'
        f'    raise {exception_name}("trusted-task-failure-probe")',
    ))
    write_candidate_bundle_manifest(source, files=('submission.py',), overwrite=True)
    artifact = PythonCandidateCodec(task_kind=TaskKind.SOLVER).bind_existing(
        source, storage_root=tmp_path / 'sealed', artifact_id='typed-failure',
        name='Typed failure', summary='Preserve failure classification', usage='prepare then query',
    )
    service, evaluation, plan = _service(tmp_path / 'execution')
    with pytest.raises(getattr(execution, exception_name)):
        service.execute_cases(artifact, (_case('typed-failure', value=1.0),), (plan,), evaluation)


def test_caught_oracle_error_is_retained_when_solver_returns_fallback(tmp_path):
    source = _candidate_bundle(tmp_path / 'source')
    entry = source / 'submission.py'
    entry.write_text(entry.read_text().replace(
        '        value = open_public_case_session(case).marker(value)',
        '        try:\n'
        '            value = open_public_case_session(case).marker(value)\n'
        '        except RuntimeError:\n'
        '            value = 2.0',
    ))
    write_candidate_bundle_manifest(source, files=('submission.py',), overwrite=True)
    artifact = PythonCandidateCodec(task_kind=TaskKind.SOLVER).bind_existing(
        source, storage_root=tmp_path / 'sealed', artifact_id='fallback',
        name='Fallback solver', summary='Preserve hidden backend failure', usage='prepare then query',
    )
    service, evaluation, plan = _service(tmp_path / 'execution')

    def failed_backend(*args):
        raise RuntimeError('private-caught-oracle-root-cause')

    outputs = service.execute_cases(
        artifact, (_case('fallback', value=1.0, use_oracle=True),), (plan,), evaluation,
        public_session_methods=('marker',), invoke_public_session=failed_backend,
        close_public_session=lambda _case: None,
    )
    assert outputs[0].prediction.fields['velocity_x'] == (2.0, 2.0)
    retained = list(service.root.glob('oracle_failures/*.json'))
    assert len(retained) == 1
    assert json.loads(retained[0].read_text())['oracle_failures'][0]['message'] == 'private-caught-oracle-root-cause'


def test_unstarted_peers_are_not_reported_as_individual_timeouts(tmp_path):
    artifact = _artifact(tmp_path)
    service, evaluation, plan = _service(tmp_path / 'execution', budget=1.5)
    with pytest.raises(CandidateExecutionTimeout):
        service.execute_cases(
            artifact, tuple(_case(f'peer-{i}', value=1.0, sleep_seconds=10) for i in range(3)),
            (plan,) * 3, evaluation,
        )
    started = [audit for audit in service.audits if audit.execution_started]
    unstarted = [audit for audit in service.audits if not audit.execution_started]
    assert len(started) == 1 and started[0].error_code == 'candidate_case_timeout'
    assert len(unstarted) == 2
    assert all(audit.error_code == 'candidate_not_started' for audit in unstarted)
    assert all(audit.batch_failure_code == 'candidate_case_timeout' for audit in unstarted)
    assert all(audit.charged_wall_seconds == 0 for audit in unstarted)



def test_timeout_cancels_running_peer_without_infrastructure_failure(tmp_path, monkeypatch):
    from pde_agent_verifier_solver.engine.common_evaluation import _is_infrastructure_failure

    artifact = _artifact(tmp_path)
    service, evaluation, plan = _service(tmp_path / 'execution', budget=2.0, workers=2)
    first_started = Event()
    original_environment = service._worker_environment
    original_popen = execution.subprocess.Popen

    def environment(task_root):
        if task_root.name.startswith('case-000001-'):
            assert first_started.wait(5)
            # Start the peer later so it is running, but not at its own deadline,
            # when the first Case times out. Keep both production clocks intact.
            time.sleep(0.7)
        return original_environment(task_root)

    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        if Path(kwargs['cwd']).name.startswith('case-000000-'):
            first_started.set()
        return process

    with monkeypatch.context() as patch:
        patch.setattr(service, '_worker_environment', environment)
        patch.setattr(execution.subprocess, 'Popen', popen)
        with pytest.raises(CandidateExecutionTimeout) as captured:
            service.execute_cases(
                artifact, (_case('timeout-first', value=1., sleep_seconds=10),
                           _case('cancelled-peer', value=2., sleep_seconds=10)),
                (plan, plan), evaluation,
            )
    error = captured.value
    audits = {audit.case_id: audit for audit in service.audits}
    assert audits['timeout-first'].error_code == 'candidate_case_timeout'
    assert audits['cancelled-peer'].error_code == 'candidate_peer_cancelled'
    assert audits['cancelled-peer'].execution_started
    assert audits['cancelled-peer'].charged_wall_seconds < 2.0
    assert all(audit.process_group_cleaned for audit in audits.values())
    assert not _is_infrastructure_failure(error)
    assert not any('cleanup also failed: _CandidatePeerCancelled' in note
                   for note in getattr(error, '__notes__', ()))
    # A failed solver does not poison an otherwise healthy execution service.
    outputs = service.execute_cases(artifact, (_case('later-good', value=3.),), (plan,), evaluation)
    assert outputs[0].prediction.fields['velocity_x'] == (3., 3.)


def test_controller_request_serialization_does_not_use_candidate_budget(tmp_path, monkeypatch):
    artifact = _artifact(tmp_path)
    service, evaluation, plan = _service(tmp_path / 'execution')
    original = execution._write_private_json

    def slow_request(path, payload, **kwargs):
        if path.name == 'request.json':
            time.sleep(0.4)
        return original(path, payload, **kwargs)

    monkeypatch.setattr(execution, '_write_private_json', slow_request)
    service.execute_cases(artifact, (_case('setup', value=1.0),), (plan,), evaluation)
    audit = service.audits[0]
    assert audit.controller_setup_wall_seconds >= 0.4
    assert audit.wall_seconds - audit.charged_wall_seconds >= 0.4
