"""Real loopback/worker coverage for cancelling one Development evaluation."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Event, Barrier, Lock
import time
from types import SimpleNamespace
from urllib.request import Request, build_opener, ProxyHandler
from urllib.error import HTTPError

import pytest
import numpy as np

from pde_agent_verifier_solver.engine.agent_capability import (
    ControllerAgentCapabilityClient, ControllerAgentCapabilityFacade,
    AgentCapabilityClientError, AGENT_CAPABILITY_RPC_REQUEST_SCHEMA,
)
from pde_agent_verifier_solver.engine.agent_execution import (
    CommandAgentProgram, AGENT_WORKER_REQUEST_SCHEMA, AGENT_REQUEST_ENV, AGENT_WORKSPACE_ENV,
)
from pde_agent_verifier_solver.engine.campaign import EvaluatorPolicy
from pde_agent_verifier_solver.engine.candidate_execution import (
    SubprocessCandidateExecutionService, PythonTaskWorkerSpec,
    CandidateExecutionCancelled, CandidateExecutionInfrastructureError,
)
from pde_agent_verifier_solver.engine.evaluator import AgentEvaluator, EvaluationResponse, _REQUEST_CANCELLATION
from pde_agent_verifier_solver.plugins.ns import NSSolverTask
from pde_agent_verifier_solver.sdk import (
    AnswerView, CaseRecord, CaseEvaluation, DevelopmentCoordinateMode, CaseAccessKind,
    GeneratorCheckpoint, GeneratorState, SampleBatch,
)
from tests.test_candidate_execution import _artifact, _case, _query, _evaluation_settings


def _wait_for(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.01)
    raise AssertionError("expected runtime condition did not occur")


class _Truth:
    def answer(self, query):
        return AnswerView(kind=query.kind, fields={
            'velocity_x': (1.0,) * len(query.points),
            'velocity_y': (-1.0,) * len(query.points),
        }, points=query.points, shape=(len(query.points),))


@pytest.fixture
def setup(tmp_path):
    artifact = _artifact(tmp_path)
    query = _query()
    settings = _evaluation_settings(query)
    service = SubprocessCandidateExecutionService(
        tmp_path / 'execution', task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),
        evaluation_settings=settings, max_workers=2, case_hard_wall_seconds=40,
        oracle_drain_timeout_seconds=0.5,
    )
    handles = []

    def create(name, *, generator=None):
        workspace = tmp_path / name
        workspace.mkdir()
        records = (
            CaseRecord(_case(name + '-slow', value=1.0, sleep_seconds=30.0 if generator is None else 0.0,
                             use_oracle=generator is not None), _Truth()),
            CaseRecord(_case(name + '-fast', value=1.0), _Truth()),
        )
        evaluator = AgentEvaluator(records, NSSolverTask(), EvaluatorPolicy(
            query, development_coordinate_mode=DevelopmentCoordinateMode.USER_FIXED_QUERY,
        ), settings, seed=7, candidate_execution=service, generator=generator,
            generator_state=(None if generator is None else
                             GeneratorState(GeneratorCheckpoint('test-oracle', {}), base={})))

        class Context:
            sampler = None
            public_problem_bundle = None
            @property
            def cases(self):
                return self.evaluator.cases
        context = Context()
        context.evaluator = evaluator
        lease = ControllerAgentCapabilityFacade(drain_timeout_seconds=3).open(
            context, workspace, name,
            CommandAgentProgram((sys.executable, '-c', 'pass'), lambda *args: artifact),
        )
        client = ControllerAgentCapabilityClient(lease.manifest, workspace=workspace, timeout_seconds=40)
        request_file = workspace / 'request.json'
        request_file.write_text(json.dumps({
            'schema': AGENT_WORKER_REQUEST_SCHEMA, 'capabilities': lease.manifest,
            'worker_settings': {'controller_rpc_timeout_seconds': 40},
            'round': {'hard_wall_seconds': 14400}, 'execution_id': name,
        }, default=dict))
        result = SimpleNamespace(workspace=workspace, evaluator=evaluator, lease=lease,
                                 client=client, request_file=request_file,
                                 ids=tuple(case.case_id for case in client.current_cases()))
        handles.append(result)
        return result
    yield SimpleNamespace(create=create, service=service)
    for handle in handles:
        if handle.lease._revoke.__self__.state._cleanup_failed:
            with pytest.raises(Exception, match="cleanup"):
                handle.lease.close()
        else:
            handle.lease.close()
    service._executor.shutdown(wait=True, cancel_futures=True)


def _active(service):
    with service._process_lock:
        return tuple(process for _owner, process in service._processes.values())


def _requests(client):
    return client.current_status()[1]


@pytest.mark.parametrize('stop_signal', [signal.SIGINT, signal.SIGTERM, signal.SIGKILL])
def test_cli_interrupt_stops_real_worker_keeps_status_and_next_request_usable(setup, stop_signal):
    run = setup.create('cli')
    env = {**os.environ, AGENT_REQUEST_ENV: str(run.request_file),
           AGENT_WORKSPACE_ENV: str(run.workspace),
           'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
    process = subprocess.Popen(
        [sys.executable, '-m', 'pde_agent_verifier_solver.engine.agent_tool',
         'evaluate', 'unused-author-bundle', run.ids[0]],
        cwd=run.workspace, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    try:
        worker = _wait_for(lambda: _active(setup.service))[0]
        started = time.monotonic()
        public_status = subprocess.run(
            [sys.executable, '-m', 'pde_agent_verifier_solver.engine.agent_tool', 'status'],
            cwd=run.workspace, env=env, capture_output=True, text=True, timeout=3,
        )
        assert public_status.returncode == 0, public_status.stderr
        rows = json.loads(public_status.stdout)['evaluations']
        assert time.monotonic() - started < 3
        assert rows[0]['state'] == 'running'
        request_id = rows[0]['request_id']
        process.send_signal(stop_signal)
        stdout, stderr = process.communicate(timeout=8)
        assert process.returncode != 0 and not stdout
        terminal = _wait_for(lambda: (row if (row := run.client.evaluation_status(request_id))['finished'] else None))
        assert terminal['state'] == 'cancelled' and terminal['cleanup_confirmed']
        assert worker.poll() is not None and not _active(setup.service)
        audits = setup.service.audits
        assert audits and all(audit.process_group_cleaned for audit in audits)
        assert {audit.error_code for audit in audits} == {'candidate_execution_cancelled'}
        if stop_signal != signal.SIGKILL:
            assert '"state": "cancelled"' in stderr
        # One accepted evaluation consumes its plan namespace even when cancelled.
        assert run.evaluator._next_plan_seed == 8
        assert not run.evaluator._evaluation_cancellation.is_set()
        result = run.client.evaluate({}, case_ids=(run.ids[1],))
        assert result is not None
        assert run.evaluator._next_plan_seed == 9
        assert setup.service.audits[-1].status == 'completed'
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=8)


def test_queued_cancel_and_duplicate_cancel_do_not_stop_owner_or_other_agent(setup):
    run, peer = setup.create('owner'), setup.create('peer')
    with ThreadPoolExecutor(max_workers=3) as executor:
        owner = executor.submit(run.client.evaluate, {}, case_ids=(run.ids[0],))
        _wait_for(lambda: _active(setup.service))
        owner_id = _requests(run.client)[0]['request_id']
        queued = executor.submit(run.client.evaluate, {}, case_ids=(run.ids[1],))
        rows = _wait_for(lambda: rows if len(rows := _requests(run.client)) == 2 else None)
        queued_id = next(row['request_id'] for row in rows if row['state'] == 'queued')
        # A valid other-Agent token cannot inspect or cancel this Agent's request.
        assert peer.client.cancel_evaluation(owner_id, wait_seconds=0)['state'] == 'unknown'
        wrong = ControllerAgentCapabilityClient(
            {**run.lease.manifest, 'token': peer.lease.manifest['token']}, workspace=run.workspace,
        )
        with pytest.raises(AgentCapabilityClientError) as denied:
            wrong.cancel_evaluation(owner_id, wait_seconds=0)
        assert denied.value.code == 'permission_denied'
        started = time.monotonic()
        first = run.client.cancel_evaluation(queued_id, wait_seconds=2)
        assert first['state'] == 'cancelled' and first['cleanup_confirmed']
        assert time.monotonic() - started < 2
        assert run.client.cancel_evaluation(queued_id, wait_seconds=0) == first
        with pytest.raises(AgentCapabilityClientError) as error:
            queued.result(timeout=2)
        assert error.value.code == 'evaluation_cancelled'
        assert not owner.done() and len(_active(setup.service)) == 1
        assert run.evaluator._next_plan_seed == 8  # queued work never allocated a plan
        other = executor.submit(peer.client.evaluate, {}, case_ids=(peer.ids[1],))
        cancelled = run.client.cancel_evaluation(owner_id, wait_seconds=3)
        assert cancelled['state'] == 'cancelled' and cancelled['cleanup_confirmed']
        with pytest.raises(AgentCapabilityClientError):
            owner.result(timeout=2)
        assert other.result(timeout=8) is not None
        assert peer.evaluator._next_plan_seed == 8
    assert run.client.evaluate({}, case_ids=(run.ids[1],)) is not None


def test_finished_request_wins_late_cancel_and_duplicate_rpc_does_not_execute(setup):
    run = setup.create('finished')
    assert run.client.evaluate({}, case_ids=(run.ids[1],)) is not None
    row = _requests(run.client)[0]
    assert row['state'] == 'completed'
    assert run.client.cancel_evaluation(row['request_id'], wait_seconds=0) == row
    # Replaying an already used ID must not dispatch another scientific operation.
    body = {'schema': AGENT_CAPABILITY_RPC_REQUEST_SCHEMA, 'request_id': row['request_id'],
            'operation': 'evaluate', 'params': {}}
    request = Request(run.lease.manifest['endpoint'], data=json.dumps(body).encode(),
                      headers={'Content-Type': 'application/json',
                               'X-PDE-Agent-Token': run.lease.manifest['token']})
    with pytest.raises(HTTPError):
        build_opener(ProxyHandler({})).open(request, timeout=3)
    assert len(setup.service.audits) == 1 and run.evaluator._next_plan_seed == 8


def test_whole_turn_revoke_still_cancels_and_drains_worker(setup):
    run = setup.create('revoke')
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(run.client.evaluate, {}, case_ids=(run.ids[0],))
        worker = _wait_for(lambda: _active(setup.service))[0]
        run.lease.close()
        with pytest.raises(AgentCapabilityClientError):
            pending.result(timeout=3)
    assert run.lease.drained and run.evaluator._evaluation_cancellation.is_set()
    assert worker.poll() is not None
    assert setup.service.audits[0].process_group_cleaned


def test_cancel_waits_for_controller_cleanup_and_reports_real_cleanup_failure(setup):
    run = setup.create('cleanup')
    entered, release = Event(), Event()
    def delayed_failure(*args):
        entered.set()
        event = _REQUEST_CANCELLATION.get()
        assert event is not None
        assert release.wait(5)
        error = CandidateExecutionCancelled('request was cancelled')
        error._candidate_cleanup_errors = (CandidateExecutionInfrastructureError('oracle did not drain'),)
        raise error
    run.evaluator.evaluate_many = delayed_failure
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(run.client.evaluate, {}, case_ids=(run.ids[0],))
        assert entered.wait(2)
        request_id = _requests(run.client)[0]['request_id']
        state = run.client.cancel_evaluation(request_id, wait_seconds=0)
        assert state['state'] == 'cancelling'
        assert not state['finished'] and not state['cleanup_confirmed']
        release.set()
        with pytest.raises(AgentCapabilityClientError) as error:
            pending.result(timeout=3)
        assert error.value.code == 'evaluation_cleanup_failed'
        state = run.client.evaluation_status(request_id)
        assert state['finished'] and state['state'] == 'cleanup_failed'
        assert not state['cleanup_confirmed']
        with pytest.raises(AgentCapabilityClientError):
            run.client.evaluate({}, case_ids=(run.ids[1],))
        assert run.client.current_cases()  # diagnostics stay available
        assert not setup.service.audits  # quarantined request never dispatched


def test_status_large_array_encoding_can_overlap_sample_registration(setup, monkeypatch):
    run = setup.create('arrays')
    array = np.arange(128 * 1024, dtype=np.float64)
    array.flags.writeable = False
    run.evaluator.register((CaseRecord(_case('array-original', value=1.0, array=array), _Truth()),))
    host = run.lease._revoke.__self__
    registered = Event()
    def sample(request):
        run.evaluator.register((CaseRecord(_case('array-new', value=1.0, array=array), _Truth()),))
        registered.set()
        return SampleBatch(cases=(run.evaluator.cases[-1],), remaining=3)
    host.state.context.sampler = SimpleNamespace(sample_cases=sample)
    sampler_client = ControllerAgentCapabilityClient(
        {**run.lease.manifest, 'sampling': {'available': True}}, workspace=run.workspace,
    )
    # Force status and sample to publish the identical large sidecar concurrently.
    # If status takes the scientific lock, this barrier times out.
    barrier, lock, count = Barrier(2, timeout=3), Lock(), 0
    original_publish = host.codec._publish_array
    def publish(value):
        nonlocal count
        with lock:
            count += 1
            synchronize = count <= 2
        if synchronize:
            barrier.wait()
        return original_publish(value)
    monkeypatch.setattr(host.codec, '_publish_array', publish)
    with ThreadPoolExecutor(max_workers=2) as executor:
        status = executor.submit(run.client.current_status)
        sampled = executor.submit(sampler_client.sample_base, 1)
        cases, _ = status.result(timeout=5)
        batch = sampled.result(timeout=5)
    assert registered.is_set() and len({case.case_id for case in cases}) == len(cases)
    for case in (*cases, *batch.cases):
        if 'array' in case.public_data:
            np.testing.assert_array_equal(case.public_data['array'], array)
    assert len(run.client.current_cases()) == 4
    assert not tuple((run.workspace / 'capability-arrays').glob('*.partial'))


def test_http_cancel_reports_actual_oracle_drain_failure_and_quarantines_service(setup):
    entered, release, closed = Event(), Event(), Event()
    class Session:
        def marker(self, value):
            entered.set()
            if not release.wait(10):
                raise RuntimeError('test oracle was not released')
            return value
        def close(self):
            closed.set()
    class Generator:
        def case_session_methods(self, access):
            return ('marker',) if access is CaseAccessKind.PUBLIC_PROBLEM else ()
        def open_case_session(self, state, case, access):
            return Session()
    run = setup.create('oracle-drain', generator=Generator())
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(run.client.evaluate, {}, case_ids=(run.ids[0],))
            assert entered.wait(5)
            worker = _active(setup.service)[0]
            request_id = _requests(run.client)[0]['request_id']
            terminal = run.client.cancel_evaluation(request_id, wait_seconds=3)
            assert terminal['state'] == 'cleanup_failed' and terminal['finished']
            assert not terminal['cleanup_confirmed']
            with pytest.raises(AgentCapabilityClientError) as failed:
                pending.result(timeout=3)
            assert failed.value.code == 'evaluation_cleanup_failed'
            assert worker.poll() is not None
            assert setup.service._unhealthy_reason is not None
            with pytest.raises(AgentCapabilityClientError):
                run.client.evaluate({}, case_ids=(run.ids[1],))
            assert run.client.current_cases()
    finally:
        release.set()
        assert closed.wait(5)


def test_cancel_stops_remaining_truth_work_without_refunding_accepted_plan_seeds(setup):
    run = setup.create('truth')
    entered, release = Event(), Event()
    calls = []
    class SlowTruth:
        def answer(self, query):
            calls.append('first')
            entered.set()
            assert release.wait(3)
            return _Truth().answer(query)
    class UnwantedTruth:
        def answer(self, query):
            calls.append('second')
            return _Truth().answer(query)
    run.evaluator.register((
        CaseRecord(_case('truth-first', value=1.0), SlowTruth()),
        CaseRecord(_case('truth-second', value=1.0), UnwantedTruth()),
    ))
    ids = tuple(case.case_id for case in run.client.current_cases())[-2:]
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(run.client.evaluate, {}, case_ids=ids)
        assert entered.wait(2)
        request_id = _requests(run.client)[0]['request_id']
        state = run.client.cancel_evaluation(request_id, wait_seconds=0)
        assert not state['finished'] and state['state'] == 'cancelling'
        release.set()
        with pytest.raises(AgentCapabilityClientError) as error:
            pending.result(timeout=3)
        assert error.value.code == 'evaluation_cancelled'
    assert calls == ['first'] and not setup.service.audits
    assert run.evaluator._next_plan_seed == 9
    assert run.client.evaluate({}, case_ids=(run.ids[1],)) is not None


def test_http_cancel_racing_completion_has_one_stable_terminal_state(setup):
    run = setup.create('race')
    for _ in range(6):
        entered, release = Event(), Event()
        def finishing(*args):
            entered.set()
            assert release.wait(3)
            return (EvaluationResponse('race-result', 0.0, {}, ()),)
        run.evaluator.evaluate_many = finishing
        with ThreadPoolExecutor(max_workers=2) as executor:
            evaluation = executor.submit(run.client.evaluate, {}, case_ids=(run.ids[1],))
            assert entered.wait(2)
            request_id = _requests(run.client)[-1]['request_id']
            cancelled = executor.submit(run.client.cancel_evaluation, request_id, wait_seconds=2)
            release.set()
            terminal = cancelled.result(timeout=3)
            assert terminal['state'] in {'cancelled', 'completed'}
            assert terminal['finished'] and terminal['cleanup_confirmed']
            if terminal['state'] == 'cancelled':
                with pytest.raises(AgentCapabilityClientError) as failure:
                    evaluation.result(timeout=2)
                assert failure.value.code == 'evaluation_cancelled'
            else:
                assert evaluation.result(timeout=2) is not None
        assert run.client.cancel_evaluation(request_id, wait_seconds=0) == terminal


def test_inprocess_cancellation_finishes_current_function_but_does_not_start_next_case(setup, monkeypatch):
    run = setup.create('inprocess')
    # This is the default AgentEvaluator route for local tasks without a
    # CandidateExecutionService, distinct from the subprocess worker path.
    run.evaluator._candidate_execution = None
    entered, release = Event(), Event()
    executed = []
    def evaluate(artifact, case, truth, plan):
        executed.append(case.case_id)
        if len(executed) == 1:
            entered.set()
            assert release.wait(3)
        return CaseEvaluation(case.case_id, 0.0, {'velocity_relative_l2': 0.0}, truth)
    monkeypatch.setattr(run.evaluator.evaluation_program, 'evaluate', evaluate)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(run.client.evaluate, {}, case_ids=run.ids)
        assert entered.wait(2)
        request_id = _requests(run.client)[0]['request_id']
        state = run.client.cancel_evaluation(request_id, wait_seconds=0)
        assert state['state'] == 'cancelling' and not state['finished']
        assert executed == [run.ids[0]]
        release.set()
        with pytest.raises(AgentCapabilityClientError) as error:
            pending.result(timeout=3)
        assert error.value.code == 'evaluation_cancelled'
    assert executed == [run.ids[0]]  # cancelled batch never entered its second Case
    assert run.client.evaluation_status(request_id)['cleanup_confirmed']
    assert run.evaluator._next_plan_seed == 9 and not setup.service.audits
    assert run.client.evaluate({}, case_ids=(run.ids[1],)) is not None
    assert executed == list(run.ids)
    assert run.evaluator._next_plan_seed == 10
