"""Real worker regressions for Controller-owned public Oracle queue accounting."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
import time

import pytest

from pde_agent_verifier_solver.engine import (
    CandidateExecutionCancelled,
    CandidateExecutionError,
    CandidateExecutionTimeout,
    PythonTaskWorkerSpec,
    SubprocessCandidateExecutionService,
)
from pde_agent_verifier_solver.engine.candidate_execution import (
    CandidatePublicSessionClient,
    _CandidateOracleHost,
)
from pde_agent_verifier_solver.engine.evaluator import _PersistentPublicCaseSession
from pde_agent_verifier_solver.plugins.ns import NSSolverTask
from pde_agent_verifier_solver.sdk import EvaluationPhase, EvaluationPlan
from tests.test_candidate_execution import (
    _artifact,
    _case,
    _evaluation_settings,
    _query,
)
from tests.test_canonical_candidate_boundary import _generator_state


def _execution(tmp_path: Path, *, workers: int, budget: float = 3.0):
    artifact = _artifact(tmp_path)
    query = _query()
    settings = _evaluation_settings(query)
    evaluation = NSSolverTask().create_evaluation(settings)
    service = SubprocessCandidateExecutionService(
        tmp_path / "execution",
        task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),
        evaluation_settings=settings,
        max_workers=workers,
        case_hard_wall_seconds=budget,
        oracle_drain_timeout_seconds=0.1,
    )
    plan = EvaluationPlan(
        phase=EvaluationPhase.OFFICIAL,
        query=query,
        settings=settings,
    )
    return artifact, evaluation, service, plan


def _persistent_session(case, operation_lock, *, opened: Event | None = None):
    class PublicSession:
        def marker(self, value):
            return float(value) + 1.0

        def close(self):
            return None

    def open_session(_state, _case, _access):
        if opened is not None:
            opened.set()
        return PublicSession()

    return _PersistentPublicCaseSession(
        opener=open_session,
        generator_state=_generator_state(),
        controller_case=case,
        candidate_case=case,
        methods=("marker",),
        operation_lock=operation_lock,
    )


def test_parallel_workers_do_not_charge_other_cases_oracle_execution(tmp_path):
    artifact, evaluation, service, plan = _execution(tmp_path, workers=3)
    cases = tuple(
        _case(f"queued-oracle-{index}", value=index, use_oracle=True)
        for index in range(3)
    )
    calls = []

    def invoke(case_id, method, args, kwargs):
        assert method == "marker"
        assert not kwargs
        calls.append(case_id)
        time.sleep(1.4)
        return float(args[0]) + 4.0

    outputs = service.execute_cases(
        artifact,
        cases,
        (plan,) * len(cases),
        evaluation,
        public_session_methods=("marker",),
        invoke_public_session=invoke,
        close_public_session=lambda _case_id: None,
    )

    assert [output.prediction.fields["velocity_x"][0] for output in outputs] == [
        4.0,
        5.0,
        6.0,
    ]
    assert sorted(calls) == sorted(case.case_id for case in cases)
    assert all(audit.status == "completed" for audit in service.audits)
    assert max(audit.wall_seconds for audit in service.audits) > 3.0
    assert max(audit.oracle_queue_wait_seconds for audit in service.audits) > 2.0
    assert all(audit.charged_wall_seconds < 3.0 for audit in service.audits)
    # The SDK's solve timing must use the same queue exclusion, even though
    # the child spends much longer blocked inside its public-session call.
    assert all(
        output.metrics["candidate_solve_wall_seconds"] < 2.8
        for output in outputs
    )


def test_own_oracle_execution_still_consumes_case_budget(tmp_path):
    artifact, evaluation, service, plan = _execution(tmp_path, workers=1)
    case = _case("own-slow-oracle", value=1.0, use_oracle=True)
    entered = Event()
    release = Event()

    def invoke(_case_id, _method, args, _kwargs):
        entered.set()
        assert release.wait(8.0), "test Oracle was not released"
        return float(args[0])

    started = time.monotonic()
    try:
        with pytest.raises(CandidateExecutionTimeout):
            service.execute_cases(
                artifact,
                (case,),
                (plan,),
                evaluation,
                public_session_methods=("marker",),
                invoke_public_session=invoke,
                close_public_session=lambda _case_id: None,
            )
    finally:
        release.set()

    assert entered.is_set(), "the timeout must exercise actual Oracle work"
    assert time.monotonic() - started < 5.0
    audit = service.audits[0]
    assert audit.error_code == "candidate_case_timeout"
    assert audit.process_group_cleaned
    assert audit.charged_wall_seconds >= 3.0
    assert audit.oracle_queue_wait_seconds < 0.5


def test_shared_generator_operation_lock_wait_is_excluded_live(tmp_path):
    artifact, evaluation, service, plan = _execution(tmp_path, workers=1)
    case = _case("external-generator-lock", value=2.0, use_oracle=True)
    operation_lock = Lock()
    operation_lock.acquire()
    session = _persistent_session(case, operation_lock)
    entered = Event()

    def invoke(_case_id, method, args, kwargs):
        entered.set()
        return session.invoke(method, args, kwargs)

    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(
            service.execute_cases,
            artifact,
            (case,),
            (plan,),
            evaluation,
            public_session_methods=("marker",),
            invoke_public_session=invoke,
            close_public_session=lambda _case_id: session.close(),
        )
        try:
            assert entered.wait(6.0)
            # Waiting itself exceeds the entire case budget. Refunding time
            # only after the RPC returns cannot make this test pass.
            time.sleep(3.2)
        finally:
            operation_lock.release()
        outputs = running.result(timeout=5.0)

    assert outputs[0].prediction.fields["velocity_x"] == (3.0, 3.0)
    audit = service.audits[0]
    assert audit.status == "completed"
    assert audit.wall_seconds > 3.2
    assert audit.oracle_queue_wait_seconds >= 3.0
    assert audit.charged_wall_seconds < 3.0
    assert outputs[0].metrics["candidate_solve_wall_seconds"] < 1.0


def test_cancellation_while_waiting_on_generator_lock_stays_bounded(tmp_path):
    artifact, evaluation, service, plan = _execution(
        tmp_path, workers=1, budget=10.0
    )
    case = _case("cancelled-generator-lock", value=2.0, use_oracle=True)
    operation_lock = Lock()
    operation_lock.acquire()
    opened = Event()
    session = _persistent_session(case, operation_lock, opened=opened)
    entered = Event()
    cancellation = Event()

    def invoke(_case_id, method, args, kwargs):
        entered.set()
        return session.invoke(method, args, kwargs)

    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(
            service.execute_cases,
            artifact,
            (case,),
            (plan,),
            evaluation,
            public_session_methods=("marker",),
            invoke_public_session=invoke,
            close_public_session=lambda _case_id: session.close(),
            cancellation_event=cancellation,
        )
        try:
            assert entered.wait(6.0)
            cancelled_at = time.monotonic()
            cancellation.set()
            with pytest.raises(CandidateExecutionCancelled):
                running.result(timeout=3.0)
            assert time.monotonic() - cancelled_at < 2.0
        finally:
            operation_lock.release()

    # Revocation must also prevent a previously queued invocation from opening
    # a plugin session after its resource eventually becomes available.
    assert not opened.wait(0.2)
    audit = service.audits[0]
    assert audit.error_code == "candidate_execution_cancelled"
    assert audit.process_group_cleaned


def test_failed_rpc_returns_trusted_queue_time_to_catching_client(tmp_path):
    owner_entered = Event()
    release_owner = Event()
    failed_work_seconds = []

    def invoke(case_id, _method, args, _kwargs):
        if case_id == "queue-owner":
            owner_entered.set()
            assert release_owner.wait(3.0), "test Oracle owner was not released"
            return float(args[0])
        started = time.monotonic()
        # This is actual work in the failing Oracle, so it must remain charged.
        Event().wait(0.08)
        failed_work_seconds.append(time.monotonic() - started)
        raise ValueError("intentional scientific Oracle failure")

    host = _CandidateOracleHost(
        tmp_path / "oracle-arrays",
        case_ids=("queue-owner", "queued-failure"),
        methods=("marker",),
        invoke=invoke,
        close=lambda _case_id: None,
    )
    host.start()
    owner_client = CandidatePublicSessionClient(host.manifest_for_case("queue-owner"))
    failed_client = CandidatePublicSessionClient(host.manifest_for_case("queued-failure"))

    def catch_failure():
        started = time.monotonic()
        with pytest.raises(CandidateExecutionError, match="public-oracle call failed"):
            failed_client.invoke("queued-failure", "marker", (2.0,), {})
        return time.monotonic() - started

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            owner = executor.submit(owner_client.invoke, "queue-owner", "marker", (1.0,), {})
            try:
                assert owner_entered.wait(2.0)
                failed = executor.submit(catch_failure)
                deadline = time.monotonic() + 2.0
                while host.state.queue_wait_seconds("queued-failure") < 0.05:
                    assert time.monotonic() < deadline, "second RPC was not admitted to the queue"
                    Event().wait(0.005)
                observed_wait = host.state.queue_wait_seconds("queued-failure")
            finally:
                release_owner.set()
            assert owner.result(timeout=2.0) == 1.0
            elapsed = failed.result(timeout=2.0)

        assert len(failed_work_seconds) == 1
        assert failed_client.queue_wait_seconds >= observed_wait
        assert elapsed - failed_client.queue_wait_seconds >= failed_work_seconds[0] * 0.9
        # A completed error must release the per-Case pending-request slot.
        assert "queued-failure" not in host.state._request_by_case
    finally:
        release_owner.set()
        host.close()
