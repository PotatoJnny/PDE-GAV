"""The solve clock excludes trusted platform waits, never numerical service."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Event, Lock
import time
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.engine.candidate_execution import (
    CANDIDATE_WORKER_REQUEST_SCHEMA,
    CandidateExecutionError,
    CandidateExecutionInfrastructureError,
    CandidatePublicSessionClient,
    _CandidateOracleHost,
    _LocalPublicSession,
)
from pde_agent_verifier_solver.engine import candidate_execution
from pde_agent_verifier_solver.engine.python_candidate import PythonCandidateProgram
from pde_agent_verifier_solver.engine.evaluator import _PersistentPublicCaseSession
from pde_agent_verifier_solver.sdk import Case, CaseSource, CaseVisibility
from pde_agent_verifier_solver.sdk import solve_budget
from pde_agent_verifier_solver import sdk
from pde_agent_verifier_solver.sdk.solve_budget import (
    ORACLE_WAIT_TIMING_SCHEMA,
    CandidateSolveBudgetMeter,
    _record_oracle_lock_wait,
    bind_candidate_solve_budget,
    candidate_solve_clock,
    candidate_solve_timing,
)


def _timing(queue=0.0, lock=0.0):
    return {
        "schema": ORACLE_WAIT_TIMING_SCHEMA,
        "queue_wait_seconds": queue,
        "lock_wait_seconds": lock,
    }


def test_clock_debits_only_wait_and_nested_tasks_reuse_meter(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(solve_budget, "perf_counter", lambda: now[0])
    with bind_candidate_solve_budget(new_case=True) as meter:
        start = candidate_solve_clock()
        now[0] += 10.0
        meter.record_oracle_response("rpc-1", _timing(3.0, 2.0), rpc_wall_seconds=8.0)
        assert candidate_solve_clock() - start == 5.0
        with bind_candidate_solve_budget() as nested:
            assert nested is meter
            assert candidate_solve_timing()["candidate_platform_wait_seconds"] == 5.0
        with bind_candidate_solve_budget(new_case=True) as other:
            assert other is not meter
            assert candidate_solve_timing()["candidate_platform_wait_seconds"] == 0.0
        assert candidate_solve_timing()["candidate_platform_wait_seconds"] == 5.0
        # Duplicate response credit would erase genuine candidate work.
        with pytest.raises(ValueError, match="already recorded"):
            meter.record_oracle_response("rpc-1", _timing(3.0, 2.0), rpc_wall_seconds=8.0)
        assert candidate_solve_clock() - start == 5.0
    assert candidate_solve_clock() == 110.0


@pytest.mark.parametrize("timing", [
    _timing(-1.0), _timing(float("nan")), _timing(float("inf")),
    _timing(True), _timing("1.0"), _timing(0.8, 0.8),
    {**_timing(), "unexpected": 0.0}, {**_timing(), "schema": "old"},
    {"schema": ORACLE_WAIT_TIMING_SCHEMA},
])
def test_invalid_or_excess_wait_never_changes_ledger(timing):
    meter = CandidateSolveBudgetMeter()
    with pytest.raises(ValueError):
        meter.record_oracle_response("bad", timing, rpc_wall_seconds=1.0)
    assert meter.snapshot()["candidate_platform_wait_seconds"] == 0.0


def _measured_call(client, case_id):
    with bind_candidate_solve_budget(new_case=True):
        wall = time.perf_counter()
        active = candidate_solve_clock()
        try:
            value = client.invoke(case_id, "marker", (), {})
            failed = False
        except CandidateExecutionError:
            value = None
            failed = True
        return {
            "wall": time.perf_counter() - wall,
            "active": candidate_solve_clock() - active,
            "value": value,
            "failed": failed,
            **candidate_solve_timing(),
        }


def test_http_queue_wait_is_per_case_and_own_oracle_compute_stays_charged(tmp_path: Path):
    entered, release = Event(), Event()

    def invoke(case_id, method, args, kwargs):
        assert method == "marker" and args == () and kwargs == {}
        if case_id == "first":
            entered.set()
            assert release.wait(3.0)
        else:
            time.sleep(0.04)  # Attributable numerical service, not platform wait.
        return case_id

    host = _CandidateOracleHost(
        tmp_path / "arrays", case_ids=("first", "second"), methods=("marker",),
        invoke=invoke, close=lambda case_id: None,
    )
    host.start()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(_measured_call, CandidatePublicSessionClient(host.manifest_for_case("first")), "first")
            assert entered.wait(2.0)
            second = pool.submit(_measured_call, CandidatePublicSessionClient(host.manifest_for_case("second")), "second")
            deadline = time.monotonic() + 2.0
            while host.state.queue_wait_seconds("second") < 0.001 and time.monotonic() < deadline:
                time.sleep(0.002)
            assert host.state.queue_wait_seconds("second") >= 0.001
            time.sleep(0.12)
            release.set()
            a, b = first.result(timeout=3.0), second.result(timeout=3.0)
        assert a["value"] == "first" and b["value"] == "second"
        assert a["active"] >= 0.12  # Its own service is not made free.
        assert b["candidate_oracle_queue_wait_seconds"] >= 0.10
        assert b["candidate_oracle_lock_wait_seconds"] == 0.0
        assert b["active"] >= 0.035
        assert b["wall"] - b["active"] == pytest.approx(b["candidate_platform_wait_seconds"], abs=0.002)
        assert a["candidate_oracle_timed_requests"] == b["candidate_oracle_timed_requests"] == 1.0
    finally:
        release.set()
        host.close()


class _ObservedLock:
    def __init__(self):
        self.lock = Lock()
        self.waiting = Event()

    def acquire(self, *args, **kwargs):
        self.waiting.set()
        return self.lock.acquire(*args, **kwargs)

    def release(self):
        self.lock.release()


@pytest.mark.parametrize("fail", [False, True])
def test_real_plugin_lock_wait_is_credited_even_on_rpc_error(tmp_path: Path, fail):
    operation_lock = _ObservedLock()
    operation_lock.lock.acquire()
    case = Case("one", CaseSource.GENERATED, CaseVisibility.DEVELOPMENT, {})

    class Session:
        def marker(self):
            time.sleep(0.03)
            if fail:
                raise ValueError("intentional numerical failure")
            return 7.0

    session = _PersistentPublicCaseSession(
        opener=lambda *args: Session(), generator_state=object(),
        controller_case=case, candidate_case=case, methods=("marker",),
        operation_lock=operation_lock,
    )
    host = _CandidateOracleHost(
        tmp_path / "arrays", case_ids=("one",), methods=("marker",),
        invoke=lambda case_id, method, args, kwargs: session.invoke(method, args, kwargs),
        close=lambda case_id: session.close(),
    )
    host.start()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_measured_call, CandidatePublicSessionClient(host.manifest_for_case("one")), "one")
            assert operation_lock.waiting.wait(2.0)
            time.sleep(0.12)
            operation_lock.release()
            result = future.result(timeout=3.0)
        assert result["failed"] is fail
        assert result["value"] == (None if fail else 7.0)
        assert result["candidate_oracle_lock_wait_seconds"] >= 0.10
        assert result["candidate_oracle_timed_requests"] == 1.0
        assert result["active"] >= 0.025
        assert result["wall"] - result["active"] == pytest.approx(result["candidate_platform_wait_seconds"], abs=0.002)
    finally:
        if operation_lock.lock.locked():
            operation_lock.release()
        host.close()


@pytest.mark.parametrize("fail", [False, True])
def test_local_public_session_accounts_lock_wait_and_preserves_exception(fail):
    case = Case("one", CaseSource.GENERATED, CaseVisibility.DEVELOPMENT, {})

    def invoke(*args):
        time.sleep(0.02)
        _record_oracle_lock_wait(0.01)
        if fail:
            raise ValueError("numerical failure")
        return 4.0

    with bind_candidate_solve_budget():
        local = _LocalPublicSession(case, ("marker",), invoke)
        if fail:
            with pytest.raises(ValueError, match="numerical failure"):
                local.marker()
        else:
            assert local.marker() == 4.0
        assert candidate_solve_timing()["candidate_oracle_lock_wait_seconds"] == 0.01
        assert candidate_solve_timing()["candidate_oracle_timed_requests"] == 1.0


def test_old_oracle_manifest_cannot_silently_disable_wait_contract(tmp_path: Path):
    host = _CandidateOracleHost(
        tmp_path / "arrays", case_ids=("one",), methods=("marker",),
        invoke=lambda *args: 1.0, close=lambda case_id: None,
    )
    try:
        manifest = dict(host.manifest_for_case("one"))
        manifest.pop("wait_timing_schema")
        with pytest.raises(CandidateExecutionInfrastructureError, match="timing contract"):
            CandidatePublicSessionClient(manifest)
    finally:
        host.close()


def test_failed_candidate_worker_retains_timing_after_meter_scope_exits(tmp_path, monkeypatch):
    case = Case("one", CaseSource.GENERATED, CaseVisibility.DEVELOPMENT, {})
    host = _CandidateOracleHost(
        tmp_path / "oracle-arrays", case_ids=("one",), methods=("marker",),
        invoke=lambda *args: 1.0, close=lambda case_id: None,
    )

    def execute(artifact, bound_case, plan):
        assert sdk.open_public_case_session(bound_case).marker() == 1.0
        raise ValueError("intentional candidate failure after oracle")

    evaluation = SimpleNamespace(
        program_id="test", supports_split_candidate_execution=True,
        execute_candidate=execute,
    )
    payload = {
        "schema": CANDIDATE_WORKER_REQUEST_SCHEMA, "execution_id": "test",
        "case_index": 0, "sidecar_root": str(tmp_path / "worker-arrays"),
        "evaluation_program_id": "test", "public_oracle": host.manifest_for_case("one"),
    }
    monkeypatch.setattr(candidate_execution, "_read_bounded_json", lambda *args: payload)
    monkeypatch.setattr(candidate_execution, "_task_from_payload", lambda *args: object())
    monkeypatch.setattr(candidate_execution, "_settings_from_payload", lambda *args: {})
    monkeypatch.setattr(candidate_execution, "_artifact_from_payload", lambda *args: object())
    monkeypatch.setattr(candidate_execution, "_case_from_payload", lambda *args: case)
    monkeypatch.setattr(candidate_execution, "_plan_from_payload", lambda *args: object())
    monkeypatch.setattr(sdk, "resolve_evaluation_program", lambda *args: evaluation)
    response_path = tmp_path / "response.json"
    host.start()
    try:
        assert candidate_execution.run_candidate_worker(tmp_path / "request.json", response_path) == 1
        response = json.loads(response_path.read_text())
        assert response["error_type"] == "ValueError"
        assert response["solve_budget_timing"]["candidate_oracle_timed_requests"] == 1.0
        assert response["solve_budget_timing"]["candidate_oracle_queue_wait_seconds"] >= 0.0
        assert candidate_solve_timing()["candidate_oracle_timed_requests"] == 0.0
    finally:
        host.close()


def test_legacy_stopwatch_scope_nests_and_restores_all_aliases_on_exceptions(tmp_path):
    # Isolate the runtime compatibility hook: no Candidate source mutation or
    # unrelated module imports are needed to exercise its scope restoration.
    program = PythonCandidateProgram(tmp_path / "unused.json", expected_tree_sha256="0" * 64)
    original = time.perf_counter
    namespace = SimpleNamespace(perf_counter=original, renamed_clock=original, time=time)
    program._loaded = namespace
    with pytest.raises(ValueError, match="outer failure"):
        with program._solve_budget_clock_scope() as outer_aliases:
            assert set(outer_aliases) == {"perf_counter", "renamed_clock"}
            assert namespace.perf_counter is candidate_solve_clock
            assert namespace.renamed_clock is candidate_solve_clock
            # A normal nested scope must not restore its parent's binding.
            with pytest.raises(RuntimeError, match="inner failure"):
                with program._solve_budget_clock_scope() as inner_aliases:
                    assert inner_aliases == ()
                    raise RuntimeError("inner failure")
            assert namespace.perf_counter is candidate_solve_clock
            # An alias created after entry belongs to the inner scope only.
            namespace.late_alias = original
            with program._solve_budget_clock_scope() as late_aliases:
                assert late_aliases == ("late_alias",)
                assert namespace.late_alias is candidate_solve_clock
            assert namespace.late_alias is original
            assert time.perf_counter is original
            raise ValueError("outer failure")
    assert namespace.perf_counter is original
    assert namespace.renamed_clock is original
    assert namespace.time is time and time.perf_counter is original
    # The RLock and scope remain usable after both exceptional exits.
    with program._solve_budget_clock_scope() as again:
        assert set(again) == {"perf_counter", "renamed_clock", "late_alias"}


def test_remaining_budget_includes_worker_startup_and_refunds_only_platform_wait(monkeypatch):
    now = [108.0]
    monkeypatch.setattr(solve_budget, "monotonic", lambda: now[0])
    assert solve_budget.candidate_solve_remaining_seconds() is None
    with bind_candidate_solve_budget(
        new_case=True, budget_seconds=900.0, started_at_monotonic=100.0,
    ) as meter:
        # Eight seconds already consumed before the Task is ready to prepare.
        assert solve_budget.candidate_solve_remaining_seconds() == 892.0
        now[0] += 10.0
        meter.record_oracle_response("rpc-budget", _timing(3.0, 2.0), rpc_wall_seconds=8.0)
        assert solve_budget.candidate_solve_remaining_seconds() == 887.0
        with bind_candidate_solve_budget(budget_seconds=5000, started_at_monotonic=now[0]):
            assert solve_budget.candidate_solve_remaining_seconds() == 887.0
        with bind_candidate_solve_budget(new_case=True):
            assert solve_budget.candidate_solve_remaining_seconds() is None
        assert solve_budget.candidate_solve_remaining_seconds() == 887.0
        now[0] += 1000.0
        assert solve_budget.candidate_solve_remaining_seconds() == 0.0
    assert solve_budget.candidate_solve_remaining_seconds() is None


@pytest.mark.parametrize("kwargs", (
    {"budget_seconds": 0}, {"budget_seconds": -1}, {"budget_seconds": True},
    {"budget_seconds": float("nan")}, {"budget_seconds": float("inf")},
    {"budget_seconds": "900"}, {"started_at_monotonic": 100},
    {"budget_seconds": 1, "started_at_monotonic": -1},
))
def test_remaining_budget_rejects_invalid_execution_limits(kwargs):
    with pytest.raises(ValueError):
        with bind_candidate_solve_budget(new_case=True, **kwargs):
            raise AssertionError("invalid limits were accepted")


def test_captured_execution_meter_credits_serialized_rpc_received_on_candidate_thread(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(solve_budget, "perf_counter", lambda: now[0])
    monkeypatch.setattr(solve_budget, "monotonic", lambda: now[0])
    with bind_candidate_solve_budget(new_case=True, budget_seconds=900.0):
        meter = solve_budget._current_candidate_solve_budget_meter()
        began = candidate_solve_clock()
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(solve_budget._current_candidate_solve_budget_meter).result() is None
            # The worker's client captured the execution meter before dispatch;
            # these successful serialized RPCs do not overlap or double-credit.
            now[0] = 106.0
            pool.submit(
                solve_budget._record_oracle_response_wait,
                "thread-rpc-1", _timing(2.0, 1.0),
                rpc_wall_seconds=5.0, meter=meter,
            ).result()
            assert candidate_solve_clock() - began == 3.0
            assert solve_budget.candidate_solve_remaining_seconds() == 897.0
            now[0] = 110.0
            pool.submit(
                solve_budget._record_oracle_response_wait,
                "thread-rpc-2", _timing(1.0, 0.5),
                rpc_wall_seconds=3.0, meter=meter,
            ).result()
        assert candidate_solve_clock() - began == 5.5
        assert solve_budget.candidate_solve_remaining_seconds() == 894.5
        assert candidate_solve_timing()["candidate_oracle_timed_requests"] == 2.0
    assert solve_budget._current_candidate_solve_budget_meter() is None
