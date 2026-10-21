"""Software transport tests; these do not claim scientific/GPU throughput."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
import time

import pytest

from pde_agent_verifier_solver.engine import candidate_execution as module
from pde_agent_verifier_solver.engine.candidate_execution import (
    CandidateExecutionError,
    CandidateExecutionInfrastructureError,
    CandidatePublicSessionClient,
    PythonTaskWorkerSpec,
    SubprocessCandidateExecutionService,
    _CandidateOracleHostGroup,
)
from pde_agent_verifier_solver.plugins.ns import NSSolverTask
from pde_agent_verifier_solver.sdk import (
    AnswerView, CandidateEvaluationOutput, EvaluationPhase, EvaluationPlan, QueryKind,
)
from pde_agent_verifier_solver.sdk.solve_budget import (
    _record_oracle_lock_wait, bind_candidate_solve_budget, candidate_solve_timing,
)
from pde_agent_verifier_solver.sdk._candidate_timing import controller_queue_wait
from tests.test_candidate_execution import _artifact, _case, _evaluation_settings, _query


def _service(root: Path, **kwargs):
    return SubprocessCandidateExecutionService(
        root, task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),
        evaluation_settings={}, max_workers=3, case_hard_wall_seconds=30, **kwargs,
    )


@pytest.mark.parametrize("value,error", [
    (0, ValueError), (-1, ValueError), (True, TypeError),
    (1.5, TypeError), ("3", TypeError),
])
def test_host_count_validation_at_construction_and_phase_assignment(tmp_path, value, error):
    with pytest.raises(error, match="oracle_host_count"):
        _service(tmp_path / "invalid", oracle_host_count=value)
    service = _service(tmp_path / "valid")
    try:
        assert service.oracle_host_count == 1
        service.oracle_host_count = 3
        with pytest.raises(error, match="oracle_host_count"):
            service.oracle_host_count = value
        assert service.oracle_host_count == 3
    finally:
        service._executor.shutdown()


@pytest.mark.parametrize("phase", [
    EvaluationPhase.DEVELOPMENT, EvaluationPhase.OFFICIAL, EvaluationPhase.TERMINAL,
])
def test_executor_keeps_order_and_uses_lanes_only_for_hidden(tmp_path, monkeypatch, phase):
    artifact = _artifact(tmp_path)
    query = _query()
    settings = _evaluation_settings(query)
    evaluation = NSSolverTask().create_evaluation(settings)
    cases = tuple(_case(f"case-{index}", value=index) for index in range(6))
    plans = tuple(EvaluationPlan(phase=phase, query=query, seed=index, settings=settings)
                  for index in range(len(cases)))
    manifests, closed = {}, []
    service = _service(tmp_path / "execution", oracle_host_count=3)

    def worker(execution_id, index, artifact, case, plan, program_id, oracle,
               call_root, cancel, cancellation_event, batch_deadline, batch_members=(),
               *, route_devices=(), requester_agent_id=None, submitted_at=None):
        del execution_id, artifact, program_id, call_root, cancel, cancellation_event, batch_deadline
        manifest = oracle.manifest_for_case(case.case_id)
        manifests[index] = manifest
        client = CandidatePublicSessionClient(manifest)
        value = client.invoke(case.case_id, "marker", (index,), {})
        # Deliberately finish in a different order than the input cohort.
        time.sleep(0.003 * (2 - index % 3))
        oracle.finish_case(case.case_id, drain_timeout_seconds=1)
        return CandidateEvaluationOutput(AnswerView(
            kind=QueryKind.POINTS, fields={"velocity_x": (value, value)},
            points=plan.query.points, shape=(2,),
        ), {})

    monkeypatch.setattr(service, "_execute_one", worker)
    try:
        outputs = service.execute_cases(
            artifact, cases, plans, evaluation, public_session_methods=("marker",),
            invoke_public_session=lambda case_id, method, args, kwargs: float(args[0]),
            close_public_session=closed.append,
        )
        count = 1 if phase is EvaluationPhase.DEVELOPMENT else 3
        assert len({manifest["endpoint"] for manifest in manifests.values()}) == count
        for index in range(len(cases)):
            assert manifests[index]["endpoint"] == manifests[index % count]["endpoint"]
            assert manifests[index]["case_id"] == cases[index].case_id
        assert [output.prediction.fields["velocity_x"][0] for output in outputs] == list(range(6))
        assert sorted(closed) == sorted(case.case_id for case in cases)
        assert not tuple(service.root.glob("evaluation-*"))
        assert service.oracle_host_count == 3  # Development did not mutate policy.
    finally:
        service._executor.shutdown()


def _call(client, case_id):
    with bind_candidate_solve_budget(new_case=True):
        value = client.invoke(case_id, "marker", (), {})
        return value, dict(candidate_solve_timing())


def test_real_http_lanes_are_independent_and_wait_ledgers_do_not_cross(tmp_path):
    entered, release = Event(), Event()
    closed = []

    def invoke(case_id, method, args, kwargs):
        assert method == "marker" and args == () and kwargs == {}
        if case_id == "blocked":
            entered.set()
            assert release.wait(5)
        elif case_id == "free":
            # A distinct, attributable lock wait in lane 1 must not leak to lane 0.
            # Notify the live parent ledger as well as the response classifier.
            with controller_queue_wait():
                began = time.perf_counter()
                time.sleep(0.03)
                _record_oracle_lock_wait(time.perf_counter() - began)
        return case_id

    group = _CandidateOracleHostGroup(
        tmp_path / "arrays", case_ids=("blocked", "free", "queued"),
        methods=("marker",), invoke=invoke, close=closed.append, host_count=2,
    )
    clients = {case_id: CandidatePublicSessionClient(group.manifest_for_case(case_id))
               for case_id in ("blocked", "free", "queued")}
    assert clients["blocked"].endpoint == clients["queued"].endpoint
    assert clients["blocked"].endpoint != clients["free"].endpoint
    group.start()
    pool = ThreadPoolExecutor(max_workers=3)
    try:
        blocked = pool.submit(_call, clients["blocked"], "blocked")
        assert entered.wait(2)
        queued = pool.submit(_call, clients["queued"], "queued")
        lane_zero = group._hosts[0].state
        deadline = time.monotonic() + 2
        while lane_zero.queue_wait_seconds("queued") < 0.001 and time.monotonic() < deadline:
            time.sleep(0.002)
        assert lane_zero.queue_wait_seconds("queued") >= 0.001
        free_value, free_timing = pool.submit(_call, clients["free"], "free").result(timeout=2)
        assert free_value == "free" and not blocked.done() and not queued.done()
        time.sleep(0.12)
        release.set()
        blocked_value, blocked_timing = blocked.result(timeout=2)
        queued_value, queued_timing = queued.result(timeout=2)
        assert (blocked_value, queued_value) == ("blocked", "queued")
        assert queued_timing["candidate_oracle_queue_wait_seconds"] >= 0.10
        assert free_timing["candidate_oracle_queue_wait_seconds"] < queued_timing["candidate_oracle_queue_wait_seconds"]
        assert free_timing["candidate_oracle_lock_wait_seconds"] >= 0.025
        for timing in (blocked_timing, queued_timing):
            assert timing["candidate_oracle_lock_wait_seconds"] == 0.0
        for timing in (blocked_timing, queued_timing, free_timing):
            assert timing["candidate_oracle_timed_requests"] == 1.0
            assert timing["candidate_platform_wait_seconds"] == pytest.approx(
                timing["candidate_oracle_queue_wait_seconds"] + timing["candidate_oracle_lock_wait_seconds"]
            )
        group.finish_case("blocked", drain_timeout_seconds=1)
        group.finish_case("blocked", drain_timeout_seconds=1)
        with pytest.raises(CandidateExecutionInfrastructureError, match="expired"):
            group.manifest_for_case("blocked")
        with pytest.raises(CandidateExecutionError):
            clients["blocked"].invoke("blocked", "marker", (), {})
    finally:
        release.set()
        pool.shutdown()
        group.close(drain_timeout_seconds=1)
        group.close(drain_timeout_seconds=1)
    assert sorted(closed) == ["blocked", "free", "queued"]
    assert all(not host._thread.is_alive() and not host._server._request_worker.is_alive()
               and not any(worker.is_alive() for worker in host.state._workers)
               for host in group._hosts)


@pytest.mark.parametrize("failure", ["init", "start", "close"])
def test_partial_failure_closes_every_constructed_lane(tmp_path, monkeypatch, failure):
    constructed, closed = [], []

    class Host:
        def __init__(self, root, **kwargs):
            self.lane = int(root.name.removeprefix("lane-"))
            if failure == "init" and self.lane == 1:
                raise RuntimeError("test init failure")
            constructed.append(self.lane)

        def start(self):
            if failure == "start" and self.lane == 1:
                raise RuntimeError("test start failure")

        def close(self, **kwargs):
            closed.append(self.lane)
            if failure == "close" and self.lane == 0:
                raise RuntimeError("test close failure")

    monkeypatch.setattr(module, "_CandidateOracleHost", Host)
    with pytest.raises(RuntimeError, match=f"test {failure} failure"):
        group = _CandidateOracleHostGroup(
            tmp_path / "arrays", case_ids=("a", "b", "c"), methods=(),
            invoke=None, close=None, host_count=3,
        )
        group.start()
        group.close()
    assert closed == constructed
