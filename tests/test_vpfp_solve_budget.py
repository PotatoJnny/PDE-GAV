"""Synthetic timing regressions, not physical or solver-quality validation."""

from functools import partial
import time

import pytest

from pde_agent_verifier_solver.engine import PythonCandidateCodec
from pde_agent_verifier_solver.plugins.vpfp.tasks import observable
from pde_agent_verifier_solver.sdk import (
    EvaluationPhase, EvaluationPlan, TaskKind, inspect_candidate_bundle,
    measure_point_query, measure_prepare_solution, write_candidate_bundle_manifest,
)
from pde_agent_verifier_solver.sdk import solve_budget as clocks
from tests.test_vpfp_split_candidate_execution import CASE, _query, _view


SOURCE = '''from time import perf_counter

def prepare_solution(case, solve_budget=None):
    began = perf_counter()
    advance()
    if cooperative and solve_budget is not None and perf_counter() - began > solve_budget:
        raise TimeoutError("frozen candidate exceeded its solve budget")
    return make_handle()
'''


def _frozen(tmp_path, *, sdk_clock=False):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    source = SOURCE
    if sdk_clock:
        source = source.replace("from time import perf_counter", (
            "from pde_agent_verifier_solver.sdk.solve_budget "
            "import candidate_solve_clock as perf_counter"))
    (bundle / "submission.py").write_text(source)
    write_candidate_bundle_manifest(bundle, files=("submission.py",))
    return PythonCandidateCodec(task_kind=TaskKind.OBSERVABLE_MODEL).bind_existing(
        bundle, storage_root=tmp_path / "frozen", artifact_id="frozen-timing",
        name="Frozen timer", summary="Synthetic timing test", usage="prepare then query",
    )


def _fake_clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(clocks, "perf_counter", lambda: now[0])
    monkeypatch.setattr(clocks, "monotonic", lambda: now[0])
    # The SDK helper's raw clock must remain raw: the compatibility scope only
    # rebinds the Candidate's alias. Explicit injection avoids real test sleeps.
    monkeypatch.setattr(observable, "measure_prepare_solution", partial(
        measure_prepare_solution, clock=lambda: now[0]))
    monkeypatch.setattr(observable, "measure_point_query", partial(
        measure_point_query, clock=lambda: now[0]))
    return now


def _advance(now, request_id, work, queue, lock=0.0):
    now[0] += work + queue + lock
    clocks._record_oracle_response_wait(request_id, {
        "schema": clocks.ORACLE_WAIT_TIMING_SCHEMA,
        "queue_wait_seconds": queue, "lock_wait_seconds": lock,
    }, rpc_wall_seconds=work + queue + lock)


@pytest.mark.parametrize("work,queue,cooperative,succeeds", [
    (49.6, 131.24, True, True),  # The observed R3 failure's charged/queue scales.
    (49.6, 240.0, True, True),  # Waiting alone exceeds the 180-second budget.
    (181.0, 240.0, True, False),
    (49.6, 240.0, False, True),
    (181.0, 240.0, False, False),  # Backend enforces even without a guard.
])
@pytest.mark.parametrize("sdk_clock", [False, True], ids=["legacy", "sdk"])
def test_frozen_prepare_excludes_only_platform_wait(
    tmp_path, monkeypatch, work, queue, cooperative, succeeds, sdk_clock,
):
    artifact = _frozen(tmp_path, sdk_clock=sdk_clock)
    module = artifact.program._module()
    original_alias = module.perf_counter
    original_time = time.perf_counter
    tree = artifact.program.tree_sha256
    now = _fake_clock(monkeypatch)
    closed = []

    def advance():
        assert module.perf_counter is clocks.candidate_solve_clock
        assert time.perf_counter is original_time
        _advance(now, "prepare-rpc", work, queue, 2.0)

    class Handle:
        def evaluate(self, query):
            assert module.perf_counter is original_alias  # Preparation only.
            _advance(now, "query-rpc", 0.5, 20.0)
            return _view(query)

        def close(self):
            closed.append(True)

    module.advance = advance
    module.make_handle = Handle
    module.cooperative = cooperative
    task = observable.VPFPObservableTask()
    plan = EvaluationPlan(EvaluationPhase.OFFICIAL, _query(), settings={
        "require_prepared_solution": True, "candidate_compute_budget": 180.0,
    })
    if succeeds:
        output = task.execute_candidate(artifact, CASE, plan)
        assert output.prediction == _view(_query())
        assert output.metrics["candidate_solve_wall_seconds"] == pytest.approx(work)
        assert output.metrics["candidate_solve_queue_wait_seconds"] == pytest.approx(queue + 2.0)
        assert output.metrics["candidate_query_wall_seconds"] == pytest.approx(0.5)
        assert output.metrics["candidate_query_queue_wait_seconds"] == pytest.approx(20.0)
        assert output.metrics["candidate_budget_clock_compat_aliases"] == (0.0 if sdk_clock else 1.0)
        assert closed == [True]
    else:
        with pytest.raises(TimeoutError, match="exceeded.*budget"):
            task.execute_candidate(artifact, CASE, plan)
        assert closed == ([] if cooperative else [True])
    assert time.perf_counter is original_time
    assert module.perf_counter is original_alias
    assert inspect_candidate_bundle(artifact.program.bundle_locator).tree_sha256 == tree
    assert clocks.candidate_solve_timing()["candidate_platform_wait_seconds"] == 0.0


@pytest.mark.parametrize("phase", ["prepare", "invalid_handle", "query"])
def test_failure_restores_legacy_alias_and_closes_returned_handle(tmp_path, monkeypatch, phase):
    artifact = _frozen(tmp_path)
    module = artifact.program._module()
    original_alias = module.perf_counter
    now = _fake_clock(monkeypatch)
    closed = []

    def advance():
        _advance(now, "prepare-rpc", 1.0, 200.0)
        if phase == "prepare":
            raise RuntimeError("prepare failed")

    class Handle:
        def evaluate(self, query):
            assert module.perf_counter is original_alias
            raise RuntimeError("query failed")

        def close(self):
            closed.append(True)

    if phase == "invalid_handle":
        Handle.evaluate = None
    module.advance, module.make_handle, module.cooperative = advance, Handle, True
    with pytest.raises((RuntimeError, TypeError)):
        observable.VPFPObservableTask().execute_candidate(artifact, CASE,
            EvaluationPlan(EvaluationPhase.OFFICIAL, _query(), settings={"candidate_compute_budget": 180.0}))
    assert closed == ([] if phase == "prepare" else [True])
    assert module.perf_counter is original_alias
    assert clocks.candidate_solve_timing()["candidate_platform_wait_seconds"] == 0.0


def test_task_reuses_worker_meter_without_resetting_upstream_budget(tmp_path, monkeypatch):
    artifact = _frozen(tmp_path)
    module = artifact.program._module()
    now = _fake_clock(monkeypatch)
    module.cooperative = True
    module.make_handle = lambda: type("Handle", (), {"evaluate": lambda self, query: _view(query)})()
    with clocks.bind_candidate_solve_budget(new_case=True, budget_seconds=900.0) as meter:
        now[0] += 10.0  # Work before Task entry is still charged by the worker.

        def advance():
            assert clocks._current_candidate_solve_budget_meter() is meter
            _advance(now, "prepare-rpc", 40.0, 240.0)

        module.advance = advance
        output = observable.VPFPObservableTask().execute_candidate(artifact, CASE,
            EvaluationPlan(EvaluationPhase.OFFICIAL, _query(), settings={"candidate_compute_budget": 180.0}))
        assert output.metrics["candidate_solve_wall_seconds"] == 40.0
        assert clocks.candidate_solve_remaining_seconds() == 850.0
        assert meter.snapshot()["candidate_platform_wait_seconds"] == 240.0


def test_agent_brief_matches_shared_budget_contract():
    brief = observable.VPFPObservableTask().agent_brief({})
    assert "remains inside the preparation wall time" not in brief
    for text in ("candidate_solve_clock", "shared locks is excluded", "numerical execution",
                 "remain charged", "Agent total wall time", "still include all waiting"):
        assert text in brief


def test_ns_remaining_case_has_no_independent_prepare_cap(tmp_path, monkeypatch):
    artifact = _frozen(tmp_path)
    module = artifact.program._module()
    now = _fake_clock(monkeypatch)
    received = []
    module.cooperative = False
    module.advance = lambda: _advance(now, "long-prepare", 700.0, 300.0)

    class Handle:
        def evaluate(self, query):
            received.append(clocks.candidate_solve_remaining_seconds())
            now[0] += 20.0
            return _view(query)

    module.make_handle = Handle
    original_prepare = module.prepare_solution

    def prepare(case, solve_budget=None):
        received.append(solve_budget)
        return original_prepare(case, solve_budget)

    module.prepare_solution = prepare
    with clocks.bind_candidate_solve_budget(new_case=True, budget_seconds=900.0):
        now[0] = 30.0  # Startup is part of the same Case.
        output = observable.VPFPObservableTask().execute_candidate(artifact, CASE,
            EvaluationPlan(EvaluationPhase.OFFICIAL, _query(), settings={
                "require_prepared_solution": True,
                "candidate_budget_mode": "ns_remaining_case_v1"}))
        assert received == [870.0, 170.0]
        assert clocks.candidate_solve_remaining_seconds() == 150.0
        assert output.metrics["candidate_solve_wall_seconds"] == 700.0


@pytest.mark.parametrize("budget", [180, 900, float("inf"), False])
def test_ns_remaining_case_rejects_separate_preparation_budget(tmp_path, budget):
    artifact = _frozen(tmp_path)
    with pytest.raises(ValueError, match="forbids a separate"):
        observable.VPFPObservableTask().execute_candidate(artifact, CASE,
            EvaluationPlan(EvaluationPhase.OFFICIAL, _query(), settings={
                "candidate_budget_mode": "ns_remaining_case_v1",
                "candidate_compute_budget": budget}))


def test_ns_remaining_case_does_not_restart_exhausted_budget(tmp_path, monkeypatch):
    artifact = _frozen(tmp_path)
    now = _fake_clock(monkeypatch)
    with clocks.bind_candidate_solve_budget(new_case=True, budget_seconds=900.0):
        now[0] = 901.0
        with pytest.raises(TimeoutError, match="exhausted before preparation"):
            observable.VPFPObservableTask().execute_candidate(artifact, CASE,
                EvaluationPlan(EvaluationPhase.OFFICIAL, _query(), settings={
                    "candidate_budget_mode": "ns_remaining_case_v1"}))


def test_ns_remaining_case_brief_is_explicit():
    brief = observable.VPFPObservableTask().agent_brief({
        "candidate_budget_mode": "ns_remaining_case_v1"})
    assert "there is no separate preparation limit" in brief
    assert "after startup/import work" in brief
    assert "when absent this Task passes solve_budget=None" not in brief
    assert "remaining shared Case time from the execution-owned budget meter" in brief
