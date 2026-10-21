"""One queue ledger supports both upstream helpers and cooperative clocks."""

import pytest

from pde_agent_verifier_solver.sdk import (
    AnswerQuery, AnswerView, Case, CaseSource, CaseVisibility, QueryKind,
)
from pde_agent_verifier_solver.sdk import continuous_solution as continuous
from pde_agent_verifier_solver.sdk import solve_budget
from pde_agent_verifier_solver.sdk._candidate_timing import controller_queue_wait_seconds


@pytest.mark.parametrize("clock_mode", ["default", "explicit_raw", "explicit_sdk"])
def test_prepare_and_query_exclude_same_wait_exactly_once(monkeypatch, clock_mode):
    now = [0.0]
    raw_clock = lambda: now[0]
    monkeypatch.setattr(solve_budget, "perf_counter", raw_clock)
    # A deterministic replacement of the default raw clock lets the default
    # API path be tested without OS scheduling tolerances or artificial sleep.
    monkeypatch.setitem(continuous.measure_prepare_solution.__kwdefaults__, "clock", raw_clock)
    monkeypatch.setitem(continuous.measure_point_query.__kwdefaults__, "clock", raw_clock)

    def advance(request_id, elapsed, queue, lock):
        now[0] += elapsed
        solve_budget._record_oracle_response_wait(request_id, {
            "schema": solve_budget.ORACLE_WAIT_TIMING_SCHEMA,
            "queue_wait_seconds": queue,
            "lock_wait_seconds": lock,
        }, rpc_wall_seconds=elapsed)

    class Handle:
        def evaluate(self, query):
            advance("query", 9.0, 2.0, 3.0)
            return AnswerView(
                kind=QueryKind.POINTS, fields={"u": (1.0, 2.0)},
                points=query.points, shape=(2,),
            )

    class Program:
        def prepare_solution(self, case, budget):
            advance("prepare", 10.0, 4.0, 3.0)
            return Handle()

    kwargs = {} if clock_mode == "default" else {
        "clock": solve_budget.candidate_solve_clock if clock_mode == "explicit_sdk" else raw_clock,
    }
    case = Case("one", CaseSource.GENERATED, CaseVisibility.DEVELOPMENT, {})
    query = AnswerQuery.at_points(fields=("u",), points=({"x": 0.0}, {"x": 1.0}))
    with solve_budget.bind_candidate_solve_budget(new_case=True):
        prepared = continuous.measure_prepare_solution(Program(), case, 4.0, **kwargs)
        assert prepared.candidate_solve_wall_seconds == 3.0
        assert prepared.candidate_solve_queue_wait_seconds == 7.0
        assert controller_queue_wait_seconds() == 7.0
        queried = continuous.measure_point_query(prepared.handle, query, **kwargs)
        assert queried.candidate_query_wall_seconds == 4.0
        assert queried.candidate_query_queue_wait_seconds == 5.0
        assert controller_queue_wait_seconds() == 12.0
        assert solve_budget.candidate_solve_clock() == 7.0
        assert continuous.SolveWallEligibilityContract(4.0).assess(prepared).eligible
        assert not continuous.SolveWallEligibilityContract(2.0).assess(prepared).eligible
    assert controller_queue_wait_seconds() == 0.0
    assert solve_budget.candidate_solve_clock() == 19.0
