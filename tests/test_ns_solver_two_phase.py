from __future__ import annotations

from pde_agent_verifier_solver.plugins.ns.tasks import NSSolverTask
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Artifact,
    Case,
    CaseSource,
    CaseVisibility,
    EvaluationPhase,
    QueryKind,
    TaskKind,
)


def _point_answer(query: AnswerQuery) -> AnswerView:
    return AnswerView(
        kind=QueryKind.POINTS,
        fields={
            "velocity_x": tuple(
                1.0 + point["t"] + point["x"] for point in query.points
            ),
            "velocity_y": tuple(
                2.0 + point["t"] - point["y"] for point in query.points
            ),
        },
        points=query.points,
        shape=(len(query.points),),
    )


class _PreparedHandle:
    def __init__(self, serial: int) -> None:
        self.serial = serial
        self.queries: list[AnswerQuery] = []

    def evaluate(self, query: AnswerQuery) -> AnswerView:
        self.queries.append(query)
        return _point_answer(query)


class _TwoPhaseCandidate:
    def __init__(self) -> None:
        self.prepared: list[tuple[str, float | None]] = []
        self.handles: list[_PreparedHandle] = []

    def prepare_solution(
        self,
        case: Case,
        solve_budget: float | None = None,
    ) -> _PreparedHandle:
        # The formal preparation signature has no query argument: hidden
        # coordinates enter only through the returned handle.
        self.prepared.append((case.case_id, solve_budget))
        handle = _PreparedHandle(len(self.handles))
        self.handles.append(handle)
        return handle


class _LegacyCandidate:
    def __init__(self, answer: AnswerView) -> None:
        self.answer = answer
        self.calls = 0

    def solve(self, case, query, budget=None):
        del case, query, budget
        self.calls += 1
        return self.answer


def _artifact(program: object, identifier: str = "candidate") -> Artifact:
    return Artifact(
        artifact_id=identifier,
        kind=TaskKind.SOLVER,
        name=identifier,
        program=program,
        summary="test",
        usage="formal two-phase NS solver",
        round_index=1,
        agent_id="agent",
    )


def _case(public_data=None) -> Case:
    return Case(
        case_id="ns-case",
        source=CaseSource.GENERATED,
        visibility=CaseVisibility.HIDDEN,
        public_data=public_data or {"equation": "NS"},
        round_index=1,
    )


def test_ns_two_phase_prepares_before_query_and_uses_a_fresh_handle_per_case_call():
    query = AnswerQuery.at_points(
        fields=("velocity_x", "velocity_y"),
        points=(
            {"t": 0.1, "x": 0.2, "y": 0.3},
            {"t": 0.4, "x": 0.5, "y": 0.6},
        ),
    )
    truth = _point_answer(query)
    candidate = _TwoPhaseCandidate()
    task = NSSolverTask()

    first = task.evaluate(_artifact(candidate), _case(), truth, query)
    second = task.evaluate(_artifact(candidate), _case(), truth, query)

    assert first.primary_error == 0.0
    assert second.primary_error == 0.0
    assert candidate.prepared == [("ns-case", None), ("ns-case", None)]
    assert len(candidate.handles) == 2
    assert candidate.handles[0] is not candidate.handles[1]
    assert candidate.handles[0].queries == [query]
    assert candidate.handles[1].queries == [query]
    assert first.metrics["candidate_formal_two_phase"] == 1.0
    assert first.metrics["candidate_query_point_count"] == 2.0
    assert first.metrics["candidate_solve_wall_seconds"] >= 0.0
    assert first.metrics["candidate_query_wall_seconds"] >= 0.0


def test_ns_legacy_solve_compatibility_remains_available_for_grid_queries():
    query = AnswerQuery.grid(
        fields=("velocity_x", "velocity_y"),
        t=1,
        x=1,
        y=2,
    )
    truth = AnswerView(
        kind=QueryKind.GRID,
        fields={
            "velocity_x": (1.0, 2.0),
            "velocity_y": (3.0, 4.0),
        },
        axes={"t": (0.0,), "x": (0.0,), "y": (0.0, 1.0)},
        shape=(1, 1, 2),
    )
    candidate = _LegacyCandidate(truth)

    result = NSSolverTask().evaluate(
        _artifact(candidate, "legacy"),
        _case(),
        truth,
        query,
    )

    assert result.primary_error == 0.0
    assert candidate.calls == 1
    assert "candidate_formal_two_phase" not in result.metrics


def test_ns_formal_two_phase_rejects_a_grid_query_before_preparation():
    query = AnswerQuery.grid(
        fields=("velocity_x", "velocity_y"),
        t=1,
        x=1,
        y=1,
    )
    truth = AnswerView(
        kind=QueryKind.GRID,
        fields={"velocity_x": (1.0,), "velocity_y": (2.0,)},
        shape=(1, 1, 1),
    )
    candidate = _TwoPhaseCandidate()

    try:
        NSSolverTask().evaluate(_artifact(candidate), _case(), truth, query)
    except ValueError as exc:
        assert "requires a POINTS query" in str(exc)
    else:
        raise AssertionError("formal two-phase NS evaluation accepted a grid")
    assert candidate.prepared == []


def test_ns_private_plans_follow_regular_and_continuous_case_schemas():
    settings = {
        "default_query": AnswerQuery.grid(
            fields=("velocity_x", "velocity_y"),
            t=2,
            x=3,
            y=3,
        ),
        "development_point_count": 32,
    }
    evaluation = NSSolverTask().create_evaluation(settings)
    regular = _case(
        {
            "question": {"time": {"t0": 0.0, "t1": 2.0}},
            "arrays": {
                "x": (0.0, 0.5, 1.0),
                "y": (0.0, 0.5, 1.0),
                "fluid_mask": (
                    (True, True, True),
                    (True, False, True),
                    (True, True, True),
                ),
            },
        }
    )

    development = evaluation.make_private_development_plan(
        regular,
        settings,
        seed=17,
    )
    official = evaluation.make_plan(
        regular,
        EvaluationPhase.OFFICIAL,
        settings,
        seed=18,
    )
    terminal = evaluation.make_plan(
        regular,
        EvaluationPhase.TERMINAL,
        settings,
        seed=19,
    )

    assert len(development.query.points) == 32
    assert len(official.query.points) == 96
    assert len(terminal.query.points) == 128
    native = {0.0, 0.5, 1.0}
    assert all(
        point["x"] in native
        and point["y"] in native
        and (point["x"], point["y"]) != (0.5, 0.5)
        and 0.0 <= point["t"] <= 2.0
        for point in (*development.query.points, *official.query.points)
    )

    continuous = _case(
        {
            "domain": {
                "time_interval": (1.0, 3.0),
                "bounds": {"x": (-1.0, 1.0), "y": (-2.0, 2.0)},
                "geometry": {
                    "obstacle_polygons": (
                        ((-0.25, -0.25), (0.25, -0.25), (0.25, 0.25), (-0.25, 0.25)),
                    )
                },
            }
        }
    )
    configured = {
        **settings,
        "official_point_count": 7,
        "terminal_point_count": 9,
    }
    first = evaluation.make_plan(
        continuous,
        EvaluationPhase.OFFICIAL,
        configured,
        seed=71,
    )
    repeated = evaluation.make_plan(
        continuous,
        EvaluationPhase.OFFICIAL,
        configured,
        seed=71,
    )
    fresh = evaluation.make_plan(
        continuous,
        EvaluationPhase.OFFICIAL,
        configured,
        seed=72,
    )
    configured_terminal = evaluation.make_plan(
        continuous,
        EvaluationPhase.TERMINAL,
        configured,
        seed=73,
    )

    assert first.query == repeated.query
    assert first.query != fresh.query
    assert len(first.query.points) == 7
    assert len(configured_terminal.query.points) == 9
    assert all(
        1.0 <= point["t"] <= 3.0
        and -1.0 <= point["x"] <= 1.0
        and -2.0 <= point["y"] <= 2.0
        and not (
            -0.25 <= point["x"] <= 0.25
            and -0.25 <= point["y"] <= 0.25
        )
        for point in (*first.query.points, *configured_terminal.query.points)
    )


def test_ns_preparation_receives_remaining_worker_budget(monkeypatch):
    from pde_agent_verifier_solver.sdk import solve_budget

    now = [105.0]
    monkeypatch.setattr(solve_budget, "monotonic", lambda: now[0])
    query = AnswerQuery.at_points(
        fields=("velocity_x", "velocity_y"),
        points=({"t": 0.1, "x": 0.2, "y": 0.3},),
    )
    candidate = _TwoPhaseCandidate()
    with solve_budget.bind_candidate_solve_budget(
        new_case=True, budget_seconds=900.0, started_at_monotonic=100.0,
    ):
        result = NSSolverTask().evaluate(_artifact(candidate), _case(), _point_answer(query), query)
    assert candidate.prepared == [("ns-case", 895.0)]
    assert result.primary_error == 0.0


def test_ns_exhausted_worker_budget_fails_before_candidate_preparation(monkeypatch):
    import pytest
    from pde_agent_verifier_solver.sdk import solve_budget

    monkeypatch.setattr(solve_budget, "monotonic", lambda: 1000.0)
    query = AnswerQuery.at_points(
        fields=("velocity_x", "velocity_y"),
        points=({"t": 0.1, "x": 0.2, "y": 0.3},),
    )
    candidate = _TwoPhaseCandidate()
    with solve_budget.bind_candidate_solve_budget(
        new_case=True, budget_seconds=900.0, started_at_monotonic=100.0,
    ), pytest.raises(TimeoutError, match="before preparation"):
        NSSolverTask().evaluate(_artifact(candidate), _case(), _point_answer(query), query)
    assert candidate.prepared == []
