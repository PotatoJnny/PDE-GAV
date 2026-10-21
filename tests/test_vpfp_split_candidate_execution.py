"""Synthetic contract tests, not physical or solver-quality validation."""
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.plugins.vpfp.tasks.observable import VPFPObservableTask
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, AnswerView, Artifact, CandidateEvaluationOutput, Case,
    CaseSource, CaseVisibility, EvaluationPhase, EvaluationPlan, QueryKind, TaskKind,
)

FIELDS = ("rho", "momentum_x", "momentum_y", "kinetic_energy")
CASE = Case("split-vpfp", CaseSource.PUBLIC, CaseVisibility.HIDDEN, {
    "question": {"collision_nu": 20., "temperature": 1.,
                 "domain": {"time": [0., 1.]}, "grid": {"x_domain": [6., 8.]}},
    "arrays": {"t": [0., 1.], "x": [0., 3.], "y": [0., 4.]},
})


def _query():
    return AnswerQuery.at_points(fields=FIELDS, points=({"t": .7, "x": 5., "y": 7.},))


def _view(query):
    return AnswerView(query.kind, fields={name: (1.,)*len(query.points) for name in FIELDS},
                      points=query.points, shape=(len(query.points),))


def _artifact(program):
    return Artifact("split", TaskKind.OBSERVABLE_MODEL, "synthetic", program, "contract", "prepare", 1, "agent")


def test_split_execution_has_no_truth_and_score_keeps_macro_contract():
    events = []
    query = _query()

    class Candidate:
        def prepare_solution(self, case, solve_budget=None):
            assert case is CASE and solve_budget == 180
            events.append("prepared")
            return SimpleNamespace(evaluate=lambda q: (events.append("queried"), _view(q))[1],
                                   close=lambda: events.append("closed"))

    task = VPFPObservableTask()
    evaluation = task.create_evaluation({"default_query": query})
    assert evaluation.supports_split_candidate_execution
    plan = EvaluationPlan(EvaluationPhase.OFFICIAL, query, settings={
        "require_prepared_solution": True, "candidate_compute_budget": 180})
    artifact = _artifact(Candidate())
    output = evaluation.execute_candidate(artifact, CASE, plan)
    assert events == ["prepared", "queried", "closed"]
    assert isinstance(output, CandidateEvaluationOutput)
    assert output.metrics["candidate_formal_two_phase"] == 1.
    assert output.metrics["candidate_query_point_count"] == 1.
    assert not any("error" in key for key in output.metrics)
    scored = evaluation.score_candidate(artifact, CASE, _view(query), plan, output)
    assert scored.primary_error == 0.
    assert events == ["prepared", "queried", "closed"]
    assert scored.metrics["candidate_formal_two_phase"] == 1.


@pytest.mark.parametrize("failure", ["query", "invalid_handle"])
def test_prepared_handle_closes_even_when_query_or_validation_fails(failure):
    closed = []
    def broken(_query):
        raise RuntimeError("query failure")
    handle = SimpleNamespace(evaluate=broken if failure == "query" else None,
                             close=lambda: closed.append(True))
    candidate = SimpleNamespace(prepare_solution=lambda case, budget: handle)
    with pytest.raises((RuntimeError, TypeError)):
        VPFPObservableTask().execute_candidate(_artifact(candidate), CASE,
            EvaluationPlan(EvaluationPhase.OFFICIAL, _query()))
    assert closed == [True]


def test_formal_plan_rejects_legacy_even_when_callable():
    candidate = SimpleNamespace(predict_observables=lambda case, query, budget: _view(query))
    with pytest.raises(TypeError, match="requires prepare_solution"):
        VPFPObservableTask().execute_candidate(_artifact(candidate), CASE,
            EvaluationPlan(EvaluationPhase.OFFICIAL, _query(), settings={"require_prepared_solution": True}))


@pytest.mark.parametrize("budget", [True, 0, -1, float("inf"), "180"])
def test_invalid_solve_budget_is_rejected_before_candidate(budget):
    with pytest.raises(ValueError, match="candidate_compute_budget"):
        VPFPObservableTask().execute_candidate(_artifact(object()), CASE,
            EvaluationPlan(EvaluationPhase.OFFICIAL, _query(), settings={"candidate_compute_budget": budget}))


@pytest.mark.parametrize("phase,key", [(EvaluationPhase.OFFICIAL, "official_point_count"),
                                      (EvaluationPhase.TERMINAL, "terminal_point_count")])
def test_private_phase_queries_use_whole_torus_frozen_seed_and_count(phase, key):
    evaluation = VPFPObservableTask().create_evaluation({"default_query": _query()})
    settings = {key: 64, "default_query": _query()}
    first = evaluation.make_plan(CASE, phase, settings, 512)
    assert first.phase is phase and first.seed == 512
    assert first.query == evaluation.make_plan(CASE, phase, settings, 512).query
    assert first.query != evaluation.make_plan(CASE, phase, settings, 513).query
    assert len(first.query.points) == 64
    assert all(0 <= p["x"] <= 6 and 0 <= p["y"] <= 8 for p in first.query.points)
    assert any(p["x"] > 3 and p["y"] > 4 for p in first.query.points)
    with pytest.raises(TypeError):
        first.query.points[0]["x"] = 999


def test_explicit_query_and_absent_count_preserve_legacy_behavior():
    query = _query()
    evaluation = VPFPObservableTask().create_evaluation({"default_query": query})
    assert evaluation.make_plan(CASE, EvaluationPhase.OFFICIAL, {}, 1).query == query
    explicit = AnswerQuery.grid(fields=FIELDS, t=2, x=2, y=2)
    assert evaluation.make_plan(CASE, EvaluationPhase.OFFICIAL, {"official_point_count": 8},
                                2, request=explicit).query == explicit


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_private_phase_count_fails_closed(count):
    evaluation = VPFPObservableTask().create_evaluation({"default_query": _query()})
    with pytest.raises((TypeError, ValueError)):
        evaluation.make_plan(CASE, EvaluationPhase.OFFICIAL, {"official_point_count": count}, 1)
