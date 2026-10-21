"""The VPFP Task actually prefers the query-blind continuous candidate ABI."""

from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.plugins.vpfp.macro_matching import VPFP_MACRO_FIELDS
from pde_agent_verifier_solver.plugins.vpfp.tasks.observable import VPFPObservableTask
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, AnswerView, Artifact, Case, CaseSource, CaseVisibility,
    QueryKind, TaskKind,
)


CASE = Case(
    case_id="continuous-vpfp-task",
    source=CaseSource.GENERATED,
    visibility=CaseVisibility.HIDDEN,
    public_data={"question": {"collision_nu": 20.0, "temperature": 1.0}},
)


def _query(kind=QueryKind.POINTS):
    if kind is QueryKind.GRID:
        return AnswerQuery.grid(fields=VPFP_MACRO_FIELDS, t=2, x=2, y=2)
    return AnswerQuery.at_points(
        fields=VPFP_MACRO_FIELDS,
        points=({"t": .81, "x": .17, "y": .23},
                {"t": .13, "x": .91, "y": .47},
                {"t": .81, "x": .17, "y": .23}),
    )


def _answer(query):
    if query.kind is QueryKind.POINTS:
        count = len(query.points)
        coordinates = {"points": query.points, "shape": (count,)}
    else:
        count = 8
        coordinates = {"axes": {"t": (0., 1.), "x": (.2, .7), "y": (.3, .9)},
                       "shape": (2, 2, 2)}
    return AnswerView(query.kind,
        {name: (float(index+1),)*count for index, name in enumerate(VPFP_MACRO_FIELDS)},
        **coordinates)


def _artifact(program):
    return Artifact("two-phase-test", TaskKind.OBSERVABLE_MODEL, "candidate",
        program, "Task ABI test", "prepare_solution then evaluate", 1, "agent")


def _run(program, kind=QueryKind.POINTS):
    query = _query(kind)
    return VPFPObservableTask().evaluate(_artifact(program), CASE, _answer(query), query)


def test_only_prepare_candidate_is_macro_only_and_query_is_revealed_after_return():
    events = []

    class Handle:
        def evaluate(self, query):
            assert events == ["prepare_started", "prepare_returned"]
            assert query.kind is QueryKind.POINTS
            assert query.points[0] == query.points[2]
            events.append("query_revealed")
            return _answer(query)

    class Candidate:
        def prepare_solution(self, *args, **kwargs):
            # The existing SDK ABI supplies only Case and None budget. No
            # scoring query, hidden truth, grid, or output resolution is passed.
            assert args == (CASE, None)
            assert kwargs == {}
            events.append("prepare_started")
            handle = Handle()
            events.append("prepare_returned")
            return handle

    result = _run(Candidate())
    assert result.primary_error == 0.0
    assert set(result.prediction.fields) == set(VPFP_MACRO_FIELDS)
    assert events == ["prepare_started", "prepare_returned", "query_revealed"]


def test_prepare_is_preferred_even_when_legacy_entry_point_would_raise():
    class Candidate:
        def prepare_solution(self, case, solve_budget=None):
            assert case is CASE and solve_budget is None
            return SimpleNamespace(evaluate=_answer)

        def predict_observables(self, *_args, **_kwargs):
            raise AssertionError("legacy entry point must not be called")

    assert _run(Candidate()).primary_error == 0.0


@pytest.mark.parametrize("failure_phase", ("prepare", "evaluate"))
def test_prepare_or_handle_failure_never_falls_back_to_legacy(failure_phase):
    original = ArithmeticError("original candidate failure")
    legacy_calls = []

    class Candidate:
        def prepare_solution(self, case, solve_budget=None):
            if failure_phase == "prepare":
                raise original

            def evaluate(query):
                raise original

            return SimpleNamespace(evaluate=evaluate)

        def predict_observables(self, case, query, budget=None):
            legacy_calls.append(query)
            return _answer(query)

    with pytest.raises(ArithmeticError) as caught:
        _run(Candidate())
    assert caught.value is original
    assert legacy_calls == []


@pytest.mark.parametrize("invalid", (None, 0, "not callable"))
def test_declared_noncallable_prepare_does_not_fall_back(invalid):
    class Candidate:
        prepare_solution = invalid

        def predict_observables(self, *_args, **_kwargs):
            raise AssertionError("noncallable preparation is not legacy absence")

    with pytest.raises(TypeError, match="callable prepare_solution"):
        _run(Candidate())


def test_declared_prepare_descriptor_attribute_error_is_not_legacy_absence():
    original = AttributeError("declared preparation cannot be resolved")

    class Candidate:
        @property
        def prepare_solution(self):
            raise original

        def predict_observables(self, *_args, **_kwargs):
            raise AssertionError("broken preparation must not switch candidate ABI")

    with pytest.raises(AttributeError) as caught:
        _run(Candidate())
    assert caught.value is original


@pytest.mark.parametrize("handle", (None, SimpleNamespace(evaluate=None)))
def test_invalid_prepared_handle_does_not_fall_back(handle):
    class Candidate:
        def prepare_solution(self, case, solve_budget=None):
            return handle

        def predict_observables(self, *_args, **_kwargs):
            raise AssertionError("invalid prepared handle must not fall back")

    with pytest.raises(TypeError, match="evaluate"):
        _run(Candidate())


@pytest.mark.parametrize("kind", (QueryKind.POINTS, QueryKind.GRID))
def test_legacy_predict_observables_is_still_compatible(kind):
    calls = []

    class Candidate:
        def predict_observables(self, case, query, budget=None):
            calls.append((case, query.kind, budget))
            return _answer(query)

    assert _run(Candidate(), kind).primary_error == 0.0
    assert calls == [(CASE, kind, None)]


def test_grid_explicitly_uses_legacy_even_when_prepare_is_available():
    class Candidate:
        def prepare_solution(self, *_args, **_kwargs):
            raise AssertionError("prepare/evaluate contract is POINTS-only")

        def predict_observables(self, case, query, budget=None):
            assert query.kind is QueryKind.GRID
            return _answer(query)

    assert _run(Candidate(), QueryKind.GRID).primary_error == 0.0


def test_only_prepare_candidate_needs_points_not_implicit_grid_conversion():
    candidate = SimpleNamespace(prepare_solution=lambda case, solve_budget=None:
                                SimpleNamespace(evaluate=_answer))
    with pytest.raises(TypeError, match="legacy POINTS/GRID compatibility"):
        _run(candidate, QueryKind.GRID)


def test_agent_brief_prioritizes_continuous_oracles_and_query_blind_preparation():
    brief = VPFPObservableTask().agent_brief({})
    for text in ("open_public_case_session", "source", "rho0", "g0",
                 "prepare_solution(case, solve_budget=None)", "handle.evaluate(query)",
                 "only after preparation returns", "rho, momentum_x",
                 "legacy compatibility", "Oracle waiting", "solve_budget=None"):
        assert text in brief
