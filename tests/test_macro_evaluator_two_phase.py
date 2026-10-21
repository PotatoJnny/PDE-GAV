import unittest

from pde_agent_verifier_solver.engine.macro_evaluator import (
    ControllerMacroEvaluator,
    MacroCaseTimingAudit,
    PreparedMacroCohort,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Case,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    MacroMatchResult,
    MacroTarget,
    QueryKind,
    open_public_case_session,
)
from pde_agent_verifier_solver.sdk.continuous_solution import TimingAssurance


POINT_QUERY = AnswerQuery.at_points(
    fields=("phi",),
    points=(
        {"t": 0.0, "s_index": 1.0},
        {"t": 1.0, "s_index": 3.0},
    ),
)
GRID_QUERY = AnswerQuery.grid(fields=("phi",), x=1)


class SequenceClock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def answer_for(query: AnswerQuery) -> AnswerView:
    if query.kind is QueryKind.POINTS:
        point_count = len(query.points)
        values = tuple(
            component
            for index in range(point_count)
            for component in (float(index + 1), float(-(index + 1)))
        )
        return AnswerView(
            kind=QueryKind.POINTS,
            fields={"phi": values},
            points=query.points,
            shape=(point_count, 2),
        )
    return AnswerView(
        kind=QueryKind.GRID,
        fields={"phi": (1.0,)},
        axes={"x": (0.0,)},
        shape=(1,),
    )


class Truth:
    def answer(self, query: AnswerQuery) -> AnswerView:
        return answer_for(query)


class ComponentMatcher:
    contract_id = "component-point-match-v1"
    fields = ("phi",)

    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events

    def prepare_target(self, case, truth, query):
        return MacroTarget(case.case_id, self.contract_id, query, truth)

    def match(self, case, prediction, target):
        if self.events is not None:
            self.events.append("match")
        if prediction.kind is not target.values.kind:
            raise ValueError("kind changed")
        if prediction.shape != target.values.shape:
            raise ValueError("shape changed")
        if prediction.points != target.values.points:
            raise ValueError("points changed")
        if prediction.fields != target.values.fields:
            raise ValueError("values changed")
        return MacroMatchResult(
            case_id=case.case_id,
            contract_id=self.contract_id,
            primary_error=0.0,
            metrics={"relative_error": 0.0},
            passed=True,
        )


class PublicSession:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def coefficient(self) -> float:
        return 1.0

    def close(self) -> None:
        self.events.append("session_close")


class PointHandle:
    def __init__(self, candidate, case, retained_method) -> None:
        self.candidate = candidate
        self.case = case
        self.retained_method = retained_method

    def evaluate(self, query: AnswerQuery) -> AnswerView:
        self.candidate.events.append("evaluate")
        self.candidate.queries.append(query)
        with self.candidate.test_case.assertRaisesRegex(
            RuntimeError,
            "does not expose",
        ):
            open_public_case_session(self.case)
        with self.candidate.test_case.assertRaisesRegex(
            RuntimeError,
            "no longer active",
        ):
            self.retained_method()
        self.candidate.session_closed_before_evaluate = True
        return answer_for(query)


class TwoPhaseCandidate:
    def __init__(self, events: list[str], test_case: unittest.TestCase) -> None:
        self.events = events
        self.test_case = test_case
        self.prepare_arguments = []
        self.queries = []
        self.session_closed_before_evaluate = False

    def prepare_solution(self, case: Case, solve_budget: float | None = None):
        self.events.append("prepare")
        self.prepare_arguments.append((case, solve_budget))
        session = open_public_case_session(case)
        retained_method = session.coefficient
        return PointHandle(self, case, retained_method)


class LegacyCandidate:
    def __init__(self) -> None:
        self.calls = []

    def predict_macros(self, case, query, budget=None):
        self.calls.append((case, query, budget))
        return answer_for(query)


class GridDualCandidate(LegacyCandidate):
    def __init__(self) -> None:
        super().__init__()
        self.prepare_calls = 0

    def prepare_solution(self, case, solve_budget=None):
        self.prepare_calls += 1
        raise AssertionError("GRID must not enter formal two-phase evaluation")


def case() -> Case:
    return Case(
        case_id="component-case",
        source=CaseSource.GENERATED,
        visibility=CaseVisibility.HIDDEN,
        public_data={"equation": "gkw-style"},
    )


def cohort(query: AnswerQuery, matcher: ComponentMatcher) -> PreparedMacroCohort:
    return PreparedMacroCohort.prepare(
        cohort_id=f"cohort-{query.kind.value}",
        records=(CaseRecord(case(), Truth()),),
        matcher=matcher,
        query=query,
    )


class MacroEvaluatorTwoPhaseTests(unittest.TestCase):
    def test_gkw_n2_two_phase_closes_session_and_separates_clocks(self):
        """Preparation is query-blind; GKW-style (N, 2) is scored later."""
        events = []
        matcher = ComponentMatcher(events)
        prepared = cohort(POINT_QUERY, matcher)
        events.clear()

        opened = []

        def session_factory(public_case):
            opened.append(public_case)
            events.append("session_open")
            return PublicSession(events)

        candidate = TwoPhaseCandidate(events, self)
        batch = ControllerMacroEvaluator(
            prepared,
            compute_budget=7.0,
            public_session_factory=session_factory,
            public_session_methods=("coefficient",),
            candidate_clock=SequenceClock(10.0, 11.5, 20.0, 20.6),
            controller_clock=SequenceClock(30.0, 31.25),
        ).evaluate("two-phase", candidate)

        self.assertTrue(batch.accepted)
        self.assertEqual(
            events,
            ["prepare", "session_open", "session_close", "evaluate", "match"],
        )
        self.assertEqual(len(candidate.prepare_arguments), 1)
        self.assertEqual(candidate.prepare_arguments[0][1], 7.0)
        self.assertEqual(candidate.queries, [POINT_QUERY])
        self.assertTrue(candidate.session_closed_before_evaluate)
        self.assertIsNot(
            candidate.prepare_arguments[0][0],
            prepared._prepared[0].public_case,
        )
        self.assertEqual(len(opened), 1)

        timing = batch.case_timings[0]
        self.assertEqual(timing.candidate_solve_wall_seconds, 1.5)
        self.assertAlmostEqual(timing.candidate_query_wall_seconds, 0.6)
        self.assertEqual(timing.candidate_query_point_count, 2)
        self.assertAlmostEqual(
            timing.candidate_query_wall_seconds_per_point,
            0.3,
        )
        self.assertEqual(timing.controller_truth_match_wall_seconds, 1.25)
        self.assertIsNone(timing.candidate_legacy_wall_seconds)
        self.assertTrue(timing.formal_timing_eligible)
        self.assertEqual(
            timing.hard_process_deadline_enforcement,
            "isolated_runner_required",
        )

        prediction = candidate.queries[0]
        self.assertIs(prediction.kind, QueryKind.POINTS)
        self.assertEqual(batch.scores[0].status, "valid")
        self.assertEqual(prepared._prepared[0].target.values.shape, (2, 2))
        self.assertEqual(
            len(prepared._prepared[0].target.values.fields["phi"]),
            4,
        )

    def test_prepare_over_budget_fails_before_query_reveal(self):
        events = []
        prepared = cohort(POINT_QUERY, ComponentMatcher())

        def session_factory(_public_case):
            events.append("session_open")
            return PublicSession(events)

        candidate = TwoPhaseCandidate(events, self)
        batch = ControllerMacroEvaluator(
            prepared,
            compute_budget=1.0,
            public_session_factory=session_factory,
            public_session_methods=("coefficient",),
            candidate_clock=SequenceClock(10.0, 11.25),
        ).evaluate("over-budget", candidate)

        self.assertFalse(batch.accepted)
        self.assertEqual(
            batch.scores[0].failures,
            ("candidate_solve_budget_exceeded",),
        )
        self.assertEqual(events, ["prepare", "session_open", "session_close"])
        self.assertEqual(candidate.queries, [])
        timing = batch.case_timings[0]
        self.assertEqual(timing.candidate_solve_wall_seconds, 1.25)
        self.assertIsNone(timing.candidate_query_wall_seconds)
        self.assertIsNone(timing.controller_truth_match_wall_seconds)
        self.assertFalse(timing.formal_timing_eligible)

    def test_point_legacy_predict_is_scored_but_timing_unassured(self):
        candidate = LegacyCandidate()
        batch = ControllerMacroEvaluator(
            cohort(POINT_QUERY, ComponentMatcher()),
            compute_budget=4.0,
            candidate_clock=SequenceClock(1.0, 1.4),
            controller_clock=SequenceClock(2.0, 2.2),
        ).evaluate("legacy-point", candidate)

        self.assertTrue(batch.accepted)
        self.assertEqual(len(candidate.calls), 1)
        self.assertEqual(candidate.calls[0][2], 4.0)
        timing = batch.case_timings[0]
        self.assertIs(
            timing.timing_assurance,
            TimingAssurance.LEGACY_ONE_SHOT_UNASSURED,
        )
        self.assertAlmostEqual(timing.candidate_legacy_wall_seconds, 0.4)
        self.assertIsNone(timing.candidate_solve_wall_seconds)
        self.assertIsNone(timing.candidate_query_wall_seconds)
        self.assertFalse(timing.formal_timing_eligible)

    def test_formal_point_query_rejects_legacy_before_query_is_revealed(self):
        candidate = LegacyCandidate()
        batch = ControllerMacroEvaluator(
            cohort(POINT_QUERY, ComponentMatcher()),
            compute_budget=4.0,
            query_seconds_per_point_limit=0.05,
        ).evaluate("legacy-formal-ineligible", candidate)

        self.assertFalse(batch.accepted)
        self.assertEqual(candidate.calls, [])
        self.assertEqual(
            batch.scores[0].failures,
            ("candidate_two_phase_required",),
        )
        timing = batch.case_timings[0]
        self.assertFalse(timing.formal_timing_eligible)
        self.assertIsNone(timing.candidate_legacy_wall_seconds)

    def test_formal_point_query_has_an_independent_latency_limit(self):
        events = []
        candidate = TwoPhaseCandidate(events, self)

        def session_factory(_public_case):
            events.append("session_open")
            return PublicSession(events)

        batch = ControllerMacroEvaluator(
            cohort(POINT_QUERY, ComponentMatcher(events)),
            compute_budget=4.0,
            query_seconds_per_point_limit=0.05,
            public_session_factory=session_factory,
            public_session_methods=("coefficient",),
            candidate_clock=SequenceClock(
                1.0,
                1.1,
                2.0,
                2.11,
            ),
        ).evaluate("deferred-query-solve", candidate)

        self.assertFalse(batch.accepted)
        self.assertEqual(
            batch.scores[0].failures,
            ("candidate_query_budget_exceeded",),
        )
        self.assertEqual(
            events,
            ["prepare", "session_open", "session_close", "evaluate"],
        )
        timing = batch.case_timings[0]
        self.assertAlmostEqual(
            timing.candidate_query_wall_seconds_per_point,
            0.055,
        )
        self.assertIsNone(timing.controller_truth_match_wall_seconds)
        self.assertTrue(timing.formal_timing_eligible)

    def test_formal_prepare_property_is_resolved_inside_solve_clock(self):
        events = []

        class Handle:
            def evaluate(self, query):
                events.append("evaluate")
                return answer_for(query)

        class Candidate:
            @property
            def prepare_solution(self):
                events.append("resolve_prepare")

                def resolved(case, solve_budget=None):
                    del case, solve_budget
                    events.append("prepare")
                    return Handle()

                return resolved

        clock_values = iter((1.0, 1.25, 2.0, 2.1))

        def clock():
            events.append("clock")
            return next(clock_values)

        batch = ControllerMacroEvaluator(
            cohort(POINT_QUERY, ComponentMatcher()),
            compute_budget=4.0,
            query_seconds_per_point_limit=1.0,
            candidate_clock=clock,
            controller_clock=SequenceClock(3.0, 3.1),
        ).evaluate("property-prepare", Candidate())

        self.assertTrue(batch.accepted)
        self.assertEqual(
            events,
            [
                "clock", "resolve_prepare", "prepare", "clock",
                "clock", "evaluate", "clock",
            ],
        )

    def test_grid_forces_explicit_legacy_path_even_if_prepare_exists(self):
        candidate = GridDualCandidate()
        batch = ControllerMacroEvaluator(
            cohort(GRID_QUERY, ComponentMatcher()),
            candidate_clock=SequenceClock(5.0, 5.25),
            controller_clock=SequenceClock(8.0, 8.1),
        ).evaluate("grid-compatibility", candidate)

        self.assertTrue(batch.accepted)
        self.assertEqual(candidate.prepare_calls, 0)
        self.assertEqual(len(candidate.calls), 1)
        timing = batch.case_timings[0]
        self.assertIs(timing.query_kind, QueryKind.GRID)
        self.assertIs(
            timing.timing_assurance,
            TimingAssurance.LEGACY_ONE_SHOT_UNASSURED,
        )
        self.assertFalse(timing.formal_timing_eligible)
        self.assertIsNone(timing.candidate_query_point_count)
        self.assertIsInstance(timing, MacroCaseTimingAudit)


if __name__ == "__main__":
    unittest.main()
