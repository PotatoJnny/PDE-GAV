from contextvars import Context
from dataclasses import fields
import unittest

from pde_agent_verifier_solver.sdk._candidate_timing import (
    bind_queue_wait_observer,
    bind_queue_wait_reader,
    controller_queue_wait,
    controller_queue_wait_seconds,
)
from pde_agent_verifier_solver.sdk.continuous_solution import (
    LegacyOneShotAdapter,
    PointQueryLatencyEligibilityContract,
    PointQueryResult,
    PreparedSolutionResult,
    SolveWallEligibilityContract,
    TimingAssurance,
    TwoPhaseEligibilityContract,
    measure_point_query,
    measure_prepare_solution,
)
from pde_agent_verifier_solver.sdk.models import (
    AnswerQuery,
    AnswerView,
    Case,
    CaseSource,
    CaseVisibility,
    QueryKind,
)


CASE = Case(
    case_id="case-1",
    source=CaseSource.GENERATED,
    visibility=CaseVisibility.HIDDEN,
    public_data={"domain": {"t": [0.0, 2.0], "x": [-1.0, 1.0]}},
)


class SequenceClock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class CandidateSelectedRepresentation:
    def __init__(self) -> None:
        self.queries: list[AnswerQuery] = []

    def evaluate(self, query: AnswerQuery) -> AnswerView:
        self.queries.append(query)
        values = tuple(point["t"] + 2.0 * point["x"] for point in query.points)
        return AnswerView(
            kind=QueryKind.POINTS,
            fields={
                field: values if field == "density" else tuple(-value for value in values)
                for field in query.fields
            },
            points=query.points,
            shape=(len(query.points),),
        )


class AdaptiveCandidate:
    def __init__(self) -> None:
        self.prepare_arguments: list[tuple[Case, float | None]] = []
        self.internal_dimensions = {"t": 37, "x": 211}
        self.prepared_representation: CandidateSelectedRepresentation | None = None

    def prepare_solution(
        self,
        case: Case,
        solve_budget: float | None = None,
    ) -> CandidateSelectedRepresentation:
        self.prepare_arguments.append((case, solve_budget))
        representation = CandidateSelectedRepresentation()
        self.prepared_representation = representation
        return representation


def point_query(point_count: int = 3) -> AnswerQuery:
    points = tuple(
        {"t": float(index) / point_count, "x": -0.8 + index / point_count}
        for index in range(point_count)
    )
    return AnswerQuery.at_points(fields=("density", "flux"), points=points)


def point_answer(point_count: int) -> AnswerView:
    points = tuple({"x": float(index)} for index in range(point_count))
    return AnswerView(
        kind=QueryKind.POINTS,
        fields={"u": tuple(float(index) for index in range(point_count))},
        points=points,
        shape=(point_count,),
    )


class ContinuousSolutionTests(unittest.TestCase):
    def test_two_phase_hides_query_then_supports_arbitrary_points(self) -> None:
        candidate = AdaptiveCandidate()
        prepared = measure_prepare_solution(
            candidate,
            CASE,
            5.0,
            clock=SequenceClock(10.0, 12.5),
        )

        self.assertEqual(candidate.prepare_arguments, [(CASE, 5.0)])
        self.assertEqual(candidate.internal_dimensions, {"t": 37, "x": 211})
        self.assertAlmostEqual(prepared.candidate_solve_wall_seconds, 2.5)
        self.assertTrue(prepared.formal_timing_assurance)

        query = point_query(3)
        evaluated = measure_point_query(
            prepared.handle,
            query,
            clock=SequenceClock(100.0, 100.75),
        )

        self.assertEqual(evaluated.answer.points, query.points)
        for actual, expected in zip(
            evaluated.answer.fields["density"],
            (-1.6, -0.6, 0.4),
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(evaluated.candidate_query_wall_seconds, 0.75)
        self.assertAlmostEqual(
            evaluated.candidate_query_wall_seconds_per_point,
            0.25,
        )
        assert candidate.prepared_representation is not None
        self.assertEqual(candidate.prepared_representation.queries, [query])

    def test_prepare_clock_includes_candidate_evaluate_property_resolution(self) -> None:
        events: list[str] = []

        class PropertyHandle:
            @property
            def evaluate(self):
                events.append("resolve_evaluate")

                def resolved(query: AnswerQuery) -> AnswerView:
                    return point_answer(len(query.points))

                return resolved

        class PropertyProgram:
            @property
            def prepare_solution(self):
                events.append("resolve_prepare")

                def resolved(case, solve_budget=None):
                    events.append("prepare")
                    return PropertyHandle()

                return resolved

        clock_values = iter((10.0, 12.0))

        def clock() -> float:
            events.append("clock")
            return next(clock_values)

        prepared = measure_prepare_solution(PropertyProgram(), CASE, clock=clock)

        self.assertEqual(
            events,
            ["clock", "resolve_prepare", "prepare", "resolve_evaluate", "clock"],
        )
        self.assertAlmostEqual(prepared.candidate_solve_wall_seconds, 2.0)

    def test_query_clock_includes_lazy_field_materialization(self) -> None:
        events: list[str] = []
        query = AnswerQuery.at_points(
            fields=("density",),
            points=({"t": 0.1}, {"t": 0.9}),
        )

        class LazyValues:
            def __iter__(self):
                events.append("materialize")
                yield 1.0
                yield 2.0

        class LazyHandle:
            def evaluate(self, revealed: AnswerQuery) -> AnswerView:
                events.append("evaluate")
                return AnswerView(
                    kind=QueryKind.POINTS,
                    fields={"density": LazyValues()},
                    points=revealed.points,
                    shape=(2,),
                )

        clock_values = iter((20.0, 23.0))

        def clock() -> float:
            events.append("clock")
            return next(clock_values)

        result = measure_point_query(LazyHandle(), query, clock=clock)

        self.assertEqual(
            events,
            ["clock", "evaluate", "materialize", "clock"],
        )
        self.assertEqual(result.answer.fields["density"], (1.0, 2.0))
        self.assertAlmostEqual(result.candidate_query_wall_seconds, 3.0)

    def test_grid_query_is_rejected_before_candidate_evaluation(self) -> None:
        handle = CandidateSelectedRepresentation()
        grid = AnswerQuery.grid(fields=("density",), t=7, x=19)

        with self.assertRaisesRegex(ValueError, "only QueryKind.POINTS"):
            measure_point_query(handle, grid, clock=SequenceClock(0.0, 1.0))

        self.assertEqual(handle.queries, [])

    def test_component_valued_point_answer_uses_flattened_shape(self) -> None:
        class ComplexComponentHandle:
            def evaluate(self, query: AnswerQuery) -> AnswerView:
                flattened = tuple(
                    component
                    for index in range(len(query.points))
                    for component in (float(index), -float(index))
                )
                return AnswerView(
                    kind=QueryKind.POINTS,
                    fields={field: flattened for field in query.fields},
                    points=query.points,
                    shape=(len(query.points), 2),
                )

        query = AnswerQuery.at_points(
            fields=("phi",),
            points=(
                {"t": 0.1, "field_row": 0.0},
                {"t": 0.7, "field_row": 2.0},
            ),
        )
        result = measure_point_query(
            ComplexComponentHandle(),
            query,
            clock=SequenceClock(3.0, 3.2),
        )

        self.assertEqual(result.answer.shape, (2, 2))
        self.assertEqual(result.answer.fields["phi"], (0.0, -0.0, 1.0, -1.0))
        self.assertEqual(result.point_count, 2)

    def test_solve_and_normalized_query_eligibility_are_independent(self) -> None:
        prepared = PreparedSolutionResult(
            handle=CandidateSelectedRepresentation(),
            candidate_solve_wall_seconds=4.0,
            solve_budget_seconds=10.0,
        )
        queries = (
            PointQueryResult(point_answer(3), 1.2, 3),
            PointQueryResult(point_answer(2), 0.8, 2),
        )
        contract = TwoPhaseEligibilityContract(
            solve=SolveWallEligibilityContract(3.0),
            query=PointQueryLatencyEligibilityContract(0.4),
        )

        result = contract.assess(prepared, queries)

        self.assertFalse(result.solve.eligible)
        self.assertTrue(result.query.eligible)
        self.assertAlmostEqual(result.query.candidate_query_wall_seconds, 2.0)
        self.assertEqual(result.query.point_count, 5)
        self.assertAlmostEqual(
            result.query.candidate_query_wall_seconds_per_point,
            0.4,
        )
        self.assertFalse(result.eligible)

        strict_query = PointQueryLatencyEligibilityContract(0.39).assess(queries)
        self.assertFalse(strict_query.eligible)
        self.assertTrue(SolveWallEligibilityContract(5.0).assess(prepared).eligible)

    def test_candidate_timing_excludes_controller_and_campaign_work(self) -> None:
        prepared = measure_prepare_solution(
            AdaptiveCandidate(),
            CASE,
            5.0,
            clock=SequenceClock(7.0, 8.0),
        )
        evaluated = measure_point_query(
            prepared.handle,
            point_query(2),
            clock=SequenceClock(20.0, 20.5),
        )

        controller_truth_wall_seconds = 30.0
        controller_matcher_wall_seconds = 40.0
        guidance_training_wall_seconds = 50.0
        generator_sample_wall_seconds = 60.0

        self.assertAlmostEqual(prepared.candidate_solve_wall_seconds, 1.0)
        self.assertAlmostEqual(evaluated.candidate_query_wall_seconds, 0.5)
        self.assertEqual(
            controller_truth_wall_seconds
            + controller_matcher_wall_seconds
            + guidance_training_wall_seconds
            + generator_sample_wall_seconds,
            180.0,
        )
        prepared_fields = {item.name for item in fields(PreparedSolutionResult)}
        query_fields = {item.name for item in fields(PointQueryResult)}
        self.assertEqual(
            prepared_fields,
            {
                "handle",
                "candidate_solve_wall_seconds",
                "solve_budget_seconds",
                "timing_assurance",
                "candidate_solve_queue_wait_seconds",
            },
        )
        self.assertEqual(
            query_fields,
            {
                "answer",
                "candidate_query_wall_seconds",
                "point_count",
                "candidate_query_queue_wait_seconds",
            },
        )
        excluded = {"truth", "matcher", "guidance", "generator", "sample"}
        self.assertFalse(
            any(
                excluded_fragment in field_name
                for field_name in prepared_fields | query_fields
                for excluded_fragment in excluded
            )
        )

    def test_prepare_excludes_queue_wait_but_charges_oracle_and_solver_work(self) -> None:
        elapsed = [0.0]
        cumulative_wait = [7.0]  # Earlier cases must not be deducted again.
        wait_started = [0.0]

        def observe(waiting: bool) -> None:
            if waiting:
                wait_started[0] = elapsed[0]
            else:
                cumulative_wait[0] += elapsed[0] - wait_started[0]

        class WaitingProgram:
            def prepare_solution(self, case, solve_budget=None):
                with controller_queue_wait():
                    elapsed[0] += 10.0  # Waiting to acquire the shared service.
                elapsed[0] += 3.0  # The oracle actually computes after acquisition.
                elapsed[0] += 2.0  # The candidate then performs its own solve.
                return CandidateSelectedRepresentation()

        with bind_queue_wait_observer(observe), bind_queue_wait_reader(
            lambda: cumulative_wait[0]
        ):
            prepared = measure_prepare_solution(
                WaitingProgram(), CASE, clock=lambda: elapsed[0]
            )

        self.assertEqual(prepared.candidate_solve_queue_wait_seconds, 10.0)
        self.assertEqual(prepared.candidate_solve_wall_seconds, 5.0)
        self.assertTrue(SolveWallEligibilityContract(5.0).assess(prepared).eligible)
        self.assertFalse(SolveWallEligibilityContract(4.9).assess(prepared).eligible)

    def test_query_excludes_queue_wait_and_preserves_per_point_eligibility(self) -> None:
        elapsed = [0.0]
        cumulative_wait = [4.0]
        wait_started = [0.0]

        def observe(waiting: bool) -> None:
            if waiting:
                wait_started[0] = elapsed[0]
            else:
                cumulative_wait[0] += elapsed[0] - wait_started[0]

        class WaitingHandle(CandidateSelectedRepresentation):
            def evaluate(self, query):
                elapsed[0] += 0.5  # Projection before a public-input request.
                with controller_queue_wait():
                    elapsed[0] += 8.0
                elapsed[0] += 1.0  # Actual public-oracle service remains charged.
                return super().evaluate(query)

        with bind_queue_wait_observer(observe), bind_queue_wait_reader(
            lambda: cumulative_wait[0]
        ):
            queried = measure_point_query(
                WaitingHandle(), point_query(3), clock=lambda: elapsed[0]
            )

        self.assertEqual(queried.candidate_query_queue_wait_seconds, 8.0)
        self.assertEqual(queried.candidate_query_wall_seconds, 1.5)
        self.assertEqual(queried.candidate_query_wall_seconds_per_point, 0.5)
        self.assertTrue(
            PointQueryLatencyEligibilityContract(0.5).assess((queried,)).eligible
        )

    def test_unbound_queue_marker_does_not_exempt_candidate_wall_time(self) -> None:
        class UnboundMarkerProgram:
            def prepare_solution(self, case, solve_budget=None):
                with controller_queue_wait():
                    return CandidateSelectedRepresentation()

        prepared = measure_prepare_solution(
            UnboundMarkerProgram(), CASE, clock=SequenceClock(10.0, 14.0)
        )
        self.assertEqual(prepared.candidate_solve_queue_wait_seconds, 0.0)
        self.assertEqual(prepared.candidate_solve_wall_seconds, 4.0)

    def test_queue_wait_bindings_are_isolated_and_restored_after_exceptions(self) -> None:
        outer: list[bool] = []
        inner: list[bool] = []
        self.assertEqual(controller_queue_wait_seconds(), 0.0)
        with bind_queue_wait_observer(outer.append), bind_queue_wait_reader(lambda: 3.0):
            self.assertEqual(controller_queue_wait_seconds(), 3.0)
            self.assertEqual(Context().run(controller_queue_wait_seconds), 0.0)

            def isolated_wait() -> None:
                with controller_queue_wait():
                    pass

            Context().run(isolated_wait)
            self.assertEqual(outer, [])
            with self.assertRaisesRegex(RuntimeError, "queue cancelled"):
                with bind_queue_wait_observer(inner.append), bind_queue_wait_reader(
                    lambda: 9.0
                ):
                    self.assertEqual(controller_queue_wait_seconds(), 9.0)
                    with controller_queue_wait():
                        raise RuntimeError("queue cancelled")
            self.assertEqual(inner, [True, False])
            self.assertEqual(controller_queue_wait_seconds(), 3.0)
            with controller_queue_wait():
                pass
            self.assertEqual(outer, [True, False])
        self.assertEqual(controller_queue_wait_seconds(), 0.0)
        with controller_queue_wait():
            pass
        self.assertEqual(outer, [True, False])

    def test_effective_wall_time_cannot_be_negative(self) -> None:
        with bind_queue_wait_reader(SequenceClock(0.0, 1.0000001)):
            prepared = measure_prepare_solution(
                AdaptiveCandidate(), CASE, clock=SequenceClock(0.0, 1.0)
            )
        self.assertEqual(prepared.candidate_solve_wall_seconds, 0.0)
        self.assertAlmostEqual(prepared.candidate_solve_queue_wait_seconds, 1.0000001)

    def test_legacy_adapter_is_explicitly_timing_unassured(self) -> None:
        calls: list[tuple[Case, AnswerQuery, float | None]] = []

        def legacy(
            case: Case,
            query: AnswerQuery,
            budget: float | None,
        ) -> AnswerView:
            calls.append((case, query, budget))
            values = tuple(point["x"] for point in query.points)
            return AnswerView(
                kind=QueryKind.POINTS,
                fields={field: values for field in query.fields},
                points=query.points,
                shape=(len(query.points),),
            )

        adapter = LegacyOneShotAdapter(legacy)
        prepared = measure_prepare_solution(
            adapter,
            CASE,
            9.0,
            clock=SequenceClock(1.0, 1.01),
        )

        self.assertEqual(calls, [])
        self.assertFalse(adapter.formal_timing_assurance)
        self.assertFalse(prepared.formal_timing_assurance)
        self.assertIs(
            prepared.timing_assurance,
            TimingAssurance.LEGACY_ONE_SHOT_UNASSURED,
        )

        query = AnswerQuery.at_points(
            fields=("u",),
            points=({"x": 0.1}, {"x": 0.9}),
        )
        evaluated = measure_point_query(
            prepared.handle,
            query,
            clock=SequenceClock(5.0, 7.0),
        )
        self.assertEqual(calls, [(CASE, query, 9.0)])
        self.assertAlmostEqual(
            evaluated.candidate_query_wall_seconds_per_point,
            1.0,
        )

        generous_contract = TwoPhaseEligibilityContract(
            solve=SolveWallEligibilityContract(100.0),
            query=PointQueryLatencyEligibilityContract(100.0),
        )
        assessment = generous_contract.assess(prepared, (evaluated,))
        self.assertTrue(assessment.solve.eligible)
        self.assertTrue(assessment.query.eligible)
        self.assertFalse(assessment.formal_timing_assurance)
        self.assertFalse(assessment.eligible)


if __name__ == "__main__":
    unittest.main()
