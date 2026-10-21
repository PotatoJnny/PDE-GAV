import unittest
from dataclasses import replace

from pde_agent_verifier_solver.engine.campaign import EvaluatorPolicy
from pde_agent_verifier_solver.engine.candidate_case import candidate_case_id
from pde_agent_verifier_solver.engine.evaluator import (
    AgentEvaluator,
    EvaluationRequest,
    HiddenVerifier,
)
from pde_agent_verifier_solver.plugins.toy import (
    ToyPoissonProblem,
    ToySineSolver,
    ToySolverTask,
    ToySolverEvaluation,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    Artifact,
    DatasetInput,
    DatasetKind,
    DevelopmentCoordinateMode,
    EvaluationPhase,
    EvaluationPlan,
    QueryKind,
    TaskKind,
    aggregate_case_evaluations,
)


class FactoryOnlyToyTask(ToySolverTask):
    """Fails loudly if the engine bypasses create_evaluation()."""

    def evaluate(self, artifact, case, truth, query):
        raise AssertionError("legacy TaskProgram.evaluate() must not be called")


class RecordingEvaluation:
    program_id = "recording-evaluation"

    def __init__(self, query: AnswerQuery) -> None:
        self.query = query
        self.plan_calls: list[tuple[str, EvaluationPhase, int]] = []
        self.evaluation_calls: list[tuple[str, str, EvaluationPlan]] = []
        self.delegate = ToySolverEvaluation(query)

    def make_plan(self, case, phase, settings, seed, request=None):
        self.plan_calls.append((case.case_id, phase, seed))
        return EvaluationPlan(
            phase=phase,
            query=request or self.query,
            seed=seed,
            settings=settings,
        )

    def evaluate(self, artifact, case, truth, plan):
        self.evaluation_calls.append((artifact.artifact_id, case.case_id, plan))
        return self.delegate.evaluate(artifact, case, truth, plan)

    def aggregate(self, artifact, case_results, plans):
        return aggregate_case_evaluations(artifact, case_results, plans)


class EvaluatorContractTests(unittest.TestCase):
    def test_answer_query_rejects_a_string_kind_before_truth_policy(self) -> None:
        with self.assertRaisesRegex(TypeError, "kind must be a QueryKind"):
            AnswerQuery(
                kind="points",
                fields=("u",),
                points=({"x": 0.5},),
            )

    def setUp(self) -> None:
        self.problem = ToyPoissonProblem()
        self.dataset = self.problem.prepare_dataset(
            DatasetInput("dataset", DatasetKind.NUMERICAL)
        )
        self.task = ToySolverTask()
        self.artifact = Artifact(
            artifact_id="solver",
            kind=TaskKind.SOLVER,
            name="test solver",
            program=ToySineSolver(2),
            summary="test",
            usage="solve",
            round_index=1,
            agent_id="agent",
        )

    def test_returns_pointwise_truth_without_exposing_continuous_truth(self) -> None:
        evaluator = AgentEvaluator(
            self.dataset.development_cases,
            self.task,
            EvaluatorPolicy(
                AnswerQuery.grid(fields=("u",), x=17),
                allow_grid_truth=True,
                allow_point_truth=True,
                development_coordinate_mode=(
                    DevelopmentCoordinateMode.AGENT_SELECTED_QUERY
                ),
            ),
        )
        case_id = candidate_case_id(
            self.dataset.development_cases[0].case.case_id
        )
        query = AnswerQuery.at_points(
            fields=("u",),
            points=({"x": 0.25}, {"x": 0.5}, {"x": 0.75}),
        )
        response = evaluator.evaluate(
            self.artifact,
            EvaluationRequest((case_id,), query=query, return_truth=True),
        )

        truth = response.cases[0].truth
        prediction = response.cases[0].prediction
        self.assertIsNotNone(truth)
        self.assertIsNotNone(prediction)
        assert truth is not None
        self.assertEqual(truth.kind, QueryKind.POINTS)
        self.assertEqual(len(truth.fields["u"]), 3)
        self.assertFalse(hasattr(truth, "answer"))

    def test_hidden_case_id_is_not_available_to_agent(self) -> None:
        evaluator = AgentEvaluator(
            self.dataset.development_cases,
            self.task,
            EvaluatorPolicy(AnswerQuery.grid(fields=("u",), x=17)),
        )
        hidden_id = self.dataset.hidden_cases[0].case.case_id
        with self.assertRaisesRegex(KeyError, "not available"):
            evaluator.evaluate(self.artifact, EvaluationRequest((hidden_id,)))

    def test_agent_evaluator_rejects_hidden_records_at_construction(self) -> None:
        with self.assertRaisesRegex(ValueError, "only accepts development Cases"):
            AgentEvaluator(
                self.dataset.hidden_cases,
                self.task,
                EvaluatorPolicy(AnswerQuery.grid(fields=("u",), x=17)),
            )

    def test_hidden_verifier_rejects_development_records(self) -> None:
        verifier = HiddenVerifier(
            self.task,
            AnswerQuery.grid(fields=("u",), x=17),
        )
        with self.assertRaisesRegex(ValueError, "only accepts hidden Cases"):
            verifier.evaluate(self.artifact, self.dataset.development_cases)

    def test_task_factory_is_the_primary_evaluation_entrypoint(self) -> None:
        evaluator = AgentEvaluator(
            self.dataset.development_cases,
            FactoryOnlyToyTask(),
            EvaluatorPolicy(AnswerQuery.grid(fields=("u",), x=17)),
        )
        case_id = candidate_case_id(
            self.dataset.development_cases[0].case.case_id
        )

        response = evaluator.evaluate(
            self.artifact,
            EvaluationRequest((case_id,)),
        )

        self.assertEqual(
            evaluator.evaluation_program.program_id,
            "toy-full-state-solver-evaluation",
        )
        self.assertEqual(len(response.cases), 1)
        self.assertIsNone(response.cases[0].truth)
        self.assertIsNone(response.cases[0].prediction)

    def test_default_response_exposes_case_scores_without_answers(self) -> None:
        query = AnswerQuery.grid(fields=("u",), x=17)
        evaluator = AgentEvaluator(
            self.dataset.development_cases,
            self.task,
            EvaluatorPolicy(query, allow_grid_truth=False),
        )
        case_id = candidate_case_id(
            self.dataset.development_cases[0].case.case_id
        )

        response = evaluator.evaluate(
            self.artifact,
            EvaluationRequest((case_id,)),
        )

        self.assertEqual(response.artifact_id, self.artifact.artifact_id)
        self.assertGreaterEqual(response.primary_error, 0.0)
        self.assertTrue(response.metrics)
        self.assertEqual(len(response.cases), 1)
        self.assertIsNone(response.cases[0].truth)
        self.assertIsNone(response.cases[0].prediction)

    def test_no_truth_batch_preserves_per_case_errors_and_public_ids(self) -> None:
        records = self.dataset.development_cases[:3]
        case_ids = tuple(candidate_case_id(record.case.case_id) for record in records)
        second_artifact = replace(self.artifact, artifact_id="second-solver")
        policy = EvaluatorPolicy(
            AnswerQuery.grid(fields=("u",), x=17),
            allow_grid_truth=False,
            allow_point_truth=False,
            development_coordinate_mode=DevelopmentCoordinateMode.USER_FIXED_QUERY,
        )
        evaluator = AgentEvaluator(records, self.task, policy)
        responses = evaluator.evaluate_many(
            (self.artifact, second_artifact), EvaluationRequest(case_ids)
        )
        for response in responses:
            self.assertEqual(tuple(row.case_id for row in response.cases), case_ids)
            self.assertEqual(len(response.cases), len(records))
            for row, record in zip(response.cases, records):
                self.assertNotEqual(row.case_id, record.case.case_id)
                self.assertIsNone(row.truth)
                self.assertIsNone(row.prediction)
                single = evaluator.evaluate(self.artifact, EvaluationRequest((row.case_id,)))
                self.assertEqual(row.primary_error, single.primary_error)
                self.assertEqual(row.metrics, single.cases[0].metrics)
            self.assertAlmostEqual(
                response.primary_error,
                sum(row.primary_error for row in response.cases) / len(records),
            )

    def test_truth_capabilities_gate_only_detailed_responses(self) -> None:
        queries_and_policies = (
            (
                AnswerQuery.grid(fields=("u",), x=17),
                EvaluatorPolicy(
                    AnswerQuery.grid(fields=("u",), x=17),
                    allow_grid_truth=False,
                    development_coordinate_mode=(
                        DevelopmentCoordinateMode.AGENT_SELECTED_QUERY
                    ),
                ),
                "grid truth views are disabled",
            ),
            (
                AnswerQuery.at_points(fields=("u",), points=({"x": 0.5},)),
                EvaluatorPolicy(
                    AnswerQuery.at_points(fields=("u",), points=({"x": 0.5},)),
                    allow_point_truth=False,
                    development_coordinate_mode=(
                        DevelopmentCoordinateMode.AGENT_SELECTED_QUERY
                    ),
                ),
                "point truth views are disabled",
            ),
        )
        case_id = candidate_case_id(
            self.dataset.development_cases[0].case.case_id
        )

        for query, policy, message in queries_and_policies:
            with self.subTest(kind=query.kind):
                evaluator = AgentEvaluator(
                    self.dataset.development_cases,
                    self.task,
                    policy,
                )
                aggregate = evaluator.evaluate(
                    self.artifact,
                    EvaluationRequest((case_id,), query=query),
                )
                self.assertEqual(len(aggregate.cases), 1)
                self.assertIsNone(aggregate.cases[0].truth)
                self.assertIsNone(aggregate.cases[0].prediction)

                with self.assertRaisesRegex(PermissionError, message):
                    evaluator.evaluate(
                        self.artifact,
                        EvaluationRequest(
                            (case_id,),
                            query=query,
                            return_truth=True,
                        ),
                    )

    def test_evaluate_many_reuses_exact_plans_for_every_artifact(self) -> None:
        query = AnswerQuery.grid(fields=("u",), x=17)
        evaluation = RecordingEvaluation(query)
        records = self.dataset.development_cases[:2]
        case_ids = tuple(
            candidate_case_id(record.case.case_id) for record in records
        )
        second_artifact = replace(
            self.artifact,
            artifact_id="solver-2",
            name="second test solver",
        )
        evaluator = AgentEvaluator(
            records,
            evaluation,
            EvaluatorPolicy(
                query,
                allow_grid_truth=True,
                development_coordinate_mode=(
                    DevelopmentCoordinateMode.USER_FIXED_QUERY
                ),
            ),
            seed=41,
        )

        responses = evaluator.evaluate_many(
            (self.artifact, second_artifact),
            EvaluationRequest(case_ids, return_truth=True),
        )

        self.assertEqual([response.artifact_id for response in responses], ["solver", "solver-2"])
        self.assertEqual(len(evaluation.plan_calls), 2)
        self.assertEqual([call[2] for call in evaluation.plan_calls], [41, 42])
        self.assertEqual(len(evaluation.evaluation_calls), 4)
        self.assertIs(evaluation.evaluation_calls[0][2], evaluation.evaluation_calls[2][2])
        self.assertIs(evaluation.evaluation_calls[1][2], evaluation.evaluation_calls[3][2])
        self.assertTrue(all(len(response.cases) == 2 for response in responses))

    def test_plan_seed_stream_is_fresh_and_deterministic_for_case_subsets(self) -> None:
        query = AnswerQuery.grid(fields=("u",), x=17)
        records = self.dataset.development_cases[:3]
        requests = (
            EvaluationRequest((candidate_case_id(records[2].case.case_id),)),
            EvaluationRequest(
                (
                    candidate_case_id(records[0].case.case_id),
                    candidate_case_id(records[2].case.case_id),
                ),
            ),
        )

        streams = []
        for _ in range(2):
            evaluation = RecordingEvaluation(query)
            evaluator = AgentEvaluator(
                records,
                evaluation,
                EvaluatorPolicy(
                    query,
                    development_coordinate_mode=(
                        DevelopmentCoordinateMode.USER_FIXED_QUERY
                    ),
                ),
                seed=73,
            )
            for request in requests:
                evaluator.evaluate(self.artifact, request)
            streams.append(evaluation.plan_calls)

        self.assertEqual(streams[0], streams[1])
        self.assertEqual([call[2] for call in streams[0]], [73, 74, 75])
        self.assertEqual(
            [call[0] for call in streams[0]],
            [
                records[2].case.case_id,
                records[0].case.case_id,
                records[2].case.case_id,
            ],
        )

    def test_controller_seed_provider_records_each_fresh_development_call(self) -> None:
        query = AnswerQuery.grid(fields=("u",), x=17)
        records = self.dataset.development_cases[:2]
        evaluation = RecordingEvaluation(query)
        requested: list[tuple[int, int, str]] = []

        def seed_for(call_index: int, case_index: int, case_id: str) -> int:
            requested.append((call_index, case_index, case_id))
            return 1_000 + call_index * 10 + case_index

        evaluator = AgentEvaluator(
            records,
            evaluation,
            EvaluatorPolicy(
                query,
                development_coordinate_mode=(
                    DevelopmentCoordinateMode.USER_FIXED_QUERY
                ),
            ),
            plan_seed_provider=seed_for,
        )
        first_raw_id, second_raw_id = (
            record.case.case_id for record in records
        )
        first_id = candidate_case_id(first_raw_id)
        second_id = candidate_case_id(second_raw_id)

        evaluator.evaluate(
            self.artifact,
            EvaluationRequest((second_id,)),
        )
        evaluator.evaluate_many(
            (self.artifact, replace(self.artifact, artifact_id="solver-2")),
            EvaluationRequest((first_id, second_id)),
        )

        self.assertEqual(
            requested,
            [
                (1, 0, second_raw_id),
                (2, 0, first_raw_id),
                (2, 1, second_raw_id),
            ],
        )
        self.assertEqual(
            [call[2] for call in evaluation.plan_calls],
            [1_010, 1_020, 1_021],
        )

    def test_private_random_points_are_fresh_reproducible_and_not_agent_selected(self) -> None:
        query = AnswerQuery.grid(fields=("u",), x=17)
        policy = EvaluatorPolicy(
            query,
            allow_point_truth=True,
            development_coordinate_mode=(
                DevelopmentCoordinateMode.PRIVATE_RANDOM_POINTS
            ),
            development_point_count=5,
        )
        case_id = candidate_case_id(
            self.dataset.development_cases[0].case.case_id
        )

        first_evaluator = AgentEvaluator(
            self.dataset.development_cases,
            self.task,
            policy,
            seed=123,
        )
        first = first_evaluator.evaluate(
            self.artifact,
            EvaluationRequest((case_id,), return_truth=True),
        )
        second = first_evaluator.evaluate(
            self.artifact,
            EvaluationRequest((case_id,), return_truth=True),
        )
        repeated = AgentEvaluator(
            self.dataset.development_cases,
            self.task,
            policy,
            seed=123,
        ).evaluate(
            self.artifact,
            EvaluationRequest((case_id,), return_truth=True),
        )

        first_points = first.cases[0].truth.points
        self.assertEqual(len(first_points), 5)
        self.assertNotEqual(first_points, second.cases[0].truth.points)
        self.assertEqual(first_points, repeated.cases[0].truth.points)
        with self.assertRaisesRegex(PermissionError, "cannot choose coordinates"):
            first_evaluator.evaluate(
                self.artifact,
                EvaluationRequest((case_id,), query=query),
            )

    def test_one_candidate_cannot_mutate_the_shared_private_plan(self) -> None:
        seen_by_second: list[tuple[dict[str, float], ...]] = []
        mutation_blocked: list[bool] = []

        class MutationAttempt:
            def solve(inner_self, case, query, budget=None):
                try:
                    query.points[0]["x"] = 999.0
                except TypeError:
                    mutation_blocked.append(True)
                return ToySineSolver(100).solve(case, query, budget)

        class Recorder:
            def solve(inner_self, case, query, budget=None):
                seen_by_second.append(
                    tuple(dict(point) for point in query.points)
                )
                return ToySineSolver(100).solve(case, query, budget)

        artifacts = (
            replace(self.artifact, artifact_id="mutation-attempt", program=MutationAttempt()),
            replace(self.artifact, artifact_id="recorder", program=Recorder()),
        )
        case_id = candidate_case_id(
            self.dataset.development_cases[0].case.case_id
        )
        evaluator = AgentEvaluator(
            self.dataset.development_cases,
            self.task,
            EvaluatorPolicy(
                AnswerQuery.grid(fields=("u",), x=17),
                development_point_count=5,
            ),
            seed=123,
        )

        evaluator.evaluate_many(artifacts, EvaluationRequest((case_id,)))

        self.assertEqual(mutation_blocked, [True])
        self.assertEqual(len(seen_by_second), 1)
        self.assertTrue(
            all(0.0 <= point["x"] <= 1.0 for point in seen_by_second[0])
        )

    def test_user_fixed_query_is_visible_but_cannot_be_overridden(self) -> None:
        query = AnswerQuery.grid(fields=("u",), x=17)
        evaluator = AgentEvaluator(
            self.dataset.development_cases,
            self.task,
            EvaluatorPolicy(
                query,
                allow_grid_truth=True,
                development_coordinate_mode=(
                    DevelopmentCoordinateMode.USER_FIXED_QUERY
                ),
            ),
            seed=51,
        )
        case_id = candidate_case_id(
            self.dataset.development_cases[0].case.case_id
        )

        self.assertEqual(evaluator.fixed_query, query)
        with self.assertRaises(TypeError):
            evaluator.fixed_query.resolution["x"] = 9
        first = evaluator.evaluate(
            self.artifact,
            EvaluationRequest((case_id,), return_truth=True),
        )
        second = evaluator.evaluate(
            self.artifact,
            EvaluationRequest((case_id,), query=query, return_truth=True),
        )
        self.assertEqual(first.cases[0].truth.axes, second.cases[0].truth.axes)
        with self.assertRaisesRegex(PermissionError, "cannot override"):
            evaluator.evaluate(
                self.artifact,
                EvaluationRequest(
                    (case_id,),
                    query=AnswerQuery.grid(fields=("u",), x=9),
                ),
            )

    def test_private_mode_fails_closed_without_a_task_sampler(self) -> None:
        query = AnswerQuery.grid(fields=("u",), x=17)
        evaluator = AgentEvaluator(
            self.dataset.development_cases,
            RecordingEvaluation(query),
            EvaluatorPolicy(query),
        )
        case_id = candidate_case_id(
            self.dataset.development_cases[0].case.case_id
        )

        with self.assertRaisesRegex(TypeError, "does not support private random"):
            evaluator.evaluate(self.artifact, EvaluationRequest((case_id,)))

    def test_custom_task_cannot_violate_the_selected_coordinate_mode(self) -> None:
        fixed = AnswerQuery.grid(fields=("u",), x=17)
        other = AnswerQuery.grid(fields=("u",), x=9)
        case_id = candidate_case_id(
            self.dataset.development_cases[0].case.case_id
        )

        class IgnoresFixedQuery(RecordingEvaluation):
            def make_plan(inner_self, case, phase, settings, seed, request=None):
                return EvaluationPlan(
                    phase=phase,
                    query=other,
                    seed=seed,
                    settings=settings,
                )

        fixed_evaluator = AgentEvaluator(
            self.dataset.development_cases,
            IgnoresFixedQuery(fixed),
            EvaluatorPolicy(
                fixed,
                development_coordinate_mode=(
                    DevelopmentCoordinateMode.USER_FIXED_QUERY
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "changed the selected coordinate"):
            fixed_evaluator.evaluate(
                self.artifact,
                EvaluationRequest((case_id,)),
            )

        class ChangesPrivateSeed(ToySolverEvaluation):
            def make_private_development_plan(
                inner_self,
                case,
                settings,
                seed,
            ):
                plan = super().make_private_development_plan(case, settings, seed)
                return EvaluationPlan(
                    phase=plan.phase,
                    query=plan.query,
                    seed=seed + 1,
                    settings=plan.settings,
                )

        private_evaluator = AgentEvaluator(
            self.dataset.development_cases,
            ChangesPrivateSeed(fixed),
            EvaluatorPolicy(fixed, development_point_count=5),
        )
        with self.assertRaisesRegex(ValueError, "changed its allocated seed"):
            private_evaluator.evaluate(
                self.artifact,
                EvaluationRequest((case_id,)),
            )

    def test_hidden_seed_provider_runs_once_per_frozen_case_plan(self) -> None:
        query = AnswerQuery.grid(fields=("u",), x=17)
        records = self.dataset.hidden_cases[:2]
        evaluation = RecordingEvaluation(query)
        requested: list[tuple[int, str]] = []

        def seed_for(case_index: int, case_id: str) -> int:
            requested.append((case_index, case_id))
            return 2_000 + case_index

        verifier = HiddenVerifier(
            evaluation,
            query,
            plan_seed_provider=seed_for,
        )
        verifier.evaluate(self.artifact, records)
        verifier.evaluate(replace(self.artifact, artifact_id="solver-2"), records)

        self.assertEqual(
            requested,
            [(index, record.case.case_id) for index, record in enumerate(records)],
        )
        self.assertEqual(
            [call[2] for call in evaluation.plan_calls],
            [2_000, 2_001],
        )

    def test_hidden_verifier_freezes_plans_for_all_candidates(self) -> None:
        query = AnswerQuery.grid(fields=("u",), x=17)
        evaluation = RecordingEvaluation(query)
        verifier = HiddenVerifier(evaluation, query, seed=41)
        records = self.dataset.hidden_cases[:2]

        verifier.evaluate(self.artifact, records)
        verifier.evaluate(self.artifact, records)

        self.assertEqual(len(evaluation.plan_calls), 2)
        self.assertEqual(
            [phase for _case_id, phase, _seed in evaluation.plan_calls],
            [EvaluationPhase.OFFICIAL, EvaluationPhase.OFFICIAL],
        )
        self.assertEqual(
            [seed for _case_id, _phase, seed in evaluation.plan_calls],
            [41, 42],
        )

    def test_hidden_verifier_materializes_stateful_truth_once_per_case(self) -> None:
        class CountingTruth:
            def __init__(inner_self, delegate) -> None:
                inner_self.delegate = delegate
                inner_self.calls = 0

            def answer(inner_self, query):
                inner_self.calls += 1
                return inner_self.delegate.answer(query)

        query = AnswerQuery.grid(fields=("u",), x=17)
        for phase in (EvaluationPhase.OFFICIAL, EvaluationPhase.TERMINAL):
            with self.subTest(phase=phase):
                truth = CountingTruth(self.dataset.hidden_cases[0].truth)
                record = replace(self.dataset.hidden_cases[0], truth=truth)
                verifier = HiddenVerifier(self.task, query, phase=phase)

                first = verifier.evaluate(self.artifact, (record,))
                second = verifier.evaluate(
                    replace(self.artifact, artifact_id=f"solver-{phase.value}"),
                    (record,),
                )

                self.assertEqual(truth.calls, 1)
                self.assertEqual(first.primary_error, second.primary_error)

    def test_terminal_verifier_marks_terminal_phase(self) -> None:
        query = AnswerQuery.grid(fields=("u",), x=17)
        evaluation = RecordingEvaluation(query)
        verifier = HiddenVerifier(
            evaluation,
            query,
            phase=EvaluationPhase.TERMINAL,
        )

        verifier.evaluate(self.artifact, self.dataset.hidden_cases[:1])

        self.assertEqual(evaluation.plan_calls[0][1], EvaluationPhase.TERMINAL)


if __name__ == "__main__":
    unittest.main()
