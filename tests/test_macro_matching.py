import math
import unittest

from pde_agent_verifier_solver.engine.campaign import EvaluatorPolicy
from pde_agent_verifier_solver.engine.candidate_case import candidate_case_id
from pde_agent_verifier_solver.engine.evaluator import (
    AgentEvaluator,
    EvaluationRequest,
)
from pde_agent_verifier_solver.plugins.registry import get_plugin
from pde_agent_verifier_solver.plugins.vpfp.macro_matching import (
    VPFP_MACRO_FIELDS,
    VPFPMacroMatcher,
)
from pde_agent_verifier_solver.plugins.vpfp.tasks import (
    VPFPObservableTask,
    VPFPReducedDynamicsTask,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Artifact,
    Case,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    DevelopmentCoordinateMode,
    MacroMatchResult,
    MacroTarget,
    QueryKind,
    TaskKind,
    evaluate_macro_prediction,
    materialize_macro_target,
)


class _Truth:
    def __init__(self, view: AnswerView) -> None:
        self.view = view
        self.calls = 0

    def answer(self, query: AnswerQuery) -> AnswerView:
        self.calls += 1
        return self.view


class _Observable:
    def __init__(self, prediction: AnswerView) -> None:
        self.prediction = prediction

    def predict_observables(self, case, query, budget=None) -> AnswerView:
        return self.prediction


class _Reduced:
    def __init__(self, prediction: AnswerView) -> None:
        self.prediction = prediction

    def initialize(self, case):
        return {"case_id": case.case_id}

    def advance(self, state, query_times, budget=None):
        return state, tuple(query_times)

    def observe(self, trajectory, query) -> AnswerView:
        return self.prediction


def _query() -> AnswerQuery:
    return AnswerQuery.grid(fields=VPFP_MACRO_FIELDS, t=2)


def _view(
    *,
    rho=(1.0, 1.0),
    momentum_x=(1.0, 1.0),
    momentum_y=(1.0, 1.0),
    kinetic_energy=(1.0, 1.0),
    kind=QueryKind.GRID,
) -> AnswerView:
    return AnswerView(
        kind=kind,
        fields={
            "rho": tuple(rho),
            "momentum_x": tuple(momentum_x),
            "momentum_y": tuple(momentum_y),
            "kinetic_energy": tuple(kinetic_energy),
        },
        axes={"t": (0.0, 1.0)} if kind is QueryKind.GRID else {},
        points=(
            ({"t": 0.0}, {"t": 1.0})
            if kind is QueryKind.POINTS
            else ()
        ),
        shape=(2,),
    )


def _case() -> Case:
    return Case(
        case_id="vpfp-case",
        source=CaseSource.GENERATED,
        visibility=CaseVisibility.DEVELOPMENT,
        public_data={
            "observations": {
                "rho": (999.0, 999.0),
                "momentum_x": (999.0, 999.0),
                "momentum_y": (999.0, 999.0),
                "kinetic_energy": (999.0, 999.0),
            }
        },
    )


def _artifact(kind: TaskKind, program: object) -> Artifact:
    return Artifact(
        artifact_id=f"candidate-{kind.value}",
        kind=kind,
        name="candidate",
        program=program,
        summary="test candidate",
        usage="test",
        round_index=0,
        agent_id="test-agent",
        metadata={"state_dimension": 1},
    )


class MacroMatchingTests(unittest.TestCase):
    def test_agent_briefs_publish_the_coordinate_metadata_contract(self) -> None:
        tasks = (VPFPObservableTask(), VPFPReducedDynamicsTask())
        for task in tasks:
            primary_metric = next(
                field for field in task.setup_fields()
                if field.name == "primary_metric"
            )
            self.assertEqual(
                primary_metric.default,
                "vpfp_macro_match_v1_relative_l2",
            )
            brief = task.agent_brief({})
            self.assertIn("case.public_data['arrays']", brief)
            self.assertIn("query.resolution", brief)
            self.assertIn("shape in (t, x, y) order", brief)
            self.assertIn("copy query.points exactly", brief)

    def test_macro_target_binds_case_contract_query_and_weights(self) -> None:
        target = MacroTarget(
            "vpfp-case",
            "contract",
            _query(),
            _view(),
            sample_weights=(1.0, 2.0),
        )
        self.assertEqual(target.sample_weights, (1.0, 2.0))

        point_values = _view(kind=QueryKind.POINTS)
        with self.assertRaisesRegex(ValueError, "query kinds"):
            MacroTarget("vpfp-case", "contract", _query(), point_values)
        with self.assertRaisesRegex(ValueError, "align"):
            MacroTarget(
                "vpfp-case",
                "contract",
                _query(),
                _view(),
                sample_weights=(1.0,),
            )
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            MacroTarget(
                "vpfp-case",
                "contract",
                _query(),
                _view(),
                sample_weights=(1.0, -1.0),
            )

    def test_macro_result_rejects_invalid_numbers_and_gate_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            MacroMatchResult("case", "contract", math.nan)
        with self.assertRaisesRegex(ValueError, "bool or None"):
            MacroMatchResult("case", "contract", 0.0, passed="yes")
        with self.assertRaisesRegex(ValueError, "cannot contain failures"):
            MacroMatchResult(
                "case",
                "contract",
                0.0,
                passed=True,
                failures=("unexpected",),
            )

    def test_target_materialization_calls_truth_once_without_retaining_it(self) -> None:
        truth = _Truth(_view())
        record = CaseRecord(_case(), truth)

        target = materialize_macro_target(
            VPFPMacroMatcher(),
            record,
            _query(),
        )

        self.assertEqual(truth.calls, 1)
        self.assertEqual(target.case_id, record.case.case_id)
        self.assertIs(target.values, truth.view)
        self.assertFalse(hasattr(target, "truth"))
        self.assertFalse(hasattr(target, "record"))

    def test_public_data_observations_do_not_define_the_macro_target(self) -> None:
        matcher = VPFPMacroMatcher()
        case = _case()
        truth = _view()
        exact = evaluate_macro_prediction(
            matcher=matcher,
            case=case,
            prediction=truth,
            truth=truth,
            query=_query(),
        )
        changed_hidden_truth = evaluate_macro_prediction(
            matcher=matcher,
            case=case,
            prediction=truth,
            truth=_view(rho=(2.0, 2.0)),
            query=_query(),
        )

        self.assertEqual(exact.primary_error, 0.0)
        self.assertGreater(changed_hidden_truth.primary_error, 0.0)
        self.assertNotIn("observations", exact.metrics)

    def test_vector_momentum_is_one_physical_macro_group(self) -> None:
        evaluation = evaluate_macro_prediction(
            matcher=VPFPMacroMatcher(),
            case=_case(),
            prediction=_view(momentum_x=(2.0, 2.0)),
            truth=_view(),
            query=_query(),
        )
        expected_momentum_error = 1.0 / math.sqrt(2.0)

        self.assertAlmostEqual(
            evaluation.metrics["momentum_relative_l2"],
            expected_momentum_error,
        )
        self.assertAlmostEqual(
            evaluation.primary_error,
            expected_momentum_error / 3.0,
        )
        self.assertEqual(
            evaluation.metrics["vpfp_macro_match_v1_relative_l2"],
            evaluation.primary_error,
        )
        self.assertAlmostEqual(
            evaluation.metrics["legacy_four_scalar_macro_relative_l2"],
            0.25,
        )
        self.assertIsNotNone(evaluation.macro_match)
        self.assertEqual(
            evaluation.metrics["macro_match_error"],
            evaluation.primary_error,
        )

    def test_optional_match_gate_is_explicit_and_fail_closed(self) -> None:
        matcher = VPFPMacroMatcher(maximum_relative_l2=0.1)
        target = matcher.prepare_target(_case(), _view(), _query())

        exact = matcher.match(_case(), _view(), target)
        failed = matcher.match(
            _case(),
            _view(rho=(2.0, 2.0)),
            target,
        )

        self.assertIs(exact.passed, True)
        self.assertEqual(exact.failures, ())
        self.assertIs(failed.passed, False)
        self.assertTrue(failed.failures)

    def test_malformed_prediction_fails_before_scoring(self) -> None:
        matcher = VPFPMacroMatcher()
        target = matcher.prepare_target(_case(), _view(), _query())
        missing = AnswerView(
            QueryKind.GRID,
            fields={"rho": (1.0, 1.0)},
            shape=(2,),
        )
        nonfinite = _view(rho=(math.inf, 1.0))

        with self.assertRaisesRegex(ValueError, "fields"):
            matcher.match(_case(), missing, target)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            matcher.match(_case(), nonfinite, target)

    def test_observable_and_reduced_tasks_share_macro_score_semantics(self) -> None:
        matcher = VPFPMacroMatcher()
        prediction = _view(momentum_x=(2.0, 2.0))
        truth = _view()
        observable = VPFPObservableTask(matcher).evaluate(
            _artifact(TaskKind.OBSERVABLE_MODEL, _Observable(prediction)),
            _case(),
            truth,
            _query(),
        )
        reduced = VPFPReducedDynamicsTask(matcher).evaluate(
            _artifact(TaskKind.REDUCED_DYNAMICS, _Reduced(prediction)),
            _case(),
            truth,
            _query(),
        )

        self.assertEqual(observable.primary_error, reduced.primary_error)
        self.assertEqual(observable.metrics, reduced.metrics)
        card = matcher.result_card(observable.macro_match)
        self.assertIn("not targets", card.summary)
        self.assertEqual(
            card.values["vpfp_macro_match_v1_relative_l2"],
            observable.primary_error,
        )

    def test_registry_discovers_matcher_without_changing_other_plugins(self) -> None:
        toy = get_plugin("toy")
        ns = get_plugin("ns")
        vpfp = get_plugin("vpfp")

        self.assertIsNone(toy.macro_matcher)
        self.assertIsNone(ns.macro_matcher)
        self.assertIs(
            vpfp.task("observable_model").macro_matcher,
            vpfp.macro_matcher,
        )
        self.assertIs(
            vpfp.task("reduced_dynamics").macro_matcher,
            vpfp.macro_matcher,
        )
        self.assertFalse(hasattr(vpfp.task("statistics"), "macro_matcher"))

    def test_agent_evaluator_returns_score_but_not_private_match_objects(self) -> None:
        truth = _Truth(_view())
        record = CaseRecord(_case(), truth)
        task = VPFPObservableTask()
        artifact = _artifact(
            TaskKind.OBSERVABLE_MODEL,
            _Observable(_view()),
        )
        evaluator = AgentEvaluator(
            (record,),
            task,
            EvaluatorPolicy(
                default_query=_query(),
                allow_grid_truth=False,
                allow_point_truth=False,
                development_coordinate_mode=(
                    DevelopmentCoordinateMode.USER_FIXED_QUERY
                ),
            ),
        )

        response = evaluator.evaluate(
            artifact,
            EvaluationRequest((candidate_case_id(record.case.case_id),)),
        )

        self.assertEqual(response.primary_error, 0.0)
        self.assertIn("macro_match_error", response.metrics)
        self.assertEqual(len(response.cases), 1)
        self.assertEqual(response.cases[0].case_id, candidate_case_id(record.case.case_id))
        self.assertEqual(response.cases[0].primary_error, 0.0)
        self.assertIsNone(response.cases[0].truth)
        self.assertIsNone(response.cases[0].prediction)
        self.assertFalse(hasattr(response.cases[0], "macro_match"))

    def test_grid_axes_and_point_coordinates_must_match_exactly(self) -> None:
        matcher = VPFPMacroMatcher()
        grid_target = matcher.prepare_target(_case(), _view(), _query())
        reversed_grid = AnswerView(
            QueryKind.GRID,
            fields=_view().fields,
            axes={"t": (1.0, 0.0)},
            shape=(2,),
        )
        with self.assertRaisesRegex(ValueError, "axes differ"):
            matcher.match(_case(), reversed_grid, grid_target)

        point_query = AnswerQuery.at_points(
            fields=VPFP_MACRO_FIELDS,
            points=({"t": 0.0}, {"t": 1.0}),
        )
        point_truth = _view(kind=QueryKind.POINTS)
        point_target = matcher.prepare_target(
            _case(),
            point_truth,
            point_query,
        )
        changed_points = AnswerView(
            QueryKind.POINTS,
            fields=point_truth.fields,
            points=({"t": 0.25}, {"t": 0.75}),
            shape=(2,),
        )
        with self.assertRaisesRegex(ValueError, "points differ"):
            matcher.match(_case(), changed_points, point_target)
        with self.assertRaisesRegex(ValueError, "points must match"):
            matcher.prepare_target(_case(), changed_points, point_query)

    def test_numpy_momentum_components_cannot_cancel_during_grouping(self) -> None:
        import numpy as np

        def array_view(momentum_x, momentum_y) -> AnswerView:
            return AnswerView(
                QueryKind.GRID,
                fields={
                    "rho": np.ones(2),
                    "momentum_x": np.asarray(momentum_x, dtype=float),
                    "momentum_y": np.asarray(momentum_y, dtype=float),
                    "kinetic_energy": np.ones(2),
                },
                axes={"t": (0.0, 1.0)},
                shape=(2,),
            )

        evaluation = evaluate_macro_prediction(
            matcher=VPFPMacroMatcher(),
            case=_case(),
            prediction=array_view((2.0, 2.0), (0.0, 0.0)),
            truth=array_view((1.0, 1.0), (1.0, 1.0)),
            query=_query(),
        )

        self.assertAlmostEqual(
            evaluation.metrics["momentum_relative_l2"],
            1.0,
        )
        self.assertAlmostEqual(evaluation.primary_error, 1.0 / 3.0)

    def test_controller_helpers_reject_misbound_targets(self) -> None:
        class MisboundMatcher:
            info = VPFPMacroMatcher.info
            fields = VPFP_MACRO_FIELDS
            contract_id = "expected-contract"

            def __init__(
                self,
                *,
                case_id=None,
                contract_id=None,
                query=None,
            ) -> None:
                self.target_case_id = case_id
                self.target_contract_id = contract_id
                self.target_query = query

            def prepare_target(self, case, truth, query) -> MacroTarget:
                return MacroTarget(
                    self.target_case_id or case.case_id,
                    self.target_contract_id or self.contract_id,
                    self.target_query or query,
                    truth,
                )

        record = CaseRecord(_case(), _Truth(_view()))
        wrong_query = AnswerQuery.grid(fields=VPFP_MACRO_FIELDS, t=3)
        cases = (
            (MisboundMatcher(case_id="other-case"), "different case"),
            (MisboundMatcher(contract_id="other-contract"), "different contract"),
            (MisboundMatcher(query=wrong_query), "different query"),
        )
        for matcher, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    materialize_macro_target(matcher, record, _query())

        with self.assertRaisesRegex(ValueError, "different query"):
            evaluate_macro_prediction(
                matcher=MisboundMatcher(query=wrong_query),
                case=_case(),
                prediction=_view(),
                truth=_view(),
                query=_query(),
            )


if __name__ == "__main__":
    unittest.main()
