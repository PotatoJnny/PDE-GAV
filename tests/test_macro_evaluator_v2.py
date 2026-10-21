from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType
import unittest
from unittest.mock import patch

from pde_agent_verifier_solver.engine.candidate_case import candidate_case_id
from pde_agent_verifier_solver.engine.macro_evaluator import (
    CaseCreditExceeded,
    ControllerMacroEvaluator,
    DevelopmentMacroEvaluator,
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
    ProgramInfo,
    QueryKind,
    open_public_case_session,
)


QUERY = AnswerQuery.grid(fields=("density",), x=1)


class CountingTruth:
    def __init__(self, value):
        self.value = float(value)
        self.answer_calls = 0

    def answer(self, query):
        self.answer_calls += 1
        return _view(self.value)


class ScalarDensityMatcher:
    info = ProgramInfo("scalar-density", "Scalar density", "Test matcher")
    contract_id = "scalar_density_match_v1"
    fields = ("density",)

    def prepare_target(self, case, truth, query):
        return MacroTarget(case.case_id, self.contract_id, query, truth)

    def match(self, case, prediction, target):
        if case.case_id != target.case_id:
            raise ValueError("case identity changed")
        if prediction.kind is not target.values.kind:
            raise ValueError("query kind changed")
        if prediction.shape != target.values.shape:
            raise ValueError("shape changed")
        actual = float(prediction.fields["density"][0])
        expected = float(target.values.fields["density"][0])
        error = abs(actual - expected) / max(abs(expected), 1.0e-30)
        return MacroMatchResult(
            case_id=case.case_id,
            contract_id=self.contract_id,
            primary_error=error,
            metrics={"density_relative_error": error},
            passed=error <= 0.5,
            failures=() if error <= 0.5 else ("density tolerance exceeded",),
        )

    def result_card(self, result):
        raise NotImplementedError


class MutatingMatcher(ScalarDensityMatcher):
    def match(self, case, prediction, target):
        result = super().match(case, prediction, target)
        case.public_data["public_values"][0] = -123.0
        target.values.fields["density"] = (999.0,)
        return result


class ConstantCandidate:
    def __init__(self, value):
        self.value = float(value)
        self.seen = []
        self.budgets = []

    def predict_macros(self, case, query, budget=None):
        self.seen.append((case, query))
        self.budgets.append(budget)
        return _view(self.value)

class PublicSessionStub:
    def __init__(self, density):
        self._density = density
        self.close_calls = 0

    def density(self):
        return self._density

    def close(self):
        self.close_calls += 1


class PublicSessionCandidate:
    def __init__(self):
        self.sessions = []
        self.methods = []

    def predict_macros(self, case, query, budget=None):
        del query, budget
        session = open_public_case_session(case)
        if session is not open_public_case_session(case):
            raise RuntimeError("public session was not memoized")
        method = session.density
        self.sessions.append(session)
        self.methods.append(method)
        return _view(method())


class MutatingCandidate(ConstantCandidate):
    def __init__(self, value, truths):
        super().__init__(value)
        self.truths = truths

    def predict_macros(self, case, query, budget=None):
        self.assert_targets_are_already_materialized()
        case.public_data["public_values"][0] = -999.0
        query.resolution["x"] = 99
        return super().predict_macros(case, query, budget=budget)

    def assert_targets_are_already_materialized(self):
        if any(truth.answer_calls != 1 for truth in self.truths):
            raise AssertionError("candidate ran before cohort targets were ready")


class PartlyFailingCandidate:
    def predict_macros(self, case, query, budget=None):
        if case.public_data["public_values"][0] == 1.0:
            raise SystemExit("candidate attempted to stop the controller")
        return _view(2.0)


class ScriptedOrderRng:
    def __init__(self, *orders):
        self.orders = list(orders)

    def shuffle(self, values):
        if not self.orders:
            raise AssertionError("test exhausted scripted evaluation orders")
        order = self.orders.pop(0)
        if sorted(order) != list(range(len(values))):
            raise AssertionError("test supplied an invalid evaluation order")
        values[:] = order


class MacroEvaluatorV2Tests(unittest.TestCase):
    def setUp(self):
        self.truths = (CountingTruth(1.0), CountingTruth(2.0))
        self.records = tuple(
            CaseRecord(_case(index), truth)
            for index, truth in enumerate(self.truths, 1)
        )
        self.matcher = ScalarDensityMatcher()
        self.cohort = PreparedMacroCohort.prepare(
            cohort_id="paired-cohort",
            records=self.records,
            matcher=self.matcher,
            query=QUERY,
        )

    def test_materializes_targets_once_and_preserves_paired_identity(self):
        self.assertEqual([truth.answer_calls for truth in self.truths], [1, 1])
        evaluator = ControllerMacroEvaluator(self.cohort)

        first = evaluator.evaluate("candidate-a", ConstantCandidate(1.0))
        second = evaluator.evaluate("candidate-b", ConstantCandidate(2.0))

        self.assertEqual([truth.answer_calls for truth in self.truths], [1, 1])
        self.assertEqual(first.cohort_sha256, second.cohort_sha256)
        self.assertEqual(first.case_ids, ("case-1", "case-2"))
        self.assertEqual(second.case_ids, first.case_ids)
        self.assertNotEqual(first.candidate_id, second.candidate_id)

    def test_many_is_case_major_with_independent_public_sessions(self):
        events = []
        opened = []
        targets = []

        class RecordingCandidate:
            def __init__(self, label, value):
                self.label = label
                self.value = value
                self.sessions = []

            def predict_macros(self, case, query, budget=None):
                del query, budget
                events.append((case.case_id, self.label))
                session = open_public_case_session(case)
                self.sessions.append(session)
                return _view(self.value)

        def factory(case):
            opened.append(case.case_id)
            target = PublicSessionStub(
                case.public_data["public_values"][0]
            )
            targets.append(target)
            return target

        first = RecordingCandidate("first", 1.0)
        second = RecordingCandidate("second", 2.0)
        evaluator = ControllerMacroEvaluator(
            self.cohort,
            public_session_factory=factory,
            public_session_methods=("density",),
            candidate_order_rng=ScriptedOrderRng((0, 1)),
        )

        batches = evaluator.evaluate_many(
            (("candidate-a", first), ("candidate-b", second))
        )

        public_ids = tuple(case.case_id for case in self.cohort.public_cases())
        self.assertEqual(
            events,
            [
                (public_ids[0], "first"),
                (public_ids[0], "second"),
                (public_ids[1], "first"),
                (public_ids[1], "second"),
            ],
        )
        self.assertEqual(opened, ["case-1", "case-1", "case-2", "case-2"])
        self.assertEqual([target.close_calls for target in targets], [1] * 4)
        self.assertEqual(
            tuple(batch.candidate_id for batch in batches),
            ("candidate-a", "candidate-b"),
        )
        self.assertEqual(batches[0].case_ids, batches[1].case_ids)
        self.assertEqual(
            batches[0].execution_order,
            batches[1].execution_order,
        )
        self.assertEqual(batches[0].execution_order, ("case-1", "case-2"))
        self.assertNotEqual(batches[0].scores, batches[1].scores)
        for session in (*first.sessions, *second.sessions):
            with self.assertRaisesRegex(RuntimeError, "no longer active"):
                session.density()

        paired = ControllerMacroEvaluator(self.cohort).evaluate_pair(
            "pair-a",
            ConstantCandidate(1.0),
            "pair-b",
            ConstantCandidate(2.0),
        )
        self.assertEqual(
            tuple(batch.candidate_id for batch in paired),
            ("pair-a", "pair-b"),
        )
        with self.assertRaisesRegex(ValueError, "at least one"):
            evaluator.evaluate_many(())

    def test_fresh_private_order_breaks_odd_even_distribution_routing(self):
        root = Case(
            case_id="paired-root",
            source=CaseSource.GENERATED,
            visibility=CaseVisibility.HIDDEN,
            public_data={"public_values": [1.0]},
        )
        child = Case(
            case_id="paired-child",
            source=CaseSource.PERTURBED,
            visibility=CaseVisibility.HIDDEN,
            public_data={"public_values": [1.0]},
            parent_case_id=root.case_id,
        )
        cohort = PreparedMacroCohort.prepare(
            cohort_id="private-order-pair",
            records=(
                CaseRecord(root, CountingTruth(1.0)),
                CaseRecord(child, CountingTruth(1.0)),
            ),
            matcher=ScalarDensityMatcher(),
            query=QUERY,
        )

        class OddEvenCandidate:
            def __init__(self):
                self.calls = 0
                self.case_ids = []

            def predict_macros(self, case, query, budget=None):
                self.case_ids.append(case.case_id)
                value = 1.0 if self.calls % 2 == 0 else 0.0
                self.calls += 1
                return _view(value)

        order_rng = ScriptedOrderRng((0, 1), (1, 0))
        evaluator = ControllerMacroEvaluator(
            cohort,
            candidate_order_rng=order_rng,
        )
        first_candidate = OddEvenCandidate()
        second_candidate = OddEvenCandidate()

        first = evaluator.evaluate("odd-even-first", first_candidate)
        second = evaluator.evaluate("odd-even-second", second_candidate)

        public_ids = tuple(case.case_id for case in cohort.public_cases())
        self.assertEqual(first_candidate.case_ids, list(public_ids))
        self.assertEqual(second_candidate.case_ids, list(reversed(public_ids)))
        self.assertEqual(first.case_ids, (root.case_id, child.case_id))
        self.assertEqual(second.case_ids, first.case_ids)
        self.assertEqual(first.root_ids, (root.case_id, root.case_id))
        self.assertEqual(second.root_ids, first.root_ids)
        self.assertEqual(first.execution_order, (root.case_id, child.case_id))
        self.assertEqual(second.execution_order, (child.case_id, root.case_id))
        self.assertEqual(
            tuple(score.primary_error for score in first.scores),
            (0.0, 1.0),
        )
        self.assertEqual(
            tuple(score.primary_error for score in second.scores),
            (1.0, 0.0),
        )
        self.assertEqual(
            tuple(timing.case_id for timing in first.case_timings),
            first.case_ids,
        )
        self.assertEqual(
            tuple(timing.case_id for timing in second.case_timings),
            second.case_ids,
        )

    def test_candidate_only_receives_deep_public_snapshots(self):
        candidate = MutatingCandidate(1.0, self.truths)
        evaluator = ControllerMacroEvaluator(self.cohort)
        result = evaluator.evaluate("mutator", candidate)

        self.assertEqual(result.scores[0].status, "valid")
        self.assertEqual(
            self.cohort.public_cases()[0].public_data["public_values"],
            [1.0],
        )
        self.assertEqual(QUERY.resolution["x"], 1)
        clean = ConstantCandidate(1.0)
        evaluator.evaluate("clean", clean)
        expected_by_id = {
            case.case_id: case.public_data["public_values"]
            for case in self.cohort.public_cases()
        }
        seen_by_id = {
            case.case_id: case.public_data["public_values"]
            for case, _query in clean.seen
        }
        self.assertEqual(seen_by_id, expected_by_id)
        self.assertTrue(all(
            query.resolution["x"] == 1 for _case, query in clean.seen
        ))

    def test_mapping_proxy_and_readonly_array_snapshots_isolate_mutation(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("NumPy is required for ndarray snapshot coverage")

        released_array = np.asarray([1.0, 2.0], dtype=np.float64)
        released_array.setflags(write=False)
        released_public = MappingProxyType(
            {
                "arrays": MappingProxyType({"x": released_array}),
                "metadata": MappingProxyType({"labels": ["released"]}),
            }
        )
        released_case = Case(
            case_id="readonly-release",
            source=CaseSource.GENERATED,
            visibility=CaseVisibility.HIDDEN,
            public_data=released_public,
        )
        released_query = AnswerQuery(
            kind=QueryKind.GRID,
            fields=("density",),
            resolution=MappingProxyType({"x": 1}),
        )
        cohort = PreparedMacroCohort.prepare(
            cohort_id="readonly-release-cohort",
            records=(CaseRecord(released_case, CountingTruth(1.0)),),
            matcher=self.matcher,
            query=released_query,
        )
        public_query = cohort.public_queries()[0]
        public_query.resolution["x"] = 7
        self.assertEqual(released_query.resolution["x"], 1)
        self.assertEqual(
            cohort.public_queries()[0].resolution["x"],
            1,
        )

        class ArrayMutatingCandidate:
            array = None
            saw_writable_array = False

            def predict_macros(self, case, query, budget=None):
                self.array = case.public_data["arrays"]["x"]
                self.saw_writable_array = bool(self.array.flags.writeable)
                self.array[0] = -99.0
                case.public_data["metadata"]["labels"].append("candidate")
                case.public_data["candidate_only"] = True
                query.resolution["x"] = 7
                return _view(1.0)

        candidate = ArrayMutatingCandidate()
        result = ControllerMacroEvaluator(cohort).evaluate("mutator", candidate)

        self.assertTrue(result.accepted)
        self.assertTrue(candidate.saw_writable_array)
        self.assertIsNot(candidate.array, released_array)
        self.assertFalse(released_array.flags.writeable)
        np.testing.assert_array_equal(released_array, [1.0, 2.0])
        self.assertEqual(released_public["metadata"]["labels"], ["released"])
        self.assertEqual(released_query.resolution["x"], 1)

        exposed = cohort.public_cases()[0]
        exposed_array = exposed.public_data["arrays"]["x"]
        self.assertTrue(exposed_array.flags.writeable)
        exposed_array[0] = -7.0
        exposed.public_data["metadata"]["labels"].append("caller")

        fresh = cohort.public_cases()[0]
        np.testing.assert_array_equal(
            fresh.public_data["arrays"]["x"],
            [1.0, 2.0],
        )
        self.assertEqual(fresh.public_data["metadata"]["labels"], ["released"])

    def test_controller_fails_closed_per_case_and_keeps_order(self):
        result = ControllerMacroEvaluator(self.cohort).evaluate(
            "partly-failing",
            PartlyFailingCandidate(),
        )

        self.assertEqual(result.case_ids, ("case-1", "case-2"))
        self.assertEqual(result.scores[0].status, "invalid")
        self.assertIsNone(result.scores[0].primary_error)
        self.assertEqual(
            result.scores[0].failures,
            ("candidate_execution_failed",),
        )
        self.assertEqual(result.scores[1].status, "valid")
        self.assertFalse(result.accepted)

    def test_development_credits_and_sanitized_optional_finite_truth(self):
        evaluator = DevelopmentMacroEvaluator(
            self.cohort,
            case_credit_limit=2,
            allow_finite_truth=True,
        )
        response = evaluator.evaluate(
            "candidate",
            ConstantCandidate(1.0),
            case_ids=("case-1",),
            return_finite_truth=True,
        )

        self.assertEqual(response.charged_case_credits, 1)
        self.assertEqual(response.remaining_case_credits, 1)
        self.assertIsInstance(response.cases[0].finite_truth, AnswerView)
        self.assertFalse(hasattr(response.cases[0], "target"))
        self.assertFalse(hasattr(response.cases[0], "truth_program"))
        with self.assertRaises(CaseCreditExceeded):
            evaluator.evaluate(
                "too-expensive",
                ConstantCandidate(1.0),
                case_ids=("case-1", "case-2"),
            )
        self.assertEqual(evaluator.charged_case_credits, 1)

    def test_truth_permission_and_bad_identity_do_not_consume_credits(self):
        evaluator = DevelopmentMacroEvaluator(
            self.cohort,
            case_credit_limit=2,
            allow_finite_truth=False,
        )
        with self.assertRaises(PermissionError):
            evaluator.evaluate(
                "candidate",
                ConstantCandidate(1.0),
                case_ids=("case-1",),
                return_finite_truth=True,
            )
        with self.assertRaises(ValueError):
            evaluator.evaluate(
                "candidate",
                ConstantCandidate(1.0),
                case_ids=("case-1", "case-1"),
            )
        with self.assertRaises(KeyError):
            evaluator.evaluate(
                "candidate",
                ConstantCandidate(1.0),
                case_ids=("unknown",),
            )
        with self.assertRaisesRegex(ValueError, "candidate_id"):
            evaluator.evaluate(
                "",
                ConstantCandidate(1.0),
                case_ids=("case-1",),
            )
        self.assertEqual(evaluator.charged_case_credits, 0)

    def test_raw_and_opaque_aliases_share_one_resolved_row(self):
        alias = candidate_case_id("case-1")
        evaluator = DevelopmentMacroEvaluator(
            self.cohort,
            case_credit_limit=2,
        )
        with self.assertRaisesRegex(ValueError, "repeat macro cases"):
            evaluator.evaluate(
                "candidate",
                ConstantCandidate(1.0),
                case_ids=("case-1", alias),
            )
        self.assertEqual(evaluator.charged_case_credits, 0)

        subset = self.cohort._subset(
            cohort_id="opaque-subset",
            case_ids=(candidate_case_id("case-2"), alias),
        )
        self.assertEqual(subset.case_ids, ("case-2", "case-1"))
        self.assertEqual(
            tuple(case.case_id for case in subset.public_cases()),
            (candidate_case_id("case-2"), alias),
        )


    def test_rejects_duplicate_case_and_empty_candidate_id(self):
        duplicate = (self.records[0], self.records[0])
        with self.assertRaisesRegex(ValueError, "unique"):
            PreparedMacroCohort.prepare(
                cohort_id="bad",
                records=duplicate,
                matcher=self.matcher,
                query=QUERY,
            )
        with self.assertRaisesRegex(ValueError, "candidate_id"):
            ControllerMacroEvaluator(self.cohort).evaluate(
                "",
                ConstantCandidate(1.0),
            )

    def test_public_session_is_bound_only_during_candidate_calls(self):
        opened = []
        targets = []

        def factory(case):
            opened.append(case)
            target = PublicSessionStub(case.public_data["public_values"][0])
            targets.append(target)
            return target

        candidate = PublicSessionCandidate()
        result = ControllerMacroEvaluator(
            self.cohort,
            public_session_factory=factory,
            public_session_methods=("density",),
            candidate_order_rng=ScriptedOrderRng((0, 1)),
        ).evaluate("session-candidate", candidate)

        self.assertTrue(result.accepted)
        self.assertEqual(
            [case.case_id for case in opened],
            ["case-1", "case-2"],
        )
        self.assertIsNot(opened[0], self.records[0].case)
        self.assertEqual(len(candidate.sessions), 2)
        self.assertEqual([target.close_calls for target in targets], [1, 1])
        with self.assertRaisesRegex(RuntimeError, "no longer active"):
            candidate.sessions[0].density
        with self.assertRaisesRegex(RuntimeError, "no longer active"):
            candidate.methods[0]()
        with self.assertRaisesRegex(RuntimeError, "does not expose"):
            open_public_case_session(self.records[0].case)
        with self.assertRaises(TypeError):
            ControllerMacroEvaluator(
                self.cohort,
                public_session_factory=object(),
                public_session_methods=("density",),
            )

    def test_mutated_candidate_snapshot_cannot_change_raw_session_case(self):
        opened = []

        class MutatingSessionCandidate:
            def predict_macros(self, case, query, budget=None):
                del query, budget
                case.public_data["public_values"][0] = -999.0
                return _view(open_public_case_session(case).density())

        def factory(case):
            opened.append(case)
            return PublicSessionStub(
                case.public_data["public_values"][0]
            )

        result = ControllerMacroEvaluator(
            self.cohort,
            public_session_factory=factory,
            public_session_methods=("density",),
            candidate_order_rng=ScriptedOrderRng((0, 1)),
        ).evaluate("mutating-session-candidate", MutatingSessionCandidate())

        self.assertTrue(result.accepted)
        self.assertEqual(
            [case.public_data["public_values"] for case in opened],
            [[1.0], [2.0]],
        )
        self.assertEqual(
            [case.public_data["public_values"] for case in self.cohort.public_cases()],
            [[1.0], [2.0]],
        )


    def test_public_session_releases_binding_when_target_close_fails(self):
        class FailingPublicSessionStub(PublicSessionStub):
            def close(self):
                self.close_calls += 1
                raise RuntimeError("target close failed")

        targets = []

        def factory(case):
            target = FailingPublicSessionStub(
                case.public_data["public_values"][0]
            )
            targets.append(target)
            return target

        candidate = PublicSessionCandidate()
        result = ControllerMacroEvaluator(
            self.cohort,
            public_session_factory=factory,
            public_session_methods=("density",),
        ).evaluate("failing-close-candidate", candidate)

        self.assertFalse(result.accepted)
        self.assertEqual([target.close_calls for target in targets], [1, 1])
        for session in candidate.sessions:
            with self.assertRaisesRegex(RuntimeError, "no longer active"):
                session.density

    def test_compute_budget_uses_shared_candidate_abi_and_is_strict(self):
        candidate = ConstantCandidate(1.0)
        result = ControllerMacroEvaluator(
            self.cohort, compute_budget=2.5
        ).evaluate("budgeted", candidate)
        self.assertTrue(result.accepted)
        self.assertEqual(candidate.budgets, [2.5, 2.5])
        for invalid in (True, 0, -1.0, float("nan"), float("inf"), "1"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    ControllerMacroEvaluator(self.cohort, compute_budget=invalid)

    def test_development_configuration_rejects_non_integer_credits(self):
        for invalid in (True, 1.5, "2"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    DevelopmentMacroEvaluator(
                        self.cohort,
                        case_credit_limit=invalid,
                    )
        with self.assertRaises(TypeError):
            DevelopmentMacroEvaluator(
                self.cohort,
                case_credit_limit=1,
                allow_finite_truth=1,
            )

    def test_public_digest_excludes_targets_but_private_evidence_binds_them(self):
        alternate = tuple(
            CaseRecord(_case(index), CountingTruth(value))
            for index, value in enumerate((7.0, 8.0), 1)
        )
        same_public = PreparedMacroCohort.prepare(
            cohort_id="paired-cohort",
            records=alternate,
            matcher=self.matcher,
            query=QUERY,
        )
        self.assertEqual(
            self.cohort.cohort_sha256,
            same_public.cohort_sha256,
        )
        self.assertNotEqual(
            self.cohort._private_target_sha256,
            same_public._private_target_sha256,
        )
        batch = ControllerMacroEvaluator(self.cohort).evaluate(
            "evidence", ConstantCandidate(1.0)
        )
        self.assertEqual(batch.target_sha256, self.cohort._private_target_sha256)
        response = DevelopmentMacroEvaluator(
            self.cohort, case_credit_limit=1
        ).evaluate("public-only", ConstantCandidate(1.0), case_ids=("case-1",))
        self.assertEqual(response.cohort_sha256, self.cohort.cohort_sha256)
        self.assertFalse(hasattr(response, "target_sha256"))

    def test_development_feedback_redacts_matcher_details_by_default(self):
        controller = ControllerMacroEvaluator(self.cohort).evaluate(
            "verbose", ConstantCandidate(10.0)
        )
        self.assertEqual(controller.scores[0].failures, ("density tolerance exceeded",))
        default = DevelopmentMacroEvaluator(
            self.cohort,
            case_credit_limit=1,
        ).evaluate(
            "default",
            ConstantCandidate(10.0),
            case_ids=("case-1",),
        )
        score = default.cases[0].score
        self.assertEqual(score.status, "rejected")
        self.assertEqual(score.metrics, {})
        self.assertEqual(score.failures, ("macro_gate_rejected",))
        allowed = DevelopmentMacroEvaluator(
            self.cohort,
            case_credit_limit=1,
            safe_metric_names=("density_relative_error",),
        ).evaluate(
            "allowed",
            ConstantCandidate(10.0),
            case_ids=("case-1",),
        )
        self.assertEqual(
            allowed.cases[0].score.metrics,
            {"density_relative_error": 9.0},
        )

    def test_disk_backed_digest_parity_lazy_reads_and_discard(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("NumPy is required for disk-backed cohort coverage")

        with TemporaryDirectory() as temporary:
            disk = PreparedMacroCohort.prepare(
                cohort_id="paired-cohort",
                records=self.records,
                matcher=self.matcher,
                query=QUERY,
                private_store_root=Path(temporary) / "private",
            )
            self.assertTrue(disk.disk_backed)
            self.assertEqual(disk.case_ids, self.cohort.case_ids)
            self.assertEqual(disk.cohort_sha256, self.cohort.cohort_sha256)
            self.assertEqual(
                disk._private_target_sha256,
                self.cohort._private_target_sha256,
            )
            manifest_path = disk.private_manifest_path
            self.assertIsNotNone(manifest_path)
            store_root = manifest_path.parent
            store = disk._prepared.store

            with patch.object(
                store,
                "read_row",
                wraps=store.read_row,
            ) as read_row:
                batches = ControllerMacroEvaluator(disk).evaluate_many(
                    (
                        ("disk-controller-a", ConstantCandidate(1.0)),
                        ("disk-controller-b", ConstantCandidate(2.0)),
                    )
                )
            self.assertEqual(len(batches), 2)
            self.assertEqual(read_row.call_count, 2)

            with patch.object(
                store,
                "read_row",
                wraps=store.read_row,
            ) as read_row:
                DevelopmentMacroEvaluator(
                    disk,
                    case_credit_limit=1,
                ).evaluate(
                    "disk-development",
                    ConstantCandidate(1.0),
                    case_ids=("case-2",),
                )
            self.assertEqual(read_row.call_count, 1)

            row = disk._prepared[0]
            mapped = row.target.values.fields["density"]
            self.assertIsInstance(mapped, np.memmap)
            self.assertFalse(mapped.flags.writeable)
            disk.discard()
            self.assertFalse(store_root.exists())
            with self.assertRaisesRegex(RuntimeError, "discarded"):
                disk.public_cases()

    def test_discard_release_callback_failure_can_be_retried(self):
        attempts = []

        def release(case_ids):
            attempts.append(case_ids)
            if len(attempts) == 1:
                raise RuntimeError("transient release failure")

        with TemporaryDirectory() as temporary:
            cohort = PreparedMacroCohort.prepare(
                cohort_id="retry-discard",
                records=self.records,
                matcher=self.matcher,
                query=QUERY,
                private_store_root=Path(temporary) / "private",
                release_cases=release,
            )
            store_root = cohort.private_manifest_path.parent
            with self.assertRaisesRegex(RuntimeError, "transient release"):
                cohort.discard()
            self.assertFalse(store_root.exists())
            self.assertFalse(cohort._discarded)

            cohort.discard()
            self.assertEqual(attempts, [cohort.case_ids, cohort.case_ids])
            with self.assertRaisesRegex(RuntimeError, "discarded"):
                cohort.public_cases()

    def test_matcher_receives_fresh_case_and_target_copies_per_call(self):
        records = (
            CaseRecord(_case(1), CountingTruth(1.0)),
            CaseRecord(_case(2), CountingTruth(2.0)),
        )
        cohort = PreparedMacroCohort.prepare(
            cohort_id="mutating-matcher",
            records=records,
            matcher=MutatingMatcher(),
            query=QUERY,
        )
        evaluator = ControllerMacroEvaluator(cohort)
        first = evaluator.evaluate("first", ConstantCandidate(1.0))
        second = evaluator.evaluate("second", ConstantCandidate(1.0))
        self.assertEqual(first.scores, second.scores)
        self.assertEqual(cohort.public_cases()[0].public_data["public_values"], [1.0])
        self.assertEqual(cohort._prepared[0].target.values.fields["density"], (1.0,))


def _case(index):
    return Case(
        case_id=f"case-{index}",
        source=CaseSource.GENERATED,
        visibility=CaseVisibility.HIDDEN,
        public_data={"public_values": [float(index)]},
    )


def _view(value):
    return AnswerView(
        kind=QueryKind.GRID,
        fields={"density": (float(value),)},
        axes={"x": (0.0,)},
        shape=(1,),
    )


if __name__ == "__main__":
    unittest.main()
