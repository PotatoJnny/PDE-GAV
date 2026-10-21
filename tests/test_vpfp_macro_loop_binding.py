import hashlib
import unittest

from pde_agent_verifier_solver.engine import (
    EvidenceStage,
    MacroAgentOutcome,
    MacroAgentSpec,
    MacroClosedLoopRunner,
    MacroLoopConfig,
)
from pde_agent_verifier_solver.plugins.vpfp import (
    VPFPContinuousGenerator,
    VPFPMacroMatcher,
    bind_vpfp_macro_loop,
)
from pde_agent_verifier_solver.plugins.vpfp.macro_matching import VPFP_MACRO_FIELDS
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    CandidateRef,
    CandidateSubmission,
    Case,
    CaseRecord,
    CaseSource,
    GenerationKind,
    QueryKind,
)


def _view(scale):
    values = (float(scale), float(scale))
    return AnswerView(
        kind=QueryKind.GRID,
        fields={field: values for field in VPFP_MACRO_FIELDS},
        axes={"t": (0.0, 1.0)},
        shape=(2,),
    )


class _Truth:
    def answer(self, query):
        return _view(1.0)


class _Backend:
    supports_perturbations = True

    def __init__(self):
        self.serial = 0
        self.generate_calls = []
        self.generated_batches = []

    def train_base(self, *args, **kwargs):
        raise AssertionError("VPFP shared loop must not train the frozen Generator")

    def fit_guidance(self, *args, **kwargs):
        raise AssertionError("VPFP shared loop must not update guidance")

    def generate(self, base_state, guidance_state, request, progress=None):
        self.generate_calls.append((base_state, guidance_state, request))
        records = []
        if request.kind is GenerationKind.BASE:
            if request.anchors:
                raise AssertionError("clean VPFP generation cannot receive anchors")
            for _index in range(request.count):
                self.serial += 1
                records.append(
                    CaseRecord(
                        Case(
                            case_id=f"vpfp-fixed-{self.serial}",
                            source=CaseSource.GENERATED,
                            visibility=request.visibility,
                            public_data={"equation": "VPFP public manufactured case"},
                            round_index=request.round_index,
                        ),
                        _Truth(),
                    )
                )
        elif request.kind is GenerationKind.PERTURBED:
            if len(request.anchors) != request.count:
                raise AssertionError(
                    "perturbed VPFP generation needs one anchor per case"
                )
            for anchor in request.anchors:
                records.append(
                    CaseRecord(
                        Case(
                            case_id=f"{anchor.case_id}-perturbed",
                            source=CaseSource.PERTURBED,
                            visibility=request.visibility,
                            public_data={"equation": "VPFP public manufactured case"},
                            round_index=request.round_index,
                            parent_case_id=anchor.case_id,
                        ),
                        _Truth(),
                    )
                )
        else:
            raise AssertionError(f"unexpected VPFP generation kind: {request.kind}")
        batch = tuple(records)
        self.generated_batches.append(batch)
        return batch


class _MacroProgram:
    def __init__(self, relative_error):
        self.relative_error = float(relative_error)

    def predict_macros(self, case, query, budget=None):
        return _view(1.0 - self.relative_error)


class _Agent:
    def run(self, context):
        candidate_id = f"vpfp-r{context.round_index}"
        digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()
        ref = CandidateRef(
            candidate_id,
            digest,
            context.round_index,
            context.agent.agent_id,
            parent_sha256=context.incumbent.bundle_sha256,
        )
        candidate = CandidateSubmission(ref, _MacroProgram(0.25))
        response = context.evaluator.evaluate(
            candidate_id,
            candidate.program,
            case_ids=tuple(case.case_id for case in context.cases),
        )
        self.last_metric_names = tuple(response.cases[0].score.metrics)
        return MacroAgentOutcome(candidate)


class VPFPMacroLoopBindingTests(unittest.TestCase):
    def test_perturbation_capability_fails_closed_without_backend_support(self):
        generator = VPFPContinuousGenerator()
        self.assertFalse(generator.capabilities.perturbations)
        fields = {field.name: field.default for field in generator.setup_fields()}
        self.assertFalse(fields["perturbations_enabled"])

    def test_fixed_vpfp_generator_and_matcher_use_the_shared_controller(self):
        backend = _Backend()
        base_state = {"checkpoint": "trained-vpfp-release"}
        generator = VPFPContinuousGenerator(backend)
        plugin = bind_vpfp_macro_loop(
            generator=generator,
            base_state=base_state,
            generator_sha256="a" * 64,
            query_factory=lambda case, profile: AnswerQuery.grid(
                fields=VPFP_MACRO_FIELDS,
                t=2,
            ),
        )
        cold_ref = CandidateRef(
            "vpfp-cold",
            "b" * 64,
            0,
            "baseline",
        )
        agent = _Agent()

        result = MacroClosedLoopRunner(
            legacy_macro_reference=True,
            config=MacroLoopConfig(
                name="vpfp-thin-binding",
                round_count=1,
                agents=(MacroAgentSpec("agent", "Agent"),),
                development_case_count=2,
                selection_case_count=2,
                evidence_stages=(
                    EvidenceStage(
                        2,
                        confidence_z=0.0,
                        min_win_rate=1.0,
                        require_cvar_no_regression=True,
                    ),
                ),
                terminal_case_count=2,
                agent_case_credits=2,
                paired_perturbations=True,
                terminal_confidence_z=0.0,
            ),
            plugin=plugin,
            cold_start=CandidateSubmission(cold_ref, _MacroProgram(0.5)),
            agents={"agent": agent},
        ).run()

        self.assertIsInstance(plugin.objective.matcher, VPFPMacroMatcher)
        self.assertEqual(result.macro_contract_id, VPFPMacroMatcher.contract_id)
        expected_published = {
            "rho_relative_l2",
            "momentum_relative_l2",
            "kinetic_energy_relative_l2",
        }
        self.assertEqual(set(result.published_metric_names), expected_published)
        self.assertEqual(
            set(result.terminal.selected_published_metrics.aggregates),
            expected_published,
        )
        self.assertTrue(result.rounds[0].promotion.promoted)
        self.assertEqual(result.selected.candidate_id, "vpfp-r1")
        self.assertTrue(result.terminal.comparison.passed)
        self.assertIn("vpfp_macro_match_v1_relative_l2", agent.last_metric_names)
        self.assertEqual(len(backend.generate_calls), 6)
        self.assertEqual(len(backend.generated_batches), 6)
        self.assertTrue(
            all(state is base_state for state, _guidance, _request in backend.generate_calls)
        )
        self.assertTrue(
            all(guidance is None for _state, guidance, _request in backend.generate_calls)
        )
        self.assertTrue(
            all(
                request.settings["fixed_generator_sha256"] == "a" * 64
                for _state, _guidance, request in backend.generate_calls
            )
        )
        cohort_case_ids = (
            result.rounds[0].development_case_ids,
            result.rounds[0].finalist_evaluations[0].batch.case_ids,
            result.terminal.selected_score.case_ids,
        )
        self.assertEqual(
            result.rounds[0].promotion.comparisons[-1].case_ids,
            result.rounds[0].finalist_evaluations[0].batch.case_ids[::2],
        )
        self.assertEqual(
            tuple(
                request.settings["cohort"]
                for _state, _guidance, request in backend.generate_calls[::2]
            ),
            ("development", "selection", "terminal"),
        )
        for cohort_index, case_ids in enumerate(cohort_case_ids):
            base_call = backend.generate_calls[2 * cohort_index][2]
            perturbed_call = backend.generate_calls[2 * cohort_index + 1][2]
            roots = backend.generated_batches[2 * cohort_index]
            children = backend.generated_batches[2 * cohort_index + 1]

            with self.subTest(cohort=base_call.settings["cohort"]):
                self.assertIs(base_call.kind, GenerationKind.BASE)
                self.assertEqual(base_call.anchors, ())
                self.assertIs(perturbed_call.kind, GenerationKind.PERTURBED)
                self.assertEqual(
                    tuple(anchor.case_id for anchor in perturbed_call.anchors),
                    tuple(record.case.case_id for record in roots),
                )
                expected_case_ids = tuple(
                    record.case.case_id
                    for pair in zip(roots, children)
                    for record in pair
                )
                if cohort_index >= 2:
                    expected_case_ids = tuple(
                        record.case.case_id for record in roots
                    )
                self.assertEqual(
                    case_ids,
                    expected_case_ids,
                )
                for root, child in zip(roots, children):
                    self.assertIs(root.case.source, CaseSource.GENERATED)
                    self.assertIsNone(root.case.parent_case_id)
                    self.assertIs(child.case.source, CaseSource.PERTURBED)
                    self.assertEqual(
                        child.case.parent_case_id,
                        root.case.case_id,
                    )


if __name__ == "__main__":
    unittest.main()
