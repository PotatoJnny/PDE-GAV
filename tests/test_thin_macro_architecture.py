import inspect
import unittest
from dataclasses import fields

import pde_agent_verifier_solver.engine.macro_closed_loop as closed_loop
import pde_agent_verifier_solver.engine.selection as selection
from pde_agent_verifier_solver.plugins.gkw.macro_loop import bind_gkw_macro_loop
from pde_agent_verifier_solver.plugins.vpfp.macro_loop import bind_vpfp_macro_loop
from pde_agent_verifier_solver.sdk import (
    FixedGeneratorCaseSource,
    MacroLoopPlugin,
)


class ThinMacroArchitectureTests(unittest.TestCase):
    def test_equation_plugin_surface_keeps_only_scientific_capabilities(self):
        self.assertEqual(
            tuple(field.name for field in fields(MacroLoopPlugin)),
            (
                "plugin_id",
                "generator",
                "objective",
                "public_session_factory",
                "public_session_methods",
                "development_solution_session_factory",
                "development_solution_session_methods",
                "public_problem_bundle",
                "public_result_kind",
                "public_result_task",
            ),
        )

    def test_shared_controller_has_no_equation_or_legacy_selection_branch(self):
        source = inspect.getsource(closed_loop)
        for forbidden in (
            "VPFP",
            "GKW",
            "task.select",
            "TaskProgram",
            "CampaignAdapter",
            "CampaignRunner",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_shared_selection_consumes_no_pde_sdk_or_field_names(self):
        source = inspect.getsource(selection)
        for forbidden in (
            "pde_agent_verifier_solver.sdk",
            "VPFP",
            "GKW",
            "density",
            "momentum",
            "energy",
            "flux",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_fixed_generator_and_pde_bindings_add_no_training_or_controller(self):
        self.assertFalse(hasattr(FixedGeneratorCaseSource, "train_base"))
        self.assertFalse(hasattr(FixedGeneratorCaseSource, "fit_guidance"))
        binding_source = inspect.getsource(bind_vpfp_macro_loop)
        self.assertNotIn("MacroClosedLoopRunner", binding_source)
        self.assertNotIn("EvidenceStage", binding_source)
        self.assertNotIn("select", binding_source)

        gkw_binding_source = inspect.getsource(bind_gkw_macro_loop)
        self.assertNotIn("MacroClosedLoopRunner", gkw_binding_source)
        self.assertNotIn("EvidenceStage", gkw_binding_source)
        self.assertNotIn("select", gkw_binding_source)

    def test_identity_and_handoff_records_have_fixed_frozen_schemas(self):
        self.assertEqual(
            tuple(field.name for field in fields(closed_loop.CandidateRef)),
            (
                "candidate_id",
                "bundle_sha256",
                "round_index",
                "agent_id",
                "parent_sha256",
                "bundle_uri",
            ),
        )
        self.assertEqual(
            tuple(field.name for field in fields(closed_loop.RoundFeedback)),
            (
                "previous_incumbent_sha256",
                "selected_sha256",
                "promoted",
                "decision_reason",
                "evidence_case_count",
                "paired_mean_improvement",
                "paired_confidence_lower",
                "paired_win_rate",
                "confidence_group_size",
                "perturbed_guardrail_passed",
                "perturbed_guardrail_confidence_lower",
            ),
        )
        self.assertEqual(
            tuple(field.name for field in fields(closed_loop.RoundHandoff)),
            (
                "round_index",
                "selected",
                "promoted",
                "challenger",
                "decision_reason",
                "feedback",
                "previous_handoff_sha256",
                "handoff_sha256",
            ),
        )
        for record_type in (
            closed_loop.CandidateRef,
            closed_loop.RoundFeedback,
            closed_loop.RoundHandoff,
        ):
            self.assertTrue(record_type.__dataclass_params__.frozen)
            names = tuple(field.name for field in fields(record_type))
            self.assertFalse(any("metadata" in name for name in names))
            self.assertFalse(any("target" in name for name in names))
            self.assertFalse(any("metric" in name for name in names))



if __name__ == "__main__":
    unittest.main()
