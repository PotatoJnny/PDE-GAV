import json
import stat
import tempfile
import unittest
from pathlib import Path

from pde_agent_verifier_solver.plugins.gkw import (
    GKWAnswerFirstGenerator,
    GKW_DEVELOPMENT_SOLUTION_SESSION_METHODS,
    GKW_PUBLIC_SESSION_METHODS,
    bind_gkw_macro_loop,
    gkw_public_problem_bundle,
)
from pde_agent_verifier_solver.sdk import (
    CONTINUOUS_POINT_SOLUTION_CONTRACT,
    PUBLIC_PROBLEM_PRIVATE_INFORMATION,
    FrozenGuidanceCapability,
    MacroLoopPlugin,
    PublicOracleMethod,
    PublicProblemBundle,
    PublicResource,
)


class _Cases:
    generator_id = "test-generator"
    artifact_sha256 = "a" * 64


class _Objective:
    pass


class PublicProblemBundleTests(unittest.TestCase):
    def test_materializes_immutable_resources_with_stable_manifest(self):
        bundle = PublicProblemBundle(
            bundle_id="test-public-problem",
            equation_id="test-equation",
            description="complete public equation",
            resources=(
                PublicResource(
                    "reference/operator.py",
                    b"VALUE = 1\n",
                    media_type="text/x-python",
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as raw:
            destination = Path(raw) / "public_resources"
            self.assertEqual(bundle.materialize(destination), destination)
            resource = destination / "reference" / "operator.py"
            self.assertEqual(resource.read_bytes(), b"VALUE = 1\n")
            self.assertEqual(stat.S_IMODE(resource.stat().st_mode), 0o444)
            manifest = json.loads(
                (destination / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["bundle_sha256"], bundle.bundle_sha256)
            self.assertEqual(
                manifest["candidate_solution_access"]["abi"],
                "prepare_solution_then_evaluate_points_v1",
            )
            self.assertFalse(
                manifest["candidate_solution_access"]["prepare_solution"][
                    "scoring_coordinates_visible"
                ]
            )
            with self.assertRaises(FileExistsError):
                bundle.materialize(destination)

    def test_resource_paths_and_private_categories_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "relative|escape"):
            PublicResource("../private.py", b"secret")
        with self.assertRaisesRegex(ValueError, "only generator"):
            PublicProblemBundle(
                bundle_id="bad",
                equation_id="equation",
                description="bad private policy",
                resources=(PublicResource("operator.py", b"x = 1\n"),),
                private_information=(
                    *PUBLIC_PROBLEM_PRIVATE_INFORMATION,
                    "pde_operator",
                ),
            )

    def test_numerical_oracle_requires_agent_selected_coordinates(self):
        with self.assertRaisesRegex(ValueError, "Agent-selected"):
            PublicOracleMethod(
                name="query_source",
                description="fixed development query",
                input_schema={"coordinates": "fixed_panel"},
                output_schema={"field": "source"},
                cost_unit="row",
                coordinate_policy="controller_fixed",
            )

    def test_macro_plugin_allowlist_matches_declared_oracle(self):
        method = PublicOracleMethod(
            name="query_source",
            description="source at arbitrary requested rows",
            input_schema={"coordinates": "Agent-selected"},
            output_schema={"field": "source"},
            cost_unit="row",
        )
        bundle = PublicProblemBundle(
            bundle_id="oracle-bundle",
            equation_id="equation",
            description="equation and oracle",
            resources=(PublicResource("operator.py", b"x = 1\n"),),
            oracle_methods=(method,),
        )
        with self.assertRaisesRegex(ValueError, "allowlist must match"):
            MacroLoopPlugin(
                "test",
                _Cases(),
                _Objective(),
                public_session_factory=lambda case: object(),
                public_session_methods=("query_initial",),
                public_problem_bundle=bundle,
            )

    def test_development_solution_contract_cannot_reuse_candidate_session(self):
        public = PublicOracleMethod(
            name="query_source",
            description="public equation source",
            input_schema={"coordinates": "Agent-selected"},
            output_schema={"field": "source"},
            cost_unit="row",
        )
        completed = PublicOracleMethod(
            name="query_completed",
            description="completed solution at Agent-selected coordinates",
            input_schema={"coordinates": "Agent-selected"},
            output_schema={"field": "macro"},
            cost_unit="point",
        )
        bundle = PublicProblemBundle(
            bundle_id="separated-oracles",
            equation_id="equation",
            description="separated equation and completed answer access",
            resources=(PublicResource("operator.py", b"x = 1\n"),),
            oracle_methods=(public,),
            development_solution_oracle_methods=(completed,),
        )

        def shared_factory(case):
            del case
            return object()

        with self.assertRaisesRegex(ValueError, "distinct factories"):
            MacroLoopPlugin(
                "test",
                _Cases(),
                _Objective(),
                public_session_factory=shared_factory,
                public_session_methods=("query_source",),
                development_solution_session_factory=shared_factory,
                development_solution_session_methods=("query_completed",),
                public_problem_bundle=bundle,
            )

    def test_guidance_manifest_is_distribution_level_and_online_when_supported(self):
        capability = FrozenGuidanceCapability(supported=True).manifest()
        self.assertEqual(capability["layer"], "generator_distribution")
        self.assertEqual(
            capability["execution_contract"],
            "sdk.guidance.GuidanceCapability",
        )
        self.assertFalse(capability["training"]["live_per_candidate_training"])
        access = capability["next_round_development_access"]
        self.assertEqual(access["kind"], "online_frozen_guidance_oracle")
        self.assertTrue(access["fresh_answer_first_sampling"])
        self.assertFalse(access["fixed_sample_quota"])
        self.assertIn("training_seed", capability["required_snapshot_bindings"])
        self.assertIn("training_budget", capability["required_snapshot_bindings"])
        self.assertEqual(
            capability["metric_cohorts"]["base_average"],
            "reported_separately",
        )
        self.assertEqual(
            capability["metric_cohorts"]["guided_adversarial"],
            "reported_separately",
        )

    def test_formal_solution_contract_is_two_phase_and_grid_independent(self):
        contract = CONTINUOUS_POINT_SOLUTION_CONTRACT
        self.assertEqual(
            contract["abi"], "prepare_solution_then_evaluate_points_v1"
        )
        self.assertEqual(
            contract["prepare_solution"]["internal_grid_or_basis"],
            "agent_selected",
        )
        self.assertFalse(
            contract["prepare_solution"]["scoring_coordinates_visible"]
        )
        self.assertTrue(contract["evaluate"]["points_capability_required"])
        self.assertTrue(
            contract["evaluate"]["query_latency_accounted_separately"]
        )


class GKWPublicProblemTests(unittest.TestCase):
    def test_gkw_exposes_real_operator_modules_and_free_coordinate_oracles(self):
        bundle = gkw_public_problem_bundle()
        paths = {resource.relative_path for resource in bundle.resources}
        self.assertEqual(
            paths,
            {
                "gkw_reference/__init__.py",
                "gkw_reference/gkw_equation_contract.py",
                "gkw_reference/gkw_public_packet.py",
                "gkw_reference/gkw_torch_precompute.py",
                "gkw_reference/gkw_torch_operator.py",
            },
        )
        operator = bundle.resource(
            "gkw_reference/gkw_torch_operator.py"
        ).data.decode("utf-8")
        precompute = bundle.resource(
            "gkw_reference/gkw_torch_precompute.py"
        ).data.decode("utf-8")
        self.assertIn("class GKWTorchSpectralOperator", operator)
        self.assertIn("def build_dataset_exact_gkw_operator", precompute)
        self.assertNotIn("gkw_selected_generator", "\n".join(paths))
        self.assertNotIn("decoder", "\n".join(paths))

        methods = {method.name: method for method in bundle.oracle_methods}
        self.assertEqual(set(methods), set(GKW_PUBLIC_SESSION_METHODS))
        source_schema = methods["query_source"].manifest()
        self.assertEqual(source_schema["coordinate_policy"], "agent_selected")
        self.assertFalse(source_schema["input_schema"]["fixed_scoring_panel"])
        arguments = source_schema["input_schema"]["arguments"]
        self.assertEqual(arguments[0]["values"], "Agent-selected")
        self.assertEqual(arguments[0]["accepted_shapes"], ["scalar", "[N]"])
        self.assertEqual(arguments[1]["shape"], ["N", 5])
        for method in methods.values():
            output = method.manifest()["output_schema"]
            self.assertEqual(output["dtype"], "float64")
            self.assertEqual(output["shape"], ["N", 2])
            self.assertEqual(output["component_order"], ["real", "imag"])
        self.assertTrue(
            source_schema["limits"][
                "larger_or_finer_agent_queries_consume_more_row_cost"
            ]
        )
        self.assertIsNone(
            source_schema["limits"]["session_total_row_quota"]
        )
        self.assertIn(
            "backend_wall_seconds",
            source_schema["limits"]["usage_accounting"],
        )
        self.assertTrue(
            source_schema["limits"]["agent_selected_source_times"]
        )
        self.assertTrue(
            source_schema["limits"][
                "legacy_session_limits_roll_over_transparently"
            ]
        )
        completed = {
            method.name: method
            for method in bundle.development_solution_oracle_methods
        }
        self.assertEqual(
            set(completed), set(GKW_DEVELOPMENT_SOLUTION_SESSION_METHODS)
        )
        completed_schema = completed["query_completed_macros"].manifest()
        self.assertFalse(
            completed_schema["input_schema"]["fixed_scoring_panel"]
        )
        self.assertIsNone(
            completed_schema["limits"]["session_total_point_quota"]
        )
        self.assertEqual(
            completed_schema["limits"]["actual_points"],
            "len(times) * len(field_indices)",
        )
        self.assertIn(
            "backend_wall_seconds",
            completed_schema["limits"]["usage_accounting"],
        )
        self.assertFalse(bundle.manifest()["guidance_capability"]["supported"])

    def test_gkw_macro_binding_publishes_bundle_and_exact_method_allowlist(self):
        plugin = bind_gkw_macro_loop(
            generator=GKWAnswerFirstGenerator(),
            base_state=object(),
            generator_sha256="b" * 64,
        )
        self.assertIs(plugin.public_problem_bundle, gkw_public_problem_bundle())
        self.assertEqual(
            plugin.public_session_methods,
            (
                "query_source",
                "query_initial_h",
                "query_initial_phi",
            ),
        )
        self.assertEqual(
            plugin.development_solution_session_methods,
            ("query_completed_macros",),
        )
        self.assertIsNot(
            plugin.public_session_factory,
            plugin.development_solution_session_factory,
        )


if __name__ == "__main__":
    unittest.main()
