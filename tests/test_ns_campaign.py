import json
import tempfile
import unittest
from pathlib import Path

from pde_agent_verifier_solver.plugins.ns import (
    NSCampaignCompiler,
    load_ns_campaign_config,
)


class NSCampaignTests(unittest.TestCase):
    def test_direct_compile_requires_explicit_legacy_staged_intent(self) -> None:
        with self.assertRaisesRegex(PermissionError, "legacy_staged_intent=True"):
            NSCampaignCompiler().compile(None)  # type: ignore[arg-type]

    def test_formal_config_compiles_the_existing_five_round_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "campaign.json"
            values = {
                "problem": "ns",
                "name": "formal NS",
                "dataset": {"format": "ns_d0_v1", "manifest": "dataset.json"},
                "implementation_root": "implementation",
                "execution": {"backend": "local", "run_root": "run"},
                "flow": {
                    "steps": 30000,
                    "checkpoint_every": 5000,
                },
                "pca": {"ranks": [64, 128, 256]},
                "campaign": {
                    "rounds": 5,
                    "cases_per_round": 1024,
                    "case_mix": {
                        "original": 256,
                        "generated": 576,
                        "perturbed": 192,
                    },
                },
                "perturbations": {
                    "pca_tangent": 48,
                    "spectral_complement": 96,
                    "pca_normal": 48,
                    "replay": 96,
                },
                "hidden": {
                    "initial_cases": 128,
                    "cases_per_round": 128,
                    "accumulate": True,
                },
            }
            values["agent"] = {
                "prompt_file": "prompt.md",
                "budget_seconds": 3600,
                "execution_mode": "persistent",
            }
            config_path.write_text(json.dumps(values))
            (config_path.parent / "prompt.md").write_text(
                "Prefer stable pressure recovery.", encoding="utf-8"
            )
            config = load_ns_campaign_config(config_path)
            plan = NSCampaignCompiler().compile(
                config,
                legacy_staged_intent=True,
            )

            self.assertEqual(len(plan.stages), 15)
            self.assertEqual(
                [stage.stage_id for stage in plan.stages],
                [
                    "project_train",
                    "project_validation",
                    "project_hidden_reference",
                    "fit_pca",
                    "select_pca",
                    "train_generator",
                    "audit_generator",
                    "select_generator",
                    "prepare_initial_hidden",
                    "adversarial_round_000",
                    "adversarial_round_001",
                    "adversarial_round_002",
                    "adversarial_round_003",
                    "adversarial_round_004",
                    "publish_result",
                ],
            )
            hidden_projection = plan.stages[2].commands[0].argv
            self.assertEqual(
                hidden_projection[hidden_projection.index("--split") + 1],
                "d0_distribution_test",
            )
            pca_selection = plan.stages[4].commands[0].argv
            self.assertEqual(pca_selection[-2:], ("--relative-tolerance", "0.05"))
            self.assertEqual(len(plan.stages[6].commands), 12)
            hidden_materialization = plan.stages[8].commands[1].argv
            self.assertIn("hidden_reference/coefficient_manifest.json", " ".join(hidden_materialization))
            first_round = plan.stages[9]
            first_loop = first_round.commands[0].argv
            generated = first_loop[first_loop.index("--generate-command") + 1]
            self.assertIn("--unperturbed-cases 576", generated)
            self.assertIn("--perturbation-cases 192", generated)
            self.assertIn("--replay-perturbation-cases 96", generated)
            self.assertEqual(first_loop[first_loop.index("--rounds") + 1], "1")
            self.assertEqual(
                first_loop[first_loop.index("--campaign-rounds-total") + 1], "5"
            )
            self.assertEqual(
                first_loop[first_loop.index("--round-index-offset") + 1], "0"
            )
            self.assertEqual(
                first_loop[first_loop.index("--cases-per-round") + 1], "1024"
            )
            self.assertEqual(
                first_loop[first_loop.index("--original-cases-per-round") + 1],
                "256",
            )
            self.assertEqual(
                first_loop[first_loop.index("--generated-cases-per-round") + 1],
                "768",
            )
            self.assertEqual(
                first_loop[
                    first_loop.index("--hidden-generated-cases-per-round") + 1
                ],
                "128",
            )
            self.assertIn("--hidden-accumulate", first_loop)
            self.assertEqual(
                first_loop[first_loop.index("--codex-persistent-search-seconds") + 1],
                "3600",
            )
            self.assertEqual(
                Path(first_loop[first_loop.index("--agent-prompt-file") + 1]).name,
                "prompt.md",
            )
            self.assertEqual(first_round.depends_on, ("prepare_initial_hidden",))

            final_round = plan.stages[13]
            final_loop = final_round.commands[0].argv
            self.assertEqual(
                final_loop[final_loop.index("--round-index-offset") + 1], "4"
            )
            self.assertEqual(final_round.depends_on, ("adversarial_round_003",))
            self.assertTrue(
                str(final_loop[final_loop.index("--initial-previous-submission") + 1])
                .endswith("round_003/submission.py")
            )
            self.assertTrue(
                any(
                    str(path).endswith("final_summary.json")
                    for path in final_round.expected_outputs
                )
            )
            self.assertEqual(plan.stages[-1].depends_on, ("adversarial_round_004",))


    def test_checked_in_template_matches_the_current_ns_experiment(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_ns_campaign_config(root / "configs" / "ns_structured_5round.json")
        plan = NSCampaignCompiler().compile(config, legacy_staged_intent=True)
        stages = {stage.stage_id: stage for stage in plan.stages}

        train = stages["project_train"].commands[0].argv
        validation = stages["project_validation"].commands[0].argv
        hidden_projection = stages["project_hidden_reference"].commands[0].argv
        self.assertEqual(train[train.index("--max-cases") + 1], "80000")
        self.assertEqual(validation[validation.index("--max-cases") + 1], "10000")
        self.assertEqual(
            hidden_projection[hidden_projection.index("--max-cases") + 1], "5000"
        )
        self.assertEqual(
            hidden_projection[hidden_projection.index("--split") + 1],
            "d0_distribution_test",
        )

        pca = stages["fit_pca"].commands[0].argv
        select_pca = stages["select_pca"].commands[0].argv
        flow = stages["train_generator"].commands[0].argv
        self.assertEqual(pca[pca.index("--ranks") + 1], "64:128:256")
        self.assertEqual(pca[pca.index("--validation-cases") + 1], "2048")
        self.assertEqual(
            select_pca[select_pca.index("--metric") + 1],
            "manifold_velocity_relative_l2",
        )
        self.assertIn("--rank-selection", flow)
        self.assertNotIn("--rank", flow)
        self.assertEqual(flow[flow.index("--steps") + 1], "30000")
        self.assertEqual(flow[flow.index("--validation-cases") + 1], "2048")
        self.assertTrue(
            str(stages["train_generator"].expected_outputs[0]).endswith(
                "flow_step_0030000.pt"
            )
        )
        generator_audit = stages["audit_generator"]
        self.assertEqual(len(generator_audit.commands), 12)
        self.assertIn("flow_step_0005000.pt", " ".join(generator_audit.commands[0].argv))
        self.assertIn("flow_step_0030000.pt", " ".join(generator_audit.commands[-2].argv))
        selected_generator = stages["select_generator"].commands[0].argv
        self.assertIn("--minimum-contract-pass-fraction", selected_generator)
        self.assertTrue(
            str(stages["select_generator"].expected_outputs[-1]).endswith(
                "selected_flow.pt"
            )
        )

        sample, materialize = stages["prepare_initial_hidden"].commands
        self.assertEqual(sample.argv[sample.argv.index("--samples") + 1], "2048")
        self.assertEqual(sample.argv[sample.argv.index("--seed") + 1], "20260810")
        self.assertEqual(
            materialize.argv[materialize.argv.index("--selection-seed") + 1],
            "20260811",
        )
        self.assertIn(
            "hidden_reference/coefficient_manifest.json", " ".join(materialize.argv)
        )

        loop = stages["adversarial_round_000"].commands[0].argv
        self.assertEqual(loop[loop.index("--rounds") + 1], "1")
        self.assertEqual(loop[loop.index("--campaign-rounds-total") + 1], "5")
        self.assertEqual(loop[loop.index("--cases-per-round") + 1], "1024")
        self.assertEqual(loop[loop.index("--original-cases-per-round") + 1], "256")
        self.assertEqual(loop[loop.index("--generated-cases-per-round") + 1], "768")
        self.assertEqual(loop[loop.index("--original-split") + 1], "adversarial_guard")
        self.assertEqual(
            loop[loop.index("--hidden-generated-cases-per-round") + 1], "128"
        )
        self.assertIn("--hidden-accumulate", loop)
        self.assertEqual(loop[loop.index("--codex-timeout-seconds") + 1], "3600")
        self.assertEqual(
            loop[loop.index("--codex-persistent-search-seconds") + 1], "3600"
        )
        self.assertEqual(loop[loop.index("--codex-model") + 1], "gpt-5.6-sol")
        self.assertEqual(loop[loop.index("--eval-workers") + 1], "4")
        self.assertIn("--solver-cache-guidance", loop)
        generated = loop[loop.index("--generate-command") + 1]
        self.assertIn("--unperturbed-cases 576", generated)
        self.assertIn("--perturbation-cases 192", generated)
        self.assertIn("--guided-fraction 0.5", generated)
        self.assertIn("--cuda-cache-clear-every 16", generated)
        hidden_generated = loop[loop.index("--hidden-generate-command") + 1]
        self.assertIn("--guided-fraction 0.5", hidden_generated)


if __name__ == "__main__":
    unittest.main()
