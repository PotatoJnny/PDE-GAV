import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pde_agent_verifier_solver.cli import _load_legacy_staged_config
from pde_agent_verifier_solver.plugins.vpfp import (
    VPFPCampaignCompiler,
    load_vpfp_campaign_config,
)


MACRO_CONTRACT = "vpfp_full_velocity_macro_match_v1"
MACRO_VALIDATION = "formal_dual_score_v1"
MACRO_ENVIRONMENT_VARIABLE = "VPFP_MACRO_MATCH_CONTRACT"


def _minimal_payload() -> dict[str, object]:
    return {
        "problem": "vpfp",
        "name": "VPFP validation test",
        "dataset": {"release_bank_report": "release.json"},
        "execution": {
            "backend": "slurm",
            "run_root": "/scratch/vpfp-validation-test",
            "virtual_environment": "venv",
        },
        "generator": {
            "base_decoder_checkpoint": "base.pt",
            "expanded_decoder_checkpoint": "expanded.pt",
        },
        "agent": {"backend": "codex_mock", "budget_seconds": 60},
        "evaluator": {
            "broad_cases": 4,
            "frontier_cases": 2,
            "macro_match_validation": MACRO_VALIDATION,
        },
        "task": {
            "kind": "observable_model",
            "incumbent_model": "submission.py",
            "macro_match_contract": MACRO_CONTRACT,
        },
    }


class VPFPCampaignTests(unittest.TestCase):
    def test_direct_compile_requires_explicit_legacy_staged_intent(self) -> None:
        with self.assertRaisesRegex(PermissionError, "legacy_staged_intent=True"):
            VPFPCampaignCompiler().compile(None)  # type: ignore[arg-type]

    def test_recipe_compiles_six_ordered_stages_and_macro_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "campaign.json"
            config_path.write_text(
                json.dumps(
                    {
                        "problem": "vpfp",
                        "name": "VPFP smoke",
                        "dataset": {"release_bank_report": "release.json"},
                        "implementation_root": "implementation",
                        "execution": {
                            "backend": "slurm",
                            "run_root": "/scratch/vpfp-unit-test",
                            "virtual_environment": "venv",
                        },
                        "generator": {
                            "base_decoder_checkpoint": "base.pt",
                            "expanded_decoder_checkpoint": "expanded.pt",
                            "generation_batch_size": 16,
                            "guidance_pool_size": 4,
                        },
                        "campaign": {"round_index": 1, "seed_base": 3600000},
                        "agent": {
                            "backend": "codex_mock",
                            "budget_seconds": 60,
                            "finalist_timeout_seconds": 300,
                            "development_case_credits": 64,
                            "prompt_file": "prompt.md",
                        },
                        "evaluator": {
                            "broad_cases": 4,
                            "frontier_cases": 2,
                            "macro_match_validation": MACRO_VALIDATION,
                        },
                        "task": {
                            "kind": "observable_model",
                            "incumbent_model": "submission.py",
                            "macro_match_contract": MACRO_CONTRACT,
                        },
                    }
                )
            )
            config = load_vpfp_campaign_config(config_path)
            plan = VPFPCampaignCompiler().compile(
                config,
                legacy_staged_intent=True,
            )

            self.assertEqual(
                [stage.stage_id for stage in plan.stages],
                [
                    "prepare_campaign",
                    "agent_a",
                    "agent_b",
                    "final_evaluation",
                    "validate_macro_contract",
                    "publish_result",
                ],
            )
            self.assertEqual(plan.stages[1].depends_on, ("prepare_campaign",))
            self.assertEqual(plan.stages[2].depends_on, ("prepare_campaign",))
            self.assertEqual(
                plan.stages[3].depends_on, ("agent_a", "agent_b")
            )
            self.assertEqual(plan.stages[4].depends_on, ("final_evaluation",))
            self.assertEqual(
                plan.stages[5].depends_on, ("validate_macro_contract",)
            )
            prepare = plan.stages[0].commands[0].argv
            self.assertIn("--agent-prompt-file", prepare)
            self.assertEqual(prepare[prepare.index("--round-index") + 1], "1")
            self.assertEqual(
                prepare[prepare.index("--guidance-pool-size") + 1], "4"
            )
            for stage in plan.stages[1:3]:
                self.assertIn("--mock-agent", stage.commands[0].argv)

            for stage in plan.stages:
                for command in stage.commands:
                    if stage.stage_id == "final_evaluation":
                        self.assertEqual(
                            command.environment[MACRO_ENVIRONMENT_VARIABLE],
                            MACRO_CONTRACT,
                        )
                    else:
                        self.assertNotIn(
                            MACRO_ENVIRONMENT_VARIABLE, command.environment
                        )

            audit_path = (
                config.run_root
                / "artifacts"
                / "formal_evaluation"
                / "macro_contract_validation.json"
            )
            validate = plan.stages[4]
            self.assertEqual(validate.expected_outputs, (audit_path,))
            validate_argv = validate.commands[0].argv
            self.assertEqual(
                validate_argv[:3],
                (
                    str(config.virtual_environment / "bin" / "python"),
                    "-m",
                    "pde_agent_verifier_solver.plugins.vpfp.macro_contract_audit",
                ),
            )

            self.assertEqual(
                validate_argv[validate_argv.index("--formal-root") + 1],
                str(config.run_root / "artifacts" / "formal_evaluation"),
            )
            self.assertEqual(
                validate_argv[validate_argv.index("--out") + 1],
                str(audit_path),
            )
            publish_argv = plan.stages[5].commands[0].argv
            self.assertEqual(
                publish_argv[publish_argv.index("--macro-contract-audit") + 1],
                str(audit_path),
            )

    def test_wrong_macro_contract_or_validation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = (
                (
                    "task",
                    "macro_match_contract",
                    "wrong",
                    r"task\.macro_match_contract",
                ),
                (
                    "evaluator",
                    "macro_match_validation",
                    "wrong",
                    r"evaluator\.macro_match_validation",
                ),
            )
            for section, key, value, message in cases:
                with self.subTest(section=section, key=key):
                    payload = _minimal_payload()
                    payload[section][key] = value  # type: ignore[index]
                    path = root / f"{section}.json"
                    path.write_text(json.dumps(payload))
                    with self.assertRaisesRegex(ValueError, message):
                        load_vpfp_campaign_config(path)

    def test_environment_paths_expand_before_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset_root = root / "external-assets"
            payload = _minimal_payload()
            payload["dataset"] = {
                "release_bank_report": "$VPFP_TEST_ASSET_ROOT/release.json"
            }
            payload["execution"] = {
                "backend": "slurm",
                "run_root": "$VPFP_TEST_ASSET_ROOT/runs/current",
                "storage_root": "$VPFP_TEST_ASSET_ROOT/storage",
                "virtual_environment": "$VPFP_TEST_ASSET_ROOT/venv",
            }
            path = root / "environment.json"
            path.write_text(json.dumps(payload))

            with patch.dict(
                os.environ,
                {"VPFP_TEST_ASSET_ROOT": str(asset_root)},
            ):
                config = load_vpfp_campaign_config(path)
                plan = VPFPCampaignCompiler().compile(
                    config,
                    legacy_staged_intent=True,
                )

            self.assertEqual(
                config.release_bank_report,
                asset_root / "release.json",
            )
            self.assertEqual(
                config.run_root,
                asset_root / "runs" / "current",
            )
            self.assertEqual(
                plan.stages[0].commands[0].environment["VPFP_RUN_STORAGE_ROOT"],
                str(asset_root / "storage"),
            )

    def test_shared_cli_loads_vpfp_through_the_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "campaign.json"
            path.write_text(
                json.dumps(
                    {
                        "problem": "vpfp",
                        "name": "registry test",
                        "dataset": {"manifest": "release.json"},
                        "execution": {
                            "backend": "slurm",
                            "run_root": "/scratch/vpfp-registry-test",
                            "virtual_environment": "venv",
                        },
                        "generator": {
                            "base_decoder_checkpoint": "base.pt",
                            "expanded_decoder_checkpoint": "expanded.pt",
                        },
                        "agent": {"backend": "codex_mock", "budget_seconds": 60},
                        "evaluator": {"broad_cases": 4, "frontier_cases": 2},
                        "task": {
                            "kind": "observable_model",
                            "incumbent_model": "submission.py",
                        },
                    }
                )
            )
            plugin, config = _load_legacy_staged_config(
                path, dataset=None, run_root=None, backend=None
            )
            self.assertEqual(plugin.plugin_id, "vpfp")
            self.assertEqual(config.name, "registry test")


if __name__ == "__main__":
    unittest.main()
