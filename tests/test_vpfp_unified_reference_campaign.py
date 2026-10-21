import json
import tempfile
import unittest
from pathlib import Path

from pde_agent_verifier_solver.plugins.vpfp.campaign import (
    MACRO_MATCH_CONTRACT,
    PRODUCT_ROOT,
    UNIFIED_MACRO_REFERENCE_PROTOCOL,
    VPFPCampaignCompiler,
    load_vpfp_campaign_config,
)


def _payload(root: Path) -> dict[str, object]:
    return {
        "schema": "pde_campaign_v1",
        "problem": "vpfp",
        "protocol": UNIFIED_MACRO_REFERENCE_PROTOCOL,
        "name": "VPFP unified macro fresh U0",
        "dataset": {"release_bank_report": "release.json"},
        "execution": {
            "backend": "slurm",
            "storage_root": "/scratch/vpfp-unified-tests",
            "run_root": "/scratch/vpfp-unified-tests/run-u0",
            "virtual_environment": str(root / "venv"),
            "account": "test-account",
            "qos": "normal",
            "preparation_partition": "gpubase_l40s_b3",
            "modules": ["scipy-stack"],
            "resources": {
                "macro_agent_round": {
                    "cpus": 6,
                    "memory": "24G",
                    "time": "07:00:00",
                }
            },
        },
        "generator": {"base_decoder_checkpoint": "base.pt"},
        "campaign": {
            "round_count": 1,
            "development_case_count": 4,
            "selection_case_count": 4,
            "promotion_case_count": 4,
            "terminal_case_count": 4,
            "seed_base": 8_240_000,
            "seed_stride": 10_000,
            "development_pool_multiplier": 1,
        },
        "agent": {
            "backend": "codex",
            "model": "gpt-5.6-sol",
            "reasoning_effort": "max",
            "timeout_seconds": 21_600,
            "development_case_credits": 16,
            "launcher": "launcher.sh",
            "codex_binary": "codex",
            "auth_file": "auth.json",
        },
        "evaluator": {
            "promotion_confidence_z": 0.0,
            "promotion_min_mean_improvement": 0.0,
            "promotion_min_win_rate": 0.0,
            "promotion_require_cvar_no_regression": False,
            "terminal_confidence_z": 0.0,
            "terminal_min_win_rate": 0.0,
            "terminal_require_cvar_no_regression": False,
        },
        "task": {
            "kind": "observable_model",
            "macro_match_contract": MACRO_MATCH_CONTRACT,
            "cold_start_model": "cold.py",
        },
    }


class VPFPUnifiedReferenceCampaignTests(unittest.TestCase):
    def test_compiles_one_normal_cpu_macro_agent_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "campaign.json"
            path.write_text(json.dumps(_payload(root)), encoding="utf-8")

            config = load_vpfp_campaign_config(path)
            plan = VPFPCampaignCompiler().compile(
                config,
                legacy_staged_intent=True,
            )

            self.assertEqual(config.protocol, UNIFIED_MACRO_REFERENCE_PROTOCOL)
            self.assertIsNone(config.expanded_decoder_checkpoint)
            self.assertIsNone(config.incumbent_model)
            self.assertEqual(config.implementation_root, PRODUCT_ROOT)
            self.assertEqual(plan.workdir, PRODUCT_ROOT)
            self.assertEqual(plan.implementation_root, PRODUCT_ROOT)
            self.assertEqual(len(plan.stages), 1)
            stage = plan.stages[0]
            self.assertEqual(stage.stage_id, "macro_agent_round")
            self.assertEqual(stage.resources.cpus, 6)
            self.assertEqual(stage.resources.memory, "24G")
            self.assertEqual(stage.resources.time_limit, "07:00:00")
            self.assertEqual(stage.resources.partition, "gpubase_l40s_b3")
            self.assertEqual(stage.resources.gres, "")
            self.assertEqual(
                stage.expected_outputs,
                (
                    config.run_root / "result" / "result.json",
                    config.run_root / "result" / "model" / "submission.py",
                ),
            )

            command = stage.commands[0]
            argv = command.argv
            self.assertEqual(
                argv[:3],
                (
                    str(root / "venv" / "bin" / "python"),
                    "-m",
                    "pde_agent_verifier_solver.plugins.vpfp.fresh_u0",
                ),
            )
            self.assertIn("--legacy-macro-reference", argv)
            self.assertEqual(
                argv[argv.index("--cold-start-candidate") + 1],
                str(root / "cold.py"),
            )
            self.assertEqual(
                argv[argv.index("--development-case-credits") + 1],
                "16",
            )
            self.assertEqual(
                argv[argv.index("--promotion-case-count") + 1],
                "4",
            )
            self.assertNotIn("--promotion-require-cvar-no-regression", argv)
            self.assertNotIn("--terminal-require-cvar-no-regression", argv)
            self.assertIn(
                "--no-promotion-require-cvar-no-regression",
                argv,
            )
            self.assertIn(
                "--no-terminal-require-cvar-no-regression", argv
            )
            self.assertEqual(
                command.environment["PDE_MACRO_MATCH_CONTRACT"],
                MACRO_MATCH_CONTRACT,
            )
            self.assertEqual(
                command.environment["PDE_MACRO_REFERENCE_ONLY"],
                "1",
            )
            self.assertEqual(plan.result_path, stage.expected_outputs[0])

    def test_rejects_nonfresh_backend_and_insufficient_credits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = (
                ("execution", "backend", "local", "Slurm backend"),
                (
                    "agent",
                    "development_case_credits",
                    3,
                    "cover at least one",
                ),
            )
            for section, key, value, message in cases:
                with self.subTest(section=section, key=key):
                    payload = _payload(root)
                    payload[section][key] = value  # type: ignore[index]
                    path = root / f"{section}-{key}.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        load_vpfp_campaign_config(path)


if __name__ == "__main__":
    unittest.main()
