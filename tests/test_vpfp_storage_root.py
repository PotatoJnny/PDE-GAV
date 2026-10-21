import json
import tempfile
import unittest
from pathlib import Path

from pde_agent_verifier_solver.plugins.vpfp import (
    VPFPCampaignCompiler,
    load_vpfp_campaign_config,
)


class VPFPStorageRootTests(unittest.TestCase):
    def test_configured_run_storage_reaches_every_stage(self) -> None:
        """Project storage is a supported alternative to shared scratch."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            storage = root / "project_runs"
            config_path = root / "campaign.json"
            config_path.write_text(
                json.dumps(
                    {
                        "problem": "vpfp",
                        "name": "project storage test",
                        "dataset": {"release_bank_report": "release.json"},
                        "implementation_root": "implementation",
                        "execution": {
                            "backend": "slurm",
                            "storage_root": str(storage),
                            "run_root": str(storage / "one_run"),
                            "virtual_environment": "venv",
                        },
                        "generator": {
                            "base_decoder_checkpoint": "base.pt",
                            "expanded_decoder_checkpoint": "expanded.pt",
                            "generation_batch_size": 16,
                            "guidance_pool_size": 4,
                        },
                        "campaign": {
                            "round_index": 2,
                            "seed_base": 3_600_000,
                        },
                        "agent": {
                            "backend": "codex_mock",
                            "budget_seconds": 60,
                            "finalist_timeout_seconds": 300,
                            "development_case_credits": 64,
                        },
                        "evaluator": {
                            "broad_cases": 4,
                            "frontier_cases": 2,
                        },
                        "task": {
                            "kind": "observable_model",
                            "incumbent_model": "submission.py",
                        },
                    }
                ),
                encoding="utf-8",
            )

            plan = VPFPCampaignCompiler().compile(
                load_vpfp_campaign_config(config_path),
                legacy_staged_intent=True,
            )
            for stage in plan.stages:
                for command in stage.commands:
                    self.assertEqual(
                        command.environment["VPFP_RUN_STORAGE_ROOT"],
                        str(storage),
                    )

            prepare = plan.stages[0].commands[0].argv
            round_position = prepare.index("--round-index")
            self.assertEqual(prepare[round_position + 1], "2")
            self.assertEqual(prepare[round_position + 2], "--seed-base")


if __name__ == "__main__":
    unittest.main()
