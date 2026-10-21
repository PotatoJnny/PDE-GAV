import json
import tempfile
import unittest
from pathlib import Path

from pde_agent_verifier_solver.panel.campaign_view import build_campaign_view


class VPFPDashboardTests(unittest.TestCase):
    def test_parallel_agent_campaign_has_a_normal_final_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "vpfp-run"
            (root / "runtime").mkdir(parents=True)
            (root / "logs").mkdir()
            (root / "result" / "model").mkdir(parents=True)
            formal_root = root / "artifacts" / "formal_evaluation"
            formal_root.mkdir(parents=True)

            stages = [
                ("prepare_campaign", "Prepare campaign"),
                ("agent_a", "Run Agent A"),
                ("agent_b", "Run Agent B"),
                ("final_evaluation", "Compare final models"),
                ("publish_result", "Publish result"),
            ]
            result_path = root / "result" / "result.json"
            (root / "run.json").write_text(
                json.dumps(
                    {
                        "name": "VPFP two-Agent smoke",
                        "problem": "vpfp",
                        "backend": "slurm",
                        "created_at": "2026-08-20T10:00:00+00:00",
                        "result_path": str(result_path),
                        "stages": [
                            {"id": stage_id, "title": title, "commands": []}
                            for stage_id, title in stages
                        ],
                    }
                ),
                encoding="utf-8",
            )
            for position, (stage_id, _title) in enumerate(stages):
                (root / "runtime" / f"{stage_id}.json").write_text(
                    json.dumps(
                        {
                            "stage_id": stage_id,
                            "status": "completed",
                            "updated_at": f"2026-08-20T10:0{position}:00+00:00",
                            "finished_at": f"2026-08-20T10:0{position}:00+00:00",
                        }
                    ),
                    encoding="utf-8",
                )
            (root / "result" / "model" / "submission.py").write_text(
                "def solve_with_budget(case, times, budget):\n    return {}\n",
                encoding="utf-8",
            )
            (formal_root / "formal_summary.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "broad_root_case_count": 4,
                        "frontier_root_case_count": 2,
                        "wall_time_seconds": 42.0,
                    }
                ),
                encoding="utf-8",
            )
            result_path.write_text(
                json.dumps(
                    {
                        "kind": "observable_model",
                        "problem": "Vlasov--Poisson--Fokker--Planck",
                        "name": "Selected VPFP macroscopic model",
                        "winner": "vpfp_max_a",
                        "primary_error": 0.125,
                        "overall_improvement": 0.25,
                        "evaluated_cases": 16,
                        "wall_time_seconds": 12.5,
                        "usage": "solve_with_budget(public_case, query_times, budget)",
                        "memory": {"note": "Both Agent workspaces retain their ideas."},
                        "evaluation": {"agent_pair": {"winner": "vpfp_max_a"}},
                    }
                ),
                encoding="utf-8",
            )

            view = build_campaign_view(root)

            self.assertEqual(view["status"], "completed")
            self.assertEqual(view["problem"], "Vlasov–Poisson–Fokker–Planck")
            self.assertEqual(view["task"], "Observable model")
            self.assertEqual(view["rounds_completed"], 1)
            self.assertEqual(view["summary"]["hidden_error"], 0.125)
            self.assertEqual(view["summary"]["evaluated_cases"], 16)
            self.assertEqual(view["summary"]["improvement"], 0.25)
            self.assertEqual(view["result"]["primary_error"], 0.125)
            self.assertEqual(view["result"]["wall_time_seconds"], 12.5)
            self.assertEqual(view["result"]["evaluated_cases"], 16)
            self.assertNotIn("paired_objective", view["result"])
            self.assertNotIn("terminal", view["result"])
            self.assertTrue(view["result"]["artifact"]["available"])
            self.assertIn(
                "solve_with_budget",
                view["result"]["artifact"]["source"],
            )
            phases = {row["id"]: row for row in view["phases"]}
            self.assertIn("result", phases)
            self.assertNotIn(str(root), json.dumps(view))


if __name__ == "__main__":
    unittest.main()
