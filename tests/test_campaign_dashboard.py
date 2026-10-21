import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from pde_agent_verifier_solver.panel.campaign_view import (
    build_campaign_view,
    build_dashboard_snapshot,
    build_hosted_dashboard_snapshot,
)
from pde_agent_verifier_solver.panel.staged_server import (
    DASHBOARD_HTML,
    _dashboard_handler,
    discover_staged_runs,
)


class CampaignDashboardTests(unittest.TestCase):
    def test_discovers_campaigns_and_puts_active_runs_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_run(root / "finished", name="Finished NS", status="completed")
            _make_run(root / "active", name="Active NS", status="running")
            (root / "not-a-run").mkdir()

            runs = discover_staged_runs(root)

            self.assertEqual([run["id"] for run in runs], ["active", "finished"])
            self.assertEqual(runs[0]["current_stage"], "Agent–Generator rounds")
            self.assertEqual(runs[0]["completed_stages"], 1)
            self.assertEqual(runs[0]["total_stages"], 2)
            self.assertGreaterEqual(runs[0]["progress_percent"], 50)
            self.assertEqual(runs[1]["status"], "completed")
            self.assertNotIn("run_root", runs[0])
            self.assertNotIn("backend", runs[0])

    def test_dashboard_api_contains_product_views_not_cluster_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_run(root / "ns-run", name="NS five rounds", status="running")
            server = ThreadingHTTPServer(("127.0.0.1", 0), _dashboard_handler(root))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            address = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with urlopen(f"{address}/", timeout=2) as response:
                    page = response.read().decode()
                with urlopen(f"{address}/api/runs", timeout=2) as response:
                    overview = json.load(response)
                with urlopen(f"{address}/api/runs/ns-run", timeout=2) as response:
                    detail = json.load(response)

                self.assertIn("Running now", page)
                self.assertIn("Recent results", page)
                self.assertNotIn("Stage log", page)
                self.assertNotIn("run_root", page)
                self.assertIn("api/runs", DASHBOARD_HTML)
                self.assertEqual(overview["schema"], "pde_panel_snapshot_v1")
                self.assertEqual(overview["runs"][0]["name"], "NS five rounds")
                self.assertEqual(detail["id"], "ns-run")
                self.assertEqual(detail["phases"][1]["status"], "running")
                serialized = json.dumps(detail)
                self.assertNotIn(str(root), serialized)
                self.assertNotIn("commands", serialized)
                self.assertNotIn("log_tail", serialized)
                self.assertNotIn("job_id", serialized)
                with self.assertRaises(HTTPError) as missing:
                    urlopen(f"{address}/api/runs/missing", timeout=2)
                self.assertEqual(missing.exception.code, 404)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_completed_result_exposes_metrics_and_download_not_a_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "result-run"
            _make_run(run_root, name="Finished solver", status="completed")
            loop = run_root / "campaign" / "agent_adversarial_loop"
            round_root = loop / "rounds" / "round_000"
            round_root.mkdir(parents=True)
            summary = {
                "round": 0,
                "generated_case_count": 2,
                "historical_replay_case_count": 0,
                "solver_feedback_case_count": 2,
                "generated_summary": _metrics(0.5, 2),
                "previous_solver_generated_summary": _metrics(0.8, 2),
                "cumulative_generated_summary": _metrics(0.5, 2),
                "hidden_holdout_summary": _metrics(0.6, 1),
                "solver_improvement_on_generated": 0.375,
                "hard_verification_summary": {"accepted": 2, "n": 2},
                "policy_update": {"enabled": True},
            }
            (round_root / "round_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            (round_root / "hidden_holdout_solver_metrics.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {"case_id": "hidden_1", "relative_l2": 0.6}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (round_root / "cumulative_generated_solver_metrics.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {"case_id": "generated_1", "relative_l2": 0.4},
                            {"case_id": "generated_2", "relative_l2": 0.6},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (loop / "final_summary.json").write_text(
                json.dumps(
                    {
                        "rounds_requested": 1,
                        "rounds_completed": 1,
                        "cases_per_round": 2,
                        "wall_time_seconds": 12.5,
                    }
                ),
                encoding="utf-8",
            )
            result_root = run_root / "result"
            (result_root / "solver").mkdir(parents=True)
            (result_root / "solver" / "submission.py").write_text(
                "def solve_case(case):\n    return case\n",
                encoding="utf-8",
            )
            result_file = result_root / "result.json"
            result_file.write_text(
                json.dumps(
                    {
                        "kind": "solver",
                        "problem": "Navier--Stokes",
                        "name": "Selected solver",
                        "primary_error": 0.6,
                        "solver": str(result_root / "solver" / "submission.py"),
                    }
                ),
                encoding="utf-8",
            )
            run_payload = json.loads((run_root / "run.json").read_text())
            run_payload["result_path"] = str(result_file)
            (run_root / "run.json").write_text(
                json.dumps(run_payload), encoding="utf-8"
            )

            detail = build_campaign_view(run_root)
            snapshot = build_dashboard_snapshot(root)
            hosted = build_hosted_dashboard_snapshot(root)
            serialized = json.dumps(snapshot)
            hosted_serialized = json.dumps(hosted)

            self.assertEqual(detail["result"]["primary_error"], 0.6)
            self.assertEqual(detail["result"]["rounds"][0]["improvement"], 0.375)
            self.assertTrue(detail["result"]["artifact"]["available"])
            self.assertIn("def solve_case", detail["result"]["artifact"]["source"])
            self.assertNotIn(str(run_root), serialized)
            self.assertNotIn('"solver":', serialized)
            self.assertNotIn('"memory": "', serialized)
            self.assertNotIn("def solve_case", hosted_serialized)
            self.assertNotIn("hidden_1", hosted_serialized)
            self.assertNotIn("generated_1", hosted_serialized)
            self.assertFalse(hosted["runs"][0]["result"]["artifact"]["available"])

    def test_standalone_macro_result_projects_objective_split_and_guardrail(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runs_root = Path(temporary)
            run_root = runs_root / "macro-result"
            _make_run(run_root, name="Fresh GKW", status="completed")
            result_file = run_root / "result" / "result.json"
            result_file.parent.mkdir(parents=True)

            def aggregate(mean: float) -> dict[str, float]:
                return {"mean": mean, "cvar": mean + 0.1, "max": mean + 0.2}

            result_file.write_text(
                json.dumps(
                    {
                        "kind": "macro_target_rom",
                        "name": "Selected GKW macroscopic model",
                        "primary_error": 0.4,
                        "rounds_completed": 4,
                        "overall_improvement": None,
                        "active_objective_improvement": 0.5,
                        "terminal_acceptance": False,
                        "macro_objective": {
                            "paired_objective": {
                                "mode": "primary_with_perturbed_guardrail",
                                "base_weight": 0.5,
                                "perturbed_weight": 0.5,
                                "guardrail_relative_margin": 0.05,
                                "guardrail_confidence_z": 1.6448536269514722,
                            }
                        },
                        "terminal": {
                            "cohort_id": "private-terminal-cohort",
                            "active_objective": "generated",
                            "distribution_aggregates": {
                                "cold_start": {
                                    "generated": aggregate(0.8),
                                    "perturbed": aggregate(0.9),
                                    "composite": aggregate(0.85),
                                },
                                "selected": {
                                    "generated": aggregate(0.4),
                                    "perturbed": aggregate(1.0),
                                    "composite": aggregate(0.7),
                                },
                            },
                            "perturbed_guardrail": {
                                "root_count": 256,
                                "candidate_mean": 1.0,
                                "incumbent_mean": 0.9,
                                "mean_excess": 0.055,
                                "standard_error": 0.01,
                                "confidence_lower": 0.0385,
                                "confidence_z": 1.6448536269514722,
                                "practical_relative_margin": 0.05,
                                "passed": False,
                                "failures": ["perturbed_regression"],
                            },
                            "supports_selected_over_cold_start": False,
                            "selected": {
                                "candidate": {
                                    "candidate_id": "private-selected-id"
                                }
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            run_payload = json.loads((run_root / "run.json").read_text())
            run_payload["result_path"] = str(result_file)
            (run_root / "run.json").write_text(
                json.dumps(run_payload), encoding="utf-8"
            )

            detail = build_campaign_view(run_root)
            hosted = build_hosted_dashboard_snapshot(runs_root)
            result = detail["result"]
            terminal = result["terminal"]
            selected = terminal["distribution_aggregates"]["selected"]

            self.assertEqual(detail["rounds_completed"], 4)
            self.assertEqual(
                result["paired_objective"]["mode"],
                "primary_with_perturbed_guardrail",
            )
            self.assertEqual(terminal["active_objective"], "generated")
            self.assertEqual(selected["generated"]["mean"], 0.4)
            self.assertFalse(terminal["perturbed_guardrail"]["passed"])
            self.assertFalse(
                terminal["supports_selected_over_cold_start"]
            )
            self.assertIsNone(result["overall_improvement"])
            self.assertEqual(result["active_objective_improvement"], 0.5)
            self.assertFalse(result["terminal_acceptance"])
            self.assertEqual(
                terminal["perturbed_guardrail"]["failures"],
                ["perturbed_regression"],
            )
            serialized = json.dumps(detail)
            self.assertNotIn("private-terminal-cohort", serialized)
            self.assertNotIn("private-selected-id", serialized)
            hosted_result = hosted["runs"][0]["result"]
            self.assertFalse(
                hosted_result["terminal"]["perturbed_guardrail"]["passed"]
            )

    def test_one_job_per_round_still_appears_as_one_five_round_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "split-rounds"
            (run_root / "runtime").mkdir(parents=True)
            (run_root / "logs").mkdir()
            stages = [
                {
                    "id": "prepare_hidden",
                    "title": "Prepare hidden cases",
                    "commands": [],
                }
            ]
            for round_index in range(5):
                stages.append(
                    {
                        "id": f"adversarial_round_{round_index:03d}",
                        "title": f"Run round {round_index + 1}",
                        "commands": [
                            [
                                "python",
                                "loop.py",
                                "--rounds",
                                "1",
                                "--campaign-rounds-total",
                                "5",
                                "--cases-per-round",
                                "1024",
                            ]
                        ],
                    }
                )
            (run_root / "run.json").write_text(
                json.dumps(
                    {
                        "name": "NS five rounds",
                        "problem": "ns",
                        "backend": "slurm",
                        "created_at": "2026-08-20T10:00:00+00:00",
                        "stages": stages,
                    }
                ),
                encoding="utf-8",
            )
            for stage_id, status in (
                ("prepare_hidden", "completed"),
                ("adversarial_round_000", "completed"),
                ("adversarial_round_001", "running"),
            ):
                (run_root / "runtime" / f"{stage_id}.json").write_text(
                    json.dumps({"stage_id": stage_id, "status": status}),
                    encoding="utf-8",
                )
            rounds_root = run_root / "campaign" / "agent_adversarial_loop" / "rounds"
            first_round = rounds_root / "round_000"
            second_round = rounds_root / "round_001"
            first_round.mkdir(parents=True)
            second_round.mkdir()
            (first_round / "round_summary.json").write_text(
                json.dumps({"round": 0, "generated_case_count": 1024}),
                encoding="utf-8",
            )

            detail = build_campaign_view(run_root)

            self.assertEqual(detail["rounds_requested"], 5)
            self.assertEqual(detail["rounds_completed"], 1)
            self.assertEqual(detail["live"]["current_round"], 2)
            self.assertIn("round 2", detail["current_activity"].lower())


def _metrics(mean: float, cases: int) -> dict[str, object]:
    return {
        "mean_relative_l2": mean,
        "median_relative_l2": mean,
        "q10_relative_l2": mean - 0.01,
        "q90_relative_l2": mean + 0.01,
        "min_relative_l2": mean - 0.02,
        "max_relative_l2": mean + 0.02,
        "n": cases,
        "ok": cases,
    }


def _make_run(run_root: Path, *, name: str, status: str) -> None:
    (run_root / "runtime").mkdir(parents=True)
    (run_root / "logs").mkdir()
    stages = [
        {"id": "prepare", "title": "Prepare data", "commands": []},
        {"id": "agent_rounds", "title": "Agent rounds", "commands": []},
    ]
    (run_root / "run.json").write_text(
        json.dumps(
            {
                "name": name,
                "problem": "ns",
                "backend": "slurm",
                "created_at": "2026-08-20T10:00:00+00:00",
                "stages": stages,
            }
        ),
        encoding="utf-8",
    )
    stage_statuses = ["completed", status]
    for index, (stage, stage_status) in enumerate(zip(stages, stage_statuses)):
        (run_root / "runtime" / f"{stage['id']}.json").write_text(
            json.dumps(
                {
                    "stage_id": stage["id"],
                    "status": stage_status,
                    "updated_at": f"2026-08-20T10:0{index}:00+00:00",
                    "finished_at": (
                        f"2026-08-20T10:0{index}:00+00:00"
                        if stage_status in {"completed", "failed"}
                        else None
                    ),
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
