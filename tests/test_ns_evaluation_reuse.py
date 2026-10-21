import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "legacy" / "ns" / "scripts" / "run_triplane_agent_adversarial_loop.py"
SPEC = importlib.util.spec_from_file_location("ns_adversarial_loop", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
NS_LOOP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NS_LOOP)

REPORT_SCRIPT = (
    ROOT / "legacy" / "ns" / "scripts" / "summarize_ns_structured_guided_perturbed_5round.py"
)
REPORT_SPEC = importlib.util.spec_from_file_location("ns_mixed_report", REPORT_SCRIPT)
assert REPORT_SPEC is not None and REPORT_SPEC.loader is not None
NS_MIXED_REPORT = importlib.util.module_from_spec(REPORT_SPEC)
REPORT_SPEC.loader.exec_module(NS_MIXED_REPORT)

def _result(case_id: str, error: float) -> dict:
    return {
        "case_id": case_id,
        "status": "ok",
        "relative_l2": error,
        "difficulty_reward": error,
        "velocity_rmse": error,
        "truth_velocity_rms": 1.0,
        "train_scaled_velocity_rmse": error,
        "wall_time_seconds": 1.0,
    }


class NSEvaluationReuseTests(unittest.TestCase):
    def test_previous_solver_evaluates_only_the_new_round_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rounds = root / "rounds"
            previous_round = rounds / "round_000"
            current_round = rounds / "round_001"
            previous_private = previous_round / "private_eval"
            current_private = current_round / "private_eval"
            previous_private.mkdir(parents=True)
            current_private.mkdir(parents=True)

            old_case = root / "cases" / "old"
            new_case = root / "cases" / "new"
            old_case.mkdir(parents=True)
            new_case.mkdir(parents=True)
            previous_solver = previous_round / "submission.py"
            previous_solver.write_text("def solve(*args, **kwargs): return None\n")

            reward_spec = {"metric": "relative_l2"}
            previous_manifest = previous_private / "solver_feedback_private_manifest.json"
            previous_manifest.write_text(
                json.dumps(
                    {
                        "truth_contract": None,
                        "reward_spec": reward_spec,
                        "cases": [
                            {"case_id": "case_000000", "private_case_dir": str(old_case)}
                        ],
                    }
                )
            )
            previous_metrics = {
                "solver": str(previous_solver),
                "private_manifest": str(previous_manifest),
                "query_density": {"nx": 0, "ny": 0, "nt": 0},
                "solver_input_mode": "query",
                "reward_spec": reward_spec,
                "results": [_result("case_000000", 0.2)],
            }
            (previous_round / "cumulative_generated_solver_metrics.json").write_text(
                json.dumps(previous_metrics)
            )

            cumulative_manifest = current_private / "solver_feedback_private_manifest.json"
            cumulative_manifest.write_text(
                json.dumps(
                    {
                        "truth_contract": None,
                        "reward_spec": reward_spec,
                        "cases": [
                            {"case_id": "case_000000", "private_case_dir": str(old_case)},
                            {"case_id": "case_000001", "private_case_dir": str(new_case)},
                        ],
                    }
                )
            )
            current_manifest = current_private / "generated_private_manifest.json"
            current_manifest.write_text(
                json.dumps(
                    {
                        "truth_contract": None,
                        "reward_spec": reward_spec,
                        "cases": [
                            {"case_id": "case_000001", "private_case_dir": str(new_case)}
                        ],
                    }
                )
            )
            public_root = current_round / "solver_workspace" / "cases"
            public_root.mkdir(parents=True)
            new_evaluation = {
                "solver": str(previous_solver),
                "cases_root": str(public_root),
                "private_manifest": str(cumulative_manifest),
                "query_density": {"nx": 0, "ny": 0, "nt": 0},
                "solver_input_mode": "query",
                "truth_contract": None,
                "reward_spec": reward_spec,
                "results": [_result("case_000001", 0.4)],
            }

            with mock.patch.object(
                NS_LOOP, "evaluate_solver_file", return_value=new_evaluation
            ) as evaluator:
                payload = NS_LOOP._evaluate_previous_solver_on_cumulative_cases(
                    submission=previous_solver,
                    public_cases_root=public_root,
                    cumulative_manifest=cumulative_manifest,
                    current_manifest=current_manifest,
                    args=SimpleNamespace(
                        eval_nx=0, eval_ny=0, eval_nt=0, failure_reward=1.0e6
                    ),
                    label="previous_solver_on_cumulative_generated_replay",
                )

            self.assertEqual(
                evaluator.call_args.kwargs["selected_case_ids"], ["case_000001"]
            )
            self.assertEqual(
                [row["case_id"] for row in payload["results"]],
                ["case_000000", "case_000001"],
            )
            self.assertEqual(payload["summary"]["n"], 2)
            self.assertEqual(
                payload["evaluation_reuse"],
                {
                    "source": str(
                        previous_round / "cumulative_generated_solver_metrics.json"
                    ),
                    "reused_historical_cases": 1,
                    "evaluated_current_cases": 1,
                },
            )

    def test_original_d0_cases_are_deterministic_and_do_not_repeat(self) -> None:
        pool = [Path(f"/dataset/case_{index:03d}") for index in range(20)]

        round_zero = NS_LOOP._select_original_cases(
            pool, round_index=0, count=5, seed=17
        )
        round_one = NS_LOOP._select_original_cases(
            pool, round_index=1, count=5, seed=17
        )

        self.assertEqual(
            round_zero,
            NS_LOOP._select_original_cases(
                pool, round_index=0, count=5, seed=17
            ),
        )
        self.assertTrue(set(round_zero).isdisjoint(round_one))

    def test_mixed_numerical_and_continuous_cases_use_per_case_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            continuous = root / "continuous"
            numerical = root / "numerical"
            continuous.mkdir()
            numerical.mkdir()
            (continuous / "hidden_continuous_case.pt").touch()

            self.assertEqual(
                NS_LOOP._truth_contract_for_cases([continuous]),
                NS_LOOP.CONTINUOUS_MMS_TRUTH_CONTRACT,
            )
            self.assertIsNone(
                NS_LOOP._truth_contract_for_cases([continuous, numerical])
            )

    def test_generated_hidden_cohorts_accumulate_by_round(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for round_index in range(2):
                case = (
                    root
                    / "rounds"
                    / f"round_{round_index:03d}"
                    / "hidden_generation"
                    / "generated_cases"
                    / f"hidden_{round_index}"
                )
                case.mkdir(parents=True)
                (case / "public_question.json").write_text("{}")

            cases = NS_LOOP._load_historical_hidden_case_dirs(
                out=root, current_round_index=2
            )
            self.assertEqual([case.name for case in cases], ["hidden_0", "hidden_1"])

    def test_mixed_report_separates_original_generated_and_perturbed_cases(self) -> None:
        manifest = {
            "cases": [
                {"case_id": "case_000000", "private_case_dir": "/private/d0_000"},
                {"case_id": "case_000001", "private_case_dir": "/private/gen_000"},
                {"case_id": "case_000002", "private_case_dir": "/private/pert_000"},
            ]
        }
        receipt = {
            "cases": [
                {"case_id": "gen_000", "role": "current_unperturbed_guided"},
                {
                    "case_id": "pert_000",
                    "role": "perturbation",
                    "perturbation_family": "pca_tangent",
                },
            ]
        }
        evaluation = {
            "results": [
                _result("case_000000", 0.1),
                _result("case_000001", 0.2),
                _result("case_000002", 0.3),
            ]
        }

        groups, complete = NS_MIXED_REPORT._group_eval(
            evaluation=evaluation,
            manifest=manifest,
            receipt=receipt,
        )

        self.assertTrue(complete)
        self.assertEqual(groups["original"]["n"], 1)
        self.assertEqual(groups["unperturbed"]["n"], 1)
        self.assertEqual(groups["pca_tangent"]["n"], 1)
        self.assertEqual(groups["spectral_complement"]["n"], 0)
        self.assertEqual(groups["pca_normal"]["n"], 0)


if __name__ == "__main__":
    unittest.main()
