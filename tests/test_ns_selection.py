import importlib.util
import tempfile
import unittest
from pathlib import Path

from pde_agent_verifier_solver.plugins.ns.selection import (
    read_validation_losses,
    select_flow_checkpoint,
    select_pca_rank,
)


class NSPCASelectionTests(unittest.TestCase):
    def test_selects_smallest_rank_within_physical_tolerance(self) -> None:
        report = {
            "ranks": [
                _pca_row(64, 0.268, 0.9956),
                _pca_row(128, 0.104, 0.9993),
                _pca_row(256, 0.032, 0.99993),
            ]
        }

        selected = select_pca_rank(report, relative_tolerance=0.05)

        self.assertEqual(selected["selected_rank"], 256)
        self.assertEqual(
            [row["eligible"] for row in selected["candidates"]],
            [False, False, True],
        )

    def test_prefers_smaller_rank_when_reconstruction_is_effectively_tied(self) -> None:
        report = {"ranks": [_pca_row(64, 0.101), _pca_row(128, 0.100)]}

        selected = select_pca_rank(report, relative_tolerance=0.02)

        self.assertEqual(selected["selected_rank"], 64)


class NSFlowSelectionTests(unittest.TestCase):
    def test_rejects_physical_failure_before_ranking_metrics(self) -> None:
        reports = {
            15_000: _flow_reports(contract_passed=7, cases=8, shift=0.0),
            30_000: _flow_reports(contract_passed=8, cases=8, shift=0.1),
        }

        selected = select_flow_checkpoint(
            validation_losses={15_000: 0.40, 30_000: 0.42},
            candidate_reports=reports,
        )

        self.assertEqual(selected["selected_step"], 30_000)
        self.assertFalse(selected["candidates"][0]["eligible"])

    def test_validation_and_distribution_ranks_are_all_visible(self) -> None:
        reports = {
            15_000: _flow_reports(contract_passed=8, cases=8, shift=0.0),
            30_000: _flow_reports(contract_passed=8, cases=8, shift=0.2),
        }

        selected = select_flow_checkpoint(
            validation_losses={15_000: 0.46, 30_000: 0.40},
            candidate_reports=reports,
        )

        self.assertIn(selected["selected_step"], {15_000, 30_000})
        for row in selected["candidates"]:
            self.assertEqual(set(row["metric_ranks"]), set(selected["weights"]))
            self.assertIsInstance(row["rank_score"], float)

    def test_empirical_gate_then_uses_lowest_validation_loss(self) -> None:
        reports = {
            15_000: _flow_reports(contract_passed=8, cases=8, shift=0.0),
            30_000: _flow_reports(contract_passed=8, cases=8, shift=0.01),
        }

        selected = select_flow_checkpoint(
            validation_losses={15_000: 0.46, 30_000: 0.40},
            candidate_reports=reports,
            selection_rule="empirical_d0_gate",
            empirical_baseline_trials=_empirical_trials(),
        )

        self.assertEqual(selected["selected_step"], 30_000)
        self.assertEqual(
            selected["rule"],
            "empirical_d0_gate_then_lowest_validation_loss",
        )
        self.assertTrue(all(row["eligible"] for row in selected["candidates"]))
        self.assertEqual(selected["baseline_trial_count"], 3)

    def test_empirical_gate_fails_instead_of_forcing_a_bad_checkpoint(self) -> None:
        reports = {
            15_000: _flow_reports(contract_passed=8, cases=8, shift=0.30),
            30_000: _flow_reports(contract_passed=8, cases=8, shift=0.40),
        }

        with self.assertRaisesRegex(
            ValueError,
            "no Flow checkpoint passed the physical and empirical D0 gates",
        ):
            select_flow_checkpoint(
                validation_losses={15_000: 0.46, 30_000: 0.40},
                candidate_reports=reports,
                selection_rule="empirical_d0_gate",
                empirical_baseline_trials=_empirical_trials(),
            )

    def test_reads_validation_rows_by_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.jsonl"
            path.write_text(
                '{"event":"train","step":100,"loss":1.0}\n'
                '{"event":"validation","step":5000,"loss":0.51}\n'
                '{"event":"validation","step":10000,"loss":0.44}\n',
                encoding="utf-8",
            )

            self.assertEqual(read_validation_losses(path), {5000: 0.51, 10000: 0.44})

    def test_materialization_wires_empirical_baseline_repeat_count(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "legacy/ns/scripts/materialize_ns_structured_flow_samples.py"
        )
        spec = importlib.util.spec_from_file_location(
            "ns_materialization_baseline_test",
            script,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metrics = module.PHYSICAL_METRICS

        def rows(count, shift):
            return [
                {
                    "continuous_exact_metrics": {
                        name: 1.0 + shift + 0.01 * index + 0.001 * metric_index
                        for metric_index, name in enumerate(metrics)
                    }
                }
                for index in range(count)
            ]

        report = module._physical_distribution(
            generated_rows=rows(8, 0.2),
            reference_full_rows=rows(16, 0.1),
            reference_pca_rows=rows(16, 0.0),
            seed=7,
            baseline_repeats=2,
        )

        self.assertEqual(
            report["joint_log_metric_distribution"]["empirical_baseline_repeats"],
            2,
        )


def _pca_row(rank: int, error: float, explained: float = 0.99):
    return {
        "rank": rank,
        "explained_variance_fraction": explained,
        "physical_reconstruction": {
            "manifold_velocity_relative_l2": {"median": error}
        },
    }


def _empirical_trials():
    return [
        {
            "latent_mmd": 0.12 + offset,
            "latent_sliced_wasserstein": 0.22 + offset,
            "latent_coverage": 0.70 - offset,
            "physical_mmd": 0.17 + offset,
            "physical_sliced_wasserstein": 0.27 + offset,
            "physical_coverage": 0.67 - offset,
        }
        for offset in (-0.01, 0.0, 0.01)
    ]


def _flow_reports(*, contract_passed: int, cases: int, shift: float):
    sample = {
        "distribution": {
            "generated_vs_validation_prdc": {
                "coverage": 0.75 - shift,
                "precision": 0.8 - shift,
            },
            "generated_vs_validation_rbf_mmd": 0.1 + shift,
            "generated_vs_validation_sliced_wasserstein": 0.2 + shift,
        }
    }
    materialized = {
        "generated_cases": cases,
        "continuous_contract_passed": contract_passed,
        "physical_distribution": {
            "joint_log_metric_distribution": {
                "generated_vs_reference_pca_prdc": {
                    "coverage": 0.7 - shift,
                    "precision": 0.8 - shift,
                },
                "generated_vs_reference_pca_rbf_mmd": 0.15 + shift,
                "generated_vs_reference_pca_sliced_wasserstein": 0.25 + shift,
            }
        },
    }
    return sample, materialized


if __name__ == "__main__":
    unittest.main()
