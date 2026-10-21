import json
import math
import tempfile
import unittest
from pathlib import Path

from pde_agent_verifier_solver.plugins.vpfp.macro_contract_audit import (
    AUDIT_SCHEMA,
    BRIDGE_RESULT_KEY,
    VPFPMacroContractAuditError,
    validate_vpfp_formal_macro_contract,
    write_vpfp_formal_macro_contract_audit,
)
from pde_agent_verifier_solver.plugins.vpfp.legacy_macro_bridge import (
    LEGACY_MACRO_BRIDGE_SCHEMA,
)
from pde_agent_verifier_solver.plugins.vpfp.macro_matching import (
    VPFPMacroMatcher,
)


class VPFPMacroContractAuditTests(unittest.TestCase):
    def test_writes_passing_sidecar_for_complete_formal_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            formal = self._formal_tree(Path(temporary))
            output = formal / "macro_contract_validation.json"

            report = write_vpfp_formal_macro_contract_audit(formal, output)

            self.assertTrue(report["passed"])
            self.assertEqual(report["schema"], AUDIT_SCHEMA)
            self.assertEqual(
                report["contract_id"], VPFPMacroMatcher.contract_id
            )
            self.assertEqual(report["stage_report_count"], 4)
            self.assertEqual(report["total_call_count"], 24)
            self.assertEqual(report["accepted_call_count"], 24)
            self.assertEqual(report["audited_macro_match_count"], 24)
            self.assertFalse(report["used_for_winner_selection"])
            self.assertEqual(json.loads(output.read_text()), report)

    def test_rejects_accepted_call_missing_bridge_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            formal = self._formal_tree(Path(temporary))
            report_path = self._stage_report(formal, "agent_pair", "broad")
            report = json.loads(report_path.read_text())
            call = report["root_case_rows"][0]["variants"]["base"]["solvers"][
                "candidate"
            ]["budgets"][0]
            del call[BRIDGE_RESULT_KEY]
            report_path.write_text(json.dumps(report))

            with self.assertRaisesRegex(
                VPFPMacroContractAuditError,
                "missing unified_macro_match",
            ):
                validate_vpfp_formal_macro_contract(formal)

    def test_rejects_wrong_contract_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            formal = self._formal_tree(Path(temporary))
            report_path = self._stage_report(formal, "agent_pair", "frontier")
            report = json.loads(report_path.read_text())
            bridge = report["root_case_rows"][0]["variants"]["base"][
                "solvers"
            ]["candidate"]["budgets"][0][BRIDGE_RESULT_KEY]
            bridge["contract"] = "other-contract"
            report_path.write_text(json.dumps(report))

            with self.assertRaisesRegex(
                VPFPMacroContractAuditError,
                "wrong macro contract id",
            ):
                validate_vpfp_formal_macro_contract(formal)

    def test_rejects_nonfinite_primary_or_metric(self) -> None:
        for target in ("primary_error", "metric"):
            with (
                self.subTest(target=target),
                tempfile.TemporaryDirectory() as temporary,
            ):
                formal = self._formal_tree(Path(temporary))
                report_path = self._stage_report(
                    formal, "champion_pair", "broad"
                )
                report = json.loads(report_path.read_text())
                bridge = report["root_case_rows"][0]["variants"]["perturbed"][
                    "solvers"
                ]["comparison"]["budgets"][0][BRIDGE_RESULT_KEY]
                if target == "primary_error":
                    bridge["primary_error"] = math.nan
                else:
                    bridge["metrics"]["rho_relative_l2"] = math.inf
                report_path.write_text(json.dumps(report))

                with self.assertRaisesRegex(
                    VPFPMacroContractAuditError,
                    "finite and nonnegative",
                ):
                    validate_vpfp_formal_macro_contract(formal)

    def test_rejects_partial_variants_solvers_or_budgets(self) -> None:
        mutations = ("variant", "solver", "budget")
        for mutation in mutations:
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as temporary,
            ):
                formal = self._formal_tree(Path(temporary))
                report_path = self._stage_report(
                    formal, "champion_pair", "frontier"
                )
                report = json.loads(report_path.read_text())
                root = report["root_case_rows"][0]
                if mutation == "variant":
                    del root["variants"]["perturbed"]
                elif mutation == "solver":
                    del root["variants"]["base"]["solvers"]["comparison"]
                else:
                    root["variants"]["base"]["solvers"]["candidate"][
                        "budgets"
                    ].clear()
                report_path.write_text(json.dumps(report))

                with self.assertRaisesRegex(
                    VPFPMacroContractAuditError,
                    "partial",
                ):
                    validate_vpfp_formal_macro_contract(formal)

    def test_rejects_missing_required_component_metric(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            formal = self._formal_tree(Path(temporary))
            report_path = self._stage_report(formal, "agent_pair", "broad")
            report = json.loads(report_path.read_text())
            bridge = report["root_case_rows"][0]["variants"]["base"][
                "solvers"
            ]["candidate"]["budgets"][0][BRIDGE_RESULT_KEY]
            del bridge["metrics"]["momentum_x_relative_l2"]
            report_path.write_text(json.dumps(report))

            with self.assertRaisesRegex(
                VPFPMacroContractAuditError,
                "missing required metrics.*momentum_x_relative_l2",
            ):
                validate_vpfp_formal_macro_contract(formal)

    def test_rejects_binding_changes_within_variant(self) -> None:
        for mutation in ("solver", "budget"):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as temporary,
            ):
                formal = self._formal_tree(Path(temporary))
                report_path = self._stage_report(
                    formal, "agent_pair", "frontier"
                )
                report = json.loads(report_path.read_text())
                solvers = report["root_case_rows"][0]["variants"]["base"][
                    "solvers"
                ]
                if mutation == "solver":
                    bridge = solvers["comparison"]["budgets"][0][
                        BRIDGE_RESULT_KEY
                    ]
                    bridge["case_id"] = "different-case"
                else:
                    bridge = solvers["candidate"]["budgets"][1][
                        BRIDGE_RESULT_KEY
                    ]
                    bridge["coordinate_digest"] = "sha256:" + "b" * 64
                report_path.write_text(json.dumps(report))

                with self.assertRaisesRegex(
                    VPFPMacroContractAuditError,
                    "changed the case or coordinate binding",
                ):
                    validate_vpfp_formal_macro_contract(formal)

    def test_rejects_binding_changes_between_formal_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            formal = self._formal_tree(Path(temporary))
            report_path = self._stage_report(formal, "champion_pair", "broad")
            report = json.loads(report_path.read_text())
            solvers = report["root_case_rows"][0]["variants"]["base"][
                "solvers"
            ]
            for solver in solvers.values():
                for call in solver["budgets"]:
                    call[BRIDGE_RESULT_KEY]["coordinate_digest"] = (
                        "sha256:" + "b" * 64
                    )
            report_path.write_text(json.dumps(report))

            with self.assertRaisesRegex(
                VPFPMacroContractAuditError,
                "broad macro bindings changed between formal pairs",
            ):
                validate_vpfp_formal_macro_contract(formal)

    def test_rejects_invalid_per_call_budget_identity(self) -> None:
        mutations = (
            ("missing", "budget identity is missing"),
            ("duplicate", "does not match its stage budget"),
            ("out_of_order", "does not match its stage budget"),
            ("wrong_value", "does not match its stage budget"),
        )
        for mutation, message in mutations:
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as temporary,
            ):
                formal = self._formal_tree(Path(temporary))
                report_path = self._stage_report(
                    formal, "agent_pair", "frontier"
                )
                report = json.loads(report_path.read_text())
                calls = report["root_case_rows"][0]["variants"]["base"][
                    "solvers"
                ]["candidate"]["budgets"]
                if mutation == "missing":
                    del calls[0]["budget"]
                elif mutation == "duplicate":
                    calls[1]["budget"] = dict(calls[0]["budget"])
                elif mutation == "out_of_order":
                    calls.reverse()
                else:
                    calls[0]["budget"]["cpus"] += 1
                report_path.write_text(json.dumps(report))

                with self.assertRaisesRegex(
                    VPFPMacroContractAuditError,
                    message,
                ):
                    validate_vpfp_formal_macro_contract(formal)

    def test_rejects_budget_schedule_changes_between_formal_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            formal = self._formal_tree(Path(temporary))
            report_path = self._stage_report(
                formal, "champion_pair", "frontier"
            )
            report = json.loads(report_path.read_text())
            changed = dict(report["budgets"][1])
            changed["wall_time_seconds"] += 5.0
            report["budgets"][1] = changed
            for root in report["root_case_rows"]:
                for variant in root["variants"].values():
                    for solver in variant["solvers"].values():
                        solver["budgets"][1]["budget"] = dict(changed)
            report_path.write_text(json.dumps(report))

            with self.assertRaisesRegex(
                VPFPMacroContractAuditError,
                "frontier budget schedule changed between formal pairs",
            ):
                validate_vpfp_formal_macro_contract(formal)

    def test_rejects_duplicate_stage_budget_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            formal = self._formal_tree(Path(temporary))
            report_path = self._stage_report(
                formal, "agent_pair", "frontier"
            )
            report = json.loads(report_path.read_text())
            duplicate = dict(report["budgets"][0])
            report["budgets"][1] = duplicate
            for root in report["root_case_rows"]:
                for variant in root["variants"].values():
                    for solver in variant["solvers"].values():
                        solver["budgets"][1]["budget"] = dict(duplicate)
            report_path.write_text(json.dumps(report))

            with self.assertRaisesRegex(
                VPFPMacroContractAuditError,
                "duplicate stage budget identity",
            ):
                validate_vpfp_formal_macro_contract(formal)

    def test_rejects_incoherent_macro_metric_payloads(self) -> None:
        mutations = (
            ("alias", "macro primary aliases disagree"),
            ("legacy_formula", "legacy four-scalar metric disagrees"),
            ("physical_formula", "macro primary disagrees with physical-field mean"),
        )
        for mutation, message in mutations:
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as temporary,
            ):
                formal = self._formal_tree(Path(temporary))
                report_path = self._stage_report(formal, "agent_pair", "broad")
                report = json.loads(report_path.read_text())
                bridge = report["root_case_rows"][0]["variants"]["base"][
                    "solvers"
                ]["candidate"]["budgets"][0][BRIDGE_RESULT_KEY]
                metrics = bridge["metrics"]
                if mutation == "alias":
                    metrics["macro_match_relative_l2"] = 0.5
                elif mutation == "legacy_formula":
                    metrics["legacy_four_scalar_macro_relative_l2"] = 0.5
                else:
                    metrics["momentum_relative_l2"] = 0.5
                report_path.write_text(json.dumps(report))

                with self.assertRaisesRegex(
                    VPFPMacroContractAuditError,
                    message,
                ):
                    validate_vpfp_formal_macro_contract(formal)

    def test_writer_failure_removes_stale_success_and_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            formal = self._formal_tree(Path(temporary))
            output = formal / "macro_contract_validation.json"
            initial = write_vpfp_formal_macro_contract_audit(formal, output)
            self.assertTrue(initial["passed"])
            self.assertTrue(output.is_file())

            report_path = self._stage_report(formal, "agent_pair", "broad")
            report = json.loads(report_path.read_text())
            call = report["root_case_rows"][0]["variants"]["base"][
                "solvers"
            ]["candidate"]["budgets"][0]
            bridge = call.pop(BRIDGE_RESULT_KEY)
            report_path.write_text(json.dumps(report))
            failure_output = output.with_name(
                "macro_contract_validation.failed.json"
            )

            with self.assertRaisesRegex(
                VPFPMacroContractAuditError,
                "missing unified_macro_match",
            ):
                write_vpfp_formal_macro_contract_audit(formal, output)

            self.assertFalse(output.exists())
            self.assertTrue(failure_output.is_file())
            failure_report = json.loads(failure_output.read_text())
            self.assertFalse(failure_report["passed"])
            self.assertIn(
                "missing unified_macro_match",
                failure_report["failure"],
            )
            self.assertFalse(
                failure_report["used_for_winner_selection"]
            )

            call[BRIDGE_RESULT_KEY] = bridge
            report_path.write_text(json.dumps(report))
            recovered = write_vpfp_formal_macro_contract_audit(formal, output)
            self.assertTrue(recovered["passed"])
            self.assertTrue(output.is_file())
            self.assertEqual(json.loads(output.read_text()), recovered)

    def _formal_tree(self, root: Path) -> Path:
        formal = root / "formal"
        private = formal / "private"
        private.mkdir(parents=True)
        summary = {
            "schema": "vpfp_long_time_robust_formal_summary_v2",
            "passed": True,
            "broad_root_case_count": 1,
            "frontier_root_case_count": 1,
        }
        summary_path = formal / "formal_summary.json"
        summary_path.write_text(json.dumps(summary))

        controller = {
            "schema": "vpfp_long_time_robust_formal_private_controller_v2",
            "public_summary": str(summary_path),
        }
        for pair_name in ("agent_pair", "champion_pair"):
            controller[pair_name] = {}
            for stage_name in ("broad", "frontier"):
                report_path = self._stage_report(formal, pair_name, stage_name)
                report_path.parent.mkdir(parents=True)
                report_path.write_text(
                    json.dumps(self._private_stage(pair_name, stage_name))
                )
                controller[pair_name][f"{stage_name}_report"] = str(
                    report_path
                )
        (private / "formal_controller_report.json").write_text(
            json.dumps(controller)
        )
        return formal

    def _stage_report(
        self,
        formal: Path,
        pair_name: str,
        stage_name: str,
    ) -> Path:
        return formal / "private" / pair_name / stage_name / "report.json"

    def _private_stage(self, _pair_name: str, stage_name: str) -> dict:
        budgets = (
            [self._budget(1, 120.0)]
            if stage_name == "broad"
            else [self._budget(0, 30.0), self._budget(1, 120.0)]
        )
        variants = {}
        for variant_name in ("base", "perturbed"):
            solvers = {}
            for solver_name in ("candidate", "comparison"):
                case_id = f"{stage_name}-{variant_name}"
                solvers[solver_name] = {
                    "budgets": [
                        self._accepted_call(case_id, budget)
                        for budget in budgets
                    ]
                }
            variants[variant_name] = {"solvers": solvers}
        return {
            "schema": "vpfp_private_paired_development_stage_v2",
            "stage_label": f"formal_{stage_name}",
            "sample_count": 1,
            "budgets": budgets,
            "root_case_rows": [{"variants": variants}],
        }

    def _budget(self, tier: int, wall_time_seconds: float) -> dict:
        return {
            "schema": "vpfp_solver_compute_budget_v1",
            "tier": tier,
            "wall_time_seconds": wall_time_seconds,
            "cpus": 4,
            "gpus": 1,
            "memory_bytes": 64 * 1024**3,
        }

    def _accepted_call(self, case_id: str, budget: dict) -> dict:
        primary = 0.125
        return {
            "budget": dict(budget),
            "accepted": True,
            BRIDGE_RESULT_KEY: {
                "schema": LEGACY_MACRO_BRIDGE_SCHEMA,
                "version": 1,
                "case_id": case_id,
                "contract": VPFPMacroMatcher.contract_id,
                "axis_order": ["t", "x", "y"],
                "shape": [3, 2, 4],
                "coordinate_digest": "sha256:" + "a" * 64,
                "primary_error": primary,
                "metrics": {
                    "macro_match_relative_l2": primary,
                    "macro_relative_l2": primary,
                    "vpfp_macro_match_v1_relative_l2": primary,
                    "legacy_four_scalar_macro_relative_l2": 0.13125,
                    "rho_relative_l2": 0.1,
                    "momentum_relative_l2": 0.15,
                    "momentum_x_relative_l2": 0.12,
                    "momentum_y_relative_l2": 0.18,
                    "kinetic_energy_relative_l2": 0.125,
                },
                "passed": None,
                "failures": [],
                "used_for_winner_selection": False,
            },
        }


if __name__ == "__main__":
    unittest.main()
