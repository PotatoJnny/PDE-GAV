import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pde_agent_verifier_solver.plugins.vpfp.macro_contract_audit import (
    AUDIT_SCHEMA,
)
from pde_agent_verifier_solver.plugins.vpfp.macro_matching import (
    VPFPMacroMatcher,
)
from pde_agent_verifier_solver.plugins.vpfp.publish import main


VALIDATOR_TARGET = (
    "pde_agent_verifier_solver.plugins.vpfp.publish."
    "validate_vpfp_formal_macro_contract"
)


def _passing_audit(formal_root: Path) -> dict[str, object]:
    return {
        "schema": AUDIT_SCHEMA,
        "passed": True,
        "formal_root": str(formal_root.resolve()),
        "contract_id": VPFPMacroMatcher.contract_id,
        "audited_macro_match_count": 16,
        "macro_primary_error_mean": 0.75,
        "macro_primary_error_maximum": 1.25,
        "used_for_winner_selection": False,
        "legacy_ranking_replaced": False,
    }


class VPFPPublishTests(unittest.TestCase):
    def test_publish_copies_the_complete_winning_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "run"
            campaign = root / "campaign"
            formal = root / "formal"
            bundle = root / "winning_bundle"
            bundle.mkdir()
            (bundle / "submission.py").write_text("VALUE = 1\n")
            (bundle / "weights.npz").write_bytes(b"weights")
            agents = []
            for agent_id in ("vpfp_max_a", "vpfp_max_b"):
                private = campaign / "controller_private" / agent_id
                report_root = private / "agent_run"
                report_root.mkdir(parents=True)
                report = {
                    "final_bundle": str(
                        bundle / "submission.py"
                        if agent_id == "vpfp_max_a"
                        else root / "unused" / "submission.py"
                    )
                }
                (report_root / "worker_report.json").write_text(json.dumps(report))
                agents.append(
                    {
                        "agent_id": agent_id,
                        "controller_private_root": str(private),
                        "workspace": str(campaign / "workspaces" / agent_id),
                    }
                )
            (campaign / "campaign_preparation.json").write_text(
                json.dumps({"agents": agents})
            )
            formal.mkdir()
            (formal / "formal_summary.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "agent_winner": "vpfp_max_a",
                        "official_agent_promotion_supported": True,
                        "agent_pair": {
                            "broad": {
                                "solvers": {
                                    "vpfp_max_a": {"high_budget_mean": 0.125}
                                }
                            }
                        },
                        "winner_vs_incumbent": {},
                    }
                )
            )
            audit = _passing_audit(formal)
            audit_path = formal / "macro_contract_validation.json"
            audit_path.write_text(json.dumps(audit))

            with patch(VALIDATOR_TARGET, return_value=audit) as validator:
                main(
                    [
                        "--run-root",
                        str(run_root),
                        "--campaign-root",
                        str(campaign),
                        "--formal-root",
                        str(formal),
                        "--macro-contract-audit",
                        str(audit_path),
                    ]
                )
            validator.assert_called_once_with(formal.resolve())

            result = json.loads(
                (run_root / "result" / "result.json").read_text()
            )
            self.assertEqual(result["winner"], "vpfp_max_a")
            self.assertEqual(result["primary_error"], 0.125)
            self.assertEqual(
                result["primary_metric"],
                "legacy_vpfp_formal_answer_first_v2",
            )
            self.assertEqual(
                result["macro_match"]["audit_primary_error_mean"],
                0.75,
            )
            self.assertEqual(
                result["macro_match"]["evidence_scope"],
                "all_accepted_formal_calls",
            )
            self.assertNotIn("primary_error_mean", result["macro_match"])
            self.assertFalse(
                result["macro_match"]["used_for_winner_selection"]
            )
            self.assertFalse(result["macro_match"]["legacy_ranking_replaced"])
            self.assertNotEqual(
                result["primary_error"],
                result["macro_match"]["audit_primary_error_mean"],
            )
            self.assertEqual(
                result["task"]["macro_match_contract"],
                VPFPMacroMatcher.contract_id,
            )
            self.assertTrue(
                (run_root / "result" / "model" / "weights.npz").is_file()
            )

    def test_publish_rejects_missing_failed_or_stale_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for scenario in ("missing", "failed", "stale"):
                with self.subTest(scenario=scenario):
                    case_root = root / scenario
                    run_root = case_root / "run"
                    campaign = case_root / "campaign"
                    formal = case_root / "formal"
                    campaign.mkdir(parents=True)
                    formal.mkdir(parents=True)
                    (campaign / "campaign_preparation.json").write_text(
                        json.dumps({"agents": []})
                    )
                    (formal / "formal_summary.json").write_text(
                        json.dumps({"passed": True})
                    )
                    audit = _passing_audit(formal)
                    audit_path = formal / "macro_contract_validation.json"
                    live_audit = audit
                    if scenario == "failed":
                        failed_audit = {**audit, "passed": False}
                        audit_path.write_text(json.dumps(failed_audit))
                    elif scenario == "stale":
                        audit_path.write_text(json.dumps(audit))
                        live_audit = {
                            **audit,
                            "audited_macro_match_count": 17,
                        }

                    args = [
                        "--run-root",
                        str(run_root),
                        "--campaign-root",
                        str(campaign),
                        "--formal-root",
                        str(formal),
                        "--macro-contract-audit",
                        str(audit_path),
                    ]
                    expected_exception = (
                        FileNotFoundError
                        if scenario == "missing"
                        else ValueError
                    )
                    with patch(VALIDATOR_TARGET, return_value=live_audit) as validator:
                        with self.assertRaises(expected_exception):
                            main(args)
                    if scenario == "stale":
                        validator.assert_called_once_with(formal.resolve())
                    else:
                        validator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
