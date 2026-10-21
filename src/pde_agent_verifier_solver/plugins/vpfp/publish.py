"""Turn a completed VPFP comparison into the model shown by the product."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from .macro_contract_audit import (
    AUDIT_SCHEMA,
    validate_vpfp_formal_macro_contract,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--macro-contract-audit", default="")
    args = parser.parse_args(argv)

    run_root = Path(args.run_root).resolve()
    campaign_root = Path(args.campaign_root).resolve()
    formal_root = Path(args.formal_root).resolve()
    macro_audit_path = (
        Path(args.macro_contract_audit).resolve()
        if args.macro_contract_audit
        else formal_root / "macro_contract_validation.json"
    )
    preparation = _read(campaign_root / "campaign_preparation.json")
    formal = _read(formal_root / "formal_summary.json")
    if formal.get("passed") is not True:
        raise ValueError("cannot publish a VPFP evaluation that did not pass")
    macro_audit = _validated_macro_audit(macro_audit_path, formal_root)
    winner = str(formal.get("agent_winner", ""))
    agent_rows = {
        str(row["agent_id"]): row
        for row in preparation.get("agents", [])
        if isinstance(row, Mapping) and row.get("agent_id")
    }
    if winner not in agent_rows:
        raise ValueError("formal VPFP winner is absent from campaign preparation")

    worker_path = (
        Path(str(agent_rows[winner]["controller_private_root"])).resolve()
        / "agent_run"
        / "worker_report.json"
    )
    worker = _read(worker_path)
    source_entrypoint = Path(str(worker.get("final_bundle", ""))).resolve()
    if source_entrypoint.name != "submission.py" or not source_entrypoint.is_file():
        raise FileNotFoundError(
            f"winning VPFP model entrypoint is missing: {source_entrypoint}"
        )

    # A solver may include learned weight files and helper modules beside
    # submission.py.  Copy the complete frozen bundle so the delivered model
    # remains runnable instead of silently dropping those files.
    model_root = run_root / "result" / "model"
    if model_root.exists():
        raise FileExistsError(model_root)
    model_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_entrypoint.parent, model_root)
    final_entrypoint = model_root / "submission.py"
    if not final_entrypoint.is_file():
        raise RuntimeError("copied VPFP model has no submission.py")

    worker_reports = {
        agent_id: str(
            Path(str(row["controller_private_root"])).resolve()
            / "agent_run"
            / "worker_report.json"
        )
        for agent_id, row in agent_rows.items()
    }
    result = {
        "kind": "observable_model",
        "problem": "Vlasov--Poisson--Fokker--Planck",
        "name": "Selected VPFP macroscopic model",
        "model": str(final_entrypoint),
        "usage": (
            "from submission import solve_with_budget; "
            "solve_with_budget(public_case, query_times, budget)"
        ),
        "winner": winner,
        "primary_error": _primary_error(formal),
        "primary_metric": "legacy_vpfp_formal_answer_first_v2",
        "official_promotion_supported": bool(
            formal.get("official_agent_promotion_supported")
        ),
        "task": {
            "outputs": ["rho", "momentum", "kinetic_energy"],
            "macro_match_contract": macro_audit["contract_id"],
            "unified_fields": [
                "rho",
                "momentum_x",
                "momentum_y",
                "kinetic_energy",
            ],
            "internal_method_is_unconstrained": True,
        },
        "reports": {
            "formal_evaluation": str(formal_root / "formal_summary.json"),
            "agent_workers": worker_reports,
            "macro_contract_audit": str(macro_audit_path),
        },
        "macro_match": {
            "contract_id": macro_audit["contract_id"],
            "primary_metric": "vpfp_macro_match_v1_relative_l2",
            "evidence_scope": "all_accepted_formal_calls",
            "audited_match_count": macro_audit["audited_macro_match_count"],
            "audit_primary_error_mean": macro_audit["macro_primary_error_mean"],
            "audit_primary_error_maximum": (
                macro_audit["macro_primary_error_maximum"]
            ),
            "used_for_winner_selection": False,
            "legacy_ranking_replaced": False,
        },
        "memory": {
            "agent_a_workspace": str(
                agent_rows.get("vpfp_max_a", {}).get("workspace", "")
            ),
            "agent_b_workspace": str(
                agent_rows.get("vpfp_max_b", {}).get("workspace", "")
            ),
            "note": (
                "Each workspace contains the Agent trajectory, feedback, "
                "and inherited solver context for this VPFP round."
            ),
        },
        "evaluation": {
            "agent_pair": formal.get("agent_pair", {}),
            "winner_vs_incumbent": formal.get("winner_vs_incumbent", {}),
        },
    }
    output = run_root / "result" / "result.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def _validated_macro_audit(
    path: Path,
    formal_root: Path,
) -> Mapping[str, Any]:
    audit = _read(path)
    if (
        audit.get("schema") != AUDIT_SCHEMA
        or audit.get("passed") is not True
        or audit.get("formal_root") != str(formal_root)
        or audit.get("used_for_winner_selection") is not False
        or audit.get("legacy_ranking_replaced") is not False
    ):
        raise ValueError("VPFP macro-contract audit is absent, failed, or stale")
    live = validate_vpfp_formal_macro_contract(formal_root)
    if audit != live:
        raise ValueError("VPFP macro-contract audit differs from formal evidence")
    return audit


def _primary_error(formal: Mapping[str, Any]) -> float | None:
    """Return the winner's broad-cohort mean answer error for the panel."""

    winner = str(formal.get("agent_winner", ""))
    agent_pair = formal.get("agent_pair")
    if not isinstance(agent_pair, Mapping):
        return None
    broad = agent_pair.get("broad")
    if not isinstance(broad, Mapping):
        return None
    solvers = broad.get("solvers")
    if not isinstance(solvers, Mapping):
        return None
    winner_metrics = solvers.get(winner)
    if not isinstance(winner_metrics, Mapping):
        return None
    value = winner_metrics.get("high_budget_mean")
    return None if value is None else float(value)


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


if __name__ == "__main__":
    main()
