#!/usr/bin/env python
"""Run independent answer-first final evaluation for one VPFP Agent campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_verifier.vpfp.development_runtime_v2 import (
    VPFPDevelopmentEvaluatorRuntime,
    VPFPDevelopmentRuntimeConfig,
)
from agent_verifier.vpfp.formal_evaluation_v2 import (
    canonical_public_cohort_sha256,
    classify_vpfp_formal_worker,
    decide_vpfp_formal_pair,
    summarize_vpfp_formal_pair_stage,
)
from agent_verifier.vpfp.oracle_training import inspect_solver_bundle
from agent_verifier.vpfp.runtime_paths import is_persistent_run_path


SUMMARY_SCHEMA = "vpfp_long_time_robust_formal_summary_v2"
PRIVATE_SCHEMA = "vpfp_long_time_robust_formal_private_controller_v2"
PREPARATION_SCHEMA = "vpfp_long_time_robust_two_max_preparation_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--broad-cases", type=int, default=12)
    parser.add_argument("--frontier-cases", type=int, default=4)
    parser.add_argument("--seed-base", type=int)
    parser.add_argument(
        "--recover-evaluator-failed-workers",
        action="store_true",
        help="screen intact bundles from the known pre-repair evaluator failure",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("VPFP formal evaluation requires a Slurm compute node")
    campaign_root = Path(args.campaign_root).resolve()
    output = Path(args.out).resolve()
    if (
        not is_persistent_run_path(campaign_root)
        or not is_persistent_run_path(output)
        or output == campaign_root
        or args.broad_cases < 4
        or args.frontier_cases < 2
    ):
        raise SystemExit("VPFP formal evaluation paths or case counts are invalid")
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "formal_summary.json"
    if summary_path.is_file():
        cached = _read_json(summary_path)
        if cached.get("schema") != SUMMARY_SCHEMA:
            raise SystemExit("existing VPFP formal summary has an unknown schema")
        print(json.dumps(cached, indent=2, sort_keys=True), flush=True)
        if cached.get("passed") is not True:
            raise SystemExit("cached VPFP formal evaluation did not pass")
        return

    preparation = _read_json(campaign_root / "campaign_preparation.json")
    if (
        preparation.get("schema") != PREPARATION_SCHEMA
        or preparation.get("passed") is not True
        or preparation.get("user_authorized_external_dispatch") is not True
        or len(preparation.get("agents") or []) != 2
    ):
        raise SystemExit("VPFP formal evaluation campaign is not qualified")

    agents = {}
    for row in preparation["agents"]:
        agent_id = str(row["agent_id"])
        worker_path = (
            Path(row["controller_private_root"]).resolve()
            / "agent_run"
            / "worker_report.json"
        )
        worker = _read_json(worker_path)
        try:
            worker_status = classify_vpfp_formal_worker(
                worker,
                expected_agent_id=agent_id,
                allow_evaluator_failed_recovery=bool(
                    args.recover_evaluator_failed_workers
                ),
            )
        except ValueError as exc:
            raise SystemExit(
                f"VPFP Agent worker is incomplete: {agent_id}: {exc}"
            ) from exc
        bundle = Path(worker["final_bundle"]).resolve()
        audit = inspect_solver_bundle(bundle)
        if audit["aggregate_sha256"] != worker.get("final_bundle_sha256"):
            raise SystemExit(f"VPFP Agent final bundle changed: {agent_id}")
        agents[agent_id] = {
            "entrypoint": bundle,
            "sha256": audit["aggregate_sha256"],
            "worker_report": worker_path,
            "worker_status": worker_status,
        }
    agent_ids = tuple(sorted(agents))
    if len(agent_ids) != 2:
        raise SystemExit("VPFP formal evaluation needs two unique Agent IDs")

    controller = _read_json(
        Path(preparation["agents"][0]["development_evaluator_config"]).resolve()
    )
    runtime_config = VPFPDevelopmentRuntimeConfig(
        **dict(controller["runtime_config"])
    )
    runtime_config = replace(
        runtime_config,
        maximum_generation_attempt_factor=max(
            16,
            int(runtime_config.maximum_generation_attempt_factor),
        ),
    )
    seed_base = (
        int(args.seed_base)
        if args.seed_base is not None
        else _deterministic_seed(campaign_root, preparation)
    )
    if seed_base < 1:
        raise SystemExit("VPFP formal evaluation seed must be positive")

    agent_pair = _run_pair(
        workspace=Path(controller["workspace"]).resolve(),
        private_root=output / "private" / "agent_pair",
        base_decoder_checkpoint=Path(controller["base_decoder_checkpoint"]).resolve(),
        expanded_decoder_checkpoint=Path(
            controller["expanded_decoder_checkpoint"]
        ).resolve(),
        runtime_config=runtime_config,
        candidate_label=agent_ids[0],
        candidate=agents[agent_ids[0]]["entrypoint"],
        comparison_label=agent_ids[1],
        comparison=agents[agent_ids[1]]["entrypoint"],
        broad_cases=int(args.broad_cases),
        frontier_cases=int(args.frontier_cases),
        seed_base=seed_base,
        checkpoint_index=int(controller["checkpoint_index"]) + 10_000,
    )
    agent_winner = agent_pair["decision"]["winner"]
    if agent_winner is None:
        raise RuntimeError("both VPFP Agent finalists failed formal safety gates")

    champion = Path(preparation["champion_solver"]).resolve()
    champion_audit = inspect_solver_bundle(champion)
    champion_pair = _run_pair(
        workspace=Path(controller["workspace"]).resolve(),
        private_root=output / "private" / "champion_pair",
        base_decoder_checkpoint=Path(controller["base_decoder_checkpoint"]).resolve(),
        expanded_decoder_checkpoint=Path(
            controller["expanded_decoder_checkpoint"]
        ).resolve(),
        runtime_config=runtime_config,
        candidate_label=agent_winner,
        candidate=agents[agent_winner]["entrypoint"],
        comparison_label="incumbent_champion",
        comparison=champion,
        broad_cases=int(args.broad_cases),
        frontier_cases=int(args.frontier_cases),
        seed_base=seed_base,
        checkpoint_index=int(controller["checkpoint_index"]) + 20_000,
    )
    cohort_match = {
        stage: (
            agent_pair["private"][f"{stage}_cohort_commitment"]
            == champion_pair["private"][f"{stage}_cohort_commitment"]
        )
        for stage in ("broad", "frontier")
    }
    champion_decision = champion_pair["decision"]
    auxiliary_required = bool(
        agent_pair["decision"]["equation_first_tiebreak_required"]
        or champion_decision["equation_first_tiebreak_required"]
    )
    worker_status = {
        label: agents[label]["worker_status"] for label in agent_ids
    }
    development_protocol_complete = all(
        row["development_feedback_protocol_complete"]
        for row in worker_status.values()
    )
    official_promoted = bool(
        development_protocol_complete
        and all(cohort_match.values())
        and not auxiliary_required
        and champion_decision["winner"] == agent_winner
        and champion_decision["clear_paired_improvement"]
    )
    passed = bool(
        all(cohort_match.values())
        and not agent_pair["decision"]["all_solvers_ineligible"]
        and not champion_decision["all_solvers_ineligible"]
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "passed": passed,
        "campaign_root": str(campaign_root),
        "regime": preparation["regime"],
        "agent_worker_status": worker_status,
        "development_feedback_protocol_complete": development_protocol_complete,
        "recovery_screen_only": not development_protocol_complete,
        "agent_bundle_sha256": {
            label: agents[label]["sha256"] for label in agent_ids
        },
        "incumbent_bundle_sha256": champion_audit["aggregate_sha256"],
        "broad_root_case_count": int(args.broad_cases),
        "frontier_root_case_count": int(args.frontier_cases),
        "agent_pair": agent_pair["public"],
        "agent_winner": agent_winner,
        "winner_vs_incumbent": champion_pair["public"],
        "same_private_cohort_commitment": cohort_match,
        "official_agent_promotion_supported": official_promoted,
        "equation_first_auxiliary": {
            "required_by_exact_answer_first_tie": auxiliary_required,
            "hidden_reference_error_used_for_selection": False,
            "status": "required_before_decision" if auxiliary_required else "not_needed",
        },
        "answer_first_exact_continuous_truth_primary": True,
        "private_case_ids_included": False,
        "per_case_errors_included": False,
        "generator_parameterization_included": False,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
    }
    private = {
        "schema": PRIVATE_SCHEMA,
        "campaign_root": str(campaign_root),
        "seed_base": seed_base,
        "agent_worker_status": worker_status,
        "development_feedback_protocol_complete": development_protocol_complete,
        "agent_pair": agent_pair["private"],
        "champion_pair": champion_pair["private"],
        "same_private_cohort_commitment": cohort_match,
        "public_summary": str(summary_path),
    }
    _write_json(output / "private" / "formal_controller_report.json", private)
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit("VPFP formal evaluation did not pass infrastructure gates")


def _run_pair(
    *,
    workspace: Path,
    private_root: Path,
    base_decoder_checkpoint: Path,
    expanded_decoder_checkpoint: Path,
    runtime_config: VPFPDevelopmentRuntimeConfig,
    candidate_label: str,
    candidate: Path,
    comparison_label: str,
    comparison: Path,
    broad_cases: int,
    frontier_cases: int,
    seed_base: int,
    checkpoint_index: int,
) -> dict[str, Any]:
    runtime = VPFPDevelopmentEvaluatorRuntime(
        workspace=workspace,
        private_root=private_root,
        base_decoder_checkpoint=base_decoder_checkpoint,
        expanded_decoder_checkpoint=expanded_decoder_checkpoint,
        comparison_solver=comparison,
        checkpoint_index=checkpoint_index,
        config=runtime_config,
        device="cuda",
    )
    candidate_audit = inspect_solver_bundle(candidate)
    formal_budgets = runtime._budgets(formal=True)
    broad = runtime._run_fresh_stage(
        candidate_path=candidate,
        candidate_sha256=candidate_audit["aggregate_sha256"],
        stage_label="formal_broad",
        stage_total_case_count=broad_cases,
        sample_count=broad_cases,
        seed=seed_base,
        budgets=(formal_budgets[1],),
    )
    frontier = runtime._run_fresh_stage(
        candidate_path=candidate,
        candidate_sha256=candidate_audit["aggregate_sha256"],
        stage_label="formal_frontier",
        stage_total_case_count=frontier_cases,
        sample_count=frontier_cases,
        seed=seed_base + 1,
        budgets=formal_budgets,
    )
    broad_summary = summarize_vpfp_formal_pair_stage(
        broad,
        candidate_label=candidate_label,
        comparison_label=comparison_label,
    )
    frontier_summary = summarize_vpfp_formal_pair_stage(
        frontier,
        candidate_label=candidate_label,
        comparison_label=comparison_label,
    )
    decision = decide_vpfp_formal_pair(broad_summary, frontier_summary)
    broad_root = runtime._stage_root(
        candidate_audit["aggregate_sha256"], "formal_broad"
    )
    frontier_root = runtime._stage_root(
        candidate_audit["aggregate_sha256"], "formal_frontier"
    )
    return {
        "public": {
            "broad": broad_summary,
            "frontier": frontier_summary,
            "decision": decision,
        },
        "decision": decision,
        "private": {
            "candidate_label": candidate_label,
            "comparison_label": comparison_label,
            "candidate_sha256": candidate_audit["aggregate_sha256"],
            "comparison_sha256": runtime.comparison_sha256,
            "broad_report": str(broad_root / "report.json"),
            "frontier_report": str(frontier_root / "report.json"),
            "broad_cohort_commitment": _cohort_commitment(
                broad,
                broad_root,
            ),
            "frontier_cohort_commitment": _cohort_commitment(
                frontier,
                frontier_root,
            ),
        },
    }


def _cohort_commitment(
    report: Mapping[str, Any],
    stage_root: Path,
) -> str:
    payload = {
        "sample_count": int(report["sample_count"]),
        "seed": int(report["seed"]),
        "generated_answer_count": int(report["generated_answer_count"]),
        "attempted_generation_count": int(report["attempted_generation_count"]),
        "public_cases_sha256": canonical_public_cohort_sha256(
            stage_root,
            expected_root_case_count=int(report["sample_count"]),
        ),
        "perturbation_metrics": [
            row["perturbation_metrics"] for row in report["root_case_rows"]
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _deterministic_seed(
    campaign_root: Path,
    preparation: Mapping[str, Any],
) -> int:
    payload = {
        "campaign_root": str(campaign_root),
        "schema": preparation["schema"],
        "regime": preparation["regime"],
        "agent_packets": [
            row["packet_sha256"] for row in preparation["agents"]
        ],
        "purpose": "vpfp_formal_answer_first_v2",
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return 1_000_000 + int(digest[:12], 16) % 900_000_000


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
