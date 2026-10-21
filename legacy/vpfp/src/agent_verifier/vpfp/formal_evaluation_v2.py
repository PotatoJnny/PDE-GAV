"""Aggregate private VPFP answer-first stages into formal pair decisions."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .development_evaluation import paired_mean_interval, tail_summary
from .development_runtime_v2 import (
    VPFP_PRIVATE_STAGE_SCHEMA,
    _summarize_solver,
)


VPFP_FORMAL_PAIR_SUMMARY_SCHEMA = "vpfp_formal_pair_summary_v2"
VPFP_FORMAL_PAIR_DECISION_SCHEMA = "vpfp_formal_pair_decision_v2"
VPFP_AGENT_WORKER_SCHEMA = "vpfp_long_time_robust_agent_worker_v2"
RECOVERABLE_FINALIST_SERVICE_FAILURE = (
    "VPFP finalist compatibility response failed because service exited with "
    "status 1"
)


def classify_vpfp_formal_worker(
    worker: Mapping[str, Any],
    *,
    expected_agent_id: str,
    allow_evaluator_failed_recovery: bool = False,
) -> dict[str, Any]:
    """Validate a complete worker or one narrowly recoverable frozen bundle."""

    if (
        worker.get("schema") != VPFP_AGENT_WORKER_SCHEMA
        or worker.get("agent_id") != expected_agent_id
        or worker.get("user_authorized_external_dispatch") is not True
        or not worker.get("final_bundle")
        or not isinstance(worker.get("final_bundle_sha256"), str)
        or len(str(worker["final_bundle_sha256"])) != 64
    ):
        raise ValueError("VPFP Agent worker identity or final bundle is invalid")

    if worker.get("passed") is True:
        return {
            "accepted": True,
            "development_feedback_protocol_complete": True,
            "recovered_after_evaluator_failure": False,
            "status": "complete",
        }
    if not allow_evaluator_failed_recovery:
        raise ValueError("VPFP Agent worker is incomplete")

    invocation = dict(worker.get("invocation") or {})
    failure = dict(worker.get("failure") or {})
    services = dict(worker.get("service_shutdown") or {})
    evaluator = dict(services.get("evaluator") or {})
    online = dict(services.get("online") or {})
    if not (
        worker.get("actual_external_agent_invoked") is True
        and invocation.get("actual_codex_invoked") is True
        and invocation.get("returncode") == 0
        and invocation.get("timed_out") is False
        and worker.get("finalist_response") is None
        and failure.get("type") == "RuntimeError"
        and failure.get("message") == RECOVERABLE_FINALIST_SERVICE_FAILURE
        and evaluator.get("started") is True
        and evaluator.get("returncode") == 1
        and evaluator.get("forced") is False
        and online.get("started") is True
        and online.get("returncode") == 0
        and online.get("forced") is False
    ):
        raise ValueError("VPFP Agent worker failure is not recoverable")

    return {
        "accepted": True,
        "development_feedback_protocol_complete": False,
        "recovered_after_evaluator_failure": True,
        "status": "recovered_frozen_bundle_after_known_evaluator_failure",
    }


def canonical_public_cohort_sha256(
    stage_root: str | Path,
    *,
    expected_root_case_count: int,
) -> str:
    """Commit the exported public equations independent of NPZ metadata."""

    root = Path(stage_root).resolve()
    expected_case_count = 2 * int(expected_root_case_count)
    if expected_case_count < 2 or not root.is_dir() or root.is_symlink():
        raise ValueError("VPFP formal public cohort root is invalid")

    complete_attempts: list[tuple[Path, list[Path]]] = []
    for attempt in sorted(root.glob("attempt_*")):
        if not attempt.is_dir() or attempt.is_symlink():
            continue
        case_dirs = sorted((attempt / "public_cases").glob("case_*"))
        if len(case_dirs) != expected_case_count:
            continue
        if all(
            case_dir.is_dir()
            and not case_dir.is_symlink()
            and (case_dir / "public_question.json").is_file()
            and not (case_dir / "public_question.json").is_symlink()
            and (case_dir / "public_arrays.npz").is_file()
            and not (case_dir / "public_arrays.npz").is_symlink()
            for case_dir in case_dirs
        ):
            complete_attempts.append((attempt, case_dirs))
    if len(complete_attempts) != 1:
        raise RuntimeError(
            "VPFP formal stage must contain exactly one complete public cohort"
        )

    digest = hashlib.sha256()
    for case_dir in complete_attempts[0][1]:
        _update_digest_record(digest, case_dir.name.encode("utf-8"))
        question = json.loads(
            (case_dir / "public_question.json").read_text(encoding="utf-8")
        )
        if not isinstance(question, dict):
            raise TypeError("VPFP public question must be a JSON object")
        _update_digest_record(
            digest,
            json.dumps(
                question,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        )
        with np.load(
            case_dir / "public_arrays.npz", allow_pickle=False
        ) as arrays:
            for name in sorted(arrays.files):
                value = np.asarray(arrays[name])
                if value.dtype.hasobject:
                    raise TypeError("VPFP public arrays cannot contain objects")
                value = np.ascontiguousarray(value)
                metadata = {
                    "name": name,
                    "dtype": value.dtype.str,
                    "shape": list(value.shape),
                }
                _update_digest_record(
                    digest,
                    json.dumps(
                        metadata,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8"),
                )
                _update_digest_record(digest, value.tobytes(order="C"))
    return digest.hexdigest()


def _update_digest_record(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    digest.update(payload)


def summarize_vpfp_formal_pair_stage(
    report: Mapping[str, Any],
    *,
    candidate_label: str,
    comparison_label: str,
) -> dict[str, Any]:
    """Return aggregate-only evidence for two solvers on one private stage."""

    row = dict(report)
    labels = (str(candidate_label), str(comparison_label))
    if (
        row.get("schema") != VPFP_PRIVATE_STAGE_SCHEMA
        or not labels[0]
        or not labels[1]
        or labels[0] == labels[1]
        or int(row.get("sample_count", 0)) <= 0
    ):
        raise ValueError("VPFP formal pair stage is invalid")
    solver_ids = ("candidate", "comparison")
    for root in row.get("root_case_rows") or []:
        variants = root.get("variants")
        if not isinstance(variants, Mapping) or set(variants) != {
            "base",
            "perturbed",
        }:
            raise ValueError("VPFP formal stage variants are incomplete")
        for variant in variants.values():
            solvers = variant.get("solvers")
            if not isinstance(solvers, Mapping) or not set(solver_ids).issubset(
                solvers
            ):
                raise ValueError("VPFP formal stage solver evidence is incomplete")

    summaries: dict[str, dict[str, Any]] = {}
    high_values: dict[str, list[float]] = {}
    for label, solver_id in zip(labels, solver_ids):
        summary = _summarize_solver([row], solver_id)
        values = [float(value) for value in summary.pop("_high_budget_values")]
        robust = dict(summary.pop("_robust"))
        high = tail_summary(values)
        summary.update(
            {
                "robust": robust,
                "high_budget_p90": float(high["p90"]),
                "high_budget_maximum": float(high["maximum"]),
                "sampled_variant_count": len(values),
            }
        )
        summaries[label] = summary
        high_values[label] = values

    paired = paired_mean_interval(high_values[labels[0]], high_values[labels[1]])
    return {
        "schema": VPFP_FORMAL_PAIR_SUMMARY_SCHEMA,
        "candidate_label": labels[0],
        "comparison_label": labels[1],
        "root_case_count": int(row["sample_count"]),
        "variant_count": 2 * int(row["sample_count"]),
        "budgets": list(row["budgets"]),
        "solvers": summaries,
        "paired_high_budget": {
            "mean_candidate_minus_comparison": float(
                paired["mean_candidate_minus_comparison"]
            ),
            "normal_approximation_95_lower": float(
                paired["normal_approximation_95_lower"]
            ),
            "normal_approximation_95_upper": float(
                paired["normal_approximation_95_upper"]
            ),
            "candidate_better_fraction": float(
                paired["candidate_better_fraction"]
            ),
        },
        "answer_first_exact_continuous_truth_primary": True,
        "base_and_structure_preserving_perturbed_required": True,
        "private_case_ids_included": False,
        "per_case_errors_included": False,
    }


def decide_vpfp_formal_pair(
    broad: Mapping[str, Any],
    frontier: Mapping[str, Any],
) -> dict[str, Any]:
    """Rank one solver pair and separately report statistical support."""

    broad_row = _validate_pair_summary(broad)
    frontier_row = _validate_pair_summary(frontier)
    labels = (
        str(broad_row["candidate_label"]),
        str(broad_row["comparison_label"]),
    )
    if (
        labels
        != (
            str(frontier_row["candidate_label"]),
            str(frontier_row["comparison_label"]),
        )
        or set(broad_row["solvers"]) != set(frontier_row["solvers"])
    ):
        raise ValueError("VPFP formal pair labels changed between stages")

    metrics = {
        label: _ranking_metrics(
            broad_row["solvers"][label],
            frontier_row["solvers"][label],
        )
        for label in labels
    }
    eligible = [
        label
        for label in labels
        if metrics[label]["safety_passed"]
    ]
    ranking = sorted(eligible, key=lambda label: (*_ranking_key(metrics[label]), label))
    ranking.extend(label for label in labels if label not in ranking)
    winner = ranking[0] if eligible else None
    exact_metric_tie = bool(
        len(eligible) == 2
        and _accuracy_key(metrics[eligible[0]])
        == _accuracy_key(metrics[eligible[1]])
    )
    paired_support = {
        "broad": _paired_supports_winner(broad_row, winner),
        "frontier": _paired_supports_winner(frontier_row, winner),
    }
    return {
        "schema": VPFP_FORMAL_PAIR_DECISION_SCHEMA,
        "ranking": ranking,
        "winner": winner,
        "all_solvers_ineligible": not eligible,
        "exact_metric_tie": exact_metric_tie,
        "equation_first_tiebreak_required": exact_metric_tie,
        "metrics": metrics,
        "paired_supports_winner": paired_support,
        "clear_paired_improvement": bool(
            winner is not None
            and paired_support["broad"]
            and paired_support["frontier"]
        ),
        "answer_first_error_used_for_ranking": True,
        "equation_first_hidden_reference_error_used_for_ranking": False,
    }


def _validate_pair_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(payload)
    if (
        row.get("schema") != VPFP_FORMAL_PAIR_SUMMARY_SCHEMA
        or row.get("answer_first_exact_continuous_truth_primary") is not True
        or row.get("private_case_ids_included") is not False
        or not isinstance(row.get("solvers"), Mapping)
        or len(row["solvers"]) != 2
    ):
        raise ValueError("VPFP formal pair summary is invalid")
    return row


def _ranking_metrics(
    broad: Mapping[str, Any],
    frontier: Mapping[str, Any],
) -> dict[str, Any]:
    broad_robust = dict(broad["robust"])
    frontier_robust = dict(frontier["robust"])
    safety = bool(
        broad.get("safety_passed") is True
        and frontier.get("safety_passed") is True
        and float(broad.get("valid_prediction_fraction", 0.0)) == 1.0
        and float(frontier.get("valid_prediction_fraction", 0.0)) == 1.0
        and float(broad.get("wall_limit_pass_fraction", 0.0)) == 1.0
        and float(frontier.get("wall_limit_pass_fraction", 0.0)) == 1.0
    )
    metrics = {
        "safety_passed": safety,
        "broad_worst_answer_first_p90": float(
            broad_robust["worst_answer_first_p90"]
        ),
        "broad_worst_time_band_p90": float(
            broad_robust["worst_time_band_p90"]
        ),
        "broad_high_budget_cvar90": float(broad["high_budget_cvar90"]),
        "frontier_worst_answer_first_p90": float(
            frontier_robust["worst_answer_first_p90"]
        ),
        "frontier_high_budget_cvar90": float(frontier["high_budget_cvar90"]),
        "frontier_geometric_mean": float(frontier["frontier_geometric_mean"]),
        "mean_solver_wall_time_seconds": float(
            0.5
            * (
                float(broad["mean_solver_wall_time_seconds"])
                + float(frontier["mean_solver_wall_time_seconds"])
            )
        ),
    }
    numeric = [
        value
        for key, value in metrics.items()
        if key != "safety_passed"
    ]
    if not all(math.isfinite(value) and value >= 0.0 for value in numeric):
        raise ValueError("VPFP formal ranking metrics are nonfinite")
    return metrics


def _accuracy_key(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(metrics["broad_worst_answer_first_p90"]),
        float(metrics["broad_worst_time_band_p90"]),
        float(metrics["broad_high_budget_cvar90"]),
        float(metrics["frontier_worst_answer_first_p90"]),
        float(metrics["frontier_high_budget_cvar90"]),
        float(metrics["frontier_geometric_mean"]),
    )


def _ranking_key(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        *_accuracy_key(metrics),
        float(metrics["mean_solver_wall_time_seconds"]),
    )


def _paired_supports_winner(
    summary: Mapping[str, Any],
    winner: str | None,
) -> bool:
    if winner is None:
        return False
    paired = dict(summary["paired_high_budget"])
    if winner == summary["candidate_label"]:
        return float(paired["normal_approximation_95_upper"]) < 0.0
    if winner == summary["comparison_label"]:
        return float(paired["normal_approximation_95_lower"]) > 0.0
    raise ValueError("VPFP formal winner is absent from the pair")
