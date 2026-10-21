from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

from agent_verifier.vpfp.compute_budget import MACRO_FIELDS


CHAMPION_CHALLENGER_SCORE_SCHEMA = "vpfp_champion_challenger_score_v1"
CHAMPION_CHALLENGER_LEADERBOARD_SCHEMA = (
    "vpfp_champion_challenger_leaderboard_v1"
)


def summarize_solver_frontier(
    rows: Iterable[Mapping[str, Any]],
    *,
    label: str,
    budget_seconds: Iterable[float],
    root_ids: Iterable[str],
) -> dict[str, Any]:
    """Summarize one solver without imposing a monotonicity requirement."""

    budgets = [float(value) for value in budget_seconds]
    roots = [str(value) for value in root_ids]
    if len(budgets) < 2 or any(value <= 0.0 for value in budgets):
        raise ValueError("a positive multi-level compute schedule is required")
    if any(fine <= coarse for coarse, fine in zip(budgets, budgets[1:])):
        raise ValueError("compute budgets must be strictly increasing")
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("root IDs must be nonempty and unique")

    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        key = (str(row.get("root_case_id") or ""), int(row.get("tier", -1)))
        if key in indexed:
            duplicates.append({"root_case_id": key[0], "tier": key[1]})
        indexed[key] = row

    tier_summaries = []
    all_valid = not duplicates
    for tier, seconds in enumerate(budgets):
        selected = [indexed.get((root, tier)) for root in roots]
        valid = [
            row
            for row in selected
            if row is not None
            and row.get("status") == "ok"
            and _finite_positive(row.get("aggregate_rel_l2"))
            and _finite_nonnegative(row.get("wall_time_seconds"))
        ]
        complete = len(valid) == len(roots)
        within_budget = bool(
            complete
            and all(
                float(row["wall_time_seconds"]) <= seconds * (1.0 + 1.0e-6)
                for row in valid
            )
        )
        all_valid = bool(all_valid and complete and within_budget)
        errors = [float(row["aggregate_rel_l2"]) for row in valid]
        tier_summaries.append(
            {
                "tier": int(tier),
                "budget_seconds": float(seconds),
                "valid_root_count": len(valid),
                "complete": bool(complete),
                "within_budget": bool(within_budget),
                "aggregate_error_mean": (
                    float(np.mean(errors)) if complete else None
                ),
                "aggregate_error_median": (
                    float(np.median(errors)) if complete else None
                ),
                "aggregate_error_q90": (
                    float(np.quantile(errors, 0.9)) if complete else None
                ),
                "mean_actual_wall_time_seconds": (
                    float(
                        np.mean(
                            [float(row["wall_time_seconds"]) for row in valid]
                        )
                    )
                    if complete
                    else None
                ),
                "mean_field_errors": {
                    field: (
                        float(
                            np.mean(
                                [float(row[f"{field}_rel_l2"]) for row in valid]
                            )
                        )
                        if complete
                        and all(
                            _finite_nonnegative(row.get(f"{field}_rel_l2"))
                            for row in valid
                        )
                        else None
                    )
                    for field in MACRO_FIELDS
                },
            }
        )

    errors = [row["aggregate_error_mean"] for row in tier_summaries]
    if all_valid:
        error_values = np.asarray(errors, dtype=np.float64)
        log_budget = np.log(np.asarray(budgets, dtype=np.float64))
        normalized_width = float(log_budget[-1] - log_budget[0])
        log_auc = float(
            np.trapz(np.log(error_values), log_budget) / normalized_width
        )
        frontier_geometric_mean = float(np.exp(log_auc))
        envelope = np.minimum.accumulate(error_values)
        envelope_log_auc = float(
            np.trapz(np.log(envelope), log_budget) / normalized_width
        )
        primary = float(error_values[-1])
        endpoint_gain = float(
            (error_values[0] - error_values[-1])
            / max(float(error_values[0]), 1.0e-30)
        )
    else:
        log_auc = None
        frontier_geometric_mean = None
        envelope_log_auc = None
        primary = None
        endpoint_gain = None

    return {
        "schema": CHAMPION_CHALLENGER_SCORE_SCHEMA,
        "label": str(label),
        "safety_passed": bool(all_valid),
        "primary_high_budget_mean_error": primary,
        "secondary_frontier_geometric_mean_error": frontier_geometric_mean,
        "secondary_log_error_auc": log_auc,
        "best_so_far_envelope_log_error_auc_diagnostic": envelope_log_auc,
        "endpoint_relative_gain_diagnostic": endpoint_gain,
        "pairwise_budget_monotonicity_required": False,
        "runtime_utilization_is_an_objective": False,
        "tier_summaries": tier_summaries,
        "duplicate_rows": duplicates,
    }


def paired_solver_comparison(
    candidate_rows: Iterable[Mapping[str, Any]],
    incumbent_rows: Iterable[Mapping[str, Any]],
    *,
    candidate_label: str,
    incumbent_label: str,
    budget_seconds: Iterable[float],
    root_ids: Iterable[str],
) -> dict[str, Any]:
    budgets = [float(value) for value in budget_seconds]
    roots = [str(value) for value in root_ids]
    candidate = _index_rows(candidate_rows)
    incumbent = _index_rows(incumbent_rows)
    tiers = []
    for tier, seconds in enumerate(budgets):
        paired = []
        for root in roots:
            left = candidate.get((root, tier))
            right = incumbent.get((root, tier))
            if not _valid_error_row(left) or not _valid_error_row(right):
                continue
            candidate_error = float(left["aggregate_rel_l2"])
            incumbent_error = float(right["aggregate_rel_l2"])
            paired.append(
                {
                    "root_case_id": root,
                    "candidate_error": candidate_error,
                    "incumbent_error": incumbent_error,
                    "candidate_to_incumbent_ratio": float(
                        candidate_error / max(incumbent_error, 1.0e-30)
                    ),
                    "candidate_minus_incumbent": float(
                        candidate_error - incumbent_error
                    ),
                }
            )
        complete = len(paired) == len(roots)
        tiers.append(
            {
                "tier": int(tier),
                "budget_seconds": float(seconds),
                "complete": bool(complete),
                "paired_root_count": len(paired),
                "mean_candidate_to_incumbent_ratio": (
                    float(
                        np.mean(
                            [row["candidate_to_incumbent_ratio"] for row in paired]
                        )
                    )
                    if complete
                    else None
                ),
                "mean_candidate_minus_incumbent": (
                    float(
                        np.mean(
                            [row["candidate_minus_incumbent"] for row in paired]
                        )
                    )
                    if complete
                    else None
                ),
                "roots": paired,
            }
        )
    return {
        "candidate": str(candidate_label),
        "incumbent": str(incumbent_label),
        "tiers": tiers,
        "high_budget_mean_ratio": tiers[-1][
            "mean_candidate_to_incumbent_ratio"
        ],
        "high_budget_mean_difference": tiers[-1][
            "mean_candidate_minus_incumbent"
        ],
    }


def rank_solver_summaries(
    summaries: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in summaries]

    def key(row: Mapping[str, Any]) -> tuple[float, float, str]:
        if not bool(row.get("safety_passed")):
            return (float("inf"), float("inf"), str(row.get("label") or ""))
        primary = float(row["primary_high_budget_mean_error"])
        secondary = float(row["secondary_frontier_geometric_mean_error"])
        return (primary, secondary, str(row.get("label") or ""))

    ranked = sorted(rows, key=key)
    return [
        {
            **row,
            "rank": int(index + 1),
            "selected": bool(index == 0 and np.isfinite(key(row)[0])),
        }
        for index, row in enumerate(ranked)
    ]


def build_leaderboard(
    solver_rows: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    incumbent_label: str,
    budget_seconds: Iterable[float],
    root_ids: Iterable[str],
) -> dict[str, Any]:
    budgets = [float(value) for value in budget_seconds]
    roots = [str(value) for value in root_ids]
    materialized_rows = {
        str(label): [dict(row) for row in rows]
        for label, rows in solver_rows.items()
    }
    if incumbent_label not in materialized_rows:
        raise ValueError("incumbent label is absent from solver rows")
    summaries = [
        summarize_solver_frontier(
            rows,
            label=label,
            budget_seconds=budgets,
            root_ids=roots,
        )
        for label, rows in materialized_rows.items()
    ]
    ranked = rank_solver_summaries(summaries)
    selected = next((row for row in ranked if row["selected"]), None)
    comparisons = {
        label: paired_solver_comparison(
            rows,
            materialized_rows[incumbent_label],
            candidate_label=label,
            incumbent_label=incumbent_label,
            budget_seconds=budgets,
            root_ids=roots,
        )
        for label, rows in materialized_rows.items()
        if label != incumbent_label
    }
    return {
        "schema": CHAMPION_CHALLENGER_LEADERBOARD_SCHEMA,
        "primary_objective": "minimum high-budget mean normalized macro error",
        "secondary_objective": "minimum log-budget frontier geometric mean error",
        "safety_gate": "finite valid outputs within each wall-time budget",
        "incumbent_label": str(incumbent_label),
        "selected_label": None if selected is None else str(selected["label"]),
        "candidate_pass_threshold_used": False,
        "budget_seconds": budgets,
        "root_ids": roots,
        "ranking": ranked,
        "paired_against_incumbent": comparisons,
    }


def _index_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    result = defaultdict(dict)
    for raw in rows:
        row = dict(raw)
        result[(str(row.get("root_case_id") or ""), int(row.get("tier", -1)))] = row
    return dict(result)


def _valid_error_row(row: Mapping[str, Any] | None) -> bool:
    return bool(
        row is not None
        and row.get("status") == "ok"
        and _finite_positive(row.get("aggregate_rel_l2"))
    )


def _finite_positive(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)) and float(value) > 0.0)
    except (TypeError, ValueError):
        return False


def _finite_nonnegative(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)) and float(value) >= 0.0)
    except (TypeError, ValueError):
        return False
