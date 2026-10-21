"""Compose exact answer-first and auxiliary VPFP robustness evidence."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .equation_first_self_convergence import (
    VPFP_SELF_CONVERGENCE_SUMMARY_SCHEMA,
)


ANSWER_FIRST_MODES = {"base", "perturbed"}


def build_vpfp_robust_selection_metrics(
    cohorts: Sequence[Mapping[str, Any]],
    solver_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Build ranking metrics while excluding equation-first reference errors."""

    rows = [dict(cohort) for cohort in cohorts]
    identifiers = [str(value) for value in solver_ids]
    if not rows or not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("VPFP robust selection needs unique solvers and cohorts")
    answer_first: list[dict[str, Any]] = []
    equation_first: list[dict[str, Any]] = []
    for cohort in rows:
        request = cohort.get("request")
        if not isinstance(request, Mapping):
            raise ValueError("VPFP robust selection request is absent")
        mode = str(request.get("generator_mode") or "")
        if mode in ANSWER_FIRST_MODES:
            answer_first.append(cohort)
        elif mode == "equation_first":
            equation_first.append(cohort)
        else:
            raise ValueError("VPFP robust selection cohort mode is invalid")
    observed_modes = {
        str(row["request"]["generator_mode"]) for row in answer_first
    }
    if observed_modes != ANSWER_FIRST_MODES or len(equation_first) != 1:
        raise ValueError("VPFP robust selection evidence is incomplete")

    equation = equation_first[0]
    metrics: dict[str, dict[str, Any]] = {}
    for solver_id in identifiers:
        try:
            answer_rows = [cohort["solvers"][solver_id] for cohort in answer_first]
            all_rows = [cohort["solvers"][solver_id] for cohort in rows]
            equation_row = equation["solvers"][solver_id]
        except KeyError as exc:
            raise ValueError(f"missing VPFP solver evidence: {solver_id}") from exc
        convergence = equation_row.get("reference_free_self_convergence")
        if (
            not isinstance(convergence, Mapping)
            or convergence.get("schema") != VPFP_SELF_CONVERGENCE_SUMMARY_SCHEMA
            or convergence.get("reference_solution_used") is not False
            or convergence.get("cross_solver_error_used") is not False
            or convergence.get("accuracy_claim") is not False
        ):
            raise ValueError("VPFP equation-first self-convergence is invalid")
        pass_fraction = float(convergence["auxiliary_pass_fraction"])
        order_raw = convergence.get("aggregate_empirical_order_median")
        order = None if order_raw is None else float(order_raw)
        if (
            not math.isfinite(pass_fraction)
            or not 0.0 <= pass_fraction <= 1.0
            or (order is not None and not math.isfinite(order))
        ):
            raise ValueError("VPFP equation-first convergence summary is nonfinite")

        p90 = [float(row["score_smaller_is_better"]["p90"]) for row in answer_rows]
        cvar90 = [
            float(row["score_smaller_is_better"].get("cvar90", value))
            for row, value in zip(answer_rows, p90)
        ]
        medians = [
            float(row["score_smaller_is_better"]["median"])
            for row in answer_rows
        ]
        time_band_values: list[float] = []
        for row in answer_rows:
            bands = row.get("time_band_p90_errors")
            if isinstance(bands, Mapping):
                time_band_values.extend(float(value) for value in bands.values())
        numeric = p90 + cvar90 + medians + time_band_values
        if not all(math.isfinite(value) and value >= 0.0 for value in numeric):
            raise ValueError("VPFP answer-first ranking metrics are invalid")

        validity = [float(row["valid_prediction_fraction"]) for row in all_rows]
        wall_pass = [float(row["wall_limit_pass_fraction"]) for row in all_rows]
        walls = [float(row["high_budget_wall_seconds"]["mean"]) for row in all_rows]
        metrics[solver_id] = {
            "worst_answer_first_cohort_p90_error": max(p90),
            "mean_answer_first_cohort_p90_error": sum(p90) / len(p90),
            "worst_answer_first_cohort_cvar90_error": max(cvar90),
            "mean_answer_first_cohort_median_error": sum(medians) / len(medians),
            "worst_answer_first_time_band_p90_error": (
                max(time_band_values) if time_band_values else max(p90)
            ),
            "minimum_valid_prediction_fraction": min(validity),
            "minimum_wall_limit_pass_fraction": min(wall_pass),
            "mean_high_budget_wall_seconds": sum(walls) / len(walls),
            "equation_first_self_convergence_pass_fraction": pass_fraction,
            "equation_first_self_convergence_empirical_order_median": order,
            "equation_first_hidden_reference_error_used_for_selection": False,
            "equation_first_cross_solver_error_used_for_selection": False,
            "answer_first_exact_hidden_error_is_primary": True,
        }
    return metrics


def rank_vpfp_robust_candidates(
    metrics: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Rank valid candidates; self-convergence is deliberately a late tie-break."""

    eligible = [
        solver_id
        for solver_id, row in metrics.items()
        if float(row["minimum_valid_prediction_fraction"]) == 1.0
        and float(row["minimum_wall_limit_pass_fraction"]) == 1.0
    ]
    if not eligible:
        raise RuntimeError("no VPFP contender passed validity and wall gates")

    def key(solver_id: str) -> tuple[Any, ...]:
        row = metrics[solver_id]
        order = row["equation_first_self_convergence_empirical_order_median"]
        return (
            float(row["worst_answer_first_cohort_p90_error"]),
            float(row["worst_answer_first_time_band_p90_error"]),
            float(row["mean_answer_first_cohort_p90_error"]),
            float(row["worst_answer_first_cohort_cvar90_error"]),
            -float(row["equation_first_self_convergence_pass_fraction"]),
            math.inf if order is None else -float(order),
            float(row["mean_high_budget_wall_seconds"]),
            solver_id,
        )

    return sorted(eligible, key=key)
