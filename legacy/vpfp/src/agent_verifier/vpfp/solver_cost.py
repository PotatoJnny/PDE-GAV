from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from agent_verifier.vpfp.robustness import metric_distribution


SOLVER_COMPUTE_SUMMARY_SCHEMA = "vpfp_solver_compute_summary_v1"
COMPUTE_AWARE_PROMOTION_SCHEMA = "vpfp_compute_aware_promotion_gate_v1"


def summarize_solver_compute(
    feedback: Mapping[str, Any],
    *,
    cvar_tail_fraction: float = 0.10,
    cvar_weight: float = 0.50,
    soft_wall_time_seconds: float = 600.0,
    hard_wall_time_seconds: float = 2700.0,
    soft_wall_time_seconds_per_case: float | None = None,
    hard_wall_time_seconds_per_case: float | None = None,
) -> dict[str, Any]:
    """Summarize accuracy and end-to-end inference cost for one cohort.

    The internal state representation is deliberately absent. This contract
    scores only numerical output quality and the resources needed to produce it.
    """

    if not 0.0 < float(cvar_tail_fraction) <= 1.0:
        raise ValueError("CVaR tail fraction must lie in (0, 1]")
    if float(cvar_weight) < 0.0:
        raise ValueError("CVaR weight must be nonnegative")
    if not 0.0 < float(soft_wall_time_seconds) <= float(hard_wall_time_seconds):
        raise ValueError("wall-time budgets must satisfy 0 < soft <= hard")
    if (soft_wall_time_seconds_per_case is None) != (
        hard_wall_time_seconds_per_case is None
    ):
        raise ValueError("per-case soft and hard budgets must be supplied together")
    if soft_wall_time_seconds_per_case is not None and not (
        0.0
        < float(soft_wall_time_seconds_per_case)
        <= float(hard_wall_time_seconds_per_case)
    ):
        raise ValueError("per-case wall-time budgets must satisfy 0 < soft <= hard")

    rows = list(feedback.get("cases") or [])
    summary = dict(feedback.get("summary") or {})
    valid_rows = [
        row
        for row in rows
        if row.get("agent_status") == "ok"
        and _is_finite_number(row.get("score_smaller_is_better"))
    ]
    all_cases_valid = bool(rows) and len(valid_rows) == len(rows)
    score_distribution = (
        metric_distribution(
            [float(row["score_smaller_is_better"]) for row in valid_rows],
            cvar_tail_fraction=float(cvar_tail_fraction),
        )
        if valid_rows
        else None
    )
    accuracy_objective = (
        float(score_distribution["mean"])
        + float(cvar_weight) * float(score_distribution["cvar_worst_tail"])
        if score_distribution is not None
        else None
    )

    call_times = [
        float(row["solver_wall_time_seconds"])
        for row in rows
        if _is_finite_number(row.get("solver_wall_time_seconds"))
        and float(row["solver_wall_time_seconds"]) >= 0.0
    ]
    runtime_distribution = (
        metric_distribution(call_times, cvar_tail_fraction=0.10)
        if call_times
        else None
    )
    load_seconds = _optional_nonnegative(
        summary.get("solver_load_wall_time_seconds")
    )
    call_total_seconds = _optional_nonnegative(
        summary.get("solver_call_total_wall_time_seconds")
    )
    if call_total_seconds is None and len(call_times) == len(rows):
        call_total_seconds = float(sum(call_times))
    measured_end_to_end = _optional_nonnegative(
        summary.get("solver_evaluation_wall_time_seconds")
    )
    if measured_end_to_end is not None:
        end_to_end_seconds = measured_end_to_end
        runtime_source = "measured_full_evaluation"
    elif load_seconds is not None and call_total_seconds is not None:
        end_to_end_seconds = load_seconds + call_total_seconds
        runtime_source = "load_plus_solver_calls"
    else:
        end_to_end_seconds = None
        runtime_source = "unavailable"

    runtime_available = end_to_end_seconds is not None
    within_soft_budget = bool(
        runtime_available
        and float(end_to_end_seconds) <= float(soft_wall_time_seconds)
    )
    within_hard_budget = bool(
        runtime_available
        and float(end_to_end_seconds) <= float(hard_wall_time_seconds)
    )
    per_case_runtime_available = bool(rows) and len(call_times) == len(rows)
    max_case_seconds = (
        float(max(call_times)) if per_case_runtime_available else None
    )
    if hard_wall_time_seconds_per_case is None:
        within_per_case_soft_budget = True
        within_per_case_hard_budget = True
        per_case_soft_excess_fraction = None
    else:
        within_per_case_soft_budget = bool(
            per_case_runtime_available
            and max_case_seconds <= float(soft_wall_time_seconds_per_case)
        )
        within_per_case_hard_budget = bool(
            per_case_runtime_available
            and max_case_seconds <= float(hard_wall_time_seconds_per_case)
        )
        per_case_soft_excess_fraction = (
            max(
                0.0,
                max_case_seconds / float(soft_wall_time_seconds_per_case) - 1.0,
            )
            if per_case_runtime_available
            else None
        )
    soft_excess_fraction = (
        max(
            0.0,
            float(end_to_end_seconds) / float(soft_wall_time_seconds) - 1.0,
        )
        if runtime_available
        else None
    )
    return {
        "schema": SOLVER_COMPUTE_SUMMARY_SCHEMA,
        "case_count": len(rows),
        "valid_case_count": len(valid_rows),
        "all_cases_valid": all_cases_valid,
        "accuracy": {
            "score_distribution": score_distribution,
            "cvar_tail_fraction": float(cvar_tail_fraction),
            "cvar_weight": float(cvar_weight),
            "mean_plus_weighted_cvar": accuracy_objective,
        },
        "runtime": {
            "available": runtime_available,
            "source": runtime_source,
            "solver_load_wall_time_seconds": load_seconds,
            "solver_call_total_wall_time_seconds": call_total_seconds,
            "solver_evaluation_wall_time_seconds": measured_end_to_end,
            "selection_wall_time_seconds": end_to_end_seconds,
            "per_case_distribution": runtime_distribution,
            "per_case_runtime_available": per_case_runtime_available,
            "solver_call_max_wall_time_seconds": max_case_seconds,
            "process_cpu_time_seconds": _optional_nonnegative(
                summary.get("solver_process_cpu_time_seconds")
            ),
            "process_cpu_to_wall_ratio": _optional_nonnegative(
                summary.get("solver_process_cpu_to_wall_ratio")
            ),
            "peak_rss_mib": _optional_nonnegative(
                summary.get("solver_peak_rss_mib")
            ),
        },
        "budget": {
            "soft_wall_time_seconds": float(soft_wall_time_seconds),
            "hard_wall_time_seconds": float(hard_wall_time_seconds),
            "within_soft_budget": within_soft_budget,
            "within_hard_budget": within_hard_budget,
            "soft_excess_fraction": soft_excess_fraction,
            "soft_wall_time_seconds_per_case": soft_wall_time_seconds_per_case,
            "hard_wall_time_seconds_per_case": hard_wall_time_seconds_per_case,
            "within_per_case_soft_budget": within_per_case_soft_budget,
            "within_per_case_hard_budget": within_per_case_hard_budget,
            "per_case_soft_excess_fraction": per_case_soft_excess_fraction,
            "hard_gate_passed": bool(
                all_cases_valid
                and within_hard_budget
                and within_per_case_hard_budget
            ),
        },
        "selection_contract": {
            "primary": "numerical_accuracy_mean_plus_weighted_cvar",
            "runtime_role": "wide_hard_constraint_and_near_accuracy_tiebreak",
            "internal_representation_restricted": False,
            "state_dimension_used_for_selection": False,
        },
    }


def select_compute_aware_candidate(
    candidates: list[Mapping[str, Any]],
    *,
    relative_accuracy_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Select a hard-budget-passing candidate from an accuracy-near-best set."""

    if not candidates:
        raise ValueError("candidate selection requires at least one candidate")
    if float(relative_accuracy_tolerance) < 0.0:
        raise ValueError("relative accuracy tolerance must be nonnegative")
    eligible: list[tuple[int, Mapping[str, Any], float, float]] = []
    for index, candidate in enumerate(candidates):
        cost = dict(candidate.get("compute_summary") or {})
        budget = dict(cost.get("budget") or {})
        accuracy = dict(cost.get("accuracy") or {})
        runtime = dict(cost.get("runtime") or {})
        objective = accuracy.get("mean_plus_weighted_cvar")
        wall_time = runtime.get("selection_wall_time_seconds")
        if (
            bool(candidate.get("eligible", True))
            and bool(budget.get("hard_gate_passed"))
            and _is_finite_number(objective)
            and _is_finite_number(wall_time)
        ):
            eligible.append(
                (index, candidate, float(objective), float(wall_time))
            )
    if not eligible:
        raise ValueError("no candidate passed validity and hard wall-time gates")
    best_accuracy = min(row[2] for row in eligible)
    tolerance = max(
        abs(best_accuracy) * float(relative_accuracy_tolerance),
        1.0e-12,
    )
    near_best = [
        row for row in eligible if row[2] <= best_accuracy + tolerance
    ]
    selected = min(near_best, key=lambda row: (row[3], row[2], row[0]))
    return {
        "schema": "vpfp_compute_aware_candidate_selection_v1",
        "selected_index": int(selected[0]),
        "best_accuracy_objective": float(best_accuracy),
        "relative_accuracy_tolerance": float(relative_accuracy_tolerance),
        "absolute_accuracy_tolerance": float(tolerance),
        "eligible_indices": [int(row[0]) for row in eligible],
        "accuracy_near_best_indices": [int(row[0]) for row in near_best],
        "selected_accuracy_objective": float(selected[2]),
        "selected_wall_time_seconds": float(selected[3]),
        "selection_rule": (
            "validity and wide hard budget, then fastest among candidates "
            "within the registered accuracy tolerance"
        ),
        "state_dimension_used_for_selection": False,
    }


def evaluate_compute_aware_promotion(
    *,
    champion_private: Mapping[str, Any],
    candidate_private: Mapping[str, Any],
    champion_guard: Mapping[str, Any],
    candidate_guard: Mapping[str, Any],
    min_relative_accuracy_improvement: float = 0.01,
    max_relative_accuracy_regression_for_speedup: float = 0.02,
    min_relative_runtime_improvement: float = 0.20,
    max_relative_guard_accuracy_regression: float = 0.02,
) -> dict[str, Any]:
    """Promote for accuracy gain or comparable accuracy with real speedup."""

    thresholds = {
        "min_relative_accuracy_improvement": float(
            min_relative_accuracy_improvement
        ),
        "max_relative_accuracy_regression_for_speedup": float(
            max_relative_accuracy_regression_for_speedup
        ),
        "min_relative_runtime_improvement": float(
            min_relative_runtime_improvement
        ),
        "max_relative_guard_accuracy_regression": float(
            max_relative_guard_accuracy_regression
        ),
    }
    if any(value < 0.0 for value in thresholds.values()):
        raise ValueError("compute-aware promotion thresholds must be nonnegative")

    cp_accuracy, cp_runtime = _objective_and_runtime(champion_private)
    np_accuracy, np_runtime = _objective_and_runtime(candidate_private)
    cg_accuracy, _ = _objective_and_runtime(champion_guard)
    ng_accuracy, _ = _objective_and_runtime(candidate_guard)
    relative_accuracy_improvement = (cp_accuracy - np_accuracy) / max(
        abs(cp_accuracy), 1.0e-12
    )
    relative_runtime_improvement = (cp_runtime - np_runtime) / max(
        abs(cp_runtime), 1.0e-12
    )
    relative_private_regression = (np_accuracy - cp_accuracy) / max(
        abs(cp_accuracy), 1.0e-12
    )
    relative_guard_regression = (ng_accuracy - cg_accuracy) / max(
        abs(cg_accuracy), 1.0e-12
    )

    candidate_budget_passed = bool(
        dict(candidate_private.get("budget") or {}).get("hard_gate_passed")
        and dict(candidate_guard.get("budget") or {}).get("hard_gate_passed")
    )
    guard_passed = (
        relative_guard_regression
        <= float(max_relative_guard_accuracy_regression)
    )
    accuracy_route = (
        relative_accuracy_improvement
        >= float(min_relative_accuracy_improvement)
    )
    speed_route = (
        relative_private_regression
        <= float(max_relative_accuracy_regression_for_speedup)
        and relative_runtime_improvement
        >= float(min_relative_runtime_improvement)
    )
    accepted = bool(
        candidate_budget_passed
        and guard_passed
        and (accuracy_route or speed_route)
    )
    return {
        "schema": COMPUTE_AWARE_PROMOTION_SCHEMA,
        "accepted": accepted,
        "primary_objective": (
            "efficient_numerical_approximation_of_required_outputs"
        ),
        "accepted_route": (
            "accuracy"
            if accepted and accuracy_route
            else "comparable_accuracy_with_speedup"
            if accepted and speed_route
            else None
        ),
        "candidate_hard_budget_passed": candidate_budget_passed,
        "guard_accuracy_gate_passed": guard_passed,
        "accuracy_route_passed": accuracy_route,
        "speed_route_passed": speed_route,
        "relative_private_accuracy_improvement": float(
            relative_accuracy_improvement
        ),
        "relative_private_accuracy_regression": float(
            relative_private_regression
        ),
        "relative_private_runtime_improvement": float(
            relative_runtime_improvement
        ),
        "relative_guard_accuracy_regression": float(
            relative_guard_regression
        ),
        "thresholds": thresholds,
        "state_dimension_used_for_promotion": False,
        "traditional_reduced_model_required": False,
    }


def _objective_and_runtime(summary: Mapping[str, Any]) -> tuple[float, float]:
    accuracy = dict(summary.get("accuracy") or {}).get(
        "mean_plus_weighted_cvar"
    )
    runtime = dict(summary.get("runtime") or {}).get(
        "selection_wall_time_seconds"
    )
    if not _is_finite_number(accuracy) or not _is_finite_number(runtime):
        raise ValueError("compute summary lacks finite accuracy or runtime")
    return float(accuracy), float(runtime)


def _optional_nonnegative(value: Any) -> float | None:
    if not _is_finite_number(value):
        return None
    result = float(value)
    return result if result >= 0.0 else None


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and np.isfinite(float(value))
    )
