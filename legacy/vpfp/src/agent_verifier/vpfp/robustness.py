from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from agent_verifier.vpfp.solver_improvement import paired_solver_statistics


ROBUSTNESS_METRICS = (
    "score_smaller_is_better",
    "rho_rel_l2",
    "g_rel_l2",
    "f_rel_l2",
    "electric_field_rel_l2",
    "vpfp_residual_excess_rel",
    "momentum_density_rel_l2",
    "kinetic_energy_density_rel_l2",
    "pressure_tensor_rel_l2",
    "entropy_density_rel_l2",
    "velocity_tail_mass_rel_l2",
)


def build_guided_target_binding(
    guided_target_solvers: Mapping[str, str],
    guided_attack_audit: Mapping[str, Any],
    solver_bundle_sha256s: Mapping[str, str],
    *,
    require_binding: bool = True,
) -> dict[str, Any]:
    """Bind each claimed target-specific attack to its reward-label solver."""

    audited = dict(guided_attack_audit.get("cohorts") or {})
    declared = {str(key): str(value) for key, value in guided_target_solvers.items()}
    unknown_cohorts = sorted(set(declared) - set(audited))
    missing_cohorts = sorted(set(audited) - set(declared))
    rows: dict[str, Any] = {}
    for cohort_name in sorted(audited):
        target_solver = declared.get(cohort_name)
        expected_sha256 = (
            str(solver_bundle_sha256s.get(target_solver) or "")
            if target_solver is not None
            else ""
        )
        provenance = dict(audited[cohort_name])
        observed = sorted(
            {
                str(value)
                for value in provenance.get("reward_target_solver_sha256s", [])
                if str(value)
            }
        )
        complete = bool(
            provenance.get("reward_target_solver_provenance_complete", False)
        )
        matched = bool(
            target_solver
            and expected_sha256
            and complete
            and observed == [expected_sha256]
        )
        rows[cohort_name] = {
            "target_solver": target_solver,
            "expected_solver_bundle_sha256": expected_sha256 or None,
            "observed_reward_target_solver_sha256s": observed,
            "reward_target_solver_provenance_complete": complete,
            "reward_guidance_checkpoint_sha256": provenance.get(
                "reward_guidance_checkpoint_sha256"
            ),
            "matched": matched,
        }
    passed = (
        not unknown_cohorts
        and not missing_cohorts
        and all(bool(row["matched"]) for row in rows.values())
    )
    return {
        "schema": "vpfp_guided_target_bundle_binding_v2",
        "required": bool(require_binding),
        "passed": bool(passed or not require_binding),
        "all_targets_matched": bool(passed),
        "unknown_cohorts": unknown_cohorts,
        "missing_cohorts": missing_cohorts,
        "cohorts": rows,
    }


def metric_distribution(
    values: list[float] | np.ndarray,
    *,
    cvar_tail_fraction: float = 0.10,
) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("metric distribution requires a nonempty vector")
    if not np.all(np.isfinite(data)):
        raise ValueError("metric distribution contains non-finite values")
    if not 0.0 < float(cvar_tail_fraction) <= 1.0:
        raise ValueError("CVaR tail fraction must lie in (0, 1]")
    tail_count = max(1, int(np.ceil(data.size * float(cvar_tail_fraction))))
    worst_tail = np.partition(data, data.size - tail_count)[-tail_count:]
    return {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "median": float(np.median(data)),
        "p90": float(np.quantile(data, 0.90)),
        "p95": float(np.quantile(data, 0.95)),
        "p99": float(np.quantile(data, 0.99)),
        "max": float(np.max(data)),
        "cvar_tail_fraction": float(cvar_tail_fraction),
        "cvar_worst_tail": float(np.mean(worst_tail)),
    }


def summarize_feedback_distribution(
    feedback: Mapping[str, Any],
    *,
    cvar_tail_fraction: float = 0.10,
) -> dict[str, Any]:
    rows = list(feedback.get("cases") or [])
    if not rows:
        raise ValueError("feedback is empty")
    failed = [
        f"{row.get('case_id')}:{row.get('agent_status')}"
        for row in rows
        if row.get("agent_status") != "ok"
    ]
    if failed:
        raise ValueError(f"solver failed robustness cases: {failed[:5]}")
    metrics: dict[str, Any] = {}
    for key in ROBUSTNESS_METRICS:
        missing = [str(row.get("case_id")) for row in rows if key not in row]
        if missing:
            raise ValueError(
                f"robustness metric {key!r} missing for cases: {missing[:5]}"
            )
        values = [float(row[key]) for row in rows]
        metrics[key] = metric_distribution(
            values,
            cvar_tail_fraction=cvar_tail_fraction,
        )
    return {
        "case_count": len(rows),
        "all_cases_valid": True,
        "case_ids": [str(row["case_id"]) for row in rows],
        "metrics": metrics,
    }


def build_robustness_report(
    feedback_by_cohort: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    champion_name: str,
    candidate_name: str,
    guided_cohorts: list[str] | None = None,
    require_equal_guided_budgets: bool = True,
    cvar_tail_fraction: float = 0.10,
    bootstrap_samples: int = 2_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Build a solver-by-cohort report without using it for candidate selection."""

    if champion_name == candidate_name:
        raise ValueError("champion and candidate names must differ")
    cohort_reports: dict[str, Any] = {}
    guided_names = list(guided_cohorts or [])
    guided_counts: dict[str, int] = {}
    for cohort_index, (cohort_name, solver_feedback) in enumerate(
        sorted(feedback_by_cohort.items())
    ):
        missing = {champion_name, candidate_name} - set(solver_feedback)
        if missing:
            raise ValueError(f"cohort {cohort_name!r} lacks solvers: {sorted(missing)}")
        summaries = {
            solver_name: summarize_feedback_distribution(
                feedback,
                cvar_tail_fraction=cvar_tail_fraction,
            )
            for solver_name, feedback in sorted(solver_feedback.items())
        }
        reference_ids = summaries[champion_name]["case_ids"]
        for solver_name, summary in summaries.items():
            if summary["case_ids"] != reference_ids:
                raise ValueError(
                    f"cohort {cohort_name!r} case order differs for {solver_name!r}"
                )
        comparison = paired_solver_statistics(
            solver_feedback[champion_name],
            solver_feedback[candidate_name],
            bootstrap_samples=int(bootstrap_samples),
            seed=int(seed) + cohort_index * 100,
        )
        cohort_reports[cohort_name] = {
            "case_count": len(reference_ids),
            "solvers": summaries,
            "candidate_vs_champion": comparison,
        }
        if cohort_name in guided_names:
            guided_counts[cohort_name] = len(reference_ids)
    equal_guided_budgets = len(set(guided_counts.values())) <= 1
    if require_equal_guided_budgets and guided_names:
        missing_guided = set(guided_names) - set(guided_counts)
        if missing_guided:
            raise ValueError(f"missing guided cohorts: {sorted(missing_guided)}")
        if not equal_guided_budgets:
            raise ValueError(f"guided attack budgets differ: {guided_counts}")

    solver_macro: dict[str, Any] = {}
    for solver_name in (champion_name, candidate_name):
        score_blocks = [
            cohort["solvers"][solver_name]["metrics"]["score_smaller_is_better"]
            for cohort in cohort_reports.values()
        ]
        solver_macro[solver_name] = {
            "cohort_count": len(score_blocks),
            "macro_mean_score": float(np.mean([row["mean"] for row in score_blocks])),
            "worst_cohort_mean_score": float(max(row["mean"] for row in score_blocks)),
            "worst_case_score_across_cohorts": float(max(row["max"] for row in score_blocks)),
            "worst_cvar_across_cohorts": float(
                max(row["cvar_worst_tail"] for row in score_blocks)
            ),
        }
    return {
        "schema": "vpfp_solver_robustness_matrix_v1",
        "used_for_candidate_selection": False,
        "champion_name": champion_name,
        "candidate_name": candidate_name,
        "guided_cohorts": guided_names,
        "guided_attack_budgets": guided_counts,
        "equal_guided_attack_budgets": equal_guided_budgets,
        "cvar_tail_fraction": float(cvar_tail_fraction),
        "cohorts": cohort_reports,
        "solver_macro": solver_macro,
    }
