from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import numpy as np


COMPONENT_KEYS = (
    "score_smaller_is_better",
    "rho_rel_l2",
    "g_rel_l2",
    "f_rel_l2",
    "electric_field_rel_l2",
    "momentum_density_rel_l2",
    "kinetic_energy_density_rel_l2",
    "pressure_tensor_rel_l2",
    "entropy_density_rel_l2",
    "velocity_tail_mass_rel_l2",
    "vpfp_residual_excess_rel",
)


def paired_solver_statistics(
    champion_feedback: Mapping[str, Any],
    candidate_feedback: Mapping[str, Any],
    *,
    group_by_case: Mapping[str, str] | None = None,
    bootstrap_samples: int = 2_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare two frozen solvers on exactly the same private cases.

    Errors are smaller-is-better. Positive improvement means that the candidate
    reduced error relative to the champion.
    """

    champion = _successful_rows(champion_feedback, label="champion")
    candidate = _successful_rows(candidate_feedback, label="candidate")
    champion_ids = set(champion)
    candidate_ids = set(candidate)
    if champion_ids != candidate_ids:
        raise ValueError(
            "solver comparison coverage mismatch: "
            f"missing_champion={sorted(candidate_ids - champion_ids)[:5]}, "
            f"missing_candidate={sorted(champion_ids - candidate_ids)[:5]}"
        )
    if not champion_ids:
        raise ValueError("solver comparison has no common successful cases")

    case_ids = sorted(champion_ids)
    group_ids = [
        str((group_by_case or {}).get(case_id) or case_id)
        for case_id in case_ids
    ]
    components: dict[str, dict[str, Any]] = {}
    for component_index, key in enumerate(COMPONENT_KEYS):
        if not all(key in champion[case_id] and key in candidate[case_id] for case_id in case_ids):
            continue
        champion_values = np.asarray(
            [float(champion[case_id][key]) for case_id in case_ids], dtype=np.float64
        )
        candidate_values = np.asarray(
            [float(candidate[case_id][key]) for case_id in case_ids], dtype=np.float64
        )
        if not np.all(np.isfinite(champion_values)) or not np.all(np.isfinite(candidate_values)):
            raise ValueError(f"non-finite {key} in solver comparison")
        improvements = champion_values - candidate_values
        champion_mean = float(np.mean(champion_values))
        candidate_mean = float(np.mean(candidate_values))
        mean_improvement = float(np.mean(improvements))
        bootstrap = _cluster_bootstrap(
            improvements,
            group_ids,
            samples=int(bootstrap_samples),
            seed=int(seed) + component_index,
        )
        components[key] = {
            "champion_mean": champion_mean,
            "candidate_mean": candidate_mean,
            "mean_improvement": mean_improvement,
            "mean_delta_candidate_minus_champion": -mean_improvement,
            "relative_mean_improvement": mean_improvement / max(abs(champion_mean), 1.0e-12),
            "candidate_win_rate": float(np.mean(candidate_values < champion_values)),
            "tie_rate": float(np.mean(candidate_values == champion_values)),
            "improvement_quantiles": {
                "q10": float(np.quantile(improvements, 0.10)),
                "q50": float(np.quantile(improvements, 0.50)),
                "q90": float(np.quantile(improvements, 0.90)),
            },
            "cluster_bootstrap": bootstrap,
        }

    score = components.get("score_smaller_is_better")
    if score is None:
        raise ValueError("solver comparison is missing score_smaller_is_better")
    return {
        "schema": "vpfp_paired_solver_statistics_v1",
        "case_count": len(case_ids),
        "condition_group_count": len(set(group_ids)),
        "all_cases_paired": True,
        "score": score,
        "components": components,
        "case_ids": case_ids,
    }


def evaluate_solver_promotion(
    private_comparison: Mapping[str, Any],
    d0_comparison: Mapping[str, Any],
    *,
    min_relative_private_improvement: float = 0.01,
    max_relative_d0_regression: float = 0.01,
    d0_epsilon_abs: float = 1.0e-8,
    require_private_ci: bool = True,
) -> dict[str, Any]:
    """Apply the champion gate to paired private and D0 comparisons."""

    private_score = _score_block(private_comparison, label="private")
    d0_score = _score_block(d0_comparison, label="D0")
    champion_d0 = float(d0_score["champion_mean"])
    candidate_d0 = float(d0_score["candidate_mean"])
    d0_slack = max(
        float(d0_epsilon_abs),
        abs(champion_d0) * float(max_relative_d0_regression),
    )
    d0_allowed = champion_d0 + d0_slack
    private_relative = float(private_score["relative_mean_improvement"])
    private_ci_low = float(private_score["cluster_bootstrap"]["ci95_low"])
    gates = [
        _gate(
            "private_relative_mean_improvement",
            private_relative,
            float(min_relative_private_improvement),
            op=">=",
        ),
        _gate("d0_no_regression", candidate_d0, d0_allowed, op="<="),
    ]
    if require_private_ci:
        gates.append(_gate("private_cluster_ci95_low", private_ci_low, 0.0, op=">"))
    accepted = all(bool(gate["passed"]) for gate in gates)
    return {
        "schema": "vpfp_solver_promotion_gate_v1",
        "accepted": accepted,
        "primary_metric": "paired_private_score_reduction",
        "generator_difficulty_is_primary": False,
        "private_score": private_score,
        "d0_score": d0_score,
        "d0_allowed_candidate_mean": d0_allowed,
        "thresholds": {
            "min_relative_private_improvement": float(min_relative_private_improvement),
            "max_relative_d0_regression": float(max_relative_d0_regression),
            "d0_epsilon_abs": float(d0_epsilon_abs),
            "require_private_ci": bool(require_private_ci),
        },
        "gates": gates,
        "reason": (
            "accepted: candidate solver improves private cases without D0 regression"
            if accepted
            else "rejected: candidate solver did not pass all solver-strength gates"
        ),
    }


def case_condition_groups(case_dirs: list[Any]) -> dict[str, str]:
    """Read controller-private condition groups without exposing them to the agent."""

    import json
    from pathlib import Path

    groups: dict[str, str] = {}
    for raw_case_dir in case_dirs:
        case_dir = Path(raw_case_dir).resolve()
        question = json.loads((case_dir / "public_question.json").read_text(encoding="utf-8"))
        case_id = str(question.get("case_id") or case_dir.name)
        metadata_path = case_dir / "generator_metadata.json"
        group_id = case_id
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            group_id = str(
                metadata.get("condition_group_id")
                or metadata.get("condition_signature")
                or case_id
            )
        if case_id in groups:
            raise ValueError(f"duplicate private case ID: {case_id}")
        groups[case_id] = group_id
    return groups


def condition_signatures(case_dirs: list[Any]) -> set[str]:
    import json
    from pathlib import Path

    signatures: set[str] = set()
    for raw_case_dir in case_dirs:
        metadata_path = Path(raw_case_dir).resolve() / "generator_metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        signature = metadata.get("condition_signature")
        if signature:
            signatures.add(str(signature))
    return signatures


def _successful_rows(feedback: Mapping[str, Any], *, label: str) -> dict[str, Mapping[str, Any]]:
    rows = list(feedback.get("cases") or [])
    if not rows:
        raise ValueError(f"{label} feedback is empty")
    result: dict[str, Mapping[str, Any]] = {}
    failures: list[str] = []
    for row in rows:
        case_id = str(row.get("case_id") or "")
        if not case_id or case_id in result:
            raise ValueError(f"invalid or duplicate {label} case ID: {case_id!r}")
        if row.get("agent_status") != "ok":
            failures.append(f"{case_id}:{row.get('agent_status')}")
        result[case_id] = row
    if failures:
        raise ValueError(f"{label} solver failed private cases: {failures[:5]}")
    return result


def _cluster_bootstrap(values, group_ids, *, samples: int, seed: int) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for group_id, value in zip(group_ids, values, strict=True):
        grouped[str(group_id)].append(float(value))
    group_means = np.asarray(
        [np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64
    )
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(
        0,
        len(group_means),
        size=(max(1, int(samples)), len(group_means)),
    )
    draws = np.mean(group_means[indices], axis=1)
    return {
        "samples": int(len(draws)),
        "cluster_count": int(len(group_means)),
        "ci90_low": float(np.quantile(draws, 0.05)),
        "ci90_high": float(np.quantile(draws, 0.95)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def _score_block(comparison: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    score = comparison.get("score")
    if not isinstance(score, Mapping):
        raise ValueError(f"{label} comparison has no score block")
    return score


def _gate(name: str, value: float, threshold: float, *, op: str) -> dict[str, Any]:
    if op == ">=":
        passed = float(value) >= float(threshold)
    elif op == ">":
        passed = float(value) > float(threshold)
    elif op == "<=":
        passed = float(value) <= float(threshold)
    else:
        raise ValueError(op)
    return {
        "name": name,
        "value": float(value),
        "op": op,
        "threshold": float(threshold),
        "passed": bool(passed),
    }
