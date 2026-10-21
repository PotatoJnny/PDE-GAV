from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


EPS = 1.0e-6


@dataclass(frozen=True)
class PolicyAcceptanceThresholds:
    min_mean_delta: float = 0.0
    min_win_rate: float = 0.50
    min_one_sided_ci90_low: float = 0.0
    max_abs_mean_log_ratio: float = 1.0e-3

    def as_dict(self) -> dict[str, float]:
        return {
            "min_mean_delta": float(self.min_mean_delta),
            "min_win_rate": float(self.min_win_rate),
            "min_one_sided_ci90_low": float(self.min_one_sided_ci90_low),
            "max_abs_mean_log_ratio": float(self.max_abs_mean_log_ratio),
        }


def compute_advantages(
    rewards: Sequence[float] | np.ndarray,
    *,
    mode: str,
    group_ids: Sequence[str] | None = None,
    epsilon: float = EPS,
) -> np.ndarray:
    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("rewards must be one-dimensional")
    if values.size == 0:
        return values.copy()
    if not np.all(np.isfinite(values)):
        raise ValueError("rewards must be finite")

    resolved = str(mode)
    if resolved == "centered":
        return values - float(np.mean(values))
    if resolved == "zscore":
        return (values - float(np.mean(values))) / max(float(np.std(values)), float(epsilon))
    if resolved == "rank":
        return _standardized_ranks(values, epsilon=float(epsilon))
    if resolved not in {"group_centered", "group_rank"}:
        raise ValueError(f"unsupported advantage mode: {resolved}")
    if group_ids is None or len(group_ids) != values.size:
        raise ValueError(f"{resolved} advantages require one group id per reward")

    groups: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        key = str(group_id)
        if not key:
            raise ValueError(f"missing group id for reward index {index}")
        groups[key].append(index)

    output = np.zeros_like(values)
    for group_id, indices in groups.items():
        if len(indices) < 2:
            raise ValueError(f"group-relative advantage needs at least two samples: {group_id}")
        group_values = values[indices]
        if float(np.std(group_values)) <= float(epsilon):
            continue
        if resolved == "group_rank":
            output[indices] = _standardized_ranks(group_values, epsilon=float(epsilon))
            continue
        total = float(np.sum(group_values))
        leave_one_out = (total - group_values) / float(len(indices) - 1)
        centered = group_values - leave_one_out
        output[indices] = centered / max(float(np.std(centered)), float(epsilon))
    return output


def advantage_diagnostics(advantages: Sequence[float], group_ids: Sequence[str] | None = None) -> dict[str, Any]:
    values = np.asarray(advantages, dtype=np.float64)
    payload: dict[str, Any] = {
        "count": int(values.size),
        "mean": float(np.mean(values)) if values.size else None,
        "std": float(np.std(values)) if values.size else None,
        "nonzero_count": int(np.count_nonzero(np.abs(values) > 1.0e-12)),
    }
    if group_ids is None:
        return payload
    groups: dict[str, list[float]] = defaultdict(list)
    for group_id, value in zip(group_ids, values):
        groups[str(group_id)].append(float(value))
    payload.update(
        {
            "group_count": len(groups),
            "nonzero_group_count": sum(any(abs(value) > 1.0e-12 for value in rows) for rows in groups.values()),
            "min_group_size": min((len(rows) for rows in groups.values()), default=0),
            "max_group_size": max((len(rows) for rows in groups.values()), default=0),
            "max_abs_group_mean": max((abs(float(np.mean(rows))) for rows in groups.values()), default=0.0),
        }
    )
    return payload


def paired_policy_statistics(
    current_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 20260709,
) -> dict[str, Any]:
    current = _reward_rows_by_pair(current_rows)
    candidate = _reward_rows_by_pair(candidate_rows)
    current_ids = set(current)
    candidate_ids = set(candidate)
    common = sorted(current_ids & candidate_ids)
    missing_current = sorted(candidate_ids - current_ids)
    missing_candidate = sorted(current_ids - candidate_ids)
    if not common:
        raise ValueError("paired policy comparison has no common pair ids")

    deltas: list[float] = []
    current_rewards: list[float] = []
    candidate_rewards: list[float] = []
    group_ids: list[str] = []
    pairs: list[dict[str, Any]] = []
    for pair_id in common:
        before = current[pair_id]
        after = candidate[pair_id]
        before_group = str(before.get("condition_group_id") or pair_id)
        after_group = str(after.get("condition_group_id") or pair_id)
        if before_group != after_group:
            raise ValueError(f"condition group mismatch for pair {pair_id}: {before_group} != {after_group}")
        before_reward = float(before["reward"])
        after_reward = float(after["reward"])
        delta = after_reward - before_reward
        current_rewards.append(before_reward)
        candidate_rewards.append(after_reward)
        deltas.append(delta)
        group_ids.append(before_group)
        pairs.append(
            {
                "pair_id": pair_id,
                "condition_group_id": before_group,
                "current_case_id": before.get("case_id"),
                "candidate_case_id": after.get("case_id"),
                "current_reward": before_reward,
                "candidate_reward": after_reward,
                "delta": delta,
            }
        )

    delta_values = np.asarray(deltas, dtype=np.float64)
    before_values = np.asarray(current_rewards, dtype=np.float64)
    after_values = np.asarray(candidate_rewards, dtype=np.float64)
    ci = cluster_bootstrap_mean_ci(
        delta_values,
        group_ids,
        samples=int(bootstrap_samples),
        seed=int(seed),
    )
    mean_delta = float(np.mean(delta_values))
    return {
        "pair_count": len(common),
        "condition_group_count": len(set(group_ids)),
        "missing_current_pair_ids": missing_current,
        "missing_candidate_pair_ids": missing_candidate,
        "all_pairs_present": not missing_current and not missing_candidate,
        "current_mean_reward": float(np.mean(before_values)),
        "candidate_mean_reward": float(np.mean(after_values)),
        "mean_delta": mean_delta,
        "median_delta": float(np.median(delta_values)),
        "relative_mean_delta": mean_delta / max(abs(float(np.mean(before_values))), 1.0e-12),
        "win_rate": float(np.mean(delta_values > 0.0)),
        "standard_error": float(np.std(delta_values, ddof=1) / np.sqrt(delta_values.size)) if delta_values.size > 1 else 0.0,
        "delta_quantiles": {
            "q10": float(np.quantile(delta_values, 0.10)),
            "q25": float(np.quantile(delta_values, 0.25)),
            "q50": float(np.quantile(delta_values, 0.50)),
            "q75": float(np.quantile(delta_values, 0.75)),
            "q90": float(np.quantile(delta_values, 0.90)),
        },
        "cluster_bootstrap": ci,
        "pairs": pairs,
    }


def cluster_bootstrap_mean_ci(
    deltas: Sequence[float] | np.ndarray,
    group_ids: Sequence[str],
    *,
    samples: int = 10_000,
    seed: int = 20260709,
) -> dict[str, float | int]:
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or values.size != len(group_ids):
        raise ValueError("cluster bootstrap needs one group id per one-dimensional delta")
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("cluster bootstrap deltas must be non-empty and finite")
    grouped: dict[str, list[float]] = defaultdict(list)
    for group_id, value in zip(group_ids, values):
        grouped[str(group_id)].append(float(value))
    group_means = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64)
    if group_means.size == 1:
        draws = np.repeat(group_means[0], max(1, int(samples)))
    else:
        rng = np.random.default_rng(int(seed))
        indices = rng.integers(0, group_means.size, size=(max(1, int(samples)), group_means.size))
        draws = np.mean(group_means[indices], axis=1)
    return {
        "samples": int(draws.size),
        "cluster_count": int(group_means.size),
        "one_sided_ci90_low": float(np.quantile(draws, 0.10)),
        "two_sided_ci90_low": float(np.quantile(draws, 0.05)),
        "two_sided_ci90_high": float(np.quantile(draws, 0.95)),
        "two_sided_ci95_low": float(np.quantile(draws, 0.025)),
        "two_sided_ci95_high": float(np.quantile(draws, 0.975)),
    }


def evaluate_policy_acceptance(
    paired: dict[str, Any],
    *,
    generated_accepted: bool,
    sample_hard_passed: bool,
    mean_log_ratio: float,
    thresholds: PolicyAcceptanceThresholds | None = None,
) -> dict[str, Any]:
    limits = thresholds or PolicyAcceptanceThresholds()
    finite = all(
        np.isfinite(float(value))
        for value in (
            paired.get("current_mean_reward"),
            paired.get("candidate_mean_reward"),
            paired.get("mean_delta"),
            paired.get("win_rate"),
            paired.get("cluster_bootstrap", {}).get("one_sided_ci90_low"),
            mean_log_ratio,
        )
    )
    gates = [
        _gate("all_pairs_present", bool(paired.get("all_pairs_present")), True),
        _gate("finite_metrics", finite, True),
        _gate("paired_mean_delta", float(paired["mean_delta"]), float(limits.min_mean_delta), op=">"),
        _gate("paired_win_rate", float(paired["win_rate"]), float(limits.min_win_rate), op=">"),
        _gate(
            "one_sided_ci90_low",
            float(paired["cluster_bootstrap"]["one_sided_ci90_low"]),
            float(limits.min_one_sided_ci90_low),
            op=">=",
        ),
        _gate("generated_acceptance", bool(generated_accepted), True),
        _gate("sample_hard_passed", bool(sample_hard_passed), True),
        _gate("abs_mean_log_ratio", abs(float(mean_log_ratio)), float(limits.max_abs_mean_log_ratio), op="<="),
    ]
    return {
        "accepted": all(bool(gate["passed"]) for gate in gates),
        "thresholds": limits.as_dict(),
        "gates": gates,
    }


def _reward_rows_by_pair(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        reward = row.get("reward")
        if not isinstance(reward, (int, float)) or not np.isfinite(float(reward)):
            continue
        pair_id = str(row.get("pair_id") or "")
        if not pair_id:
            raise ValueError(f"reward row has no pair_id: {row.get('case_id')}")
        if pair_id in result:
            raise ValueError(f"duplicate pair_id in reward table: {pair_id}")
        result[pair_id] = dict(row)
    return result


def _standardized_ranks(values: np.ndarray, *, epsilon: float) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = 0.5 * float(start + end - 1)
        ranks[order[start:end]] = average_rank
        start = end
    centered = ranks - float(np.mean(ranks))
    std = float(np.std(centered))
    if std <= float(epsilon):
        return np.zeros_like(centered)
    return centered / std


def _gate(name: str, value: Any, threshold: Any, *, op: str = "==") -> dict[str, Any]:
    if op == ">":
        passed = float(value) > float(threshold)
    elif op == ">=":
        passed = float(value) >= float(threshold)
    elif op == "<=":
        passed = float(value) <= float(threshold)
    elif op == "==":
        passed = value == threshold
    else:
        raise ValueError(f"unsupported acceptance operator: {op}")
    return {"name": name, "value": value, "op": op, "threshold": threshold, "passed": bool(passed)}
