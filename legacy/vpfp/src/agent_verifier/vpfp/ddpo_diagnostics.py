from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


def reward_variance_decomposition(
    rewards: Sequence[float] | np.ndarray,
    group_ids: Sequence[str],
) -> dict[str, Any]:
    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 1 or values.size != len(group_ids):
        raise ValueError("reward decomposition needs one group id per one-dimensional reward")
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("rewards must be non-empty and finite")

    groups: dict[str, list[float]] = defaultdict(list)
    for group_id, reward in zip(group_ids, values):
        key = str(group_id)
        if not key:
            raise ValueError("condition group ids must be non-empty")
        groups[key].append(float(reward))

    overall_mean = float(np.mean(values))
    total_variance = float(np.mean(np.square(values - overall_mean)))
    between_sum = 0.0
    within_sum = 0.0
    within_stds: list[float] = []
    group_sizes: list[int] = []
    group_means: list[float] = []
    group_maxima: list[float] = []
    group_minima: list[float] = []
    for rows in groups.values():
        group_values = np.asarray(rows, dtype=np.float64)
        group_mean = float(np.mean(group_values))
        between_sum += group_values.size * (group_mean - overall_mean) ** 2
        within_sum += float(np.sum(np.square(group_values - group_mean)))
        within_stds.append(float(np.std(group_values)))
        group_sizes.append(int(group_values.size))
        group_means.append(group_mean)
        group_maxima.append(float(np.max(group_values)))
        group_minima.append(float(np.min(group_values)))

    between_variance = between_sum / float(values.size)
    within_variance = within_sum / float(values.size)
    denominator = max(total_variance, np.finfo(np.float64).eps)
    top_5_count = max(1, int(np.ceil(0.05 * values.size)))
    top_10_count = max(1, int(np.ceil(0.10 * values.size)))
    sorted_rewards = np.sort(values)
    within_oracle_mean = float(np.mean(group_maxima))
    return {
        "reward_count": int(values.size),
        "condition_group_count": len(groups),
        "min_group_size": min(group_sizes),
        "max_group_size": max(group_sizes),
        "reward_mean": overall_mean,
        "reward_std": float(np.std(values)),
        "reward_min": float(np.min(values)),
        "reward_max": float(np.max(values)),
        "reward_q90": float(np.quantile(values, 0.90)),
        "reward_q95": float(np.quantile(values, 0.95)),
        "reward_q99": float(np.quantile(values, 0.99)),
        "reward_top_5pct_cvar": float(np.mean(sorted_rewards[-top_5_count:])),
        "reward_top_10pct_cvar": float(np.mean(sorted_rewards[-top_10_count:])),
        "total_variance": total_variance,
        "between_condition_variance": between_variance,
        "within_condition_variance": within_variance,
        "between_variance_fraction": between_variance / denominator,
        "within_variance_fraction": within_variance / denominator,
        "variance_reconstruction_error": abs(total_variance - between_variance - within_variance),
        "mean_within_condition_std": float(np.mean(within_stds)),
        "median_within_condition_std": float(np.median(within_stds)),
        "zero_within_variance_groups": sum(value <= 1.0e-12 for value in within_stds),
        "mean_within_condition_range": float(np.mean(np.asarray(group_maxima) - np.asarray(group_minima))),
        "within_condition_max_oracle_reward": within_oracle_mean,
        "within_condition_max_oracle_relative_uplift": (within_oracle_mean - float(np.mean(group_means)))
        / max(abs(float(np.mean(group_means))), np.finfo(np.float64).eps),
    }


def cosine_agreement(vectors: Mapping[str, Sequence[float] | np.ndarray]) -> dict[str, Any]:
    if len(vectors) < 2:
        raise ValueError("cosine agreement requires at least two vectors")
    names = list(vectors)
    arrays = [np.asarray(vectors[name], dtype=np.float64).reshape(-1) for name in names]
    dimensions = {array.size for array in arrays}
    if len(dimensions) != 1 or not dimensions or next(iter(dimensions)) == 0:
        raise ValueError("agreement vectors must have one common positive dimension")
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("agreement vectors must be finite")
    norms = np.asarray([np.linalg.norm(array) for array in arrays], dtype=np.float64)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("agreement vectors must have nonzero norms")

    matrix = np.empty((len(arrays), len(arrays)), dtype=np.float64)
    for row, left in enumerate(arrays):
        for column, right in enumerate(arrays):
            matrix[row, column] = float(np.dot(left, right) / (norms[row] * norms[column]))
    off_diagonal = matrix[np.triu_indices(len(arrays), k=1)]
    normalized = [array / norm for array, norm in zip(arrays, norms)]
    consensus = np.mean(np.stack(normalized), axis=0)
    consensus_norm = float(np.linalg.norm(consensus))
    consensus_cosines = []
    if consensus_norm > np.finfo(np.float64).eps:
        consensus_cosines = [float(np.dot(array, consensus) / consensus_norm) for array in normalized]
    return {
        "names": names,
        "dimension": int(next(iter(dimensions))),
        "norms": {name: float(norm) for name, norm in zip(names, norms)},
        "matrix": matrix.tolist(),
        "off_diagonal": {
            "count": int(off_diagonal.size),
            "mean": float(np.mean(off_diagonal)),
            "median": float(np.median(off_diagonal)),
            "min": float(np.min(off_diagonal)),
            "max": float(np.max(off_diagonal)),
            "positive_fraction": float(np.mean(off_diagonal > 0.0)),
        },
        "consensus_norm": consensus_norm,
        "consensus_cosines": {name: value for name, value in zip(names, consensus_cosines)},
    }
