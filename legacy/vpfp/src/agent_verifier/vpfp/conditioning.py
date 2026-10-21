from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from agent_verifier.vpfp.autoencoder import features_to_rho_g
from agent_verifier.vpfp.operators import EPS, VPFPGrid, f_from_rho_g, make_axes, relative_l2

Array = np.ndarray

DEFAULT_SHAPE_CONDITION_KEYS = (
    "rho_temporal_rel_l2",
    "f_temporal_rel_l2",
    "mean_abs_u_vx",
    "mean_abs_u_vy",
    "rho_std",
    "f_rms",
)


def shape_condition_from_features(
    features: Array,
    grid: VPFPGrid,
    basis_config,
    *,
    coeffs_are_logits: bool,
    keys: Sequence[str] = DEFAULT_SHAPE_CONDITION_KEYS,
) -> dict[str, float]:
    """Return low-dimensional VPFP solution-shape descriptors.

    These values are public-training descriptors used to condition the generator;
    generated cases still derive their public source from the sampled hidden
    answer and do not expose the descriptor as a public target.
    """

    rho, g, _coeffs = features_to_rho_g(features, grid, basis_config, coeffs_are_logits=coeffs_are_logits)
    f = f_from_rho_g(rho, g)
    axes = make_axes(grid)
    vx = axes["vx"][None, None, None, :, None]
    vy = axes["vy"][None, None, None, None, :]
    weight = float(grid.velocity_cell_area)
    ux = np.sum(g * vx, axis=(-2, -1)) * weight
    uy = np.sum(g * vy, axis=(-2, -1)) * weight
    metrics = {
        "rho_temporal_rel_l2": relative_l2(rho - rho[:1], rho),
        "f_temporal_rel_l2": relative_l2(f - f[:1], f),
        "mean_abs_u_vx": float(np.mean(np.abs(ux))),
        "mean_abs_u_vy": float(np.mean(np.abs(uy))),
        "rho_std": float(np.std(rho)),
        "f_rms": float(np.sqrt(np.mean(f**2))),
    }
    return {str(key): float(metrics[str(key)]) for key in keys}


def condition_vector_from_metrics(metrics: dict[str, Any], keys: Sequence[str] = DEFAULT_SHAPE_CONDITION_KEYS) -> list[float]:
    return [float(metrics[str(key)]) for key in keys]


def fit_condition_normalization(vectors: Iterable[Sequence[float]], *, min_std: float = 1.0e-6) -> dict[str, Any]:
    arr = np.asarray(list(vectors), dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"condition vectors must have shape [N,D], got {arr.shape}")
    mean = np.mean(arr, axis=0)
    std = np.maximum(np.std(arr, axis=0), float(min_std))
    return {
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "min_std": float(min_std),
    }


def normalize_condition_vector(vector: Sequence[float], normalization: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(vector, dtype=float)
    mean = np.asarray(normalization.get("mean"), dtype=float)
    std = np.maximum(np.asarray(normalization.get("std"), dtype=float), EPS)
    if arr.shape != mean.shape or arr.shape != std.shape:
        raise ValueError(f"condition vector shape {arr.shape} does not match normalization shape {mean.shape}")
    return ((arr - mean) / std).astype(np.float32)


def reorder_condition_vector(row: dict[str, Any], keys: Sequence[str]) -> list[float]:
    row_keys = tuple(str(key) for key in row.get("condition_keys", keys))
    vector = list(row.get("condition_vector", []))
    if len(row_keys) != len(vector):
        raise ValueError(f"condition row {row.get('case_id')} has inconsistent keys/vector lengths")
    values = {key: float(value) for key, value in zip(row_keys, vector)}
    missing = [str(key) for key in keys if str(key) not in values]
    if missing:
        raise ValueError(f"condition row {row.get('case_id')} is missing keys {missing}")
    return [values[str(key)] for key in keys]
