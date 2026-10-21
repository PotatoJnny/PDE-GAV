from __future__ import annotations

from typing import Any

import numpy as np

from agent_verifier.vpfp.online_training import (
    fit_translation_invariant_structured_normalization,
)


CALIBRATION_SCHEMA = "vpfp_structured_equivariant_affine_calibration_v1"


def fit_equivariant_affine_calibration(
    reference,
    candidate,
    config,
    *,
    min_scale: float = 0.5,
    max_scale: float = 2.5,
    min_std: float = 1.0e-4,
) -> dict[str, Any]:
    """Fit a diagonal affine map that commutes with torus translations."""

    reference_array = _matrix(reference, config, name="reference")
    candidate_array = _matrix(candidate, config, name="candidate")
    if min(reference_array.shape[0], candidate_array.shape[0]) < 2:
        raise ValueError("calibration requires at least two rows per split")
    if (
        not np.isfinite(float(min_scale))
        or not np.isfinite(float(max_scale))
        or not 0.0 < float(min_scale) <= float(max_scale)
    ):
        raise ValueError("calibration scale bounds are invalid")
    if not np.isfinite(float(min_std)) or float(min_std) <= 0.0:
        raise ValueError("calibration min_std must be positive")

    reference_norm = fit_translation_invariant_structured_normalization(
        reference_array,
        config,
        min_std=float(min_std),
    )
    candidate_norm = fit_translation_invariant_structured_normalization(
        candidate_array,
        config,
        min_std=float(min_std),
    )
    reference_mean = np.asarray(reference_norm["mean"], dtype=np.float64)
    reference_std = np.asarray(reference_norm["std"], dtype=np.float64)
    candidate_mean = np.asarray(candidate_norm["mean"], dtype=np.float64)
    candidate_std = np.asarray(candidate_norm["std"], dtype=np.float64)
    raw_scale = reference_std / candidate_std
    scale = np.clip(raw_scale, float(min_scale), float(max_scale))
    shift = reference_mean - scale * candidate_mean
    _validate_group_contract(scale, shift, config)
    return {
        "schema": CALIBRATION_SCHEMA,
        "group_action": "joint_torus_translation_phase_rotation",
        "reference_count": int(reference_array.shape[0]),
        "candidate_count": int(candidate_array.shape[0]),
        "min_scale": float(min_scale),
        "max_scale": float(max_scale),
        "min_std": float(min_std),
        "scale": scale,
        "shift": shift,
        "raw_scale": raw_scale,
        "clipped_low_count": int(np.count_nonzero(raw_scale < float(min_scale))),
        "clipped_high_count": int(np.count_nonzero(raw_scale > float(max_scale))),
    }


def apply_equivariant_affine_calibration(values, calibration, config):
    array = _matrix(values, config, name="values")
    if calibration.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("unexpected structured calibration schema")
    scale = np.asarray(calibration.get("scale"), dtype=np.float64)
    shift = np.asarray(calibration.get("shift"), dtype=np.float64)
    expected = (int(config.active_latent_dim),)
    if scale.shape != expected or shift.shape != expected:
        raise ValueError("structured calibration shape mismatch")
    if not np.all(np.isfinite(scale)) or not np.all(np.isfinite(shift)):
        raise ValueError("structured calibration contains non-finite values")
    _validate_group_contract(scale, shift, config)
    calibrated = array * scale[None, :] + shift[None, :]
    if not np.all(np.isfinite(calibrated)):
        raise FloatingPointError("structured calibration output is non-finite")
    return calibrated


def calibration_scale_summary(calibration) -> dict[str, float | int]:
    raw = np.asarray(calibration["raw_scale"], dtype=np.float64)
    scale = np.asarray(calibration["scale"], dtype=np.float64)
    return {
        "raw_min": float(np.min(raw)),
        "raw_p05": float(np.quantile(raw, 0.05)),
        "raw_median": float(np.median(raw)),
        "raw_p95": float(np.quantile(raw, 0.95)),
        "raw_max": float(np.max(raw)),
        "applied_min": float(np.min(scale)),
        "applied_median": float(np.median(scale)),
        "applied_max": float(np.max(scale)),
        "clipped_low_count": int(calibration["clipped_low_count"]),
        "clipped_high_count": int(calibration["clipped_high_count"]),
    }


def _matrix(values, config, *, name: str) -> np.ndarray:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    result = np.asarray(array, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != int(config.active_latent_dim):
        raise ValueError(f"{name} does not match the structured latent layout")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _validate_group_contract(scale, shift, config) -> None:
    rho_scale = np.asarray(scale[: config.rho_latent_dim]).reshape(
        config.rho_time_count,
        config.rho_mode_count,
        2,
    )
    rho_shift = np.asarray(shift[: config.rho_latent_dim]).reshape(
        config.rho_time_count,
        config.rho_mode_count,
        2,
    )
    if not np.array_equal(rho_scale[..., 0], rho_scale[..., 1]):
        raise ValueError("rho calibration scales must be tied by Fourier pair")
    if np.any(rho_shift != 0.0):
        raise ValueError("rho calibration shifts must vanish")

    local_scale = np.asarray(scale[config.rho_latent_dim :]).reshape(
        config.local_channels,
        config.local_time_count,
        config.local_spatial_feature_count,
    )
    local_shift = np.asarray(shift[config.rho_latent_dim :]).reshape(
        config.local_channels,
        config.local_time_count,
        config.local_spatial_feature_count,
    )
    paired_scale = local_scale[..., 1:].reshape(
        config.local_channels,
        config.local_time_count,
        -1,
        2,
    )
    paired_shift = local_shift[..., 1:].reshape(
        config.local_channels,
        config.local_time_count,
        -1,
        2,
    )
    if not np.array_equal(paired_scale[..., 0], paired_scale[..., 1]):
        raise ValueError("local calibration scales must be tied by Fourier pair")
    if np.any(paired_shift != 0.0):
        raise ValueError("nonconstant local calibration shifts must vanish")
