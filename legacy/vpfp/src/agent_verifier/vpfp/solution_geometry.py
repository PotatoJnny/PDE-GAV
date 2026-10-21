"""Aggregate diagnostics for generated VPFP answer coordinates."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch


def grouped_solution_geometry(
    blocks: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    """Compare candidate geometry with two disjoint reference draws."""

    required = {"reference_a", "reference_b", "generated"}
    if set(blocks) != required:
        raise ValueError(f"geometry blocks must be exactly {sorted(required)}")
    shapes = {tuple(torch.as_tensor(value).shape) for value in blocks.values()}
    if len(shapes) != 1:
        raise ValueError("all geometry blocks must have the same shape")
    shape = next(iter(shapes))
    if len(shape) < 2 or int(shape[0]) < 2:
        raise ValueError("geometry blocks need at least two cases")

    statistics = {
        name: solution_set_geometry(values) for name, values in blocks.items()
    }
    reference_pairwise = max(
        statistics["reference_a"]["pairwise_distance_median"], 1.0e-12
    )
    reference_rms = max(
        statistics["reference_a"]["centered_rms"], 1.0e-12
    )
    return {
        "sets": statistics,
        "reference_null": {
            "pairwise_distance_ratio": float(
                statistics["reference_b"]["pairwise_distance_median"]
                / reference_pairwise
            ),
            "centered_rms_ratio": float(
                statistics["reference_b"]["centered_rms"] / reference_rms
            ),
            **cross_solution_geometry(
                blocks["reference_a"], blocks["reference_b"]
            ),
        },
        "generated_to_reference": {
            "pairwise_distance_ratio": float(
                statistics["generated"]["pairwise_distance_median"]
                / reference_pairwise
            ),
            "centered_rms_ratio": float(
                statistics["generated"]["centered_rms"] / reference_rms
            ),
            **cross_solution_geometry(
                blocks["reference_a"], blocks["generated"]
            ),
        },
    }


def solution_set_geometry(values: torch.Tensor) -> dict[str, float]:
    flat = torch.as_tensor(values, dtype=torch.float64).reshape(
        int(values.shape[0]), -1
    )
    if not bool(torch.all(torch.isfinite(flat)).item()):
        raise ValueError("solution geometry input is non-finite")
    centered = flat - flat.mean(dim=0, keepdim=True)
    coordinate_std = torch.sqrt(torch.mean(centered.square(), dim=0))
    distances = torch.pdist(flat) / math.sqrt(float(flat.shape[1]))
    gram = centered @ centered.T / float(flat.shape[1])
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    probability = eigenvalues / eigenvalues.sum().clamp_min(1.0e-15)
    positive = probability > 0.0
    effective_rank = torch.exp(
        -torch.sum(probability[positive] * torch.log(probability[positive]))
    )
    return {
        "absolute_mean": float(torch.mean(torch.abs(flat))),
        "rms": float(torch.sqrt(torch.mean(flat.square()))),
        "centered_rms": float(torch.sqrt(torch.mean(centered.square()))),
        "coordinate_std_mean": float(torch.mean(coordinate_std)),
        "coordinate_std_median": float(torch.median(coordinate_std)),
        "coordinate_std_q05": float(torch.quantile(coordinate_std, 0.05)),
        "coordinate_std_q95": float(torch.quantile(coordinate_std, 0.95)),
        "pairwise_distance_median": float(torch.median(distances)),
        "case_covariance_effective_rank": float(effective_rank),
    }


def cross_solution_geometry(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float]:
    reference_flat = torch.as_tensor(reference, dtype=torch.float64).reshape(
        int(reference.shape[0]), -1
    )
    candidate_flat = torch.as_tensor(candidate, dtype=torch.float64).reshape(
        int(candidate.shape[0]), -1
    )
    if reference_flat.shape != candidate_flat.shape:
        raise ValueError("cross-geometry arrays must have the same shape")
    reference_centered = reference_flat - reference_flat.mean(
        dim=0, keepdim=True
    )
    candidate_centered = candidate_flat - candidate_flat.mean(
        dim=0, keepdim=True
    )
    mean_scale = torch.sqrt(torch.mean(reference_centered.square())).clamp_min(
        1.0e-12
    )
    mean_shift = torch.sqrt(
        torch.mean(
            (candidate_flat.mean(0) - reference_flat.mean(0)).square()
        )
    )
    reference_gram = reference_centered @ reference_centered.T
    candidate_gram = candidate_centered @ candidate_centered.T
    reference_spectrum = torch.linalg.eigvalsh(reference_gram).clamp_min(0.0)
    candidate_spectrum = torch.linalg.eigvalsh(candidate_gram).clamp_min(0.0)
    reference_spectrum = reference_spectrum / reference_spectrum.sum().clamp_min(
        1.0e-12
    )
    candidate_spectrum = candidate_spectrum / candidate_spectrum.sum().clamp_min(
        1.0e-12
    )
    return {
        "normalized_mean_shift": float(mean_shift / mean_scale),
        "case_covariance_spectrum_l2_distance": float(
            torch.linalg.vector_norm(candidate_spectrum - reference_spectrum)
        ),
    }


def classify_micro_geometry(
    *,
    normalized_micro_spread_ratio: float,
    micro_pairwise_ratio: float,
    cycle_ratio: float,
) -> dict[str, object]:
    """Produce a diagnostic hypothesis, never an admission decision."""

    values = (
        normalized_micro_spread_ratio,
        micro_pairwise_ratio,
        cycle_ratio,
    )
    if not all(
        math.isfinite(float(value)) and value >= 0.0 for value in values
    ):
        raise ValueError("micro geometry ratios must be finite and nonnegative")
    spread_preserved = bool(
        0.75 <= normalized_micro_spread_ratio <= 1.35
        and 0.75 <= micro_pairwise_ratio <= 1.35
    )
    off_manifold = bool(cycle_ratio >= 2.0)
    if spread_preserved and off_manifold:
        conclusion = (
            "normalized_micro_spread_preserved_but_codes_off_manifold"
        )
        next_model = (
            "invertible_nonlinear_micro_coordinates_then_shared_diffusion"
        )
    elif not spread_preserved:
        conclusion = (
            "diffusion_under_or_over_disperses_normalized_micro_coordinates"
        )
        next_model = (
            "repair_micro_score_training_or_sampler_before_new_coordinates"
        )
    else:
        conclusion = (
            "linear_geometry_inconclusive_without_strong_cycle_failure"
        )
        next_model = (
            "run_targeted_nonlinear_two_sample_and_sampler_ablation"
        )
    return {
        "conclusion": conclusion,
        "normalized_spread_preserved": spread_preserved,
        "micro_code_off_manifold": off_manifold,
        "recommended_next_model": next_model,
        "thresholds_are_diagnostic_not_admission_gates": True,
    }


def reliability_weighted_grouped_answer_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    projection_weights: torch.Tensor,
    *,
    macro_coordinate_count: int,
    macro_fraction: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compare answer coordinates without introducing equation-side targets."""

    predicted_values = torch.as_tensor(predicted)
    target_values = torch.as_tensor(
        target, dtype=predicted_values.dtype, device=predicted_values.device
    )
    weights = torch.as_tensor(
        projection_weights,
        dtype=predicted_values.dtype,
        device=predicted_values.device,
    )
    if (
        predicted_values.ndim != 2
        or target_values.shape != predicted_values.shape
        or weights.shape != (int(predicted_values.shape[1]),)
    ):
        raise ValueError("answer loss tensors have incompatible shapes")
    boundary = int(macro_coordinate_count)
    if not 0 < boundary < int(predicted_values.shape[1]):
        raise ValueError("answer loss needs a strict macro/micro boundary")
    if not 0.0 < float(macro_fraction) < 1.0:
        raise ValueError("answer macro fraction must lie in (0,1)")
    if not bool(
        torch.all(torch.isfinite(predicted_values)).item()
        and torch.all(torch.isfinite(target_values)).item()
        and torch.all(torch.isfinite(weights)).item()
    ):
        raise ValueError("answer loss inputs must be finite")
    if bool(torch.any(weights < 0.0).item()):
        raise ValueError("answer projection weights cannot be negative")

    weighted_error = (predicted_values - target_values).square() * weights.square()
    macro_denominator = weights[:boundary].square().sum().clamp_min(1.0e-12)
    micro_denominator = weights[boundary:].square().sum().clamp_min(1.0e-12)
    macro_loss = weighted_error[:, :boundary].sum(dim=1).mean()
    macro_loss = macro_loss / macro_denominator
    micro_loss = weighted_error[:, boundary:].sum(dim=1).mean()
    micro_loss = micro_loss / micro_denominator
    fraction = float(macro_fraction)
    loss = fraction * macro_loss + (1.0 - fraction) * micro_loss
    return loss, {
        "answer_macro_mse": macro_loss,
        "answer_micro_mse": micro_loss,
    }


def reliability_weighted_grouped_answer_relative_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    projection_weights: torch.Tensor,
    *,
    macro_coordinate_count: int,
    macro_fraction: float,
    denominator_floor_fraction: float = 1.0e-3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Average per-answer relative errors with a low-energy stability floor."""

    predicted_values = torch.as_tensor(predicted)
    target_values = torch.as_tensor(
        target, dtype=predicted_values.dtype, device=predicted_values.device
    )
    weights = torch.as_tensor(
        projection_weights,
        dtype=predicted_values.dtype,
        device=predicted_values.device,
    )
    if (
        predicted_values.ndim != 2
        or target_values.shape != predicted_values.shape
        or weights.shape != (int(predicted_values.shape[1]),)
    ):
        raise ValueError("answer loss tensors have incompatible shapes")
    boundary = int(macro_coordinate_count)
    if not 0 < boundary < int(predicted_values.shape[1]):
        raise ValueError("answer loss needs a strict macro/micro boundary")
    if not 0.0 < float(macro_fraction) < 1.0:
        raise ValueError("answer macro fraction must lie in (0,1)")
    if not 0.0 < float(denominator_floor_fraction) <= 1.0:
        raise ValueError("relative answer denominator floor is invalid")
    if not bool(
        torch.all(torch.isfinite(predicted_values)).item()
        and torch.all(torch.isfinite(target_values)).item()
        and torch.all(torch.isfinite(weights)).item()
    ):
        raise ValueError("answer loss inputs must be finite")
    if bool(torch.any(weights < 0.0).item()):
        raise ValueError("answer projection weights cannot be negative")

    squared_weights = weights.square()
    weighted_error = (predicted_values - target_values).square() * squared_weights
    weighted_target = target_values.square() * squared_weights
    sample_energy_floor = float(denominator_floor_fraction) * (
        weighted_target.detach().sum(dim=1)
    )

    def relative_group(start: int, stop: int) -> torch.Tensor:
        error_energy = weighted_error[:, start:stop].sum(dim=1)
        target_energy = weighted_target[:, start:stop].sum(dim=1)
        denominator = torch.maximum(
            target_energy, sample_energy_floor
        ).clamp_min(
            torch.finfo(target_energy.dtype).eps
        )
        return torch.mean(error_energy / denominator)

    macro_loss = relative_group(0, boundary)
    micro_loss = relative_group(boundary, int(predicted_values.shape[1]))
    fraction = float(macro_fraction)
    loss = fraction * macro_loss + (1.0 - fraction) * micro_loss
    return loss, {
        "answer_macro_relative_mse": macro_loss,
        "answer_micro_relative_mse": micro_loss,
    }
