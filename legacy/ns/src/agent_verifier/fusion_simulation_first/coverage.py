"""Coverage metrics and compact physics features for generated windows."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


FEATURE_NAMES = (
    "log_h_scale",
    "log_phi_scale",
    "h_plane_log_rms",
    "h_plane_peak_to_rms",
    "h_zonal_fraction",
    "h_high_k_fraction",
    "h_spectral_entropy",
    "phi_plane_log_rms",
    "phi_plane_peak_to_rms",
    "phi_zonal_fraction",
    "phi_high_k_fraction",
    "phi_spectral_entropy",
    "vpar_tail_fraction",
    "mu_tail_fraction",
    "velocity_roughness",
)


def _squared_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("distance inputs must be [samples, features] with matching width")
    distances = (
        np.sum(first * first, axis=1)[:, None]
        + np.sum(second * second, axis=1)[None, :]
        - 2.0 * first @ second.T
    )
    return np.maximum(distances, 0.0)


def _neighbor_radii(values: np.ndarray, neighbors: int) -> np.ndarray:
    if values.shape[0] < 2:
        return np.full((values.shape[0],), np.inf, dtype=np.float64)
    k = min(max(neighbors, 1), values.shape[0] - 1)
    distances = _squared_distances(values, values)
    np.fill_diagonal(distances, np.inf)
    return np.partition(distances, k - 1, axis=1)[:, k - 1]


def prdc_metrics(real: np.ndarray, generated: np.ndarray, *, neighbors: int = 5) -> dict[str, float]:
    real = np.asarray(real, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if real.shape[0] == 0 or generated.shape[0] == 0:
        raise ValueError("PRDC inputs must be nonempty")
    real_radii = _neighbor_radii(real, neighbors)
    generated_radii = _neighbor_radii(generated, neighbors)
    distances = _squared_distances(generated, real)
    precision = np.mean(np.any(distances <= real_radii[None, :], axis=1))
    recall = np.mean(np.any(distances <= generated_radii[:, None], axis=0))
    k = min(max(neighbors, 1), max(real.shape[0] - 1, 1))
    density = np.mean(np.sum(distances <= real_radii[None, :], axis=1) / k)
    coverage = np.mean(np.min(distances, axis=0) <= real_radii)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "density": float(density),
        "coverage": float(coverage),
    }


def rbf_mmd(real: np.ndarray, generated: np.ndarray) -> float:
    real = np.asarray(real, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    combined = np.concatenate((real, generated), axis=0)
    distances = _squared_distances(combined, combined)
    positive = distances[distances > 0.0]
    bandwidth = float(np.median(positive)) if positive.size else 1.0
    bandwidth = max(bandwidth, 1.0e-12)

    def kernel(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return np.exp(-_squared_distances(first, second) / (2.0 * bandwidth))

    k_rr = kernel(real, real)
    k_gg = kernel(generated, generated)
    k_rg = kernel(real, generated)
    if real.shape[0] > 1:
        rr = (np.sum(k_rr) - np.trace(k_rr)) / (real.shape[0] * (real.shape[0] - 1))
    else:
        rr = 1.0
    if generated.shape[0] > 1:
        gg = (np.sum(k_gg) - np.trace(k_gg)) / (
            generated.shape[0] * (generated.shape[0] - 1)
        )
    else:
        gg = 1.0
    return float(max(rr + gg - 2.0 * np.mean(k_rg), 0.0))


def effective_rank(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] < 2:
        return 1.0
    singular = np.linalg.svd(values - np.mean(values, axis=0), compute_uv=False)
    energy = singular * singular
    if not np.any(energy > 0.0):
        return 1.0
    probabilities = energy / np.sum(energy)
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1.0e-30)))
    return float(np.exp(entropy))


def standardize_features(reference: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    reference = np.asarray(reference, dtype=np.float64)
    mean = np.mean(reference, axis=0)
    scale = np.std(reference, axis=0)
    scale = np.where(scale < 1.0e-6, 1.0, scale)
    return tuple((np.asarray(values, dtype=np.float64) - mean) / scale for values in (reference, *others))


def reference_pca_projection(
    reference: np.ndarray,
    *others: np.ndarray,
    max_components: int = 16,
    variance_fraction: float = 0.95,
) -> tuple[tuple[np.ndarray, ...], int, float]:
    """Project all splits through a PCA fit only on the reference split."""

    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[0] < 2:
        raise ValueError("PCA reference must contain at least two samples")
    if max_components <= 0 or not 0.0 < variance_fraction <= 1.0:
        raise ValueError("invalid PCA component or variance limit")
    arrays = (reference, *(np.asarray(values, dtype=np.float64) for values in others))
    if any(values.ndim != 2 or values.shape[1] != reference.shape[1] for values in arrays):
        raise ValueError("all PCA inputs must have matching feature widths")

    mean = np.mean(reference, axis=0, keepdims=True)
    _left, singular, right = np.linalg.svd(reference - mean, full_matrices=False)
    energy = np.square(singular)
    total_energy = float(np.sum(energy))
    if total_energy <= 0.0:
        component_count = 1
        retained_fraction = 1.0
    else:
        cumulative = np.cumsum(energy) / total_energy
        component_count = int(np.searchsorted(cumulative, variance_fraction) + 1)
        component_count = min(component_count, max_components, right.shape[0])
        retained_fraction = float(cumulative[component_count - 1])
    components = right[:component_count]
    projected = tuple((values - mean) @ components.T for values in arrays)
    return projected, component_count, retained_fraction


def sliced_wasserstein(
    first: np.ndarray,
    second: np.ndarray,
    *,
    projections: int = 128,
    quantiles: int = 256,
    seed: int = 20260715,
) -> float:
    """Deterministic sliced 1-Wasserstein distance in standardized feature space."""

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if (
        first.ndim != 2
        or second.ndim != 2
        or first.shape[1] != second.shape[1]
        or first.shape[0] == 0
        or second.shape[0] == 0
    ):
        raise ValueError("sliced Wasserstein inputs must be nonempty and aligned")
    if projections <= 0 or quantiles <= 1:
        raise ValueError("projection and quantile counts must be positive")

    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(projections, first.shape[1]))
    directions /= np.maximum(
        np.linalg.norm(directions, axis=1, keepdims=True),
        1.0e-12,
    )
    probabilities = np.linspace(0.0, 1.0, quantiles)
    first_quantiles = np.quantile(first @ directions.T, probabilities, axis=0)
    second_quantiles = np.quantile(second @ directions.T, probabilities, axis=0)
    return float(np.mean(np.abs(first_quantiles - second_quantiles)))


def central_interval_coverage(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    """Measure whether candidate central-90% intervals cover reference central-80%."""

    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if (
        reference.ndim != 2
        or candidate.ndim != 2
        or reference.shape[1] != candidate.shape[1]
        or reference.shape[0] == 0
        or candidate.shape[0] == 0
    ):
        raise ValueError("interval coverage inputs must be nonempty and aligned")
    reference_low, reference_high = np.quantile(
        reference, (0.10, 0.90), axis=0
    )
    candidate_low, candidate_high = np.quantile(
        candidate, (0.05, 0.95), axis=0
    )
    covered = (candidate_low <= reference_low) & (
        candidate_high >= reference_high
    )
    reference_width = np.maximum(reference_high - reference_low, 1.0e-12)
    candidate_width = candidate_high - candidate_low
    width_ratio = candidate_width / reference_width
    return {
        "covered_feature_fraction": float(np.mean(covered)),
        "covered_feature_count": int(np.count_nonzero(covered)),
        "feature_count": int(covered.size),
        "covered_mask": covered.tolist(),
        "width_ratio_by_feature": width_ratio.tolist(),
        "lower_margin_by_feature": (
            (reference_low - candidate_low) / reference_width
        ).tolist(),
        "upper_margin_by_feature": (
            (candidate_high - reference_high) / reference_width
        ).tolist(),
        "median_width_ratio": float(np.median(width_ratio)),
    }


def compare_distributions(
    train: np.ndarray,
    holdout: np.ndarray,
    generated: np.ndarray,
    *,
    neighbors: int = 5,
    baseline_seed: int = 20260715,
) -> dict[str, Any]:
    train_z, holdout_z, generated_z = standardize_features(
        train, holdout, generated
    )
    (
        train_pca,
        holdout_pca,
        generated_pca,
    ), pca_components, pca_retained_fraction = reference_pca_projection(
        train_z,
        holdout_z,
        generated_z,
    )
    baseline_count = min(train_z.shape[0], generated_z.shape[0])
    if baseline_count <= 0:
        raise ValueError("empirical candidate baseline must be nonempty")
    if baseline_count == train_z.shape[0]:
        baseline_indices = np.arange(baseline_count, dtype=np.int64)
    else:
        baseline_rng = np.random.default_rng(baseline_seed)
        baseline_indices = np.sort(
            baseline_rng.choice(
                train_z.shape[0],
                size=baseline_count,
                replace=False,
            )
        )
    empirical_candidate_z = train_z[baseline_indices]
    empirical_candidate_pca = train_pca[baseline_indices]
    generated_holdout_swd = sliced_wasserstein(
        holdout_z, generated_z
    )
    train_holdout_swd = sliced_wasserstein(train_z, holdout_z)
    return {
        "empirical_candidate_baseline": {
            "seed": int(baseline_seed),
            "sample_count": int(baseline_count),
            "train_sample_count": int(train_z.shape[0]),
            "generated_sample_count": int(generated_z.shape[0]),
            "reference": "holdout",
            "candidate": "deterministic_train_subsample",
        },
        "generated_vs_train": prdc_metrics(
            train_z, generated_z, neighbors=neighbors
        ),
        "generated_vs_holdout": prdc_metrics(
            holdout_z, generated_z, neighbors=neighbors
        ),
        "empirical_candidate_vs_holdout": prdc_metrics(
            holdout_z, empirical_candidate_z, neighbors=neighbors
        ),
        "train_vs_holdout": prdc_metrics(
            train_z, holdout_z, neighbors=neighbors
        ),
        "pca_generated_vs_train": prdc_metrics(
            train_pca, generated_pca, neighbors=neighbors
        ),
        "pca_generated_vs_holdout": prdc_metrics(
            holdout_pca, generated_pca, neighbors=neighbors
        ),
        "pca_empirical_candidate_vs_holdout": prdc_metrics(
            holdout_pca, empirical_candidate_pca, neighbors=neighbors
        ),
        "pca_train_vs_holdout": prdc_metrics(
            train_pca, holdout_pca, neighbors=neighbors
        ),
        "pca_components": pca_components,
        "pca_retained_variance_fraction": pca_retained_fraction,
        "mmd_generated_train": rbf_mmd(train_z, generated_z),
        "mmd_generated_holdout": rbf_mmd(holdout_z, generated_z),
        "sliced_wasserstein_generated_train": sliced_wasserstein(
            train_z, generated_z
        ),
        "sliced_wasserstein_generated_holdout": generated_holdout_swd,
        "sliced_wasserstein_empirical_candidate_holdout": sliced_wasserstein(
            holdout_z, empirical_candidate_z
        ),
        "sliced_wasserstein_train_holdout": train_holdout_swd,
        "sliced_wasserstein_holdout_ratio": (
            generated_holdout_swd / max(train_holdout_swd, 1.0e-12)
        ),
        "generated_interval_coverage_of_train": central_interval_coverage(
            train_z, generated_z
        ),
        "generated_interval_coverage_of_holdout": central_interval_coverage(
            holdout_z, generated_z
        ),
        "empirical_candidate_interval_coverage_of_holdout": (
            central_interval_coverage(holdout_z, empirical_candidate_z)
        ),
        "effective_rank_train": effective_rank(train_z),
        "effective_rank_holdout": effective_rank(holdout_z),
        "effective_rank_generated": effective_rank(generated_z),
    }


def _plane_features(values: np.ndarray) -> tuple[float, float, float, float, float]:
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[-1] == 2:
        field = values[..., 0] + 1j * values[..., 1]
    elif values.ndim == 2:
        field = values.astype(np.complex128)
    else:
        raise ValueError("plane must be [nx, ny] or [nx, ny, 2]")
    magnitude_squared = np.abs(field) ** 2
    rms = math.sqrt(float(np.mean(magnitude_squared)) + 1.0e-30)
    peak_ratio = float(np.max(np.abs(field)) / max(rms, 1.0e-15))
    power = np.abs(np.fft.fft2(field)) ** 2
    total = float(np.sum(power)) + 1.0e-30
    zonal = float(np.sum(power[:, 0]) / total)
    kx = np.fft.fftfreq(field.shape[0])[:, None]
    ky = np.fft.fftfreq(field.shape[1])[None, :]
    high = np.sqrt(kx * kx + ky * ky) >= 0.25
    high_fraction = float(np.sum(power[high]) / total)
    probabilities = power.ravel() / total
    entropy = -float(
        np.sum(probabilities * np.log(np.maximum(probabilities, 1.0e-30)))
    ) / math.log(max(probabilities.size, 2))
    return math.log(max(rms, 1.0e-15)), peak_ratio, zonal, high_fraction, entropy


def physics_feature_vector(
    h_plane: np.ndarray,
    h_velocity: np.ndarray,
    phi_plane: np.ndarray,
    *,
    h_scale: float,
    phi_scale: float,
) -> np.ndarray:
    h_features = _plane_features(h_plane)
    phi_features = _plane_features(phi_plane)
    velocity = np.asarray(h_velocity)
    if velocity.ndim != 3 or velocity.shape[-1] != 2:
        raise ValueError("velocity column must have shape [nvpar, nmu, 2]")
    energy = np.sum(np.square(velocity, dtype=np.float64), axis=-1)
    total = float(np.sum(energy)) + 1.0e-30
    v_cut = max(1, velocity.shape[0] // 4)
    v_tail = float((np.sum(energy[:v_cut]) + np.sum(energy[-v_cut:])) / total)
    mu_cut = max(1, velocity.shape[1] // 4)
    mu_tail = float(np.sum(energy[:, -mu_cut:]) / total)
    roughness = (
        float(np.sum(np.diff(velocity, axis=0) ** 2))
        + float(np.sum(np.diff(velocity, axis=1) ** 2))
    ) / total
    result = np.asarray(
        (
            math.log(max(h_scale, 1.0e-15)),
            math.log(max(phi_scale, 1.0e-15)),
            *h_features,
            *phi_features,
            v_tail,
            mu_tail,
            roughness,
        ),
        dtype=np.float64,
    )
    if result.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(result)):
        raise ValueError("physics feature extraction produced invalid output")
    return result
