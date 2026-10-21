from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

import numpy as np


Array = np.ndarray


def rho_fourier_projection_diagnostic(
    rho: Array,
    x: Array,
    y: Array,
    *,
    background_density: float,
    max_mode: int,
    rho_contrast: float,
) -> dict[str, Any]:
    """Measure the best discrete rho fit before neural-network optimization.

    The design matrix matches the continuous decoder's real Fourier convention.
    The radially scaled fit is feasible under its sufficient L1 positivity bound;
    it is not claimed to be the optimal L1-constrained projection.
    """

    values = np.asarray(rho, dtype=np.float64)
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    background = float(background_density)
    contrast = float(rho_contrast)
    cutoff = int(max_mode)
    if values.ndim != 3:
        raise ValueError("rho must have shape [nt,nx,ny]")
    if values.shape[1:] != (x_values.size, y_values.size):
        raise ValueError("rho spatial shape must match x and y")
    if min(x_values.size, y_values.size) < 2:
        raise ValueError("x and y must contain at least two points")
    if cutoff < 1:
        raise ValueError("max_mode must be positive")
    if background <= 0.0:
        raise ValueError("background_density must be positive")
    if not 0.0 < contrast < 1.0:
        raise ValueError("rho_contrast must lie in (0,1)")
    if not np.all(np.isfinite(values)):
        raise ValueError("rho must be finite")

    design, pseudoinverse, rank, singular_values = _fourier_projection_operator(
        tuple(float(value) for value in x_values),
        tuple(float(value) for value in y_values),
        cutoff,
    )
    target = values.reshape(values.shape[0], -1) - background
    coefficients = pseudoinverse @ target.T
    unconstrained = background + (design @ coefficients).T
    coefficient_l1 = np.sum(np.abs(coefficients), axis=0)
    budget = contrast * background
    radial_scale = np.minimum(
        1.0,
        budget / np.maximum(coefficient_l1, np.finfo(np.float64).tiny),
    )
    feasible_coefficients = coefficients * radial_scale[None, :]
    radial_feasible = background + (design @ feasible_coefficients).T
    flattened = values.reshape(values.shape[0], -1)
    denominator = max(float(np.sum(flattened**2)), 1.0e-30)

    def relative_l2(prediction: Array) -> float:
        return float(
            np.sqrt(np.sum((prediction - flattened) ** 2) / denominator)
        )

    spatial_mean = np.mean(flattened, axis=1)
    mean_error = np.broadcast_to(
        (spatial_mean - background)[:, None], flattened.shape
    )
    smallest_singular = (
        float(singular_values[-1]) if singular_values.size else 0.0
    )
    condition_number = (
        float(singular_values[0] / smallest_singular)
        if smallest_singular > 0.0
        else float("inf")
    )
    return {
        "schema": "vpfp_rho_fourier_projection_diagnostic_v1",
        "max_mode": cutoff,
        "time_count": int(values.shape[0]),
        "spatial_shape": [int(values.shape[1]), int(values.shape[2])],
        "mode_count": len(_real_fourier_half_modes(cutoff)),
        "design_column_count": int(design.shape[1]),
        "design_rank": int(rank),
        "design_full_column_rank": bool(rank == design.shape[1]),
        "design_condition_number": condition_number,
        "background_density": background,
        "rho_contrast": contrast,
        "coefficient_l1_budget": budget,
        "coefficient_l1": _summary(coefficient_l1),
        "radial_scale": _summary(radial_scale),
        "l1_budget_satisfied_fraction": float(
            np.mean(coefficient_l1 <= budget * (1.0 + 1.0e-10))
        ),
        "fixed_mean_rel_l2": float(
            np.sqrt(np.sum(mean_error**2) / denominator)
        ),
        "unconstrained_projection_rel_l2": relative_l2(unconstrained),
        "radial_l1_feasible_rel_l2": relative_l2(radial_feasible),
        "radial_l1_feasible_min_rho": float(np.min(radial_feasible)),
    }


def summarize_rho_fourier_diagnostics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one diagnostic row is required")
    keys = (
        "fixed_mean_rel_l2",
        "unconstrained_projection_rel_l2",
        "radial_l1_feasible_rel_l2",
        "l1_budget_satisfied_fraction",
    )
    return {
        key: _summary(np.asarray([float(row[key]) for row in rows]))
        for key in keys
    }


@lru_cache(maxsize=32)
def _fourier_projection_operator(
    x_values: tuple[float, ...],
    y_values: tuple[float, ...],
    max_mode: int,
):
    x_axis = np.asarray(x_values, dtype=np.float64)
    y_axis = np.asarray(y_values, dtype=np.float64)
    lx = _periodic_length(x_axis)
    ly = _periodic_length(y_axis)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="ij")
    columns = []
    for kx, ky in _real_fourier_half_modes(max_mode):
        phase = 2.0 * np.pi * (
            float(kx) * x_grid / lx + float(ky) * y_grid / ly
        )
        columns.extend(
            (np.cos(phase).reshape(-1), np.sin(phase).reshape(-1))
        )
    design = np.stack(columns, axis=-1)
    left, singular_values, right = np.linalg.svd(
        design, full_matrices=False
    )
    tolerance = (
        max(design.shape)
        * np.finfo(np.float64).eps
        * float(singular_values[0])
    )
    retained = singular_values > tolerance
    pseudoinverse = (
        right[retained].T / singular_values[retained]
    ) @ left[:, retained].T
    return (
        design,
        pseudoinverse,
        int(np.sum(retained)),
        singular_values,
    )




def hermite_coefficient_field_diagnostic(
    coefficients: Array,
    *,
    spatial_cutoffs: Sequence[int] = (2, 4, 6),
    temporal_orders: Sequence[int] = (2, 4, 6, 8),
) -> dict[str, Any]:
    """Summarize smoothness and spectral support of one Hermite field."""

    from scipy.fft import dct

    values = np.asarray(coefficients, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError(
            "coefficients must have shape [nt,nx,ny,hermite_modes]"
        )
    if min(values.shape) < 1 or not np.all(np.isfinite(values)):
        raise ValueError("coefficient fields must be finite and nonempty")
    cutoffs = tuple(sorted({int(value) for value in spatial_cutoffs}))
    orders = tuple(sorted({int(value) for value in temporal_orders}))
    if not cutoffs or min(cutoffs) < 0:
        raise ValueError("spatial cutoffs must be nonnegative")
    if not orders or min(orders) < 0:
        raise ValueError("temporal orders must be nonnegative")

    norm = np.linalg.norm(values, axis=-1)
    total_energy = float(np.sum(values**2))
    if total_energy <= np.finfo(np.float64).tiny:
        raise ValueError("coefficient field has zero energy")

    spatial_spectrum = np.fft.fftn(
        values,
        axes=(1, 2),
        norm="ortho",
    )
    spatial_energy = np.abs(spatial_spectrum) ** 2
    kx = np.rint(np.fft.fftfreq(values.shape[1]) * values.shape[1])
    ky = np.rint(np.fft.fftfreq(values.shape[2]) * values.shape[2])
    spatial_retained = {}
    for cutoff in cutoffs:
        mask = (
            (np.abs(kx)[:, None] <= cutoff)
            & (np.abs(ky)[None, :] <= cutoff)
        )
        spatial_retained[str(cutoff)] = float(
            np.sum(spatial_energy[:, mask, :]) / total_energy
        )

    temporal_spectrum = dct(values, type=2, axis=0, norm="ortho")
    temporal_energy = temporal_spectrum**2
    temporal_retained = {
        str(order): float(
            np.sum(temporal_energy[: min(order + 1, values.shape[0])])
            / total_energy
        )
        for order in orders
    }

    temporal_distance = np.linalg.norm(
        np.diff(values, axis=0),
        axis=-1,
    )
    spatial_x_distance = np.linalg.norm(
        values - np.roll(values, 1, axis=1),
        axis=-1,
    )
    spatial_y_distance = np.linalg.norm(
        values - np.roll(values, 1, axis=2),
        axis=-1,
    )
    return {
        "schema": "vpfp_hermite_coefficient_field_diagnostic_v1",
        "shape": [int(value) for value in values.shape],
        "coefficient_norm_abs_error": _summary(np.abs(norm - 1.0)),
        "zeroth_coefficient": {
            **_summary(values[..., 0]),
            "min": float(np.min(values[..., 0])),
        },
        "spatial_retained_energy_fraction": spatial_retained,
        "temporal_dct_retained_energy_fraction": temporal_retained,
        "neighbor_coefficient_l2": {
            "time": _summary(temporal_distance),
            "x_periodic": _summary(spatial_x_distance),
            "y_periodic": _summary(spatial_y_distance),
        },
    }


def hermite_coefficient_corpus_rank_diagnostic(
    coefficient_fields: Sequence[Array],
    *,
    max_slice_samples: int = 32768,
    seed: int = 20260713,
) -> dict[str, Any]:
    """Estimate case-level and local coefficient-subspace dimensions."""

    fields = [
        np.asarray(values, dtype=np.float32)
        for values in coefficient_fields
    ]
    if not fields:
        raise ValueError("at least one coefficient field is required")
    shape = fields[0].shape
    if len(shape) != 4:
        raise ValueError("coefficient fields must be four-dimensional")
    if any(values.shape != shape for values in fields):
        raise ValueError("all coefficient fields must have the same shape")
    if any(not np.all(np.isfinite(values)) for values in fields):
        raise ValueError("coefficient fields must be finite")
    if int(max_slice_samples) < 1:
        raise ValueError("max_slice_samples must be positive")

    case_matrix = np.stack(
        [values.reshape(-1) for values in fields],
        axis=0,
    )
    case_matrix -= np.mean(case_matrix, axis=0, keepdims=True)
    case_gram = (
        case_matrix @ case_matrix.T
        / max(int(case_matrix.shape[1]), 1)
    )
    case_eigenvalues = np.linalg.eigvalsh(
        np.asarray(case_gram, dtype=np.float64)
    )[::-1]

    rng = np.random.default_rng(int(seed))
    per_case = max(
        1,
        int(np.ceil(int(max_slice_samples) / len(fields))),
    )
    sampled = []
    for values in fields:
        rows = values.reshape(-1, values.shape[-1])
        count = min(per_case, rows.shape[0])
        if count == rows.shape[0]:
            selected = np.arange(count)
        else:
            selected = rng.choice(rows.shape[0], size=count, replace=False)
        sampled.append(rows[selected])
    slice_matrix = np.concatenate(sampled, axis=0)[: int(max_slice_samples)]
    slice_matrix = np.asarray(slice_matrix, dtype=np.float64)
    slice_matrix -= np.mean(slice_matrix, axis=0, keepdims=True)
    slice_covariance = (
        slice_matrix.T @ slice_matrix
        / max(int(slice_matrix.shape[0]), 1)
    )
    slice_eigenvalues = np.linalg.eigvalsh(slice_covariance)[::-1]

    return {
        "schema": "vpfp_hermite_coefficient_corpus_rank_v1",
        "case_count": len(fields),
        "field_shape": [int(value) for value in shape],
        "sampled_slice_count": int(slice_matrix.shape[0]),
        "case_field_pca": _eigenvalue_rank_summary(case_eigenvalues),
        "slice_coefficient_pca": _eigenvalue_rank_summary(
            slice_eigenvalues
        ),
    }


def summarize_hermite_coefficient_field_diagnostics(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one coefficient diagnostic is required")
    spatial_keys = tuple(rows[0]["spatial_retained_energy_fraction"])
    temporal_keys = tuple(rows[0]["temporal_dct_retained_energy_fraction"])
    return {
        "case_count": len(rows),
        "coefficient_norm_abs_error_max": float(
            max(
                row["coefficient_norm_abs_error"]["max"]
                for row in rows
            )
        ),
        "zeroth_coefficient_min": float(
            min(row["zeroth_coefficient"]["min"] for row in rows)
        ),
        "spatial_retained_energy_fraction": {
            key: _summary(
                np.asarray(
                    [
                        row["spatial_retained_energy_fraction"][key]
                        for row in rows
                    ]
                )
            )
            for key in spatial_keys
        },
        "temporal_dct_retained_energy_fraction": {
            key: _summary(
                np.asarray(
                    [
                        row["temporal_dct_retained_energy_fraction"][key]
                        for row in rows
                    ]
                )
            )
            for key in temporal_keys
        },
        "neighbor_coefficient_l2": {
            axis: {
                statistic: _summary(
                    np.asarray(
                        [
                            row["neighbor_coefficient_l2"][axis][statistic]
                            for row in rows
                        ]
                    )
                )
                for statistic in ("mean", "p90", "max")
            }
            for axis in ("time", "x_periodic", "y_periodic")
        },
    }


def _eigenvalue_rank_summary(eigenvalues: Array) -> dict[str, Any]:
    spectrum = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
    total = float(np.sum(spectrum))
    if spectrum.ndim != 1 or spectrum.size < 1 or total <= 1.0e-30:
        return {
            "ambient_rank": int(spectrum.size),
            "numerical_rank": 0,
            "effective_rank": 0.0,
            "stable_rank": 0.0,
            "components_for_fraction": {
                "0.90": 0,
                "0.95": 0,
                "0.99": 0,
            },
            "leading_energy_fraction": {},
        }
    probability = spectrum / total
    positive = probability > 0.0
    cumulative = np.cumsum(probability)
    tolerance = max(
        float(spectrum[0]) * max(spectrum.size, 1) * np.finfo(float).eps,
        1.0e-30,
    )
    return {
        "ambient_rank": int(spectrum.size),
        "numerical_rank": int(np.sum(spectrum > tolerance)),
        "effective_rank": float(
            np.exp(-np.sum(probability[positive] * np.log(probability[positive])))
        ),
        "stable_rank": float(total / max(float(spectrum[0]), 1.0e-30)),
        "components_for_fraction": {
            f"{fraction:.2f}": int(
                np.searchsorted(cumulative, fraction, side="left") + 1
            )
            for fraction in (0.90, 0.95, 0.99)
        },
        "leading_energy_fraction": {
            str(count): float(cumulative[min(count, spectrum.size) - 1])
            for count in (1, 2, 4, 8, 16, 32, 64, 128)
            if count <= spectrum.size
        },
    }

def _periodic_length(axis: Array) -> float:
    spacing = np.diff(axis)
    if not np.all(spacing > 0.0):
        raise ValueError("periodic axes must be strictly increasing")
    if not np.allclose(spacing, spacing[0], rtol=2.0e-6, atol=2.0e-7):
        raise ValueError("periodic axes must be uniformly spaced")
    return float(axis[-1] - axis[0] + spacing[0])


def _real_fourier_half_modes(max_mode: int) -> list[tuple[int, int]]:
    return [
        (kx, ky)
        for kx in range(-int(max_mode), int(max_mode) + 1)
        for ky in range(-int(max_mode), int(max_mode) + 1)
        if kx > 0 or (kx == 0 and ky > 0)
    ]


def _summary(values: Array) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("diagnostic summaries require nonempty finite values")
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }
