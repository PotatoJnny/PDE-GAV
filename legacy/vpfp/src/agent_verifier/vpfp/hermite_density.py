from __future__ import annotations

from typing import Any, Sequence

import numpy as np


SQUARED_HERMITE_REPRESENTATION = (
    "conditional_squared_orthonormal_hermite_density_v1"
)


def total_degree_mode_pairs(max_total_degree: int) -> tuple[tuple[int, int], ...]:
    degree = int(max_total_degree)
    if degree < 0:
        raise ValueError("max_total_degree must be nonnegative")
    return tuple(
        (first, total - first)
        for total in range(degree + 1)
        for first in range(total + 1)
    )


def normalized_hermite_functions(
    values: np.ndarray | Sequence[float],
    *,
    max_degree: int,
    scale: float,
) -> np.ndarray:
    """Evaluate orthonormal Hermite functions on R with dilation scale."""

    points = np.asarray(values, dtype=np.float64)
    degree = int(max_degree)
    dilation = float(scale)
    if degree < 0:
        raise ValueError("max_degree must be nonnegative")
    if not np.isfinite(dilation) or dilation <= 0.0:
        raise ValueError("scale must be finite and positive")
    if not np.all(np.isfinite(points)):
        raise ValueError("values must be finite")

    xi = points / dilation
    output = np.empty(points.shape + (degree + 1,), dtype=np.float64)
    output[..., 0] = (
        np.pi ** (-0.25)
        / np.sqrt(dilation)
        * np.exp(-0.5 * xi**2)
    )
    if degree >= 1:
        output[..., 1] = np.sqrt(2.0) * xi * output[..., 0]
    for order in range(1, degree):
        output[..., order + 1] = (
            np.sqrt(2.0 / float(order + 1))
            * xi
            * output[..., order]
            - np.sqrt(float(order) / float(order + 1))
            * output[..., order - 1]
        )
    return output


def tensor_hermite_basis(
    vx: np.ndarray | Sequence[float],
    vy: np.ndarray | Sequence[float],
    *,
    max_total_degree: int,
    scale: float,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    velocity_x = np.asarray(vx, dtype=np.float64)
    velocity_y = np.asarray(vy, dtype=np.float64)
    if velocity_x.ndim != 1 or velocity_y.ndim != 1:
        raise ValueError("vx and vy must be one-dimensional")
    modes = total_degree_mode_pairs(max_total_degree)
    x_functions = normalized_hermite_functions(
        velocity_x,
        max_degree=max_total_degree,
        scale=scale,
    )
    y_functions = normalized_hermite_functions(
        velocity_y,
        max_degree=max_total_degree,
        scale=scale,
    )
    basis = np.stack(
        [
            (
                x_functions[:, first, None]
                * y_functions[None, :, second]
            ).reshape(-1)
            for first, second in modes
        ],
        axis=-1,
    )
    return basis, modes


def squared_hermite_projection_diagnostic(
    density_slices: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    velocity_weights: np.ndarray,
    *,
    max_total_degree: int,
    temperature: float,
    rcond: float = 1.0e-10,
) -> dict[str, Any]:
    """Project sqrt(g) and evaluate the exactly normalized squared expansion."""

    density = np.asarray(density_slices, dtype=np.float64)
    weights = np.asarray(velocity_weights, dtype=np.float64)
    expected_shape = (np.asarray(vx).size, np.asarray(vy).size)
    if density.ndim < 3 or density.shape[-2:] != expected_shape:
        raise ValueError(
            "density_slices must end in the velocity-grid shape"
        )
    if weights.shape != expected_shape:
        raise ValueError("velocity_weights shape does not match velocity grid")
    if not np.all(np.isfinite(density)) or np.min(density) < -1.0e-12:
        raise ValueError("density_slices must be finite and nonnegative")
    if not np.all(np.isfinite(weights)) or np.min(weights) <= 0.0:
        raise ValueError("velocity_weights must be finite and positive")
    theta = float(temperature)
    if not np.isfinite(theta) or theta <= 0.0:
        raise ValueError("temperature must be finite and positive")

    basis, modes = tensor_hermite_basis(
        vx,
        vy,
        max_total_degree=max_total_degree,
        scale=np.sqrt(2.0 * theta),
    )
    flat_weights = weights.reshape(-1)
    sqrt_weights = np.sqrt(flat_weights)
    weighted_design = sqrt_weights[:, None] * basis
    rank = int(np.linalg.matrix_rank(weighted_design))
    singular_values = np.linalg.svd(
        weighted_design, full_matrices=False, compute_uv=False
    )
    pseudo_inverse = np.linalg.pinv(weighted_design, rcond=float(rcond))

    target = np.maximum(density, 0.0).reshape(-1, flat_weights.size)
    weighted_sqrt_target = np.sqrt(target) * sqrt_weights[None, :]
    coefficients = weighted_sqrt_target @ pseudo_inverse.T
    coefficient_norm_squared = np.sum(coefficients**2, axis=-1)
    valid = coefficient_norm_squared > np.finfo(np.float64).tiny
    if not np.all(valid):
        coefficients[~valid] = 0.0
        coefficients[~valid, 0] = 1.0
        coefficient_norm_squared[~valid] = 1.0

    normalized_coefficients = coefficients / np.sqrt(
        coefficient_norm_squared[:, None]
    )
    reconstructed_sqrt = normalized_coefficients @ basis.T
    prediction = reconstructed_sqrt**2
    target_l2 = np.sum(target**2 * flat_weights[None, :], axis=-1)
    density_error = np.sqrt(
        np.sum((prediction - target) ** 2 * flat_weights[None, :], axis=-1)
        / np.maximum(target_l2, np.finfo(np.float64).tiny)
    )
    target_sqrt = np.sqrt(target)
    target_sqrt_l2 = np.sum(
        target_sqrt**2 * flat_weights[None, :], axis=-1
    )
    sqrt_error = np.sqrt(
        np.sum(
            (reconstructed_sqrt - target_sqrt) ** 2
            * flat_weights[None, :],
            axis=-1,
        )
        / np.maximum(target_sqrt_l2, np.finfo(np.float64).tiny)
    )
    window_mass = prediction @ flat_weights
    highest_degree = np.asarray(
        [first + second == int(max_total_degree) for first, second in modes]
    )
    highest_degree_energy = np.sum(
        normalized_coefficients[:, highest_degree] ** 2, axis=-1
    )

    return {
        "basis_count": len(modes),
        "design_rank": rank,
        "design_full_column_rank": bool(rank == len(modes)),
        "design_condition_number": float(
            singular_values[0] / max(singular_values[-1], 1.0e-300)
        ),
        "normalized_coefficients": normalized_coefficients,
        "density_rel_l2": density_error,
        "sqrt_density_rel_l2": sqrt_error,
        "window_mass_abs_error": np.abs(window_mass - 1.0),
        "highest_degree_energy_fraction": highest_degree_energy,
    }


def torch_tensor_hermite_basis(
    torch_module,
    velocity,
    *,
    modes: Sequence[tuple[int, int]],
    scale,
):
    """Differentiable tensor Hermite basis for per-row velocities and scales."""

    torch = torch_module
    values = torch.as_tensor(velocity)
    if values.ndim != 2 or int(values.shape[-1]) != 2:
        raise ValueError("velocity must have shape [N,2]")
    if not modes:
        raise ValueError("modes must be nonempty")
    max_degree = max(max(first, second) for first, second in modes)
    scale_values = torch.as_tensor(
        scale, dtype=values.dtype, device=values.device
    )
    if scale_values.ndim == 0:
        scale_values = scale_values.expand(values.shape[0])
    scale_values = torch.broadcast_to(
        scale_values.reshape(-1), (values.shape[0],)
    )
    if bool(torch.any(scale_values <= 0.0)):
        raise ValueError("scale must be positive")

    def one_dimensional(points):
        xi = points / scale_values
        functions = [
            torch.pi ** (-0.25)
            / torch.sqrt(scale_values)
            * torch.exp(-0.5 * xi.square())
        ]
        if max_degree >= 1:
            functions.append(np.sqrt(2.0) * xi * functions[0])
        for order in range(1, max_degree):
            functions.append(
                np.sqrt(2.0 / float(order + 1))
                * xi
                * functions[order]
                - np.sqrt(float(order) / float(order + 1))
                * functions[order - 1]
            )
        return torch.stack(functions, dim=-1)

    x_functions = one_dimensional(values[:, 0])
    y_functions = one_dimensional(values[:, 1])
    return torch.stack(
        [
            x_functions[:, first] * y_functions[:, second]
            for first, second in modes
        ],
        dim=-1,
    )


def torch_squared_hermite_density(
    torch_module,
    velocity,
    coefficients,
    *,
    temperature,
    modes: Sequence[tuple[int, int]],
    floor_mixture: float = 0.0,
):
    """Evaluate a full-space density with exact analytic normalization."""

    torch = torch_module
    values = torch.as_tensor(velocity)
    theta = torch.as_tensor(
        temperature, dtype=values.dtype, device=values.device
    )
    if theta.ndim == 0:
        theta = theta.expand(values.shape[0])
    theta = torch.broadcast_to(theta.reshape(-1), (values.shape[0],))
    if bool(torch.any(theta <= 0.0)):
        raise ValueError("temperature must be positive")
    floor = float(floor_mixture)
    if not 0.0 <= floor < 1.0:
        raise ValueError("floor_mixture must lie in [0,1)")

    basis = torch_tensor_hermite_basis(
        torch,
        values,
        modes=modes,
        scale=torch.sqrt(2.0 * theta),
    )
    coefficient_rows = torch.as_tensor(
        coefficients, dtype=values.dtype, device=values.device
    )
    if coefficient_rows.ndim == 1:
        coefficient_rows = coefficient_rows[None, :]
    coefficient_rows = torch.broadcast_to(
        coefficient_rows, (values.shape[0], len(modes))
    )
    norm_squared = torch.sum(coefficient_rows.square(), dim=-1).clamp_min(
        torch.finfo(values.dtype).tiny
    )
    amplitude = torch.sum(coefficient_rows * basis, dim=-1)
    density = amplitude.square() / norm_squared
    maxwellian = torch.exp(
        -0.5 * torch.sum(values.square(), dim=-1) / theta
    ) / (2.0 * torch.pi * theta)
    return (1.0 - floor) * density + floor * maxwellian


def torch_squared_hermite_density_grid(
    torch_module,
    velocity,
    coefficients,
    *,
    temperature,
    modes: Sequence[tuple[int, int]],
    floor_mixture: float = 0.0,
):
    """Evaluate coefficient rows on one shared full-space velocity grid."""

    torch = torch_module
    values = torch.as_tensor(velocity)
    if values.ndim != 2 or int(values.shape[-1]) != 2:
        raise ValueError("velocity must have shape [V,2]")
    coefficient_rows = torch.as_tensor(
        coefficients, dtype=values.dtype, device=values.device
    )
    if coefficient_rows.ndim == 1:
        coefficient_rows = coefficient_rows[None, :]
    if (
        coefficient_rows.ndim != 2
        or int(coefficient_rows.shape[-1]) != len(modes)
    ):
        raise ValueError("coefficients must have shape [N,mode_count]")
    theta = torch.as_tensor(
        temperature, dtype=values.dtype, device=values.device
    )
    if int(theta.numel()) != 1 or not bool(torch.all(theta > 0.0)):
        raise ValueError("shared-grid temperature must be one positive scalar")
    theta = theta.reshape(())
    floor = float(floor_mixture)
    if not 0.0 <= floor < 1.0:
        raise ValueError("floor_mixture must lie in [0,1)")

    basis = torch_tensor_hermite_basis(
        torch,
        values,
        modes=modes,
        scale=torch.sqrt(2.0 * theta),
    )
    norm_squared = torch.sum(coefficient_rows.square(), dim=-1).clamp_min(
        torch.finfo(values.dtype).tiny
    )
    amplitude = coefficient_rows @ basis.transpose(0, 1)
    density = amplitude.square() / norm_squared[:, None]
    maxwellian = torch.exp(
        -0.5 * torch.sum(values.square(), dim=-1) / theta
    ) / (2.0 * torch.pi * theta)
    return (1.0 - floor) * density + floor * maxwellian[None, :]
