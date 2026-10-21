from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from agent_verifier.vpfp.hermite_density import total_degree_mode_pairs


MACRO_OUTPUT_CONTRACT = "vpfp_continuous_macro_output_v1"
MACRO_TRUTH_SCHEMA = "vpfp_continuous_fullspace_macro_truth_v1"
MACRO_OUTPUT_KEYS = ("rho", "momentum", "kinetic_energy")


def is_continuous_macro_question(question: dict[str, Any]) -> bool:
    return str(question.get("solver_output_contract_version") or "") == (
        MACRO_OUTPUT_CONTRACT
    )


def torch_squared_hermite_moments(
    torch_module,
    coefficients,
    *,
    modes: Sequence[tuple[int, int]],
    temperature,
    floor_mixture: float = 0.0,
) -> dict[str, Any]:
    """Evaluate exact full-space moments of a squared Hermite amplitude.

    The returned moments belong to the normalized velocity density ``g``. The
    multiplication recurrence is applied in a degree-extended space, so modes
    leaving and re-entering the represented subspace are not truncated.
    """

    torch = torch_module
    values = torch.as_tensor(coefficients)
    mode_pairs = tuple((int(first), int(second)) for first, second in modes)
    if values.ndim < 1 or int(values.shape[-1]) != len(mode_pairs):
        raise ValueError("coefficient shape does not match Hermite modes")
    if not mode_pairs or len(set(mode_pairs)) != len(mode_pairs):
        raise ValueError("Hermite modes must be nonempty and unique")
    if min(min(pair) for pair in mode_pairs) < 0:
        raise ValueError("Hermite mode degrees must be nonnegative")

    batch_shape = tuple(int(value) for value in values.shape[:-1])
    flat = values.reshape(-1, len(mode_pairs))
    theta = torch.as_tensor(
        temperature,
        dtype=values.dtype,
        device=values.device,
    )
    theta = torch.broadcast_to(theta, batch_shape).reshape(-1)
    if bool(torch.any(~torch.isfinite(theta)).item()) or bool(
        torch.any(theta <= 0.0).item()
    ):
        raise ValueError("temperature must be finite and positive")
    floor = float(floor_mixture)
    if not np.isfinite(floor) or not 0.0 <= floor < 1.0:
        raise ValueError("floor_mixture must lie in [0,1)")

    maximum_degree = max(first + second for first, second in mode_pairs)
    extended_modes = total_degree_mode_pairs(maximum_degree + 1)
    extended_index = {
        pair: index for index, pair in enumerate(extended_modes)
    }
    original_indices = torch.as_tensor(
        [extended_index[pair] for pair in mode_pairs],
        dtype=torch.long,
        device=values.device,
    )
    embedded = torch.zeros(
        (int(flat.shape[0]), len(extended_modes)),
        dtype=values.dtype,
        device=values.device,
    ).index_add(1, original_indices, flat)

    def multiply_velocity(axis: int):
        source_indices: list[int] = []
        destination_indices: list[int] = []
        factors: list[float] = []
        for source_index, pair in enumerate(mode_pairs):
            degree = pair[axis]
            raised = list(pair)
            raised[axis] += 1
            source_indices.append(source_index)
            destination_indices.append(extended_index[tuple(raised)])
            factors.append(np.sqrt(float(degree + 1)))
            if degree:
                lowered = list(pair)
                lowered[axis] -= 1
                source_indices.append(source_index)
                destination_indices.append(extended_index[tuple(lowered)])
                factors.append(np.sqrt(float(degree)))
        source = torch.as_tensor(
            source_indices, dtype=torch.long, device=values.device
        )
        destination = torch.as_tensor(
            destination_indices, dtype=torch.long, device=values.device
        )
        factor = torch.as_tensor(
            factors, dtype=values.dtype, device=values.device
        )
        contributions = (
            flat.index_select(1, source)
            * factor[None, :]
            * torch.sqrt(theta)[:, None]
        )
        return torch.zeros_like(embedded).index_add(
            1, destination, contributions
        )

    velocity_x = multiply_velocity(0)
    velocity_y = multiply_velocity(1)
    tiny = torch.finfo(values.dtype).tiny
    norm_squared = torch.sum(flat.square(), dim=-1).clamp_min(tiny)
    square_weight = 1.0 - floor
    mean_x = square_weight * torch.sum(
        embedded * velocity_x, dim=-1
    ) / norm_squared
    mean_y = square_weight * torch.sum(
        embedded * velocity_y, dim=-1
    ) / norm_squared
    second_xx = square_weight * torch.sum(
        velocity_x.square(), dim=-1
    ) / norm_squared + floor * theta
    second_xy = square_weight * torch.sum(
        velocity_x * velocity_y, dim=-1
    ) / norm_squared
    second_yy = square_weight * torch.sum(
        velocity_y.square(), dim=-1
    ) / norm_squared + floor * theta

    return {
        "mass": torch.ones_like(theta).reshape(batch_shape),
        "mean_velocity": torch.stack((mean_x, mean_y), dim=-1).reshape(
            *batch_shape, 2
        ),
        "raw_second_moment": torch.stack(
            (second_xx, second_xy, second_yy), dim=-1
        ).reshape(*batch_shape, 3),
        "kinetic_energy": (0.5 * (second_xx + second_yy)).reshape(
            batch_shape
        ),
    }


def evaluate_continuous_hermite_macro_points(
    model,
    latent,
    points_txy,
    *,
    micro_latent=None,
    equation_condition=None,
    chunk_size: int = 8192,
) -> dict[str, np.ndarray]:
    """Evaluate exact full-space macro fields at Agent-selected (t,x,y) points."""

    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    points = np.asarray(points_txy, dtype=np.float64)
    if points.ndim == 1:
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"macro truth points must have shape [N,3], got {points.shape}"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError("macro truth points must be finite")
    if not hasattr(model, "hermite_modes") or not hasattr(
        model.config, "velocity_floor_mixture"
    ):
        raise TypeError("macro truth requires a continuous Hermite decoder")
    rows: dict[str, list[np.ndarray]] = {
        "rho": [],
        "momentum": [],
        "kinetic_energy": [],
        "raw_second_moment": [],
    }
    torch = model.torch
    model.eval()
    with torch.no_grad():
        for start in range(0, int(points.shape[0]), int(chunk_size)):
            txy = points[start : start + int(chunk_size)]
            phase_points = np.column_stack(
                (txy, np.zeros((txy.shape[0], 2), dtype=np.float64))
            )
            fields = model.fields(
                phase_points,
                latent,
                micro_latent=micro_latent,
                equation_condition=equation_condition,
            )
            moments = torch_squared_hermite_moments(
                torch,
                fields["hermite_coefficients"],
                modes=model.hermite_modes,
                temperature=fields["equation_condition"][:, 1],
                floor_mixture=float(model.config.velocity_floor_mixture),
            )
            rho = fields["rho"]
            rows["rho"].append(rho.detach().cpu().numpy())
            rows["momentum"].append(
                (rho[:, None] * moments["mean_velocity"]).detach().cpu().numpy()
            )
            rows["kinetic_energy"].append(
                (rho * moments["kinetic_energy"]).detach().cpu().numpy()
            )
            rows["raw_second_moment"].append(
                (rho[:, None] * moments["raw_second_moment"])
                .detach()
                .cpu()
                .numpy()
            )
    result = {
        name: np.concatenate(parts, axis=0) for name, parts in rows.items()
    }
    if not all(np.all(np.isfinite(value)) for value in result.values()):
        raise FloatingPointError("continuous macro point truth is non-finite")
    return result


def evaluate_continuous_hermite_macro_grid(
    model,
    latent,
    *,
    grid,
    time,
    micro_latent=None,
    equation_condition=None,
    chunk_size: int = 8192,
) -> dict[str, np.ndarray]:
    """Evaluate exact full-space macro fields on a controller spatial grid."""

    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    if not hasattr(model, "hermite_modes") or not hasattr(
        model.config, "velocity_floor_mixture"
    ):
        raise TypeError("macro truth requires a continuous Hermite decoder")
    grid.validate()
    time.validate()
    shape = (int(time.nt), int(grid.nx), int(grid.ny))
    total = int(np.prod(shape))
    axes_t = np.linspace(float(time.t0), float(time.t1), int(time.nt))
    axes_x = np.linspace(
        0.0, float(grid.x_domain[0]), int(grid.nx), endpoint=False
    )
    axes_y = np.linspace(
        0.0, float(grid.x_domain[1]), int(grid.ny), endpoint=False
    )
    rho = np.empty(total, dtype=np.float64)
    momentum = np.empty((total, 2), dtype=np.float64)
    kinetic_energy = np.empty(total, dtype=np.float64)
    raw_second = np.empty((total, 3), dtype=np.float64)
    torch = model.torch
    model.eval()
    with torch.no_grad():
        for start in range(0, total, int(chunk_size)):
            stop = min(total, start + int(chunk_size))
            indices = np.arange(start, stop, dtype=np.int64)
            it, ix, iy = np.unravel_index(indices, shape)
            points = np.column_stack(
                (
                    axes_t[it],
                    axes_x[ix],
                    axes_y[iy],
                    np.zeros(stop - start),
                    np.zeros(stop - start),
                )
            )
            fields = model.fields(
                points,
                latent,
                micro_latent=micro_latent,
                equation_condition=equation_condition,
            )
            moments = torch_squared_hermite_moments(
                torch,
                fields["hermite_coefficients"],
                modes=model.hermite_modes,
                temperature=fields["equation_condition"][:, 1],
                floor_mixture=float(model.config.velocity_floor_mixture),
            )
            local_rho = fields["rho"]
            rho[start:stop] = local_rho.detach().cpu().numpy()
            momentum[start:stop] = (
                local_rho[:, None] * moments["mean_velocity"]
            ).detach().cpu().numpy()
            kinetic_energy[start:stop] = (
                local_rho * moments["kinetic_energy"]
            ).detach().cpu().numpy()
            raw_second[start:stop] = (
                local_rho[:, None] * moments["raw_second_moment"]
            ).detach().cpu().numpy()

    outputs = {
        "rho": rho.reshape(shape),
        "momentum": momentum.reshape(*shape, 2),
        "kinetic_energy": kinetic_energy.reshape(shape),
        "raw_second_moment": raw_second.reshape(*shape, 3),
    }
    if not all(np.all(np.isfinite(value)) for value in outputs.values()):
        raise FloatingPointError("continuous macro truth is non-finite")
    return outputs
