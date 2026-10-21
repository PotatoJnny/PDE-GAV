from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import resample

from agent_verifier.vpfp.operators import VPFPGrid, VPFPTime
from agent_verifier.vpfp.torch_reference import (
    VPFPTorchSolverConfig,
    solve_torch_vpfp_batch,
)


MACRO_FIELDS = ("rho", "momentum", "kinetic_energy")


@dataclass(frozen=True)
class KineticAuditResolution:
    nx: int
    ny: int
    nvx: int
    nvy: int
    nt: int
    cfl: float


def kinetic_resolution_for_tier(
    public_case: dict[str, Any], tier: int
) -> KineticAuditResolution:
    """Choose public, intermediate, or declared reference resolution."""

    grid = public_case["grid"]
    time = public_case["time"]
    level = min(max(int(tier), 0), 2)
    if level == 0:
        return KineticAuditResolution(
            nx=int(grid.nx),
            ny=int(grid.ny),
            nvx=int(grid.nvx),
            nvy=int(grid.nvy),
            nt=int(time.nt),
            cfl=0.12,
        )

    reference = dict(
        dict(public_case["question"].get("development_view") or {}).get(
            "reference_resolution"
        )
        or {}
    )
    reference_values = {
        "nx": int(reference.get("nx", 2 * int(grid.nx))),
        "ny": int(reference.get("ny", 2 * int(grid.ny))),
        "nvx": int(reference.get("nvx", 2 * int(grid.nvx) - 1)),
        "nvy": int(reference.get("nvy", 2 * int(grid.nvy) - 1)),
        "nt": int(reference.get("nt", 2 * int(time.nt) - 1)),
    }
    if level == 2:
        return KineticAuditResolution(**reference_values, cfl=0.05)

    def midpoint(public: int, refined: int) -> int:
        value = int(np.ceil(0.5 * (public + refined)))
        return max(public, value)

    return KineticAuditResolution(
        nx=midpoint(int(grid.nx), reference_values["nx"]),
        ny=midpoint(int(grid.ny), reference_values["ny"]),
        nvx=midpoint(int(grid.nvx), reference_values["nvx"]),
        nvy=midpoint(int(grid.nvy), reference_values["nvy"]),
        nt=midpoint(int(time.nt), reference_values["nt"]),
        cfl=0.075,
    )


def solve_interpolated_full_kinetic(
    public_case: dict[str, Any],
    query_times: np.ndarray,
    *,
    tier: int,
    moment_match_initial: bool = True,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """Solve full VPFP after reconstructing a finer view of public samples.

    This is deliberately an audit candidate, not a reference oracle. It only
    uses public arrays and public resolution metadata. Generator parameters and
    equation seeds are never consumed.
    """

    resolution = kinetic_resolution_for_tier(public_case, tier)
    arrays = public_case["arrays"]
    question = public_case["question"]
    public_grid = public_case["grid"]
    public_time = public_case["time"]
    grid = VPFPGrid(
        nx=resolution.nx,
        ny=resolution.ny,
        nvx=resolution.nvx,
        nvy=resolution.nvy,
        x_domain=tuple(float(value) for value in public_grid.x_domain),
        v_bounds=tuple(float(value) for value in public_grid.v_bounds),
        dtype="float32",
    )
    time = VPFPTime(
        t0=float(public_time.t0),
        t1=float(public_time.t1),
        nt=resolution.nt,
    )
    target_times = np.linspace(time.t0, time.t1, time.nt, dtype=np.float64)
    target_vx = np.linspace(grid.v_bounds[0], grid.v_bounds[1], grid.nvx)
    target_vy = np.linspace(grid.v_bounds[2], grid.v_bounds[3], grid.nvy)

    rho0 = _periodic_space_resample(
        np.asarray(arrays["rho0"], dtype=np.float64),
        grid.nx,
        grid.ny,
        axes=(0, 1),
    )
    momentum0 = _periodic_space_resample(
        np.asarray(arrays["momentum0"], dtype=np.float64),
        grid.nx,
        grid.ny,
        axes=(0, 1),
    )
    kinetic0 = _periodic_space_resample(
        np.asarray(arrays["kinetic_energy0"], dtype=np.float64),
        grid.nx,
        grid.ny,
        axes=(0, 1),
    )
    g0 = _reconstruct_positive_g0(
        np.asarray(arrays["g0"], dtype=np.float64),
        source_vx=np.asarray(arrays["vx"], dtype=np.float64),
        source_vy=np.asarray(arrays["vy"], dtype=np.float64),
        target_vx=target_vx,
        target_vy=target_vy,
        nx=grid.nx,
        ny=grid.ny,
    )
    weights = _trapezoidal_weights(target_vx, target_vy)
    if moment_match_initial:
        g0 = _match_density_velocity_moments(
            g0,
            target_vx,
            target_vy,
            weights,
            momentum0=momentum0,
            kinetic0=kinetic0,
            rho0=rho0,
        )
    else:
        g0 = _normalize_velocity(g0, weights)

    source = _resample_source(
        np.asarray(arrays["source"], dtype=np.float64),
        source_times=np.asarray(arrays["t"], dtype=np.float64),
        source_vx=np.asarray(arrays["vx"], dtype=np.float64),
        source_vy=np.asarray(arrays["vy"], dtype=np.float64),
        target_times=target_times,
        target_vx=target_vx,
        target_vy=target_vy,
        nx=grid.nx,
        ny=grid.ny,
    )
    prepared = {
        "schema": "vpfp_torch_prepared_equation_batch_v1",
        "f0": (rho0[..., None, None] * g0)[None].astype(np.float32),
        "source_frames": source[None].astype(np.float32),
        "collision_nu": np.asarray(
            [float(public_case["collision_nu"])], dtype=np.float32
        ),
        "temperature": np.asarray(
            [float(question["temperature"])], dtype=np.float32
        ),
        "preparation_wall_time_seconds": 0.0,
        "source_interpolation": "linear_output_frames",
    }
    solution = solve_torch_vpfp_batch(
        prepared,
        grid=grid,
        time=time,
        config=VPFPTorchSolverConfig(
            internal_dt_max=0.001,
            cfl=float(resolution.cfl),
            positivity_floor=1.0e-13,
            rho_floor=1.0e-10,
            dtype="float32",
            source_interpolation="linear_output_frames",
        ),
        device=device,
    )
    macro = _torch_macro_trajectory(solution["f"], target_vx, target_vy, weights)
    restricted = {
        name: _periodic_space_resample(
            value,
            int(public_grid.nx),
            int(public_grid.ny),
            axes=(1, 2),
        )
        for name, value in macro.items()
    }
    output = {
        name: np.asarray(
            CubicSpline(target_times, value, axis=0)(query_times),
            dtype=np.float64,
        )
        for name, value in restricted.items()
    }
    _restore_public_initial_values(public_case, query_times, output)
    _validate_macro_output(output)
    return output


def solve_refined_moment_candidate(
    public_case: dict[str, Any],
    query_times: np.ndarray,
    *,
    champion_module: Any,
    maximum_degree: int,
    spatial_factor: float,
    maximum_step: float,
    learned_correction: bool,
) -> dict[str, np.ndarray]:
    """Run a higher-order member of the champion's raw-moment hierarchy."""

    nx = int(public_case["grid"].nx)
    ny = int(public_case["grid"].ny)
    nx_internal = max(nx, int(np.ceil(float(spatial_factor) * nx)))
    ny_internal = max(ny, int(np.ceil(float(spatial_factor) * ny)))
    physical = champion_module.solve_spatially_refined(
        public_case,
        nx_internal,
        ny_internal,
        _collision_scale=1.0,
        _maximum_degree=int(maximum_degree),
        _maximum_step=float(maximum_step),
        _initial_log_refine_blend=1.0,
        _initial_log_refine_n=45,
        _log_space_g0=True,
        _log_velocity_kind="cubic",
        _initial_log_refine_degree_blends=(1.0,),
        _initial_second_reconstruction="spline_ensemble",
    )
    background = float(public_case["question"].get("background_density", 1.0))
    if len(physical["rho"]) > 1:
        mean_error = background - physical["rho"][1:].mean(axis=(1, 2))
        physical["rho"][1:] += mean_error[:, None, None]

    source_times = np.asarray(public_case["arrays"]["t"], dtype=np.float64)
    output = {
        name: np.asarray(
            CubicSpline(source_times, physical[name], axis=0)(query_times),
            dtype=np.float64,
        )
        for name in MACRO_FIELDS
    }
    if learned_correction:
        correction, correction_times = champion_module.canonical_correction(
            public_case, physical, 2
        )
        residual, residual_times = champion_module.residual_correction(
            public_case, physical
        )
        correction = champion_module.resize_correction(correction, nx, ny)
        residual = champion_module.resize_correction(residual, nx, ny)
        combined = (
            CubicSpline(correction_times, correction, axis=0)(query_times)
            + CubicSpline(residual_times, residual, axis=0)(query_times)
        )
        output["rho"] += combined[:, 0]
        output["momentum"][..., 0] += combined[:, 1]
        output["momentum"][..., 1] += combined[:, 2]
        output["kinetic_energy"] += combined[:, 3]
    _restore_public_initial_values(public_case, query_times, output)
    _validate_macro_output(output)
    return output


def equation_feature_tensor(public_case: dict[str, Any]) -> np.ndarray:
    """Build a fixed 17x17 equation tensor for direct and residual learning."""

    arrays = public_case["arrays"]
    rho0 = _periodic_space_resample(
        np.asarray(arrays["rho0"], dtype=np.float64), 17, 17, axes=(0, 1)
    )
    momentum0 = _periodic_space_resample(
        np.asarray(arrays["momentum0"], dtype=np.float64), 17, 17, axes=(0, 1)
    )
    kinetic0 = _periodic_space_resample(
        np.asarray(arrays["kinetic_energy0"], dtype=np.float64),
        17,
        17,
        axes=(0, 1),
    )
    g0 = _canonical_velocity_field(public_case, "g0", time_dependent=False)
    source = _canonical_velocity_field(public_case, "source", time_dependent=True)
    weights = _trapezoidal_weights(
        np.linspace(float(arrays["vx"][0]), float(arrays["vx"][-1]), 15),
        np.linspace(float(arrays["vy"][0]), float(arrays["vy"][-1]), 15),
    )
    vx = np.linspace(float(arrays["vx"][0]), float(arrays["vx"][-1]), 15)
    vy = np.linspace(float(arrays["vy"][0]), float(arrays["vy"][-1]), 15)
    velocity_x = vx[:, None]
    velocity_y = vy[None, :]
    f0 = rho0[..., None, None] * g0
    channels: list[np.ndarray] = [
        rho0,
        momentum0[..., 0],
        momentum0[..., 1],
        kinetic0,
    ]
    for total in range(7):
        for degree_y in range(total + 1):
            degree_x = total - degree_y
            channels.append(
                np.sum(
                    f0
                    * weights
                    * velocity_x**degree_x
                    * velocity_y**degree_y,
                    axis=(-2, -1),
                )
            )
    for frame in source:
        channels.extend(
            (
                np.sum(frame * weights, axis=(-2, -1)),
                np.sum(frame * weights * velocity_x, axis=(-2, -1)),
                np.sum(frame * weights * velocity_y, axis=(-2, -1)),
                np.sum(
                    frame
                    * weights
                    * 0.5
                    * (velocity_x**2 + velocity_y**2),
                    axis=(-2, -1),
                ),
            )
        )
    log_g = np.moveaxis(
        np.log(np.maximum(g0, 1.0e-35)), (-2, -1), (0, 1)
    ).reshape(225, 17, 17)
    channels.extend(log_g)
    ones = np.ones((17, 17), dtype=np.float64)
    nu = float(public_case["collision_nu"])
    temperature = float(public_case["question"]["temperature"])
    channels.extend(
        (
            ones * nu,
            ones * temperature,
            ones * nu * temperature,
            ones * nu**2,
            ones * temperature**2,
        )
    )
    return np.asarray(channels, dtype=np.float32)


def canonical_macro_tensor(
    public_case: dict[str, Any],
    macro: dict[str, np.ndarray],
    *,
    source_times: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return four canonical time knots as 16 channels on a 17x17 torus."""

    public_times = np.asarray(
        public_case["arrays"]["t"] if source_times is None else source_times,
        dtype=np.float64,
    )
    canonical_times = np.linspace(public_times[0], public_times[-1], 4)
    channels = []
    for time_index in range(4):
        channels.extend(
            (
                CubicSpline(public_times, macro["rho"], axis=0)(canonical_times)[
                    time_index
                ],
                CubicSpline(
                    public_times, macro["momentum"], axis=0
                )(canonical_times)[time_index, ..., 0],
                CubicSpline(
                    public_times, macro["momentum"], axis=0
                )(canonical_times)[time_index, ..., 1],
                CubicSpline(
                    public_times, macro["kinetic_energy"], axis=0
                )(canonical_times)[time_index],
            )
        )
    value = np.asarray(channels, dtype=np.float64)
    if value.shape[-2:] != (17, 17):
        value = _periodic_space_resample(value, 17, 17, axes=(-2, -1))
    return value.astype(np.float32), canonical_times


def macro_from_canonical_tensor(
    public_case: dict[str, Any],
    values: np.ndarray,
    canonical_times: np.ndarray,
    query_times: np.ndarray,
) -> dict[str, np.ndarray]:
    array = np.asarray(values, dtype=np.float64).reshape(4, 4, 17, 17)
    nx = int(public_case["grid"].nx)
    ny = int(public_case["grid"].ny)
    if (nx, ny) != (17, 17):
        array = _periodic_space_resample(array, nx, ny, axes=(-2, -1))
    interpolated = CubicSpline(canonical_times, array, axis=0)(query_times)
    output = {
        "rho": interpolated[:, 0],
        "momentum": np.stack(
            (interpolated[:, 1], interpolated[:, 2]), axis=-1
        ),
        "kinetic_energy": interpolated[:, 3],
    }
    _restore_public_initial_values(public_case, query_times, output)
    _validate_macro_output(output)
    return output


def _canonical_velocity_field(
    public_case: dict[str, Any], name: str, *, time_dependent: bool
) -> np.ndarray:
    arrays = public_case["arrays"]
    value = np.asarray(arrays[name], dtype=np.float64)
    if time_dependent:
        target_times = np.linspace(
            float(arrays["t"][0]), float(arrays["t"][-1]), 4
        )
        value = CubicSpline(
            np.asarray(arrays["t"], dtype=np.float64), value, axis=0
        )(target_times)
        value = _periodic_space_resample(value, 17, 17, axes=(1, 2))
    else:
        value = _periodic_space_resample(value, 17, 17, axes=(0, 1))
    target_vx = np.linspace(float(arrays["vx"][0]), float(arrays["vx"][-1]), 15)
    target_vy = np.linspace(float(arrays["vy"][0]), float(arrays["vy"][-1]), 15)
    value = CubicSpline(
        np.asarray(arrays["vx"], dtype=np.float64), value, axis=-2
    )(target_vx)
    return CubicSpline(
        np.asarray(arrays["vy"], dtype=np.float64), value, axis=-1
    )(target_vy)


def _resample_source(
    source: np.ndarray,
    *,
    source_times: np.ndarray,
    source_vx: np.ndarray,
    source_vy: np.ndarray,
    target_times: np.ndarray,
    target_vx: np.ndarray,
    target_vy: np.ndarray,
    nx: int,
    ny: int,
) -> np.ndarray:
    value = CubicSpline(source_times, source, axis=0)(target_times)
    value = _periodic_space_resample(value, nx, ny, axes=(1, 2))
    value = CubicSpline(source_vx, value, axis=-2)(target_vx)
    value = CubicSpline(source_vy, value, axis=-1)(target_vy)
    return np.asarray(value, dtype=np.float32)


def _reconstruct_positive_g0(
    g0: np.ndarray,
    *,
    source_vx: np.ndarray,
    source_vy: np.ndarray,
    target_vx: np.ndarray,
    target_vy: np.ndarray,
    nx: int,
    ny: int,
) -> np.ndarray:
    log_value = np.log(np.maximum(g0, 1.0e-35))
    log_value = _periodic_space_resample(log_value, nx, ny, axes=(0, 1))
    log_value = CubicSpline(source_vx, log_value, axis=-2)(target_vx)
    log_value = CubicSpline(source_vy, log_value, axis=-1)(target_vy)
    log_value -= np.max(log_value, axis=(-2, -1), keepdims=True)
    return np.exp(np.clip(log_value, -80.0, 0.0))


def _match_density_velocity_moments(
    base: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    weights: np.ndarray,
    *,
    momentum0: np.ndarray,
    kinetic0: np.ndarray,
    rho0: np.ndarray,
) -> np.ndarray:
    shape = base.shape[:2]
    points = int(base.shape[-2] * base.shape[-1])
    features = np.stack(
        np.broadcast_arrays(
            vx[:, None],
            vy[None, :],
            0.5 * (vx[:, None] ** 2 + vy[None, :] ** 2),
        ),
        axis=-1,
    ).reshape(points, 3)
    log_base = np.log(np.maximum(base.reshape(-1, points), 1.0e-35))
    target = np.stack(
        (
            momentum0[..., 0] / np.maximum(rho0, 1.0e-12),
            momentum0[..., 1] / np.maximum(rho0, 1.0e-12),
            kinetic0 / np.maximum(rho0, 1.0e-12),
        ),
        axis=-1,
    ).reshape(-1, 3)
    multipliers = np.zeros_like(target)
    weight = weights.reshape(1, points)
    eye = np.eye(3, dtype=np.float64)[None]
    for _ in range(8):
        logits = log_base + multipliers @ features.T
        logits -= np.max(logits, axis=1, keepdims=True)
        probability = np.exp(np.clip(logits, -80.0, 0.0)) * weight
        probability /= np.maximum(
            np.sum(probability, axis=1, keepdims=True), 1.0e-30
        )
        mean = probability @ features
        residual = target - mean
        second = np.einsum(
            "bp,pi,pj->bij", probability, features, features, optimize=True
        )
        covariance = second - mean[:, :, None] * mean[:, None, :] + 1.0e-7 * eye
        try:
            step = np.linalg.solve(covariance, residual[..., None])[..., 0]
        except np.linalg.LinAlgError:
            break
        step[:, :2] = np.clip(step[:, :2], -0.5, 0.5)
        step[:, 2] = np.clip(step[:, 2], -0.05, 0.05)
        multipliers += step
        if float(np.max(np.abs(residual))) < 2.0e-7:
            break
    logits = log_base + multipliers @ features.T
    logits -= np.max(logits, axis=1, keepdims=True)
    matched = np.exp(np.clip(logits, -80.0, 0.0)).reshape(
        *shape, len(vx), len(vy)
    )
    return _normalize_velocity(matched, weights)


def _normalize_velocity(value: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mass = np.sum(value * weights, axis=(-2, -1), keepdims=True)
    return value / np.maximum(mass, 1.0e-30)


def _trapezoidal_weights(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    wx = np.ones(len(vx), dtype=np.float64)
    wy = np.ones(len(vy), dtype=np.float64)
    wx[[0, -1]] = 0.5
    wy[[0, -1]] = 0.5
    return wx[:, None] * wy[None, :] * (vx[1] - vx[0]) * (vy[1] - vy[0])


def _torch_macro_trajectory(
    f: Any, vx: np.ndarray, vy: np.ndarray, weights: np.ndarray
) -> dict[str, np.ndarray]:
    import torch

    weight = torch.as_tensor(weights, dtype=f.dtype, device=f.device)
    velocity_x = torch.as_tensor(vx[:, None], dtype=f.dtype, device=f.device)
    velocity_y = torch.as_tensor(vy[None, :], dtype=f.dtype, device=f.device)
    density = torch.sum(f[0] * weight, dim=(-2, -1))
    momentum_x = torch.sum(f[0] * weight * velocity_x, dim=(-2, -1))
    momentum_y = torch.sum(f[0] * weight * velocity_y, dim=(-2, -1))
    energy = torch.sum(
        f[0]
        * weight
        * 0.5
        * (velocity_x.square() + velocity_y.square()),
        dim=(-2, -1),
    )
    return {
        "rho": density.detach().cpu().numpy().astype(np.float64),
        "momentum": torch.stack((momentum_x, momentum_y), dim=-1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64),
        "kinetic_energy": energy.detach().cpu().numpy().astype(np.float64),
    }


def _periodic_space_resample(
    value: np.ndarray, nx: int, ny: int, *, axes: tuple[int, int]
) -> np.ndarray:
    result = resample(np.asarray(value), int(nx), axis=axes[0])
    result = resample(result, int(ny), axis=axes[1])
    return np.asarray(result.real)


def _restore_public_initial_values(
    public_case: dict[str, Any],
    query_times: np.ndarray,
    output: dict[str, np.ndarray],
) -> None:
    at_initial = np.isclose(
        np.asarray(query_times, dtype=np.float64),
        float(public_case["time"].t0),
        rtol=0.0,
        atol=1.0e-13,
    )
    if not np.any(at_initial):
        return
    arrays = public_case["arrays"]
    output["rho"][at_initial] = np.asarray(arrays["rho0"], dtype=np.float64)
    output["momentum"][at_initial] = np.asarray(
        arrays["momentum0"], dtype=np.float64
    )
    output["kinetic_energy"][at_initial] = np.asarray(
        arrays["kinetic_energy0"], dtype=np.float64
    )


def _validate_macro_output(output: dict[str, np.ndarray]) -> None:
    for name in MACRO_FIELDS:
        if not np.all(np.isfinite(output[name])):
            raise FloatingPointError(f"non-finite {name} in method expansion audit")
