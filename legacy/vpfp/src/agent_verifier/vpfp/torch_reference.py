from __future__ import annotations

import time as walltime
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from agent_verifier.vpfp.high_fidelity import (
    RICH_EQUATION_FAMILY,
    evaluate_continuous_equation,
)
from agent_verifier.vpfp.operators import VPFPGrid, VPFPTime


@dataclass(frozen=True)
class VPFPTorchSolverConfig:
    internal_dt_max: float = 0.008
    cfl: float = 0.18
    positivity_floor: float = 1.0e-13
    rho_floor: float = 1.0e-10
    dtype: str = "float32"
    source_interpolation: str = "linear_output_frames"

    def validate(self) -> None:
        if self.internal_dt_max <= 0.0:
            raise ValueError("internal_dt_max must be positive")
        if self.cfl <= 0.0:
            raise ValueError("cfl must be positive")
        if min(self.positivity_floor, self.rho_floor) <= 0.0:
            raise ValueError("positivity and density floors must be positive")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if self.source_interpolation != "linear_output_frames":
            raise ValueError("unsupported source interpolation")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare_torch_equation_batch(
    specs: Sequence[dict[str, Any]],
    *,
    grid: VPFPGrid,
    time: VPFPTime,
    dtype: str = "float32",
) -> dict[str, Any]:
    """Evaluate continuous initial data and source frames for a Torch solve."""

    if not specs:
        raise ValueError("at least one VPFP equation specification is required")
    grid.validate()
    time.validate()
    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be float32 or float64")
    numpy_dtype = np.float32 if dtype == "float32" else np.float64
    started = walltime.perf_counter()
    equations = [
        evaluate_continuous_equation(spec, grid=grid, time=time)
        for spec in specs
    ]
    f0 = np.stack(
        [
            np.asarray(equation["f0_continuous"], dtype=numpy_dtype)
            for equation in equations
        ]
    )
    source = np.stack(
        [
            np.asarray(equation["source"], dtype=numpy_dtype)
            for equation in equations
        ]
    )
    collision_nu = np.asarray(
        [float(spec["collision_nu"]) for spec in specs],
        dtype=numpy_dtype,
    )
    temperature = np.asarray(
        [float(spec["temperature"]) for spec in specs],
        dtype=numpy_dtype,
    )
    if not np.all(np.isfinite(f0)) or not np.all(np.isfinite(source)):
        raise FloatingPointError("prepared VPFP equation batch is non-finite")
    return {
        "schema": "vpfp_torch_prepared_equation_batch_v1",
        "f0": f0,
        "source_frames": source,
        "collision_nu": collision_nu,
        "temperature": temperature,
        "preparation_wall_time_seconds": walltime.perf_counter() - started,
        "source_interpolation": "linear_output_frames",
    }


def solve_torch_vpfp_batch(
    prepared: dict[str, Any],
    *,
    grid: VPFPGrid,
    time: VPFPTime,
    config: VPFPTorchSolverConfig | None = None,
    device: str | Any = "cpu",
) -> dict[str, Any]:
    """Solve a batch with RK4, spectral x operators, and conservative OU FV."""

    torch = _torch()
    cfg = config or VPFPTorchSolverConfig()
    cfg.validate()
    grid.validate()
    time.validate()
    if prepared.get("schema") != "vpfp_torch_prepared_equation_batch_v1":
        raise ValueError("unexpected prepared VPFP batch schema")
    if prepared.get("source_interpolation") != cfg.source_interpolation:
        raise ValueError("prepared source interpolation does not match solver")
    torch_dtype = torch.float32 if cfg.dtype == "float32" else torch.float64
    target_device = torch.device(device)
    transfer_started = walltime.perf_counter()
    f = torch.as_tensor(
        prepared["f0"],
        dtype=torch_dtype,
        device=target_device,
    ).clone()
    source_frames = torch.as_tensor(
        prepared["source_frames"],
        dtype=torch_dtype,
        device=target_device,
    )
    collision_nu = torch.as_tensor(
        prepared["collision_nu"],
        dtype=torch_dtype,
        device=target_device,
    )
    temperature = torch.as_tensor(
        prepared["temperature"],
        dtype=torch_dtype,
        device=target_device,
    )
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    transfer_seconds = walltime.perf_counter() - transfer_started

    expected_f = (
        len(collision_nu),
        grid.nx,
        grid.ny,
        grid.nvx,
        grid.nvy,
    )
    expected_source = (len(collision_nu), time.nt) + expected_f[1:]
    if tuple(f.shape) != expected_f:
        raise ValueError(
            f"prepared f0 shape {tuple(f.shape)} does not match {expected_f}"
        )
    if tuple(source_frames.shape) != expected_source:
        raise ValueError(
            "prepared source shape does not match batch, time, and grid"
        )
    if tuple(temperature.shape) != tuple(collision_nu.shape):
        raise ValueError("temperature and collision arrays must match")

    trapezoidal_weights = _trapezoidal_velocity_weights(
        torch,
        grid=grid,
        dtype=torch_dtype,
        device=target_device,
    )
    repair_mass = torch.zeros_like(collision_nu)
    repair_steps = torch.zeros(
        collision_nu.shape,
        dtype=torch.int64,
        device=target_device,
    )
    trajectory = [f.clone()]
    _rho, _phi, electric = _trapezoidal_field(
        torch,
        f,
        grid=grid,
        rho_floor=float(cfg.rho_floor),
    )
    rhs_evaluations = 0
    substeps_by_frame = []
    stable_bounds = []
    min_step = float("inf")
    max_step = 0.0

    solve_started = walltime.perf_counter()
    with torch.inference_mode():
        for frame in range(time.nt - 1):
            frame_start = float(time.t0) + float(frame) * float(time.dt)
            frame_end = frame_start + float(time.dt)
            electric_max = torch.amax(
                torch.abs(electric),
                dim=(1, 2, 3),
            )
            stable_per_case = _stable_step_bound(
                torch,
                grid=grid,
                collision_nu=collision_nu,
                temperature=temperature,
                electric_max=electric_max,
                cfl=float(cfg.cfl),
            )
            stable_dt = float(torch.min(stable_per_case).item())
            stable_bounds.append(stable_dt)
            step_cap = min(float(cfg.internal_dt_max), stable_dt)
            substeps = max(
                1,
                int(np.ceil((frame_end - frame_start) / step_cap)),
            )
            substeps_by_frame.append(substeps)
            h = (frame_end - frame_start) / float(substeps)
            min_step = min(min_step, h)
            max_step = max(max_step, h)
            for substep in range(substeps):
                current_time = frame_start + float(substep) * h
                k1 = _rhs(
                    torch,
                    f,
                    current_time,
                    source_frames=source_frames,
                    collision_nu=collision_nu,
                    temperature=temperature,
                    grid=grid,
                    time=time,
                    rho_floor=float(cfg.rho_floor),
                )
                k2 = _rhs(
                    torch,
                    f + 0.5 * h * k1,
                    current_time + 0.5 * h,
                    source_frames=source_frames,
                    collision_nu=collision_nu,
                    temperature=temperature,
                    grid=grid,
                    time=time,
                    rho_floor=float(cfg.rho_floor),
                )
                k3 = _rhs(
                    torch,
                    f + 0.5 * h * k2,
                    current_time + 0.5 * h,
                    source_frames=source_frames,
                    collision_nu=collision_nu,
                    temperature=temperature,
                    grid=grid,
                    time=time,
                    rho_floor=float(cfg.rho_floor),
                )
                k4 = _rhs(
                    torch,
                    f + h * k3,
                    current_time + h,
                    source_frames=source_frames,
                    collision_nu=collision_nu,
                    temperature=temperature,
                    grid=grid,
                    time=time,
                    rho_floor=float(cfg.rho_floor),
                )
                rhs_evaluations += 4
                next_f = f + (h / 6.0) * (
                    k1 + 2.0 * k2 + 2.0 * k3 + k4
                )
                repair = torch.clamp(
                    float(cfg.positivity_floor) - next_f,
                    min=0.0,
                )
                repair_steps += torch.any(
                    repair > 0.0,
                    dim=(1, 2, 3, 4),
                ).to(dtype=torch.int64)
                repair_mass += (
                    torch.sum(
                        repair
                        * trapezoidal_weights[None, None, None],
                        dim=(1, 2, 3, 4),
                    )
                    * float(grid.dx)
                    * float(grid.dy)
                )
                f = torch.clamp(next_f, min=float(cfg.positivity_floor))
            _rho, _phi, electric = _trapezoidal_field(
                torch,
                f,
                grid=grid,
                rho_floor=float(cfg.rho_floor),
            )
            trajectory.append(f.clone())

        f_trajectory = torch.stack(trajectory, dim=1)
        rho_uniform, g_uniform = _uniform_rho_g(
            torch,
            f_trajectory,
            grid=grid,
            rho_floor=float(cfg.rho_floor),
        )
        phi_uniform, electric_uniform = _poisson_field(
            torch,
            rho_uniform,
            grid=grid,
        )
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    solve_seconds = walltime.perf_counter() - solve_started
    if not bool(torch.all(torch.isfinite(f_trajectory)).item()):
        raise FloatingPointError("Torch VPFP solve became non-finite")
    return {
        "schema": "vpfp_torch_batched_solution_v1",
        "f": f_trajectory,
        "rho": rho_uniform,
        "g": g_uniform,
        "phi": phi_uniform,
        "electric": electric_uniform,
        "diagnostics": {
            "backend": "torch_batched_rk4_spectral_x_chang_cooper_ou_fv_v2",
            "config": cfg.to_dict(),
            "batch_size": int(f.shape[0]),
            "rhs_evaluations": int(rhs_evaluations),
            "substeps_by_frame": substeps_by_frame,
            "stable_dt_bound_min": float(min(stable_bounds)),
            "min_internal_step": float(min_step),
            "max_internal_step": float(max_step),
            "repair_steps": repair_steps.detach().cpu().numpy(),
            "repair_mass": repair_mass.detach().cpu().numpy(),
            "preparation_wall_time_seconds": float(
                prepared.get("preparation_wall_time_seconds", 0.0)
            ),
            "host_to_device_wall_time_seconds": float(transfer_seconds),
            "solve_wall_time_seconds": float(solve_seconds),
        },
    }


def _rhs(
    torch,
    f,
    time_value: float,
    *,
    source_frames,
    collision_nu,
    temperature,
    grid: VPFPGrid,
    time: VPFPTime,
    rho_floor: float,
):
    rho = _velocity_integral(torch, f, grid=grid)
    _phi, electric = _poisson_field(torch, rho, grid=grid)
    dfdx, dfdy = _spectral_spatial_gradient(torch, f, grid=grid)
    acceleration, collision = _velocity_operators(
        torch,
        f,
        electric,
        temperature=temperature,
        grid=grid,
    )
    vx = torch.linspace(
        float(grid.v_bounds[0]),
        float(grid.v_bounds[1]),
        int(grid.nvx),
        dtype=f.dtype,
        device=f.device,
    )[None, None, None, :, None]
    vy = torch.linspace(
        float(grid.v_bounds[2]),
        float(grid.v_bounds[3]),
        int(grid.nvy),
        dtype=f.dtype,
        device=f.device,
    )[None, None, None, None, :]
    source = _interpolate_source(
        source_frames,
        time_value=time_value,
        time=time,
    )
    return (
        source
        - vx * dfdx
        - vy * dfdy
        - acceleration
        + collision_nu[:, None, None, None, None] * collision
    )


def _trapezoidal_field(
    torch,
    f,
    *,
    grid: VPFPGrid,
    rho_floor: float,
):
    rho = torch.clamp(
        _velocity_integral(torch, f, grid=grid),
        min=float(rho_floor),
    )
    phi, electric = _poisson_field(torch, rho, grid=grid)
    return rho, phi, electric


def _poisson_field(torch, rho, *, grid: VPFPGrid):
    centered = rho - torch.mean(rho, dim=(-2, -1), keepdim=True)
    rho_hat = torch.fft.fftn(centered, dim=(-2, -1))
    kx = (
        2.0
        * torch.pi
        * torch.fft.fftfreq(
            int(grid.nx),
            d=float(grid.dx),
            dtype=rho.dtype,
            device=rho.device,
        )
    )[:, None]
    ky = (
        2.0
        * torch.pi
        * torch.fft.fftfreq(
            int(grid.ny),
            d=float(grid.dy),
            dtype=rho.dtype,
            device=rho.device,
        )
    )[None, :]
    k2 = kx.square() + ky.square()
    safe_k2 = torch.where(k2 > 0.0, k2, torch.ones_like(k2))
    phi_hat = rho_hat / safe_k2
    phi_hat = torch.where(k2 > 0.0, phi_hat, torch.zeros_like(phi_hat))
    phi = torch.fft.ifftn(phi_hat, dim=(-2, -1)).real
    ex = torch.fft.ifftn(
        -1j * kx * phi_hat,
        dim=(-2, -1),
    ).real
    ey = torch.fft.ifftn(
        -1j * ky * phi_hat,
        dim=(-2, -1),
    ).real
    return phi, torch.stack((ex, ey), dim=-3)


def _spectral_spatial_gradient(torch, values, *, grid: VPFPGrid):
    transformed = torch.fft.fftn(values, dim=(1, 2))
    kx = (
        2.0
        * torch.pi
        * torch.fft.fftfreq(
            int(grid.nx),
            d=float(grid.dx),
            dtype=values.dtype,
            device=values.device,
        )
    )[None, :, None, None, None]
    ky = (
        2.0
        * torch.pi
        * torch.fft.fftfreq(
            int(grid.ny),
            d=float(grid.dy),
            dtype=values.dtype,
            device=values.device,
        )
    )[None, None, :, None, None]
    dfdx = torch.fft.ifftn(
        1j * kx * transformed,
        dim=(1, 2),
    ).real
    dfdy = torch.fft.ifftn(
        1j * ky * transformed,
        dim=(1, 2),
    ).real
    return dfdx, dfdy


def _velocity_operators(
    torch,
    values,
    electric,
    *,
    temperature,
    grid: VPFPGrid,
):
    ex = electric[:, 0, :, :, None, None]
    ey = electric[:, 1, :, :, None, None]
    acceleration = torch.zeros_like(values)
    flux_x = ex * 0.5 * (
        values[..., 1:, :] + values[..., :-1, :]
    )
    acceleration[..., 1:-1, :] = (
        flux_x[..., 1:, :] - flux_x[..., :-1, :]
    ) / float(grid.dvx)
    acceleration[..., 0, :] = (
        2.0 * flux_x[..., 0, :] / float(grid.dvx)
    )
    acceleration[..., -1, :] = (
        -2.0 * flux_x[..., -1, :] / float(grid.dvx)
    )
    flux_y = ey * 0.5 * (
        values[..., :, 1:] + values[..., :, :-1]
    )
    acceleration[..., :, 1:-1] += (
        flux_y[..., :, 1:] - flux_y[..., :, :-1]
    ) / float(grid.dvy)
    acceleration[..., :, 0] += (
        2.0 * flux_y[..., :, 0] / float(grid.dvy)
    )
    acceleration[..., :, -1] += (
        -2.0 * flux_y[..., :, -1] / float(grid.dvy)
    )

    vx = torch.linspace(
        float(grid.v_bounds[0]),
        float(grid.v_bounds[1]),
        int(grid.nvx),
        dtype=values.dtype,
        device=values.device,
    )
    vy = torch.linspace(
        float(grid.v_bounds[2]),
        float(grid.v_bounds[3]),
        int(grid.nvy),
        dtype=values.dtype,
        device=values.device,
    )
    face_vx = 0.5 * (vx[1:] + vx[:-1])
    face_vy = 0.5 * (vy[1:] + vy[:-1])
    theta = temperature[:, None, None, None, None]
    delta_x = _torch_chang_cooper_delta(
        torch,
        face_vx[None, None, None, :, None]
        * float(grid.dvx)
        / theta,
    )
    face_state_x = (
        (1.0 - delta_x) * values[..., 1:, :]
        + delta_x * values[..., :-1, :]
    )
    collision_flux_x = (
        face_vx[None, None, None, :, None] * face_state_x
        + theta
        * (values[..., 1:, :] - values[..., :-1, :])
        / float(grid.dvx)
    )
    collision = torch.zeros_like(values)
    collision[..., 1:-1, :] = (
        collision_flux_x[..., 1:, :]
        - collision_flux_x[..., :-1, :]
    ) / float(grid.dvx)
    collision[..., 0, :] = (
        2.0 * collision_flux_x[..., 0, :] / float(grid.dvx)
    )
    collision[..., -1, :] = (
        -2.0 * collision_flux_x[..., -1, :] / float(grid.dvx)
    )
    delta_y = _torch_chang_cooper_delta(
        torch,
        face_vy[None, None, None, None, :]
        * float(grid.dvy)
        / theta,
    )
    face_state_y = (
        (1.0 - delta_y) * values[..., :, 1:]
        + delta_y * values[..., :, :-1]
    )
    collision_flux_y = (
        face_vy[None, None, None, None, :] * face_state_y
        + theta
        * (values[..., :, 1:] - values[..., :, :-1])
        / float(grid.dvy)
    )
    collision[..., :, 1:-1] += (
        collision_flux_y[..., :, 1:]
        - collision_flux_y[..., :, :-1]
    ) / float(grid.dvy)
    collision[..., :, 0] += (
        2.0 * collision_flux_y[..., :, 0] / float(grid.dvy)
    )
    collision[..., :, -1] += (
        -2.0 * collision_flux_y[..., :, -1] / float(grid.dvy)
    )
    return acceleration, collision


def _torch_chang_cooper_delta(torch, peclet):
    small = torch.abs(peclet) < 1.0e-4
    safe = torch.where(small, torch.ones_like(peclet), peclet)
    direct = 1.0 / safe - 1.0 / torch.expm1(safe)
    series = 0.5 - peclet / 12.0 + peclet**3 / 720.0
    return torch.where(small, series, direct)


def _interpolate_source(source_frames, *, time_value: float, time: VPFPTime):
    position = (float(time_value) - float(time.t0)) / float(time.dt)
    lower = min(max(int(np.floor(position)), 0), int(time.nt) - 2)
    alpha = min(max(position - float(lower), 0.0), 1.0)
    return torch_lerp(
        source_frames[:, lower],
        source_frames[:, lower + 1],
        alpha,
    )


def torch_lerp(first, second, weight: float):
    return first + float(weight) * (second - first)


def _velocity_integral(torch, values, *, grid: VPFPGrid):
    weights = _trapezoidal_velocity_weights(
        torch,
        grid=grid,
        dtype=values.dtype,
        device=values.device,
    )
    return torch.sum(values * weights, dim=(-2, -1))


def _trapezoidal_velocity_weights(
    torch,
    *,
    grid: VPFPGrid,
    dtype,
    device,
):
    wx = torch.ones(int(grid.nvx), dtype=dtype, device=device)
    wy = torch.ones(int(grid.nvy), dtype=dtype, device=device)
    wx[0] = wx[-1] = 0.5
    wy[0] = wy[-1] = 0.5
    return (
        wx[:, None]
        * wy[None, :]
        * float(grid.dvx)
        * float(grid.dvy)
    )


def _uniform_rho_g(
    torch,
    f,
    *,
    grid: VPFPGrid,
    rho_floor: float,
):
    positive = torch.clamp(f, min=0.0)
    cell_area = float(grid.dvx) * float(grid.dvy)
    rho = torch.clamp(
        torch.sum(positive, dim=(-2, -1)) * cell_area,
        min=float(rho_floor),
    )
    g = positive / rho[..., None, None]
    normalization = torch.sum(g, dim=(-2, -1), keepdim=True) * cell_area
    g = g / torch.clamp(normalization, min=1.0e-12)
    return rho, g


def _stable_step_bound(
    torch,
    *,
    grid: VPFPGrid,
    collision_nu,
    temperature,
    electric_max,
    cfl: float,
):
    vmax_x = max(abs(grid.v_bounds[0]), abs(grid.v_bounds[1]))
    vmax_y = max(abs(grid.v_bounds[2]), abs(grid.v_bounds[3]))
    transport_rate = (
        float(vmax_x) / float(grid.dx)
        + float(vmax_y) / float(grid.dy)
    )
    electric_rate = torch.abs(electric_max) * (
        1.0 / float(grid.dvx) + 1.0 / float(grid.dvy)
    )
    collision_rate = collision_nu * (
        float(vmax_x) / float(grid.dvx)
        + float(vmax_y) / float(grid.dvy)
        + 2.0
        * temperature
        * (
            1.0 / float(grid.dvx) ** 2
            + 1.0 / float(grid.dvy) ** 2
        )
    )
    return float(cfl) / torch.clamp(
        transport_rate + electric_rate + collision_rate,
        min=1.0e-12,
    )


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the VPFP Torch solver") from exc
    return torch



def pack_torch_equation_specs(
    specs: Sequence[dict[str, Any]],
    *,
    dtype: str = "float32",
) -> dict[str, Any]:
    """Pack rich continuous equation specs without materializing grid fields."""

    if not specs:
        raise ValueError("at least one VPFP equation specification is required")
    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be float32 or float64")
    for spec in specs:
        if (
            spec.get("schema") != "vpfp_continuous_equation_spec_v3"
            or spec.get("family") != RICH_EQUATION_FAMILY
        ):
            raise ValueError(
                "on-device equation evaluation requires the rich v3 family"
            )

    started = walltime.perf_counter()
    numpy_dtype = np.float32 if dtype == "float32" else np.float64
    batch = len(specs)
    rho_count = max(len(spec["rho_modes"]) for spec in specs)
    component_count = max(len(spec["g0_components"]) for spec in specs)
    component_mode_count = max(
        len(component[key])
        for spec in specs
        for component in spec["g0_components"]
        for key in (
            "weight_modes",
            "mean_x_modes",
            "mean_y_modes",
            "log_std_x_modes",
            "log_std_y_modes",
        )
    )
    source_count = max(len(spec["source_modes"]) for spec in specs)

    rho_modes = np.zeros((batch, rho_count, 4), dtype=numpy_dtype)
    rho_mask = np.zeros((batch, rho_count), dtype=bool)
    component_base_logit = np.zeros(
        (batch, component_count), dtype=numpy_dtype
    )
    component_base_mean = np.zeros(
        (batch, component_count, 2), dtype=numpy_dtype
    )
    component_base_log_std = np.zeros_like(component_base_mean)
    component_correlation = np.zeros(
        (batch, component_count), dtype=numpy_dtype
    )
    component_mask = np.zeros((batch, component_count), dtype=bool)
    component_modes = np.zeros(
        (batch, component_count, 5, component_mode_count, 4),
        dtype=numpy_dtype,
    )
    component_mode_mask = np.zeros(
        (batch, component_count, 5, component_mode_count), dtype=bool
    )
    source_modes = np.zeros((batch, source_count, 15), dtype=numpy_dtype)
    source_mask = np.zeros((batch, source_count), dtype=bool)
    component_mode_keys = (
        "weight_modes",
        "mean_x_modes",
        "mean_y_modes",
        "log_std_x_modes",
        "log_std_y_modes",
    )
    maximum_velocity_degree = 0
    for case_index, spec in enumerate(specs):
        for mode_index, mode in enumerate(spec["rho_modes"]):
            rho_modes[case_index, mode_index] = _pack_spatial_mode(mode)
            rho_mask[case_index, mode_index] = True
        for component_index, component in enumerate(spec["g0_components"]):
            component_base_logit[case_index, component_index] = float(
                component["base_logit"]
            )
            component_base_mean[case_index, component_index] = component[
                "base_mean"
            ]
            component_base_log_std[case_index, component_index] = component[
                "base_log_std"
            ]
            component_correlation[case_index, component_index] = float(
                component["correlation"]
            )
            component_mask[case_index, component_index] = True
            for family_index, key in enumerate(component_mode_keys):
                for mode_index, mode in enumerate(component[key]):
                    component_modes[
                        case_index,
                        component_index,
                        family_index,
                        mode_index,
                    ] = _pack_spatial_mode(mode)
                    component_mode_mask[
                        case_index,
                        component_index,
                        family_index,
                        mode_index,
                    ] = True
        for mode_index, mode in enumerate(spec["source_modes"]):
            maximum_velocity_degree = max(
                maximum_velocity_degree,
                int(mode["velocity_degree"][0]),
                int(mode["velocity_degree"][1]),
            )
            source_modes[case_index, mode_index] = np.asarray(
                (
                    float(mode["amplitude"]),
                    float(mode["kx"]),
                    float(mode["ky"]),
                    float(mode["omega"]),
                    float(mode["phase"]),
                    float(mode["velocity_center"][0]),
                    float(mode["velocity_center"][1]),
                    float(mode["velocity_std"][0]),
                    float(mode["velocity_std"][1]),
                    float(mode["velocity_correlation"]),
                    float(mode["velocity_degree"][0]),
                    float(mode["velocity_degree"][1]),
                    float(mode["velocity_wave"][0]),
                    float(mode["velocity_wave"][1]),
                    float(mode["velocity_phase"]),
                ),
                dtype=numpy_dtype,
            )
            source_mask[case_index, mode_index] = True

    return {
        "schema": "vpfp_torch_equation_parameter_batch_v1",
        "dtype": dtype,
        "rho_modes": rho_modes,
        "rho_mask": rho_mask,
        "rho_log_partition": np.asarray(
            [
                0.0
                if spec.get("rho_log_partition") is None
                else float(spec["rho_log_partition"])
                for spec in specs
            ],
            dtype=numpy_dtype,
        ),
        "rho_log_partition_mask": np.asarray(
            [spec.get("rho_log_partition") is not None for spec in specs],
            dtype=bool,
        ),
        "rho_normalization_quadrature_order": int(
            specs[0]["rho_normalization_quadrature_order"]
        ),
        "background_density": np.asarray(
            [float(spec["background_density"]) for spec in specs],
            dtype=numpy_dtype,
        ),
        "component_base_logit": component_base_logit,
        "component_base_mean": component_base_mean,
        "component_base_log_std": component_base_log_std,
        "component_correlation": component_correlation,
        "component_mask": component_mask,
        "component_modes": component_modes,
        "component_mode_mask": component_mode_mask,
        "source_modes": source_modes,
        "source_mask": source_mask,
        "source_velocity_degree_max": int(maximum_velocity_degree),
        "collision_nu": np.asarray(
            [float(spec["collision_nu"]) for spec in specs],
            dtype=numpy_dtype,
        ),
        "temperature": np.asarray(
            [float(spec["temperature"]) for spec in specs],
            dtype=numpy_dtype,
        ),
        "parameter_pack_wall_time_seconds": walltime.perf_counter() - started,
    }


def evaluate_torch_equation_parameter_batch(
    parameters: dict[str, Any],
    *,
    grid: VPFPGrid,
    time: VPFPTime,
    device: str | Any = "cpu",
) -> dict[str, Any]:
    """Materialize rich initial data and source frames directly with Torch."""

    torch = _torch()
    if parameters.get("schema") != "vpfp_torch_equation_parameter_batch_v1":
        raise ValueError("unexpected Torch equation parameter schema")
    grid.validate()
    time.validate()
    dtype_name = str(parameters["dtype"])
    if dtype_name not in {"float32", "float64"}:
        raise ValueError("packed equation dtype must be float32 or float64")
    torch_dtype = torch.float32 if dtype_name == "float32" else torch.float64
    target_device = torch.device(device)
    started = walltime.perf_counter()

    def tensor(key: str, *, boolean: bool = False):
        return torch.as_tensor(
            parameters[key],
            dtype=torch.bool if boolean else torch_dtype,
            device=target_device,
        )

    rho_modes = tensor("rho_modes")
    rho_mask = tensor("rho_mask", boolean=True)
    x = (
        torch.arange(grid.nx, dtype=torch_dtype, device=target_device)
        * float(grid.dx)
    )
    y = (
        torch.arange(grid.ny, dtype=torch_dtype, device=target_device)
        * float(grid.dy)
    )
    x_phase = 2.0 * torch.pi * x / float(grid.x_domain[0])
    y_phase = 2.0 * torch.pi * y / float(grid.x_domain[1])
    rho_angle = (
        rho_modes[..., 1, None, None] * x_phase[None, None, :, None]
        + rho_modes[..., 2, None, None] * y_phase[None, None, None, :]
        + rho_modes[..., 3, None, None]
    )
    log_rho = torch.sum(
        rho_modes[..., 0, None, None]
        * torch.cos(rho_angle)
        * rho_mask[..., None, None],
        dim=1,
    )
    rho_log_partition = tensor("rho_log_partition")
    partition_mask = tensor("rho_log_partition_mask", boolean=True)
    if not bool(torch.all(partition_mask).item()):
        computed_partition = _torch_rho_log_partition(
            torch,
            rho_modes,
            rho_mask,
            order=int(parameters["rho_normalization_quadrature_order"]),
        )
        rho_log_partition = torch.where(
            partition_mask, rho_log_partition, computed_partition
        )
    rho0 = tensor("background_density")[:, None, None] * torch.exp(
        log_rho - rho_log_partition[:, None, None]
    )

    component_fields = _torch_spatial_mode_fields(
        torch,
        tensor("component_modes"),
        tensor("component_mode_mask", boolean=True),
        x_phase=x_phase,
        y_phase=y_phase,
    )
    component_mask = tensor("component_mask", boolean=True)
    logits = tensor("component_base_logit")[..., None, None] + component_fields[
        :, :, 0
    ]
    logits = logits.masked_fill(~component_mask[..., None, None], -torch.inf)
    mixture_weights = torch.softmax(logits, dim=1)
    base_mean = tensor("component_base_mean")
    base_log_std = tensor("component_base_log_std")
    mean_x = base_mean[..., 0, None, None] + component_fields[:, :, 1]
    mean_y = base_mean[..., 1, None, None] + component_fields[:, :, 2]
    std_x = torch.exp(
        base_log_std[..., 0, None, None] + component_fields[:, :, 3]
    )
    std_y = torch.exp(
        base_log_std[..., 1, None, None] + component_fields[:, :, 4]
    )
    correlations = tensor("component_correlation")
    vx = torch.linspace(
        float(grid.v_bounds[0]),
        float(grid.v_bounds[1]),
        int(grid.nvx),
        dtype=torch_dtype,
        device=target_device,
    )
    vy = torch.linspace(
        float(grid.v_bounds[2]),
        float(grid.v_bounds[3]),
        int(grid.nvy),
        dtype=torch_dtype,
        device=target_device,
    )
    g0 = torch.zeros(
        (rho0.shape[0], grid.nx, grid.ny, grid.nvx, grid.nvy),
        dtype=torch_dtype,
        device=target_device,
    )
    for component_index in range(int(component_mask.shape[1])):
        atom = _torch_anisotropic_gaussian(
            torch,
            vx[None, None, None, :, None],
            vy[None, None, None, None, :],
            mean_x=mean_x[:, component_index, :, :, None, None],
            mean_y=mean_y[:, component_index, :, :, None, None],
            std_x=std_x[:, component_index, :, :, None, None],
            std_y=std_y[:, component_index, :, :, None, None],
            correlation=correlations[:, component_index, None, None, None, None],
        )
        g0 += mixture_weights[:, component_index, :, :, None, None] * atom
    f0 = rho0[..., None, None] * g0

    output_times = torch.linspace(
        float(time.t0),
        float(time.t1),
        int(time.nt),
        dtype=torch_dtype,
        device=target_device,
    )
    source_parameters = tensor("source_modes")
    source_mask = tensor("source_mask", boolean=True)
    source = torch.zeros(
        (rho0.shape[0], time.nt, grid.nx, grid.ny, grid.nvx, grid.nvy),
        dtype=torch_dtype,
        device=target_device,
    )
    maximum_velocity_degree = int(parameters["source_velocity_degree_max"])
    for mode_index in range(int(source_mask.shape[1])):
        mode = source_parameters[:, mode_index]
        angle = (
            mode[:, 1, None, None, None] * x_phase[None, None, :, None]
            + mode[:, 2, None, None, None] * y_phase[None, None, None, :]
            + mode[:, 3, None, None, None]
            * output_times[None, :, None, None]
            + mode[:, 4, None, None, None]
        )
        center_x = mode[:, 5, None, None]
        center_y = mode[:, 6, None, None]
        std_mode_x = mode[:, 7, None, None]
        std_mode_y = mode[:, 8, None, None]
        velocity_x = vx[None, :, None]
        velocity_y = vy[None, None, :]
        profile = _torch_anisotropic_gaussian(
            torch,
            velocity_x,
            velocity_y,
            mean_x=center_x,
            mean_y=center_y,
            std_x=std_mode_x,
            std_y=std_mode_y,
            correlation=mode[:, 9, None, None],
        )
        normalized_x = (velocity_x - center_x) / std_mode_x
        normalized_y = (velocity_y - center_y) / std_mode_y
        profile *= _torch_normalized_hermite(
            torch,
            normalized_x,
            mode[:, 10],
            maximum_degree=maximum_velocity_degree,
        )
        profile *= _torch_normalized_hermite(
            torch,
            normalized_y,
            mode[:, 11],
            maximum_degree=maximum_velocity_degree,
        )
        profile *= torch.cos(
            mode[:, 12, None, None] * (velocity_x - center_x)
            + mode[:, 13, None, None] * (velocity_y - center_y)
            + mode[:, 14, None, None]
        )
        source += (
            mode[:, 0, None, None, None, None, None]
            * source_mask[:, mode_index, None, None, None, None, None]
            * torch.sin(angle)[..., None, None]
            * profile[:, None, None, None, :, :]
        )

    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    evaluation_seconds = walltime.perf_counter() - started
    if not bool(torch.all(torch.isfinite(f0)).item()) or not bool(
        torch.all(torch.isfinite(source)).item()
    ):
        raise FloatingPointError("on-device VPFP equation evaluation is non-finite")
    pack_seconds = float(parameters.get("parameter_pack_wall_time_seconds", 0.0))
    return {
        "schema": "vpfp_torch_prepared_equation_batch_v1",
        "f0": f0,
        "source_frames": source,
        "collision_nu": tensor("collision_nu"),
        "temperature": tensor("temperature"),
        "parameter_pack_wall_time_seconds": pack_seconds,
        "equation_evaluation_wall_time_seconds": float(evaluation_seconds),
        "preparation_wall_time_seconds": pack_seconds + evaluation_seconds,
        "preparation_backend": "torch_on_device_rich_v3",
        "source_interpolation": "linear_output_frames",
    }


def _pack_spatial_mode(mode: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        (
            float(mode["amplitude"]),
            float(mode["kx"]),
            float(mode["ky"]),
            float(mode["phase"]),
        ),
        dtype=np.float64,
    )


def _torch_spatial_mode_fields(
    torch,
    modes,
    mask,
    *,
    x_phase,
    y_phase,
):
    angle = (
        modes[..., 1, None, None]
        * x_phase[None, None, None, None, :, None]
        + modes[..., 2, None, None]
        * y_phase[None, None, None, None, None, :]
        + modes[..., 3, None, None]
    )
    return torch.sum(
        modes[..., 0, None, None]
        * torch.cos(angle)
        * mask[..., None, None],
        dim=3,
    )


def _torch_anisotropic_gaussian(
    torch,
    vx,
    vy,
    *,
    mean_x,
    mean_y,
    std_x,
    std_y,
    correlation,
):
    clipped_correlation = torch.clamp(correlation, -0.95, 0.95)
    safe_std_x = torch.clamp(std_x, min=1.0e-6)
    safe_std_y = torch.clamp(std_y, min=1.0e-6)
    dx = (vx - mean_x) / safe_std_x
    dy = (vy - mean_y) / safe_std_y
    one_minus = torch.clamp(1.0 - clipped_correlation.square(), min=1.0e-8)
    exponent = -0.5 * (
        dx.square() - 2.0 * clipped_correlation * dx * dy + dy.square()
    ) / one_minus
    normalization = (
        2.0
        * torch.pi
        * safe_std_x
        * safe_std_y
        * torch.sqrt(one_minus)
    )
    return torch.exp(exponent) / normalization


def _torch_normalized_hermite(
    torch,
    values,
    degrees,
    *,
    maximum_degree: int,
):
    raw_previous = torch.ones_like(values)
    functions = [raw_previous]
    if int(maximum_degree) >= 1:
        raw_current = values
        functions.append(raw_current)
        factorial = 1.0
        for order in range(1, int(maximum_degree)):
            raw_following = values * raw_current - float(order) * raw_previous
            factorial *= float(order + 1)
            functions.append(raw_following / np.sqrt(factorial))
            raw_previous, raw_current = raw_current, raw_following
    output = torch.zeros_like(values)
    integer_degrees = degrees.to(dtype=torch.int64)
    for degree, function in enumerate(functions):
        output = torch.where(
            integer_degrees[:, None, None] == degree,
            function,
            output,
        )
    return output


def _torch_rho_log_partition(
    torch,
    modes,
    mask,
    *,
    order: int,
):
    if int(order) < 2:
        raise ValueError("density normalization order must be at least two")
    theta = (
        2.0
        * torch.pi
        * torch.arange(order, dtype=modes.dtype, device=modes.device)
        / float(order)
    )
    angle = (
        modes[..., 1, None, None] * theta[None, None, :, None]
        + modes[..., 2, None, None] * theta[None, None, None, :]
        + modes[..., 3, None, None]
    )
    log_density = torch.sum(
        modes[..., 0, None, None]
        * torch.cos(angle)
        * mask[..., None, None],
        dim=1,
    )
    return torch.logsumexp(
        log_density.reshape(log_density.shape[0], -1),
        dim=-1,
    ) - np.log(float(order * order))
