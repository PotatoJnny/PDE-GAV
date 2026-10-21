from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from agent_verifier.vpfp.basis import (
    VPFPBasisConfig,
    project_g_to_basis,
    velocity_basis,
)
from agent_verifier.vpfp.continuous_v2 import STANDARD_VPFP_EQUATION_VERSION
from agent_verifier.vpfp.io import VPFPCase
from agent_verifier.vpfp.numerical import rho_g_from_f
from agent_verifier.vpfp.operators import (
    EPS,
    VPFPGrid,
    VPFPTime,
    f_from_rho_g,
    make_axes,
    poisson_electric_field,
    relative_l2,
)


Array = np.ndarray
RHO_NORMALIZATION_QUADRATURE_ORDER = 256
RICH_EQUATION_FAMILY = "spatially_conditioned_gaussian_packets_v3"
LEGACY_BASIS_EQUATION_FAMILY = "fixed_velocity_basis_v2"


@dataclass(frozen=True)
class VPFPHighFidelityConfig:
    internal_dt_max: float = 0.001
    cfl: float = 0.18
    positivity_floor: float = 1.0e-13
    rho_floor: float = 1.0e-10
    equation_family: str = RICH_EQUATION_FAMILY
    max_spatial_mode: int = 4
    g0_component_range: tuple[int, int] = (4, 10)
    g0_weight_mode_count: int = 3
    g0_parameter_mode_count: int = 2
    velocity_center_range: float = 2.0
    velocity_std_range: tuple[float, float] = (0.65, 1.80)
    velocity_correlation_max: float = 0.60
    source_velocity_degree_max: int = 3
    source_velocity_wave_max: float = 1.5
    rho_mode_count: int = 4
    coefficient_mode_count: int = 2
    source_mode_count: int = 12
    density_log_amplitude: float = 0.10
    coefficient_amplitude: float = 0.80
    source_amplitude_range: tuple[float, float] = (0.0005, 0.005)
    collision_nu_range: tuple[float, float] = (0.02, 0.08)
    temperature_range: tuple[float, float] = (0.8, 1.2)
    background_density: float = 1.0
    gci_safety_factor: float = 1.25
    basis: VPFPBasisConfig = VPFPBasisConfig()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["g0_component_range"] = [
            int(value) for value in self.g0_component_range
        ]
        payload["velocity_std_range"] = [
            float(value) for value in self.velocity_std_range
        ]
        payload["source_amplitude_range"] = [
            float(value) for value in self.source_amplitude_range
        ]
        payload["collision_nu_range"] = [
            float(value) for value in self.collision_nu_range
        ]
        payload["temperature_range"] = [
            float(value) for value in self.temperature_range
        ]
        payload["basis"] = self.basis.to_dict()
        return payload

    @classmethod
    def from_dict(
        cls, data: dict[str, Any] | None
    ) -> "VPFPHighFidelityConfig":
        if not data:
            config = cls()
        else:
            payload = dict(data)
            for name in (
                "velocity_std_range",
                "source_amplitude_range",
                "collision_nu_range",
                "temperature_range",
            ):
                if name in payload:
                    payload[name] = tuple(float(value) for value in payload[name])
            if "g0_component_range" in payload:
                payload["g0_component_range"] = tuple(
                    int(value) for value in payload["g0_component_range"]
                )
            payload["basis"] = VPFPBasisConfig.from_dict(payload.get("basis"))
            config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if min(
            float(self.internal_dt_max),
            float(self.cfl),
            float(self.positivity_floor),
            float(self.rho_floor),
            float(self.background_density),
            float(self.gci_safety_factor),
        ) <= 0.0:
            raise ValueError("VPFP numerical scales and floors must be positive")
        if self.equation_family not in {
            RICH_EQUATION_FAMILY,
            LEGACY_BASIS_EQUATION_FAMILY,
        }:
            raise ValueError(f"unsupported equation family: {self.equation_family}")
        if min(
            int(self.max_spatial_mode),
            int(self.g0_weight_mode_count),
            int(self.g0_parameter_mode_count),
            int(self.rho_mode_count),
            int(self.coefficient_mode_count),
            int(self.source_mode_count),
        ) < 1:
            raise ValueError("VPFP equation mode counts must be positive")
        if not (
            1 <= int(self.g0_component_range[0])
            <= int(self.g0_component_range[1])
        ):
            raise ValueError("g0_component_range must be positive and ordered")
        if not (
            0.0 < float(self.velocity_std_range[0])
            <= float(self.velocity_std_range[1])
        ):
            raise ValueError("velocity_std_range must be positive and ordered")
        if not 0.0 <= float(self.velocity_correlation_max) < 1.0:
            raise ValueError("velocity_correlation_max must lie in [0, 1)")
        for name, values, strictly_positive in (
            ("source_amplitude_range", self.source_amplitude_range, False),
            ("collision_nu_range", self.collision_nu_range, False),
            ("temperature_range", self.temperature_range, True),
        ):
            lower, upper = (float(values[0]), float(values[1]))
            if lower > upper or lower < 0.0 or (strictly_positive and lower <= 0.0):
                qualifier = "positive" if strictly_positive else "nonnegative"
                raise ValueError(f"{name} must be {qualifier} and ordered")


def sample_continuous_equation_spec(
    *,
    seed: int,
    config: VPFPHighFidelityConfig,
    defer_rho_log_partition: bool = False,
) -> dict[str, Any]:
    if config.equation_family == RICH_EQUATION_FAMILY:
        return _sample_rich_equation_spec(
            seed=seed,
            config=config,
            defer_rho_log_partition=bool(defer_rho_log_partition),
        )
    if config.equation_family != LEGACY_BASIS_EQUATION_FAMILY:
        raise ValueError(f"unsupported equation family: {config.equation_family}")
    if defer_rho_log_partition:
        raise ValueError(
            "deferred density normalization is only supported for rich v3 specs"
        )

    rng = np.random.default_rng(int(seed))
    atom_count = velocity_basis(VPFPGrid(), config.basis)[0].shape[0]
    rho_modes = []
    for _ in range(int(config.rho_mode_count)):
        kx, ky = _sample_nonzero_mode(rng, int(config.max_spatial_mode))
        rho_modes.append(
            {
                "amplitude": float(
                    rng.normal(scale=float(config.density_log_amplitude))
                ),
                "kx": kx,
                "ky": ky,
                "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
            }
        )
    coefficient_modes: list[dict[str, Any]] = []
    for atom in range(atom_count):
        for _ in range(int(config.coefficient_mode_count)):
            kx, ky = _sample_nonzero_mode(rng, int(config.max_spatial_mode))
            coefficient_modes.append(
                {
                    "atom": int(atom),
                    "amplitude": float(
                        rng.normal(scale=float(config.coefficient_amplitude))
                    ),
                    "kx": kx,
                    "ky": ky,
                    "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
                }
            )
    source_amplitude = float(rng.uniform(*config.source_amplitude_range))
    source_modes = []
    for _ in range(int(config.source_mode_count)):
        kx, ky = _sample_nonzero_mode(rng, int(config.max_spatial_mode))
        source_modes.append(
            {
                "atom": int(rng.integers(0, atom_count)),
                "amplitude": float(
                    source_amplitude
                    * rng.normal()
                    / np.sqrt(max(1, int(config.source_mode_count)))
                ),
                "kx": kx,
                "ky": ky,
                "omega": float(rng.uniform(0.3, 1.8)),
                "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
            }
        )
    return {
        "schema": "vpfp_continuous_equation_spec_v2",
        "equation_version": STANDARD_VPFP_EQUATION_VERSION,
        "seed": int(seed),
        "rho_modes": rho_modes,
        "coefficient_base_logits": rng.normal(
            scale=0.35, size=atom_count
        ).astype(float).tolist(),
        "coefficient_modes": coefficient_modes,
        "source_modes": source_modes,
        "collision_nu": float(rng.uniform(*config.collision_nu_range)),
        "temperature": float(rng.uniform(*config.temperature_range)),
        "background_density": float(config.background_density),
        "rho_log_partition": _rho_log_partition(rho_modes),
        "rho_normalization_quadrature_order": RHO_NORMALIZATION_QUADRATURE_ORDER,
        "basis": config.basis.to_dict(),
        "boundary": {
            "space": "periodic",
            "velocity": "truncated_zero_total_flux_finite_volume_reference",
        },
        "collision_operator": "nu * div_v(v*f + temperature*grad_v(f))",
    }


def _sample_rich_equation_spec(
    *,
    seed: int,
    config: VPFPHighFidelityConfig,
    defer_rho_log_partition: bool,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    minimum_components, maximum_components = (
        int(config.g0_component_range[0]),
        int(config.g0_component_range[1]),
    )
    if minimum_components < 1 or maximum_components < minimum_components:
        raise ValueError("g0_component_range must be positive and ordered")
    if not 0.0 <= float(config.velocity_correlation_max) < 1.0:
        raise ValueError("velocity_correlation_max must lie in [0, 1)")
    if not 0.0 < float(config.velocity_std_range[0]) <= float(
        config.velocity_std_range[1]
    ):
        raise ValueError("velocity_std_range must be positive and ordered")

    rho_modes = []
    for _ in range(int(config.rho_mode_count)):
        kx, ky = _sample_nonzero_mode(rng, int(config.max_spatial_mode))
        rho_modes.append(
            {
                "amplitude": float(
                    rng.normal(scale=float(config.density_log_amplitude))
                ),
                "kx": kx,
                "ky": ky,
                "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
            }
        )

    component_count = int(
        rng.integers(minimum_components, maximum_components + 1)
    )
    g0_components = []
    log_std_min = float(np.log(config.velocity_std_range[0]))
    log_std_max = float(np.log(config.velocity_std_range[1]))
    for _ in range(component_count):
        base_log_std = rng.uniform(log_std_min, log_std_max, size=2)
        g0_components.append(
            {
                "base_logit": float(rng.normal(scale=0.7)),
                "base_mean": rng.uniform(
                    -float(config.velocity_center_range),
                    float(config.velocity_center_range),
                    size=2,
                )
                .astype(float)
                .tolist(),
                "base_log_std": base_log_std.astype(float).tolist(),
                "correlation": float(
                    rng.uniform(
                        -float(config.velocity_correlation_max),
                        float(config.velocity_correlation_max),
                    )
                ),
                "weight_modes": _sample_scalar_spatial_modes(
                    rng,
                    count=int(config.g0_weight_mode_count),
                    max_mode=int(config.max_spatial_mode),
                    amplitude=0.8,
                ),
                "mean_x_modes": _sample_scalar_spatial_modes(
                    rng,
                    count=int(config.g0_parameter_mode_count),
                    max_mode=int(config.max_spatial_mode),
                    amplitude=0.35,
                ),
                "mean_y_modes": _sample_scalar_spatial_modes(
                    rng,
                    count=int(config.g0_parameter_mode_count),
                    max_mode=int(config.max_spatial_mode),
                    amplitude=0.35,
                ),
                "log_std_x_modes": _sample_scalar_spatial_modes(
                    rng,
                    count=int(config.g0_parameter_mode_count),
                    max_mode=int(config.max_spatial_mode),
                    amplitude=0.12,
                ),
                "log_std_y_modes": _sample_scalar_spatial_modes(
                    rng,
                    count=int(config.g0_parameter_mode_count),
                    max_mode=int(config.max_spatial_mode),
                    amplitude=0.12,
                ),
            }
        )

    source_amplitude = float(rng.uniform(*config.source_amplitude_range))
    source_modes = []
    for _ in range(int(config.source_mode_count)):
        kx, ky = _sample_nonzero_mode(rng, int(config.max_spatial_mode))
        total_degree = int(
            rng.integers(0, int(config.source_velocity_degree_max) + 1)
        )
        degree_x = int(rng.integers(0, total_degree + 1))
        degree_y = total_degree - degree_x
        source_modes.append(
            {
                "amplitude": float(
                    source_amplitude
                    * rng.normal()
                    / np.sqrt(max(1, int(config.source_mode_count)))
                ),
                "kx": kx,
                "ky": ky,
                "omega": float(rng.uniform(0.3, 4.0)),
                "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
                "velocity_center": rng.uniform(
                    -float(config.velocity_center_range),
                    float(config.velocity_center_range),
                    size=2,
                )
                .astype(float)
                .tolist(),
                "velocity_std": np.exp(
                    rng.uniform(log_std_min, log_std_max, size=2)
                )
                .astype(float)
                .tolist(),
                "velocity_correlation": float(
                    rng.uniform(
                        -float(config.velocity_correlation_max),
                        float(config.velocity_correlation_max),
                    )
                ),
                "velocity_degree": [degree_x, degree_y],
                "velocity_wave": rng.uniform(
                    -float(config.source_velocity_wave_max),
                    float(config.source_velocity_wave_max),
                    size=2,
                )
                .astype(float)
                .tolist(),
                "velocity_phase": float(rng.uniform(0.0, 2.0 * np.pi)),
            }
        )
    return {
        "schema": "vpfp_continuous_equation_spec_v3",
        "family": RICH_EQUATION_FAMILY,
        "equation_version": STANDARD_VPFP_EQUATION_VERSION,
        "seed": int(seed),
        "rho_modes": rho_modes,
        "g0_components": g0_components,
        "source_modes": source_modes,
        "collision_nu": float(rng.uniform(*config.collision_nu_range)),
        "temperature": float(rng.uniform(*config.temperature_range)),
        "background_density": float(config.background_density),
        "rho_log_partition": (
            None
            if defer_rho_log_partition
            else _rho_log_partition(rho_modes)
        ),
        "rho_normalization_quadrature_order": RHO_NORMALIZATION_QUADRATURE_ORDER,
        "velocity_representation": (
            "spatially_conditioned_fullspace_anisotropic_gaussian_mixture"
        ),
        "source_representation": (
            "signed_fullspace_gaussian_hermite_wave_packets"
        ),
        "basis_role": "none_in_equation_generation",
        "boundary": {
            "space": "periodic",
            "velocity": "R2_fullspace_with_reference_solver_flux_truncation",
        },
        "collision_operator": "nu * div_v(v*f + temperature*grad_v(f))",
    }


def _sample_scalar_spatial_modes(
    rng: np.random.Generator,
    *,
    count: int,
    max_mode: int,
    amplitude: float,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for _ in range(int(count)):
        kx, ky = _sample_nonzero_mode(rng, int(max_mode))
        rows.append(
            {
                "amplitude": float(
                    rng.normal(scale=float(amplitude) / np.sqrt(max(1, count)))
                ),
                "kx": kx,
                "ky": ky,
                "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
            }
        )
    return rows

def evaluate_continuous_equation(
    spec: dict[str, Any],
    *,
    grid: VPFPGrid,
    time: VPFPTime,
) -> dict[str, Any]:
    axes = make_axes(grid, time)
    rho_factor = _evaluate_rho0(spec, grid)
    g_continuous = _evaluate_g0(spec, grid)
    f0_continuous = f_from_rho_g(rho_factor, g_continuous)
    rho0, g0 = rho_g_from_f(
        f0_continuous, grid
    )
    source = _evaluate_source(spec, grid, np.asarray(axes["t"], dtype=float))
    return {
        "rho0": rho0,
        "g0": g0,
        "f0_continuous": f0_continuous,
        "source": source,
        "collision_nu": float(spec["collision_nu"]),
        "temperature": float(spec["temperature"]),
        "background_density": float(spec["background_density"]),
    }


def solve_high_fidelity_equation(
    spec: dict[str, Any],
    *,
    grid: VPFPGrid,
    time: VPFPTime,
    config: VPFPHighFidelityConfig,
) -> dict[str, Any]:
    equation = evaluate_continuous_equation(spec, grid=grid, time=time)
    axes = make_axes(grid, time)
    output_times = np.asarray(axes["t"], dtype=float)
    f = np.asarray(equation["f0_continuous"], dtype=float).copy()
    rho, g = _rho_g_from_f_trapezoidal(
        f, grid, rho_floor=float(config.rho_floor)
    )
    phi, electric = poisson_electric_field(rho[None], grid)
    f_rows = [f.copy()]
    rho_rows = [rho]
    g_rows = [g]
    phi_rows = [phi[0]]
    electric_rows = [electric[0]]
    repairs = 0
    repair_mass = 0.0
    rhs_evaluations = 0
    max_rhs_rms = 0.0
    min_step = float("inf")
    max_step = 0.0
    stable_bounds: list[float] = []

    for frame in range(time.nt - 1):
        frame_start = float(output_times[frame])
        frame_end = float(output_times[frame + 1])
        stable_dt = _stable_step_bound(
            grid,
            collision_nu=float(spec["collision_nu"]),
            temperature=float(spec["temperature"]),
            cfl=float(config.cfl),
            electric_max=float(np.max(np.abs(electric_rows[-1]))),
        )
        stable_bounds.append(stable_dt)
        step_cap = min(float(config.internal_dt_max), stable_dt)
        substeps = max(1, int(np.ceil((frame_end - frame_start) / max(step_cap, EPS))))
        h = (frame_end - frame_start) / float(substeps)
        min_step = min(min_step, h)
        max_step = max(max_step, h)
        for substep in range(substeps):
            current_time = frame_start + substep * h
            k1 = _high_fidelity_rhs(
                f,
                current_time,
                spec=spec,
                grid=grid,
                rho_floor=float(config.rho_floor),
            )
            k2 = _high_fidelity_rhs(
                f + 0.5 * h * k1,
                current_time + 0.5 * h,
                spec=spec,
                grid=grid,
                rho_floor=float(config.rho_floor),
            )
            k3 = _high_fidelity_rhs(
                f + 0.5 * h * k2,
                current_time + 0.5 * h,
                spec=spec,
                grid=grid,
                rho_floor=float(config.rho_floor),
            )
            k4 = _high_fidelity_rhs(
                f + h * k3,
                current_time + h,
                spec=spec,
                grid=grid,
                rho_floor=float(config.rho_floor),
            )
            rhs_evaluations += 4
            max_rhs_rms = max(
                max_rhs_rms,
                float(
                    max(
                        np.sqrt(np.mean(k1**2)),
                        np.sqrt(np.mean(k2**2)),
                        np.sqrt(np.mean(k3**2)),
                        np.sqrt(np.mean(k4**2)),
                    )
                ),
            )
            next_f = f + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            floor = float(config.positivity_floor)
            repair = np.maximum(floor - next_f, 0.0)
            if np.any(repair > 0.0):
                repairs += 1
                repair_mass += float(
                    np.sum(_velocity_integral(repair, grid)) * grid.dx * grid.dy
                )
            f = np.maximum(next_f, floor)
            if not np.all(np.isfinite(f)):
                raise FloatingPointError("high-fidelity VPFP integration became non-finite")
        rho, g = _rho_g_from_f_trapezoidal(
            f, grid, rho_floor=float(config.rho_floor)
        )
        phi, electric = poisson_electric_field(rho[None], grid)
        f_rows.append(f.copy())
        rho_rows.append(rho)
        g_rows.append(g)
        phi_rows.append(phi[0])
        electric_rows.append(electric[0])

    solution = {
        "f": np.asarray(f_rows, dtype=np.float64),
        "rho": np.asarray(rho_rows, dtype=np.float64),
        "g": np.asarray(g_rows, dtype=np.float64),
        "phi": np.asarray(phi_rows, dtype=np.float64),
        "electric": np.asarray(electric_rows, dtype=np.float64),
    }
    source = np.asarray(equation["source"], dtype=float)
    diagnostics = {
        "solver": "rk4_spectral_x_chang_cooper_ou_fv_v_high_fidelity_v4",
        "equation_version": STANDARD_VPFP_EQUATION_VERSION,
        "collision_operator": "nu * div_v(v*f + temperature*grad_v(f))",
        "temperature": float(spec["temperature"]),
        "rhs_evaluations": int(rhs_evaluations),
        "internal_dt_max_requested": float(config.internal_dt_max),
        "stable_dt_bound": float(min(stable_bounds)),
        "min_internal_step": float(min_step),
        "max_internal_step": float(max_step),
        "positivity_repair_steps": int(repairs),
        "positivity_repair_mass": float(repair_mass),
        "max_rhs_rms": float(max_rhs_rms),
        "mass_balance": _mass_balance_diagnostic(
            solution["rho"],
            source,
            grid=grid,
            time=time,
        ),
        "kinetic": kinetic_diagnostics(
            solution["rho"],
            solution["g"],
            solution["electric"],
            grid=grid,
            velocity_quadrature="trapezoidal",
        ),
    }
    return {**solution, "diagnostics": diagnostics}


def refinement_grid(
    base: VPFPGrid,
    *,
    level: int,
    dimension: str,
) -> VPFPGrid:
    factor = 2 ** int(level)
    if int(level) < 0:
        raise ValueError("refinement level must be nonnegative")
    spatial = dimension in {"joint", "spatial"}
    velocity = dimension in {"joint", "velocity"}
    if dimension not in {"joint", "spatial", "velocity", "temporal"}:
        raise ValueError(f"unsupported refinement dimension: {dimension}")
    return VPFPGrid(
        nx=base.nx * factor if spatial else base.nx,
        ny=base.ny * factor if spatial else base.ny,
        nvx=(base.nvx - 1) * factor + 1 if velocity else base.nvx,
        nvy=(base.nvy - 1) * factor + 1 if velocity else base.nvy,
        x_domain=base.x_domain,
        v_bounds=base.v_bounds,
        dtype=base.dtype,
    )


def expand_velocity_domain_grid(
    base: VPFPGrid,
    *,
    scale: float,
) -> VPFPGrid:
    base.validate()
    requested = float(scale)
    if requested < 1.0:
        raise ValueError("velocity-domain scale must be at least one")
    if abs(requested - 1.0) < 1.0e-12:
        return base
    extra_x = max(
        1, int(np.ceil(0.5 * (requested - 1.0) * (base.nvx - 1)))
    )
    extra_y = max(
        1, int(np.ceil(0.5 * (requested - 1.0) * (base.nvy - 1)))
    )
    return VPFPGrid(
        nx=base.nx,
        ny=base.ny,
        nvx=base.nvx + 2 * extra_x,
        nvy=base.nvy + 2 * extra_y,
        x_domain=base.x_domain,
        v_bounds=(
            base.v_bounds[0] - extra_x * base.dvx,
            base.v_bounds[1] + extra_x * base.dvx,
            base.v_bounds[2] - extra_y * base.dvy,
            base.v_bounds[3] + extra_y * base.dvy,
        ),
        dtype=base.dtype,
    )


def refinement_time(
    base: VPFPTime,
    *,
    level: int,
    dimension: str,
) -> VPFPTime:
    factor = 2 ** int(level)
    temporal = dimension in {"joint", "temporal"}
    return VPFPTime(
        t0=base.t0,
        t1=base.t1,
        nt=(base.nt - 1) * factor + 1 if temporal else base.nt,
    )


def restrict_solution(
    solution: dict[str, Any],
    *,
    fine_grid: VPFPGrid,
    fine_time: VPFPTime,
    target_grid: VPFPGrid,
    target_time: VPFPTime,
) -> dict[str, Array]:
    time_index = _endpoint_nested_indices(fine_time.nt, target_time.nt, "time")
    x_index = _periodic_nested_indices(fine_grid.nx, target_grid.nx, "x")
    y_index = _periodic_nested_indices(fine_grid.ny, target_grid.ny, "y")
    vx_index = _endpoint_nested_indices(fine_grid.nvx, target_grid.nvx, "vx")
    vy_index = _endpoint_nested_indices(fine_grid.nvy, target_grid.nvy, "vy")

    f = _take_axes(
        np.asarray(solution["f"], dtype=float),
        (
            (0, time_index),
            (1, x_index),
            (2, y_index),
            (3, vx_index),
            (4, vy_index),
        ),
    )
    rho, g = rho_g_from_f(
        f, target_grid
    )
    phi, electric = poisson_electric_field(rho, target_grid)
    return {
        "rho": rho,
        "g": g,
        "phi": phi,
        "electric": electric,
        "f": f_from_rho_g(rho, g),
    }


def restrict_macro_moments(
    solution: dict[str, Any],
    *,
    fine_grid: VPFPGrid,
    fine_time: VPFPTime,
    target_grid: VPFPGrid,
    target_time: VPFPTime,
) -> dict[str, Array]:
    """Integrate velocity moments before restricting a nested reference grid."""

    time_index = _endpoint_nested_indices(fine_time.nt, target_time.nt, "time")
    x_index = _periodic_nested_indices(fine_grid.nx, target_grid.nx, "x")
    y_index = _periodic_nested_indices(fine_grid.ny, target_grid.ny, "y")
    f = np.asarray(solution["f"], dtype=float)
    expected = (
        fine_time.nt,
        fine_grid.nx,
        fine_grid.ny,
        fine_grid.nvx,
        fine_grid.nvy,
    )
    if tuple(f.shape) != expected:
        raise ValueError(
            f"reference f has shape {tuple(f.shape)}, expected {expected}"
        )
    axes = make_axes(fine_grid)
    vx = np.asarray(axes["vx"], dtype=float)[None, None, None, :, None]
    vy = np.asarray(axes["vy"], dtype=float)[None, None, None, None, :]
    weights = np.full(
        (fine_grid.nvx, fine_grid.nvy),
        fine_grid.velocity_cell_area,
        dtype=float,
    )
    rho = np.sum(f * weights, axis=(-2, -1))
    momentum = np.stack(
        (
            np.sum(f * vx * weights, axis=(-2, -1)),
            np.sum(f * vy * weights, axis=(-2, -1)),
        ),
        axis=-1,
    )
    second_xx = np.sum(f * vx**2 * weights, axis=(-2, -1))
    second_xy = np.sum(f * vx * vy * weights, axis=(-2, -1))
    second_yy = np.sum(f * vy**2 * weights, axis=(-2, -1))
    raw_second = np.stack((second_xx, second_xy, second_yy), axis=-1)
    kinetic = 0.5 * (second_xx + second_yy)

    restriction = ((0, time_index), (1, x_index), (2, y_index))
    return {
        "rho": _take_axes(rho, restriction),
        "momentum": _take_axes(momentum, restriction),
        "kinetic_energy": _take_axes(kinetic, restriction),
        "raw_second_moment": _take_axes(raw_second, restriction),
    }



def convergence_report(
    levels: list[dict[str, Any]],
    *,
    target_grid: VPFPGrid,
    target_time: VPFPTime,
    dimension: str,
    safety_factor: float = 1.25,
) -> dict[str, Any]:
    if len(levels) < 2:
        raise ValueError("convergence study needs at least two levels")
    projected = [
        restrict_solution(
            row["solution"],
            fine_grid=row["grid"],
            fine_time=row["time"],
            target_grid=target_grid,
            target_time=target_time,
        )
        for row in levels
    ]
    h_values = [
        float(
            row.get(
                "characteristic_spacing",
                _refinement_spacing(row["grid"], row["time"], dimension),
            )
        )
        for row in levels
    ]
    if any(not np.isfinite(value) or value <= 0.0 for value in h_values):
        raise ValueError(f"invalid convergence spacings: {h_values}")
    refinement_ratios = [
        float(h_values[index] / h_values[index + 1])
        for index in range(len(h_values) - 1)
    ]
    if any(value <= 1.0 for value in refinement_ratios):
        raise ValueError(f"convergence levels are not increasingly fine: {h_values}")
    if len(refinement_ratios) > 1 and not np.allclose(
        refinement_ratios, refinement_ratios[-1], rtol=1.0e-6, atol=1.0e-12
    ):
        raise ValueError(f"non-geometric refinement ratios: {refinement_ratios}")
    metrics = {}
    for key in ("f", "rho", "g", "electric"):
        differences = [
            relative_l2(projected[index][key] - projected[index + 1][key], projected[index + 1][key])
            for index in range(len(projected) - 1)
        ]
        block: dict[str, Any] = {
            "successive_relative_differences": [float(value) for value in differences],
            "observed_order": None,
            "richardson_relative_error": None,
            "gci_fine_relative": None,
            "successive_difference_ratio": None,
            "roundoff_limited": bool(differences[-1] <= 1.0e-12),
        }
        if len(differences) >= 2:
            r = float(refinement_ratios[-1])
            observed_order = (
                float(np.log(differences[-2] / differences[-1]) / np.log(r))
                if r > 1.0 and min(differences[-2:]) > 10.0 * np.finfo(np.float64).eps
                else float("nan")
            )
            if np.isfinite(observed_order) and observed_order > 0.0:
                denominator = r**observed_order - 1.0
                richardson = differences[-1] / max(denominator, EPS)
                gci = float(safety_factor) * richardson
                difference_ratio = differences[-2] / max(
                    differences[-1], np.finfo(np.float64).tiny
                )
                block.update(
                    {
                        "observed_order": float(observed_order),
                        "richardson_relative_error": float(richardson),
                        "gci_fine_relative": float(gci),
                        "successive_difference_ratio": float(difference_ratio),
                    }
                )
        metrics[key] = block
    return {
        "schema": "vpfp_grid_convergence_gci_v1",
        "dimension": dimension,
        "level_count": len(levels),
        "refinement_ratios": refinement_ratios,
        "resolutions": [
            {
                "grid": row["grid"].to_dict(),
                "time": row["time"].to_dict(),
                "characteristic_spacing": float(h),
                "solver_diagnostics": row["solution"]["diagnostics"],
            }
            for row, h in zip(levels, h_values, strict=True)
        ],
        "metrics": metrics,
        "safety_factor": float(safety_factor),
        "all_primary_orders_positive": all(
            metrics[key]["observed_order"] is not None
            and np.isfinite(float(metrics[key]["observed_order"]))
            and float(metrics[key]["observed_order"]) > 0.0
            for key in ("f", "rho", "g")
        ),
    }


def make_high_fidelity_case(
    *,
    case_id: str,
    spec: dict[str, Any],
    target_grid: VPFPGrid,
    target_time: VPFPTime,
    reference_grid: VPFPGrid,
    reference_time: VPFPTime,
    reference_solution: dict[str, Any],
    convergence: dict[str, Any],
    config: VPFPHighFidelityConfig,
    macro_output: bool = False,
) -> VPFPCase:
    equation = evaluate_continuous_equation(
        spec,
        grid=target_grid,
        time=target_time,
    )
    hidden = restrict_solution(
        reference_solution,
        fine_grid=reference_grid,
        fine_time=reference_time,
        target_grid=target_grid,
        target_time=target_time,
    )
    macro_truth = (
        restrict_macro_moments(
            reference_solution,
            fine_grid=reference_grid,
            fine_time=reference_time,
            target_grid=target_grid,
            target_time=target_time,
        )
        if macro_output
        else None
    )
    initial_rho_rel = relative_l2(hidden["rho"][0] - equation["rho0"], equation["rho0"])
    initial_g_rel = relative_l2(hidden["g"][0] - equation["g0"], equation["g0"])
    if max(initial_rho_rel, initial_g_rel) > 1.0e-12:
        raise RuntimeError(
            "nested reference initial state does not match the public continuous spec: "
            f"rho={initial_rho_rel}, g={initial_g_rel}"
        )
    is_rich_family = spec.get("schema") == "vpfp_continuous_equation_spec_v3"
    if is_rich_family and macro_output:
        basis_projection_f_rel_l2 = None
        published_coeffs = None
        published_basis = None
    else:
        coeffs, basis_reconstruction, basis_metadata = project_g_to_basis(
            hidden["g"],
            target_grid,
            config.basis,
        )
        hidden_f = f_from_rho_g(hidden["rho"], hidden["g"])
        basis_f = f_from_rho_g(hidden["rho"], basis_reconstruction)
        basis_projection_f_rel_l2 = relative_l2(
            basis_f - hidden_f, hidden_f
        )
        published_coeffs = (
            None if is_rich_family else np.asarray(coeffs, dtype=np.float64)
        )
        published_basis = None if is_rich_family else basis_metadata
    case_family = (
        "rich_fullspace_velocity_standard_ou_vpfp_high_fidelity_d0"
        if is_rich_family
        else "continuous_low_mode_standard_ou_vpfp_high_fidelity_d0"
    )
    source_family = (
        "signed_fullspace_gaussian_hermite_wave_packet_source"
        if is_rich_family
        else "continuous_low_mode_signed_velocity_basis_source"
    )
    representation = (
        "factorized_rho_g_full_velocity_grid"
        if is_rich_family
        else "factorized_rho_g_velocity_basis"
    )
    spec_json = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    signature = hashlib.sha256(spec_json.encode("utf-8")).hexdigest()
    diagnostics = {
        "solver": "standard_ou_high_fidelity_reference_projection_v2",
        "equation_version": STANDARD_VPFP_EQUATION_VERSION,
        "source_origin": "randomized_public_equation",
        "reference_fidelity": "refined_equation_first_numerical_solution",
        "continuous_spec_sha256": signature,
        "initial_rho_projection_rel_l2": float(initial_rho_rel),
        "initial_g_projection_rel_l2": float(initial_g_rel),
        "legacy_basis_projection_f_rel_l2": (
            None
            if basis_projection_f_rel_l2 is None
            else float(basis_projection_f_rel_l2)
        ),
        "equation_family": spec.get("family", LEGACY_BASIS_EQUATION_FAMILY),
        "reference_grid": reference_grid.to_dict(),
        "reference_time": reference_time.to_dict(),
        "reference_solver": reference_solution["diagnostics"],
        "convergence": convergence,
        "solver_output_contract": (
            "vpfp_continuous_macro_output_v1"
            if macro_output
            else "full_state"
        ),
        "macro_truth_method": (
            "reference_velocity_quadrature_before_time_space_restriction"
            if macro_output
            else None
        ),
        "kinetic": kinetic_diagnostics(
            hidden["rho"],
            hidden["g"],
            hidden["electric"],
            grid=target_grid,
        ),
        "g_normalization_max_abs_error": float(
            np.max(
                np.abs(
                    np.sum(hidden["g"], axis=(-2, -1))
                    * target_grid.velocity_cell_area
                    - 1.0
                )
            )
        ),
    }
    return VPFPCase(
        case_id=case_id,
        grid=target_grid,
        time=target_time,
        collision_nu=float(spec["collision_nu"]),
        rho=np.asarray(hidden["rho"], dtype=np.float64),
        g=np.asarray(hidden["g"], dtype=np.float64),
        source=np.asarray(equation["source"], dtype=np.float64),
        phi=np.asarray(hidden["phi"], dtype=np.float64),
        electric=np.asarray(hidden["electric"], dtype=np.float64),
        diagnostics=diagnostics,
        case_spec={
            "schema": "vpfp_standard_ou_equation_first_reference_v1",
            "case_id": case_id,
            "family": case_family,
            "source_origin": "randomized_public_equation",
            "truth_primary": "equation_first_refined_numerical_solution",
            "solver_output_contract": (
                "vpfp_continuous_macro_output_v1"
                if macro_output
                else "full_state"
            ),
            "derivation_method": (
                "continuous equation spec solved on refined grids and projected "
                "to the public target grid"
            ),
            "continuous_equation_spec": spec,
            "continuous_spec_sha256": signature,
        },
        generator_metadata={
            "backend": "rk4_spectral_x_chang_cooper_ou_fv_v_reference_v4",
            "representation": representation,
            "source_origin": "randomized_public_equation",
            "reference_fidelity": "refined_equation_first_numerical_solution",
            "condition_signature": signature,
        },
        basis_coeffs=published_coeffs,
        basis_metadata=published_basis,
        public_metadata={
            "source_family": {
                "family": source_family,
                "continuous_spec_sha256": signature,
            },
            "basis": published_basis,
            "question_overrides": {
                "equation": STANDARD_VPFP_EQUATION_VERSION,
                "equation_version": STANDARD_VPFP_EQUATION_VERSION,
                "task": "predict the numerical reference view of rho(t,x,y) and g(t,x,y,vx,vy) for an equation-first standard-OU VPFP case",
                "temperature": float(spec["temperature"]),
                "background_density": float(spec["background_density"]),
                "collision_operator": "nu * div_v(v*f + temperature*grad_v(f))",
                "domain": {
                    "space": "two-dimensional periodic torus",
                    "velocity": "R^2, numerically truncated only by the reference solver",
                },
                "velocity_boundary": "reference_solver_truncation_with_zero_total_flux; not a physical full-space boundary",
                "normalization": "equation-first numerical reference uses the declared velocity quadrature on its evaluation view",
                "poisson": {
                    "equation": "-Delta_x phi = rho - background_density",
                    "electric_field": "E = -grad_x phi",
                    "x_boundary": "periodic",
                    "implementation": "Fourier multiplier on the spatial torus",
                },
            },
        },
        source_origin="randomized_public_equation",
        macro_truth=macro_truth,
    )


def kinetic_diagnostics(
    rho: Array,
    g: Array,
    electric: Array,
    *,
    grid: VPFPGrid,
    velocity_quadrature: str = "rectangle",
) -> dict[str, Any]:
    axes = make_axes(grid)
    vx = np.asarray(axes["vx"], dtype=float)[:, None]
    vy = np.asarray(axes["vy"], dtype=float)[None, :]
    if velocity_quadrature == "trapezoidal":
        weights = _trapezoidal_velocity_weights(grid)
    elif velocity_quadrature == "rectangle":
        weights = np.full((grid.nvx, grid.nvy), grid.velocity_cell_area)
    else:
        raise ValueError(f"unknown velocity quadrature: {velocity_quadrature}")
    f = f_from_rho_g(rho, g)
    spatial_area = grid.dx * grid.dy
    mass = np.sum(rho, axis=(-2, -1)) * spatial_area
    momentum_x = (
        np.sum(f * vx * weights, axis=(-4, -3, -2, -1))
        * spatial_area
    )
    momentum_y = (
        np.sum(f * vy * weights, axis=(-4, -3, -2, -1))
        * spatial_area
    )
    kinetic_energy = (
        0.5
        * np.sum(f * (vx**2 + vy**2) * weights, axis=(-4, -3, -2, -1))
        * spatial_area
    )
    electric_energy = (
        0.5
        * np.sum(np.asarray(electric, dtype=float) ** 2, axis=(-3, -2, -1))
        * spatial_area
    )
    entropy = (
        np.sum(
            np.maximum(f, EPS) * np.log(np.maximum(f, EPS)) * weights,
            axis=(-4, -3, -2, -1),
        )
        * spatial_area
    )
    boundary_fraction = 0.80
    vmax_x = max(abs(grid.v_bounds[0]), abs(grid.v_bounds[1]))
    vmax_y = max(abs(grid.v_bounds[2]), abs(grid.v_bounds[3]))
    tail = (np.abs(vx) >= boundary_fraction * vmax_x) | (
        np.abs(vy) >= boundary_fraction * vmax_y
    )
    tail_mass = (
        np.sum(f * tail * weights, axis=(-4, -3, -2, -1))
        * spatial_area
    )
    return {
        "mass": _series_summary(mass),
        "momentum_x": _series_summary(momentum_x),
        "momentum_y": _series_summary(momentum_y),
        "kinetic_energy": _series_summary(kinetic_energy),
        "electric_energy": _series_summary(electric_energy),
        "entropy_f_log_f": _series_summary(entropy),
        "velocity_tail_mass": _series_summary(tail_mass),
        "velocity_tail_boundary_fraction": boundary_fraction,
    }


def velocity_tail_mass_fraction(kinetic: dict[str, Any]) -> float:
    """Return the worst velocity-tail mass relative to the least total mass."""

    tail_mass_max = float(kinetic["velocity_tail_mass"]["max"])
    total_mass_min = abs(float(kinetic["mass"]["min"]))
    value = tail_mass_max / max(total_mass_min, 1.0e-12)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("velocity-tail mass fraction must be finite and nonnegative")
    return float(value)


def _high_fidelity_rhs(
    f: Array,
    time_value: float,
    *,
    spec: dict[str, Any],
    grid: VPFPGrid,
    rho_floor: float,
) -> Array:
    stage_f = np.asarray(f, dtype=float)
    rho = _velocity_integral(stage_f, grid)
    if np.any(rho <= float(rho_floor)):
        raise FloatingPointError("RK4 stage has nonpositive density")
    _phi, electric = poisson_electric_field(rho[None], grid)
    dfdx, dfdy = _spectral_spatial_gradient(f, grid)
    acceleration, collision = _conservative_standard_ou_velocity_operators(
        f,
        electric[0],
        grid,
        temperature=float(spec["temperature"]),
    )
    axes = make_axes(grid)
    vx = np.asarray(axes["vx"], dtype=float)[None, None, :, None]
    vy = np.asarray(axes["vy"], dtype=float)[None, None, None, :]
    source = _evaluate_source(
        spec,
        grid,
        np.asarray([float(time_value)], dtype=float),
    )[0]
    return (
        source
        - vx * dfdx
        - vy * dfdy
        - acceleration
        + float(spec["collision_nu"]) * collision
    )


def _spectral_spatial_gradient(values: Array, grid: VPFPGrid) -> tuple[Array, Array]:
    transformed = np.fft.fftn(values, axes=(0, 1))
    kx = 2.0 * np.pi * np.fft.fftfreq(grid.nx, d=grid.dx)[:, None, None, None]
    ky = 2.0 * np.pi * np.fft.fftfreq(grid.ny, d=grid.dy)[None, :, None, None]
    dfdx = np.fft.ifftn(1j * kx * transformed, axes=(0, 1)).real
    dfdy = np.fft.ifftn(1j * ky * transformed, axes=(0, 1)).real
    return dfdx, dfdy


def _conservative_velocity_operators(
    values: Array,
    electric: Array,
    grid: VPFPGrid,
) -> tuple[Array, Array]:
    arr = np.asarray(values, dtype=float)
    field = np.asarray(electric, dtype=float)
    ex = field[0][..., None, None]
    ey = field[1][..., None, None]
    acceleration = np.zeros_like(arr)
    flux_x = ex * 0.5 * (
        arr[..., 1:, :] + arr[..., :-1, :]
    )
    acceleration[..., 1:-1, :] = (
        flux_x[..., 1:, :] - flux_x[..., :-1, :]
    ) / grid.dvx
    acceleration[..., 0, :] = 2.0 * flux_x[..., 0, :] / grid.dvx
    acceleration[..., -1, :] = -2.0 * flux_x[..., -1, :] / grid.dvx
    flux_y = ey * 0.5 * (
        arr[..., :, 1:] + arr[..., :, :-1]
    )
    acceleration[..., :, 1:-1] += (
        flux_y[..., :, 1:] - flux_y[..., :, :-1]
    ) / grid.dvy
    acceleration[..., :, 0] += 2.0 * flux_y[..., :, 0] / grid.dvy
    acceleration[..., :, -1] += -2.0 * flux_y[..., :, -1] / grid.dvy

    laplacian = np.zeros_like(arr)
    grad_x = (arr[..., 1:, :] - arr[..., :-1, :]) / grid.dvx
    laplacian[..., 1:-1, :] = (
        grad_x[..., 1:, :] - grad_x[..., :-1, :]
    ) / grid.dvx
    laplacian[..., 0, :] = 2.0 * grad_x[..., 0, :] / grid.dvx
    laplacian[..., -1, :] = -2.0 * grad_x[..., -1, :] / grid.dvx
    grad_y = (arr[..., :, 1:] - arr[..., :, :-1]) / grid.dvy
    laplacian[..., :, 1:-1] += (
        grad_y[..., :, 1:] - grad_y[..., :, :-1]
    ) / grid.dvy
    laplacian[..., :, 0] += 2.0 * grad_y[..., :, 0] / grid.dvy
    laplacian[..., :, -1] += -2.0 * grad_y[..., :, -1] / grid.dvy
    return acceleration, laplacian


def _conservative_standard_ou_velocity_operators(
    values: Array,
    electric: Array,
    grid: VPFPGrid,
    *,
    temperature: float,
) -> tuple[Array, Array]:
    """Return conservative electric acceleration and standard OU collision."""

    if float(temperature) <= 0.0:
        raise ValueError("OU temperature must be positive")
    arr = np.asarray(values, dtype=float)
    acceleration, _laplacian = _conservative_velocity_operators(
        arr, electric, grid
    )
    vx = np.linspace(
        grid.v_bounds[0], grid.v_bounds[1], grid.nvx, dtype=float
    )
    vy = np.linspace(
        grid.v_bounds[2], grid.v_bounds[3], grid.nvy, dtype=float
    )
    face_vx = 0.5 * (vx[1:] + vx[:-1])
    face_vy = 0.5 * (vy[1:] + vy[:-1])
    delta_x = _chang_cooper_delta(
        face_vx * float(grid.dvx) / float(temperature)
    )
    delta_y = _chang_cooper_delta(
        face_vy * float(grid.dvy) / float(temperature)
    )
    face_state_x = (
        (1.0 - delta_x[None, None, :, None]) * arr[..., 1:, :]
        + delta_x[None, None, :, None] * arr[..., :-1, :]
    )
    flux_x = (
        face_vx[None, None, :, None] * face_state_x
        + float(temperature)
        * (arr[..., 1:, :] - arr[..., :-1, :])
        / grid.dvx
    )
    collision = np.zeros_like(arr)
    collision[..., 1:-1, :] = (
        flux_x[..., 1:, :] - flux_x[..., :-1, :]
    ) / grid.dvx
    collision[..., 0, :] = 2.0 * flux_x[..., 0, :] / grid.dvx
    collision[..., -1, :] = -2.0 * flux_x[..., -1, :] / grid.dvx
    face_state_y = (
        (1.0 - delta_y[None, None, None, :]) * arr[..., :, 1:]
        + delta_y[None, None, None, :] * arr[..., :, :-1]
    )
    flux_y = (
        face_vy[None, None, None, :] * face_state_y
        + float(temperature)
        * (arr[..., :, 1:] - arr[..., :, :-1])
        / grid.dvy
    )
    collision[..., :, 1:-1] += (
        flux_y[..., :, 1:] - flux_y[..., :, :-1]
    ) / grid.dvy
    collision[..., :, 0] += 2.0 * flux_y[..., :, 0] / grid.dvy
    collision[..., :, -1] += -2.0 * flux_y[..., :, -1] / grid.dvy
    return acceleration, collision


def _chang_cooper_delta(peclet: Array) -> Array:
    """Return the equilibrium-preserving Chang-Cooper face weight."""

    values = np.asarray(peclet, dtype=float)
    small = np.abs(values) < 1.0e-4
    safe = np.where(small, 1.0, values)
    direct = 1.0 / safe - 1.0 / np.expm1(safe)
    series = 0.5 - values / 12.0 + values**3 / 720.0
    return np.where(small, series, direct)


def _trapezoidal_velocity_weights(grid: VPFPGrid) -> Array:
    wx = np.ones(grid.nvx, dtype=float)
    wy = np.ones(grid.nvy, dtype=float)
    wx[[0, -1]] = 0.5
    wy[[0, -1]] = 0.5
    return wx[:, None] * wy[None, :] * grid.dvx * grid.dvy


def _velocity_integral(values: Array, grid: VPFPGrid) -> Array:
    weights = _trapezoidal_velocity_weights(grid)
    return np.sum(np.asarray(values, dtype=float) * weights, axis=(-2, -1))


def _rho_g_from_f_trapezoidal(
    f: Array,
    grid: VPFPGrid,
    *,
    rho_floor: float,
) -> tuple[Array, Array]:
    f_arr = np.maximum(np.asarray(f, dtype=float), 0.0)
    rho = _velocity_integral(f_arr, grid)
    rho = np.maximum(rho, float(rho_floor))
    g = f_arr / rho[..., None, None]
    normalization = _velocity_integral(g, grid)
    g = g / np.maximum(normalization[..., None, None], EPS)
    return rho.astype(float), g.astype(float)


def _rho_log_partition(
    modes: list[dict[str, Any]],
    *,
    order: int = RHO_NORMALIZATION_QUADRATURE_ORDER,
) -> float:
    if int(order) < 2:
        raise ValueError("rho normalization quadrature order must be at least two")
    theta = 2.0 * np.pi * np.arange(int(order), dtype=float) / float(order)
    theta_x = theta[:, None]
    theta_y = theta[None, :]
    log_density = np.zeros((int(order), int(order)), dtype=float)
    for mode in modes:
        angle = (
            int(mode["kx"]) * theta_x
            + int(mode["ky"]) * theta_y
            + float(mode["phase"])
        )
        log_density += float(mode["amplitude"]) * np.cos(angle)
    maximum = float(np.max(log_density))
    log_partition = maximum + float(
        np.log(np.mean(np.exp(log_density - maximum)))
    )
    if not np.isfinite(log_partition):
        raise ValueError("rho log-partition is non-finite")
    return log_partition


def _evaluate_rho0(spec: dict[str, Any], grid: VPFPGrid) -> Array:
    axes = make_axes(grid)
    x = np.asarray(axes["x"], dtype=float)[:, None]
    y = np.asarray(axes["y"], dtype=float)[None, :]
    log_rho = np.zeros((grid.nx, grid.ny), dtype=float)
    for mode in spec["rho_modes"]:
        angle = (
            2.0 * np.pi * int(mode["kx"]) * x / float(grid.x_domain[0])
            + 2.0 * np.pi * int(mode["ky"]) * y / float(grid.x_domain[1])
            + float(mode["phase"])
        )
        log_rho += float(mode["amplitude"]) * np.cos(angle)
    background = float(spec["background_density"])
    raw_log_partition = spec.get("rho_log_partition")
    log_partition = (
        _rho_log_partition(spec["rho_modes"])
        if raw_log_partition is None
        else float(raw_log_partition)
    )
    if background <= 0.0 or not np.isfinite(log_partition):
        raise ValueError("invalid density background or log-partition")
    return background * np.exp(log_rho - log_partition)


def _continuous_velocity_atoms(spec: dict[str, Any], grid: VPFPGrid) -> Array:
    cfg = VPFPBasisConfig.from_dict(spec.get("basis"))
    axes = make_axes(grid)
    vx = np.asarray(axes["vx"], dtype=float)[:, None]
    vy = np.asarray(axes["vy"], dtype=float)[None, :]
    atoms: list[Array] = []
    for sigma in cfg.thermal_speeds:
        atoms.append(
            _full_space_maxwellian(vx, vy, ux=0.0, uy=0.0, sigma=sigma)
        )
    for stream in cfg.stream_speeds:
        sigma = max(0.35, float(cfg.stream_sigma_scale) * float(stream))
        plus = _full_space_maxwellian(vx, vy, ux=stream, uy=0.0, sigma=sigma)
        minus = _full_space_maxwellian(vx, vy, ux=-stream, uy=0.0, sigma=sigma)
        atoms.append(0.5 * plus + 0.5 * minus)
        if cfg.include_shifted_streams:
            atoms.extend((plus, minus))
        if cfg.include_y_streams:
            plus_y = _full_space_maxwellian(vx, vy, ux=0.0, uy=stream, sigma=sigma)
            minus_y = _full_space_maxwellian(vx, vy, ux=0.0, uy=-stream, sigma=sigma)
            atoms.append(0.5 * plus_y + 0.5 * minus_y)
            if cfg.include_shifted_streams:
                atoms.extend((plus_y, minus_y))
    return np.asarray(atoms, dtype=float)


def _full_space_maxwellian(
    vx: Array,
    vy: Array,
    *,
    ux: float,
    uy: float,
    sigma: float,
) -> Array:
    variance = max(float(sigma) ** 2, EPS)
    return np.exp(
        -0.5 * (((vx - float(ux)) ** 2 + (vy - float(uy)) ** 2) / variance)
    ) / (2.0 * np.pi * variance)


def _evaluate_rich_g0(spec: dict[str, Any], grid: VPFPGrid) -> Array:
    axes = make_axes(grid)
    x = np.asarray(axes["x"], dtype=float)[:, None]
    y = np.asarray(axes["y"], dtype=float)[None, :]
    vx = np.asarray(axes["vx"], dtype=float)[None, None, :, None]
    vy = np.asarray(axes["vy"], dtype=float)[None, None, None, :]
    logits = []
    atoms = []
    for component in spec["g0_components"]:
        logit = float(component["base_logit"]) + _spatial_mode_field(
            component["weight_modes"], x=x, y=y, grid=grid
        )
        mean_x = float(component["base_mean"][0]) + _spatial_mode_field(
            component["mean_x_modes"], x=x, y=y, grid=grid
        )
        mean_y = float(component["base_mean"][1]) + _spatial_mode_field(
            component["mean_y_modes"], x=x, y=y, grid=grid
        )
        log_std_x = float(component["base_log_std"][0]) + _spatial_mode_field(
            component["log_std_x_modes"], x=x, y=y, grid=grid
        )
        log_std_y = float(component["base_log_std"][1]) + _spatial_mode_field(
            component["log_std_y_modes"], x=x, y=y, grid=grid
        )
        std_x = np.exp(log_std_x)
        std_y = np.exp(log_std_y)
        correlation = float(component["correlation"])
        atoms.append(
            _anisotropic_gaussian(
                vx,
                vy,
                mean_x=mean_x[..., None, None],
                mean_y=mean_y[..., None, None],
                std_x=std_x[..., None, None],
                std_y=std_y[..., None, None],
                correlation=correlation,
            )
        )
        logits.append(logit)
    logits_array = np.stack(logits, axis=-1)
    logits_array -= np.max(logits_array, axis=-1, keepdims=True)
    weights = np.exp(logits_array)
    weights /= np.sum(weights, axis=-1, keepdims=True)
    atoms_array = np.stack(atoms, axis=2)
    return np.sum(weights[:, :, :, None, None] * atoms_array, axis=2)


def _evaluate_rich_source(
    spec: dict[str, Any],
    grid: VPFPGrid,
    times: Array,
) -> Array:
    axes = make_axes(grid)
    x = np.asarray(axes["x"], dtype=float)[None, :, None]
    y = np.asarray(axes["y"], dtype=float)[None, None, :]
    t = np.asarray(times, dtype=float)[:, None, None]
    vx = np.asarray(axes["vx"], dtype=float)[:, None]
    vy = np.asarray(axes["vy"], dtype=float)[None, :]
    source = np.zeros(
        (len(times), grid.nx, grid.ny, grid.nvx, grid.nvy), dtype=float
    )
    for mode in spec["source_modes"]:
        angle = (
            2.0 * np.pi * int(mode["kx"]) * x / float(grid.x_domain[0])
            + 2.0 * np.pi * int(mode["ky"]) * y / float(grid.x_domain[1])
            + float(mode["omega"]) * t
            + float(mode["phase"])
        )
        center_x, center_y = (float(value) for value in mode["velocity_center"])
        std_x, std_y = (float(value) for value in mode["velocity_std"])
        correlation = float(mode["velocity_correlation"])
        envelope = _anisotropic_gaussian(
            vx,
            vy,
            mean_x=center_x,
            mean_y=center_y,
            std_x=std_x,
            std_y=std_y,
            correlation=correlation,
        )
        normalized_x = (vx - center_x) / std_x
        normalized_y = (vy - center_y) / std_y
        degree_x, degree_y = (int(value) for value in mode["velocity_degree"])
        polynomial = _normalized_hermite_array(
            normalized_x, degree_x
        ) * _normalized_hermite_array(normalized_y, degree_y)
        wave_x, wave_y = (float(value) for value in mode["velocity_wave"])
        oscillation = np.cos(
            wave_x * (vx - center_x)
            + wave_y * (vy - center_y)
            + float(mode["velocity_phase"])
        )
        profile = envelope * polynomial * oscillation
        source += (
            float(mode["amplitude"])
            * np.sin(angle)[..., None, None]
            * profile[None, None, None, :, :]
        )
    return source


def _spatial_mode_field(
    modes: list[dict[str, Any]],
    *,
    x: Array,
    y: Array,
    grid: VPFPGrid,
) -> Array:
    output = np.zeros((grid.nx, grid.ny), dtype=float)
    for mode in modes:
        angle = (
            2.0 * np.pi * int(mode["kx"]) * x / float(grid.x_domain[0])
            + 2.0 * np.pi * int(mode["ky"]) * y / float(grid.x_domain[1])
            + float(mode["phase"])
        )
        output += float(mode["amplitude"]) * np.cos(angle)
    return output


def _anisotropic_gaussian(
    vx: Array,
    vy: Array,
    *,
    mean_x: Array | float,
    mean_y: Array | float,
    std_x: Array | float,
    std_y: Array | float,
    correlation: float,
) -> Array:
    rho = float(np.clip(correlation, -0.95, 0.95))
    sx = np.maximum(np.asarray(std_x, dtype=float), 1.0e-6)
    sy = np.maximum(np.asarray(std_y, dtype=float), 1.0e-6)
    dx = (vx - mean_x) / sx
    dy = (vy - mean_y) / sy
    one_minus = max(1.0 - rho**2, 1.0e-8)
    exponent = -0.5 * (dx**2 - 2.0 * rho * dx * dy + dy**2) / one_minus
    normalization = 2.0 * np.pi * sx * sy * np.sqrt(one_minus)
    return np.exp(exponent) / normalization


def _normalized_hermite_array(values: Array, degree: int) -> Array:
    if int(degree) == 0:
        return np.ones_like(values)
    previous = np.ones_like(values)
    current = np.asarray(values, dtype=float)
    if int(degree) == 1:
        return current
    normalization = 1.0
    for order in range(1, int(degree)):
        following = values * current - float(order) * previous
        normalization *= float(order + 1)
        previous, current = current, following
    return current / np.sqrt(normalization)

def _evaluate_g0(spec: dict[str, Any], grid: VPFPGrid) -> Array:
    if spec.get("schema") == "vpfp_continuous_equation_spec_v3":
        return _evaluate_rich_g0(spec, grid)
    axes = make_axes(grid)
    x = np.asarray(axes["x"], dtype=float)[:, None]
    y = np.asarray(axes["y"], dtype=float)[None, :]
    atoms = _continuous_velocity_atoms(spec, grid)
    logits = np.broadcast_to(
        np.asarray(spec["coefficient_base_logits"], dtype=float),
        (grid.nx, grid.ny, atoms.shape[0]),
    ).copy()
    for mode in spec["coefficient_modes"]:
        angle = (
            2.0 * np.pi * int(mode["kx"]) * x / float(grid.x_domain[0])
            + 2.0 * np.pi * int(mode["ky"]) * y / float(grid.x_domain[1])
            + float(mode["phase"])
        )
        logits[..., int(mode["atom"])] += float(mode["amplitude"]) * np.cos(angle)
    logits -= np.max(logits, axis=-1, keepdims=True)
    coefficients = np.exp(logits)
    coefficients /= np.sum(coefficients, axis=-1, keepdims=True)
    return np.einsum("xyk,kvw->xyvw", coefficients, atoms)


def _evaluate_source(
    spec: dict[str, Any],
    grid: VPFPGrid,
    times: Array,
) -> Array:
    if spec.get("schema") == "vpfp_continuous_equation_spec_v3":
        return _evaluate_rich_source(spec, grid, times)
    axes = make_axes(grid)
    x = np.asarray(axes["x"], dtype=float)[None, :, None]
    y = np.asarray(axes["y"], dtype=float)[None, None, :]
    t = np.asarray(times, dtype=float)[:, None, None]
    atoms = _continuous_velocity_atoms(spec, grid)
    coefficients = np.zeros(
        (len(times), grid.nx, grid.ny, atoms.shape[0]),
        dtype=float,
    )
    for mode in spec["source_modes"]:
        angle = (
            2.0 * np.pi * int(mode["kx"]) * x / float(grid.x_domain[0])
            + 2.0 * np.pi * int(mode["ky"]) * y / float(grid.x_domain[1])
            + float(mode["omega"]) * t
            + float(mode["phase"])
        )
        coefficients[..., int(mode["atom"])] += float(mode["amplitude"]) * np.sin(
            angle
        )
    return np.einsum("txyk,kvw->txyvw", coefficients, atoms)


def _stable_step_bound(
    grid: VPFPGrid,
    *,
    collision_nu: float,
    temperature: float,
    cfl: float,
    electric_max: float = 0.0,
) -> float:
    vmax_x = max(abs(grid.v_bounds[0]), abs(grid.v_bounds[1]))
    vmax_y = max(abs(grid.v_bounds[2]), abs(grid.v_bounds[3]))
    transport_rate = vmax_x / grid.dx + vmax_y / grid.dy
    electric_rate = abs(float(electric_max)) * (1.0 / grid.dvx + 1.0 / grid.dvy)
    collision_rate = float(collision_nu) * (
        vmax_x / grid.dvx
        + vmax_y / grid.dvy
        + 2.0
        * float(temperature)
        * (1.0 / grid.dvx**2 + 1.0 / grid.dvy**2)
    )
    return float(cfl) / max(transport_rate + electric_rate + collision_rate, EPS)


def _mass_balance_diagnostic(
    rho: Array,
    source: Array,
    *,
    grid: VPFPGrid,
    time: VPFPTime,
) -> dict[str, float]:
    spatial_area = grid.dx * grid.dy
    mass = np.sum(rho, axis=(-2, -1)) * spatial_area
    source_density = _velocity_integral(source, grid)
    source_rate = np.sum(source_density, axis=(-2, -1)) * spatial_area
    axes = make_axes(grid, time)
    expected_delta = float(np.trapz(source_rate, np.asarray(axes["t"], dtype=float)))
    observed_delta = float(mass[-1] - mass[0])
    return {
        "initial_mass": float(mass[0]),
        "final_mass": float(mass[-1]),
        "observed_mass_change": observed_delta,
        "source_integrated_mass_change": expected_delta,
        "absolute_balance_error": abs(observed_delta - expected_delta),
        "relative_balance_error": abs(observed_delta - expected_delta)
        / max(abs(expected_delta), abs(observed_delta), EPS),
        "normalized_balance_error": abs(observed_delta - expected_delta)
        / max(abs(float(mass[0])), EPS),
    }


def _take_axes(values: Array, axis_indices: tuple[tuple[int, Array], ...]) -> Array:
    result = np.asarray(values)
    for axis, indices in axis_indices:
        result = np.take(result, indices, axis=axis)
    return result


def _periodic_nested_indices(fine: int, coarse: int, label: str) -> Array:
    if fine % coarse != 0:
        raise ValueError(f"{label} grids are not periodic-nested: {fine}, {coarse}")
    return np.arange(coarse, dtype=int) * (fine // coarse)


def _endpoint_nested_indices(fine: int, coarse: int, label: str) -> Array:
    if (fine - 1) % (coarse - 1) != 0:
        raise ValueError(f"{label} grids are not endpoint-nested: {fine}, {coarse}")
    return np.arange(coarse, dtype=int) * ((fine - 1) // (coarse - 1))


def _refinement_spacing(
    grid: VPFPGrid,
    time: VPFPTime,
    dimension: str,
) -> float:
    if dimension == "spatial":
        return max(grid.dx, grid.dy)
    if dimension == "velocity":
        return max(grid.dvx, grid.dvy)
    if dimension == "temporal":
        return time.dt
    if dimension == "joint":
        return float((grid.dx * grid.dy * grid.dvx * grid.dvy * time.dt) ** 0.2)
    raise ValueError(dimension)


def _sample_nonzero_mode(
    rng: np.random.Generator,
    max_mode: int,
) -> tuple[int, int]:
    while True:
        kx = int(rng.integers(0, max(1, int(max_mode) + 1)))
        ky = int(rng.integers(0, max(1, int(max_mode) + 1)))
        if kx != 0 or ky != 0:
            return kx, ky


def _series_summary(values: Array) -> dict[str, Any]:
    data = np.asarray(values, dtype=float)
    return {
        "initial": float(data[0]),
        "final": float(data[-1]),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "absolute_change": float(data[-1] - data[0]),
        "relative_change": float(
            abs(data[-1] - data[0]) / max(abs(float(data[0])), EPS)
        ),
        "values": data.astype(float).tolist(),
    }
