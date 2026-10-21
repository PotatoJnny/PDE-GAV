from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from agent_verifier.vpfp.basis import VPFPBasisConfig, basis_projection_metrics, decode_basis_coeffs, project_g_to_basis, velocity_basis
from agent_verifier.vpfp.io import VPFPCase
from agent_verifier.vpfp.operators import (
    EPS,
    VPFPGrid,
    VPFPTime,
    f_from_rho_g,
    make_axes,
    neumann_grad_v,
    neumann_laplacian_v,
    normalize_g,
    poisson_electric_field,
    relative_l2,
)

Array = np.ndarray


@dataclass(frozen=True)
class VPFPNumericalConfig:
    density_amplitude_range: tuple[float, float] = (0.03, 0.16)
    source_amplitude_range: tuple[float, float] = (0.002, 0.018)
    collision_nu_range: tuple[float, float] = (0.02, 0.08)
    max_spatial_mode: int = 3
    source_mode_count: int = 4
    coeff_logit_scale: float = 0.9
    internal_dt_max: float = 0.005
    positivity_floor: float = 1.0e-10
    rho_floor: float = 1.0e-8
    basis: VPFPBasisConfig = VPFPBasisConfig()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["basis"] = self.basis.to_dict()
        for key in ("density_amplitude_range", "source_amplitude_range", "collision_nu_range"):
            data[key] = [float(v) for v in getattr(self, key)]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VPFPNumericalConfig":
        if not data:
            return cls()
        return cls(
            density_amplitude_range=tuple(float(v) for v in data.get("density_amplitude_range", cls.density_amplitude_range)),
            source_amplitude_range=tuple(float(v) for v in data.get("source_amplitude_range", cls.source_amplitude_range)),
            collision_nu_range=tuple(float(v) for v in data.get("collision_nu_range", cls.collision_nu_range)),
            max_spatial_mode=int(data.get("max_spatial_mode", cls.max_spatial_mode)),
            source_mode_count=int(data.get("source_mode_count", cls.source_mode_count)),
            coeff_logit_scale=float(data.get("coeff_logit_scale", cls.coeff_logit_scale)),
            internal_dt_max=float(data.get("internal_dt_max", cls.internal_dt_max)),
            positivity_floor=float(data.get("positivity_floor", cls.positivity_floor)),
            rho_floor=float(data.get("rho_floor", cls.rho_floor)),
            basis=VPFPBasisConfig.from_dict(data.get("basis")),
        )


def make_numerical_d0_case(
    *,
    case_id: str,
    grid: VPFPGrid,
    time: VPFPTime,
    config: VPFPNumericalConfig,
    rng: np.random.Generator,
    generator_metadata: dict[str, Any] | None = None,
) -> VPFPCase:
    equation = sample_public_equation(grid=grid, time=time, config=config, rng=rng)
    solved = solve_public_equation(
        rho0=equation["rho0"],
        g0=equation["g0"],
        source=equation["source"],
        grid=grid,
        time=time,
        collision_nu=equation["collision_nu"],
        internal_dt_max=float(config.internal_dt_max),
        positivity_floor=float(config.positivity_floor),
        rho_floor=float(config.rho_floor),
    )
    coeffs, recon_g, basis_metadata = project_g_to_basis(solved["g"], grid, config.basis)
    projection = basis_projection_metrics(solved["rho"], solved["g"], grid, config.basis)
    diagnostics = {
        **solved["diagnostics"],
        **projection,
        "rho_min": float(np.min(solved["rho"])),
        "rho_max": float(np.max(solved["rho"])),
        "g_min": float(np.min(solved["g"])),
        "g_max": float(np.max(solved["g"])),
        "source_rms": float(np.sqrt(np.mean(equation["source"] ** 2))),
        "basis_recon_g_rms": float(np.sqrt(np.mean(recon_g**2))),
        "source_origin": "randomized_public_equation",
    }
    case_spec = {
        "case_id": case_id,
        "family": "numerical_vpfp_d0_lowmode_source",
        "source_origin": "randomized_public_equation",
        "derivation_method": "public_equation_solved_by_deterministic_coarse_vpfp_grid_solver",
        "hidden_answer_primary": ["rho_true", "g_true"],
        "public_derivation": "rho0, g0, source, coefficients are sampled publicly; hidden truth is a deterministic coarse numerical solve",
        "parameters": equation["source_family"],
    }
    metadata = {
        "backend": "deterministic_coarse_vpfp_grid_solver",
        "representation": "factorized_rho_g_velocity_basis",
        "source_origin": "randomized_public_equation",
        "supports_gpu_generator_extension": True,
        "sample_seed": generator_metadata.get("sample_seed") if generator_metadata else None,
        **dict(generator_metadata or {}),
    }
    return VPFPCase(
        case_id=case_id,
        grid=grid,
        time=time,
        collision_nu=float(equation["collision_nu"]),
        rho=solved["rho"].astype(np.float64),
        g=solved["g"].astype(np.float64),
        source=equation["source"].astype(np.float64),
        phi=solved["phi"].astype(np.float64),
        electric=solved["electric"].astype(np.float64),
        diagnostics=diagnostics,
        case_spec=case_spec,
        generator_metadata=metadata,
        basis_coeffs=coeffs.astype(np.float64),
        basis_metadata=basis_metadata,
        public_metadata={"source_family": equation["source_family"], "basis": basis_metadata},
        source_origin="randomized_public_equation",
    )


def sample_public_equation(*, grid: VPFPGrid, time: VPFPTime, config: VPFPNumericalConfig, rng: np.random.Generator) -> dict[str, Any]:
    axes = make_axes(grid, time)
    atoms, basis_metadata = velocity_basis(grid, config.basis)
    rho0 = _sample_rho0(grid, config, rng)
    coeffs0 = _sample_coeffs0(grid, atoms.shape[0], config, rng)
    g0 = decode_basis_coeffs(coeffs0, atoms)
    source, source_meta = _sample_source(grid, time, atoms, config, rng)
    collision_nu = float(rng.uniform(*config.collision_nu_range))
    return {
        "rho0": rho0,
        "g0": g0,
        "source": source,
        "collision_nu": collision_nu,
        "basis_metadata": basis_metadata,
        "source_family": {
            **source_meta,
            "collision_nu": collision_nu,
            "density_amplitude_range": [float(v) for v in config.density_amplitude_range],
            "source_amplitude_range": [float(v) for v in config.source_amplitude_range],
            "grid_shape": [grid.nx, grid.ny, grid.nvx, grid.nvy],
            "time_values": [float(v) for v in axes["t"]],
        },
    }


def solve_public_equation(
    *,
    rho0: Array,
    g0: Array,
    source: Array,
    grid: VPFPGrid,
    time: VPFPTime,
    collision_nu: float,
    internal_dt_max: float = 0.005,
    positivity_floor: float = 1.0e-10,
    rho_floor: float = 1.0e-8,
) -> dict[str, Any]:
    axes = make_axes(grid, time)
    times = axes["t"]
    source_arr = np.asarray(source, dtype=float)
    expected_source = (time.nt, grid.nx, grid.ny, grid.nvx, grid.nvy)
    if source_arr.shape != expected_source:
        raise ValueError(f"source shape must be {expected_source}, got {source_arr.shape}")
    f = f_from_rho_g(np.asarray(rho0, dtype=float), np.asarray(g0, dtype=float))
    rho, g = rho_g_from_f(f, grid, rho_floor=rho_floor)
    rho_rows = [rho]
    g_rows = [g]
    phi, electric = poisson_electric_field(rho[None, ...], grid)
    phi_rows = [phi[0]]
    electric_rows = [electric[0]]
    repair_mass = 0.0
    repair_steps = 0
    max_rhs_rms = 0.0
    for frame in range(time.nt - 1):
        frame_dt = float(times[frame + 1] - times[frame])
        substeps = max(1, int(np.ceil(frame_dt / max(float(internal_dt_max), EPS))))
        h = frame_dt / float(substeps)
        for sub in range(substeps):
            alpha = float(sub) / float(substeps)
            src = (1.0 - alpha) * source_arr[frame] + alpha * source_arr[frame + 1]
            rho, g = rho_g_from_f(f, grid, rho_floor=rho_floor)
            _phi, electric_now = poisson_electric_field(rho[None, ...], grid)
            rhs = _rhs(f, rho, g, src, electric_now[0], grid=grid, collision_nu=collision_nu)
            max_rhs_rms = max(max_rhs_rms, float(np.sqrt(np.mean(rhs**2))))
            next_f = f + h * rhs
            neg = np.minimum(next_f, 0.0)
            if np.any(neg < 0.0):
                repair_mass += float(np.sum(np.abs(neg)) * grid.velocity_cell_area)
                repair_steps += 1
            f = np.maximum(next_f, float(positivity_floor))
        rho, g = rho_g_from_f(f, grid, rho_floor=rho_floor)
        phi, electric = poisson_electric_field(rho[None, ...], grid)
        rho_rows.append(rho)
        g_rows.append(g)
        phi_rows.append(phi[0])
        electric_rows.append(electric[0])
    rho_out = np.asarray(rho_rows, dtype=float)
    g_out = np.asarray(g_rows, dtype=float)
    g_mass = np.sum(g_out, axis=(-2, -1)) * grid.velocity_cell_area
    f_out = f_from_rho_g(rho_out, g_out)
    diagnostics = {
        "solver": "explicit_upwind_coarse_vpfp_v1",
        "internal_dt_max": float(internal_dt_max),
        "positivity_floor": float(positivity_floor),
        "positivity_repair_mass": float(repair_mass),
        "positivity_repair_steps": int(repair_steps),
        "g_normalization_max_abs_error": float(np.max(np.abs(g_mass - 1.0))),
        "f_rms": float(np.sqrt(np.mean(f_out**2))),
        "rho_rms": float(np.sqrt(np.mean(rho_out**2))),
        "max_rhs_rms": float(max_rhs_rms),
        "repeat_initial_difficulty": _repeat_initial_difficulty(rho_out, g_out),
    }
    return {
        "rho": rho_out,
        "g": g_out,
        "phi": np.asarray(phi_rows, dtype=float),
        "electric": np.asarray(electric_rows, dtype=float),
        "diagnostics": diagnostics,
    }


def rho_g_from_f(f: Array, grid: VPFPGrid, *, rho_floor: float = 1.0e-8) -> tuple[Array, Array]:
    f_arr = np.maximum(np.asarray(f, dtype=float), 0.0)
    rho = np.sum(f_arr, axis=(-2, -1)) * grid.velocity_cell_area
    rho = np.maximum(rho, float(rho_floor))
    g = f_arr / rho[..., None, None]
    g = normalize_g(g, grid)
    return rho.astype(float), g.astype(float)


def _rhs(f: Array, rho: Array, g: Array, source: Array, electric: Array, *, grid: VPFPGrid, collision_nu: float) -> Array:
    axes = make_axes(grid)
    vx = axes["vx"][None, None, :, None]
    vy = axes["vy"][None, None, None, :]
    adv_x = vx * _periodic_upwind(f, vx, axis=0, spacing=grid.dx)
    adv_y = vy * _periodic_upwind(f, vy, axis=1, spacing=grid.dy)
    dg_dvx, dg_dvy = neumann_grad_v(g, grid)
    lap_g = neumann_laplacian_v(g, grid)
    ex = electric[0, :, :][:, :, None, None]
    ey = electric[1, :, :][:, :, None, None]
    rho_v = rho[:, :, None, None]
    return np.asarray(source, dtype=float) - adv_x - adv_y - rho_v * (ex * dg_dvx + ey * dg_dvy) + float(collision_nu) * rho_v * lap_g


def _periodic_upwind(values: Array, speed: Array, *, axis: int, spacing: float) -> Array:
    arr = np.asarray(values, dtype=float)
    backward = (arr - np.roll(arr, 1, axis=axis)) / float(spacing)
    forward = (np.roll(arr, -1, axis=axis) - arr) / float(spacing)
    return np.where(speed >= 0.0, backward, forward)


def _sample_rho0(grid: VPFPGrid, config: VPFPNumericalConfig, rng: np.random.Generator) -> Array:
    axes = make_axes(grid)
    x = axes["x"][:, None]
    y = axes["y"][None, :]
    field = np.zeros((grid.nx, grid.ny), dtype=float)
    for _ in range(4):
        kx = int(rng.integers(1, max(2, int(config.max_spatial_mode) + 1)))
        ky = int(rng.integers(0, max(1, int(config.max_spatial_mode) + 1)))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        amp = float(rng.normal())
        field += amp * np.cos(kx * x + ky * y + phase)
    field = field / max(float(np.sqrt(np.mean(field**2))), EPS)
    density_amp = float(rng.uniform(*config.density_amplitude_range))
    return np.maximum(1.0 + density_amp * field, 0.05).astype(float)


def _sample_coeffs0(grid: VPFPGrid, atom_count: int, config: VPFPNumericalConfig, rng: np.random.Generator) -> Array:
    axes = make_axes(grid)
    x = axes["x"][:, None]
    y = axes["y"][None, :]
    logits = rng.normal(scale=0.35, size=(grid.nx, grid.ny, atom_count))
    for atom in range(atom_count):
        for _ in range(2):
            kx = int(rng.integers(0, max(1, int(config.max_spatial_mode) + 1)))
            ky = int(rng.integers(0, max(1, int(config.max_spatial_mode) + 1)))
            if kx == 0 and ky == 0:
                kx = 1
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            logits[..., atom] += float(config.coeff_logit_scale) * float(rng.normal()) * np.sin(kx * x + ky * y + phase)
    logits -= np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def _sample_source(grid: VPFPGrid, time: VPFPTime, atoms: Array, config: VPFPNumericalConfig, rng: np.random.Generator) -> tuple[Array, dict[str, Any]]:
    axes = make_axes(grid, time)
    t = axes["t"][:, None, None]
    x = axes["x"][None, :, None]
    y = axes["y"][None, None, :]
    atom_count = int(atoms.shape[0])
    coeff = np.zeros((time.nt, grid.nx, grid.ny, atom_count), dtype=float)
    modes = []
    for _ in range(int(config.source_mode_count)):
        atom = int(rng.integers(0, atom_count))
        kx = int(rng.integers(0, max(1, int(config.max_spatial_mode) + 1)))
        ky = int(rng.integers(0, max(1, int(config.max_spatial_mode) + 1)))
        if kx == 0 and ky == 0:
            kx = 1
        omega = float(rng.uniform(0.3, 1.8))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        sign = float(rng.choice([-1.0, 1.0]))
        mode = sign * np.sin(kx * x + ky * y + omega * (t - float(time.t0)) + phase)
        coeff[..., atom] += mode
        modes.append({"atom": atom, "kx": kx, "ky": ky, "omega": omega, "phase": phase, "sign": sign})
    coeff = coeff / max(float(np.sqrt(np.mean(coeff**2))), EPS)
    amplitude = float(rng.uniform(*config.source_amplitude_range))
    source = amplitude * np.einsum("txyk,kvw->txyvw", coeff, atoms)
    return source.astype(float), {
        "family": "low_mode_signed_velocity_basis_source",
        "source_amplitude": amplitude,
        "source_mode_count": int(config.source_mode_count),
        "modes": modes,
    }


def _repeat_initial_difficulty(rho: Array, g: Array) -> float:
    f = f_from_rho_g(rho, g)
    pred = np.repeat(f[:1], f.shape[0], axis=0)
    return relative_l2(pred - f, f)
