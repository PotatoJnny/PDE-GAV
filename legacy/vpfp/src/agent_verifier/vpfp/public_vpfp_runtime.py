from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EPS = 1.0e-12
STANDARD_VPFP_EQUATION_VERSION = "vpfp_ou_poisson_torus_rv_v2"
MACRO_OUTPUT_CONTRACT = "vpfp_continuous_macro_output_v1"
SCALABLE_RESOLUTION_CONTRACT = "vpfp_scalable_resolution_solver_v1"
COMPUTE_BUDGET_CONTRACT = "vpfp_compute_budget_solver_v1"


@dataclass(frozen=True)
class VPFPGrid:
    nx: int
    ny: int
    nvx: int
    nvy: int
    x_domain: tuple[float, float]
    v_bounds: tuple[float, float, float, float]
    dtype: str = "float64"

    @property
    def dx(self) -> float:
        return float(self.x_domain[0]) / float(self.nx)

    @property
    def dy(self) -> float:
        return float(self.x_domain[1]) / float(self.ny)

    @property
    def dvx(self) -> float:
        return float(self.v_bounds[1] - self.v_bounds[0]) / float(max(self.nvx - 1, 1))

    @property
    def dvy(self) -> float:
        return float(self.v_bounds[3] - self.v_bounds[2]) / float(max(self.nvy - 1, 1))

    @property
    def velocity_cell_area(self) -> float:
        return self.dvx * self.dvy


@dataclass(frozen=True)
class VPFPTime:
    t0: float
    t1: float
    nt: int

    @property
    def dt(self) -> float:
        return (float(self.t1) - float(self.t0)) / float(self.nt - 1)


def load_public_case(case_dir: str | Path) -> dict[str, Any]:
    path = Path(case_dir)
    question = json.loads((path / "public_question.json").read_text())
    grid_data = dict(question["grid"])
    grid_data["x_domain"] = tuple(float(value) for value in grid_data["x_domain"])
    grid_data["v_bounds"] = tuple(float(value) for value in grid_data["v_bounds"])
    grid = VPFPGrid(**grid_data)
    time = VPFPTime(**question["time"])
    with np.load(path / "public_arrays.npz") as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    return {
        "case_id": str(question["case_id"]),
        "question": question,
        "arrays": arrays,
        "grid": grid,
        "time": time,
        "collision_nu": float(question["collision_nu"]),
    }


def load_submission(path: str | Path):
    source = Path(path)
    spec = importlib.util.spec_from_file_location("vpfp_submission", source)
    if spec is None or spec.loader is None:
        raise ImportError(source)
    module = importlib.util.module_from_spec(spec)
    inserted = False
    module_dir = str(source.parent.resolve())
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
        inserted = True
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            sys.path.remove(module_dir)
    if not any(
        hasattr(module, name)
        for name in ("solve", "solve_case", "solve_at_resolution", "solve_with_budget")
    ):
        raise AttributeError(
            f"{source} must define solve(case), solve_case(case_dir), "
            "solve_at_resolution(case, resolution), or solve_with_budget(...)"
        )
    return module


def run_submission(module, case_dir: str | Path) -> Any:
    case = load_public_case(case_dir)
    question = case["question"]
    if str(question.get("solver_resolution_contract_version") or "") == (
        SCALABLE_RESOLUTION_CONTRACT
    ):
        if not hasattr(module, "solve_at_resolution"):
            raise AttributeError(
                "scalable-resolution case requires "
                "solve_at_resolution(case, resolution)"
            )
        request = dict(question.get("resolution_request") or {})
        resolution = dict(request.get("resolution") or {})
        return module.solve_at_resolution(case, resolution)
    if hasattr(module, "solve"):
        return module.solve(case)
    return module.solve_case(str(case_dir))


def evaluate_public_output(case: dict[str, Any], output: Any) -> dict[str, Any]:
    case_id = str(case["case_id"])
    if not isinstance(output, dict):
        return _failure(case_id, "output_not_dict")
    if _is_macro_output_contract(case["question"]):
        return _evaluate_macro_public_output(case, output)
    standard_continuous = _is_standard_continuous(case["question"])
    rho_value = output.get("rho", output.get("rho_pred"))
    g_value = output.get("g", output.get("g_pred"))
    coeff_value = output.get("basis_coeffs", output.get("coeffs_pred"))
    if rho_value is None or (g_value is None and coeff_value is None):
        return _failure(case_id, "missing_rho_or_g")
    if standard_continuous and g_value is None:
        return _failure(case_id, "continuous_case_requires_full_g")
    rho = np.asarray(rho_value, dtype=float)
    expected_rho = tuple(int(value) for value in case["question"]["target_shapes"]["rho"])
    expected_g = tuple(int(value) for value in case["question"]["target_shapes"]["g"])
    if tuple(rho.shape) != expected_rho:
        return _failure(case_id, "bad_rho_shape", actual_shape=list(rho.shape), expected_shape=list(expected_rho))
    if g_value is None:
        atoms = velocity_basis(case["grid"], (case["question"].get("basis") or {}).get("config"))
        coeffs = np.asarray(coeff_value, dtype=float)
        expected_coeffs = expected_g[:-2] + (atoms.shape[0],)
        if tuple(coeffs.shape) != expected_coeffs:
            return _failure(
                case_id,
                "bad_basis_coeff_shape",
                actual_shape=list(coeffs.shape),
                expected_shape=list(expected_coeffs),
            )
        g = decode_basis_coeffs(coeffs, atoms)
    else:
        g = np.asarray(g_value, dtype=float)
    if tuple(g.shape) != expected_g:
        return _failure(case_id, "bad_g_shape", actual_shape=list(g.shape), expected_shape=list(expected_g))
    if not np.all(np.isfinite(rho)) or not np.all(np.isfinite(g)):
        return _failure(case_id, "nonfinite")

    arrays = case["arrays"]
    source = np.asarray(arrays["source"], dtype=float)
    weights = np.asarray(arrays["velocity_weights"], dtype=float)
    if standard_continuous:
        residual_value = standard_ou_residual(
            rho,
            g,
            source,
            grid=case["grid"],
            time=case["time"],
            collision_nu=case["collision_nu"],
            temperature=float(case["question"]["temperature"]),
        )
    else:
        residual_value = residual(
            rho,
            g,
            source,
            grid=case["grid"],
            time=case["time"],
            collision_nu=case["collision_nu"],
        )
    g_mass = np.sum(g * weights, axis=(-2, -1))
    rho_initial = relative_l2(rho[0] - arrays["rho0"], arrays["rho0"])
    g_initial = relative_l2(g[0] - arrays["g0"], arrays["g0"])
    residual_rel = relative_l2(residual_value, source)
    rho_negative = relative_l2(np.minimum(rho, 0.0), rho)
    g_negative = relative_l2(np.minimum(g, 0.0), g)
    if standard_continuous:
        objective = residual_rel + rho_initial + g_initial + rho_negative + g_negative
        return {
            "case_id": case_id,
            "status": "ok",
            "equation_version": STANDARD_VPFP_EQUATION_VERSION,
            "rho_shape": list(rho.shape),
            "g_shape": list(g.shape),
            "finite": True,
            "rho_initial_rel_l2": float(rho_initial),
            "g_initial_rel_l2": float(g_initial),
            "vpfp_residual_rel": float(residual_rel),
            "g_normalization_contract": "analytic_fullspace_not_materialized_window",
            "g_window_mass_min": float(np.min(g_mass)),
            "g_window_mass_max": float(np.max(g_mass)),
            "rho_positivity_violation": float(rho_negative),
            "g_positivity_violation": float(g_negative),
            "public_objective_smaller_is_better": float(objective),
        }
    g_norm = float(np.max(np.abs(g_mass - 1.0)))
    objective = residual_rel + rho_initial + g_initial + g_norm + rho_negative + g_negative
    return {
        "case_id": case_id,
        "status": "ok",
        "rho_shape": list(rho.shape),
        "g_shape": list(g.shape),
        "finite": True,
        "rho_initial_rel_l2": float(rho_initial),
        "g_initial_rel_l2": float(g_initial),
        "vpfp_residual_rel": float(residual_rel),
        "g_normalization_max_abs_error": g_norm,
        "rho_positivity_violation": float(rho_negative),
        "g_positivity_violation": float(g_negative),
        "public_objective_smaller_is_better": float(objective),
    }



def _evaluate_macro_public_output(
    case: dict[str, Any], output: dict[str, Any]
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    rho_value = output.get("rho", output.get("rho_pred"))
    momentum_value = output.get("momentum", output.get("j"))
    kinetic_value = output.get("kinetic_energy", output.get("K"))
    if rho_value is None or momentum_value is None or kinetic_value is None:
        return _failure(case_id, "missing_macro_fields")
    rho = np.asarray(rho_value, dtype=float)
    momentum = np.asarray(momentum_value, dtype=float)
    kinetic = np.asarray(kinetic_value, dtype=float)
    target_shapes = case["question"]["target_shapes"]
    expected_rho = tuple(int(value) for value in target_shapes["rho"])
    expected_momentum = tuple(
        int(value) for value in target_shapes["momentum"]
    )
    expected_kinetic = tuple(
        int(value) for value in target_shapes["kinetic_energy"]
    )
    actual_shapes = {
        "rho": list(rho.shape),
        "momentum": list(momentum.shape),
        "kinetic_energy": list(kinetic.shape),
    }
    if (
        rho.shape != expected_rho
        or momentum.shape != expected_momentum
        or kinetic.shape != expected_kinetic
    ):
        return _failure(
            case_id,
            "bad_macro_shape",
            actual_shapes=actual_shapes,
            expected_shapes={
                name: list(value)
                for name, value in (
                    ("rho", expected_rho),
                    ("momentum", expected_momentum),
                    ("kinetic_energy", expected_kinetic),
                )
            },
        )
    if not all(np.all(np.isfinite(value)) for value in (rho, momentum, kinetic)):
        return _failure(case_id, "nonfinite")
    arrays = case["arrays"]
    if "momentum0" not in arrays or "kinetic_energy0" not in arrays:
        return _failure(case_id, "missing_public_initial_macros")
    rho_initial = relative_l2(rho[0] - arrays["rho0"], arrays["rho0"])
    momentum_initial = relative_l2(
        momentum[0] - arrays["momentum0"], arrays["momentum0"]
    )
    kinetic_initial = relative_l2(
        kinetic[0] - arrays["kinetic_energy0"], arrays["kinetic_energy0"]
    )
    rho_negative = relative_l2(np.minimum(rho, 0.0), rho)
    kinetic_negative = relative_l2(np.minimum(kinetic, 0.0), kinetic)
    objective = (
        rho_initial
        + momentum_initial
        + kinetic_initial
        + rho_negative
        + kinetic_negative
    )
    return {
        "case_id": case_id,
        "status": "ok",
        "equation_version": STANDARD_VPFP_EQUATION_VERSION,
        "solver_output_contract_version": MACRO_OUTPUT_CONTRACT,
        "rho_shape": list(rho.shape),
        "momentum_shape": list(momentum.shape),
        "kinetic_energy_shape": list(kinetic.shape),
        "finite": True,
        "rho_initial_rel_l2": float(rho_initial),
        "momentum_initial_rel_l2": float(momentum_initial),
        "kinetic_energy_initial_rel_l2": float(kinetic_initial),
        "rho_positivity_violation": float(rho_negative),
        "kinetic_energy_positivity_violation": float(kinetic_negative),
        "public_objective_smaller_is_better": float(objective),
    }


def _is_macro_output_contract(question: dict[str, Any]) -> bool:
    return str(question.get("solver_output_contract_version") or "") == (
        MACRO_OUTPUT_CONTRACT
    )

def velocity_basis(grid: VPFPGrid, config: dict[str, Any] | None = None) -> np.ndarray:
    cfg = dict(config or {})
    thermal_speeds = tuple(float(value) for value in cfg.get("thermal_speeds", (0.75, 1.0, 1.35)))
    stream_speeds = tuple(float(value) for value in cfg.get("stream_speeds", (1.25, 1.85)))
    sigma_scale = float(cfg.get("stream_sigma_scale", 0.65))
    include_y = bool(cfg.get("include_y_streams", True))
    include_shifted = bool(cfg.get("include_shifted_streams", False))
    vx = np.linspace(grid.v_bounds[0], grid.v_bounds[1], grid.nvx)[:, None]
    vy = np.linspace(grid.v_bounds[2], grid.v_bounds[3], grid.nvy)[None, :]
    atoms: list[np.ndarray] = []
    for sigma in thermal_speeds:
        atoms.append(_normalize_atom(_maxwellian(vx, vy, 0.0, 0.0, sigma), grid))
    for stream in stream_speeds:
        sigma = max(0.35, sigma_scale * stream)
        plus_x = _maxwellian(vx, vy, stream, 0.0, sigma)
        minus_x = _maxwellian(vx, vy, -stream, 0.0, sigma)
        atoms.append(_normalize_atom(0.5 * (plus_x + minus_x), grid))
        if include_shifted:
            atoms.extend((_normalize_atom(plus_x, grid), _normalize_atom(minus_x, grid)))
        if include_y:
            plus_y = _maxwellian(vx, vy, 0.0, stream, sigma)
            minus_y = _maxwellian(vx, vy, 0.0, -stream, sigma)
            atoms.append(_normalize_atom(0.5 * (plus_y + minus_y), grid))
            if include_shifted:
                atoms.extend((_normalize_atom(plus_y, grid), _normalize_atom(minus_y, grid)))
    return np.asarray(atoms, dtype=float)


def decode_basis_coeffs(coeffs: np.ndarray, atoms: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(coeffs, dtype=float), 0.0)
    total = np.sum(values, axis=-1, keepdims=True)
    bad = np.squeeze(total <= EPS, axis=-1)
    if np.any(bad):
        values = values.copy()
        values[bad, :] = 1.0 / float(values.shape[-1])
        total = np.sum(values, axis=-1, keepdims=True)
    values = values / np.maximum(total, EPS)
    return np.einsum("...k,kvw->...vw", values, np.asarray(atoms, dtype=float))


def _is_standard_continuous(question: dict[str, Any]) -> bool:
    return str(question.get("equation_version") or question.get("equation")) == (
        STANDARD_VPFP_EQUATION_VERSION
    )


def standard_ou_residual(
    rho: np.ndarray,
    g: np.ndarray,
    source: np.ndarray,
    *,
    grid: VPFPGrid,
    time: VPFPTime,
    collision_nu: float,
    temperature: float,
) -> np.ndarray:
    """Finite-window diagnostic for an equation posed on full velocity space."""

    f = np.asarray(rho, dtype=float)[..., None, None] * np.asarray(g, dtype=float)
    electric = poisson_electric_field(rho, grid)
    df_dt = np.gradient(f, time.dt, axis=0, edge_order=2 if time.nt >= 3 else 1)
    df_dx = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * grid.dx)
    df_dy = (np.roll(f, -1, axis=2) - np.roll(f, 1, axis=2)) / (2.0 * grid.dy)
    edge_vx = 2 if grid.nvx >= 3 else 1
    edge_vy = 2 if grid.nvy >= 3 else 1
    df_dvx = np.gradient(f, grid.dvx, axis=3, edge_order=edge_vx)
    df_dvy = np.gradient(f, grid.dvy, axis=4, edge_order=edge_vy)
    d2f_dvx2 = np.gradient(df_dvx, grid.dvx, axis=3, edge_order=edge_vx)
    d2f_dvy2 = np.gradient(df_dvy, grid.dvy, axis=4, edge_order=edge_vy)
    vx = np.linspace(grid.v_bounds[0], grid.v_bounds[1], grid.nvx)[None, None, None, :, None]
    vy = np.linspace(grid.v_bounds[2], grid.v_bounds[3], grid.nvy)[None, None, None, None, :]
    field = (
        electric[:, 0, :, :, None, None] * df_dvx
        + electric[:, 1, :, :, None, None] * df_dvy
    )
    collision = 2.0 * f + vx * df_dvx + vy * df_dvy + float(temperature) * (
        d2f_dvx2 + d2f_dvy2
    )
    computed = (
        df_dt
        + vx * df_dx
        + vy * df_dy
        + field
        - float(collision_nu) * collision
    )
    return computed - np.asarray(source, dtype=float)


def residual(
    rho: np.ndarray,
    g: np.ndarray,
    source: np.ndarray,
    *,
    grid: VPFPGrid,
    time: VPFPTime,
    collision_nu: float,
) -> np.ndarray:
    f = rho[..., None, None] * g
    electric = poisson_electric_field(rho, grid)
    df_dt = np.gradient(f, time.dt, axis=0, edge_order=2 if time.nt >= 3 else 1)
    df_dx = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * grid.dx)
    df_dy = (np.roll(f, -1, axis=2) - np.roll(f, 1, axis=2)) / (2.0 * grid.dy)
    dg_dvx = np.zeros_like(g)
    dg_dvy = np.zeros_like(g)
    dg_dvx[..., 1:-1, :] = (g[..., 2:, :] - g[..., :-2, :]) / (2.0 * grid.dvx)
    dg_dvy[..., :, 1:-1] = (g[..., :, 2:] - g[..., :, :-2]) / (2.0 * grid.dvy)
    lap_g = neumann_laplacian_v(g, grid)
    vx = np.linspace(grid.v_bounds[0], grid.v_bounds[1], grid.nvx)[None, None, None, :, None]
    vy = np.linspace(grid.v_bounds[2], grid.v_bounds[3], grid.nvy)[None, None, None, None, :]
    rho_v = rho[..., None, None]
    field_term = rho_v * (
        electric[:, 0, :, :, None, None] * dg_dvx
        + electric[:, 1, :, :, None, None] * dg_dvy
    )
    computed = df_dt + vx * df_dx + vy * df_dy + field_term - float(collision_nu) * rho_v * lap_g
    return computed - np.asarray(source, dtype=float)


def poisson_electric_field(rho: np.ndarray, grid: VPFPGrid) -> np.ndarray:
    centered = rho - np.mean(rho, axis=(-2, -1), keepdims=True)
    rho_hat = np.fft.fftn(centered, axes=(-2, -1))
    kx = 2.0 * np.pi * np.fft.fftfreq(grid.nx, d=grid.dx)[:, None]
    ky = 2.0 * np.pi * np.fft.fftfreq(grid.ny, d=grid.dy)[None, :]
    k2 = kx**2 + ky**2
    phi_hat = np.zeros_like(rho_hat, dtype=complex)
    np.divide(rho_hat, k2, out=phi_hat, where=k2 > 0.0)
    ex = np.fft.ifftn(-1j * kx * phi_hat, axes=(-2, -1)).real
    ey = np.fft.ifftn(-1j * ky * phi_hat, axes=(-2, -1)).real
    return np.stack((ex, ey), axis=1)


def neumann_laplacian_v(values: np.ndarray, grid: VPFPGrid) -> np.ndarray:
    lap = np.zeros_like(values, dtype=float)
    lap[..., 1:-1, :] += (values[..., 2:, :] - 2.0 * values[..., 1:-1, :] + values[..., :-2, :]) / grid.dvx**2
    lap[..., 0, :] += (values[..., 1, :] - values[..., 0, :]) / grid.dvx**2
    lap[..., -1, :] += (values[..., -2, :] - values[..., -1, :]) / grid.dvx**2
    lap[..., :, 1:-1] += (values[..., :, 2:] - 2.0 * values[..., :, 1:-1] + values[..., :, :-2]) / grid.dvy**2
    lap[..., :, 0] += (values[..., :, 1] - values[..., :, 0]) / grid.dvy**2
    lap[..., :, -1] += (values[..., :, -2] - values[..., :, -1]) / grid.dvy**2
    return lap


def relative_l2(numerator: np.ndarray, denominator: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(numerator, dtype=float)) / (np.linalg.norm(np.asarray(denominator, dtype=float)) + EPS))


def _normalize_atom(atom: np.ndarray, grid: VPFPGrid) -> np.ndarray:
    values = np.maximum(np.asarray(atom, dtype=float), 0.0)
    return values / max(float(np.sum(values) * grid.velocity_cell_area), EPS)


def _maxwellian(vx: np.ndarray, vy: np.ndarray, ux: float, uy: float, sigma: float) -> np.ndarray:
    variance = max(float(sigma) ** 2, EPS)
    return np.exp(-0.5 * (((vx - ux) ** 2 + (vy - uy) ** 2) / variance)) / (2.0 * np.pi * variance)


def _failure(case_id: str, status: str, **details: Any) -> dict[str, Any]:
    return {"case_id": case_id, "status": status, "finite": False, **details}
