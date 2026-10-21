from __future__ import annotations

import json
import math
import multiprocessing as mp
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from agent_verifier.neural_generator.triplane_dataset import (
    Array,
    GeometryContract,
    KiteBaffleGeometryConfig,
    build_kite_baffle_geometry,
    case_metrics,
    divergence,
    mms_forcing_from_velocity_pressure,
    write_case,
    write_geometry_contract,
)


FAMILIES: tuple[str, ...] = (
    "body_force_wake_acceleration",
    "pressure_gradient_channel_drive",
    "oscillatory_cross_shear",
    "counter_rotating_vortex_pair",
    "wake_reversal_pulse",
    "sidewall_eddy_drive",
    "baffle_impingement_pulse",
    "inlet_region_body_force_jet",
    "rotary_solenoidal_drive",
    "shear_layer_vortex_sling",
)


@dataclass(frozen=True)
class D0NumericalConfig:
    geometry: KiteBaffleGeometryConfig = KiteBaffleGeometryConfig()
    nt: int = 32
    t0: float = 0.0
    t1: float = 0.12
    viscosity: float = 0.01
    internal_dt_max: float = 0.0025
    cfl: float = 0.35
    diffusion_cfl: float = 0.20
    pressure_iterations: int = 80
    pressure_relaxation: float = 0.85
    velocity_clip: float = 8.0
    min_repeat_initial_difficulty: float = 0.03
    max_divergence_rel: float = 5.0e-2
    max_boundary_velocity_rel: float = 1.0e-6
    max_obstacle_velocity_rel: float = 1.0e-6
    max_forcing_obstacle_rel: float = 1.0e-7

    def validate(self) -> None:
        self.geometry.validate()
        if self.nt < 3 or self.t1 <= self.t0:
            raise ValueError("D0 numerical generation needs nt >= 3 and t1 > t0")
        if self.viscosity <= 0.0:
            raise ValueError("viscosity must be positive")
        if self.internal_dt_max <= 0.0:
            raise ValueError("internal_dt_max must be positive")
        if self.pressure_iterations < 1:
            raise ValueError("pressure_iterations must be positive")


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    seed: int
    family: str
    parameters: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "seed": int(self.seed), "family": self.family, "parameters": self.parameters}


@dataclass
class NumericalCase:
    spec: CaseSpec
    t: Array
    velocity: Array
    pressure: Array
    forcing: Array
    diagnostics: dict[str, Any]


def build_d0_release(
    out_dir: str | Path,
    *,
    case_count: int,
    config: D0NumericalConfig,
    seed: int,
    jobs: int = 1,
    start_index: int = 0,
    max_candidates: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    config.validate()
    out = Path(out_dir)
    cases_root = out / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    geometry = build_kite_baffle_geometry(config.geometry)
    write_geometry_contract(out / "geometry_contract.json", geometry)
    target_count = int(case_count)
    if target_count < 0:
        raise ValueError("case_count must be non-negative")
    candidate_limit = int(max_candidates) if max_candidates is not None else max(target_count * 3, target_count + 100)
    if candidate_limit < target_count:
        raise ValueError("max_candidates must be at least case_count")
    existing_rows = _existing_case_rows(cases_root) if resume else []
    rows = list(existing_rows)
    seen_case_ids = {str(row["case_id"]) for row in rows}
    accepted_count = int(sum(1 for row in rows if row.get("status") == "accepted"))
    candidate_offset = 0
    worker_count = max(1, int(jobs))

    def next_specs(batch_count: int) -> list[CaseSpec]:
        nonlocal candidate_offset
        specs: list[CaseSpec] = []
        while len(specs) < batch_count and candidate_offset < candidate_limit:
            candidate_id = start_index + candidate_offset
            candidate_offset += 1
            case_id = f"d0_kite_baffle_{candidate_id:06d}"
            if case_id in seen_case_ids:
                continue
            seen_case_ids.add(case_id)
            specs.append(sample_case_spec(seed + candidate_id, case_id=case_id))
        return specs

    def batch_size() -> int:
        remaining_accepts = target_count - accepted_count
        remaining_candidates = candidate_limit - candidate_offset
        return min(worker_count, remaining_accepts, remaining_candidates)

    if worker_count <= 1:
        while accepted_count < target_count and candidate_offset < candidate_limit:
            specs = next_specs(batch_size())
            if not specs:
                break
            batch_rows = [_generate_and_write_one(cases_root, config, geometry, spec) for spec in specs]
            rows.extend(batch_rows)
            accepted_count += int(sum(1 for row in batch_rows if row["status"] == "accepted"))
    else:
        with mp.get_context("spawn").Pool(processes=worker_count) as pool:
            while accepted_count < target_count and candidate_offset < candidate_limit:
                specs = next_specs(batch_size())
                if not specs:
                    break
                tasks = [(str(cases_root), config, geometry.to_json_dict(), spec) for spec in specs]
                batch_rows = pool.map(_generate_and_write_one_spawn, tasks)
                rows.extend(batch_rows)
                accepted_count += int(sum(1 for row in batch_rows if row["status"] == "accepted"))
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    manifest = {
        "case_count_requested": target_count,
        "case_count_written": accepted_count,
        "candidate_count_attempted": len(rows),
        "candidate_count_existing": len(existing_rows),
        "candidate_count_generated_this_run": len(rows) - len(existing_rows),
        "max_candidates": candidate_limit,
        "resume": bool(resume),
        "start_index": int(start_index),
        "seed": int(seed),
        "config": _config_to_dict(config),
        "families": list(FAMILIES),
        "status_counts": status_counts,
        "cases": rows,
    }
    (out / "d0_generation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _existing_case_rows(cases_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not cases_root.exists():
        return rows
    for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        diagnostics_path = case_dir / "diagnostics.json"
        spec_path = case_dir / "case_spec.json"
        failure_path = case_dir / "generation_failed.json"
        if diagnostics_path.exists() and spec_path.exists():
            diagnostics = json.loads(diagnostics_path.read_text())
            spec = json.loads(spec_path.read_text())
            accepted = bool(diagnostics.get("accepted"))
            rows.append(
                {
                    "case_id": case_dir.name,
                    "case_dir": str(case_dir),
                    "family": str(spec.get("family", "unknown")),
                    "status": "accepted" if accepted else "rejected",
                    "filter_failures": list(diagnostics.get("filter_failures", [])),
                    "source": "existing",
                }
            )
        elif failure_path.exists():
            failure = json.loads(failure_path.read_text())
            rows.append(
                {
                    "case_id": case_dir.name,
                    "case_dir": str(case_dir),
                    "family": str(failure.get("family", "unknown")),
                    "status": "failed",
                    "error": str(failure.get("error", "")),
                    "source": "existing",
                }
            )
    return rows


def sample_case_spec(seed: int, *, case_id: str | None = None) -> CaseSpec:
    rng = np.random.default_rng(int(seed))
    family = str(FAMILIES[int(rng.integers(0, len(FAMILIES)))])
    common = {
        "amplitude": float(rng.uniform(0.8, 2.6)),
        "temporal_frequency": float(rng.uniform(0.7, 4.0)),
        "temporal_phase": float(rng.uniform(0.0, 2.0 * np.pi)),
        "x_width": float(rng.uniform(0.055, 0.20)),
        "y_width": float(rng.uniform(0.055, 0.22)),
        "x_center": float(rng.uniform(0.18, 0.82)),
        "y_center": float(rng.uniform(0.20, 0.80)),
        "secondary_phase": float(rng.uniform(0.0, 2.0 * np.pi)),
        "pulse_time": float(rng.uniform(0.025, 0.095)),
        "pulse_width": float(rng.uniform(0.012, 0.045)),
        "mode_x": float(int(rng.integers(1, 5))),
        "mode_y": float(int(rng.integers(1, 5))),
        "bias": float(rng.uniform(-0.35, 0.35)),
    }
    return CaseSpec(case_id=case_id or f"d0_kite_baffle_{int(seed):08d}", seed=int(seed), family=family, parameters=common)


def generate_numerical_case(config: D0NumericalConfig, geometry: GeometryContract, spec: CaseSpec) -> NumericalCase:
    config.validate()
    rng = np.random.default_rng(spec.seed)
    times = np.linspace(config.t0, config.t1, config.nt, dtype=np.float64)
    velocity = np.zeros((config.nt, 2, config.geometry.nx, config.geometry.ny), dtype=np.float64)
    pressure = np.zeros((config.nt, config.geometry.nx, config.geometry.ny), dtype=np.float64)
    drive_forcing = np.stack([_force_field(geometry, spec, float(t)) for t in times], axis=0)
    u = _initial_velocity(geometry, rng)
    p = np.zeros_like(pressure[0])
    u = _apply_no_slip(u, geometry)
    velocity[0] = u
    pressure[0] = p

    current_t = float(times[0])
    substeps = 0
    min_dt = math.inf
    max_dt = 0.0
    for frame_idx, target_t in enumerate(times[1:], start=1):
        while current_t < float(target_t) - 1.0e-12:
            remaining = float(target_t) - current_t
            speed = max(float(np.max(np.sqrt(np.sum(u**2, axis=0)))), 1.0e-8)
            advective_dt = config.cfl * min(config.geometry.dx, config.geometry.dy) / speed
            diffusive_dt = config.diffusion_cfl * min(config.geometry.dx, config.geometry.dy) ** 2 / config.viscosity
            dt = min(config.internal_dt_max, advective_dt, diffusive_dt, remaining)
            f_mid = _force_field(geometry, spec, current_t + 0.5 * dt)
            u, p = _projection_step(u, p, f_mid, geometry, config, dt)
            current_t += dt
            substeps += 1
            min_dt = min(min_dt, dt)
            max_dt = max(max_dt, dt)
        velocity[frame_idx] = u
        pressure[frame_idx] = p
    public_forcing = mms_forcing_from_velocity_pressure(
        velocity=velocity,
        pressure=pressure,
        t=times,
        viscosity=config.viscosity,
        geometry=geometry,
    )
    diagnostics = {
        "substeps": int(substeps),
        "min_internal_dt": float(0.0 if math.isinf(min_dt) else min_dt),
        "max_internal_dt": float(max_dt),
        "family": spec.family,
        "public_forcing_derivation": "mms_from_hidden_velocity_pressure",
        "numerical_drive_forcing_rms": float(np.sqrt(np.mean(drive_forcing[:, :, geometry.fluid_mask] ** 2))),
    }
    metrics = case_metrics(velocity=velocity, pressure=pressure, forcing=public_forcing, t=times, geometry=geometry)
    diagnostics.update(metrics)
    diagnostics["filter_failures"] = d0_filter_failures(metrics, config)
    diagnostics["accepted"] = not diagnostics["filter_failures"]
    return NumericalCase(spec=spec, t=times, velocity=velocity, pressure=pressure, forcing=public_forcing, diagnostics=diagnostics)


def d0_filter_failures(metrics: dict[str, float], config: D0NumericalConfig) -> list[str]:
    failures = []
    if not all(np.isfinite(float(v)) for v in metrics.values()):
        failures.append("non_finite_metric")
    if metrics.get("repeat_initial_difficulty", 0.0) < config.min_repeat_initial_difficulty:
        failures.append("repeat_initial_difficulty_degenerate")
    if metrics.get("divergence_rel", math.inf) > config.max_divergence_rel:
        failures.append("divergence_rel")
    if metrics.get("outer_boundary_velocity_rel", math.inf) > config.max_boundary_velocity_rel:
        failures.append("outer_boundary_velocity_rel")
    if metrics.get("obstacle_velocity_rel", math.inf) > config.max_obstacle_velocity_rel:
        failures.append("obstacle_velocity_rel")
    if metrics.get("forcing_inside_obstacle_rel", math.inf) > config.max_forcing_obstacle_rel:
        failures.append("forcing_inside_obstacle_rel")
    if metrics.get("velocity_rms", 0.0) <= 1.0e-8:
        failures.append("velocity_trivial")
    if metrics.get("forcing_rms", 0.0) <= 1.0e-8:
        failures.append("forcing_trivial")
    return failures


def _generate_and_write_one(root: Path, config: D0NumericalConfig, geometry: GeometryContract, spec: CaseSpec) -> dict[str, Any]:
    try:
        case = generate_numerical_case(config, geometry, spec)
        case_dir = root / spec.case_id
        write_case(
            case_dir,
            case_id=spec.case_id,
            geometry=geometry,
            t=case.t,
            velocity=case.velocity,
            pressure=case.pressure,
            forcing=case.forcing,
            viscosity=config.viscosity,
            case_spec=spec.to_dict(),
            diagnostics=case.diagnostics,
            case_source="kite_baffle_numerical_projection_d0",
        )
        return {
            "case_id": spec.case_id,
            "case_dir": str(case_dir),
            "family": spec.family,
            "status": "accepted" if case.diagnostics.get("accepted") else "rejected",
            "filter_failures": list(case.diagnostics.get("filter_failures", [])),
        }
    except Exception as exc:  # noqa: BLE001 - failed generated cases must be auditable.
        failure_dir = root / spec.case_id
        failure_dir.mkdir(parents=True, exist_ok=True)
        payload = {"case_id": spec.case_id, "family": spec.family, "error_type": type(exc).__name__, "error": str(exc)}
        (failure_dir / "generation_failed.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return {"case_id": spec.case_id, "case_dir": str(failure_dir), "family": spec.family, "status": "failed", "error": str(exc)}


def _generate_and_write_one_spawn(task: tuple[str, D0NumericalConfig, dict[str, Any], CaseSpec]) -> dict[str, Any]:
    root, config, _geometry_json, spec = task
    geometry = build_kite_baffle_geometry(config.geometry)
    return _generate_and_write_one(Path(root), config, geometry, spec)


def _initial_velocity(geometry: GeometryContract, rng: np.random.Generator) -> Array:
    xx, yy = np.meshgrid(geometry.x, geometry.y, indexing="ij")
    psi = np.zeros_like(xx, dtype=np.float64)
    wall = _wall_factor(xx, yy)
    obstacle = _obstacle_factor(geometry)
    for _ in range(3):
        kx = int(rng.integers(1, 4))
        ky = int(rng.integers(1, 4))
        amp = float(rng.uniform(-0.015, 0.015))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        psi += amp * wall * obstacle * np.sin(np.pi * kx * xx + phase) * np.sin(np.pi * ky * yy)
    u = np.gradient(psi, geometry.y, axis=1, edge_order=2)
    v = -np.gradient(psi, geometry.x, axis=0, edge_order=2)
    return np.stack([u, v], axis=0)


def _force_field(geometry: GeometryContract, spec: CaseSpec, t: float) -> Array:
    family_fn = _FAMILY_DISPATCH.get(spec.family)
    if family_fn is None:
        raise ValueError(f"unknown D0 force family: {spec.family}")
    xx, yy = np.meshgrid(geometry.x, geometry.y, indexing="ij")
    force = family_fn(xx, yy, float(t), spec.parameters)
    force = np.asarray(force, dtype=np.float64)
    if force.shape != (2, geometry.config.nx, geometry.config.ny):
        raise ValueError(f"force family {spec.family} returned shape {force.shape}")
    force[:, ~geometry.fluid_mask] = 0.0
    force[:, geometry.outer_boundary_mask] = 0.0
    return force


def _projection_step(u: Array, p: Array, forcing: Array, geometry: GeometryContract, config: D0NumericalConfig, dt: float) -> tuple[Array, Array]:
    adv = _advection(u, geometry.config.dx, geometry.config.dy)
    lap = np.stack([_laplacian2(u[c], geometry.config.dx, geometry.config.dy) for c in range(2)], axis=0)
    tentative = u + float(dt) * (-adv + config.viscosity * lap + forcing)
    tentative = np.clip(tentative, -config.velocity_clip, config.velocity_clip)
    tentative = _apply_no_slip(tentative, geometry)
    rhs = _divergence2(tentative, geometry.config.dx, geometry.config.dy) / float(dt)
    p_new = _solve_pressure_jacobi(rhs, p, geometry.fluid_mask, geometry.config.dx, geometry.config.dy, config)
    gradp = _pressure_gradient2(p_new, geometry.fluid_mask, geometry.config.dx, geometry.config.dy)
    corrected = tentative - float(dt) * gradp
    corrected = _apply_no_slip(corrected, geometry)
    return corrected, p_new



def project_velocity_to_incompressible(
    velocity: Array,
    geometry: GeometryContract,
    *,
    pressure_iterations: int = 400,
    pressure_relaxation: float = 0.8,
) -> tuple[Array, Array]:
    frames = np.asarray(velocity, dtype=np.float64)
    if frames.ndim != 4 or frames.shape[1] != 2:
        raise ValueError(f"expected velocity shaped (nt,2,nx,ny), got {frames.shape}")
    cfg = D0NumericalConfig(
        geometry=geometry.config,
        pressure_iterations=int(pressure_iterations),
        pressure_relaxation=float(pressure_relaxation),
    )
    projected = np.empty_like(frames, dtype=np.float64)
    correction_pressure = np.empty((frames.shape[0], frames.shape[2], frames.shape[3]), dtype=np.float64)
    zero_pressure = np.zeros((frames.shape[2], frames.shape[3]), dtype=np.float64)
    for idx, frame in enumerate(frames):
        u0 = _apply_no_slip(frame, geometry)
        rhs = _divergence2(u0, geometry.config.dx, geometry.config.dy)
        phi = _solve_pressure_jacobi(rhs, zero_pressure, geometry.fluid_mask, geometry.config.dx, geometry.config.dy, cfg)
        corrected = _apply_no_slip(u0 - _pressure_gradient2(phi, geometry.fluid_mask, geometry.config.dx, geometry.config.dy), geometry)
        projected[idx] = corrected
        correction_pressure[idx] = phi
    return projected, correction_pressure

def _apply_no_slip(u: Array, geometry: GeometryContract) -> Array:
    out = np.asarray(u, dtype=np.float64).copy()
    solid_or_wall = np.asarray(geometry.obstacle_mask | geometry.outer_boundary_mask, dtype=bool)
    out[:, solid_or_wall] = 0.0
    return out


def _advection(u: Array, dx: float, dy: float) -> Array:
    out = np.empty_like(u, dtype=np.float64)
    for c in range(2):
        dudx = _upwind_derivative(u[c], u[0], dx, axis=0)
        dudy = _upwind_derivative(u[c], u[1], dy, axis=1)
        out[c] = u[0] * dudx + u[1] * dudy
    return out


def _upwind_derivative(field: Array, advector: Array, spacing: float, axis: int) -> Array:
    forward = (np.roll(field, -1, axis=axis) - field) / float(spacing)
    backward = (field - np.roll(field, 1, axis=axis)) / float(spacing)
    if axis == 0:
        forward[-1, :] = 0.0
        backward[0, :] = 0.0
    else:
        forward[:, -1] = 0.0
        backward[:, 0] = 0.0
    return np.where(advector >= 0.0, backward, forward)


def _laplacian2(field: Array, dx: float, dy: float) -> Array:
    east = np.roll(field, -1, axis=0)
    west = np.roll(field, 1, axis=0)
    north = np.roll(field, -1, axis=1)
    south = np.roll(field, 1, axis=1)
    east[-1, :] = field[-1, :]
    west[0, :] = field[0, :]
    north[:, -1] = field[:, -1]
    south[:, 0] = field[:, 0]
    return (east - 2.0 * field + west) / dx**2 + (north - 2.0 * field + south) / dy**2


def _divergence2(u: Array, dx: float, dy: float) -> Array:
    return _central_derivative(u[0], dx, axis=0) + _central_derivative(u[1], dy, axis=1)


def _central_derivative(field: Array, spacing: float, axis: int) -> Array:
    fwd = np.roll(field, -1, axis=axis)
    back = np.roll(field, 1, axis=axis)
    if axis == 0:
        fwd[-1, :] = field[-1, :]
        back[0, :] = field[0, :]
    else:
        fwd[:, -1] = field[:, -1]
        back[:, 0] = field[:, 0]
    return (fwd - back) / (2.0 * float(spacing))


def _pressure_gradient2(pressure: Array, fluid_mask: Array, dx: float, dy: float) -> Array:
    east = np.roll(pressure, -1, axis=0)
    west = np.roll(pressure, 1, axis=0)
    north = np.roll(pressure, -1, axis=1)
    south = np.roll(pressure, 1, axis=1)
    me = np.roll(fluid_mask, -1, axis=0)
    mw = np.roll(fluid_mask, 1, axis=0)
    mn = np.roll(fluid_mask, -1, axis=1)
    ms = np.roll(fluid_mask, 1, axis=1)
    east[-1, :] = pressure[-1, :]
    west[0, :] = pressure[0, :]
    north[:, -1] = pressure[:, -1]
    south[:, 0] = pressure[:, 0]
    me[-1, :] = False
    mw[0, :] = False
    mn[:, -1] = False
    ms[:, 0] = False
    east = np.where(me, east, pressure)
    west = np.where(mw, west, pressure)
    north = np.where(mn, north, pressure)
    south = np.where(ms, south, pressure)
    return np.stack([(east - west) / (2.0 * dx), (north - south) / (2.0 * dy)], axis=0)


def _solve_pressure_jacobi(rhs: Array, pressure: Array, fluid_mask: Array, dx: float, dy: float, config: D0NumericalConfig) -> Array:
    p = np.asarray(pressure, dtype=np.float64).copy()
    b = np.asarray(rhs, dtype=np.float64) * fluid_mask
    b = b - float(np.mean(b[fluid_mask]))
    wx = 1.0 / dx**2
    wy = 1.0 / dy**2
    denom = 2.0 * (wx + wy)
    relaxation = float(config.pressure_relaxation)
    for _ in range(config.pressure_iterations):
        east = np.roll(p, -1, axis=0)
        west = np.roll(p, 1, axis=0)
        north = np.roll(p, -1, axis=1)
        south = np.roll(p, 1, axis=1)
        me = np.roll(fluid_mask, -1, axis=0)
        mw = np.roll(fluid_mask, 1, axis=0)
        mn = np.roll(fluid_mask, -1, axis=1)
        ms = np.roll(fluid_mask, 1, axis=1)
        east[-1, :] = p[-1, :]
        west[0, :] = p[0, :]
        north[:, -1] = p[:, -1]
        south[:, 0] = p[:, 0]
        me[-1, :] = False
        mw[0, :] = False
        mn[:, -1] = False
        ms[:, 0] = False
        east = np.where(me, east, p)
        west = np.where(mw, west, p)
        north = np.where(mn, north, p)
        south = np.where(ms, south, p)
        updated = (wx * (east + west) + wy * (north + south) - b) / denom
        p = np.where(fluid_mask, (1.0 - relaxation) * p + relaxation * updated, 0.0)
        p[fluid_mask] -= float(np.mean(p[fluid_mask]))
    return p


def _wall_factor(xx: Array, yy: Array) -> Array:
    return (xx**2) * ((1.0 - xx) ** 2) * (yy**2) * ((1.0 - yy) ** 2)


def _obstacle_factor(geometry: GeometryContract) -> Array:
    sdf = np.asarray(geometry.obstacle_signed_distance, dtype=np.float64)
    scale = 2.5 * max(geometry.config.dx, geometry.config.dy)
    return np.clip(sdf / max(scale, 1.0e-12), 0.0, 1.0) ** 2


def _gaussian(xx: Array, yy: Array, x0: float, y0: float, sx: float, sy: float) -> Array:
    return np.exp(-0.5 * (((xx - x0) / sx) ** 2 + ((yy - y0) / sy) ** 2))


def _temporal(t: float, params: dict[str, float]) -> float:
    return math.sin(2.0 * math.pi * params["temporal_frequency"] * t + params["temporal_phase"])


def _pulse(t: float, params: dict[str, float]) -> float:
    return math.exp(-0.5 * ((t - params["pulse_time"]) / params["pulse_width"]) ** 2)


def _wake_acceleration(xx: Array, yy: Array, t: float, p: dict[str, float]) -> Array:
    g = _gaussian(xx, yy, 0.58, 0.52, p["x_width"] * 1.4, p["y_width"])
    amp = p["amplitude"] * (0.8 + 0.4 * _temporal(t, p))
    return np.stack([amp * g, 0.25 * amp * (yy - 0.52) / max(p["y_width"], 1.0e-6) * g], axis=0)


def _channel_drive(xx: Array, yy: Array, t: float, p: dict[str, float]) -> Array:
    profile = np.sin(np.pi * yy) ** 2
    osc = 1.0 + 0.35 * _temporal(t, p)
    return np.stack([p["amplitude"] * osc * profile, np.zeros_like(xx)], axis=0)


def _cross_shear(xx: Array, yy: Array, t: float, p: dict[str, float]) -> Array:
    shear = np.sin(2.0 * np.pi * p["mode_y"] * yy + p["secondary_phase"])
    envelope = np.sin(np.pi * xx) ** 2
    return np.stack([np.zeros_like(xx), p["amplitude"] * _temporal(t, p) * envelope * shear], axis=0)


def _vortex_pair(xx: Array, yy: Array, t: float, p: dict[str, float]) -> Array:
    x0 = 0.48 + 0.10 * math.sin(2.0 * math.pi * t + p["secondary_phase"])
    g1 = _gaussian(xx, yy, x0, 0.38, p["x_width"], p["y_width"])
    g2 = _gaussian(xx, yy, x0 + 0.12, 0.64, p["x_width"], p["y_width"])
    stream = p["amplitude"] * _temporal(t, p) * (g1 - g2)
    fx = np.gradient(stream, yy[0], axis=1, edge_order=2)
    fy = -np.gradient(stream, xx[:, 0], axis=0, edge_order=2)
    return np.stack([fx, fy], axis=0)


def _wake_reversal(xx: Array, yy: Array, t: float, p: dict[str, float]) -> Array:
    g = _gaussian(xx, yy, 0.62, 0.50, p["x_width"] * 1.8, p["y_width"] * 1.2)
    direction = math.tanh((t - p["pulse_time"]) / max(p["pulse_width"], 1.0e-6))
    return np.stack([-p["amplitude"] * direction * g, 0.18 * p["amplitude"] * _temporal(t, p) * g], axis=0)


def _sidewall_eddy(xx: Array, yy: Array, t: float, p: dict[str, float]) -> Array:
    y0 = 0.10 if p["bias"] >= 0 else 0.90
    g = _gaussian(xx, yy, 0.42, y0, p["x_width"] * 1.4, p["y_width"] * 0.8)
    stream = p["amplitude"] * _temporal(t, p) * g
    fx = np.gradient(stream, yy[0], axis=1, edge_order=2)
    fy = -np.gradient(stream, xx[:, 0], axis=0, edge_order=2)
    return np.stack([fx, fy], axis=0)


def _baffle_impingement(xx: Array, yy: Array, t: float, p: dict[str, float]) -> Array:
    g = _gaussian(xx, yy, 0.70, 0.44, p["x_width"], p["y_width"])
    amp = p["amplitude"] * (0.4 + _pulse(t, p))
    return np.stack([0.5 * amp * g, -amp * (xx - 0.70) / max(p["x_width"], 1.0e-6) * g], axis=0)


def _inlet_jet(xx: Array, yy: Array, t: float, p: dict[str, float]) -> Array:
    y0 = p["y_center"]
    g = _gaussian(xx, yy, 0.12, y0, p["x_width"] * 0.7, p["y_width"])
    return np.stack([p["amplitude"] * (1.0 + 0.25 * _temporal(t, p)) * g, 0.15 * p["amplitude"] * (yy - y0) * g], axis=0)


def _rotary_drive(xx: Array, yy: Array, t: float, p: dict[str, float]) -> Array:
    x0, y0 = p["x_center"], p["y_center"]
    g = _gaussian(xx, yy, x0, y0, p["x_width"], p["y_width"])
    amp = p["amplitude"] * _temporal(t, p)
    return np.stack([-amp * (yy - y0) * g, amp * (xx - x0) * g], axis=0)


def _vortex_sling(xx: Array, yy: Array, t: float, p: dict[str, float]) -> Array:
    center = 0.50 + 0.10 * math.sin(2.0 * math.pi * p["temporal_frequency"] * t + p["secondary_phase"])
    g = _gaussian(xx, yy, 0.50, center, p["x_width"] * 1.6, p["y_width"] * 0.7)
    layer = np.tanh((yy - center) / max(0.4 * p["y_width"], 1.0e-6))
    return np.stack([p["amplitude"] * _pulse(t, p) * g, p["amplitude"] * _temporal(t, p) * layer * g], axis=0)


_FAMILY_DISPATCH: dict[str, Callable[[Array, Array, float, dict[str, float]], Array]] = {
    "body_force_wake_acceleration": _wake_acceleration,
    "pressure_gradient_channel_drive": _channel_drive,
    "oscillatory_cross_shear": _cross_shear,
    "counter_rotating_vortex_pair": _vortex_pair,
    "wake_reversal_pulse": _wake_reversal,
    "sidewall_eddy_drive": _sidewall_eddy,
    "baffle_impingement_pulse": _baffle_impingement,
    "inlet_region_body_force_jet": _inlet_jet,
    "rotary_solenoidal_drive": _rotary_drive,
    "shear_layer_vortex_sling": _vortex_sling,
}


def _config_to_dict(config: D0NumericalConfig) -> dict[str, Any]:
    data = asdict(config)
    data["geometry"] = asdict(config.geometry)
    return data
