from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agent_verifier.vpfp.io import VPFPCase
from agent_verifier.vpfp.operators import (
    VPFPGrid,
    VPFPTime,
    compute_source,
    f_from_rho_g,
    make_axes,
    normalize_g,
    relative_l2,
    residual,
)

Array = np.ndarray
EPS = 1.0e-12


@dataclass(frozen=True)
class VPFPManufactureConfig:
    family: str = "landau"
    amplitude_range: tuple[float, float] = (0.03, 0.12)
    drift_range: tuple[float, float] = (0.02, 0.18)
    thermal_speed_range: tuple[float, float] = (0.75, 1.25)
    stream_speed_range: tuple[float, float] = (1.0, 2.0)
    temporal_frequency_range: tuple[float, float] = (0.6, 2.0)
    damping_range: tuple[float, float] = (0.1, 0.6)
    min_rho: float = 0.05


def make_mms_case(
    *,
    case_id: str,
    grid: VPFPGrid,
    time: VPFPTime,
    collision_nu: float,
    config: VPFPManufactureConfig,
    rng: np.random.Generator,
    generator_metadata: dict[str, Any] | None = None,
) -> VPFPCase:
    rho, g, spec = sample_factorized_answer(grid=grid, time=time, config=config, rng=rng)
    source, phi, electric = compute_source(rho, g, grid=grid, time=time, collision_nu=collision_nu)
    res = residual(rho, g, source, grid=grid, time=time, collision_nu=collision_nu)
    f = f_from_rho_g(rho, g)
    g_mass = np.sum(g, axis=(-2, -1)) * grid.velocity_cell_area
    diagnostics = {
        "truth_vpfp_residual_rel": relative_l2(res, source) if np.linalg.norm(source) > EPS else float(np.linalg.norm(res)),
        "rho_min": float(np.min(rho)),
        "rho_max": float(np.max(rho)),
        "g_min": float(np.min(g)),
        "g_max": float(np.max(g)),
        "g_normalization_max_abs_error": float(np.max(np.abs(g_mass - 1.0))),
        "f_rms": float(np.sqrt(np.mean(f**2))),
        "rho_rms": float(np.sqrt(np.mean(rho**2))),
        "source_rms": float(np.sqrt(np.mean(source**2))),
        "electric_rms": float(np.sqrt(np.mean(electric**2))),
        "field_energy_mean": float(0.5 * np.mean(np.sum(electric**2, axis=1))),
        "repeat_initial_difficulty": _repeat_initial_difficulty(rho, g),
        "factorization": "rho_times_conditional_g",
    }
    case_spec = {
        "case_id": case_id,
        "family": spec["family"],
        "parameters": spec,
        "derivation_method": "manufactured_solution_to_public_question",
        "hidden_answer_primary": ["rho_true", "g_true"],
        "public_derivation": "rho and g were sampled first; Poisson field and VPFP source were derived from the hidden factorized answer",
    }
    metadata = {
        "backend": "cpu_mms_factorized",
        "representation": "factorized_rho_g",
        "supports_gpu_generator_extension": True,
        "checkpoint": None,
        "conditioning": spec["family"],
        "sample_seed": generator_metadata.get("sample_seed") if generator_metadata else None,
        "latent_id": generator_metadata.get("latent_id") if generator_metadata else None,
        **dict(generator_metadata or {}),
    }
    return VPFPCase(
        case_id=case_id,
        grid=grid,
        time=time,
        collision_nu=float(collision_nu),
        rho=rho.astype(np.float64),
        g=g.astype(np.float64),
        source=source.astype(np.float64),
        phi=phi.astype(np.float64),
        electric=electric.astype(np.float64),
        diagnostics=diagnostics,
        case_spec=case_spec,
        generator_metadata=metadata,
    )


def sample_factorized_answer(
    *,
    grid: VPFPGrid,
    time: VPFPTime,
    config: VPFPManufactureConfig,
    rng: np.random.Generator,
) -> tuple[Array, Array, dict[str, Any]]:
    axes = make_axes(grid, time)
    t = axes["t"][:, None, None]
    x = axes["x"][None, :, None]
    y = axes["y"][None, None, :]
    vx = axes["vx"][None, None, None, :, None]
    vy = axes["vy"][None, None, None, None, :]
    family = str(config.family)
    if family == "mixed":
        family = "landau" if float(rng.random()) < 0.5 else "two_stream"
    if family not in {"landau", "two_stream"}:
        raise ValueError(f"unsupported VPFP MMS family: {family}")

    amp = float(rng.uniform(*config.amplitude_range))
    drift_amp = float(rng.uniform(*config.drift_range))
    sigma = float(rng.uniform(*config.thermal_speed_range))
    omega = float(rng.uniform(*config.temporal_frequency_range))
    gamma = float(rng.uniform(*config.damping_range))
    kx = int(rng.integers(1, 3))
    ky = int(rng.integers(0, 3))
    if ky == 0 and kx == 0:
        kx = 1
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    theta = kx * x + ky * y - omega * t + phase
    if family == "landau":
        envelope = np.exp(-gamma * (t - float(time.t0)))
        rho = 1.0 + amp * envelope * np.cos(theta)
        ux = drift_amp * envelope * np.sin(theta)
        uy = 0.5 * drift_amp * envelope * np.cos(theta)
        g_raw = _maxwellian(vx, vy, ux[..., None, None], uy[..., None, None], sigma)
    else:
        stream = float(rng.uniform(*config.stream_speed_range))
        envelope = 1.0 + 0.25 * np.sin(omega * t + phase)
        rho = 1.0 + amp * np.cos(theta) * np.cos(omega * t + phase)
        mix = np.clip(0.5 + 0.20 * amp * np.sin(theta), 0.05, 0.95)
        sigma_stream = max(0.35, 0.65 * sigma)
        g_plus = _maxwellian(vx, vy, stream, 0.0, sigma_stream)
        g_minus = _maxwellian(vx, vy, -stream, 0.0, sigma_stream)
        thermal = _maxwellian(vx, vy, drift_amp * np.sin(theta)[..., None, None], 0.0, sigma)
        g_raw = envelope[..., None, None] * (mix[..., None, None] * g_plus + (1.0 - mix[..., None, None]) * g_minus)
        g_raw = 0.9 * g_raw + 0.1 * thermal
    rho = np.maximum(rho, float(config.min_rho))
    g = normalize_g(g_raw, grid)
    spec = {
        "family": family,
        "amplitude": amp,
        "drift_amplitude": drift_amp,
        "thermal_speed": sigma,
        "temporal_frequency": omega,
        "damping": gamma,
        "kx": kx,
        "ky": ky,
        "phase": phase,
    }
    if family == "two_stream":
        spec["stream_speed"] = stream
    return rho.astype(float), g.astype(float), spec


def _maxwellian(vx: Array, vy: Array, ux: Array | float, uy: Array | float, sigma: float) -> Array:
    var = max(float(sigma) ** 2, EPS)
    return np.exp(-0.5 * (((vx - ux) ** 2 + (vy - uy) ** 2) / var)) / (2.0 * np.pi * var)


def _repeat_initial_difficulty(rho: Array, g: Array) -> float:
    f = f_from_rho_g(rho, g)
    pred = np.repeat(f[:1], f.shape[0], axis=0)
    return relative_l2(pred - f, f)
