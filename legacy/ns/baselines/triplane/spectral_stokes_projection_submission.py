# Public-only reference solver for triplane generated Navier-Stokes cases.
# Intended as the first-round previous_solver/submission.py baseline for Codex agents.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def solve_case(case_dir: str, query_grid=None):
    case_path = Path(case_dir)
    with np.load(case_path / "public_arrays.npz") as data:
        x = np.asarray(data["x"], dtype=np.float64)
        y = np.asarray(data["y"], dtype=np.float64)
        t = np.asarray(data["t"], dtype=np.float64)
        fluid = np.asarray(data["fluid_mask"], dtype=bool)
        u0 = np.asarray(data["initial_velocity"], dtype=np.float64)
        forcing = np.asarray(data["forcing"], dtype=np.float64)
    viscosity = _load_viscosity(case_path)

    velocity = _spectral_stokes_projection(u0, forcing, t, x, y, viscosity)

    velocity = _apply_fluid_mask(velocity, fluid)
    if query_grid is not None:
        velocity = _resample_velocity(
            velocity,
            old_t=t,
            old_x=x,
            old_y=y,
            new_t=np.asarray(query_grid.get("t", t), dtype=np.float64),
            new_x=np.asarray(query_grid.get("x", x), dtype=np.float64),
            new_y=np.asarray(query_grid.get("y", y), dtype=np.float64),
        )
    return {
        "velocity": velocity.astype(np.float64, copy=False),
        "metadata": {
            "method": "spectral_stokes_projection",
            "viscosity": float(viscosity),
            "note": "public-only spectral Stokes/projection reference solver; uses periodic projection and then reapplies the fluid mask",
        },
    }


def _load_viscosity(case_path: Path) -> float:
    question_path = case_path / "public_question.json"
    if question_path.exists():
        value = json.loads(question_path.read_text()).get("viscosity")
        if value is not None:
            return float(value)
    return 0.01



def _spectral_stokes_projection(
    u0: np.ndarray,
    forcing: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    viscosity: float,
) -> np.ndarray:
    out = np.empty((t.size, *u0.shape), dtype=np.float64)
    kx, ky, k2 = _wave_numbers(x, y)
    state_hat_u = np.fft.fftn(u0[0])
    state_hat_v = np.fft.fftn(u0[1])
    state_hat_u, state_hat_v = _project_hat(state_hat_u, state_hat_v, kx, ky, k2)
    out[0] = np.stack(
        [np.fft.ifftn(state_hat_u).real, np.fft.ifftn(state_hat_v).real],
        axis=0,
    )
    for i in range(1, t.size):
        dt = float(t[i] - t[i - 1])
        force_avg = 0.5 * (forcing[i - 1] + forcing[i])
        fu = np.fft.fftn(force_avg[0])
        fv = np.fft.fftn(force_avg[1])
        denom = 1.0 + float(viscosity) * dt * k2
        state_hat_u = (state_hat_u + dt * fu) / denom
        state_hat_v = (state_hat_v + dt * fv) / denom
        state_hat_u, state_hat_v = _project_hat(state_hat_u, state_hat_v, kx, ky, k2)
        out[i] = np.stack(
            [np.fft.ifftn(state_hat_u).real, np.fft.ifftn(state_hat_v).real],
            axis=0,
        )
    return out



def _wave_numbers(x: np.ndarray, y: np.ndarray):
    dx = float(np.mean(np.diff(x))) if x.size > 1 else 1.0
    dy = float(np.mean(np.diff(y))) if y.size > 1 else 1.0
    kx_1d = 2.0 * np.pi * np.fft.fftfreq(x.size, d=dx)
    ky_1d = 2.0 * np.pi * np.fft.fftfreq(y.size, d=dy)
    kx, ky = np.meshgrid(kx_1d, ky_1d, indexing="ij")
    k2 = kx * kx + ky * ky
    return kx, ky, k2


def _project_hat(uh: np.ndarray, vh: np.ndarray, kx: np.ndarray, ky: np.ndarray, k2: np.ndarray):
    dot = kx * uh + ky * vh
    scale = np.zeros_like(k2, dtype=np.complex128)
    mask = k2 > 0.0
    scale[mask] = dot[mask] / k2[mask]
    return uh - kx * scale, vh - ky * scale


def _apply_fluid_mask(velocity: np.ndarray, fluid: np.ndarray) -> np.ndarray:
    out = np.asarray(velocity, dtype=np.float64).copy()
    solid = ~np.asarray(fluid, dtype=bool)
    out[:, :, solid] = 0.0
    return out


def _resample_velocity(
    velocity: np.ndarray,
    *,
    old_t: np.ndarray,
    old_x: np.ndarray,
    old_y: np.ndarray,
    new_t: np.ndarray,
    new_x: np.ndarray,
    new_y: np.ndarray,
) -> np.ndarray:
    out = _interp_axis(velocity, old_t, new_t, axis=0)
    out = _interp_axis(out, old_x, new_x, axis=2)
    out = _interp_axis(out, old_y, new_y, axis=3)
    return out


def _interp_axis(arr: np.ndarray, old_axis: np.ndarray, new_axis: np.ndarray, *, axis: int) -> np.ndarray:
    old = np.asarray(old_axis, dtype=np.float64).reshape(-1)
    new = np.asarray(new_axis, dtype=np.float64).reshape(-1)
    if old.size == new.size and np.allclose(old, new, rtol=1.0e-12, atol=1.0e-12):
        return np.asarray(arr, dtype=np.float64)
    moved = np.moveaxis(np.asarray(arr, dtype=np.float64), axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    interp = np.empty((new.size, flat.shape[1]), dtype=np.float64)
    for idx in range(flat.shape[1]):
        interp[:, idx] = np.interp(new, old, flat[:, idx])
    out = interp.reshape((new.size, *moved.shape[1:]))
    return np.moveaxis(out, 0, axis)
