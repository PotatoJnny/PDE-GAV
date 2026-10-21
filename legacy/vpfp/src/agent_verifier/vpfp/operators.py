from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

Array = np.ndarray
EPS = 1.0e-12


@dataclass(frozen=True)
class VPFPGrid:
    nx: int = 8
    ny: int = 8
    nvx: int = 9
    nvy: int = 9
    x_domain: tuple[float, float] = (2.0 * np.pi, 2.0 * np.pi)
    v_bounds: tuple[float, float, float, float] = (-5.0, 5.0, -5.0, 5.0)
    dtype: str = "float64"

    def validate(self) -> None:
        if min(self.nx, self.ny) < 4:
            raise ValueError("VPFP x grid needs nx, ny >= 4")
        if min(self.nvx, self.nvy) < 3:
            raise ValueError("VPFP v grid needs nvx, nvy >= 3")
        if self.x_domain[0] <= 0.0 or self.x_domain[1] <= 0.0:
            raise ValueError("VPFP x_domain lengths must be positive")
        vx0, vx1, vy0, vy1 = self.v_bounds
        if vx1 <= vx0 or vy1 <= vy0:
            raise ValueError("VPFP v_bounds must be ordered")

    @property
    def dx(self) -> float:
        return float(self.x_domain[0]) / float(self.nx)

    @property
    def dy(self) -> float:
        return float(self.x_domain[1]) / float(self.ny)

    @property
    def dvx(self) -> float:
        vx0, vx1, _vy0, _vy1 = self.v_bounds
        return float(vx1 - vx0) / float(max(self.nvx - 1, 1))

    @property
    def dvy(self) -> float:
        _vx0, _vx1, vy0, vy1 = self.v_bounds
        return float(vy1 - vy0) / float(max(self.nvy - 1, 1))

    @property
    def velocity_cell_area(self) -> float:
        return self.dvx * self.dvy

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VPFPTime:
    t0: float = 0.0
    t1: float = 0.5
    nt: int = 5

    def validate(self) -> None:
        if self.nt < 3:
            raise ValueError("VPFP MVP needs nt >= 3 for second-order time derivatives")
        if self.t1 <= self.t0:
            raise ValueError("VPFP time interval must satisfy t1 > t0")

    @property
    def dt(self) -> float:
        return (float(self.t1) - float(self.t0)) / float(self.nt - 1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_axes(grid: VPFPGrid, time: VPFPTime | None = None) -> dict[str, Array]:
    grid.validate()
    axes = {
        "x": np.linspace(0.0, grid.x_domain[0], grid.nx, endpoint=False, dtype=grid.dtype),
        "y": np.linspace(0.0, grid.x_domain[1], grid.ny, endpoint=False, dtype=grid.dtype),
        "vx": np.linspace(grid.v_bounds[0], grid.v_bounds[1], grid.nvx, dtype=grid.dtype),
        "vy": np.linspace(grid.v_bounds[2], grid.v_bounds[3], grid.nvy, dtype=grid.dtype),
    }
    if time is not None:
        time.validate()
        axes["t"] = np.linspace(time.t0, time.t1, time.nt, dtype=grid.dtype)
    return axes


def velocity_weights(grid: VPFPGrid) -> Array:
    return np.full((grid.nvx, grid.nvy), grid.velocity_cell_area, dtype=grid.dtype)


def f_from_rho_g(rho: Array, g: Array) -> Array:
    rho_arr = np.asarray(rho, dtype=float)
    g_arr = np.asarray(g, dtype=float)
    return rho_arr[..., None, None] * g_arr


def normalize_g(g_raw: Array, grid: VPFPGrid) -> Array:
    arr = np.maximum(np.asarray(g_raw, dtype=float), 0.0)
    mass = np.sum(arr, axis=(-2, -1), keepdims=True) * grid.velocity_cell_area
    return arr / np.maximum(mass, EPS)


def poisson_electric_field(rho: Array, grid: VPFPGrid) -> tuple[Array, Array]:
    """Solve -Delta_x phi = rho - mean_x(rho), E = -grad_x phi."""

    rho_arr = np.asarray(rho, dtype=float)
    if rho_arr.shape[-2:] != (grid.nx, grid.ny):
        raise ValueError(f"rho trailing shape must be {(grid.nx, grid.ny)}, got {rho_arr.shape}")
    rho_centered = rho_arr - np.mean(rho_arr, axis=(-2, -1), keepdims=True)
    rho_hat = np.fft.fftn(rho_centered, axes=(-2, -1))
    kx = 2.0 * np.pi * np.fft.fftfreq(grid.nx, d=grid.dx)[:, None]
    ky = 2.0 * np.pi * np.fft.fftfreq(grid.ny, d=grid.dy)[None, :]
    k2 = kx**2 + ky**2
    phi_hat = np.zeros_like(rho_hat, dtype=complex)
    np.divide(rho_hat, k2, out=phi_hat, where=k2 > 0.0)
    phi = np.fft.ifftn(phi_hat, axes=(-2, -1)).real
    ex = np.fft.ifftn(-1j * kx * phi_hat, axes=(-2, -1)).real
    ey = np.fft.ifftn(-1j * ky * phi_hat, axes=(-2, -1)).real
    return phi, np.stack([ex, ey], axis=-3)


def time_derivative(values: Array, time: VPFPTime) -> Array:
    arr = np.asarray(values, dtype=float)
    edge_order = 2 if arr.shape[0] >= 3 else 1
    return np.gradient(arr, time.dt, axis=0, edge_order=edge_order)


def periodic_grad_x(values: Array, grid: VPFPGrid) -> tuple[Array, Array]:
    arr = np.asarray(values, dtype=float)
    d_x = (np.roll(arr, -1, axis=1) - np.roll(arr, 1, axis=1)) / (2.0 * grid.dx)
    d_y = (np.roll(arr, -1, axis=2) - np.roll(arr, 1, axis=2)) / (2.0 * grid.dy)
    return d_x, d_y


def neumann_grad_v(values: Array, grid: VPFPGrid) -> tuple[Array, Array]:
    arr = np.asarray(values, dtype=float)
    gx = np.zeros_like(arr, dtype=float)
    gy = np.zeros_like(arr, dtype=float)
    gx[..., 1:-1, :] = (arr[..., 2:, :] - arr[..., :-2, :]) / (2.0 * grid.dvx)
    gy[..., :, 1:-1] = (arr[..., :, 2:] - arr[..., :, :-2]) / (2.0 * grid.dvy)
    return gx, gy


def neumann_laplacian_v(values: Array, grid: VPFPGrid) -> Array:
    arr = np.asarray(values, dtype=float)
    lap = np.zeros_like(arr, dtype=float)
    lap[..., 1:-1, :] += (arr[..., 2:, :] - 2.0 * arr[..., 1:-1, :] + arr[..., :-2, :]) / (grid.dvx**2)
    lap[..., 0, :] += (arr[..., 1, :] - arr[..., 0, :]) / (grid.dvx**2)
    lap[..., -1, :] += (arr[..., -2, :] - arr[..., -1, :]) / (grid.dvx**2)
    lap[..., :, 1:-1] += (arr[..., :, 2:] - 2.0 * arr[..., :, 1:-1] + arr[..., :, :-2]) / (grid.dvy**2)
    lap[..., :, 0] += (arr[..., :, 1] - arr[..., :, 0]) / (grid.dvy**2)
    lap[..., :, -1] += (arr[..., :, -2] - arr[..., :, -1]) / (grid.dvy**2)
    return lap


def compute_source(rho: Array, g: Array, *, grid: VPFPGrid, time: VPFPTime, collision_nu: float) -> tuple[Array, Array, Array]:
    f = f_from_rho_g(rho, g)
    phi, electric = poisson_electric_field(rho, grid)
    df_dt = time_derivative(f, time)
    df_dx, df_dy = periodic_grad_x(f, grid)
    dg_dvx, dg_dvy = neumann_grad_v(g, grid)
    lap_g = neumann_laplacian_v(g, grid)
    axes = make_axes(grid)
    vx = axes["vx"][None, None, None, :, None]
    vy = axes["vy"][None, None, None, None, :]
    ex = electric[:, 0, :, :][:, :, :, None, None]
    ey = electric[:, 1, :, :][:, :, :, None, None]
    rho_v = np.asarray(rho, dtype=float)[:, :, :, None, None]
    source = df_dt + vx * df_dx + vy * df_dy + rho_v * (ex * dg_dvx + ey * dg_dvy)
    source = source - float(collision_nu) * rho_v * lap_g
    return source.astype(float), phi.astype(float), electric.astype(float)


def residual(rho: Array, g: Array, source: Array, *, grid: VPFPGrid, time: VPFPTime, collision_nu: float) -> Array:
    computed, _phi, _electric = compute_source(
        rho,
        g,
        grid=grid,
        time=time,
        collision_nu=collision_nu,
    )
    return computed - np.asarray(source, dtype=float)


def relative_l2(num: Array, denom: Array) -> float:
    return float(np.linalg.norm(np.asarray(num, dtype=float)) / (np.linalg.norm(np.asarray(denom, dtype=float)) + EPS))
