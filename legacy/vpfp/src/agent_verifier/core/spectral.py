from __future__ import annotations

import numpy as np

from agent_verifier.core.types import GridSpec

Array = np.ndarray


def make_physical_grid(grid: GridSpec) -> Array:
    axes = [
        np.linspace(0.0, grid.domain[i], grid.shape[i], endpoint=False, dtype=grid.dtype)
        for i in range(3)
    ]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    return np.stack([x, y, z], axis=-1)


def _freq_axis(n: int, length: float) -> Array:
    return 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)


def make_wavenumbers(grid: GridSpec) -> tuple[Array, Array, Array]:
    nx, ny, nz = grid.shape
    lx, ly, lz = grid.domain
    kx = _freq_axis(nx, lx)[:, None, None]
    ky = _freq_axis(ny, ly)[None, :, None]
    kz = _freq_axis(nz, lz)[None, None, :]
    return kx, ky, kz


def _non_nyquist_mask(grid: GridSpec) -> Array:
    axes = [_freq_axis(grid.shape[i], grid.domain[i]) for i in range(3)]
    masks = []
    for n, axis in zip(grid.shape, axes):
        if n % 2 == 0:
            masks.append(np.abs(axis) < np.max(np.abs(axis)))
        else:
            masks.append(np.ones_like(axis, dtype=bool))
    return (
        masks[0][:, None, None]
        & masks[1][None, :, None]
        & masks[2][None, None, :]
    )


def fftn_vec(u: Array) -> Array:
    return np.fft.fftn(u, axes=(-3, -2, -1))


def ifftn_vec(uhat: Array) -> Array:
    out = np.fft.ifftn(uhat, axes=(-3, -2, -1))
    return np.real_if_close(out, tol=1000).real


def divergence(u: Array, grid: GridSpec) -> Array:
    kx, ky, kz = make_wavenumbers(grid)
    uhat = fftn_vec(np.asarray(u))
    div_hat = 1j * (kx * uhat[0] + ky * uhat[1] + kz * uhat[2])
    return np.fft.ifftn(div_hat, axes=(-3, -2, -1)).real


def gradient_scalar(phi: Array, grid: GridSpec) -> Array:
    kx, ky, kz = make_wavenumbers(grid)
    phihat = np.fft.fftn(phi, axes=(-3, -2, -1))
    return np.stack(
        [
            np.fft.ifftn(1j * kx * phihat, axes=(-3, -2, -1)).real,
            np.fft.ifftn(1j * ky * phihat, axes=(-3, -2, -1)).real,
            np.fft.ifftn(1j * kz * phihat, axes=(-3, -2, -1)).real,
        ],
        axis=0,
    )


def laplacian_vec(u: Array, grid: GridSpec) -> Array:
    kx, ky, kz = make_wavenumbers(grid)
    k2 = kx**2 + ky**2 + kz**2
    return ifftn_vec(-k2[None, ...] * fftn_vec(np.asarray(u)))


def curl(a: Array, grid: GridSpec) -> Array:
    kx, ky, kz = make_wavenumbers(grid)
    ahat = fftn_vec(np.asarray(a))
    cx = 1j * (ky * ahat[2] - kz * ahat[1])
    cy = 1j * (kz * ahat[0] - kx * ahat[2])
    cz = 1j * (kx * ahat[1] - ky * ahat[0])
    return ifftn_vec(np.stack([cx, cy, cz], axis=0))


def helmholtz_project(v: Array, grid: GridSpec) -> Array:
    v = np.asarray(v)
    kx, ky, kz = make_wavenumbers(grid)
    k2 = kx**2 + ky**2 + kz**2
    vhat = fftn_vec(v)
    dot = kx * vhat[0] + ky * vhat[1] + kz * vhat[2]
    factor = np.zeros_like(k2, dtype=vhat.dtype)
    np.divide(dot, k2, out=factor, where=k2 > 0)
    phat = np.empty_like(vhat)
    phat[0] = vhat[0] - kx * factor
    phat[1] = vhat[1] - ky * factor
    phat[2] = vhat[2] - kz * factor
    phat *= _non_nyquist_mask(grid)[None, ...]
    return ifftn_vec(phat)


def lowpass_project(v: Array, grid: GridSpec, kmax: int) -> Array:
    kx, ky, kz = make_wavenumbers(grid)
    mask = (kx**2 + ky**2 + kz**2) <= float(kmax) ** 2
    return ifftn_vec(fftn_vec(np.asarray(v)) * mask[None, ...])


def dealias_mask(grid: GridSpec) -> Array:
    kx, ky, kz = make_wavenumbers(grid)
    cutoffs = [
        (2.0 / 3.0) * (grid.shape[i] // 2) * (2.0 * np.pi / grid.domain[i])
        for i in range(3)
    ]
    return (
        (np.abs(kx) <= cutoffs[0])
        & (np.abs(ky) <= cutoffs[1])
        & (np.abs(kz) <= cutoffs[2])
    )


def dealias_vec(v: Array, grid: GridSpec) -> Array:
    return ifftn_vec(fftn_vec(np.asarray(v)) * dealias_mask(grid)[None, ...])
