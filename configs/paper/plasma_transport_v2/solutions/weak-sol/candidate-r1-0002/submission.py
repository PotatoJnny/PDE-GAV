"""Static, non-Maxwellian initial-moment VPFP approximation.

This deliberately conservative checkpoint reconstructs all requested moments
of the public initial distribution with high-order velocity quadrature.  It
does not yet evolve them, but unlike S0 it retains flow, anisotropic stress,
and central heat flux.
"""
from __future__ import annotations

from math import isfinite

import numpy as np
from numpy.polynomial.legendre import leggauss

from pde_agent_verifier_solver.sdk import AnswerQuery, AnswerView, QueryKind, open_public_case_session
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock


FIELDS = (
    "rho", "electric_x", "electric_y", "momentum_x", "momentum_y",
    "pressure_xx", "pressure_xy", "pressure_yy", "heat_flux_x", "heat_flux_y",
)
NX = 24
NV = 80
VMAX = 18.0
ORACLE_BATCH = 250_000


def _spectral_resample(values, xnew, periods):
    """Evaluate a periodic tensor-product grid at a new tensor grid."""
    a = np.asarray(values, dtype=np.float64)
    n0, n1 = a.shape[:2]
    k0 = np.fft.fftfreq(n0) * n0
    k1 = np.fft.fftfreq(n1) * n1
    e0 = np.exp(2j * np.pi * np.outer(xnew[0] / periods[0], k0))
    e1 = np.exp(2j * np.pi * np.outer(xnew[1] / periods[1], k1))
    hat = np.fft.fftn(a, axes=(0, 1)) / (n0 * n1)
    if a.ndim == 2:
        return np.einsum("ia,ab,jb->ij", e0, hat, e1, optimize=True).real
    return np.einsum("ia,abc,jb->ijc", e0, hat, e1, optimize=True).real


def _poisson(rho, periods):
    nx, ny = rho.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=periods[0] / nx)[:, None]
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=periods[1] / ny)[None, :]
    k2 = kx * kx + ky * ky
    rh = np.fft.fft2(rho - rho.mean())
    ph = np.zeros_like(rh)
    np.divide(rh, k2, out=ph, where=k2 > 0)
    return np.stack((np.fft.ifft2(-1j * kx * ph).real,
                     np.fft.ifft2(-1j * ky * ph).real))


class Prepared:
    def __init__(self, grids, periods, interval):
        self.periods = np.asarray(periods, dtype=np.float64)
        self.interval = np.asarray(interval, dtype=np.float64)
        self.coeffs = {
            name: np.fft.fft2(np.asarray(grid, dtype=np.float64)) / (NX * NX)
            for name, grid in grids.items()
        }
        self.closed = False

    def _interpolate(self, coeff, xy):
        modes = np.fft.fftfreq(NX) * NX
        ex = np.exp(2j * np.pi * np.outer(xy[:, 0] / self.periods[0], modes))
        ey = np.exp(2j * np.pi * np.outer(xy[:, 1] / self.periods[1], modes))
        return np.einsum("pa,ab,pb->p", ex, coeff, ey, optimize=True).real

    def evaluate(self, query):
        if self.closed:
            raise RuntimeError("prepared handle is closed")
        if not isinstance(query, AnswerQuery) or query.kind is not QueryKind.POINTS:
            raise ValueError("POINTS query required")
        if (not query.fields or len(set(query.fields)) != len(query.fields)
                or not set(query.fields) <= set(FIELDS)):
            raise ValueError("unknown or duplicate transport field")
        points = np.asarray([(p["t"], p["x"], p["y"]) for p in query.points], dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0 or not np.isfinite(points).all():
            raise ValueError("finite nonempty (t,x,y) points required")
        if np.any(points[:, 0] < self.interval[0]) or np.any(points[:, 0] > self.interval[1]):
            raise ValueError("query outside prepared interval")
        values = {name: self._interpolate(self.coeffs[name], points[:, 1:]) for name in query.fields}
        if any(not np.isfinite(v).all() for v in values.values()):
            raise ValueError("nonfinite interpolation result")
        return AnswerView(kind=query.kind,
                          fields={k: tuple(map(float, values[k])) for k in query.fields},
                          points=query.points, shape=(len(points),))

    def close(self):
        self.closed = True


def prepare_solution(case, solve_budget=None):
    started = candidate_solve_clock()
    if solve_budget is not None and (isinstance(solve_budget, bool)
            or not isinstance(solve_budget, (int, float))
            or not isfinite(solve_budget) or solve_budget <= 0):
        raise ValueError("solve_budget must be finite and positive")

    def check(reserve=10.0):
        if solve_budget is not None and candidate_solve_clock() - started > max(0.0, solve_budget - reserve):
            raise TimeoutError("insufficient remaining Case budget")

    q = case.public_data["question"]
    arrays = case.public_data["arrays"]
    periods = np.asarray(q["x_domain"], dtype=np.float64)
    interval = np.asarray(q["time_domain"], dtype=np.float64)
    if periods.shape != (2,) or np.any(periods <= 0) or interval.shape != (2,):
        raise ValueError("invalid public domain")

    axes = [np.arange(NX, dtype=np.float64) * periods[d] / NX for d in range(2)]
    rho = _spectral_resample(arrays["rho0"], axes, periods)
    momentum = _spectral_resample(arrays["momentum0"], axes, periods)
    kinetic = _spectral_resample(arrays["kinetic_energy0"], axes, periods)
    rho = np.maximum(rho, 1.e-10)
    u = momentum / rho[..., None]

    nodes, weights = leggauss(NV)
    v = VMAX * nodes
    w = VMAX * weights
    vx, vy = np.meshgrid(v, v, indexing="ij")
    vw = np.outer(w, w)
    xg, yg = np.meshgrid(*axes, indexing="ij")
    xy = np.column_stack((xg.ravel(), yg.ravel()))
    nvel = NV * NV
    total = len(xy) * nvel
    gflat = np.empty(total, dtype=np.float64)
    session = open_public_case_session(case)
    flat_vx, flat_vy = vx.ravel(), vy.ravel()
    for lo in range(0, total, ORACLE_BATCH):
        check()
        hi = min(total, lo + ORACLE_BATCH)
        ids = np.arange(lo, hi, dtype=np.int64)
        sidx, vidx = np.divmod(ids, nvel)
        request = np.column_stack((xy[sidx, 0], xy[sidx, 1], flat_vx[vidx], flat_vy[vidx]))
        got = np.asarray(session.g0(request), dtype=np.float64)
        if got.shape != (hi - lo,) or not np.isfinite(got).all():
            raise ValueError("invalid g0 Oracle response")
        gflat[lo:hi] = got
    g = gflat.reshape(NX, NX, NV, NV)
    cx = vx[None, None, :, :] - u[:, :, 0, None, None]
    cy = vy[None, None, :, :] - u[:, :, 1, None, None]
    mass = np.einsum("xyij,ij->xy", g, vw, optimize=True)
    if np.any(mass <= 0) or not np.isfinite(mass).all():
        raise ValueError("invalid quadrature mass")
    scale = rho / mass
    pxx = scale * np.einsum("xyij,xyij,ij->xy", g, cx * cx, vw, optimize=True)
    pxy = scale * np.einsum("xyij,xyij,ij->xy", g, cx * cy, vw, optimize=True)
    pyy = scale * np.einsum("xyij,xyij,ij->xy", g, cy * cy, vw, optimize=True)
    # The public kinetic-energy preview gives the exact pressure trace.  Use it
    # to remove the tiny full-space tail error without altering anisotropy.
    trace_exact = np.maximum(2 * kinetic - rho * np.sum(u * u, axis=-1), 1.e-12)
    trace_scale = trace_exact / np.maximum(pxx + pyy, 1.e-12)
    pxx, pxy, pyy = pxx * trace_scale, pxy * trace_scale, pyy * trace_scale
    c2 = cx * cx + cy * cy
    qx = scale * np.einsum("xyij,xyij,ij->xy", g, 0.5 * c2 * cx, vw, optimize=True)
    qy = scale * np.einsum("xyij,xyij,ij->xy", g, 0.5 * c2 * cy, vw, optimize=True)
    electric = _poisson(rho, periods)
    grids = {
        "rho": rho, "electric_x": electric[0], "electric_y": electric[1],
        "momentum_x": momentum[..., 0], "momentum_y": momentum[..., 1],
        "pressure_xx": pxx, "pressure_xy": pxy, "pressure_yy": pyy,
        "heat_flux_x": qx, "heat_flux_y": qy,
    }
    check(reserve=3.0)
    if any(not np.isfinite(a).all() for a in grids.values()):
        raise ValueError("nonfinite reconstructed moment")
    prepared = Prepared(grids, periods, interval)
    # Retain the resolved initial kinetic representation for entrypoints that
    # apply an analytic homogeneous propagator.  It remains candidate-owned
    # state constructed entirely during preparation.
    prepared.phase_f0 = g * scale[:, :, None, None]
    prepared.phase_v = v
    prepared.phase_w = w
    return prepared
