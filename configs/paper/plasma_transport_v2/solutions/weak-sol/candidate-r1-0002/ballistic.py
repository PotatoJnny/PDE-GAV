"""Source-driven discrete-velocity VPFP approximation.

The dominant free-streaming operator is integrated exactly in spatial Fourier
space.  The public source is incorporated by an exponential-midpoint Duhamel
step.  A finely reconstructed static solution supplies the exact t=0 anchor;
the coarse kinetic solve contributes only its causal time-dependent increment.
"""
from __future__ import annotations

from math import isfinite

import numpy as np
from numpy.polynomial.legendre import leggauss

from pde_agent_verifier_solver.sdk import AnswerQuery, AnswerView, QueryKind, open_public_case_session
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock

import submission as static_solver


FIELDS = static_solver.FIELDS
NX = 16
NT = 16
ORACLE_BATCH = 250_000
DYNAMIC_SCALE = 1.0


def _velocity_rule():
    pieces = ((-20.0, -10.0, 8), (-10.0, 10.0, 32), (10.0, 20.0, 8))
    nodes, weights = [], []
    for left, right, count in pieces:
        z, w = leggauss(count)
        nodes.append(0.5 * (left + right) + 0.5 * (right - left) * z)
        weights.append(0.5 * (right - left) * w)
    v, w = np.concatenate(nodes), np.concatenate(weights)
    order = np.argsort(v)
    return v[order], w[order]


def _poisson(rho, periods):
    kx = 2 * np.pi * np.fft.fftfreq(NX, d=periods[0] / NX)[:, None]
    ky = 2 * np.pi * np.fft.fftfreq(NX, d=periods[1] / NX)[None, :]
    k2 = kx * kx + ky * ky
    rh = np.fft.fft2(rho - rho.mean())
    ph = np.zeros_like(rh)
    np.divide(rh, k2, out=ph, where=k2 > 0)
    return np.stack((np.fft.ifft2(-1j * kx * ph).real,
                     np.fft.ifft2(-1j * ky * ph).real))


def _macros(f, vx, vy, weights, periods):
    def moment(power):
        return np.einsum("xyij,ij->xy", f, power * weights, optimize=True)

    rho = moment(np.ones_like(vx))
    rho_safe = np.maximum(rho, 1.e-8)
    jx, jy = moment(vx), moment(vy)
    ux, uy = jx / rho_safe, jy / rho_safe
    m20, m11, m02 = moment(vx * vx), moment(vx * vy), moment(vy * vy)
    m30, m21 = moment(vx ** 3), moment(vx * vx * vy)
    m12, m03 = moment(vx * vy * vy), moment(vy ** 3)
    pxx = m20 - rho_safe * ux * ux
    pxy = m11 - rho_safe * ux * uy
    pyy = m02 - rho_safe * uy * uy
    qx = 0.5 * (m30 + m12 - ux * (3 * m20 + m02) - 2 * uy * m11
                + 2 * rho_safe * ux * (ux * ux + uy * uy))
    qy = 0.5 * (m03 + m21 - uy * (3 * m02 + m20) - 2 * ux * m11
                + 2 * rho_safe * uy * (ux * ux + uy * uy))
    electric = _poisson(rho, periods)
    return {
        "rho": rho, "electric_x": electric[0], "electric_y": electric[1],
        "momentum_x": jx, "momentum_y": jy,
        "pressure_xx": pxx, "pressure_xy": pxy, "pressure_yy": pyy,
        "heat_flux_x": qx, "heat_flux_y": qy,
    }


class Prepared:
    def __init__(self, static, deltas, periods, interval):
        self.static = static
        self.periods = np.asarray(periods, dtype=np.float64)
        self.interval = np.asarray(interval, dtype=np.float64)
        self.modes = np.fft.fftfreq(NX) * NX
        self.delta_hat = {
            name: np.fft.fft2(np.asarray(values), axes=(-2, -1)) / (NX * NX)
            for name, values in deltas.items()
        }
        self.closed = False

    def _dynamic(self, coeff, points):
        ex = np.exp(2j * np.pi * np.outer(points[:, 1] / self.periods[0], self.modes))
        ey = np.exp(2j * np.pi * np.outer(points[:, 2] / self.periods[1], self.modes))
        by_time = np.einsum("pa,tab,pb->pt", ex, coeff, ey, optimize=True).real
        z = ((points[:, 0] - self.interval[0])
             * NT / (self.interval[1] - self.interval[0]))
        lo = np.minimum(np.floor(z).astype(int), NT - 1)
        lo = np.maximum(lo, 0)
        frac = np.clip(z - lo, 0.0, 1.0)
        rows = np.arange(len(points))
        return (1 - frac) * by_time[rows, lo] + frac * by_time[rows, lo + 1]

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
        # Reuse the converged high-resolution static Fourier representation.
        static_values = {name: self.static._interpolate(self.static.coeffs[name], points[:, 1:])
                         for name in query.fields}
        values = {name: static_values[name] + DYNAMIC_SCALE * self._dynamic(self.delta_hat[name], points)
                  for name in query.fields}
        if any(not np.isfinite(v).all() for v in values.values()):
            raise ValueError("nonfinite kinetic prediction")
        return AnswerView(kind=query.kind,
                          fields={k: tuple(map(float, values[k])) for k in query.fields},
                          points=query.points, shape=(len(points),))

    def close(self):
        if not self.closed:
            self.static.close()
        self.closed = True


def prepare_solution(case, solve_budget=None):
    started = candidate_solve_clock()
    if solve_budget is not None and (isinstance(solve_budget, bool)
            or not isinstance(solve_budget, (int, float)) or not isfinite(solve_budget) or solve_budget <= 0):
        raise ValueError("solve_budget must be finite and positive")

    def check(reserve=20.0):
        if solve_budget is not None and candidate_solve_clock() - started > max(0.0, solve_budget - reserve):
            raise TimeoutError("insufficient remaining Case budget")

    static = static_solver.prepare_solution(case, solve_budget)
    q = case.public_data["question"]
    arrays = case.public_data["arrays"]
    periods = np.asarray(q["x_domain"], dtype=np.float64)
    interval = np.asarray(q["time_domain"], dtype=np.float64)
    duration = float(interval[1] - interval[0])
    dt = duration / NT
    axes = [np.arange(NX, dtype=np.float64) * periods[d] / NX for d in range(2)]
    rho0 = np.maximum(static_solver._spectral_resample(arrays["rho0"], axes, periods), 1.e-10)
    xg, yg = np.meshgrid(*axes, indexing="ij")
    xy = np.column_stack((xg.ravel(), yg.ravel()))
    v, w = _velocity_rule()
    nv = len(v)
    vx, vy = np.meshgrid(v, v, indexing="ij")
    vw = np.outer(w, w)
    flat_vx, flat_vy = vx.ravel(), vy.ravel()
    nvel, total = nv * nv, NX * NX * nv * nv
    session = open_public_case_session(case)

    def oracle_grid(method, time_value=None):
        result = np.empty(total, dtype=np.float64)
        for lo in range(0, total, ORACLE_BATCH):
            check()
            hi = min(total, lo + ORACLE_BATCH)
            ids = np.arange(lo, hi, dtype=np.int64)
            sidx, vidx = np.divmod(ids, nvel)
            if time_value is None:
                request = np.column_stack((xy[sidx, 0], xy[sidx, 1],
                                           flat_vx[vidx], flat_vy[vidx]))
            else:
                request = np.column_stack((np.full(hi - lo, time_value),
                                           xy[sidx, 0], xy[sidx, 1],
                                           flat_vx[vidx], flat_vy[vidx]))
            got = np.asarray(getattr(session, method)(request), dtype=np.float64)
            if got.shape != (hi - lo,) or not np.isfinite(got).all():
                raise ValueError("invalid public Oracle response")
            result[lo:hi] = got
        return result.reshape(NX, NX, nv, nv)

    g0 = oracle_grid("g0")
    mass = np.einsum("xyij,ij->xy", g0, vw, optimize=True)
    if np.any(mass <= 0):
        raise ValueError("nonpositive initial quadrature mass")
    f = g0 * (rho0 / mass)[:, :, None, None]
    del g0
    kx = (2 * np.pi * np.fft.fftfreq(NX, d=periods[0] / NX))[:, None, None, None]
    ky = (2 * np.pi * np.fft.fftfreq(NX, d=periods[1] / NX))[None, :, None, None]
    phase_rate = kx * vx[None, None, :, :] + ky * vy[None, None, :, :]

    def stream(values, step):
        transformed = np.fft.fft2(values, axes=(0, 1))
        return np.fft.ifft2(transformed * np.exp(-1j * step * phase_rate), axes=(0, 1)).real

    first = _macros(f, vx, vy, vw, periods)
    history = {name: [first[name]] for name in FIELDS}
    for n in range(NT):
        check()
        tm = float(interval[0] + (n + 0.5) * dt)
        source = oracle_grid("source", tm)
        f = stream(f, dt) + dt * stream(source, 0.5 * dt)
        del source
        current = _macros(f, vx, vy, vw, periods)
        for name in FIELDS:
            history[name].append(current[name])
    del f
    deltas = {name: np.stack(history[name]) - history[name][0][None, :, :] for name in FIELDS}
    check(reserve=5.0)
    if any(not np.isfinite(v).all() for v in deltas.values()):
        raise ValueError("nonfinite kinetic increment")
    return Prepared(static, deltas, periods, interval)
