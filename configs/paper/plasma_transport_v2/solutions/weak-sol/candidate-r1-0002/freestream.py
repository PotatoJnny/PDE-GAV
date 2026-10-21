"""Exact collisionless propagation of the resolved public initial state."""
from __future__ import annotations

from math import isfinite

import numpy as np
import torch

from pde_agent_verifier_solver.sdk import AnswerQuery, AnswerView, QueryKind

import submission as static_solver


FIELDS = static_solver.FIELDS
NX = static_solver.NX
NT = 24
BLEND = 1.0


def _torch_macros(f, vx, vy, weight, periods):
    def m(power):
        return torch.sum(f * power[None, None, :, :] * weight[None, None, :, :], dim=(-2, -1))
    rho = m(torch.ones_like(vx))
    rs = torch.clamp(rho, min=1.e-8)
    jx, jy = m(vx), m(vy)
    ux, uy = jx / rs, jy / rs
    m20, m11, m02 = m(vx * vx), m(vx * vy), m(vy * vy)
    m30, m21, m12, m03 = m(vx ** 3), m(vx * vx * vy), m(vx * vy * vy), m(vy ** 3)
    pxx, pxy, pyy = m20 - rs * ux * ux, m11 - rs * ux * uy, m02 - rs * uy * uy
    qx = 0.5 * (m30 + m12 - ux * (3 * m20 + m02) - 2 * uy * m11
                + 2 * rs * ux * (ux * ux + uy * uy))
    qy = 0.5 * (m03 + m21 - uy * (3 * m02 + m20) - 2 * ux * m11
                + 2 * rs * uy * (ux * ux + uy * uy))
    rn = rho.detach().cpu().numpy()
    electric = static_solver._poisson(rn, periods)
    return {
        "rho": rn, "electric_x": electric[0], "electric_y": electric[1],
        "momentum_x": jx.detach().cpu().numpy(), "momentum_y": jy.detach().cpu().numpy(),
        "pressure_xx": pxx.detach().cpu().numpy(), "pressure_xy": pxy.detach().cpu().numpy(),
        "pressure_yy": pyy.detach().cpu().numpy(),
        "heat_flux_x": qx.detach().cpu().numpy(), "heat_flux_y": qy.detach().cpu().numpy(),
    }


class Prepared:
    def __init__(self, base, deltas, periods, interval):
        self.base = base
        self.periods = np.asarray(periods, dtype=np.float64)
        self.interval = np.asarray(interval, dtype=np.float64)
        self.modes = np.fft.fftfreq(NX) * NX
        self.delta_hat = {k: np.fft.fft2(a, axes=(-2, -1)) / (NX * NX) for k, a in deltas.items()}
        self.closed = False

    def _delta(self, coeff, points):
        ex = np.exp(2j * np.pi * np.outer(points[:, 1] / self.periods[0], self.modes))
        ey = np.exp(2j * np.pi * np.outer(points[:, 2] / self.periods[1], self.modes))
        all_t = np.einsum("pa,tab,pb->pt", ex, coeff, ey, optimize=True).real
        z = (points[:, 0] - self.interval[0]) * NT / (self.interval[1] - self.interval[0])
        lo = np.clip(np.floor(z).astype(int), 0, NT - 1)
        frac = np.clip(z - lo, 0., 1.)
        rows = np.arange(len(points))
        return (1 - frac) * all_t[rows, lo] + frac * all_t[rows, lo + 1]

    def evaluate(self, query):
        if self.closed:
            raise RuntimeError("prepared handle is closed")
        if not isinstance(query, AnswerQuery) or query.kind is not QueryKind.POINTS:
            raise ValueError("POINTS query required")
        if (not query.fields or len(set(query.fields)) != len(query.fields)
                or not set(query.fields) <= set(FIELDS)):
            raise ValueError("unknown or duplicate field")
        p = np.asarray([(x["t"], x["x"], x["y"]) for x in query.points], dtype=np.float64)
        if p.ndim != 2 or p.shape[1:] != (3,) or len(p) == 0 or not np.isfinite(p).all():
            raise ValueError("finite nonempty points required")
        if np.any(p[:, 0] < self.interval[0]) or np.any(p[:, 0] > self.interval[1]):
            raise ValueError("query outside prepared interval")
        values = {}
        for name in query.fields:
            anchor = self.base._interpolate(self.base.coeffs[name], p[:, 1:])
            values[name] = anchor + BLEND * self._delta(self.delta_hat[name], p)
        if any(not np.isfinite(x).all() for x in values.values()):
            raise ValueError("nonfinite free-stream prediction")
        return AnswerView(kind=query.kind,
                          fields={k: tuple(map(float, values[k])) for k in query.fields},
                          points=query.points, shape=(len(p),))

    def close(self):
        if not self.closed:
            self.base.close()
        self.closed = True


def prepare_solution(case, solve_budget=None):
    if solve_budget is not None and (isinstance(solve_budget, bool)
            or not isinstance(solve_budget, (int, float)) or not isfinite(solve_budget) or solve_budget <= 0):
        raise ValueError("solve_budget must be finite and positive")
    base = static_solver.prepare_solution(case, solve_budget)
    q = case.public_data["question"]
    periods = np.asarray(q["x_domain"], dtype=np.float64)
    interval = np.asarray(q["time_domain"], dtype=np.float64)
    dt = float(interval[1] - interval[0]) / NT
    device = torch.device("cuda:0")
    f = torch.as_tensor(base.phase_f0, dtype=torch.float64, device=device)
    v = torch.as_tensor(base.phase_v, dtype=torch.float64, device=device)
    w = torch.as_tensor(base.phase_w, dtype=torch.float64, device=device)
    vx, vy = torch.meshgrid(v, v, indexing="ij")
    weight = w[:, None] * w[None, :]
    kx = torch.as_tensor(2 * np.pi * np.fft.fftfreq(NX, d=periods[0] / NX),
                         dtype=torch.float64, device=device)[:, None, None, None]
    ky = torch.as_tensor(2 * np.pi * np.fft.fftfreq(NX, d=periods[1] / NX),
                         dtype=torch.float64, device=device)[None, :, None, None]
    phase = torch.exp(-1j * dt * (kx * vx[None, None, :, :] + ky * vy[None, None, :, :]))
    first = _torch_macros(f, vx, vy, weight, periods)
    history = {name: [first[name]] for name in FIELDS}
    for _ in range(NT):
        f = torch.fft.ifft2(torch.fft.fft2(f, dim=(0, 1)) * phase, dim=(0, 1)).real
        current = _torch_macros(f, vx, vy, weight, periods)
        for name in FIELDS:
            history[name].append(current[name])
    del f
    torch.cuda.synchronize()
    deltas = {name: np.stack(values) - values[0][None, :, :] for name, values in history.items()}
    if any(not np.isfinite(x).all() for x in deltas.values()):
        raise ValueError("nonfinite free-stream increment")
    return Prepared(base, deltas, periods, interval)
