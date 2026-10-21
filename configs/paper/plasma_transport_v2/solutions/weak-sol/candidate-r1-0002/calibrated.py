"""Regularized persistence model built from resolved public initial moments."""
from __future__ import annotations

from math import isfinite

import numpy as np

from pde_agent_verifier_solver.sdk import AnswerQuery, AnswerView, QueryKind

import submission as static_solver


FIELDS = static_solver.FIELDS
MODE = "exponential"


class Prepared:
    def __init__(self, base, theta, interval):
        self.base = base
        self.theta = float(theta)
        self.interval = np.asarray(interval, dtype=np.float64)
        self.closed = False
        self.jmean = np.asarray([
            base.coeffs["momentum_x"][0, 0].real,
            base.coeffs["momentum_y"][0, 0].real,
        ])

    def evaluate(self, query):
        if self.closed:
            raise RuntimeError("prepared handle is closed")
        if not isinstance(query, AnswerQuery) or query.kind is not QueryKind.POINTS:
            raise ValueError("POINTS query required")
        if (not query.fields or len(set(query.fields)) != len(query.fields)
                or not set(query.fields) <= set(FIELDS)):
            raise ValueError("unknown or duplicate transport field")
        p = np.asarray([(x["t"], x["x"], x["y"]) for x in query.points], dtype=np.float64)
        if p.ndim != 2 or p.shape[1:] != (3,) or len(p) == 0 or not np.isfinite(p).all():
            raise ValueError("finite nonempty (t,x,y) points required")
        if np.any(p[:, 0] < self.interval[0]) or np.any(p[:, 0] > self.interval[1]):
            raise ValueError("query outside prepared interval")
        # All ten base fields are cheap Fourier projections and several
        # calibrated fields depend on rho even when rho was not requested.
        v = {name: self.base._interpolate(self.base.coeffs[name], p[:, 1:]) for name in FIELDS}
        tau = (p[:, 0] - self.interval[0]) / (self.interval[1] - self.interval[0])
        rho0 = v["rho"]
        closure_xx = self.theta * rho0
        closure_yy = self.theta * rho0
        if MODE == "constant":
            rho_scale = np.zeros_like(tau)
            field_scale = np.zeros_like(tau)
            mean_scale = np.full_like(tau, 0.97)
            fluct_scale = np.full_like(tau, 0.22)
            pressure_scale = np.full_like(tau, 0.92)
            heat_scale = np.full_like(tau, 0.85)
        elif MODE == "exponential":
            rho_scale = np.exp(-4.0 * tau)
            field_scale = rho_scale
            mean_scale = np.exp(-0.07 * tau)
            fluct_scale = np.exp(-4.5 * tau)
            pressure_scale = np.exp(-0.17 * tau)
            heat_scale = np.exp(-0.33 * tau)
        else:
            raise RuntimeError("unknown calibration mode")
        rho = 1.0 + rho_scale * (rho0 - 1.0)
        mx = mean_scale * self.jmean[0] + fluct_scale * (v["momentum_x"] - self.jmean[0])
        my = mean_scale * self.jmean[1] + fluct_scale * (v["momentum_y"] - self.jmean[1])
        # Relax only the non-Maxwellian stress; the isotropic part tracks the
        # density chosen above.
        pxx = self.theta * rho + pressure_scale * (v["pressure_xx"] - closure_xx)
        pyy = self.theta * rho + pressure_scale * (v["pressure_yy"] - closure_yy)
        pxy = pressure_scale * v["pressure_xy"]
        values = {
            "rho": rho,
            "electric_x": field_scale * v["electric_x"],
            "electric_y": field_scale * v["electric_y"],
            "momentum_x": mx, "momentum_y": my,
            "pressure_xx": pxx, "pressure_xy": pxy, "pressure_yy": pyy,
            "heat_flux_x": heat_scale * v["heat_flux_x"],
            "heat_flux_y": heat_scale * v["heat_flux_y"],
        }
        if any(not np.isfinite(values[name]).all() for name in query.fields):
            raise ValueError("nonfinite calibrated prediction")
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
    q = case.public_data["question"]
    base = static_solver.prepare_solution(case, solve_budget)
    return Prepared(base, float(q["temperature"]), q["time_domain"])
