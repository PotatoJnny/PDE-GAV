"""Simple v2 baseline: hold initial density and use a local Maxwellian closure.

This is deliberately not a kinetic time integrator. It queries only public
rho0 on its own fixed spatial mesh, solves periodic Poisson once, and holds
n/E fixed. The declared closure is u=0, P=n*theta*I, q_h=0. It ignores source
and evolution; zero heat flux is a baseline approximation, never hidden truth.
"""
from __future__ import annotations

from math import isfinite

import numpy as np

from pde_agent_verifier_solver.sdk import AnswerQuery, AnswerView, QueryKind, open_public_case_session
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock

FIELDS = ("rho", "electric_x", "electric_y", "momentum_x", "momentum_y",
          "pressure_xx", "pressure_xy", "pressure_yy", "heat_flux_x", "heat_flux_y")
GRID_SIZE = 32


class Prepared:
    def __init__(self, rho, electric, periods, interval, theta):
        self.rho, self.electric = rho, electric
        self.periods, self.interval, self.theta = periods, interval, theta
        self.closed = False

    def _interpolate(self, grid, points):
        scaled = np.mod(points[:, 1:] / self.periods, 1.) * GRID_SIZE
        lower = np.floor(scaled).astype(int)
        fraction = scaled - lower
        values = np.zeros(len(points), dtype=np.float64)
        for dx in (0, 1):
            for dy in (0, 1):
                weight = (fraction[:, 0] if dx else 1 - fraction[:, 0])
                weight = weight * (fraction[:, 1] if dy else 1 - fraction[:, 1])
                values += weight * grid[(lower[:, 0] + dx) % GRID_SIZE,
                                        (lower[:, 1] + dy) % GRID_SIZE]
        return values

    def evaluate(self, query):
        if self.closed:
            raise RuntimeError("simple VPFP baseline handle is closed")
        if not isinstance(query, AnswerQuery) or query.kind is not QueryKind.POINTS:
            raise ValueError("simple VPFP baseline accepts POINTS only")
        if not query.fields or len(set(query.fields)) != len(query.fields) or not set(query.fields) <= set(FIELDS):
            raise ValueError("unknown v2 transport field")
        points = np.asarray([(p["t"], p["x"], p["y"]) for p in query.points], dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.isfinite(points).all():
            raise ValueError("finite nonempty t,x,y points required")
        if np.any(points[:, 0] < self.interval[0]) or np.any(points[:, 0] > self.interval[1]):
            raise ValueError("query leaves prepared time interval")
        rho = self._interpolate(self.rho, points)
        zero = np.zeros(len(points), dtype=np.float64)
        values = {"rho": rho, "electric_x": self._interpolate(self.electric[0], points),
                  "electric_y": self._interpolate(self.electric[1], points),
                  "momentum_x": zero, "momentum_y": zero, "pressure_xx": self.theta * rho,
                  "pressure_xy": zero, "pressure_yy": self.theta * rho,
                  "heat_flux_x": zero, "heat_flux_y": zero}
        if any(not np.isfinite(v).all() for v in values.values()) or np.any(rho <= 0):
            raise ValueError("baseline produced invalid macros")
        return AnswerView(kind=query.kind, fields={k: tuple(map(float, values[k])) for k in query.fields},
                          points=query.points, shape=(len(points),))

    def close(self):
        self.closed = True


def prepare_solution(case, solve_budget=None):
    started = candidate_solve_clock()
    if solve_budget is not None and (isinstance(solve_budget, bool)
            or not isinstance(solve_budget, (int, float))
            or not isfinite(solve_budget) or solve_budget <= 0):
        raise ValueError("solve_budget must be finite and positive")

    def check_budget():
        if solve_budget is not None and candidate_solve_clock() - started > solve_budget:
            raise TimeoutError("simple VPFP preparation exceeded the remaining Case budget")

    question = case.public_data["question"]
    periods = np.asarray(question["grid"]["x_domain"], dtype=np.float64)
    interval = np.asarray(question["time_domain"], dtype=np.float64)
    theta = float(question["temperature"])
    if (periods.shape != (2,) or not np.isfinite(periods).all() or np.any(periods <= 0)
            or interval.shape != (2,) or not np.isfinite(interval).all() or interval[1] <= interval[0]
            or not isfinite(theta) or theta <= 0):
        raise ValueError("positive public periods/temperature and ordered time interval required")
    axes = [np.arange(GRID_SIZE) * length / GRID_SIZE for length in periods]
    grid = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([a.reshape(-1) for a in grid])
    limit = question.get("numerical_access", {}).get("max_points_per_request", len(points))
    if type(limit) is not int or limit <= 0:
        raise ValueError("public Oracle request limit must be positive")
    check_budget()
    session = open_public_case_session(case)
    chunks = []
    for start in range(0, len(points), limit):
        check_budget()
        batch = points[start:start + limit]
        rho = np.asarray(session.rho0(batch), dtype=np.float64)
        if rho.shape != (len(batch),) or not np.isfinite(rho).all() or np.any(rho <= 0):
            raise ValueError("public initial density must be finite and positive")
        chunks.append(rho)
    density = np.concatenate(chunks).reshape(GRID_SIZE, GRID_SIZE)
    wave = np.meshgrid(*(2*np.pi*np.fft.fftfreq(GRID_SIZE, d=length/GRID_SIZE)
                         for length in periods), indexing="ij")
    k2 = wave[0]**2 + wave[1]**2
    modes = np.fft.fftn(density - density.mean())
    potential = np.zeros_like(modes)
    np.divide(modes, k2, out=potential, where=k2 > 0)
    electric = tuple(np.fft.ifftn(-1j * k * potential).real for k in wave)
    check_budget()
    if any(not np.isfinite(value).all() for value in electric):
        raise ValueError("nonfinite baseline Poisson field")
    return Prepared(density, electric, periods, interval, theta)
