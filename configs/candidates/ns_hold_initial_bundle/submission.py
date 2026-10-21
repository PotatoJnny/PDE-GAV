"""Weak physical NS cold start for the formal continuous POINTS ABI.

The candidate queries only the public initial condition during query-blind
preparation, builds its own spatial grid, and holds that velocity in time.
It is deliberately a baseline, not a claim of an accurate Navier--Stokes
solver.  Grid interpolation preserves the sampled outer no-slip trace and the
initial divergence-free field up to interpolation error.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

import numpy as np

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Case,
    QueryKind,
    open_public_case_session,
)


_VELOCITY_FIELDS = ("velocity_x", "velocity_y")


class _HoldInitialVelocity:
    def __init__(self, case: Case, x, y, velocity) -> None:
        self._case_id = case.case_id
        self._x = np.asarray(x, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.float64)
        self._velocity = np.asarray(velocity, dtype=np.float64)
        domain = case.public_data["domain"]
        self._time_interval = tuple(float(value) for value in domain["time_interval"])

    def evaluate(self, query: AnswerQuery) -> AnswerView:
        if not isinstance(query, AnswerQuery):
            raise TypeError("NS hold-initial projection needs AnswerQuery")
        if query.kind is not QueryKind.POINTS:
            raise ValueError("NS formal cold start supports POINTS queries only")
        if set(query.fields) != set(_VELOCITY_FIELDS):
            raise ValueError("NS formal cold start needs both velocity components")
        coordinates = np.asarray(
            [
                [float(point[name]) for name in ("t", "x", "y")]
                for point in query.points
            ],
            dtype=np.float64,
        )
        if coordinates.shape != (len(query.points), 3):
            raise ValueError("NS scoring points must use physical (t,x,y)")
        if not np.isfinite(coordinates).all():
            raise ValueError("NS scoring points must be finite")
        t = coordinates[:, 0]
        if np.any(t < self._time_interval[0]) or np.any(t > self._time_interval[1]):
            raise ValueError("NS scoring time leaves the public interval")
        velocity = _bilinear_vector(
            self._x,
            self._y,
            self._velocity,
            coordinates[:, 1],
            coordinates[:, 2],
        )
        return AnswerView(
            kind=QueryKind.POINTS,
            fields={
                "velocity_x": tuple(float(value) for value in velocity[:, 0]),
                "velocity_y": tuple(float(value) for value in velocity[:, 1]),
            },
            points=query.points,
            shape=(len(query.points),),
        )


def prepare_solution(
    case: Case,
    solve_budget: float | None = None,
) -> _HoldInitialVelocity:
    """Prepare an Agent-owned initial-condition interpolant before query reveal."""

    if not isinstance(case, Case):
        raise TypeError("NS hold-initial baseline needs a Case")
    _validate_budget(solve_budget)
    domain = case.public_data.get("domain")
    if not isinstance(domain, Mapping):
        raise ValueError("NS hold-initial baseline needs a public domain")
    bounds = domain.get("bounds")
    if not isinstance(bounds, Mapping):
        raise ValueError("NS hold-initial baseline needs public spatial bounds")
    x0, x1 = (float(value) for value in bounds["x"])
    y0, y1 = (float(value) for value in bounds["y"])
    resolution = 33 if solve_budget is None or float(solve_budget) >= 0.05 else 17
    x = np.linspace(x0, x1, resolution, dtype=np.float64)
    y = np.linspace(y0, y1, resolution, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack((xx.reshape(-1), yy.reshape(-1)))

    protocol = case.public_data.get("oracle_protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("NS hold-initial baseline needs the public oracle contract")
    limits = protocol.get("limits")
    if not isinstance(limits, Mapping):
        raise ValueError("NS hold-initial baseline needs public oracle limits")
    maximum = int(limits["max_points_per_request"])
    if maximum < 1:
        raise ValueError("NS public oracle point limit must be positive")

    session = open_public_case_session(case)
    pieces = [
        np.asarray(
            session.query_initial_velocity(points[start : start + maximum]),
            dtype=np.float64,
        )
        for start in range(0, len(points), maximum)
    ]
    velocity = np.concatenate(pieces, axis=0).reshape(resolution, resolution, 2)
    if not np.isfinite(velocity).all():
        raise ValueError("NS public initial velocity is non-finite")
    return _HoldInitialVelocity(case, x, y, velocity)


def _bilinear_vector(x_axis, y_axis, values, x, y):
    if np.any(x < x_axis[0]) or np.any(x > x_axis[-1]):
        raise ValueError("NS scoring x leaves the public domain")
    if np.any(y < y_axis[0]) or np.any(y > y_axis[-1]):
        raise ValueError("NS scoring y leaves the public domain")
    x_upper = np.searchsorted(x_axis, x, side="right")
    y_upper = np.searchsorted(y_axis, y, side="right")
    x_upper = np.clip(x_upper, 1, len(x_axis) - 1)
    y_upper = np.clip(y_upper, 1, len(y_axis) - 1)
    x_lower = x_upper - 1
    y_lower = y_upper - 1
    x_weight = (x - x_axis[x_lower]) / (x_axis[x_upper] - x_axis[x_lower])
    y_weight = (y - y_axis[y_lower]) / (y_axis[y_upper] - y_axis[y_lower])
    return (
        values[x_lower, y_lower] * ((1.0 - x_weight) * (1.0 - y_weight))[:, None]
        + values[x_upper, y_lower] * (x_weight * (1.0 - y_weight))[:, None]
        + values[x_lower, y_upper] * ((1.0 - x_weight) * y_weight)[:, None]
        + values[x_upper, y_upper] * (x_weight * y_weight)[:, None]
    )


def _validate_budget(value: float | None) -> None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError("candidate solve budget must be finite and positive")
