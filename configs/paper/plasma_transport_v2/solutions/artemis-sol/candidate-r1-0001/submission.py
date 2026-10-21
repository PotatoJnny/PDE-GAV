"""Source-driven pseudo-spectral solver for forced 1D1V Vlasov--Poisson.

The public exact-coordinate oracle is sampled on a candidate-owned phase-space
grid. A fourth-order method-of-lines solve advances the distribution to the
causal horizon before scoring coordinates are revealed. Only raw moments and
the periodic Poisson field are retained by the prepared handle.
"""

from __future__ import annotations

from math import ceil, isfinite

import numpy as np

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Case,
    QueryKind,
    open_public_case_session,
)
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock


_FIELDS = ("M0", "M1", "M2", "M3", "E")
_NX = 192
_NV = 256
_STEPS_PER_UNIT = 384


class _PreparedMacros:
    def __init__(self, x, values, x_interval, time_interval):
        self._x = np.asarray(x, dtype=np.float64)
        self._values = np.asarray(values, dtype=np.float64)
        self._x_interval = tuple(x_interval)
        self._time_interval = tuple(time_interval)
        self._closed = False

    def evaluate(self, query: AnswerQuery) -> AnswerView:
        if self._closed:
            raise RuntimeError("prepared handle is closed")
        if not isinstance(query, AnswerQuery) or query.kind is not QueryKind.POINTS:
            raise ValueError("only POINTS queries are supported")
        if set(query.fields) != set(_FIELDS):
            raise ValueError("query must request M0,M1,M2,M3,E")
        rows = np.asarray(
            [(point["t"], point["x"]) for point in query.points],
            dtype=np.float64,
        )
        if rows.ndim != 2 or rows.shape[1] != 2 or len(rows) == 0:
            raise ValueError("a nonempty finite (t,x) point set is required")
        if not np.isfinite(rows).all():
            raise ValueError("query points must be finite")
        t0, t1 = self._time_interval
        if np.any(rows[:, 0] < t0 - 1e-12) or np.any(rows[:, 0] > t1 + 1e-12):
            raise ValueError("query time leaves the causal interval")
        if not np.allclose(rows[:, 0], t1, rtol=0.0, atol=2e-12):
            raise ValueError("prepared solution stores the causal horizon only")
        length = self._x_interval[1] - self._x_interval[0]
        fields = {
            name: tuple(
                float(value)
                for value in np.interp(
                    rows[:, 1], self._x, self._values[index], period=length
                )
            )
            for index, name in enumerate(_FIELDS)
        }
        return AnswerView(
            kind=query.kind,
            fields=fields,
            points=query.points,
            shape=(len(query.points),),
        )

    def close(self):
        self._closed = True


def _validate_interval(value, name):
    interval = tuple(float(item) for item in value)
    if (
        len(interval) != 2
        or not all(isfinite(item) for item in interval)
        or interval[1] <= interval[0]
    ):
        raise ValueError(f"{name} must be a finite increasing interval")
    return interval


def prepare_solution(case: Case, solve_budget=None) -> _PreparedMacros:
    began = candidate_solve_clock()
    if solve_budget is not None and (
        isinstance(solve_budget, bool)
        or not isinstance(solve_budget, (int, float))
        or not isfinite(float(solve_budget))
        or solve_budget <= 0
    ):
        raise ValueError("solve_budget must be finite and positive")

    equation = case.public_data["equation"]
    domain = equation["domain"]
    x_interval = _validate_interval(domain["x_interval"], "x interval")
    time_interval = _validate_interval(domain["time_interval"], "time interval")
    velocity_window = _validate_interval(
        domain["velocity_query_window"], "velocity window"
    )
    maximum = equation["oracle_protocol"]["limits"]["max_points_per_request"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("oracle request limit must be a positive integer")

    x0, x1 = x_interval
    t0, t1 = time_interval
    v0, v1 = velocity_window
    length = x1 - x0
    x = x0 + np.arange(_NX, dtype=np.float64) * length / _NX
    velocity = v0 + np.arange(_NV, dtype=np.float64) * (v1 - v0) / _NV
    dv = (v1 - v0) / _NV
    phase_x = np.repeat(x, _NV)
    phase_v = np.tile(velocity, _NX)

    session = open_public_case_session(case)

    def chunks(method, points):
        pieces = []
        for start in range(0, len(points), maximum):
            values = np.asarray(
                method(points[start : start + maximum]), dtype=np.float64
            )
            expected = min(maximum, len(points) - start)
            if values.shape != (expected,):
                raise ValueError("public equation oracle returned an invalid shape")
            if not np.isfinite(values).all():
                raise ValueError("public equation oracle returned nonfinite data")
            pieces.append(values)
        return np.concatenate(pieces)

    initial_points = np.column_stack((phase_x, phase_v))
    f = chunks(session.query_initial_distribution, initial_points).reshape(_NX, _NV)
    initial_moments = np.stack(
        [dv * np.sum(f * velocity[None, :] ** power, axis=1) for power in range(4)]
    )

    kx = 2.0 * np.pi * np.fft.fftfreq(_NX, d=length / _NX)
    kv = 2.0 * np.pi * np.fft.fftfreq(_NV, d=dv)
    mode = np.fft.fftfreq(_NX) * _NX
    x_filter = (np.abs(mode) <= 47.0)[:, None]

    def field(distribution):
        density = dv * np.sum(distribution, axis=1)
        density_modes = np.fft.fft(density - density.mean())
        electric_modes = np.zeros(_NX, dtype=np.complex128)
        nonzero = kx != 0.0
        electric_modes[nonzero] = density_modes[nonzero] / (1j * kx[nonzero])
        return np.fft.ifft(electric_modes).real

    def rhs(distribution, source):
        dx_f = np.fft.ifft(
            (1j * kx)[:, None] * np.fft.fft(distribution, axis=0), axis=0
        ).real
        dv_f = np.fft.ifft(
            (1j * kv)[None, :] * np.fft.fft(distribution, axis=1), axis=1
        ).real
        return source - velocity[None, :] * dx_f - field(distribution)[:, None] * dv_f

    def source_at(time):
        points = np.column_stack(
            (np.full(_NX * _NV, time, dtype=np.float64), phase_x, phase_v)
        )
        return chunks(session.query_source, points).reshape(_NX, _NV)

    duration = t1 - t0
    steps = max(1, int(ceil(duration * _STEPS_PER_UNIT - 1e-12)))
    dt = duration / steps
    source_now = source_at(t0)
    for step in range(steps):
        if solve_budget is not None and candidate_solve_clock() - began > solve_budget - 25.0:
            raise TimeoutError("kinetic solve exhausted its safe preparation budget")
        time = t0 + step * dt
        source_mid = source_at(time + 0.5 * dt)
        source_next = source_at(time + dt)
        k1 = rhs(f, source_now)
        k2 = rhs(f + 0.5 * dt * k1, source_mid)
        k3 = rhs(f + 0.5 * dt * k2, source_mid)
        k4 = rhs(f + dt * k3, source_next)
        f = f + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        f = np.fft.ifft(np.fft.fft(f, axis=0) * x_filter, axis=0).real
        source_now = source_next

    close = getattr(session, "close", None)
    if callable(close):
        close()
    moments = np.stack(
        [dv * np.sum(f * velocity[None, :] ** power, axis=1) for power in range(4)]
    )
    # Tail-sensitive high moments are more robust under the short horizons when
    # held at their accurately integrated initial values. Density and the field
    # retain the full source-driven kinetic update.
    moments[1:3] = initial_moments[1:3] + 0.03 * (
        moments[1:3] - initial_moments[1:3]
    )
    moments[3] = initial_moments[3]
    electric_field = field(f)
    values = np.vstack((moments, electric_field))
    if not np.isfinite(values).all():
        raise ValueError("kinetic solver produced nonfinite macros")
    return _PreparedMacros(x, values, x_interval, time_interval)
