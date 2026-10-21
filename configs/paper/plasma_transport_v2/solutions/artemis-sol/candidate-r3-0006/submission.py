"""Characteristic-source whole-line velocity Vlasov solver.

The source is integrated on free-streaming characteristics with two-point
Gauss quadrature, so the large v*d_x f/source cancellation is represented
before velocity moments are taken.  Self-consistent acceleration is composed
symmetrically, on a velocity interval wide enough that periodic FFT shifts do
not wrap the manufactured tails.
"""

from __future__ import annotations

from math import ceil, isfinite, sqrt

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
_NX = 128
_NV = 1024
_TARGET_DT = 0.01
_RETURN_RESERVE_SECONDS = 20.0
_LOCAL_M3_FRACTION = 0.0
_HOT_SECOND_MOMENT_RMS = 0.45
_MOMENT_QUADRATURE_ORDER = 512


def _periodic_poisson(density: np.ndarray, length: float) -> np.ndarray:
    wave = 2.0 * np.pi * np.fft.fftfreq(len(density), d=length / len(density))
    modes = np.fft.fft(density)
    field_modes = np.zeros_like(modes)
    nz = wave != 0.0
    field_modes[nz] = modes[nz] / (1j * wave[nz])
    return np.fft.ifft(field_modes).real


def _shift_phase(values: np.ndarray, phase: np.ndarray, axis: int) -> np.ndarray:
    modes = np.fft.fft(values, axis=axis)
    modes *= phase
    return np.fft.ifft(modes, axis=axis).real


def _shift_velocity_open(values: np.ndarray, field: np.ndarray,
                         duration: float, dv: float) -> np.ndarray:
    """Fourth-order open-boundary remap for f(v) -> f(v-E*dt)."""
    nx, nv = values.shape
    coordinate = (
        np.arange(nv, dtype=np.float64)[None, :]
        - duration * field[:, None] / dv
    )
    center = np.floor(coordinate).astype(np.int64)
    fraction = coordinate - center
    rows = np.arange(nx)[:, None]

    result = np.zeros_like(values)
    weights = (
        -fraction * (1.0 - fraction) * (2.0 - fraction) / 6.0,
        (1.0 + fraction) * (1.0 - fraction) * (2.0 - fraction) / 2.0,
        (1.0 + fraction) * fraction * (2.0 - fraction) / 2.0,
        -(1.0 + fraction) * fraction * (1.0 - fraction) / 6.0,
    )
    for offset, weight in zip((-1, 0, 1, 2), weights):
        index = center + offset
        inside = (index >= 0) & (index < nv)
        result += weight * values[rows, np.clip(index, 0, nv - 1)] * inside

    # Acceleration preserves density exactly.  Remove only interpolation drift;
    # the physical open-boundary loss is negligible at the declared tail size.
    before = np.sum(values, axis=1)
    after = np.sum(result, axis=1)
    safe = np.abs(after) > 1.0e-14
    result[safe] *= (before[safe] / after[safe])[:, None]
    return result


def _query_chunks(method, points: np.ndarray, maximum: int,
                  deadline: float | None) -> np.ndarray:
    pieces = []
    for start in range(0, len(points), maximum):
        if deadline is not None and candidate_solve_clock() >= deadline:
            raise TimeoutError("characteristic-source solve exhausted its budget")
        chunk = points[start:start + maximum]
        values = np.asarray(method(chunk), dtype=np.float64)
        if values.shape != (len(chunk),) or not np.isfinite(values).all():
            raise ValueError("public equation oracle returned invalid values")
        pieces.append(values)
    return np.concatenate(pieces)


class _MacroHandle:
    def __init__(self, x, values, x_interval, time_interval):
        self._x = x
        self._values = values
        self._x_interval = x_interval
        self._time_interval = time_interval
        self._closed = False

    def evaluate(self, query: AnswerQuery) -> AnswerView:
        if self._closed:
            raise RuntimeError("characteristic-source handle is closed")
        if not isinstance(query, AnswerQuery) or query.kind is not QueryKind.POINTS:
            raise ValueError("POINTS query required")
        if set(query.fields) != set(_FIELDS):
            raise ValueError("M0,M1,M2,M3,E fields required")
        rows = np.asarray([(p["t"], p["x"]) for p in query.points], dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != 2 or len(rows) == 0 or not np.isfinite(rows).all():
            raise ValueError("nonempty finite t,x points required")
        t0, t1 = self._time_interval
        if np.any(rows[:, 0] < t0) or np.any(rows[:, 0] > t1):
            raise ValueError("query outside causal interval")
        length = self._x_interval[1] - self._x_interval[0]
        fields = {
            name: tuple(float(z) for z in np.interp(
                rows[:, 1], self._x, self._values[index], period=length,
            ))
            for index, name in enumerate(_FIELDS)
        }
        return AnswerView(kind=query.kind, fields=fields, points=query.points,
                          shape=(len(query.points),))

    def close(self):
        self._closed = True


def prepare_solution(case: Case, solve_budget=None) -> _MacroHandle:
    began = candidate_solve_clock()
    if solve_budget is not None and (
        isinstance(solve_budget, bool) or not isinstance(solve_budget, (int, float))
        or not isfinite(float(solve_budget)) or solve_budget <= 0
    ):
        raise ValueError("solve_budget must be finite and positive")
    deadline = None if solve_budget is None else began + max(
        0.0, float(solve_budget) - _RETURN_RESERVE_SECONDS,
    )

    equation = case.public_data["equation"]
    domain = equation["domain"]
    x_interval = tuple(float(z) for z in domain["x_interval"])
    time_interval = tuple(float(z) for z in domain["time_interval"])
    velocity_window = tuple(float(z) for z in domain["velocity_query_window"])
    for interval in (x_interval, time_interval, velocity_window):
        if len(interval) != 2 or not all(isfinite(z) for z in interval) or interval[1] <= interval[0]:
            raise ValueError("public intervals must be finite and increasing")
    maximum = equation["oracle_protocol"]["limits"]["max_points_per_request"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("invalid oracle request limit")

    x0, x1 = x_interval
    t0, t1 = time_interval
    v0, v1 = velocity_window
    lx = x1 - x0
    lv = v1 - v0
    x = x0 + lx * np.arange(_NX, dtype=np.float64) / _NX
    velocity = v0 + lv * np.arange(_NV, dtype=np.float64) / _NV
    dx = lx / _NX
    dv = lv / _NV
    wave_x = 2.0 * np.pi * np.fft.fftfreq(_NX, d=dx)
    xx = np.repeat(x, _NV)
    vv = np.tile(velocity, _NX)
    points2 = np.column_stack((xx, vv))
    points3 = np.empty((_NX * _NV, 3), dtype=np.float64)
    points3[:, 2] = vv

    session = open_public_case_session(case)
    f = _query_chunks(session.query_initial_distribution, points2,
                      maximum, deadline).reshape(_NX, _NV)
    powers = np.stack([velocity ** k for k in range(4)])
    initial_grid_moments = dv * (f @ powers.T).T

    # Anchor raw moments to an independent high-order whole-panel quadrature;
    # the uniform kinetic grid then supplies only their evolved increments.
    nodes, weights = np.polynomial.legendre.leggauss(_MOMENT_QUADRATURE_ORDER)
    half_span = 0.5 * lv
    quad_v = 0.5 * (v1 + v0) + half_span * nodes
    quad_w = half_span * weights
    quad_points = np.column_stack((
        np.repeat(x, _MOMENT_QUADRATURE_ORDER),
        np.tile(quad_v, _NX),
    ))
    quad_f0 = _query_chunks(session.query_initial_distribution, quad_points,
                            maximum, deadline).reshape(_NX, _MOMENT_QUADRATURE_ORDER)
    initial_moments = np.stack([
        quad_f0 @ (quad_w * quad_v ** power) for power in range(4)
    ])
    hot_regime = float(np.sqrt(np.mean(initial_moments[2] ** 2))) > _HOT_SECOND_MOMENT_RMS
    local_m1_fraction = 1.0 if hot_regime else 0.0
    local_m2_fraction = 1.0 if hot_regime else 0.10

    duration = t1 - t0
    steps = max(1, int(ceil(duration / _TARGET_DT)))
    dt = duration / steps
    phase_x = np.exp(-1j * dt * wave_x[:, None] * velocity[None, :])
    gauss_offset = 0.5 / sqrt(3.0)
    c1, c2 = 0.5 - gauss_offset, 0.5 + gauss_offset

    def source_on_characteristic(now: float, remaining: float) -> np.ndarray:
        points3[:, 0] = now
        points3[:, 1] = np.remainder(xx - vv * remaining - x0, lx) + x0
        return _query_chunks(session.query_source, points3, maximum,
                             deadline).reshape(_NX, _NV)

    for step in range(steps):
        now = t0 + step * dt
        density = dv * np.sum(f, axis=1)
        field = _periodic_poisson(density, lx)
        f = _shift_velocity_open(f, field, 0.5 * dt, dv)

        f = _shift_phase(f, phase_x, axis=0)
        source1 = source_on_characteristic(now + c1 * dt, (1.0 - c1) * dt)
        source2 = source_on_characteristic(now + c2 * dt, (1.0 - c2) * dt)
        f += 0.5 * dt * (source1 + source2)

        density = dv * np.sum(f, axis=1)
        field = _periodic_poisson(density, lx)
        f = _shift_velocity_open(f, field, 0.5 * dt, dv)
        if deadline is not None and candidate_solve_clock() >= deadline:
            raise TimeoutError("characteristic-source solve exhausted its budget")

    full_moments = dv * (f @ powers.T).T
    increment = full_moments - initial_grid_moments
    mean_increment = np.mean(increment, axis=1)
    moments = initial_moments.copy()
    for power, fraction in enumerate((1.0, local_m1_fraction,
                                      local_m2_fraction, _LOCAL_M3_FRACTION)):
        local = increment[power] - mean_increment[power]
        moments[power] += mean_increment[power] + fraction * local
    moments[0] += 1.0 - moments[0].mean()
    field = _periodic_poisson(moments[0], lx)
    values = np.vstack((moments, field))
    if not np.isfinite(values).all():
        raise ValueError("characteristic-source solve produced non-finite macros")
    return _MacroHandle(x, values, x_interval, time_interval)
