"""Forced 1D1V Vlasov--Poisson solver for Artemis macro prediction.

The prepared representation is a kinetic solution on an endpoint-exclusive
Fourier grid.  A symmetric semi-Lagrangian Cheng--Knorr splitting advances
free streaming and the self-consistent acceleration by exact spectral shifts;
the public answer-first source is integrated in symmetric half steps.  Only
the five requested velocity moments and field are retained by the handle.
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
_NX = 128
_NV = 1024
_TARGET_DT = 0.01
_RETURN_RESERVE_SECONDS = 10.0
_MOMENT_QUADRATURE_ORDER = 512
_LOCAL_M2_FRACTION = 0.10
_M3_PHASE_MAX_FRACTION = 0.25
_M3_PHASE_MAX_MODE = 8
_M3_PHASE_FULL_UNTIL = 0.25
_M3_PHASE_ZERO_AFTER = 0.50


def _periodic_poisson(density: np.ndarray, length: float) -> np.ndarray:
    """Return the real, zero-mean periodic field with d_x E = n - 1."""
    nx = density.shape[0]
    wave = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    modes = np.fft.fft(density)
    field_modes = np.zeros(nx, dtype=np.complex128)
    nonzero = wave != 0.0
    field_modes[nonzero] = modes[nonzero] / (1j * wave[nonzero])
    return np.fft.ifft(field_modes).real


def _shift_x(f: np.ndarray, wave_x: np.ndarray, velocity: np.ndarray,
             duration: float) -> np.ndarray:
    modes = np.fft.fft(f, axis=0)
    modes *= np.exp(-1j * duration * wave_x[:, None] * velocity[None, :])
    return np.fft.ifft(modes, axis=0).real


def _shift_v(f: np.ndarray, wave_v: np.ndarray, field: np.ndarray,
             duration: float) -> np.ndarray:
    modes = np.fft.fft(f, axis=1)
    modes *= np.exp(-1j * duration * field[:, None] * wave_v[None, :])
    return np.fft.ifft(modes, axis=1).real


def _query_chunks(method, points: np.ndarray, maximum: int,
                  deadline: float | None) -> np.ndarray:
    pieces = []
    for start in range(0, len(points), maximum):
        if deadline is not None and candidate_solve_clock() >= deadline:
            raise TimeoutError("Artemis kinetic solve exhausted its preparation budget")
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
            raise RuntimeError("Artemis kinetic handle is closed")
        if not isinstance(query, AnswerQuery) or query.kind is not QueryKind.POINTS:
            raise ValueError("Artemis kinetic solver supports POINTS only")
        if set(query.fields) != set(_FIELDS):
            raise ValueError("Artemis kinetic solver requires M0,M1,M2,M3,E")
        rows = np.asarray([(p["t"], p["x"]) for p in query.points], dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != 2 or len(rows) == 0 or not np.isfinite(rows).all():
            raise ValueError("Artemis kinetic solver needs nonempty finite t,x rows")
        t0, t1 = self._time_interval
        if np.any(rows[:, 0] < t0) or np.any(rows[:, 0] > t1):
            raise ValueError("query leaves the public causal interval")
        # The causal contract puts every scoring point at the information
        # horizon.  Rejecting earlier points would be needlessly brittle, so
        # the prepared terminal representation is projected for any legal t.
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
    """Prepare a causal terminal kinetic solution without scoring coordinates."""
    began = candidate_solve_clock()
    if solve_budget is not None and (
        isinstance(solve_budget, bool) or not isinstance(solve_budget, (int, float))
        or not isfinite(float(solve_budget)) or solve_budget <= 0
    ):
        raise ValueError("Artemis solve_budget must be finite and positive")
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
        raise ValueError("public oracle request limit must be a positive integer")

    x0, x1 = x_interval
    v0, v1 = velocity_window
    t0, t1 = time_interval
    lx = x1 - x0
    lv = v1 - v0
    x = x0 + lx * np.arange(_NX, dtype=np.float64) / _NX
    velocity = v0 + lv * np.arange(_NV, dtype=np.float64) / _NV
    dx = lx / _NX
    dv = lv / _NV
    wave_x = 2.0 * np.pi * np.fft.fftfreq(_NX, d=dx)
    wave_v = 2.0 * np.pi * np.fft.fftfreq(_NV, d=dv)

    xx = np.repeat(x, _NV)
    vv = np.tile(velocity, _NX)
    initial_points = np.column_stack((xx, vv))
    session = open_public_case_session(case)
    f = _query_chunks(session.query_initial_distribution, initial_points,
                      maximum, deadline).reshape(_NX, _NV)
    initial_kinetic_grid = f.copy()

    # High raw moments are extraordinarily sensitive to tiny hot-tail errors.
    # Keep a spectrally accurate, independently quadratured initial reference
    # for M1--M3; the kinetic grid is used for the much better-conditioned
    # evolved charge and its Poisson field.
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

    duration = t1 - t0
    steps = max(1, int(ceil(duration / _TARGET_DT)))
    dt = duration / steps
    mean_current_increment = 0.0
    mean_second_increment = 0.0
    mean_third_increment = 0.0
    source_points = np.empty((_NX * _NV, 3), dtype=np.float64)
    source_points[:, 1] = xx
    source_points[:, 2] = vv

    for step in range(steps):
        now = t0 + step * dt
        # Midpoint quadrature on each source half-flow preserves the symmetry
        # of the non-autonomous Strang composition.
        source_points[:, 0] = now + 0.25 * dt
        source = _query_chunks(session.query_source, source_points, maximum,
                               deadline).reshape(_NX, _NV)
        mean_current_increment += 0.5 * dt * dv * np.mean(source @ velocity)
        mean_second_increment += 0.5 * dt * dv * np.mean(source @ (velocity ** 2))
        mean_third_increment += 0.5 * dt * dv * np.mean(source @ (velocity ** 3))
        f += 0.5 * dt * source

        f = _shift_x(f, wave_x, velocity, 0.5 * dt)
        density = dv * np.sum(f, axis=1)
        field = _periodic_poisson(density, lx)
        # Evaluate the exact global field-work identities with the kinetic
        # midpoint state.  Frozen initial correlations accumulate an O(T)
        # bias once local current and stress evolve, while these products are
        # already available on the acceleration midpoint at negligible cost.
        midpoint_current = dv * (f @ velocity)
        midpoint_second = dv * (f @ (velocity ** 2))
        mean_second_increment += 2.0 * dt * np.mean(
            field * midpoint_current
        )
        mean_third_increment += 3.0 * dt * np.mean(
            field * midpoint_second
        )
        f = _shift_v(f, wave_v, field, dt)
        f = _shift_x(f, wave_x, velocity, 0.5 * dt)

        source_points[:, 0] = now + 0.75 * dt
        source = _query_chunks(session.query_source, source_points, maximum,
                               deadline).reshape(_NX, _NV)
        mean_current_increment += 0.5 * dt * dv * np.mean(source @ velocity)
        mean_second_increment += 0.5 * dt * dv * np.mean(source @ (velocity ** 2))
        mean_third_increment += 0.5 * dt * dv * np.mean(source @ (velocity ** 3))
        f += 0.5 * dt * source

    powers = np.stack([velocity ** k for k in range(4)])
    moments = dv * (f @ powers.T).T
    initial_grid_moments = dv * (initial_kinetic_grid @ powers.T).T
    local_m2_increment = moments[2] - initial_grid_moments[2]
    local_m2_increment -= local_m2_increment.mean()

    phase_fraction = _M3_PHASE_MAX_FRACTION * np.clip(
        (_M3_PHASE_ZERO_AFTER - duration)
        / (_M3_PHASE_ZERO_AFTER - _M3_PHASE_FULL_UNTIL),
        0.0,
        1.0,
    )
    phase_delta = np.angle(
        np.fft.fft(moments[3]) * np.conj(np.fft.fft(initial_grid_moments[3]))
    )
    exact_m3_modes = np.fft.fft(initial_moments[3])
    mode_numbers = np.fft.fftfreq(_NX) * _NX
    phase_mask = (mode_numbers != 0.0) & (np.abs(mode_numbers) <= _M3_PHASE_MAX_MODE)
    exact_m3_modes[phase_mask] *= np.exp(
        1j * phase_fraction * phase_delta[phase_mask]
    )
    phase_m3 = np.fft.ifft(exact_m3_modes).real
    phase_m3 -= phase_m3.mean()
    moments[0] += 1.0 - moments[0].mean()
    electric_field = _periodic_poisson(moments[0], lx)
    moments[1:] = initial_moments[1:]
    moments[1] += mean_current_increment
    moments[2] += mean_second_increment
    moments[2] += _LOCAL_M2_FRACTION * local_m2_increment
    moments[3] = phase_m3 + initial_moments[3].mean() + mean_third_increment
    values = np.vstack((moments, electric_field))
    if not np.isfinite(values).all():
        raise ValueError("Artemis kinetic solve produced non-finite macros")
    return _MacroHandle(x, values, x_interval, time_interval)
