"""Weak, query-blind initial-moment baseline for canonical Artemis.

Only public f0 is used: integrate over the declared finite velocity query
window, solve periodic Poisson for that initial density, and hold all five
macros fixed in time. This deliberately ignores source-driven evolution and
the unqueried velocity tails. It is an auditable S0, not a qualified solver.
The bundle needs only NumPy and the public SDK; it owns no model assets.
"""

from __future__ import annotations

from math import isfinite
from time import perf_counter

import numpy as np

from pde_agent_verifier_solver.sdk import (
    AnswerQuery, AnswerView, Case, QueryKind, open_public_case_session,
)


_FIELDS = ("M0", "M1", "M2", "M3", "E")
_VELOCITY_QUADRATURE_ORDER = 128


class _InitialMacros:
    def __init__(self, x, values, x_interval, time_interval):
        self._x = x
        self._values = values
        self._x_interval = x_interval
        self._time_interval = time_interval
        self._closed = False

    def evaluate(self, query: AnswerQuery) -> AnswerView:
        if self._closed:
            raise RuntimeError("Artemis prepared handle is closed")
        if not isinstance(query, AnswerQuery) or query.kind is not QueryKind.POINTS:
            raise ValueError("Artemis hold-initial supports POINTS only")
        if set(query.fields) != set(_FIELDS):
            raise ValueError("Artemis hold-initial requires M0,M1,M2,M3,E")
        rows = np.asarray([(point["t"], point["x"]) for point in query.points], dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != 2 or not len(rows) or not np.isfinite(rows).all():
            raise ValueError("Artemis hold-initial needs nonempty finite t,x rows")
        t0, t1 = self._time_interval
        if np.any(rows[:, 0] < t0) or np.any(rows[:, 0] > t1):
            raise ValueError("Artemis hold-initial query leaves the causal interval")
        length = self._x_interval[1] - self._x_interval[0]
        fields = {
            name: tuple(float(value) for value in np.interp(
                rows[:, 1], self._x, self._values[index], period=length,
            ))
            for index, name in enumerate(_FIELDS)
        }
        return AnswerView(kind=query.kind, fields=fields, points=query.points,
                          shape=(len(query.points),))

    def close(self):
        self._closed = True


def prepare_solution(case: Case, solve_budget=None) -> _InitialMacros:
    """Compute every native-x initial macro before scoring points are known."""
    began = perf_counter()
    if solve_budget is not None and (
        isinstance(solve_budget, bool) or not isinstance(solve_budget, (int, float))
        or not isfinite(float(solve_budget)) or solve_budget <= 0
    ):
        raise ValueError("Artemis solve_budget must be finite and positive")
    equation = case.public_data["equation"]
    domain = equation["domain"]
    x_interval = tuple(float(value) for value in domain["x_interval"])
    time_interval = tuple(float(value) for value in domain["time_interval"])
    velocity_window = tuple(float(value) for value in domain["velocity_query_window"])
    for interval in (x_interval, time_interval, velocity_window):
        if len(interval) != 2 or not all(isfinite(value) for value in interval) or interval[1] <= interval[0]:
            raise ValueError("Artemis public intervals must be finite and increasing")
    x = np.array(case.public_data["arrays"]["x"], dtype=np.float64, copy=True)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Artemis hold-initial requires a finite native spatial panel")
    length = x_interval[1] - x_interval[0]
    expected_x = x_interval[0] + np.arange(len(x)) * length / len(x)
    if not np.allclose(x, expected_x, rtol=1e-10, atol=1e-12):
        raise ValueError("Artemis hold-initial requires a uniform endpoint-exclusive periodic panel")
    nodes, weights = np.polynomial.legendre.leggauss(_VELOCITY_QUADRATURE_ORDER)
    half_span = (velocity_window[1] - velocity_window[0]) / 2
    velocity = (velocity_window[1] + velocity_window[0]) / 2 + half_span * nodes
    weights = half_span * weights
    points = np.column_stack((np.repeat(x, len(velocity)), np.tile(velocity, len(x))))
    maximum = equation["oracle_protocol"]["limits"]["max_points_per_request"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("Artemis initial oracle needs a positive integer request limit")
    session = open_public_case_session(case)
    samples = []
    for start in range(0, len(points), maximum):
        if solve_budget is not None and perf_counter() - began > solve_budget:
            raise TimeoutError("Artemis hold-initial exceeded preparation budget")
        chunk = points[start:start + maximum]
        values = np.asarray(session.query_initial_distribution(chunk), dtype=np.float64)
        if values.shape != (len(chunk),) or not np.isfinite(values).all():
            raise ValueError("Artemis public initial oracle returned invalid values")
        samples.append(values)
    f0 = np.concatenate(samples).reshape(len(x), len(velocity))
    moments = np.stack([f0 @ (weights * velocity**power) for power in range(4)])
    wave_numbers = 2 * np.pi * np.fft.fftfreq(len(x), d=length / len(x))
    density_modes = np.fft.fft(moments[0] - moments[0].mean())
    field_modes = np.zeros(len(x), dtype=np.complex128)
    nonzero = wave_numbers != 0
    field_modes[nonzero] = density_modes[nonzero] / (1j * wave_numbers[nonzero])
    electric_field = np.fft.ifft(field_modes).real
    values = np.vstack((moments, electric_field))
    if not np.isfinite(values).all():
        raise ValueError("Artemis hold-initial produced non-finite macros")
    return _InitialMacros(x, values, x_interval, time_interval)
