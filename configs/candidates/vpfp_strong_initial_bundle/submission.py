"""Query-blind continuous wrapper around the frozen strong VPFP S0.

The original numerical algorithm is in legacy_solver.py unchanged. This
candidate, not the framework, chooses its 13x33x33x9x9 numerical grid. All
source/g0/rho0 samples are requested through the public continuous Oracle
during preparation. Its finite macro solution is subsequently interpolated
linearly in time and bilinearly on the spatial torus for arbitrary POINTS.
No exact-continuous-solver or interpolation-accuracy qualification is claimed.
"""
from __future__ import annotations

from math import isfinite, prod
from time import perf_counter

import numpy as np

from pde_agent_verifier_solver.sdk import (
    AnswerQuery, AnswerView, Case, CaseSource, CaseVisibility, QueryKind,
    open_public_case_session,
)
from . import legacy_solver


FIELDS = ("rho", "momentum_x", "momentum_y", "kinetic_energy")
SHAPE = (13, 33, 33, 9, 9)
CHUNK_SIZE = 8192
LEGACY_SOLVER_SHA256 = "c3fd9984dc05d1a8ecdb698e7c63861e98f201cadb0cfba9ad1b78d9e20867bd"


def _numeric(value, shape, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.shape != shape:
        raise ValueError(f"S0 {name} must be a real array of shape {shape}")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"S0 {name} contains nonfinite values")
    return result


def _chunks(axes, chunk_size):
    shape = tuple(len(axis) for axis in axes)
    for start in range(0, prod(shape), chunk_size):
        indices = np.unravel_index(np.arange(start, min(start+chunk_size, prod(shape))), shape)
        yield start, np.column_stack([axis[index] for axis, index in zip(axes, indices)])


class _PreparedMacros:
    def __init__(self, t, x, y, fields):
        self._axes = tuple(np.array(a, dtype=float, copy=True) for a in (t, x, y))
        if any(a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all()
               or not np.all(np.diff(a) > 0) for a in self._axes):
            raise ValueError("S0 prepared axes must be finite and increasing")
        if any(not np.allclose(np.diff(a), a[1]-a[0], rtol=1e-12, atol=0)
               for a in self._axes):
            raise ValueError("S0 interpolation requires uniform axes")
        shape = tuple(len(a) for a in self._axes)
        self._fields = {name: _numeric(fields[name], shape, name) for name in FIELDS}
        self._closed = False

    def evaluate(self, query):
        if self._closed:
            raise RuntimeError("S0 prepared solution has been closed")
        if not isinstance(query, AnswerQuery) or query.kind is not QueryKind.POINTS:
            raise ValueError("S0 prepared solution accepts POINTS only")
        if not query.fields or len(set(query.fields)) != len(query.fields) or not set(query.fields) <= set(FIELDS):
            raise ValueError("S0 predicts only the declared VPFP macro fields")
        points = np.asarray([(p["t"], p["x"], p["y"]) for p in query.points], dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError("S0 requires nonempty finite t,x,y points")
        t, x, y = self._axes
        if np.any(points[:, 0] < t[0]) or np.any(points[:, 0] > t[-1]):
            raise ValueError("S0 query time leaves the prepared interval")
        indices, fractions = [], []
        for column, axis in enumerate(self._axes):
            coordinate = (points[:, column]-axis[0])/(axis[1]-axis[0])
            if column:
                coordinate = np.mod(coordinate, len(axis))
                lower = np.floor(coordinate).astype(int)
            else:
                lower = np.clip(np.floor(coordinate).astype(int), 0, len(axis)-2)
            indices.append(lower)
            fractions.append(coordinate-lower)
        it, ix, iy = indices
        at, ax, ay = fractions
        fields = {}
        for name in query.fields:
            values = self._fields[name]
            prediction = np.zeros(len(points), dtype=float)
            for dt in (0, 1):
                for dx in (0, 1):
                    for dy in (0, 1):
                        weight = (at if dt else 1-at)*(ax if dx else 1-ax)*(ay if dy else 1-ay)
                        prediction += weight*values[it+dt, (ix+dx) % len(x), (iy+dy) % len(y)]
            if not np.isfinite(prediction).all():
                raise ValueError("S0 interpolation produced nonfinite macros")
            fields[name] = tuple(float(v) for v in prediction)
        return AnswerView(QueryKind.POINTS, fields=fields, points=query.points, shape=(len(points),))

    def close(self):
        self._closed = True
        self._fields.clear()


def prepare_solution(case, solve_budget=None):
    """Solve on the S0-owned grid before any scoring query is supplied."""
    began = perf_counter()
    if solve_budget is not None and (
        isinstance(solve_budget, bool) or not isinstance(solve_budget, (int, float))
        or not isfinite(solve_budget) or solve_budget <= 0
    ):
        raise ValueError("S0 solve_budget must be finite and positive")

    def check_budget():
        if solve_budget is not None and perf_counter()-began > solve_budget:
            raise TimeoutError("S0 preparation exceeded its solve budget")

    public = case.public_data
    original = public["arrays"]
    question = {}
    for name in ("collision_nu", "temperature", "background_density"):
        value = public["question"][name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"S0 {name} must be a real scalar")
        question[name] = float(value)
        if not isfinite(question[name]) or question[name] <= 0:
            raise ValueError(f"S0 {name} must be finite and positive")
    if not 10 <= question["collision_nu"] <= 30 or not .8 <= question["temperature"] <= float(np.float32(1.2)):
        raise ValueError("S0 is scoped to strong collision nu=[10,30], T=[.8,1.2]")
    arrays = {name: _numeric(original[name], (count,), name)
              for name, count in zip(("t", "x", "y", "vx", "vy"), SHAPE)}
    for name, axis in arrays.items():
        if not np.all(np.diff(axis) > 0) or not np.allclose(np.diff(axis), axis[1]-axis[0], rtol=1e-12, atol=0):
            raise ValueError(f"S0 uses a uniform {name} grid")
    if arrays["t"][0] != 0 or arrays["t"][-1] != 1:
        raise ValueError("S0 is scoped to the [0,1] time interval")
    if any(arrays[name][0] != 0 for name in ("x", "y")):
        raise ValueError("S0 periodic axes start at zero")
    if any(not np.array_equal(arrays[name], np.linspace(-5., 5., 9)) for name in ("vx", "vy")):
        raise ValueError("S0 chooses nine velocity points on [-5,5]")
    _, nx, ny, nvx, nvy = SHAPE
    for name, shape in {"momentum0": (nx, ny, 2), "kinetic_energy0": (nx, ny),
                        "velocity_weights": (nvx, nvy)}.items():
        arrays[name] = _numeric(original[name], shape, name)
    if np.any(arrays["velocity_weights"] <= 0):
        raise ValueError("S0 velocity quadrature weights must be positive")
    limit = public["question"].get("numerical_access", {}).get("max_points_per_request", CHUNK_SIZE)
    if type(limit) is not int or limit < 1:
        raise ValueError("S0 public Oracle point limit must be positive")
    chunk_size = min(CHUNK_SIZE, limit)
    check_budget()
    # This is a borrowed SDK capability, not a candidate-owned session. The
    # Evaluator closes/revokes it at scope exit, including preparation errors.
    # Only source/rho0/g0 are public methods; close is intentionally absent.
    session = open_public_case_session(case)
    for method, names in (("rho0", ("x", "y")), ("g0", ("x", "y", "vx", "vy")),
                          ("source", ("t", "x", "y", "vx", "vy"))):
        axes = tuple(arrays[name] for name in names)
        shape = tuple(len(a) for a in axes)
        values = np.empty(prod(shape), dtype=float)
        operation = getattr(session, method)
        for start, points in _chunks(axes, chunk_size):
            check_budget()
            values[start:start+len(points)] = _numeric(operation(points), (len(points),), method)
        arrays[method] = values.reshape(shape)
    if np.any(arrays["rho0"] <= 0) or np.any(arrays["g0"] < 0):
        raise ValueError("S0 initial density must be positive and g0 nonnegative")
    # Preserve the validated experiment's anonymous public input semantics.
    # In particular the legacy empirical is_perturbed branch sees no label.
    legacy_case = Case(case_id="frozen-strong-public-question", source=CaseSource.PUBLIC,
        public_data={"question": question, "arrays": arrays}, visibility=CaseVisibility.HIDDEN,
        parent_case_id=None, round_index=None, tags=())
    grid = AnswerQuery.grid(fields=FIELDS, t=SHAPE[0], x=nx, y=ny)
    check_budget()
    answer = legacy_solver.predict_macros(legacy_case, grid, budget=solve_budget)
    check_budget()
    if answer.kind is not QueryKind.GRID or tuple(answer.shape) != SHAPE[:3]:
        raise ValueError("frozen S0 returned a mismatched native macro grid")
    if any(tuple(answer.axes[name]) != tuple(arrays[name]) for name in ("t", "x", "y")):
        raise ValueError("frozen S0 returned mismatched native coordinates")
    fields = {name: np.asarray(answer.fields[name], dtype=float).reshape(SHAPE[:3]) for name in FIELDS}
    return _PreparedMacros(arrays["t"], arrays["x"], arrays["y"], fields)
