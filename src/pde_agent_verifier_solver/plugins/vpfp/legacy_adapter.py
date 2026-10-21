"""Trusted in-process adapter for the historical VPFP solver ABI.

The formal VPFP campaign still runs untrusted submissions in its established
subprocess evaluator. This adapter has a narrower purpose: expose a trusted
``solve_with_budget(public_case, query_times, budget)`` implementation as a
unified observable program without changing either ABI. The caller injects the
runner so importing the unified plugin never mutates ``sys.path`` or takes a
dependency on the legacy package.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Case,
    QueryKind,
)

from .macro_matching import VPFP_MACRO_FIELDS


LEGACY_SOLVER_OUTPUT_CONTRACT = "vpfp_continuous_macro_output_v1"
LEGACY_COMPUTE_BUDGET_CONTRACT = "vpfp_compute_budget_solver_v1"
LEGACY_COMPUTE_BUDGET_SCHEMA = "vpfp_solver_compute_budget_v1"
LEGACY_MACRO_PUBLIC_ARRAY_KEYS = frozenset(
    {
        "x",
        "y",
        "vx",
        "vy",
        "t",
        "velocity_weights",
        "rho0",
        "g0",
        "source",
        "momentum0",
        "kinetic_energy0",
    }
)


class VPFPLegacyRunner(Protocol):
    """Injected equivalent of legacy ``run_budgeted_submission``."""

    def __call__(
        self,
        submission: object,
        public_case: Mapping[str, object],
        query_times: object,
        budget: Mapping[str, object],
    ) -> tuple[Mapping[str, object], float]: ...


@dataclass(frozen=True)
class _DTypeGuardedSubmission:
    """Reject unsafe raw solver arrays before a runner can coerce them."""

    wrapped_submission: object

    @property
    def __wrapped__(self) -> object:
        return self.wrapped_submission

    def __getattr__(self, name: str) -> object:
        return getattr(self.wrapped_submission, name)

    def solve_with_budget(self, public_case, query_times, budget) -> object:
        output = self.wrapped_submission.solve_with_budget(
            public_case, query_times, budget
        )
        _validated_raw_prediction_arrays(output)
        return output


@dataclass(frozen=True)
class _LegacyGrid:
    nx: int
    ny: int
    nvx: int
    nvy: int
    x_domain: tuple[float, float]
    v_bounds: tuple[float, float, float, float]
    dtype: str = "float64"

    @property
    def dx(self) -> float:
        return float(self.x_domain[0]) / float(self.nx)

    @property
    def dy(self) -> float:
        return float(self.x_domain[1]) / float(self.ny)

    @property
    def dvx(self) -> float:
        return float(self.v_bounds[1] - self.v_bounds[0]) / float(
            max(self.nvx - 1, 1)
        )

    @property
    def dvy(self) -> float:
        return float(self.v_bounds[3] - self.v_bounds[2]) / float(
            max(self.nvy - 1, 1)
        )

    @property
    def velocity_cell_area(self) -> float:
        return self.dvx * self.dvy


@dataclass(frozen=True)
class _LegacyTime:
    t0: float
    t1: float
    nt: int

    @property
    def dt(self) -> float:
        return (float(self.t1) - float(self.t0)) / float(self.nt - 1)


@dataclass(frozen=True)
class VPFPLegacyObservableAdapter:
    """Present one trusted historical VPFP solver as an observable program.

    The configured compute budget is deliberately structured. A unified float
    budget cannot encode the legacy tier and resource binding, so per-call float
    overrides fail closed instead of silently changing the solver contract.
    """

    submission: object
    compute_budget: Mapping[str, object]
    runner: VPFPLegacyRunner | Callable[..., object]

    def __post_init__(self) -> None:
        if not callable(getattr(self.submission, "solve_with_budget", None)):
            raise TypeError(
                "legacy VPFP submission must define callable solve_with_budget"
            )
        if not callable(self.runner):
            raise TypeError("legacy VPFP runner must be callable")
        object.__setattr__(
            self,
            "compute_budget",
            MappingProxyType(_validated_budget(self.compute_budget)),
        )

    def predict_macros(
        self,
        case: Case,
        query: AnswerQuery,
        budget: float | None = None,
    ) -> AnswerView:
        """Expose the shared macro-only ABI without changing solver behavior."""

        return self.predict_observables(case, query, budget=budget)

    def predict_observables(
        self,
        case: Case,
        query: AnswerQuery,
        budget: float | None = None,
    ) -> AnswerView:
        """Run the legacy solver on its native spatial grid and adapt its output."""

        if budget is not None:
            raise ValueError(
                "legacy VPFP adapter uses its configured structured budget; "
                "a float budget override is not supported"
            )
        _validate_query(query)
        public_case, native_axes = _legacy_public_case(
            case, self.compute_budget
        )
        _validate_native_spatial_resolution(query, native_axes)
        query_times = _requested_time_axis(
            native_axes["t"],
            query.resolution.get("t"),
        )

        run_result = self.runner(
            _DTypeGuardedSubmission(self.submission),
            public_case,
            query_times.copy(),
            dict(self.compute_budget),
        )
        if not isinstance(run_result, tuple) or len(run_result) != 2:
            raise TypeError(
                "legacy VPFP runner must return (prediction, wall_seconds)"
            )
        raw_prediction, wall_seconds = run_result
        if (
            isinstance(wall_seconds, bool)
            or not isinstance(wall_seconds, (int, float))
            or not isfinite(float(wall_seconds))
            or float(wall_seconds) < 0.0
        ):
            raise ValueError("legacy VPFP runner returned invalid wall time")

        nx = len(native_axes["x"])
        ny = len(native_axes["y"])
        scalar_shape = (len(query_times), nx, ny)
        prediction = _coerce_prediction(raw_prediction, scalar_shape)
        momentum = prediction["momentum"]
        return AnswerView(
            kind=QueryKind.GRID,
            fields={
                "rho": tuple(prediction["rho"].reshape(-1, order="C").tolist()),
                "momentum_x": tuple(
                    momentum[..., 0].reshape(-1, order="C").tolist()
                ),
                "momentum_y": tuple(
                    momentum[..., 1].reshape(-1, order="C").tolist()
                ),
                "kinetic_energy": tuple(
                    prediction["kinetic_energy"]
                    .reshape(-1, order="C")
                    .tolist()
                ),
            },
            axes={
                "t": tuple(float(value) for value in query_times),
                "x": tuple(float(value) for value in native_axes["x"]),
                "y": tuple(float(value) for value in native_axes["y"]),
            },
            shape=scalar_shape,
        )


def _validated_budget(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("legacy VPFP compute budget must be a mapping")
    payload = dict(value)
    required = {
        "schema",
        "tier",
        "wall_time_seconds",
        "cpus",
        "gpus",
        "memory_bytes",
    }
    if set(payload) != required:
        raise ValueError(
            f"legacy VPFP compute budget keys must be {sorted(required)}"
        )
    if payload["schema"] != LEGACY_COMPUTE_BUDGET_SCHEMA:
        raise ValueError("unexpected legacy VPFP compute-budget schema")

    tier = payload["tier"]
    if isinstance(tier, bool) or not isinstance(tier, int) or tier < 0:
        raise ValueError("legacy VPFP budget tier must be a nonnegative integer")
    wall = payload["wall_time_seconds"]
    if (
        isinstance(wall, bool)
        or not isinstance(wall, (int, float))
        or not isfinite(float(wall))
        or float(wall) <= 0.0
    ):
        raise ValueError("legacy VPFP wall time must be finite and positive")
    for name in ("cpus", "memory_bytes"):
        item = payload[name]
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError(f"legacy VPFP {name} must be a positive integer")
    gpus = payload["gpus"]
    if isinstance(gpus, bool) or not isinstance(gpus, int) or gpus < 0:
        raise ValueError("legacy VPFP gpus must be a nonnegative integer")
    return payload


def _validate_query(query: AnswerQuery) -> None:
    if query.kind is not QueryKind.GRID:
        raise ValueError("legacy VPFP observable adapter supports GRID queries only")
    if len(query.fields) != len(VPFP_MACRO_FIELDS) or set(query.fields) != set(
        VPFP_MACRO_FIELDS
    ):
        raise ValueError(
            "legacy VPFP observable adapter requires all registered macro fields"
        )
    unexpected = set(query.resolution) - {"t", "x", "y"}
    if unexpected:
        raise ValueError(
            "legacy VPFP grid query contains unsupported axes: "
            f"{sorted(unexpected)}"
        )
    for name, count in query.resolution.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(
                f"legacy VPFP query resolution {name!r} must be a positive integer"
            )


def _normalize_question_contracts(
    question: dict[str, object],
    compute_budget: Mapping[str, object],
) -> None:
    output_contract = question.get("solver_output_contract_version")
    if (
        not isinstance(output_contract, str)
        or output_contract != LEGACY_SOLVER_OUTPUT_CONTRACT
    ):
        raise ValueError(
            "legacy VPFP public solver output contract must be exactly "
            f"{LEGACY_SOLVER_OUTPUT_CONTRACT!r}, got {output_contract!r}"
        )

    key = "solver_compute_budget_contract_version"
    if key in question:
        budget_contract = question[key]
        if (
            not isinstance(budget_contract, str)
            or budget_contract != LEGACY_COMPUTE_BUDGET_CONTRACT
        ):
            raise ValueError(
                "legacy VPFP public compute-budget contract must be exactly "
                f"{LEGACY_COMPUTE_BUDGET_CONTRACT!r}, got {budget_contract!r}"
            )
        return

    # Release-bank v2 cases created before the budgeted formal rollout omit this
    # marker. Compatibility is safe here because the adapter already validated
    # the structured budget and augments only its private question copy.
    if compute_budget.get("schema") != LEGACY_COMPUTE_BUDGET_SCHEMA:
        raise RuntimeError("legacy VPFP adapter lost its structured budget binding")
    question[key] = LEGACY_COMPUTE_BUDGET_CONTRACT


def _legacy_public_case(
    case: Case,
    compute_budget: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    import numpy as np

    question_value = case.public_data.get("question")
    arrays_value = case.public_data.get("arrays")
    if not isinstance(question_value, Mapping) or not isinstance(
        arrays_value, Mapping
    ):
        raise ValueError("legacy VPFP adapter needs public question and arrays")
    question = deepcopy(dict(question_value))
    if str(question.get("case_id", "")) != case.case_id:
        raise ValueError("legacy VPFP public question case id does not match Case")
    _normalize_question_contracts(question, compute_budget)
    if set(arrays_value) != LEGACY_MACRO_PUBLIC_ARRAY_KEYS:
        raise ValueError(
            "legacy VPFP public arrays use unexpected keys: "
            f"{sorted(arrays_value)}"
        )
    arrays = {}
    for name, raw in arrays_value.items():
        value = np.asarray(raw)
        if (
            value.dtype.hasobject
            or np.iscomplexobj(value)
            or not np.issubdtype(value.dtype, np.number)
        ):
            raise ValueError(f"legacy VPFP public array {name!r} is not numerical")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"legacy VPFP public array {name!r} is non-finite")
        arrays[str(name)] = value.copy()

    grid_value = question.get("grid")
    time_value = question.get("time")
    if not isinstance(grid_value, Mapping) or not isinstance(time_value, Mapping):
        raise ValueError("legacy VPFP public question needs grid and time objects")
    grid_payload = dict(grid_value)
    time_payload = dict(time_value)
    try:
        grid_payload["x_domain"] = tuple(
            float(item) for item in grid_payload["x_domain"]
        )
        grid_payload["v_bounds"] = tuple(
            float(item) for item in grid_payload["v_bounds"]
        )
        grid = _LegacyGrid(**grid_payload)
        time = _LegacyTime(**time_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("legacy VPFP public grid or time schema is invalid") from exc
    _validate_public_shapes(arrays, grid, time)

    collision = question.get("collision_nu")
    if (
        isinstance(collision, bool)
        or not isinstance(collision, (int, float))
        or not isfinite(float(collision))
        or float(collision) < 0.0
    ):
        raise ValueError("legacy VPFP collision frequency is invalid")
    native_axes = {
        "t": arrays["t"].copy(),
        "x": arrays["x"].copy(),
        "y": arrays["y"].copy(),
    }
    return (
        {
            "case_id": case.case_id,
            "question": question,
            "arrays": arrays,
            "grid": grid,
            "time": time,
            "collision_nu": float(collision),
        },
        native_axes,
    )


def _validate_public_shapes(
    arrays: Mapping[str, object],
    grid: _LegacyGrid,
    time: _LegacyTime,
) -> None:
    import numpy as np

    integer_values = (grid.nx, grid.ny, grid.nvx, grid.nvy, time.nt)
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in integer_values
    ):
        raise ValueError("legacy VPFP grid sizes must be integers")
    if min(integer_values) < 1 or time.nt < 2:
        raise ValueError("legacy VPFP grid and time sizes are invalid")
    if (
        len(grid.x_domain) != 2
        or len(grid.v_bounds) != 4
        or grid.x_domain[0] <= 0.0
        or grid.x_domain[1] <= 0.0
        or grid.v_bounds[1] <= grid.v_bounds[0]
        or grid.v_bounds[3] <= grid.v_bounds[2]
        or not time.t1 > time.t0
    ):
        raise ValueError("legacy VPFP grid or time bounds are invalid")

    expected = {
        "x": (grid.nx,),
        "y": (grid.ny,),
        "vx": (grid.nvx,),
        "vy": (grid.nvy,),
        "t": (time.nt,),
        "velocity_weights": (grid.nvx, grid.nvy),
        "rho0": (grid.nx, grid.ny),
        "g0": (grid.nx, grid.ny, grid.nvx, grid.nvy),
        "source": (time.nt, grid.nx, grid.ny, grid.nvx, grid.nvy),
        "momentum0": (grid.nx, grid.ny, 2),
        "kinetic_energy0": (grid.nx, grid.ny),
    }
    for name, shape in expected.items():
        if np.asarray(arrays[name]).shape != shape:
            raise ValueError(
                f"legacy VPFP public array {name!r} has shape "
                f"{np.asarray(arrays[name]).shape}, expected {shape}"
            )
    for name in ("t", "x", "y", "vx", "vy"):
        values = np.asarray(arrays[name], dtype=np.float64)
        if values.size > 1 and not np.all(np.diff(values) > 0.0):
            raise ValueError(f"legacy VPFP public axis {name!r} must be ordered")
    scale = max(1.0, abs(float(time.t0)), abs(float(time.t1)))
    if not (
        np.isclose(
            arrays["t"][0], time.t0, rtol=0.0, atol=1.0e-12 * scale
        )
        and np.isclose(
            arrays["t"][-1], time.t1, rtol=0.0, atol=1.0e-12 * scale
        )
    ):
        raise ValueError("legacy VPFP public time axis differs from its question")


def _validate_native_spatial_resolution(
    query: AnswerQuery,
    axes: Mapping[str, object],
) -> None:
    for name in ("x", "y"):
        requested = query.resolution.get(name)
        if requested is not None and requested != len(axes[name]):
            raise ValueError(
                "legacy VPFP observable adapter requires native "
                f"{name}-resolution {len(axes[name])}, got {requested}"
            )


def _requested_time_axis(native: object, count: int | None):
    import numpy as np

    values = np.asarray(native, dtype=np.float64)
    if count is None or count == len(values):
        return values.copy()
    if count == 1:
        return np.asarray([(float(values[0]) + float(values[-1])) / 2.0])
    return np.linspace(float(values[0]), float(values[-1]), int(count))


def _validated_raw_prediction_arrays(output: object) -> dict[str, Any]:
    import numpy as np

    if not isinstance(output, Mapping):
        raise TypeError("legacy VPFP solver output must be a mapping")
    aliases = {
        "rho": ("rho", "rho_pred"),
        "momentum": ("momentum", "j"),
        "kinetic_energy": ("kinetic_energy", "K"),
    }
    result = {}
    for field, names in aliases.items():
        raw = next(
            (
                output[name]
                for name in names
                if output.get(name) is not None
            ),
            None,
        )
        if raw is None:
            raise ValueError(f"legacy VPFP solver output is missing {field}")
        value = np.asarray(raw)
        if value.dtype.hasobject:
            raise ValueError(
                f"legacy VPFP {field} uses unsupported object dtype"
            )
        if np.iscomplexobj(value):
            raise ValueError(
                f"legacy VPFP {field} contains complex values"
            )
        if not np.issubdtype(value.dtype, np.number) or not np.isrealobj(value):
            raise ValueError(
                f"legacy VPFP {field} must use a real numeric dtype"
            )
        result[field] = value
    return result


def _coerce_prediction(
    output: object,
    scalar_shape: tuple[int, int, int],
) -> dict[str, Any]:
    import numpy as np

    expected = {
        "rho": scalar_shape,
        "momentum": (*scalar_shape, 2),
        "kinetic_energy": scalar_shape,
    }
    result = {}
    for field, raw_value in _validated_raw_prediction_arrays(output).items():
        value = raw_value.astype(np.float64, copy=False)
        if value.shape != expected[field]:
            raise ValueError(
                f"legacy VPFP {field} has shape {value.shape}, "
                f"expected {expected[field]}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"legacy VPFP {field} contains non-finite values")
        result[field] = value.copy()
    return result


__all__ = [
    "LEGACY_COMPUTE_BUDGET_CONTRACT",
    "LEGACY_COMPUTE_BUDGET_SCHEMA",
    "LEGACY_SOLVER_OUTPUT_CONTRACT",
    "LEGACY_MACRO_PUBLIC_ARRAY_KEYS",
    "VPFPLegacyObservableAdapter",
    "VPFPLegacyRunner",
]
