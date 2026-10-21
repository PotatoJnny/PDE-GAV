from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import CubicSpline

try:
    from resource_limits import wall_time_limit
except ModuleNotFoundError:  # Package import used by repository tests.
    from agent_verifier.vpfp.resource_limits import wall_time_limit


COMPUTE_BUDGET_CONTRACT = "vpfp_compute_budget_solver_v1"
COMPUTE_BUDGET_SCHEMA = "vpfp_solver_compute_budget_v1"
BUDGET_RESPONSE_REPORT_SCHEMA = "vpfp_compute_budget_response_v1"
MACRO_FIELDS = ("rho", "momentum", "kinetic_energy")
ALIASES = {
    "rho": ("rho", "rho_pred"),
    "momentum": ("momentum", "j"),
    "kinetic_energy": ("kinetic_energy", "K"),
}


class ComputeBudgetContractError(ValueError):
    pass


@dataclass(frozen=True)
class SolverComputeBudget:
    tier: int
    wall_time_seconds: float
    cpus: int
    gpus: int
    memory_bytes: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SolverComputeBudget":
        payload = dict(value)
        if payload.pop("schema", None) != COMPUTE_BUDGET_SCHEMA:
            raise ComputeBudgetContractError("unexpected compute-budget schema")
        required = {"tier", "wall_time_seconds", "cpus", "gpus", "memory_bytes"}
        if set(payload) != required:
            raise ComputeBudgetContractError(
                f"compute-budget keys must be {sorted(required)}"
            )
        tier = payload["tier"]
        if isinstance(tier, bool) or not isinstance(tier, int) or tier < 0:
            raise ComputeBudgetContractError("compute-budget tier must be nonnegative")
        wall_time = payload["wall_time_seconds"]
        if isinstance(wall_time, bool) or not isinstance(wall_time, (int, float)):
            raise ComputeBudgetContractError("wall_time_seconds must be numeric")
        resources = {}
        for name in ("cpus", "memory_bytes"):
            item = payload[name]
            if isinstance(item, bool) or not isinstance(item, int) or item < 1:
                raise ComputeBudgetContractError(f"{name} must be a positive integer")
            resources[name] = int(item)
        gpus = payload["gpus"]
        if isinstance(gpus, bool) or not isinstance(gpus, int) or gpus < 0:
            raise ComputeBudgetContractError("gpus must be a nonnegative integer")
        if float(wall_time) <= 0.0 or not np.isfinite(float(wall_time)):
            raise ComputeBudgetContractError("wall_time_seconds must be finite and positive")
        return cls(
            tier=int(tier),
            wall_time_seconds=float(wall_time),
            cpus=resources["cpus"],
            gpus=int(gpus),
            memory_bytes=resources["memory_bytes"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {"schema": COMPUTE_BUDGET_SCHEMA, **asdict(self)}


def make_budget_schedule(
    wall_time_seconds: list[float],
    *,
    cpus: int,
    gpus: int,
    memory_bytes: int,
) -> list[SolverComputeBudget]:
    if len(wall_time_seconds) < 2:
        raise ComputeBudgetContractError("at least two compute budgets are required")
    rows = [
        SolverComputeBudget.from_dict(
            {
                "schema": COMPUTE_BUDGET_SCHEMA,
                "tier": tier,
                "wall_time_seconds": float(seconds),
                "cpus": int(cpus),
                "gpus": int(gpus),
                "memory_bytes": int(memory_bytes),
            }
        )
        for tier, seconds in enumerate(wall_time_seconds)
    ]
    if any(
        fine.wall_time_seconds <= coarse.wall_time_seconds
        for coarse, fine in zip(rows, rows[1:], strict=False)
    ):
        raise ComputeBudgetContractError(
            "compute-budget wall times must be strictly increasing"
        )
    return rows


def validate_query_times(
    public_case: Mapping[str, Any], query_times: Any
) -> np.ndarray:
    values = np.asarray(query_times, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise ComputeBudgetContractError("query_times must be a nonempty 1D array")
    if not np.all(np.isfinite(values)):
        raise ComputeBudgetContractError("query_times contain non-finite values")
    time_grid = public_case["time"]
    t0 = float(time_grid.t0)
    t1 = float(time_grid.t1)
    tolerance = 1.0e-12 * max(1.0, abs(t0), abs(t1))
    if float(np.min(values)) < t0 - tolerance or float(np.max(values)) > t1 + tolerance:
        raise ComputeBudgetContractError(
            f"query_times must lie in the public interval [{t0}, {t1}]"
        )
    return values


def coerce_budget_output(
    output: Any,
    public_case: Mapping[str, Any],
    query_times: Any,
) -> dict[str, np.ndarray]:
    if not isinstance(output, Mapping):
        raise ComputeBudgetContractError("solver output must be a mapping")
    times = validate_query_times(public_case, query_times)
    grid = public_case["grid"]
    scalar_shape = (int(times.size), int(grid.nx), int(grid.ny))
    expected = {
        "rho": scalar_shape,
        "momentum": (*scalar_shape, 2),
        "kinetic_energy": scalar_shape,
    }
    result = {}
    for field in MACRO_FIELDS:
        raw = next(
            (
                output[name]
                for name in ALIASES[field]
                if output.get(name) is not None
            ),
            None,
        )
        if raw is None:
            raise ComputeBudgetContractError(f"solver output is missing {field}")
        value = np.asarray(raw, dtype=np.float64)
        if value.shape != expected[field]:
            raise ComputeBudgetContractError(
                f"{field} has shape {value.shape}, expected {expected[field]}"
            )
        if not np.all(np.isfinite(value)):
            raise ComputeBudgetContractError(f"{field} contains non-finite values")
        result[field] = value
    return result


def run_budgeted_submission(
    module: Any,
    public_case: Mapping[str, Any],
    query_times: Any,
    budget: SolverComputeBudget | Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], float]:
    normalized = (
        budget
        if isinstance(budget, SolverComputeBudget)
        else SolverComputeBudget.from_dict(budget)
    )
    times = validate_query_times(public_case, query_times)
    entrypoint = getattr(module, "solve_with_budget", None)
    if entrypoint is None or not callable(entrypoint):
        raise ComputeBudgetContractError(
            "submission must define solve_with_budget(public_case, query_times, budget)"
        )
    started = time.perf_counter()
    with wall_time_limit(normalized.wall_time_seconds):
        output = entrypoint(
            public_case,
            np.array(times, copy=True),
            normalized.to_dict(),
        )
    elapsed = float(time.perf_counter() - started)
    return coerce_budget_output(output, public_case, times), elapsed


def interpolate_macro_times(
    source_times: Any,
    fields: Mapping[str, Any],
    query_times: Any,
) -> dict[str, np.ndarray]:
    source = np.asarray(source_times, dtype=np.float64)
    query = np.asarray(query_times, dtype=np.float64)
    if source.ndim != 1 or source.size < 4:
        raise ComputeBudgetContractError(
            "truth interpolation requires at least four source time samples"
        )
    if not np.all(np.diff(source) > 0.0):
        raise ComputeBudgetContractError("source truth times must be strictly increasing")
    if query.ndim != 1 or not np.all(np.isfinite(query)):
        raise ComputeBudgetContractError("query_times must be a finite 1D array")
    result = {}
    for field in MACRO_FIELDS:
        value = np.asarray(fields[field], dtype=np.float64)
        if value.shape[0] != source.size:
            raise ComputeBudgetContractError(
                f"{field} truth time axis does not match source_times"
            )
        interpolated = np.asarray(
            CubicSpline(source, value, axis=0, extrapolate=False)(query),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(interpolated)):
            raise ComputeBudgetContractError(
                f"interpolated {field} truth contains non-finite values"
            )
        result[field] = interpolated
    return result


def macro_error_metrics(
    prediction: Mapping[str, Any], truth: Mapping[str, Any]
) -> dict[str, float]:
    metrics = {}
    for field in MACRO_FIELDS:
        predicted = np.asarray(prediction[field], dtype=np.float64)
        reference = np.asarray(truth[field], dtype=np.float64)
        metrics[field] = float(
            np.linalg.norm(predicted - reference)
            / (np.linalg.norm(reference) + 1.0e-12)
        )
    metrics["aggregate"] = float(np.mean(list(metrics.values())))
    return metrics
