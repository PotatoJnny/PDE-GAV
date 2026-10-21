from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from agent_verifier.vpfp.high_fidelity import refinement_grid, refinement_time
from agent_verifier.vpfp.operators import (
    EPS,
    VPFPGrid,
    VPFPTime,
    relative_l2,
)
from agent_verifier.vpfp.resource_limits import wall_time_limit


SCALABLE_RESOLUTION_CONTRACT = "vpfp_scalable_resolution_solver_v1"
CONVERGENCE_REPORT_SCHEMA = "vpfp_answer_first_refinement_report_v1"
MACRO_FIELDS = ("rho", "momentum", "kinetic_energy")


class ResolutionContractError(ValueError):
    pass


@dataclass(frozen=True)
class VPFPResolution:
    nx: int
    ny: int
    nvx: int
    nvy: int
    nt: int

    @classmethod
    def from_grid_time(
        cls, grid: VPFPGrid, time_grid: VPFPTime
    ) -> "VPFPResolution":
        return cls(
            nx=int(grid.nx),
            ny=int(grid.ny),
            nvx=int(grid.nvx),
            nvy=int(grid.nvy),
            nt=int(time_grid.nt),
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "VPFPResolution":
        required = {"nx", "ny", "nvx", "nvy", "nt"}
        if set(value) != required:
            raise ResolutionContractError(
                f"resolution keys must be {sorted(required)}, got {sorted(value)}"
            )
        if any(
            isinstance(value[name], bool) or not isinstance(value[name], int)
            for name in required
        ):
            raise ResolutionContractError("resolution values must be integers")
        result = cls(**{name: int(value[name]) for name in required})
        result.validate()
        return result

    def validate(self) -> None:
        if min(self.nx, self.ny) < 4:
            raise ResolutionContractError("nx and ny must be at least 4")
        if min(self.nvx, self.nvy, self.nt) < 3:
            raise ResolutionContractError("nvx, nvy, and nt must be at least 3")

    @property
    def source_element_count(self) -> int:
        return int(self.nt * self.nx * self.ny * self.nvx * self.nvy)

    @property
    def macro_element_count(self) -> int:
        return int(4 * self.nt * self.nx * self.ny)

    def to_dict(self) -> dict[str, int]:
        return {name: int(value) for name, value in asdict(self).items()}


def make_refinement_ladder(
    base_grid: VPFPGrid,
    base_time: VPFPTime,
    *,
    level_count: int,
    dimension: str,
    max_source_elements: int,
) -> list[dict[str, Any]]:
    if int(level_count) < 3:
        raise ResolutionContractError(
            "a convergence study needs at least three levels"
        )
    if int(max_source_elements) < 1:
        raise ResolutionContractError("max_source_elements must be positive")
    rows: list[dict[str, Any]] = []
    for level in range(int(level_count)):
        grid = refinement_grid(base_grid, level=level, dimension=dimension)
        time_grid = refinement_time(
            base_time, level=level, dimension=dimension
        )
        resolution = VPFPResolution.from_grid_time(grid, time_grid)
        if resolution.source_element_count > int(max_source_elements):
            raise ResolutionContractError(
                "refinement level exceeds the public source-element envelope: "
                f"{resolution.source_element_count} > {int(max_source_elements)}"
            )
        rows.append(
            {
                "level": int(level),
                "dimension": str(dimension),
                "grid": grid,
                "time": time_grid,
                "resolution": resolution,
                "characteristic_spacing": float(2.0 ** (-level)),
            }
        )
    validate_refinement_ladder(rows, dimension=dimension)
    return rows


def validate_refinement_ladder(
    rows: list[dict[str, Any]], *, dimension: str
) -> None:
    if dimension not in {"spatial", "velocity", "temporal", "joint"}:
        raise ResolutionContractError(
            f"unsupported refinement dimension: {dimension}"
        )
    if len(rows) < 3:
        raise ResolutionContractError(
            "a convergence study needs at least three levels"
        )
    resolutions = [
        (
            row["resolution"]
            if isinstance(row.get("resolution"), VPFPResolution)
            else VPFPResolution.from_dict(dict(row["resolution"]))
        )
        for row in rows
    ]
    for coarse, fine in zip(resolutions, resolutions[1:], strict=False):
        expected = {
            "nx": 2 * coarse.nx if dimension in {"spatial", "joint"} else coarse.nx,
            "ny": 2 * coarse.ny if dimension in {"spatial", "joint"} else coarse.ny,
            "nvx": (
                2 * (coarse.nvx - 1) + 1
                if dimension in {"velocity", "joint"}
                else coarse.nvx
            ),
            "nvy": (
                2 * (coarse.nvy - 1) + 1
                if dimension in {"velocity", "joint"}
                else coarse.nvy
            ),
            "nt": (
                2 * (coarse.nt - 1) + 1
                if dimension in {"temporal", "joint"}
                else coarse.nt
            ),
        }
        if fine.to_dict() != expected:
            raise ResolutionContractError(
                "refinement ladder is not endpoint-nested dyadic refinement: "
                f"expected {expected}, got {fine.to_dict()}"
            )


def resolution_request(
    row: dict[str, Any],
    *,
    max_source_elements: int,
    study_id: str,
) -> dict[str, Any]:
    resolution = row["resolution"]
    if not isinstance(resolution, VPFPResolution):
        resolution = VPFPResolution.from_dict(dict(resolution))
    return {
        "schema": SCALABLE_RESOLUTION_CONTRACT,
        "study_id": str(study_id),
        "dimension": str(row["dimension"]),
        "level": int(row["level"]),
        "characteristic_spacing": float(row["characteristic_spacing"]),
        "resolution": resolution.to_dict(),
        "resource_envelope": {
            "max_source_elements": int(max_source_elements),
        },
        "semantics": {
            "public_arrays_are_samples_not_primary_truth": True,
            "same_hidden_continuous_case_across_levels": True,
            "solver_must_use_actual_case_grid": True,
        },
    }


def run_scalable_submission(
    module,
    public_case: dict[str, Any],
    request: dict[str, Any],
    *,
    max_wall_time_seconds: float | None = None,
) -> tuple[Any, float]:
    entrypoint = getattr(module, "solve_at_resolution", None)
    if entrypoint is None or not callable(entrypoint):
        raise ResolutionContractError(
            "submission must define solve_at_resolution(public_case, resolution)"
        )
    started = time.perf_counter()
    with wall_time_limit(max_wall_time_seconds):
        output = entrypoint(public_case, dict(request["resolution"]))
    return output, float(time.perf_counter() - started)


def coerce_macro_output(
    output: Any, resolution: VPFPResolution
) -> dict[str, np.ndarray]:
    if not isinstance(output, dict):
        raise ResolutionContractError("solver output must be a mapping")
    aliases = {
        "rho": ("rho", "rho_pred"),
        "momentum": ("momentum", "j"),
        "kinetic_energy": ("kinetic_energy", "K"),
    }
    values: dict[str, np.ndarray] = {}
    for field, names in aliases.items():
        raw = next((output[name] for name in names if output.get(name) is not None), None)
        if raw is None:
            raise ResolutionContractError(f"solver output is missing {field}")
        values[field] = np.asarray(raw, dtype=float)
    scalar_shape = (resolution.nt, resolution.nx, resolution.ny)
    expected = {
        "rho": scalar_shape,
        "momentum": (*scalar_shape, 2),
        "kinetic_energy": scalar_shape,
    }
    for field, value in values.items():
        if value.shape != expected[field]:
            raise ResolutionContractError(
                f"{field} has shape {value.shape}, expected {expected[field]}"
            )
        if not np.all(np.isfinite(value)):
            raise ResolutionContractError(f"{field} contains non-finite values")
    return values


def restrict_macros(
    fields: dict[str, np.ndarray],
    *,
    fine: VPFPResolution,
    target: VPFPResolution,
) -> dict[str, np.ndarray]:
    time_index = _endpoint_nested_indices(fine.nt, target.nt, "time")
    x_index = _periodic_nested_indices(fine.nx, target.nx, "x")
    y_index = _periodic_nested_indices(fine.ny, target.ny, "y")
    restricted: dict[str, np.ndarray] = {}
    for field in MACRO_FIELDS:
        value = np.asarray(fields[field], dtype=float)
        value = np.take(value, time_index, axis=0)
        value = np.take(value, x_index, axis=1)
        value = np.take(value, y_index, axis=2)
        restricted[field] = value
    return restricted


def macro_error_metrics(
    prediction: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
) -> dict[str, float]:
    metrics = {
        field: relative_l2(
            np.asarray(prediction[field]) - np.asarray(truth[field]),
            np.asarray(truth[field]),
        )
        for field in MACRO_FIELDS
    }
    metrics["aggregate"] = float(
        np.mean([metrics[field] for field in MACRO_FIELDS])
    )
    return metrics


def continuity_balance_metrics(
    prediction: dict[str, np.ndarray],
    *,
    source: np.ndarray,
    grid: VPFPGrid,
    time_grid: VPFPTime,
) -> dict[str, float | str]:
    rho = np.asarray(prediction["rho"], dtype=float)
    momentum = np.asarray(prediction["momentum"], dtype=float)
    source_array = np.asarray(source, dtype=float)
    expected_source = (
        time_grid.nt,
        grid.nx,
        grid.ny,
        grid.nvx,
        grid.nvy,
    )
    if source_array.shape != expected_source:
        raise ResolutionContractError(
            f"source has shape {source_array.shape}, expected {expected_source}"
        )
    weights = trapezoidal_velocity_weights(grid)
    source_mass = np.sum(source_array * weights, axis=(-2, -1))
    d_rho_dt = np.gradient(
        rho,
        time_grid.dt,
        axis=0,
        edge_order=2 if time_grid.nt >= 3 else 1,
    )
    divergence = _periodic_spectral_divergence(momentum, grid)
    residual = d_rho_dt + divergence - source_mass
    scale = (
        np.linalg.norm(d_rho_dt)
        + np.linalg.norm(divergence)
        + np.linalg.norm(source_mass)
        + EPS
    )
    local_relative = float(np.linalg.norm(residual) / scale)

    spatial_area = float(grid.dx * grid.dy)
    mass = np.sum(rho, axis=(-2, -1)) * spatial_area
    source_total = np.sum(source_mass, axis=(-2, -1)) * spatial_area
    d_mass_dt = np.gradient(
        mass,
        time_grid.dt,
        edge_order=2 if time_grid.nt >= 3 else 1,
    )
    global_residual = d_mass_dt - source_total
    global_scale = (
        np.linalg.norm(d_mass_dt) + np.linalg.norm(source_total) + EPS
    )
    return {
        "continuity_window_relative_l2": local_relative,
        "global_mass_balance_window_relative_l2": float(
            np.linalg.norm(global_residual) / global_scale
        ),
        "source_velocity_integration": "trapezoidal_on_materialized_window",
    }


def build_convergence_report(
    records: list[dict[str, Any]],
    *,
    dimension: str,
    min_observed_order: float = 0.0,
    monotonic_rtol: float = 0.02,
    exact_floor: float = 1.0e-12,
) -> dict[str, Any]:
    if len(records) < 3:
        raise ResolutionContractError(
            "a convergence report needs at least three records"
        )
    resolutions = [
        (
            row["resolution"]
            if isinstance(row["resolution"], VPFPResolution)
            else VPFPResolution.from_dict(dict(row["resolution"]))
        )
        for row in records
    ]
    spacings = [float(row["characteristic_spacing"]) for row in records]
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in spacings
    ) or any(
        fine >= coarse for coarse, fine in zip(spacings, spacings[1:], strict=False)
    ):
        raise ResolutionContractError(
            f"characteristic spacings must strictly decrease: {spacings}"
        )

    level_metrics: list[dict[str, Any]] = []
    errors_by_field = {field: [] for field in (*MACRO_FIELDS, "aggregate")}
    for row, resolution in zip(records, resolutions, strict=True):
        prediction = coerce_macro_output(row["prediction"], resolution)
        truth = coerce_macro_output(row["truth"], resolution)
        errors = macro_error_metrics(prediction, truth)
        for field, value in errors.items():
            errors_by_field[field].append(float(value))
        physics = continuity_balance_metrics(
            prediction,
            source=np.asarray(row["source"], dtype=float),
            grid=row["grid"],
            time_grid=row["time"],
        )
        level_metrics.append(
            {
                "level": int(row["level"]),
                "resolution": resolution.to_dict(),
                "characteristic_spacing": float(row["characteristic_spacing"]),
                "source_element_count": int(resolution.source_element_count),
                "macro_element_count": int(resolution.macro_element_count),
                "wall_time_seconds": float(row["wall_time_seconds"]),
                "errors": errors,
                "physics": physics,
            }
        )

    error_convergence = {
        field: _exact_error_convergence(
            values,
            spacings,
            min_observed_order=float(min_observed_order),
            monotonic_rtol=float(monotonic_rtol),
            exact_floor=float(exact_floor),
        )
        for field, values in errors_by_field.items()
    }
    base_resolution = resolutions[0]
    common_predictions = [
        restrict_macros(
            coerce_macro_output(row["prediction"], resolution),
            fine=resolution,
            target=base_resolution,
        )
        for row, resolution in zip(records, resolutions, strict=True)
    ]
    self_differences = []
    for coarse, fine in zip(
        common_predictions, common_predictions[1:], strict=False
    ):
        self_differences.append(
            macro_error_metrics(coarse, fine)["aggregate"]
        )
    self_convergence = _successive_difference_convergence(
        self_differences,
        spacings,
        exact_floor=float(exact_floor),
    )
    runtime = [float(row["wall_time_seconds"]) for row in records]
    source_counts = [
        float(resolution.source_element_count) for resolution in resolutions
    ]
    cost = {
        "wall_time_seconds": runtime,
        "source_element_counts": [int(value) for value in source_counts],
        "runtime_scaling_exponent_vs_source_elements": _log_slope(
            source_counts, runtime
        ),
        "aggregate_error_scaling_exponent_vs_wall_time": _log_slope(
            runtime, errors_by_field["aggregate"]
        ),
    }
    aggregate = error_convergence["aggregate"]
    field_not_worse = all(
        block["finest_not_worse_than_coarsest"]
        for name, block in error_convergence.items()
        if name in MACRO_FIELDS
    )
    gates = {
        "all_outputs_finite_and_shape_valid": True,
        "aggregate_error_monotone_nonincreasing": bool(
            aggregate["monotone_nonincreasing"]
        ),
        "aggregate_finest_strictly_better_or_exact": bool(
            aggregate["finest_strictly_better_or_exact"]
        ),
        "aggregate_positive_asymptotic_order_or_exact": bool(
            aggregate["positive_asymptotic_order_or_exact"]
        ),
        "no_macro_field_finest_worse_than_coarsest": bool(field_not_worse),
    }
    return {
        "schema": CONVERGENCE_REPORT_SCHEMA,
        "solver_resolution_contract_version": SCALABLE_RESOLUTION_CONTRACT,
        "dimension": str(dimension),
        "level_count": len(records),
        "truth_semantics": (
            "hidden continuous answer sampled independently on every level"
        ),
        "level_metrics": level_metrics,
        "exact_error_convergence": error_convergence,
        "solver_self_convergence_diagnostic_only": self_convergence,
        "cost_scaling": cost,
        "gates": gates,
        "accepted": bool(all(gates.values())),
        "notes": [
            "Self-convergence is diagnostic only; a resolution-independent wrong answer can self-converge.",
            "Continuity balance integrates the source only over the materialized velocity window and is not a full-space identity gate.",
        ],
    }


def trapezoidal_velocity_weights(grid: VPFPGrid) -> np.ndarray:
    weights_x = np.ones(int(grid.nvx), dtype=float)
    weights_y = np.ones(int(grid.nvy), dtype=float)
    weights_x[[0, -1]] = 0.5
    weights_y[[0, -1]] = 0.5
    return (
        weights_x[:, None]
        * weights_y[None, :]
        * float(grid.dvx)
        * float(grid.dvy)
    )


def _exact_error_convergence(
    errors: list[float],
    spacings: list[float],
    *,
    min_observed_order: float,
    monotonic_rtol: float,
    exact_floor: float,
) -> dict[str, Any]:
    values = np.asarray(errors, dtype=float)
    orders: list[float | None] = []
    for index in range(len(values) - 1):
        coarse = float(values[index])
        fine = float(values[index + 1])
        ratio = float(spacings[index] / spacings[index + 1])
        if max(coarse, fine) <= exact_floor:
            orders.append(None)
        elif min(coarse, fine) <= 0.0:
            orders.append(None)
        else:
            orders.append(float(np.log(coarse / fine) / np.log(ratio)))
    finite_orders = [
        float(value)
        for value in orders
        if value is not None and np.isfinite(float(value))
    ]
    monotone = all(
        float(fine)
        <= float(coarse) * (1.0 + float(monotonic_rtol)) + float(exact_floor)
        for coarse, fine in zip(values, values[1:], strict=False)
    )
    exact = bool(float(values[-1]) <= float(exact_floor))
    strict_improvement = bool(
        exact
        or float(values[-1])
        < float(values[0]) * (1.0 - max(1.0e-6, monotonic_rtol))
    )
    asymptotic_order = finite_orders[-1] if finite_orders else None
    return {
        "errors": [float(value) for value in values],
        "pairwise_observed_orders": orders,
        "median_observed_order": (
            None if not finite_orders else float(np.median(finite_orders))
        ),
        "asymptotic_observed_order": asymptotic_order,
        "monotone_nonincreasing": bool(monotone),
        "finest_strictly_better_or_exact": strict_improvement,
        "positive_asymptotic_order_or_exact": bool(
            exact
            or (
                asymptotic_order is not None
                and float(asymptotic_order) > float(min_observed_order)
            )
        ),
        "finest_not_worse_than_coarsest": bool(
            exact
            or float(values[-1])
            <= float(values[0]) * (1.0 + float(monotonic_rtol))
            + float(exact_floor)
        ),
        "roundoff_or_exact_floor_reached": exact,
    }


def _successive_difference_convergence(
    differences: list[float],
    spacings: list[float],
    *,
    exact_floor: float,
) -> dict[str, Any]:
    order = None
    if len(differences) >= 2:
        coarse = float(differences[-2])
        fine = float(differences[-1])
        ratio = float(spacings[-2] / spacings[-1])
        if min(coarse, fine) > exact_floor:
            order = float(np.log(coarse / fine) / np.log(ratio))
    return {
        "successive_relative_differences_on_coarsest_grid": [
            float(value) for value in differences
        ],
        "observed_order": order,
        "is_acceptance_gate": False,
    }


def _periodic_spectral_divergence(
    momentum: np.ndarray, grid: VPFPGrid
) -> np.ndarray:
    values = np.asarray(momentum, dtype=float)
    if values.shape[-1] != 2:
        raise ResolutionContractError("momentum final axis must have length two")
    kx = 2.0 * np.pi * np.fft.fftfreq(grid.nx, d=grid.dx)[:, None]
    ky = 2.0 * np.pi * np.fft.fftfreq(grid.ny, d=grid.dy)[None, :]
    jx_hat = np.fft.fftn(values[..., 0], axes=(-2, -1))
    jy_hat = np.fft.fftn(values[..., 1], axes=(-2, -1))
    return np.fft.ifftn(
        1j * kx * jx_hat + 1j * ky * jy_hat,
        axes=(-2, -1),
    ).real


def _endpoint_nested_indices(
    fine_count: int, target_count: int, axis: str
) -> np.ndarray:
    if target_count < 2 or (fine_count - 1) % (target_count - 1) != 0:
        raise ResolutionContractError(
            f"{axis} endpoint grids are not nested: {fine_count}, {target_count}"
        )
    stride = (fine_count - 1) // (target_count - 1)
    return np.arange(target_count, dtype=int) * stride


def _periodic_nested_indices(
    fine_count: int, target_count: int, axis: str
) -> np.ndarray:
    if target_count < 1 or fine_count % target_count != 0:
        raise ResolutionContractError(
            f"{axis} periodic grids are not nested: {fine_count}, {target_count}"
        )
    stride = fine_count // target_count
    return np.arange(target_count, dtype=int) * stride


def _log_slope(x_values: list[float], y_values: list[float]) -> float | None:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if int(np.sum(mask)) < 2 or np.allclose(x[mask], x[mask][0]):
        return None
    return float(np.polyfit(np.log(x[mask]), np.log(y[mask]), deg=1)[0])
