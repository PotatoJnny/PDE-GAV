from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from agent_verifier.vpfp.analytic_source import (
    ANALYTIC_SOURCE_BACKEND,
    IEEE_COORDINATE_MATMUL_PRECISION,
    ieee_float32_coordinate_matmul,
    manufacture_standard_vpfp_source_analytic,
    resolve_structured_source_backend,
)
from agent_verifier.vpfp.continuous_macros import (
    evaluate_continuous_hermite_macro_points,
)
from agent_verifier.vpfp.continuous_v2 import manufacture_standard_vpfp_source


@dataclass(frozen=True)
class VPFPOracleQueryLimits:
    max_points_per_call: int = 262_144
    max_total_points: int = 20_000_000
    evaluation_chunk_size: int = 512

    def validate(self) -> None:
        if min(
            self.max_points_per_call,
            self.max_total_points,
            self.evaluation_chunk_size,
        ) < 1:
            raise ValueError("oracle query limits must be positive")
        if self.max_points_per_call > self.max_total_points:
            raise ValueError("max_points_per_call cannot exceed max_total_points")


class VPFPContinuousPrivateCase:
    """Controller-private continuous case.

    This class is not a security sandbox. The independent agent receives only
    the public proxy or a numerical service endpoint plus explicit context
    instructions. Completed truth is released separately and intentionally.
    """

    def __init__(
        self,
        *,
        case_id: str,
        model,
        latent,
        micro_latent=None,
        equation_condition=None,
        limits: VPFPOracleQueryLimits = VPFPOracleQueryLimits(),
        source_derivative_backend: str = "auto",
    ) -> None:
        limits.validate()
        self.case_id = str(case_id)
        self._model = model
        self._latent = latent
        self._micro_latent = micro_latent
        self._equation_condition = _validated_equation_condition(equation_condition)
        self._limits = limits
        self._source_derivative_backend = resolve_structured_source_backend(
            model,
            micro_latent=micro_latent,
            requested=source_derivative_backend,
        )

    def public_oracle(self) -> "VPFPContinuousPublicOracle":
        metadata = {
            "schema": "vpfp_continuous_public_oracle_v1",
            "case_id": self.case_id,
            "equation": self._public_equation_metadata(),
            "query_contract": {
                "point_order": ["t", "x", "y", "vx", "vy"],
                "source": "source(points[N,5]) -> values[N]",
                "rho0": "rho0(points[N,2] in x,y order) -> values[N]",
                "g0": "g0(points[N,4] in x,y,vx,vy order) -> values[N]",
                "arbitrary_agent_selected_grids": True,
                "query_cost_is_metered": True,
                "source_derivative_backend": self._source_derivative_backend,
                "coordinate_matmul_precision": IEEE_COORDINATE_MATMUL_PRECISION,
            },
            "limits": asdict(self._limits),
        }
        return VPFPContinuousPublicOracle(
            metadata=metadata,
            limits=self._limits,
            source_query=self._query_source,
            rho0_query=self._query_rho0,
            g0_query=self._query_g0,
        )

    def truth(self, points) -> dict[str, np.ndarray]:
        """Controller-only arbitrary-point hidden-answer access."""

        query = _validated_points(points, width=5, name="truth points")
        rows: dict[str, list[np.ndarray]] = {
            name: [] for name in ("rho", "g", "f", "phi", "electric")
        }
        torch = _torch()
        with ieee_float32_coordinate_matmul(self._model), torch.no_grad():
            for chunk in _chunks(query, self._limits.evaluation_chunk_size):
                values = self._model.fields(
                    chunk,
                    self._latent,
                    **self._field_kwargs(),
                )
                for name in rows:
                    rows[name].append(_numpy(values[name]))
        return {name: np.concatenate(parts, axis=0) for name, parts in rows.items()}

    def released_truth(self, points) -> dict[str, np.ndarray]:
        """Explicit completed-answer release hook for solver training."""

        return self.truth(points)

    def released_macros(self, points_txy) -> dict[str, np.ndarray]:
        """Release exact full-space moments at completed-answer points."""

        with ieee_float32_coordinate_matmul(self._model):
            return evaluate_continuous_hermite_macro_points(
                self._model,
                self._latent,
                points_txy,
                micro_latent=self._micro_latent,
                equation_condition=self._equation_condition,
                chunk_size=self._limits.evaluation_chunk_size,
            )

    def _field_kwargs(self) -> dict[str, Any]:
        values = {"micro_latent": self._micro_latent}
        if self._equation_condition is not None:
            values["equation_condition"] = self._equation_condition
        return values

    def _public_equation_metadata(self) -> dict[str, Any]:
        metadata = dict(self._model.public_equation_metadata())
        if self._equation_condition is not None:
            metadata.update(
                {
                    "collision_nu": float(self._equation_condition[0]),
                    "temperature": float(self._equation_condition[1]),
                }
            )
        return metadata

    def _query_source(self, points: np.ndarray) -> np.ndarray:
        rows = []
        for chunk in _chunks(points, self._limits.evaluation_chunk_size):
            if self._source_derivative_backend == ANALYTIC_SOURCE_BACKEND:
                with ieee_float32_coordinate_matmul(self._model):
                    values = manufacture_standard_vpfp_source_analytic(
                        self._model,
                        chunk,
                        self._latent,
                        equation_condition=self._equation_condition,
                    )
                rows.append(_numpy(values["source"]))
                continue
            values = manufacture_standard_vpfp_source(
                self._model,
                chunk,
                self._latent,
                micro_latent=self._micro_latent,
                equation_condition=self._equation_condition,
                create_graph=False,
            )
            rows.append(_numpy(values["source"]))
        return np.concatenate(rows, axis=0)

    def _query_rho0(self, xy: np.ndarray) -> np.ndarray:
        t0 = float(self._model.config.time_domain[0])
        points = np.column_stack(
            (
                np.full(xy.shape[0], t0),
                xy,
                np.zeros((xy.shape[0], 2), dtype=float),
            )
        )
        return self._field_component(points, "rho")

    def _query_g0(self, xyv: np.ndarray) -> np.ndarray:
        t0 = float(self._model.config.time_domain[0])
        points = np.column_stack((np.full(xyv.shape[0], t0), xyv))
        return self._field_component(points, "g")

    def _field_component(self, points: np.ndarray, name: str) -> np.ndarray:
        torch = _torch()
        rows = []
        with ieee_float32_coordinate_matmul(self._model), torch.no_grad():
            for chunk in _chunks(points, self._limits.evaluation_chunk_size):
                values = self._model.fields(
                    chunk,
                    self._latent,
                    **self._field_kwargs(),
                )
                rows.append(_numpy(values[name]))
        return np.concatenate(rows, axis=0)


class VPFPContinuousPublicOracle:
    """Metered numerical access to public equation data only."""

    def __init__(
        self,
        *,
        metadata: dict[str, Any],
        limits: VPFPOracleQueryLimits,
        source_query: Callable[[np.ndarray], np.ndarray],
        rho0_query: Callable[[np.ndarray], np.ndarray],
        g0_query: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self.metadata = metadata
        self._limits = limits
        self._source_query = source_query
        self._rho0_query = rho0_query
        self._g0_query = g0_query
        self._total_points = 0
        self._calls = {"source": 0, "rho0": 0, "g0": 0}
        self._points = {"source": 0, "rho0": 0, "g0": 0}
        self._seconds = {"source": 0.0, "rho0": 0.0, "g0": 0.0}

    def source(self, points) -> np.ndarray:
        query = _validated_points(points, width=5, name="source points")
        time_domain = self.metadata["equation"]["domain"]["time"]
        tolerance = 1.0e-12 * max(1.0, abs(float(time_domain[1] - time_domain[0])))
        if np.any(query[:, 0] < float(time_domain[0]) - tolerance) or np.any(
            query[:, 0] > float(time_domain[1]) + tolerance
        ):
            raise ValueError("source query time lies outside the declared time domain")
        return self._metered("source", query, self._source_query)

    def rho0(self, points_xy) -> np.ndarray:
        query = _validated_points(points_xy, width=2, name="rho0 points")
        return self._metered("rho0", query, self._rho0_query)

    def g0(self, points_xyv) -> np.ndarray:
        query = _validated_points(points_xyv, width=4, name="g0 points")
        return self._metered("g0", query, self._g0_query)

    def query_report(self) -> dict[str, Any]:
        return {
            "schema": "vpfp_continuous_public_oracle_query_report_v1",
            "case_id": self.metadata["case_id"],
            "total_points": int(self._total_points),
            "calls": dict(self._calls),
            "points": dict(self._points),
            "seconds": dict(self._seconds),
        }

    def solver_instruction_text(self) -> str:
        return """# Continuous VPFP Public Context

The primary object is a hidden continuous function on T^2_x x R^2_v. Grid
arrays are numerical oracle samples, not the definition of that function. You
may query source, rho0, and g0 on arbitrary finite grids that you select.
Runtime, memory, and oracle query counts are included in solver evaluation.

Use only the documented public methods and metadata. Do not inspect Python
private attributes, callback closures, controller files, model weights, latent
variables, or unreleased hidden answers. Historical completed answers are
available only when the controller explicitly places them in a released-truth
training bank. A final frozen solver is evaluated on fresh public-only cases.

The equation, domain, coefficients, normalization, and point ordering are in
oracle.metadata. Velocity is mathematically unbounded; any finite velocity
window is your numerical approximation and should be checked by tail/window
convergence.
"""

    def _metered(self, name: str, points: np.ndarray, callback) -> np.ndarray:
        count = int(points.shape[0])
        if count > self._limits.max_points_per_call:
            raise ValueError(
                f"{name} query has {count} points; limit is {self._limits.max_points_per_call}"
            )
        if self._total_points + count > self._limits.max_total_points:
            raise RuntimeError("continuous VPFP oracle total query budget exceeded")
        started = time.perf_counter()
        values = np.asarray(callback(points), dtype=float)
        elapsed = time.perf_counter() - started
        if values.shape != (count,):
            raise RuntimeError(f"{name} oracle returned shape {values.shape}, expected {(count,)}")
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(f"{name} oracle returned non-finite values")
        self._total_points += count
        self._calls[name] += 1
        self._points[name] += count
        self._seconds[name] += float(elapsed)
        return values


def _validated_points(values, *, width: int, name: str) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if points.ndim == 1:
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != width:
        raise ValueError(f"{name} must have shape [N,{width}], got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be finite")
    return points


def _validated_equation_condition(value):
    if value is None:
        return None
    condition = np.asarray(value, dtype=float)
    if condition.shape != (2,) or not np.all(np.isfinite(condition)):
        raise ValueError(
            "equation_condition must be a finite (collision_nu, temperature) pair"
        )
    if np.any(condition <= 0.0):
        raise ValueError("collision_nu and temperature must be positive")
    return condition


def _chunks(values: np.ndarray, size: int):
    for start in range(0, int(values.shape[0]), int(size)):
        yield values[start : start + int(size)]


def _numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy().astype(float, copy=False)


def _torch():
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("torch is required for the continuous VPFP oracle") from exc
    return torch
