"""Reference-free equation-first self-convergence evidence for VPFP solvers.

Answer-first hidden functions remain the accuracy truth.  This module only
measures whether one candidate changes coherently when it is given increasing
internal resolution on the same equation and the same physical query points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


VPFP_SELF_CONVERGENCE_SCHEMA = "vpfp_equation_first_self_convergence_v2"
VPFP_SELF_CONVERGENCE_SUMMARY_SCHEMA = (
    "vpfp_equation_first_self_convergence_summary_v2"
)
VPFP_MACRO_FIELDS = ("rho", "momentum", "kinetic_energy")


@dataclass(frozen=True)
class VPFPEquationFirstSelfConvergenceConfig:
    minimum_levels: int = 3
    geometric_ratio_tolerance: float = 1.0e-6
    roundoff_relative_floor: float = 1.0e-10
    minimum_informative_field_fraction: float = 1.0 / 3.0
    minimum_reported_order: float = 0.0

    def __post_init__(self) -> None:
        if self.minimum_levels < 3:
            raise ValueError("VPFP self-convergence needs at least three levels")
        if not (
            math.isfinite(self.geometric_ratio_tolerance)
            and self.geometric_ratio_tolerance > 0.0
            and math.isfinite(self.roundoff_relative_floor)
            and self.roundoff_relative_floor > 0.0
            and math.isfinite(self.minimum_informative_field_fraction)
            and 0.0 < self.minimum_informative_field_fraction <= 1.0
            and math.isfinite(self.minimum_reported_order)
        ):
            raise ValueError("invalid VPFP self-convergence thresholds")


def _finite_field(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite and nonempty")
    return array


def _broadcast_weights(weights: Any | None, shape: tuple[int, ...]) -> np.ndarray:
    if weights is None:
        return np.ones(shape, dtype=np.float64)
    array = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError("VPFP self-convergence weights must be nonnegative")
    try:
        return np.broadcast_to(array, shape)
    except ValueError as exc:
        raise ValueError("VPFP self-convergence weights do not broadcast") from exc


def _weighted_norm(value: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weights * np.square(value))))


def _median(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return None if not finite else float(np.median(np.asarray(finite)))


def analyze_vpfp_equation_first_self_convergence(
    level_rows: Sequence[Mapping[str, Any]],
    *,
    field_weights: Mapping[str, Any] | None = None,
    config: VPFPEquationFirstSelfConvergenceConfig | None = None,
) -> dict[str, Any]:
    """Analyze nested candidate outputs without any numerical reference.

    Every level must provide ``name``, ``resolution_scale`` and ``predictions``.
    Predictions are already restricted to identical physical ``(t,x,y)``
    queries.  A larger resolution scale means a smaller effective mesh width.
    """

    cfg = config or VPFPEquationFirstSelfConvergenceConfig()
    rows = [dict(row) for row in level_rows]
    if len(rows) < cfg.minimum_levels:
        raise ValueError("insufficient VPFP equation-first refinement levels")
    names = [str(row.get("name") or "") for row in rows]
    scales = np.asarray(
        [float(row.get("resolution_scale", math.nan)) for row in rows],
        dtype=np.float64,
    )
    if (
        any(not name for name in names)
        or len(set(names)) != len(names)
        or not np.isfinite(scales).all()
        or np.any(scales <= 0.0)
        or np.any(np.diff(scales) <= 0.0)
    ):
        raise ValueError("invalid VPFP equation-first refinement sequence")
    ratios = scales[1:] / scales[:-1]
    ratio = float(np.exp(np.mean(np.log(ratios))))
    if np.max(np.abs(ratios / ratio - 1.0)) > cfg.geometric_ratio_tolerance:
        raise ValueError("VPFP self-convergence requires geometric refinement")

    predictions = [row.get("predictions") for row in rows]
    if any(not isinstance(value, Mapping) for value in predictions):
        raise ValueError("VPFP refinement predictions are absent")
    fields = tuple(sorted(str(name) for name in predictions[0]))
    if fields != tuple(sorted(VPFP_MACRO_FIELDS)) or any(
        tuple(sorted(str(name) for name in value)) != fields
        for value in predictions
    ):
        raise ValueError("VPFP refinement fields do not match the macro contract")
    weights_by_field = dict(field_weights or {})
    if set(weights_by_field) - set(fields):
        raise ValueError("VPFP self-convergence weights contain unknown fields")

    reports: dict[str, dict[str, Any]] = {}
    informative_orders: list[float] = []
    informative_count = 0
    monotone_count = 0
    for field in fields:
        values = [
            _finite_field(value[field], label=f"{names[index]}:{field}")
            for index, value in enumerate(predictions)
        ]
        shape = values[0].shape
        if any(value.shape != shape for value in values[1:]):
            raise ValueError(f"VPFP refinement field {field} shapes do not align")
        weights = _broadcast_weights(weights_by_field.get(field), shape)
        solution_scale = max(
            max(_weighted_norm(value, weights) for value in values),
            np.finfo(np.float64).tiny,
        )
        differences = [
            _weighted_norm(right - left, weights)
            for left, right in zip(values, values[1:])
        ]
        relative = [value / solution_scale for value in differences]
        informative_pairs = [
            value > cfg.roundoff_relative_floor for value in relative
        ]
        resolution_independent = not any(informative_pairs)
        orders: list[float] = []
        if not resolution_independent:
            for coarse, fine in zip(differences, differences[1:]):
                if (
                    coarse / solution_scale <= cfg.roundoff_relative_floor
                    or fine / solution_scale <= cfg.roundoff_relative_floor
                ):
                    continue
                orders.append(float(math.log(coarse / fine) / math.log(ratio)))
        order = _median(orders)
        monotone = bool(
            not resolution_independent
            and all(right < left for left, right in zip(differences, differences[1:]))
        )
        if order is not None:
            informative_count += 1
            informative_orders.append(order)
        if monotone:
            monotone_count += 1
        reports[field] = {
            "solution_weighted_l2_scale": solution_scale,
            "successive_absolute_cauchy_differences": differences,
            "successive_relative_cauchy_differences": relative,
            "pair_is_above_roundoff_floor": informative_pairs,
            "resolution_independent": resolution_independent,
            "monotone_difference_reduction": monotone,
            "observed_orders": orders,
            "empirical_order_median": order,
        }

    field_count = len(fields)
    informative_fraction = informative_count / field_count
    aggregate_order = _median(informative_orders)
    auxiliary_passed = bool(
        informative_fraction >= cfg.minimum_informative_field_fraction
        and aggregate_order is not None
        and aggregate_order >= cfg.minimum_reported_order
    )
    return {
        "schema": VPFP_SELF_CONVERGENCE_SCHEMA,
        "passed_as_auxiliary_evidence": auxiliary_passed,
        "reference_solution_used": False,
        "cross_solver_error_used": False,
        "candidate_self_convergence_only": True,
        "accuracy_claim": False,
        "selection_role": "auxiliary_tiebreak_only",
        "strict_monotonicity_is_acceptance_gate": False,
        "constant_or_resolution_ignoring_output_gets_credit": False,
        "level_names": names,
        "resolution_scales": [float(value) for value in scales],
        "geometric_refinement_ratio": ratio,
        "field_count": field_count,
        "informative_field_fraction": informative_fraction,
        "monotone_field_fraction": monotone_count / field_count,
        "aggregate_empirical_order_median": aggregate_order,
        "fields": reports,
    }


def summarize_vpfp_equation_first_self_convergence(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [dict(report) for report in reports]
    if not rows:
        raise ValueError("VPFP self-convergence summary needs cases")
    for row in rows:
        if (
            row.get("schema") != VPFP_SELF_CONVERGENCE_SCHEMA
            or row.get("reference_solution_used") is not False
            or row.get("cross_solver_error_used") is not False
            or row.get("accuracy_claim") is not False
        ):
            raise ValueError("invalid VPFP self-convergence case report")
    orders = [
        float(row["aggregate_empirical_order_median"])
        for row in rows
        if row.get("aggregate_empirical_order_median") is not None
    ]
    return {
        "schema": VPFP_SELF_CONVERGENCE_SUMMARY_SCHEMA,
        "case_count": len(rows),
        "auxiliary_pass_fraction": float(
            np.mean([bool(row["passed_as_auxiliary_evidence"]) for row in rows])
        ),
        "aggregate_empirical_order_median": _median(orders),
        "reference_solution_used": False,
        "cross_solver_error_used": False,
        "accuracy_claim": False,
        "selection_role": "auxiliary_tiebreak_only",
    }
