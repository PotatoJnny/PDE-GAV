"""Small metric helpers shared by VPFP tasks."""

from __future__ import annotations

from math import sqrt
from statistics import fmean
from typing import Mapping, Sequence

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    ArtifactEvaluation,
    Case,
    CaseEvaluation,
)

from ..._private_queries import (
    default_query_fields,
    development_point_count,
    public_axis_bounds,
    random_point_query,
)


VPFP_MACRO_FIELDS = (
    "rho",
    "momentum_x",
    "momentum_y",
    "kinetic_energy",
)


def private_vpfp_macro_query(
    case: Case,
    settings: Mapping[str, object],
    seed: int,
) -> AnswerQuery:
    """Sample the full public torus, not only the stored endpoint-free grid."""

    bounds = public_axis_bounds(case, ("t", "x", "y"))
    question = case.public_data.get("question")
    if isinstance(question, Mapping):
        domain = question.get("domain")
        grid = question.get("grid")
        if isinstance(domain, Mapping) and isinstance(grid, Mapping):
            import numpy as np

            try:
                time_domain = np.asarray(domain.get("time"), dtype=np.float64)
                spatial_periods = np.asarray(
                    grid.get("x_domain"), dtype=np.float64
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "VPFP public domain bounds must be numeric"
                ) from exc
            if (
                time_domain.shape != (2,)
                or spatial_periods.shape != (2,)
                or not np.all(np.isfinite(time_domain))
                or not np.all(np.isfinite(spatial_periods))
                or time_domain[1] <= time_domain[0]
                or np.any(spatial_periods <= 0.0)
            ):
                raise ValueError(
                    "VPFP public domain bounds must have positive extent"
                )
            bounds = {
                "t": (float(time_domain[0]), float(time_domain[1])),
                "x": (0.0, float(spatial_periods[0])),
                "y": (0.0, float(spatial_periods[1])),
            }
    return random_point_query(
        fields=default_query_fields(settings),
        bounds=bounds,
        point_count=development_point_count(settings),
        seed=seed,
    )


def field_evaluation(
    case_id: str,
    prediction: AnswerView,
    truth: AnswerView,
    fields: Sequence[str] = VPFP_MACRO_FIELDS,
) -> CaseEvaluation:
    errors = {
        field: relative_l2(
            prediction.fields[field],
            truth.fields[field],
        )
        for field in fields
    }
    primary = fmean(errors.values())
    return CaseEvaluation(
        case_id=case_id,
        primary_error=primary,
        metrics={
            "macro_relative_l2": primary,
            **{
                f"{name}_relative_l2": value
                for name, value in errors.items()
            },
        },
        prediction=prediction,
    )


def choose_smallest_error(
    candidates: Sequence[ArtifactEvaluation],
    incumbent: ArtifactEvaluation | None,
) -> ArtifactEvaluation:
    choices = tuple(candidates) + ((incumbent,) if incumbent else ())
    return min(choices, key=lambda value: value.primary_error)


def relative_l2(actual, expected) -> float:
    if len(actual) != len(expected):
        raise ValueError(
            "prediction and truth contain different sample counts"
        )
    squared_error = sum(
        (float(a) - float(b)) ** 2
        for a, b in zip(actual, expected)
    )
    squared_truth = sum(float(value) ** 2 for value in expected)
    return sqrt(squared_error / max(squared_truth, 1.0e-30))


def scalar_truth_statistics(
    truth: AnswerView,
    requests: Sequence[str],
) -> Mapping[str, float]:
    result: dict[str, float] = {}
    for request in requests:
        field, separator, statistic = request.rpartition(".")
        if not separator or field not in truth.fields:
            raise ValueError(
                "statistics names use '<field>.mean' or '<field>.rms'"
            )
        values = tuple(float(value) for value in truth.fields[field])
        if statistic == "mean":
            result[request] = fmean(values)
        elif statistic == "rms":
            result[request] = sqrt(fmean(value * value for value in values))
        else:
            raise ValueError(f"unsupported truth statistic {statistic!r}")
    return result
