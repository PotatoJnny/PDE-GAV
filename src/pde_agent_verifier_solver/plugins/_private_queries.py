"""Plugin-owned helpers for fresh Development point queries.

The shared engine owns access control and seeds.  PDE Tasks own the physical
domain from which those seeds may select coordinates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from random import Random

from pde_agent_verifier_solver.sdk import AnswerQuery, Case


def development_point_count(settings: Mapping[str, object]) -> int:
    """Return the user-configured number of private Development points."""

    value = settings.get("development_point_count")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            "private Development evaluation needs a positive "
            "development_point_count"
        )
    return value


def default_query_fields(settings: Mapping[str, object]) -> tuple[str, ...]:
    """Use the Run's Task query only as a field-selection template."""

    query = settings.get("default_query")
    if not isinstance(query, AnswerQuery):
        raise TypeError("private Development evaluation needs a default_query")
    return tuple(query.fields)


def public_axis_bounds(case: Case, names: Sequence[str]) -> dict[str, tuple[float, float]]:
    """Validate ordered public regular-grid axes and return endpoint bounds."""

    arrays = case.public_data.get("arrays")
    if not isinstance(arrays, Mapping):
        raise ValueError("private Development query needs public Case arrays")

    import numpy as np

    bounds: dict[str, tuple[float, float]] = {}
    for name in names:
        values = np.asarray(arrays.get(name))
        if (
            values.ndim != 1
            or values.size < 2
            or values.dtype.kind not in "iuf"
            or not np.all(np.isfinite(values))
            or not np.all(np.diff(values.astype(float)) > 0.0)
        ):
            raise ValueError(
                f"private Development axis {name!r} must be finite and increasing"
            )
        bounds[name] = (float(values[0]), float(values[-1]))
    return bounds


def random_point_query(
    *,
    fields: Sequence[str],
    bounds: Mapping[str, tuple[float, float]],
    point_count: int,
    seed: int,
) -> AnswerQuery:
    """Draw a reproducible finite POINTS query inside declared public bounds."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("private Development query seed must be nonnegative")
    if isinstance(point_count, bool) or not isinstance(point_count, int) or point_count < 1:
        raise ValueError("private Development point count must be positive")
    prepared_bounds: dict[str, tuple[float, float]] = {}
    for name, interval in bounds.items():
        try:
            lower, upper = (float(value) for value in interval)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"private Development bounds for {name!r} are invalid"
            ) from exc
        if not (isfinite(lower) and isfinite(upper)) or upper <= lower:
            raise ValueError(
                f"private Development bounds for {name!r} must be finite and nonempty"
            )
        prepared_bounds[name] = (lower, upper)
    if not prepared_bounds:
        raise ValueError("private Development query needs at least one coordinate axis")

    random = Random(seed)
    points = tuple(
        {
            name: lower + (upper - lower) * random.random()
            for name, (lower, upper) in prepared_bounds.items()
        }
        for _index in range(point_count)
    )
    return AnswerQuery.at_points(fields=tuple(fields), points=points)


def random_query_from_public_axes(
    case: Case,
    settings: Mapping[str, object],
    seed: int,
    *,
    axis_names: Sequence[str],
) -> AnswerQuery:
    """Build a private query from public regular-grid endpoint bounds."""

    return random_point_query(
        fields=default_query_fields(settings),
        bounds=public_axis_bounds(case, axis_names),
        point_count=development_point_count(settings),
        seed=seed,
    )


__all__ = [
    "default_query_fields",
    "development_point_count",
    "public_axis_bounds",
    "random_point_query",
    "random_query_from_public_axes",
]
