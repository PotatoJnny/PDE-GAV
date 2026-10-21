"""Stable GKW macroscopic ABI expressed without importing the legacy runtime.

The source contract was frozen by commit ``6947a64``.  This module keeps the
small public boundary in the unified tree while numerical-oracle sessions and
the five-dimensional hidden state remain owned by the established GKW
implementation.
"""

from __future__ import annotations

from math import isfinite
from numbers import Integral, Real
from typing import Any, Mapping

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Case,
    QueryKind,
)


GKW_STABLE_SOURCE_COMMIT = "6947a64c81b34dfbaa386fa58d1700bcc18e112d"
GKW_LEGACY_MACRO_OUTPUT_CONTRACT = "fusion_gkw_macro_output_v1"
GKW_LEGACY_COMPUTE_BUDGET_SCHEMA = "fusion_gkw_macro_compute_budget_v1"
GKW_MACRO_MATCH_CONTRACT = "gkw_continuous_macro_match_v1"
GKW_MACRO_FIELDS = (
    "particle_density",
    "parallel_momentum",
    "kinetic_energy",
    "phi",
)
GKW_GRID_AXIS_ORDER = ("t", "field_row", "complex_component")
GKW_COMPLEX_COMPONENT_AXIS = (0.0, 1.0)


def validate_gkw_compute_budget(value: Mapping[str, object]) -> dict[str, object]:
    """Validate and copy the exact structured budget used by stable GKW."""

    if not isinstance(value, Mapping):
        raise TypeError("legacy GKW compute budget must be a mapping")
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
            f"legacy GKW compute budget keys must be {sorted(required)}"
        )
    if payload["schema"] != GKW_LEGACY_COMPUTE_BUDGET_SCHEMA:
        raise ValueError("unexpected legacy GKW compute-budget schema")
    tier = payload["tier"]
    if isinstance(tier, bool) or not isinstance(tier, int) or tier < 0:
        raise ValueError("legacy GKW budget tier must be a nonnegative integer")
    wall = payload["wall_time_seconds"]
    if (
        isinstance(wall, bool)
        or not isinstance(wall, (int, float))
        or not isfinite(float(wall))
        or float(wall) <= 0.0
    ):
        raise ValueError("legacy GKW wall time must be finite and positive")
    for name in ("cpus", "memory_bytes"):
        item = payload[name]
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError(f"legacy GKW {name} must be a positive integer")
    gpus = payload["gpus"]
    if isinstance(gpus, bool) or not isinstance(gpus, int) or gpus < 0:
        raise ValueError("legacy GKW gpus must be a nonnegative integer")
    return payload


def gkw_macro_query_panel(case: Case) -> tuple[Any, Any]:
    """Return copied public query times and ``(s,kx,ky)`` index rows."""

    import numpy as np

    query = case.public_data.get("macro_query")
    if not isinstance(query, Mapping):
        raise ValueError("GKW Case needs a public macro_query panel")
    if set(query) != {"times", "field_indices", "output_contract"}:
        raise ValueError("GKW public macro-query keys changed")
    contract = query.get("output_contract")
    if contract != GKW_LEGACY_MACRO_OUTPUT_CONTRACT:
        raise ValueError("GKW Case uses an unexpected macro output contract")
    times = np.asarray(query.get("times"), dtype=np.float64)
    indices = np.asarray(query.get("field_indices"))
    if (
        times.ndim != 1
        or times.size <= 0
        or not np.isfinite(times).all()
        or not np.all(np.diff(times) > 0.0)
    ):
        raise ValueError("GKW public macro query times must increase and be finite")
    if (
        indices.ndim != 2
        or indices.shape[0] <= 0
        or indices.shape[1] != 3
        or not np.issubdtype(indices.dtype, np.integer)
    ):
        raise ValueError("GKW public field indices must be integer [N,3]")
    indices = indices.astype(np.int64, copy=True)
    if len(np.unique(indices, axis=0)) != len(indices):
        raise ValueError("GKW public field-index panel contains duplicates")

    arrays = gkw_public_arrays(case)
    shape = tuple(len(arrays[name]) for name in ("s", "kx", "ky"))
    for axis, extent in enumerate(shape):
        if np.any(indices[:, axis] < 0) or np.any(indices[:, axis] >= extent):
            raise IndexError("GKW public field index leaves the public grid")
    return times.copy(), indices


def gkw_public_arrays(case: Case) -> Mapping[str, object]:
    arrays = case.public_data.get("arrays")
    if not isinstance(arrays, Mapping):
        raise ValueError("GKW Case needs public equation arrays")
    required = {"s", "kx", "ky", "B", "ints", "parseval_ky"}
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"GKW Case is missing public arrays: {sorted(missing)}")
    return arrays


def gkw_public_query_domain(case: Case) -> tuple[tuple[float, float], tuple[int, int, int]]:
    """Return the continuous-time/full-field domain exposed by the equation.

    Unlike ``gkw_macro_query_panel``, this contract contains no realization-
    specific scoring coordinates.  It is derived entirely from the public
    numerical-oracle interval and the public ``(s,kx,ky)`` arrays.
    """

    import numpy as np

    oracle = case.public_data.get("oracle_protocol")
    if not isinstance(oracle, Mapping):
        raise ValueError("GKW Case needs a public numerical-oracle contract")
    interval = np.asarray(oracle.get("time_interval_closed"), dtype=np.float64)
    if (
        interval.shape != (2,)
        or not np.isfinite(interval).all()
        or float(interval[1]) <= float(interval[0])
    ):
        raise ValueError("GKW public time interval is invalid")

    arrays = gkw_public_arrays(case)
    shape = tuple(len(arrays[name]) for name in ("s", "kx", "ky"))
    if any(extent < 1 for extent in shape):
        raise ValueError("GKW public field grid is empty")
    declared_shape = np.asarray(oracle.get("field_grid_shape"))
    if (
        declared_shape.shape != (3,)
        or declared_shape.dtype.kind not in "iu"
        or tuple(int(value) for value in declared_shape) != shape
    ):
        raise ValueError("GKW public field-grid shape differs from its arrays")
    return (float(interval[0]), float(interval[1])), shape


def validate_gkw_macro_query(
    case: Case,
    query: AnswerQuery,
) -> tuple[Any, Any]:
    """Validate a native grid query and return its exact public panel."""

    if set(query.fields) != set(GKW_MACRO_FIELDS):
        raise ValueError("GKW macro query requires all registered macro fields")
    if query.kind is not QueryKind.GRID:
        raise ValueError("this helper requires a GKW GRID query")
    required_axes = {"t", "field_row"}
    if set(query.resolution) != required_axes:
        raise ValueError(
            "GKW macro grid query axes must be exactly "
            f"{sorted(required_axes)}"
        )
    times, indices = gkw_macro_query_panel(case)
    expected = {"t": len(times), "field_row": len(indices)}
    for name, count in query.resolution.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"GKW query resolution {name!r} must be positive")
        if count != expected[name]:
            raise ValueError(
                f"GKW adapter requires the native {name} panel of "
                f"{expected[name]} entries"
            )
    return times, indices


def gkw_point_query_coordinates(
    case: Case,
    query: AnswerQuery,
    *,
    require_native_panel: bool = True,
) -> tuple[Any, Any]:
    """Validate explicit GKW ``(t,s_index,kx_index,ky_index)`` points."""

    import numpy as np

    if set(query.fields) != set(GKW_MACRO_FIELDS):
        raise ValueError("GKW macro query requires all registered macro fields")
    if query.kind is not QueryKind.POINTS:
        raise ValueError("this helper requires a GKW POINTS query")
    (lower, upper), shape = gkw_public_query_domain(case)
    panel_times = panel_indices = None
    if require_native_panel:
        panel_times, panel_indices = gkw_macro_query_panel(case)
    times = []
    indices = []
    required = {"t", "s_index", "kx_index", "ky_index"}
    for point in query.points:
        if set(point) != required:
            raise ValueError(
                "GKW points require t, s_index, kx_index and ky_index"
            )
        time_value = point["t"]
        if (
            isinstance(time_value, bool)
            or not isinstance(time_value, Real)
            or not isfinite(float(time_value))
            or float(time_value) < lower
            or float(time_value) > upper
            or (
                require_native_panel
                and not np.any(
                    np.isclose(
                        panel_times, float(time_value), rtol=0.0, atol=1.0e-12
                    )
                )
            )
        ):
            raise ValueError("GKW point time leaves the allowed query panel")
        row = []
        for name, extent in zip(
            ("s_index", "kx_index", "ky_index"), shape
        ):
            value = point[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) < 0
                or int(value) >= extent
            ):
                raise ValueError(f"GKW point {name} leaves the public grid")
            row.append(int(value))
        if require_native_panel and not np.any(
            np.all(panel_indices == np.asarray(row)[None, :], axis=1)
        ):
            raise ValueError("GKW point field index is not on the released panel")
        times.append(float(time_value))
        indices.append(row)
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(indices, dtype=np.int64),
    )


def gkw_field_quadrature_weights(
    case: Case,
    field_indices: Any,
) -> Any:
    """Stable ``abs(B * ints * parseval_ky)`` field weights."""

    import numpy as np

    arrays = gkw_public_arrays(case)
    indices = np.asarray(field_indices, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("GKW field weights require integer [N,3] indices")
    magnetic = np.asarray(arrays["B"], dtype=np.float64)
    parallel = np.asarray(arrays["ints"], dtype=np.float64)
    parseval = np.asarray(arrays["parseval_ky"], dtype=np.float64)
    weights = np.abs(
        magnetic[indices[:, 0]]
        * parallel[indices[:, 0]]
        * parseval[indices[:, 2]]
    )
    if (
        weights.shape != (len(indices),)
        or not np.isfinite(weights).all()
        or np.any(weights < 0.0)
        or not np.any(weights > 0.0)
    ):
        raise ValueError("GKW macro field weights are invalid")
    return weights


def coerce_gkw_macro_output(
    output: Any,
    *,
    query_count: int,
    field_count: int,
) -> dict[str, Any]:
    """Require the exact stable four-field real/imag output mapping."""

    import numpy as np

    if not isinstance(output, Mapping) or set(output) != set(GKW_MACRO_FIELDS):
        raise ValueError("GKW macro output fields must match the contract exactly")
    expected = (int(query_count), int(field_count), 2)
    result = {}
    for name in GKW_MACRO_FIELDS:
        raw = np.asarray(output[name])
        if raw.dtype.kind not in "iuf":
            raise TypeError(f"GKW macro {name} must use a real numeric dtype")
        value = raw.astype(np.float64, copy=True)
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError(
                f"GKW macro {name} must be finite with shape {expected}"
            )
        result[name] = value
    return result


def gkw_answer_view_from_pairs(
    query: AnswerQuery,
    values: Mapping[str, Any],
    *,
    times: Any,
    field_indices: Any,
) -> AnswerView:
    """Encode stable complex pairs as finite unified float fields."""

    import numpy as np

    if query.kind is QueryKind.GRID:
        coerced = coerce_gkw_macro_output(
            values,
            query_count=len(times),
            field_count=len(field_indices),
        )
        shape = (len(times), len(field_indices), 2)
        return AnswerView(
            kind=QueryKind.GRID,
            fields={
                name: tuple(value.reshape(-1, order="C").tolist())
                for name, value in coerced.items()
            },
            axes={
                "t": tuple(float(item) for item in times),
                "field_row": tuple(float(item) for item in range(len(field_indices))),
                "complex_component": GKW_COMPLEX_COMPONENT_AXIS,
            },
            shape=shape,
        )
    if query.kind is not QueryKind.POINTS:
        raise ValueError("unsupported GKW answer query kind")
    point_count = len(query.points)
    if not isinstance(values, Mapping) or set(values) != set(GKW_MACRO_FIELDS):
        raise ValueError("GKW point macro fields must match the contract exactly")
    result = {}
    for name in GKW_MACRO_FIELDS:
        raw = np.asarray(values[name])
        if raw.dtype.kind not in "iuf":
            raise TypeError(f"GKW macro {name} must use a real numeric dtype")
        value = raw.astype(np.float64, copy=True)
        if value.shape != (point_count, 2) or not np.isfinite(value).all():
            raise ValueError(
                f"GKW point macro {name} must be finite with shape {(point_count, 2)}"
            )
        result[name] = tuple(value.reshape(-1, order="C").tolist())
    return AnswerView(
        kind=QueryKind.POINTS,
        fields=result,
        points=query.points,
        shape=(point_count, 2),
    )
