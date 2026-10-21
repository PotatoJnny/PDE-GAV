"""Sample controller-only truth arrays on finite grids or points.

NS and VPFP store truth as NumPy arrays on regular time/space grids. Their
Problem classes share this deliberately small interpolation helper. NumPy is
imported inside the function, so the framework and Toy PDE remain lightweight.
"""

from __future__ import annotations

from itertools import product
from typing import Mapping, Sequence

from pde_agent_verifier_solver.sdk import AnswerQuery, AnswerView, QueryKind


def answer_from_regular_grid(
    query: AnswerQuery,
    *,
    axis_order: Sequence[str],
    native_axes: Mapping[str, object],
    fields: Mapping[str, object],
) -> AnswerView:
    """Sample scalar fields that all use the same regular grid.

    Every field must have shape [len(axis) for axis in axis_order]. Grid
    queries use linear interpolation. Point queries use multilinear
    interpolation. Returned values are flattened in C order.
    """

    import numpy as np

    axes = {
        name: np.asarray(native_axes[name], dtype=np.float64)
        for name in axis_order
    }
    arrays = {name: np.asarray(value) for name, value in fields.items()}
    expected = tuple(len(axes[name]) for name in axis_order)
    for name, value in arrays.items():
        if value.shape != expected:
            raise ValueError(
                f"field {name!r} has shape {value.shape}; expected {expected}"
            )
    missing = set(query.fields) - set(arrays)
    if missing:
        raise ValueError(f"truth does not provide requested fields: {sorted(missing)}")

    if query.kind is QueryKind.GRID:
        requested_axes = {
            name: _requested_axis(np, axes[name], query.resolution.get(name))
            for name in axis_order
        }
        sampled = {
            name: _resample_grid(
                np,
                arrays[name],
                tuple(axes[axis] for axis in axis_order),
                tuple(requested_axes[axis] for axis in axis_order),
            )
            for name in query.fields
        }
        shape = tuple(len(requested_axes[name]) for name in axis_order)
        return AnswerView(
            kind=query.kind,
            fields={
                name: tuple(value.reshape(-1).tolist())
                for name, value in sampled.items()
            },
            axes={
                name: tuple(value.tolist())
                for name, value in requested_axes.items()
            },
            shape=shape,
        )

    sampled_points = {name: [] for name in query.fields}
    for point in query.points:
        coordinates = tuple(float(point[name]) for name in axis_order)
        for name in query.fields:
            sampled_points[name].append(
                _multilinear_point(
                    np,
                    arrays[name],
                    tuple(axes[axis] for axis in axis_order),
                    coordinates,
                )
            )
    return AnswerView(
        kind=query.kind,
        fields={name: tuple(values) for name, values in sampled_points.items()},
        points=query.points,
        shape=(len(query.points),),
    )


def _requested_axis(np, native, count: int | None):
    if count is None or int(count) == len(native):
        return native
    if int(count) <= 0:
        raise ValueError("requested grid resolutions must be positive")
    if int(count) == 1:
        return np.asarray([(float(native[0]) + float(native[-1])) / 2.0])
    return np.linspace(float(native[0]), float(native[-1]), int(count))


def _resample_grid(np, values, old_axes, new_axes):
    result = np.asarray(values, dtype=np.float64)
    for dimension, (old, new) in enumerate(zip(old_axes, new_axes)):
        if len(old) == len(new) and np.array_equal(old, new):
            continue
        moved = np.moveaxis(result, dimension, 0)
        flat = moved.reshape(len(old), -1)
        interpolated = np.empty((len(new), flat.shape[1]), dtype=np.float64)
        for column in range(flat.shape[1]):
            interpolated[:, column] = np.interp(new, old, flat[:, column])
        moved = interpolated.reshape((len(new),) + moved.shape[1:])
        result = np.moveaxis(moved, 0, dimension)
    return result


def _multilinear_point(np, values, axes, coordinates):
    brackets = []
    for axis, coordinate in zip(axes, coordinates):
        if coordinate < float(axis[0]) or coordinate > float(axis[-1]):
            raise ValueError(
                f"query coordinate {coordinate} lies outside the truth grid"
            )
        upper = int(np.searchsorted(axis, coordinate, side="right"))
        upper = min(max(upper, 1), len(axis) - 1)
        lower = upper - 1
        span = float(axis[upper] - axis[lower])
        weight = 0.0 if span == 0.0 else (
            coordinate - float(axis[lower])
        ) / span
        brackets.append((lower, upper, weight))

    result = 0.0
    for corners in product((0, 1), repeat=len(axes)):
        index = []
        weight = 1.0
        for choose_upper, (lower, upper, upper_weight) in zip(corners, brackets):
            index.append(upper if choose_upper else lower)
            weight *= upper_weight if choose_upper else 1.0 - upper_weight
        result += weight * float(values[tuple(index)])
    return result
