"""Fresh VPFP cold start: hold the public initial macroscopic fields fixed.

This is a deliberately weak one-file reference candidate.  It contains no
trained weights, does not import the VPFP plugin implementation, and exposes
only the public macro candidate ABI expected by the reference runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

import numpy as np

from pde_agent_verifier_solver.sdk import AnswerQuery, AnswerView, Case, QueryKind


_MACRO_FIELDS = ("rho", "momentum_x", "momentum_y", "kinetic_energy")


def predict_macros(
    case: Case,
    query: AnswerQuery,
    budget: float | None = None,
) -> AnswerView:
    """Repeat public t=0 moments over the requested time grid."""

    if not isinstance(case, Case):
        raise TypeError("VPFP hold-initial candidate needs a Case")
    if not isinstance(query, AnswerQuery):
        raise TypeError("VPFP hold-initial candidate needs an AnswerQuery")
    if query.kind is not QueryKind.GRID:
        raise ValueError("VPFP hold-initial candidate supports GRID queries only")
    if len(query.fields) != len(_MACRO_FIELDS) or set(query.fields) != set(
        _MACRO_FIELDS
    ):
        raise ValueError(
            "VPFP hold-initial candidate requires rho, momentum_x, "
            "momentum_y, and kinetic_energy"
        )
    if set(query.resolution) != {"t", "x", "y"}:
        raise ValueError(
            "VPFP hold-initial candidate requires t/x/y grid resolutions"
        )
    if budget is not None and (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not isfinite(float(budget))
        or float(budget) <= 0.0
    ):
        raise ValueError("candidate compute budget must be finite and positive")

    arrays_value = case.public_data.get("arrays")
    if not isinstance(arrays_value, Mapping):
        raise ValueError("VPFP hold-initial candidate needs public arrays")

    native_t = _axis(arrays_value.get("t"), "t")
    x = _axis(arrays_value.get("x"), "x")
    y = _axis(arrays_value.get("y"), "y")
    for name, axis in (("x", x), ("y", y)):
        requested = query.resolution.get(name)
        if requested is not None and requested != len(axis):
            raise ValueError(
                "VPFP hold-initial candidate requires native "
                f"{name}-resolution {len(axis)}, got {requested}"
            )
    requested_t = _requested_axis(native_t, query.resolution.get("t"))
    nx = len(x)
    ny = len(y)
    rho0 = _array(arrays_value.get("rho0"), "rho0", (nx, ny))
    momentum0 = _array(
        arrays_value.get("momentum0"),
        "momentum0",
        (nx, ny, 2),
    )
    kinetic_energy0 = _array(
        arrays_value.get("kinetic_energy0"),
        "kinetic_energy0",
        (nx, ny),
    )

    shape = (len(requested_t), nx, ny)
    rho = np.broadcast_to(rho0, shape)
    momentum = np.broadcast_to(momentum0, (*shape, 2))
    kinetic_energy = np.broadcast_to(kinetic_energy0, shape)
    return AnswerView(
        kind=QueryKind.GRID,
        fields={
            "rho": _flatten(rho),
            "momentum_x": _flatten(momentum[..., 0]),
            "momentum_y": _flatten(momentum[..., 1]),
            "kinetic_energy": _flatten(kinetic_energy),
        },
        axes={
            "t": tuple(float(value) for value in requested_t),
            "x": tuple(float(value) for value in x),
            "y": tuple(float(value) for value in y),
        },
        shape=shape,
    )


def _axis(value: object, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"VPFP public axis {name} must be numeric") from exc
    if raw.ndim != 1 or raw.size < 1 or raw.dtype.kind not in "iuf":
        raise ValueError(f"VPFP public axis {name} must be nonempty real 1D")
    result = raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"VPFP public axis {name} is non-finite")
    if len(result) > 1 and not np.all(np.diff(result) > 0.0):
        raise ValueError(f"VPFP public axis {name} must be strictly increasing")
    return result


def _array(value: object, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"VPFP public array {name} must be numeric") from exc
    if raw.shape != shape or raw.dtype.kind not in "iuf":
        raise ValueError(
            f"VPFP public array {name} must be real with shape {shape}"
        )
    result = raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"VPFP public array {name} is non-finite")
    return result


def _requested_axis(native: np.ndarray, count: int | None) -> np.ndarray:
    if count is None or count == len(native):
        return native.copy()
    if count == 1:
        return np.asarray(
            [(float(native[0]) + float(native[-1])) / 2.0],
            dtype=np.float64,
        )
    return np.linspace(
        float(native[0]),
        float(native[-1]),
        int(count),
        dtype=np.float64,
    )


def _flatten(array: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in array.reshape(-1, order="C"))
