"""Diagnostic bridge from legacy VPFP arrays to the unified macro matcher.

The legacy campaign represents a sampled macro trajectory as three arrays:
``rho[t, x, y]``, ``momentum[t, x, y, 2]``, and
``kinetic_energy[t, x, y]``.  This module binds those arrays to their exact
query coordinates and delegates scoring to :class:`VPFPMacroMatcher`.

Bridge payloads are migration diagnostics.  They are deliberately marked as
not participating in winner selection until campaign-level parity is accepted.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Sequence

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Case,
    CaseSource,
    CaseVisibility,
    QueryKind,
    evaluate_macro_prediction,
)

from .macro_matching import VPFP_MACRO_FIELDS, VPFPMacroMatcher


LEGACY_MACRO_BRIDGE_SCHEMA = "vpfp_legacy_macro_match_payload_v1"
LEGACY_MACRO_BRIDGE_VERSION = 1
VPFP_MACRO_AXIS_ORDER = ("t", "x", "y")


def legacy_macro_arrays_to_answer_view(
    *,
    rho: object,
    momentum: object,
    kinetic_energy: object,
    query_times: Sequence[float],
    x: Sequence[float],
    y: Sequence[float],
) -> AnswerView:
    """Convert one legacy macro bundle to the unified grid response ABI.

    Axes must be finite, nonempty, one-dimensional, and strictly increasing.
    Array layout is intentionally strict: scalar fields have shape
    ``(len(query_times), len(x), len(y))`` and momentum has that shape plus a
    final component dimension of length two.  Fields are flattened in C order,
    matching the declared ``(t, x, y)`` shape.
    """

    axes = _validated_axes(query_times=query_times, x=x, y=y)
    return _answer_view_from_axes(
        rho=rho,
        momentum=momentum,
        kinetic_energy=kinetic_energy,
        axes=axes,
    )


def evaluate_legacy_macro_arrays(
    *,
    case_id: str,
    query_times: Sequence[float],
    x: Sequence[float],
    y: Sequence[float],
    prediction_rho: object,
    prediction_momentum: object,
    prediction_kinetic_energy: object,
    truth_rho: object,
    truth_momentum: object,
    truth_kinetic_energy: object,
    matcher: VPFPMacroMatcher | None = None,
) -> dict[str, object]:
    """Score legacy prediction arrays against controller-side truth arrays.

    The returned object contains JSON-native scalars and containers only.  It
    is versioned and coordinate-bound so a parity artifact cannot silently be
    reused for a different irregular grid.
    """

    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("legacy VPFP macro bridge requires a nonempty case_id")

    axes = _validated_axes(query_times=query_times, x=x, y=y)
    prediction = _answer_view_from_axes(
        rho=prediction_rho,
        momentum=prediction_momentum,
        kinetic_energy=prediction_kinetic_energy,
        axes=axes,
    )
    truth = _answer_view_from_axes(
        rho=truth_rho,
        momentum=truth_momentum,
        kinetic_energy=truth_kinetic_energy,
        axes=axes,
    )
    shape = prediction.shape
    query = AnswerQuery.grid(
        fields=VPFP_MACRO_FIELDS,
        t=shape[0],
        x=shape[1],
        y=shape[2],
    )
    case = Case(
        case_id=case_id,
        source=CaseSource.INITIAL,
        visibility=CaseVisibility.DEVELOPMENT,
        public_data={},
        tags=("vpfp", "legacy-macro-bridge"),
    )
    active_matcher = matcher if matcher is not None else VPFPMacroMatcher()
    evaluation = evaluate_macro_prediction(
        matcher=active_matcher,
        case=case,
        prediction=prediction,
        truth=truth,
        query=query,
    )
    result = evaluation.macro_match
    if result is None:  # Defensive: this helper requires the matcher path.
        raise RuntimeError("VPFP macro matcher did not return a match result")

    return {
        "schema": LEGACY_MACRO_BRIDGE_SCHEMA,
        "version": LEGACY_MACRO_BRIDGE_VERSION,
        "contract": active_matcher.contract_id,
        "case_id": case_id,
        "axis_order": list(VPFP_MACRO_AXIS_ORDER),
        "shape": list(shape),
        "coordinate_digest": _coordinate_digest(axes, shape),
        "primary_error": float(result.primary_error),
        "metrics": {
            str(name): float(value)
            for name, value in result.metrics.items()
        },
        "passed": result.passed,
        "failures": list(result.failures),
        "used_for_winner_selection": False,
    }


def _validated_axes(
    *,
    query_times: Sequence[float],
    x: Sequence[float],
    y: Sequence[float],
) -> dict[str, tuple[float, ...]]:
    return {
        "t": _validated_axis("query_times", query_times),
        "x": _validated_axis("x", x),
        "y": _validated_axis("y", y),
    }


def _validated_axis(
    name: str,
    values: Sequence[float],
) -> tuple[float, ...]:
    import numpy as np

    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"legacy VPFP axis {name} must be numeric") from exc
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(
            f"legacy VPFP axis {name} must be a nonempty 1D sequence"
        )
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"legacy VPFP axis {name} must be real numeric data")
    converted = raw.astype(float, copy=False)
    if not np.isfinite(converted).all():
        raise ValueError(f"legacy VPFP axis {name} must be finite")
    if converted.size > 1 and not np.all(np.diff(converted) > 0.0):
        raise ValueError(
            f"legacy VPFP axis {name} must be strictly increasing"
        )
    return tuple(
        0.0 if float(value) == 0.0 else float(value)
        for value in converted
    )


def _answer_view_from_axes(
    *,
    rho: object,
    momentum: object,
    kinetic_energy: object,
    axes: dict[str, tuple[float, ...]],
) -> AnswerView:
    scalar_shape = tuple(len(axes[name]) for name in VPFP_MACRO_AXIS_ORDER)
    rho_array = _validated_array("rho", rho, scalar_shape)
    momentum_array = _validated_array(
        "momentum",
        momentum,
        scalar_shape + (2,),
    )
    energy_array = _validated_array(
        "kinetic_energy",
        kinetic_energy,
        scalar_shape,
    )
    return AnswerView(
        kind=QueryKind.GRID,
        fields={
            "rho": _flatten(rho_array),
            "momentum_x": _flatten(momentum_array[..., 0]),
            "momentum_y": _flatten(momentum_array[..., 1]),
            "kinetic_energy": _flatten(energy_array),
        },
        axes=dict(axes),
        shape=scalar_shape,
    )


def _validated_array(name: str, value: object, shape: tuple[int, ...]):
    import numpy as np

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"legacy VPFP {name} must be a numeric array") from exc
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"legacy VPFP {name} must contain real numeric data")
    if raw.shape != shape:
        raise ValueError(
            f"legacy VPFP {name} shape {raw.shape} does not match {shape}"
        )
    converted = raw.astype(float, copy=False)
    if not np.isfinite(converted).all():
        raise ValueError(f"legacy VPFP {name} must contain only finite values")
    return converted


def _flatten(array) -> tuple[float, ...]:
    return tuple(float(value) for value in array.ravel(order="C"))


def _coordinate_digest(
    axes: dict[str, tuple[float, ...]],
    shape: tuple[int, ...],
) -> str:
    document = {
        "axis_order": list(VPFP_MACRO_AXIS_ORDER),
        "axes": {name: list(axes[name]) for name in VPFP_MACRO_AXIS_ORDER},
        "shape": list(shape),
    }
    encoded = json.dumps(
        document,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + sha256(encoded).hexdigest()


__all__ = [
    "LEGACY_MACRO_BRIDGE_SCHEMA",
    "LEGACY_MACRO_BRIDGE_VERSION",
    "VPFP_MACRO_AXIS_ORDER",
    "evaluate_legacy_macro_arrays",
    "legacy_macro_arrays_to_answer_view",
]
