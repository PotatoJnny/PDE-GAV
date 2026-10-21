"""Equation-owned VPFP macro programs used by reference experiments."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Case,
    QueryKind,
)

if TYPE_CHECKING:
    from pde_agent_verifier_solver.sdk import CandidateSubmission

from .macro_matching import VPFP_MACRO_FIELDS


VPFP_HOLD_INITIAL_CANDIDATE_ID = "vpfp-hold-initial-macros-v1"
VPFP_HOLD_INITIAL_AGENT_ID = "vpfp-built-in-cold-start"


class _VPFPHoldInitialPrepared:
    """Query-blind copy of public initial macros with periodic interpolation."""

    def __init__(self, case: Case) -> None:
        import numpy as np

        arrays = _public_arrays(case)
        self._t = _axis(np, arrays.get("t"), "t").copy()
        self._x = _uniform_periodic_axis(arrays.get("x"), "x")
        self._y = _uniform_periodic_axis(arrays.get("y"), "y")
        nx, ny = len(self._x), len(self._y)
        momentum = _array(
            np, arrays.get("momentum0"), "momentum0", (nx, ny, 2)
        ).copy()
        self._fields = {
            "rho": _array(
                np, arrays.get("rho0"), "rho0", (nx, ny)
            ).copy(),
            "momentum_x": momentum[..., 0].copy(),
            "momentum_y": momentum[..., 1].copy(),
            "kinetic_energy": _array(
                np,
                arrays.get("kinetic_energy0"),
                "kinetic_energy0",
                (nx, ny),
            ).copy(),
        }

    def evaluate(self, query: AnswerQuery) -> AnswerView:
        import numpy as np

        if not isinstance(query, AnswerQuery):
            raise TypeError("VPFP hold-initial projection needs AnswerQuery")
        if query.kind is not QueryKind.POINTS:
            raise ValueError(
                "VPFP continuous cold start supports POINTS queries only"
            )
        _require_all_macro_fields(query)
        coordinates = _point_coordinates(query)
        times = coordinates[:, 0]
        if np.any(times < self._t[0]) or np.any(times > self._t[-1]):
            raise ValueError("VPFP scoring time leaves the public interval")
        values = {
            name: _periodic_bilinear(
                self._x,
                self._y,
                field,
                coordinates[:, 1],
                coordinates[:, 2],
            )
            for name, field in self._fields.items()
        }
        return AnswerView(
            kind=QueryKind.POINTS,
            fields={
                name: tuple(float(value) for value in values[name])
                for name in query.fields
            },
            points=query.points,
            shape=(len(query.points),),
        )


@dataclass(frozen=True)
class VPFPHoldInitialMacroProgram:
    """Weak, valid cold starter that holds public initial moments fixed.

    The program contains no learned parameters and has never participated in
    an Agent iteration. It is intentionally weak enough for a fresh learning
    curve while still satisfying the exact macro response ABI.

    This in-package executable is reference-only until a formal bundle
    attestor and isolated candidate executor seal and run it out of process.
    """

    reference_only = True
    formal_claims_allowed = False

    def prepare_solution(
        self,
        case: Case,
        solve_budget: float | None = None,
    ) -> _VPFPHoldInitialPrepared:
        """Copy public initial data before private scoring points are known."""

        if not isinstance(case, Case):
            raise TypeError("VPFP hold-initial program needs a Case")
        _validate_budget(solve_budget)
        return _VPFPHoldInitialPrepared(case)

    def predict_observables(
        self,
        case: Case,
        query: AnswerQuery,
        budget: float | None = None,
    ) -> AnswerView:
        """Canonical Observable Task entry point."""

        return self.predict_macros(case, query, budget=budget)

    def predict_macros(
        self,
        case: Case,
        query: AnswerQuery,
        budget: float | None = None,
    ) -> AnswerView:
        if not isinstance(case, Case):
            raise TypeError("VPFP hold-initial program needs a Case")
        if not isinstance(query, AnswerQuery):
            raise TypeError("VPFP hold-initial program needs an AnswerQuery")
        _validate_budget(budget)
        if query.kind is QueryKind.POINTS:
            return _VPFPHoldInitialPrepared(case).evaluate(query)
        if query.kind is not QueryKind.GRID:
            raise ValueError(
                "VPFP hold-initial program received an unknown query kind"
            )
        _require_all_macro_fields(query)
        if set(query.resolution) != {"t", "x", "y"}:
            raise ValueError(
                "VPFP hold-initial program requires t/x/y grid resolutions"
            )
        arrays_value = _public_arrays(case)
        import numpy as np

        axes = {
            name: _axis(np, arrays_value.get(name), name)
            for name in ("t", "x", "y")
        }
        for name in ("x", "y"):
            requested = query.resolution.get(name)
            if requested is not None and requested != len(axes[name]):
                raise ValueError(
                    "VPFP hold-initial program requires native "
                    f"{name}-resolution {len(axes[name])}, got {requested}"
                )
        query_times = _requested_axis(
            np,
            axes["t"],
            query.resolution.get("t"),
        )
        nx = len(axes["x"])
        ny = len(axes["y"])
        rho0 = _array(
            np,
            arrays_value.get("rho0"),
            "rho0",
            (nx, ny),
        )
        momentum0 = _array(
            np,
            arrays_value.get("momentum0"),
            "momentum0",
            (nx, ny, 2),
        )
        kinetic_energy0 = _array(
            np,
            arrays_value.get("kinetic_energy0"),
            "kinetic_energy0",
            (nx, ny),
        )
        shape = (len(query_times), nx, ny)
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
                "t": tuple(float(value) for value in query_times),
                "x": tuple(float(value) for value in axes["x"]),
                "y": tuple(float(value) for value in axes["y"]),
            },
            shape=shape,
        )


def make_vpfp_hold_initial_cold_start(
    *,
    candidate_id: str = VPFP_HOLD_INITIAL_CANDIDATE_ID,
    agent_id: str = VPFP_HOLD_INITIAL_AGENT_ID,
) -> CandidateSubmission:
    """Create the round-zero reference candidate from its actual source bytes."""

    from pde_agent_verifier_solver.sdk import CandidateRef, CandidateSubmission

    source = Path(__file__).resolve()
    bundle_sha256 = sha256(source.read_bytes()).hexdigest()
    return CandidateSubmission(
        ref=CandidateRef(
            candidate_id=candidate_id,
            bundle_sha256=bundle_sha256,
            round_index=0,
            agent_id=agent_id,
            bundle_uri=str(source),
        ),
        program=VPFPHoldInitialMacroProgram(),
        program_factory=VPFPHoldInitialMacroProgram,
    )


def _public_arrays(case: Case) -> Mapping[str, object]:
    arrays = case.public_data.get("arrays")
    if not isinstance(arrays, Mapping):
        raise ValueError("VPFP hold-initial program needs public arrays")
    return arrays


def _require_all_macro_fields(query: AnswerQuery) -> None:
    if len(query.fields) != len(VPFP_MACRO_FIELDS) or set(
        query.fields
    ) != set(VPFP_MACRO_FIELDS):
        raise ValueError(
            "VPFP hold-initial program requires all registered macro fields"
        )


def _uniform_periodic_axis(value: object, name: str):
    import numpy as np

    axis = _axis(np, value, name).copy()
    if len(axis) < 2:
        raise ValueError(f"VPFP periodic axis {name} needs at least two points")
    spacing = np.diff(axis)
    tolerance = 64.0 * np.finfo(np.float64).eps * max(
        1.0, float(np.max(np.abs(axis)))
    )
    if not np.allclose(
        spacing, spacing[0], rtol=1.0e-10, atol=tolerance
    ):
        raise ValueError(
            f"VPFP periodic axis {name} must be uniformly spaced"
        )
    return axis


def _point_coordinates(query: AnswerQuery):
    import numpy as np

    try:
        coordinates = np.asarray(
            [
                [float(point[name]) for name in ("t", "x", "y")]
                for point in query.points
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "VPFP scoring points must use numeric (t,x,y)"
        ) from exc
    if coordinates.shape != (len(query.points), 3) or not np.all(
        np.isfinite(coordinates)
    ):
        raise ValueError(
            "VPFP scoring points must be finite physical (t,x,y)"
        )
    return coordinates


def _periodic_bilinear(x_axis, y_axis, values, x, y):
    import numpy as np

    dx = float(x_axis[1] - x_axis[0])
    dy = float(y_axis[1] - y_axis[0])
    x_phase = np.mod((x - x_axis[0]) / dx, len(x_axis))
    y_phase = np.mod((y - y_axis[0]) / dy, len(y_axis))
    x_lower = np.floor(x_phase).astype(np.int64)
    y_lower = np.floor(y_phase).astype(np.int64)
    x_upper = (x_lower + 1) % len(x_axis)
    y_upper = (y_lower + 1) % len(y_axis)
    x_weight = x_phase - x_lower
    y_weight = y_phase - y_lower
    return (
        values[x_lower, y_lower] * (1.0 - x_weight) * (1.0 - y_weight)
        + values[x_upper, y_lower] * x_weight * (1.0 - y_weight)
        + values[x_lower, y_upper] * (1.0 - x_weight) * y_weight
        + values[x_upper, y_upper] * x_weight * y_weight
    )


def _validate_budget(value: float | None) -> None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError("candidate solve budget must be finite and positive")


def _axis(np, value: object, name: str):
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


def _array(np, value: object, name: str, shape: tuple[int, ...]):
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


def _requested_axis(np, native, count: int | None):
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


def _flatten(array) -> tuple[float, ...]:
    return tuple(float(value) for value in array.reshape(-1, order="C"))


__all__ = [
    "VPFP_HOLD_INITIAL_AGENT_ID",
    "VPFP_HOLD_INITIAL_CANDIDATE_ID",
    "VPFPHoldInitialMacroProgram",
    "make_vpfp_hold_initial_cold_start",
]
