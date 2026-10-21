"""Trusted adapter from the stable GKW macro ABI to Observable Model."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

from pde_agent_verifier_solver.sdk import AnswerQuery, AnswerView, Case, QueryKind

from .contract import (
    GKW_MACRO_FIELDS,
    coerce_gkw_macro_output,
    gkw_answer_view_from_pairs,
    gkw_point_query_coordinates,
    validate_gkw_compute_budget,
    validate_gkw_macro_query,
)
from .session_binding import validate_gkw_public_session


class GKWLegacyRunner(Protocol):
    """Injected trusted wrapper around the established budgeted runner."""

    def __call__(
        self,
        solver: object,
        public_session: object,
        query_times: object,
        field_indices: object,
        budget: Mapping[str, object],
    ) -> tuple[Mapping[str, object], float]: ...


@dataclass(frozen=True)
class GKWLegacyRuntimeRunner:
    """Adapt the real stable runner's budget object without importing it."""

    run_submission: Callable[..., tuple[Mapping[str, object], float]]
    budget_from_dict: Callable[[Mapping[str, object]], object]

    def __post_init__(self) -> None:
        if not callable(self.run_submission):
            raise TypeError("GKW runtime run_submission must be callable")
        if not callable(self.budget_from_dict):
            raise TypeError("GKW runtime budget_from_dict must be callable")

    def __call__(
        self,
        solver,
        public_session,
        query_times,
        field_indices,
        budget,
    ):
        structured_budget = self.budget_from_dict(dict(budget))
        return self.run_submission(
            solver,
            public_session,
            query_times,
            field_indices,
            structured_budget,
        )

@dataclass(frozen=True)
class _DTypeGuardedMacroSolver:
    wrapped_solver: Callable[..., object]

    @property
    def __wrapped__(self) -> Callable[..., object]:
        return self.wrapped_solver

    def __call__(self, public_session, query_times, field_indices, budget):
        output = self.wrapped_solver(
            public_session, query_times, field_indices, budget
        )
        return _validate_raw_output(output, len(query_times), len(field_indices))


@dataclass(frozen=True)
class GKWLegacyObservableAdapter:
    """Expose one trusted ``solve_macros_with_budget`` implementation."""

    submission: object
    compute_budget: Mapping[str, object]
    runner: GKWLegacyRunner | Callable[..., object]
    public_session_factory: Callable[[Case], object]

    def __post_init__(self) -> None:
        solver = _solver_entrypoint(self.submission)
        if not callable(self.runner):
            raise TypeError("legacy GKW runner must be callable")
        if not callable(self.public_session_factory):
            raise TypeError("GKW public_session_factory must be callable")
        object.__setattr__(self, "_solver", solver)
        object.__setattr__(
            self,
            "compute_budget",
            MappingProxyType(validate_gkw_compute_budget(self.compute_budget)),
        )

    def predict_observables(
        self,
        case: Case,
        query: AnswerQuery,
        budget: float | None = None,
    ) -> AnswerView:
        if budget is not None:
            raise ValueError(
                "legacy GKW adapter uses a structured budget; "
                "a float budget override is not supported"
            )
        if query.kind is QueryKind.GRID:
            times, indices = validate_gkw_macro_query(case, query)
            values = self._run(case, times, indices)
            return gkw_answer_view_from_pairs(
                query,
                values,
                times=times,
                field_indices=indices,
            )
        point_times, point_indices = gkw_point_query_coordinates(
            case, query, require_native_panel=False
        )
        unique_times, time_inverse = _unique_rows(point_times.reshape(-1, 1))
        unique_indices, field_inverse = _unique_rows(point_indices)
        grid_values = self._run(case, unique_times[:, 0], unique_indices)
        selected = {
            name: values[time_inverse, field_inverse, :]
            for name, values in grid_values.items()
        }
        return gkw_answer_view_from_pairs(
            query,
            selected,
            times=point_times,
            field_indices=point_indices,
        )

    def predict_macros(
        self,
        case: Case,
        query: AnswerQuery,
        budget: float | None = None,
    ) -> AnswerView:
        """Expose the same checked result through the unified candidate ABI."""

        return self.predict_observables(case, query, budget)

    def _run(self, case: Case, times, indices) -> dict[str, Any]:
        session = self.public_session_factory(case)
        validate_gkw_public_session(case, session)
        result = self.runner(
            _DTypeGuardedMacroSolver(self._solver),
            session,
            times.copy(),
            indices.copy(),
            dict(self.compute_budget),
        )
        validate_gkw_public_session(case, session)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("legacy GKW runner must return (prediction, wall_seconds)")
        raw, wall_seconds = result
        if (
            isinstance(wall_seconds, bool)
            or not isinstance(wall_seconds, (int, float))
            or not isfinite(float(wall_seconds))
            or float(wall_seconds) < 0.0
        ):
            raise ValueError("legacy GKW runner returned invalid wall time")
        return coerce_gkw_macro_output(
            raw,
            query_count=len(times),
            field_count=len(indices),
        )


def _solver_entrypoint(submission: object) -> Callable[..., object]:
    if callable(submission):
        return submission
    entrypoint = getattr(submission, "solve_macros_with_budget", None)
    if not callable(entrypoint):
        raise TypeError(
            "legacy GKW submission must define solve_macros_with_budget"
        )
    return entrypoint


def _validate_raw_output(
    output: object, query_count: int, field_count: int
) -> dict[str, Any]:
    import numpy as np

    if not isinstance(output, Mapping):
        raise ValueError("GKW macro output must be a mapping")
    snapshot = dict(output)
    if set(snapshot) != set(GKW_MACRO_FIELDS):
        raise ValueError("GKW macro output fields must match exactly")
    expected = (query_count, field_count, 2)
    result = {}
    for name in GKW_MACRO_FIELDS:
        value = np.asarray(snapshot[name])
        if value.dtype.kind not in "iuf":
            raise TypeError(f"GKW macro {name} must use a real numeric dtype")
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError(
                f"GKW macro {name} must be finite with shape {expected}"
            )

        result[name] = value.copy()
    return result



def _unique_rows(values):
    import numpy as np

    unique, first, inverse = np.unique(
        values, axis=0, return_index=True, return_inverse=True
    )
    order = np.argsort(first)
    reordered = unique[order]
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return reordered, remap[inverse]
