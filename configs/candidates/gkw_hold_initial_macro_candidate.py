"""Fresh GKW cold start: strict two-phase zero-macro baseline."""

from __future__ import annotations

from math import isfinite

import numpy as np

from pde_agent_verifier_solver.plugins.gkw.contract import (
    GKW_MACRO_FIELDS,
    gkw_answer_view_from_pairs,
    gkw_point_query_coordinates,
    validate_gkw_macro_query,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Case,
    QueryKind,
)



class _ZeroMacroSolution:
    """Query-blind zero perturbation, defined on the full public domain."""

    def __init__(self, case: Case) -> None:
        self._case = case

    def evaluate(self, query: AnswerQuery) -> AnswerView:
        if not isinstance(query, AnswerQuery):
            raise TypeError("GKW zero-macro projection needs AnswerQuery")
        times, field_indices = gkw_point_query_coordinates(
            self._case,
            query,
            require_native_panel=False,
        )
        zeros = np.zeros(
            (len(times), 2),
            dtype=np.float64,
        )
        return gkw_answer_view_from_pairs(
            query,
            {name: zeros.copy() for name in GKW_MACRO_FIELDS},
            times=times,
            field_indices=field_indices,
        )


def prepare_solution(
    case: Case,
    solve_budget: float | None = None,
) -> _ZeroMacroSolution:
    """Prepare an O(1) zero baseline before scoring points are revealed."""

    if not isinstance(case, Case):
        raise TypeError("GKW zero baseline needs Case")
    _validate_budget(solve_budget)
    return _ZeroMacroSolution(case)


def predict_macros(
    case: Case,
    query: AnswerQuery,
    budget: float | None = None,
) -> AnswerView:
    """Legacy GRID entry point for the same zero-macro baseline."""

    if not isinstance(case, Case) or not isinstance(query, AnswerQuery):
        raise TypeError("GKW zero baseline needs Case and AnswerQuery")
    if query.kind is not QueryKind.GRID:
        raise ValueError("GKW initial hold supports the native GRID query only")
    if set(query.fields) != set(GKW_MACRO_FIELDS):
        raise ValueError("GKW initial hold requires all four macro fields")
    _validate_budget(budget)

    times, indices = validate_gkw_macro_query(case, query)
    values = {
        name: np.zeros(
            (len(times), len(indices), 2),
            dtype=np.float64,
        )
        for name in GKW_MACRO_FIELDS
    }
    return gkw_answer_view_from_pairs(
        query,
        values,
        times=times,
        field_indices=indices,
    )



def _validate_budget(value: float | None) -> None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError("candidate budget must be finite and positive")
