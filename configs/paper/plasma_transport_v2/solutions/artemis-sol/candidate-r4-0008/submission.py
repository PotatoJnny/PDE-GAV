"""Dual-grid Artemis macro solver.

The selected Nx=128 solver supplies the high moments, while an independent
Nx=192 solve supplies only density and electric field. Both continuous
representations are prepared before scoring coordinates are visible.
"""

from __future__ import annotations

from math import isfinite

from pde_agent_verifier_solver.sdk import AnswerQuery, AnswerView, Case
from pde_agent_verifier_solver.sdk.solve_budget import candidate_solve_clock

import solver128
import solver192


class _DualHandle:
    def __init__(self, high_moment_handle, charge_handle):
        self._high = high_moment_handle
        self._charge = charge_handle
        self._closed = False

    def evaluate(self, query: AnswerQuery) -> AnswerView:
        if self._closed:
            raise RuntimeError("dual-grid Artemis handle is closed")
        high = self._high.evaluate(query)
        charge = self._charge.evaluate(query)
        blended_m2 = tuple(
            0.5 * (left + right)
            for left, right in zip(high.fields["M2"], charge.fields["M2"])
        )
        fields = {
            "M0": charge.fields["M0"],
            "M1": high.fields["M1"],
            "M2": blended_m2,
            "M3": high.fields["M3"],
            "E": charge.fields["E"],
        }
        return AnswerView(kind=query.kind, fields=fields, points=query.points,
                          shape=(len(query.points),))

    def close(self):
        if not self._closed:
            self._closed = True
            self._high.close()
            self._charge.close()


def prepare_solution(case: Case, solve_budget=None) -> _DualHandle:
    began = candidate_solve_clock()
    if solve_budget is not None and (
        isinstance(solve_budget, bool) or not isinstance(solve_budget, (int, float))
        or not isfinite(float(solve_budget)) or solve_budget <= 0
    ):
        raise ValueError("dual-grid solve_budget must be finite and positive")

    high = solver128.prepare_solution(case, solve_budget)
    if solve_budget is None:
        remaining = None
    else:
        remaining = float(solve_budget) - (candidate_solve_clock() - began)
        if remaining <= 20.0:
            high.close()
            raise TimeoutError("insufficient shared budget for Nx=192 charge solve")
    try:
        charge = solver192.prepare_solution(case, remaining)
    except Exception:
        high.close()
        raise
    return _DualHandle(high, charge)
