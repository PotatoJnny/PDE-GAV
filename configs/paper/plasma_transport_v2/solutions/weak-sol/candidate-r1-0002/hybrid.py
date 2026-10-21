"""Calibrated persistence model with pressure-only free-stream correction."""
from __future__ import annotations

from math import isfinite

import numpy as np

from pde_agent_verifier_solver.sdk import AnswerQuery, AnswerView, QueryKind

import calibrated
import freestream


PRESSURE = {"pressure_xx", "pressure_xy", "pressure_yy"}


class Prepared:
    def __init__(self, kinetic, calibrated_handle):
        self.kinetic = kinetic
        self.calibrated = calibrated_handle
        self.closed = False

    def evaluate(self, query):
        if self.closed:
            raise RuntimeError("prepared handle is closed")
        if not isinstance(query, AnswerQuery) or query.kind is not QueryKind.POINTS:
            raise ValueError("POINTS query required")
        empirical = self.calibrated.evaluate(query)
        ballistic = self.kinetic.evaluate(query)
        fields = {name: (ballistic.fields[name] if name in PRESSURE else empirical.fields[name])
                  for name in query.fields}
        return AnswerView(kind=query.kind, fields=fields, points=query.points, shape=(len(query.points),))

    def close(self):
        if not self.closed:
            # Both child handles share the same base; close it exactly once.
            self.kinetic.close()
            self.calibrated.closed = True
        self.closed = True


def prepare_solution(case, solve_budget=None):
    if solve_budget is not None and (isinstance(solve_budget, bool)
            or not isinstance(solve_budget, (int, float)) or not isfinite(solve_budget) or solve_budget <= 0):
        raise ValueError("solve_budget must be finite and positive")
    kinetic = freestream.prepare_solution(case, solve_budget)
    q = case.public_data["question"]
    empirical = calibrated.Prepared(kinetic.base, float(q["temperature"]), q["time_domain"])
    return Prepared(kinetic, empirical)
