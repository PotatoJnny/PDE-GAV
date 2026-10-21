"""Candidate shapes for the built-in Task categories.

These are intentionally structural protocols. An Artifact may use any internal
method; the selected Task decides which customer-facing calls it requires.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from .models import AnswerQuery, AnswerView, Case


class FullStateSolverProgram(Protocol):
    def solve(self, case: Case, query: AnswerQuery, budget: float | None = None) -> AnswerView: ...


class ObservableModelProgram(Protocol):
    """Predict requested observables; internals need not be low dimensional."""

    def predict_observables(
        self,
        case: Case,
        query: AnswerQuery,
        budget: float | None = None,
    ) -> AnswerView: ...


class ReducedDynamicsProgram(Protocol):
    """A true ROM with explicit reduced state, evolution, and observation."""

    def initialize(self, case: Case) -> object: ...

    def advance(
        self,
        state: object,
        query_times: Sequence[float],
        budget: float | None = None,
    ) -> object: ...

    def observe(self, reduced_trajectory: object, query: AnswerQuery) -> AnswerView: ...


class StatisticsProgram(Protocol):
    def statistics(
        self,
        case: Case,
        requested: Sequence[str],
        settings: Mapping[str, object],
    ) -> Mapping[str, object]: ...
