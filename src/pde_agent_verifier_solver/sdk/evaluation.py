"""Task-owned scientific evaluation contracts.

An EvaluationProgram defines finite queries, per-Case metrics, and aggregation.
Development Evaluator and Hidden Verifier provide access control and execute the
same program; they do not define PDE-specific errors.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from numbers import Real
from statistics import fmean
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

from .models import (
    AnswerQuery,
    AnswerView,
    Artifact,
    ArtifactEvaluation,
    Case,
    CaseEvaluation,
)


class EvaluationPhase(str, Enum):
    DEVELOPMENT = "development"
    OFFICIAL = "official"
    TERMINAL = "terminal"


class DevelopmentCoordinateMode(str, Enum):
    """Who chooses the finite coordinates used for Development evaluation."""

    PRIVATE_RANDOM_POINTS = "private_random_points"
    USER_FIXED_QUERY = "user_fixed_query"
    AGENT_SELECTED_QUERY = "agent_selected_query"


@dataclass(frozen=True)
class _FrozenMetricMapping(Mapping[str, float]):
    """A small pickle-safe immutable mapping for process-boundary results."""

    _items: tuple[tuple[str, float], ...]

    def __getitem__(self, key: str) -> float:
        for name, value in self._items:
            if name == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (name for name, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)


@dataclass(frozen=True)
class CandidateEvaluationOutput:
    """Finite Candidate output produced before private truth is available.

    The deliberately narrow value can cross a process boundary without
    carrying a ``TruthProgram`` or a materialized private ``AnswerView`` into
    Candidate execution. Metrics here describe execution only; scientific
    errors are added later by the Task-owned scorer.
    """

    prediction: AnswerView
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, AnswerView):
            raise TypeError("candidate evaluation output needs an AnswerView")
        prepared: dict[str, float] = {}
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("candidate execution metric names must be nonempty")
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(
                    f"candidate execution metric {name!r} must be a real number"
                )
            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError(
                    f"candidate execution metric {name!r} must be finite"
                )
            prepared[name] = numeric
        object.__setattr__(
            self,
            "metrics",
            _FrozenMetricMapping(tuple(prepared.items())),
        )


@dataclass(frozen=True)
class EvaluationPlan:
    """One immutable finite evaluation view for one Case."""

    phase: EvaluationPhase
    query: AnswerQuery
    seed: int = 0
    settings: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.phase, EvaluationPhase):
            raise TypeError("evaluation phase must be an EvaluationPhase")
        if not isinstance(self.query, AnswerQuery):
            raise TypeError("evaluation plan needs an AnswerQuery")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("evaluation seed must be a nonnegative integer")
        object.__setattr__(
            self,
            "query",
            AnswerQuery(
                kind=self.query.kind,
                fields=tuple(self.query.fields),
                resolution=MappingProxyType(dict(self.query.resolution)),
                points=tuple(
                    MappingProxyType(dict(point))
                    for point in self.query.points
                ),
            ),
        )
        object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))


class EvaluationProgram(Protocol):
    """Task-owned query, scoring, and aggregation policy."""

    @property
    def program_id(self) -> str: ...

    def make_plan(
        self,
        case: Case,
        phase: EvaluationPhase,
        settings: Mapping[str, object],
        seed: int,
        request: AnswerQuery | None = None,
    ) -> EvaluationPlan: ...

    def evaluate(
        self,
        artifact: Artifact,
        case: Case,
        truth: AnswerView,
        plan: EvaluationPlan,
    ) -> CaseEvaluation: ...

    def aggregate(
        self,
        artifact: Artifact,
        case_results: Sequence[CaseEvaluation],
        plans: Sequence[EvaluationPlan],
    ) -> ArtifactEvaluation: ...


class LegacyTaskEvaluationProgram:
    """Temporary adapter for TaskProgram implementations not yet migrated."""

    def __init__(self, task: object, default_query: AnswerQuery) -> None:
        if not isinstance(default_query, AnswerQuery):
            raise TypeError("legacy Task evaluation needs a default query")
        if not callable(getattr(task, "evaluate", None)):
            raise TypeError("Task needs create_evaluation() or legacy evaluate()")
        self._task = task
        self._default_query = default_query

    @property
    def program_id(self) -> str:
        info = getattr(self._task, "info", None)
        task_id = getattr(info, "program_id", type(self._task).__name__)
        return f"{task_id}:legacy-evaluation"

    def make_plan(
        self,
        case: Case,
        phase: EvaluationPhase,
        settings: Mapping[str, object],
        seed: int,
        request: AnswerQuery | None = None,
    ) -> EvaluationPlan:
        del case
        return EvaluationPlan(
            phase=phase,
            query=request or self._default_query,
            seed=seed,
            settings=settings,
        )

    def evaluate(
        self,
        artifact: Artifact,
        case: Case,
        truth: AnswerView,
        plan: EvaluationPlan,
    ) -> CaseEvaluation:
        return self._task.evaluate(artifact, case, truth, plan.query)

    def aggregate(
        self,
        artifact: Artifact,
        case_results: Sequence[CaseEvaluation],
        plans: Sequence[EvaluationPlan],
    ) -> ArtifactEvaluation:
        return aggregate_case_evaluations(artifact, case_results, plans)


def resolve_evaluation_program(
    task_or_evaluation: object,
    settings: Mapping[str, object],
) -> EvaluationProgram:
    """Return a Task's EvaluationProgram, with an explicit legacy fallback."""

    if _is_evaluation_program(task_or_evaluation):
        return task_or_evaluation
    default_query = settings.get("default_query")
    if not isinstance(default_query, AnswerQuery):
        raise TypeError("evaluation settings need an AnswerQuery default_query")
    factory = getattr(task_or_evaluation, "create_evaluation", None)
    if callable(factory):
        evaluation = factory(settings)
        if not _is_evaluation_program(evaluation):
            raise TypeError("TaskProgram.create_evaluation() returned an invalid program")
        return evaluation
    return LegacyTaskEvaluationProgram(task_or_evaluation, default_query)


def aggregate_case_evaluations(
    artifact: Artifact,
    case_results: Sequence[CaseEvaluation],
    plans: Sequence[EvaluationPlan],
) -> ArtifactEvaluation:
    """Default equal-Case aggregation; a Task may provide another rule."""

    results = tuple(case_results)
    prepared_plans = tuple(plans)
    if not results:
        raise ValueError("cannot aggregate an empty evaluation")
    if len(results) != len(prepared_plans):
        raise ValueError("evaluation results and plans must have equal length")
    names = sorted({name for result in results for name in result.metrics})
    metrics = {
        name: fmean(result.metrics[name] for result in results if name in result.metrics)
        for name in names
    }
    return ArtifactEvaluation(
        artifact=artifact,
        primary_error=fmean(result.primary_error for result in results),
        metrics=metrics,
        cases=results,
    )


def _is_evaluation_program(value: object) -> bool:
    return all(
        callable(getattr(value, name, None))
        for name in ("make_plan", "evaluate", "aggregate")
    ) and isinstance(getattr(value, "program_id", None), str)


__all__ = [
    "CandidateEvaluationOutput",
    "DevelopmentCoordinateMode",
    "EvaluationPhase",
    "EvaluationPlan",
    "EvaluationProgram",
    "LegacyTaskEvaluationProgram",
    "aggregate_case_evaluations",
    "resolve_evaluation_program",
]
