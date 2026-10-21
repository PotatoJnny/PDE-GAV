"""Independent controller-side matching of macroscopic quantities.

Data observations may be present in an Agent-visible :class:`Case` and may
also be used to train a Generator.  They are not macro targets.  A macro
target is prepared only from a finite controller-side truth view, then matched
against a candidate prediction by a PDE-owned implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, prod
from typing import Mapping, Protocol

from .models import (
    AnswerQuery,
    AnswerView,
    Case,
    CaseEvaluation,
    CaseRecord,
    PanelCard,
    ProgramInfo,
)


@dataclass(frozen=True)
class MacroTarget:
    """Finite, controller-private macroscopic target for one case and query."""

    case_id: str
    contract_id: str
    query: AnswerQuery
    values: AnswerView
    sample_weights: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.case_id.strip() or not self.contract_id.strip():
            raise ValueError("macro target ids cannot be empty")
        if self.query.kind is not self.values.kind:
            raise ValueError("macro target query and values use different query kinds")
        if set(self.query.fields) != set(self.values.fields):
            raise ValueError("macro target fields must match the query exactly")

        lengths = {len(values) for values in self.values.fields.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
            raise ValueError("macro target fields need one common nonzero sample count")
        sample_count = next(iter(lengths))
        if self.values.shape and prod(self.values.shape) != sample_count:
            raise ValueError("macro target shape does not match its flattened fields")

        for values in self.values.fields.values():
            if any(not isfinite(float(value)) for value in values):
                raise ValueError("macro target values must be finite")
        if self.sample_weights:
            if len(self.sample_weights) != sample_count:
                raise ValueError("macro target weights must align with every sample")
            if any(
                not isfinite(float(value)) or float(value) < 0.0
                for value in self.sample_weights
            ):
                raise ValueError("macro target weights must be finite and nonnegative")
            if not any(float(value) > 0.0 for value in self.sample_weights):
                raise ValueError("macro target weights cannot all be zero")


@dataclass(frozen=True)
class MacroMatchResult:
    """Numerical match plus an optional PDE-owned acceptance decision."""

    case_id: str
    contract_id: str
    primary_error: float
    metrics: Mapping[str, float] = field(default_factory=dict)
    passed: bool | None = None
    failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.case_id.strip() or not self.contract_id.strip():
            raise ValueError("macro match ids cannot be empty")
        if not isfinite(float(self.primary_error)) or self.primary_error < 0.0:
            raise ValueError("macro match primary error must be finite and nonnegative")
        if any(not isfinite(float(value)) for value in self.metrics.values()):
            raise ValueError("macro match metrics must be finite")
        if self.passed is not None and not isinstance(self.passed, bool):
            raise ValueError("macro match passed must be bool or None")
        if self.passed is True and self.failures:
            raise ValueError("a passing macro match cannot contain failures")


class MacroMatcher(Protocol):
    """PDE-owned macro projection/metric contract used outside data fitting."""

    @property
    def info(self) -> ProgramInfo: ...

    @property
    def contract_id(self) -> str: ...

    @property
    def fields(self) -> tuple[str, ...]: ...

    def prepare_target(
        self,
        case: Case,
        truth: AnswerView,
        query: AnswerQuery,
    ) -> MacroTarget: ...

    def match(
        self,
        case: Case,
        prediction: AnswerView,
        target: MacroTarget,
    ) -> MacroMatchResult: ...

    def result_card(self, result: MacroMatchResult) -> PanelCard: ...


def materialize_macro_target(
    matcher: MacroMatcher,
    record: CaseRecord,
    query: AnswerQuery,
) -> MacroTarget:
    """Materialize one target without retaining the underlying TruthProgram."""

    truth = record.truth.answer(query)
    target = matcher.prepare_target(record.case, truth, query)
    _validate_prepared_target(matcher, record.case, query, target)
    return target


def _validate_prepared_target(
    matcher: MacroMatcher,
    case: Case,
    query: AnswerQuery,
    target: MacroTarget,
) -> None:
    if target.case_id != case.case_id:
        raise ValueError("macro matcher prepared a target for a different case")
    if target.contract_id != matcher.contract_id:
        raise ValueError("macro matcher prepared a target for a different contract")
    if target.query != query:
        raise ValueError("macro matcher prepared a target for a different query")


def evaluate_macro_prediction(
    *,
    matcher: MacroMatcher,
    case: Case,
    prediction: AnswerView,
    truth: AnswerView,
    query: AnswerQuery,
) -> CaseEvaluation:
    """Match a Task prediction while keeping the target controller-private."""

    target = matcher.prepare_target(case, truth, query)
    _validate_prepared_target(matcher, case, query, target)

    result = matcher.match(case, prediction, target)
    if result.case_id != case.case_id:
        raise ValueError("macro matcher returned a result for a different case")
    if result.contract_id != matcher.contract_id:
        raise ValueError("macro matcher returned a result for a different contract")

    metrics = dict(result.metrics)
    metrics["macro_match_error"] = result.primary_error
    metrics.pop("macro_match_passed", None)
    if result.passed is not None:
        metrics["macro_match_passed"] = 1.0 if result.passed else 0.0
    return CaseEvaluation(
        case_id=case.case_id,
        primary_error=result.primary_error,
        metrics=metrics,
        prediction=prediction,
        macro_match=result,
    )
