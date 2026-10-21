"""Contract for choosing the program carried to the next round.

Every candidate is evaluated and reported before this contract is called.
``PromotionProgram`` does not decide what the panel hides or shows. It only
turns a multi-distribution report into one explicit next-round choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence


PROMOTION_DISTRIBUTIONS = (
    "original",
    "base",
    "guided",
    "base_perturbed",
    "guided_perturbed",
    # Derived from all real Cases; never a Generator sampling source.
    "overall",
)


@dataclass(frozen=True)
class PromotionMetricSummary:
    """One candidate's aggregate error on a source or explicit overall mixture."""

    case_count: int
    mean: float
    cvar: float
    maximum: float
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.case_count, bool)
            or not isinstance(self.case_count, int)
            or self.case_count < 1
        ):
            raise ValueError("promotion metric case_count must be positive")
        values = (float(self.mean), float(self.cvar), float(self.maximum))
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError("promotion metrics must be finite and nonnegative")
        object.__setattr__(self, "mean", values[0])
        object.__setattr__(self, "cvar", values[1])
        object.__setattr__(self, "maximum", values[2])
        if not isinstance(self.metrics, Mapping):
            raise TypeError("promotion named metrics must be a mapping")
        metrics: dict[str, float] = {}
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("promotion metric names cannot be empty")
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("promotion metric values must be numeric")
            number = float(value)
            if not isfinite(number):
                raise ValueError("promotion metric values must be finite")
            metrics[name] = number
        object.__setattr__(self, "metrics", MappingProxyType(metrics))


@dataclass(frozen=True)
class PromotionCandidateReport:
    """Everything a promotion rule may use for one candidate."""

    candidate_id: str
    candidate_digest: str
    technically_eligible: bool
    distributions: Mapping[str, PromotionMetricSummary]
    guardrails: Mapping[str, bool] = field(default_factory=dict)
    failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("promotion candidate_id cannot be empty")
        if not isinstance(self.candidate_digest, str) or not self.candidate_digest.strip():
            raise ValueError("promotion candidate_digest cannot be empty")
        if not isinstance(self.technically_eligible, bool):
            raise TypeError("promotion technically_eligible must be bool")
        distributions = dict(self.distributions)
        if any(name not in PROMOTION_DISTRIBUTIONS for name in distributions):
            raise ValueError("promotion report contains an unknown distribution")
        if any(
            not isinstance(value, PromotionMetricSummary)
            for value in distributions.values()
        ):
            raise TypeError("promotion distributions need metric summaries")
        guardrails = dict(self.guardrails)
        if any(
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(value, bool)
            for name, value in guardrails.items()
        ):
            raise TypeError("promotion guardrails must map names to bool values")
        failures = tuple(self.failures)
        if any(not isinstance(value, str) or not value for value in failures):
            raise ValueError("promotion failures must be nonempty strings")
        object.__setattr__(self, "distributions", MappingProxyType(distributions))
        object.__setattr__(self, "guardrails", MappingProxyType(guardrails))
        object.__setattr__(self, "failures", failures)


@dataclass(frozen=True)
class PromotionContext:
    """Round information and user settings passed to a PromotionProgram."""

    round_index: int
    settings: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.round_index, bool)
            or not isinstance(self.round_index, int)
            or self.round_index < 1
        ):
            raise ValueError("promotion round_index must be positive")
        object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))


@dataclass(frozen=True)
class PromotionRankingEntry:
    """One row shown in the round's candidate decision table."""

    candidate_digest: str
    rank: int | None
    eligible: bool
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_digest, str) or not self.candidate_digest:
            raise ValueError("promotion ranking digest cannot be empty")
        if self.rank is not None and (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 1
        ):
            raise ValueError("promotion ranking rank must be positive or None")
        if not isinstance(self.eligible, bool):
            raise TypeError("promotion ranking eligible must be bool")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("promotion ranking reason cannot be empty")


@dataclass(frozen=True)
class PromotionDecision:
    """The one program inherited by the next round plus the complete ranking.

    ``promoted`` is true for the initial selection when no incumbent exists.
    """

    selected_digest: str
    promoted: bool
    reason: str
    ranking: tuple[PromotionRankingEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.selected_digest, str) or not self.selected_digest:
            raise ValueError("promotion selected_digest cannot be empty")
        if not isinstance(self.promoted, bool):
            raise TypeError("promotion promoted must be bool")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("promotion reason cannot be empty")
        ranking = tuple(self.ranking)
        if not ranking:
            raise ValueError("promotion decision needs a complete ranking")
        digests = tuple(row.candidate_digest for row in ranking)
        if len(digests) != len(set(digests)):
            raise ValueError("promotion ranking contains duplicate candidates")
        if self.selected_digest not in digests:
            raise ValueError("promotion selected candidate is absent from ranking")
        object.__setattr__(self, "ranking", ranking)


class PromotionProgram(Protocol):
    """Select one next-round program from complete official reports.

    ``incumbent_digest`` is ``None`` for the first selection without ``S0``.
    """

    @property
    def program_id(self) -> str: ...

    def select(
        self,
        incumbent_digest: str | None,
        reports: Sequence[PromotionCandidateReport],
        context: PromotionContext,
    ) -> PromotionDecision: ...


__all__ = [
    "PROMOTION_DISTRIBUTIONS",
    "PromotionCandidateReport",
    "PromotionContext",
    "PromotionDecision",
    "PromotionMetricSummary",
    "PromotionProgram",
    "PromotionRankingEntry",
]
