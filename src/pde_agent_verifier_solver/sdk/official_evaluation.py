"""Small contracts for the hidden Cases used after each Agent round.

The shared framework decides *how many* Cases of each kind are required.  The
Dataset and Generator programs still own the scientific construction of those
Cases.  Keeping the counts separate is important: a user's promotion rule may
care about average Base performance while still asking the panel to report
Guided and perturbed performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from .promotion import PromotionMetricSummary


@dataclass(frozen=True)
class HiddenSuiteSpec:
    """Requested Case counts for one fresh controller-only evaluation suite."""

    original: int = 0
    base: int = 0
    guided: int = 0
    base_perturbed: int = 0
    guided_perturbed: int = 0

    def __post_init__(self) -> None:
        values = {
            "original": self.original,
            "base": self.base,
            "guided": self.guided,
            "base_perturbed": self.base_perturbed,
            "guided_perturbed": self.guided_perturbed,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"hidden {name} count must be a nonnegative integer")
        if not any(values.values()):
            raise ValueError("a hidden suite needs at least one Case")
        # Version 1 creates at most one perturbed child for each clean parent.
        # The PDE Generator owns the perturbation itself.
        if self.base_perturbed > self.base:
            raise ValueError("hidden base_perturbed cannot exceed hidden base")
        if self.guided_perturbed > self.guided:
            raise ValueError("hidden guided_perturbed cannot exceed hidden guided")

    @property
    def case_count(self) -> int:
        return (
            self.original
            + self.base
            + self.guided
            + self.base_perturbed
            + self.guided_perturbed
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "original": self.original,
            "base": self.base,
            "guided": self.guided,
            "base_perturbed": self.base_perturbed,
            "guided_perturbed": self.guided_perturbed,
        }


@dataclass(frozen=True)
class OfficialEvaluationSpec:
    """One fresh, non-cumulative official suite for each round.

    ``terminal`` is retained only as an input compatibility bridge. New
    Campaigns declare final evaluation once, at Campaign scope, through
    :class:`TerminalEvaluationSpec`.
    """

    round: HiddenSuiteSpec
    terminal: HiddenSuiteSpec | None = None


class TerminalGuidanceSource(str, Enum):
    """Generator state used to seal one terminal cohort."""

    BASE = "base"
    ROUND = "round_guided"
    FINAL = "final_guided"


@dataclass(frozen=True)
class TerminalCohortSpec:
    """A named terminal cohort generated at an explicitly defined time."""

    cohort_id: str
    suite: HiddenSuiteSpec
    guidance_source: TerminalGuidanceSource = TerminalGuidanceSource.BASE
    round_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cohort_id, str) or not self.cohort_id.strip():
            raise ValueError("terminal cohort_id cannot be empty")
        if self.cohort_id == "overall":
            raise ValueError("terminal cohort_id 'overall' is reserved for derived evaluation")
        if not isinstance(self.guidance_source, TerminalGuidanceSource):
            raise TypeError("terminal guidance_source must be TerminalGuidanceSource")
        if self.guidance_source is TerminalGuidanceSource.ROUND:
            if (
                isinstance(self.round_index, bool)
                or not isinstance(self.round_index, int)
                or self.round_index < 1
            ):
                raise ValueError("round-guided terminal cohort needs a positive round_index")
        elif self.round_index is not None:
            raise ValueError("only round-guided terminal cohorts accept round_index")

        guided_count = self.suite.guided + self.suite.guided_perturbed
        base_count = (
            self.suite.original
            + self.suite.base
            + self.suite.base_perturbed
        )
        if self.guidance_source is TerminalGuidanceSource.BASE:
            if guided_count:
                raise ValueError("base terminal cohort cannot request guided Cases")
        else:
            if base_count:
                raise ValueError(
                    "guided terminal cohort must contain only guided distributions"
                )
            if not guided_count:
                raise ValueError("guided terminal cohort needs guided Cases")


@dataclass(frozen=True)
class TerminalMetricSpec:
    """Diagnostic used to identify the best terminal round."""

    cohort_id: str
    distribution: str
    statistic: str = "mean"

    def __post_init__(self) -> None:
        if not self.cohort_id.strip():
            raise ValueError("terminal metric needs a cohort_id")
        if not self.distribution.strip():
            raise ValueError("terminal metric needs a distribution")
        if (self.cohort_id == "overall") != (self.distribution == "overall"):
            raise ValueError(
                "overall terminal ranking requires both cohort_id and distribution to be 'overall'"
            )
        if self.statistic not in {"mean", "cvar", "maximum"}:
            raise ValueError("terminal statistic must be mean, cvar, or maximum")


@dataclass(frozen=True)
class TerminalEvaluationSpec:
    """Campaign-level final exam evaluated for every selected ``S_1..S_R``.

    Cohorts are never mixed implicitly. The complete cross-product of round
    selections and named cohorts is reported. ``ranking`` is diagnostic only:
    it may identify ``best_terminal_round`` but can never replace delivered
    program ``S_R``. Explicit ``overall/overall`` ranking combines all real
    Cases across the cohorts, retaining each cohort's separate report.
    """

    cohorts: tuple[TerminalCohortSpec, ...]
    ranking: TerminalMetricSpec

    def __post_init__(self) -> None:
        cohorts = tuple(self.cohorts)
        if not cohorts:
            raise ValueError("terminal evaluation needs at least one cohort")
        ids = tuple(cohort.cohort_id for cohort in cohorts)
        if len(ids) != len(set(ids)):
            raise ValueError("terminal cohort ids must be unique")
        if self.ranking.cohort_id != "overall" and self.ranking.cohort_id not in ids:
            raise ValueError("terminal ranking references an unknown cohort")
        object.__setattr__(self, "cohorts", cohorts)


@dataclass(frozen=True)
class TerminalCandidateReport:
    """One round selection evaluated across arbitrary named terminal cohorts."""

    candidate_id: str
    candidate_digest: str
    technically_eligible: bool
    distributions: Mapping[str, PromotionMetricSummary]
    guardrails: Mapping[str, bool] = field(default_factory=dict)
    failures: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("terminal candidate_id cannot be empty")
        if not isinstance(self.candidate_digest, str) or not self.candidate_digest.strip():
            raise ValueError("terminal candidate_digest cannot be empty")
        if not isinstance(self.technically_eligible, bool):
            raise TypeError("terminal technically_eligible must be bool")
        distributions = dict(self.distributions)
        if any(not isinstance(name, str) or not name.strip() for name in distributions):
            raise ValueError("terminal distribution names cannot be empty")
        if any(
            not isinstance(value, PromotionMetricSummary)
            for value in distributions.values()
        ):
            raise TypeError("terminal distributions need metric summaries")
        guardrails = dict(self.guardrails)
        if any(
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(value, bool)
            for name, value in guardrails.items()
        ):
            raise TypeError("terminal guardrails must map names to bool values")
        failures = tuple(self.failures)
        if any(not isinstance(value, str) or not value for value in failures):
            raise ValueError("terminal failures must be nonempty strings")
        object.__setattr__(self, "distributions", MappingProxyType(distributions))
        object.__setattr__(self, "guardrails", MappingProxyType(guardrails))
        object.__setattr__(self, "failures", failures)


def terminal_from_legacy_suite(
    suite: HiddenSuiteSpec,
    *,
    score_distribution: str,
) -> TerminalEvaluationSpec:
    """Translate the former single terminal suite without changing its meaning."""

    cohort_id = "base_model"
    return TerminalEvaluationSpec(
        cohorts=(TerminalCohortSpec(cohort_id=cohort_id, suite=suite),),
        ranking=TerminalMetricSpec(
            cohort_id="overall" if score_distribution == "overall" else cohort_id,
            distribution=score_distribution,
        ),
    )


__all__ = [
    "HiddenSuiteSpec",
    "OfficialEvaluationSpec",
    "TerminalCohortSpec",
    "TerminalCandidateReport",
    "TerminalEvaluationSpec",
    "TerminalGuidanceSource",
    "TerminalMetricSpec",
    "terminal_from_legacy_suite",
]
