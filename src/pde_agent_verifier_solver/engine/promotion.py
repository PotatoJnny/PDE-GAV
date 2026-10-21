"""Default and user-supplied candidate promotion programs."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from math import isfinite
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from pde_agent_verifier_solver.sdk.promotion import (
    PROMOTION_DISTRIBUTIONS,
    PromotionCandidateReport,
    PromotionContext,
    PromotionDecision,
    PromotionMetricSummary,
    PromotionRankingEntry,
)


DEFAULT_PROMOTION_PRESET = "base_improvement_with_robustness_guardrails"
MINIMIZE_METRIC_PROMOTION_PRESET = "minimize_metric"
_PROMOTION_STATISTICS = ("mean", "cvar", "maximum")


@dataclass(frozen=True)
class PromotionMetricReference:
    """One user-selected value in an official hidden evaluation report."""

    distribution: str
    statistic: str

    def __post_init__(self) -> None:
        if self.distribution not in PROMOTION_DISTRIBUTIONS:
            raise ValueError(f"unknown promotion distribution: {self.distribution!r}")
        if self.statistic not in _PROMOTION_STATISTICS:
            raise ValueError("promotion statistic must be mean, cvar, or maximum")

    @property
    def label(self) -> str:
        return f"{self.distribution}.{self.statistic}"

    def value(self, report: PromotionCandidateReport) -> float:
        return float(getattr(report.distributions[self.distribution], self.statistic))


@dataclass(frozen=True)
class MinimizeMetricPromotion:
    """Choose the technically valid candidate that minimizes selected metrics.

    ``primary`` is the only metric used unless the user explicitly supplies tie
    breakers or guardrails. A Guided or perturbed regression therefore does not
    silently veto a Base-only rule.
    """

    primary: PromotionMetricReference = PromotionMetricReference("base", "mean")
    require_improvement: bool = True
    minimum_improvement: float = 0.0
    tie_breakers: tuple[PromotionMetricReference, ...] = ()
    guardrails: tuple[PromotionMetricReference, ...] = ()
    settings: Mapping[str, object] = field(default_factory=dict)
    program_id: str = MINIMIZE_METRIC_PROMOTION_PRESET

    def __post_init__(self) -> None:
        if not isinstance(self.primary, PromotionMetricReference):
            raise TypeError("promotion primary must be a metric reference")
        if not isinstance(self.require_improvement, bool):
            raise TypeError("promotion require_improvement must be bool")
        if (
            isinstance(self.minimum_improvement, bool)
            or not isinstance(self.minimum_improvement, Real)
        ):
            raise TypeError("promotion minimum_improvement must be numeric")
        improvement = float(self.minimum_improvement)
        if not isfinite(improvement) or improvement < 0.0:
            raise ValueError("promotion minimum_improvement must be nonnegative")
        if improvement > 0.0 and not self.require_improvement:
            raise ValueError(
                "positive minimum_improvement requires require_improvement=true"
            )
        tie_breakers = tuple(self.tie_breakers)
        guardrails = tuple(self.guardrails)
        if any(
            not isinstance(value, PromotionMetricReference)
            for value in tie_breakers + guardrails
        ):
            raise TypeError("promotion metric rules must be metric references")
        ranking_metrics = (self.primary,) + tie_breakers
        if len(ranking_metrics) != len(set(ranking_metrics)):
            raise ValueError("promotion ranking metrics cannot repeat")
        if len(guardrails) != len(set(guardrails)):
            raise ValueError("promotion guardrails cannot repeat")
        object.__setattr__(self, "minimum_improvement", improvement)
        object.__setattr__(self, "tie_breakers", tie_breakers)
        object.__setattr__(self, "guardrails", guardrails)
        object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))

    def select(
        self,
        incumbent_digest: str | None,
        reports: Sequence[PromotionCandidateReport],
        context: PromotionContext,
    ) -> PromotionDecision:
        del context
        rows = tuple(reports)
        by_digest = {row.candidate_digest: row for row in rows}
        if len(by_digest) != len(rows):
            raise ValueError("promotion reports contain duplicate candidates")
        incumbent: PromotionCandidateReport | None = None
        if incumbent_digest is not None:
            if incumbent_digest not in by_digest:
                raise ValueError("promotion reports do not contain the incumbent")
            incumbent = by_digest[incumbent_digest]
            if not incumbent.technically_eligible:
                raise ValueError("minimize_metric incumbent is technically ineligible")
            missing = self._missing_metric(incumbent)
            if missing is not None:
                raise ValueError(
                    f"minimize_metric incumbent is missing {missing} metrics"
                )

        eligible: list[PromotionCandidateReport] = []
        ineligible_reason: dict[str, str] = {}
        for row in rows:
            reason = self._ineligible_reason(row, incumbent)
            if reason is None:
                eligible.append(row)
            else:
                ineligible_reason[row.candidate_digest] = reason

        if not eligible:
            raise ValueError("minimize_metric has no eligible initial candidate")
        ordered = sorted(
            eligible,
            key=lambda row: self._ranking_key(row, incumbent_digest),
        )
        leader = ordered[0]
        improvement_required_but_missing = False
        if (
            incumbent is not None
            and self.require_improvement
            and leader.candidate_digest != incumbent_digest
            and not self._meets_required_improvement(leader, incumbent)
        ):
            leader = incumbent
            improvement_required_but_missing = True

        ranking_by_digest = {
            row.candidate_digest: index + 1 for index, row in enumerate(ordered)
        }
        ranking = tuple(
            PromotionRankingEntry(
                candidate_digest=row.candidate_digest,
                rank=ranking_by_digest.get(row.candidate_digest),
                eligible=row.candidate_digest in ranking_by_digest,
                reason=(
                    "eligible_metric_ranking"
                    if row.candidate_digest in ranking_by_digest
                    else ineligible_reason[row.candidate_digest]
                ),
            )
            for row in rows
        )
        promoted = (
            incumbent_digest is None
            or leader.candidate_digest != incumbent_digest
        )
        if incumbent_digest is None:
            reason = "initial_candidate_selected_by_configured_metrics"
        elif promoted:
            reason = "configured_metric_minimized"
        elif improvement_required_but_missing:
            reason = "incumbent_kept_minimum_improvement_not_met"
        else:
            reason = "incumbent_kept_by_configured_metrics"
        return PromotionDecision(
            selected_digest=leader.candidate_digest,
            promoted=promoted,
            reason=reason,
            ranking=ranking,
        )

    def _ineligible_reason(
        self,
        row: PromotionCandidateReport,
        incumbent: PromotionCandidateReport | None,
    ) -> str | None:
        if not row.technically_eligible:
            return "technical_evaluation_failed"
        missing = self._missing_metric(row)
        if missing is not None:
            return f"{missing}_evaluation_missing"
        if incumbent is None:
            return None
        if row.candidate_digest == incumbent.candidate_digest:
            return None
        for metric in self.guardrails:
            if metric.value(row) > metric.value(incumbent):
                return f"{metric.label}_non_regression_failed"
        return None

    def _missing_metric(self, row: PromotionCandidateReport) -> str | None:
        metrics = (self.primary,) + self.tie_breakers + self.guardrails
        for metric in metrics:
            if metric.distribution not in row.distributions:
                return metric.distribution
        return None

    def _ranking_key(
        self,
        row: PromotionCandidateReport,
        incumbent_digest: str | None,
    ) -> tuple[object, ...]:
        metrics = (self.primary,) + self.tie_breakers
        # Preserve the incumbent when every configured value ties. The digest
        # only makes challenger ordering stable.
        return tuple(metric.value(row) for metric in metrics) + (
            (
                0
                if incumbent_digest is not None
                and row.candidate_digest == incumbent_digest
                else 1
            ),
            row.candidate_digest,
        )

    def _meets_required_improvement(
        self,
        candidate: PromotionCandidateReport,
        incumbent: PromotionCandidateReport,
    ) -> bool:
        improvement = self.primary.value(incumbent) - self.primary.value(candidate)
        if self.minimum_improvement == 0.0:
            return improvement > 0.0
        return improvement >= self.minimum_improvement


@dataclass(frozen=True)
class BaseImprovementWithRobustnessGuardrails:
    """Rank Base performance after explicit, separately reported guardrails."""

    relative_margin: float = 0.05
    guardrail_distributions: tuple[str, ...] = (
        "guided",
        "guided_perturbed",
        "original",
    )
    require_base_improvement: bool = True
    settings: Mapping[str, object] = field(default_factory=dict)
    program_id: str = DEFAULT_PROMOTION_PRESET

    def __post_init__(self) -> None:
        margin = float(self.relative_margin)
        if not isfinite(margin) or margin < 0.0:
            raise ValueError("promotion relative_margin must be nonnegative")
        distributions = tuple(self.guardrail_distributions)
        allowed = {
            "original",
            "guided",
            "base_perturbed",
            "guided_perturbed",
        }
        if any(value not in allowed for value in distributions):
            raise ValueError("default promotion has an unknown guardrail distribution")
        if len(distributions) != len(set(distributions)):
            raise ValueError("default promotion guardrails cannot repeat")
        if not isinstance(self.require_base_improvement, bool):
            raise TypeError("require_base_improvement must be bool")
        object.__setattr__(self, "relative_margin", margin)
        object.__setattr__(self, "guardrail_distributions", distributions)
        object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))

    def select(
        self,
        incumbent_digest: str | None,
        reports: Sequence[PromotionCandidateReport],
        context: PromotionContext,
    ) -> PromotionDecision:
        del context
        rows = tuple(reports)
        by_digest = {row.candidate_digest: row for row in rows}
        if len(by_digest) != len(rows):
            raise ValueError("promotion reports contain duplicate candidates")
        incumbent: PromotionCandidateReport | None = None
        if incumbent_digest is not None:
            if incumbent_digest not in by_digest:
                raise ValueError("promotion reports do not contain the incumbent")
            incumbent = by_digest[incumbent_digest]
            if not incumbent.technically_eligible:
                raise ValueError(
                    "default promotion incumbent is technically ineligible"
                )
            if "base" not in incumbent.distributions:
                raise ValueError("default promotion requires incumbent Base metrics")

        eligible: list[PromotionCandidateReport] = []
        ineligible_reason: dict[str, str] = {}
        for row in rows:
            reason = self._ineligible_reason(row, incumbent)
            if reason is None:
                eligible.append(row)
            else:
                ineligible_reason[row.candidate_digest] = reason

        # The current incumbent is the fallback if every challenger fails.
        if incumbent is not None and incumbent not in eligible:
            eligible.append(incumbent)
            ineligible_reason.pop(incumbent_digest, None)
        if not eligible:
            raise ValueError("default promotion has no eligible initial candidate")
        ordered = sorted(
            eligible,
            key=lambda row: (
                self._base_key(row),
                (
                    0
                    if incumbent_digest is not None
                    and row.candidate_digest == incumbent_digest
                    else 1
                ),
                row.candidate_digest,
            ),
        )
        leader = ordered[0]
        if (
            incumbent is not None
            and self.require_base_improvement
            and leader.candidate_digest != incumbent_digest
            and not self._strictly_improves_base(leader, incumbent)
        ):
            leader = incumbent

        ranking_by_digest = {
            row.candidate_digest: index + 1 for index, row in enumerate(ordered)
        }
        ranking = tuple(
            PromotionRankingEntry(
                candidate_digest=row.candidate_digest,
                rank=ranking_by_digest.get(row.candidate_digest),
                eligible=row.candidate_digest in ranking_by_digest,
                reason=(
                    "eligible_base_ranking"
                    if row.candidate_digest in ranking_by_digest
                    else ineligible_reason[row.candidate_digest]
                ),
            )
            for row in rows
        )
        promoted = (
            incumbent_digest is None
            or leader.candidate_digest != incumbent_digest
        )
        return PromotionDecision(
            selected_digest=leader.candidate_digest,
            promoted=promoted,
            reason=(
                "initial_candidate_selected_by_base_metrics"
                if incumbent_digest is None
                else (
                    "base_improved_with_robustness_guardrails"
                    if promoted
                    else "incumbent_kept_after_robustness_guardrails"
                )
            ),
            ranking=ranking,
        )

    def _ineligible_reason(
        self,
        row: PromotionCandidateReport,
        incumbent: PromotionCandidateReport | None,
    ) -> str | None:
        if not row.technically_eligible:
            return "technical_evaluation_failed"
        if "base" not in row.distributions:
            return "base_evaluation_missing"
        # The built-in guardrails are non-regression comparisons. With no S0
        # there is no incumbent baseline, so first selection ranks complete
        # technically valid Base reports directly.
        if incumbent is None:
            return None
        for name, passed in row.guardrails.items():
            if not passed:
                return f"{name}_failed"
        if row.candidate_digest == incumbent.candidate_digest:
            return None
        for name in self.guardrail_distributions:
            candidate_metric = row.distributions.get(name)
            incumbent_metric = incumbent.distributions.get(name)
            if candidate_metric is None or incumbent_metric is None:
                # BASE perturbation is an explicit opt-in guardrail, never an
                # optional member of the legacy default tuple. Missing either
                # side cannot establish its requested non-regression check.
                if name == "base_perturbed":
                    return f"{name}_evaluation_missing"
                continue
            if (
                candidate_metric.mean
                > (1.0 + self.relative_margin) * incumbent_metric.mean
            ):
                return f"{name}_relative_margin_failed"
        return None

    @staticmethod
    def _base_key(row: PromotionCandidateReport) -> tuple[float, float, float]:
        metric = row.distributions["base"]
        return (metric.mean, metric.cvar, metric.maximum)

    @staticmethod
    def _strictly_improves_base(
        candidate: PromotionCandidateReport,
        incumbent: PromotionCandidateReport,
    ) -> bool:
        return BaseImprovementWithRobustnessGuardrails._base_key(
            candidate
        ) < BaseImprovementWithRobustnessGuardrails._base_key(incumbent)


class FunctionPromotionProgram:
    """Adapter for a user function named by the Campaign config."""

    def __init__(self, function: Callable[..., object], *, program_id: str) -> None:
        if not callable(function):
            raise TypeError("custom promotion entrypoint must be callable")
        if not isinstance(program_id, str) or not program_id.strip():
            raise ValueError("custom promotion program_id cannot be empty")
        self._function = function
        self.program_id = program_id.strip()

    def select(
        self,
        incumbent_digest: str | None,
        reports: Sequence[PromotionCandidateReport],
        context: PromotionContext,
    ) -> PromotionDecision:
        result = self._function(incumbent_digest, tuple(reports), context)
        if not isinstance(result, PromotionDecision):
            raise TypeError("custom promotion function must return PromotionDecision")
        return result


def _metric_reference_from_config(
    value: object,
    *,
    field_name: str,
) -> PromotionMetricReference:
    if not isinstance(value, Mapping):
        raise TypeError(f"promotion {field_name} must be an object")
    raw = dict(value)
    if set(raw) != {"distribution", "statistic"}:
        raise ValueError(
            f"promotion {field_name} needs exactly distribution and statistic"
        )
    distribution = raw["distribution"]
    statistic = raw["statistic"]
    if not isinstance(distribution, str) or not isinstance(statistic, str):
        raise TypeError(
            f"promotion {field_name} distribution and statistic must be strings"
        )
    return PromotionMetricReference(distribution, statistic)


def _metric_reference_sequence_from_config(
    value: object,
    *,
    field_name: str,
) -> tuple[PromotionMetricReference, ...]:
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError(f"promotion {field_name} must be a sequence")
    try:
        entries = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"promotion {field_name} must be a sequence") from exc
    return tuple(
        _metric_reference_from_config(
            entry,
            field_name=f"{field_name}[{index}]",
        )
        for index, entry in enumerate(entries)
    )


def build_promotion_program(
    specification: Mapping[str, object] | None,
    *,
    config_dir: Path | None = None,
):
    """Build the framework preset or load one user Python entrypoint."""

    spec = dict(specification or {"preset": DEFAULT_PROMOTION_PRESET})
    settings_value = spec.pop("settings", {})
    if not isinstance(settings_value, Mapping):
        raise TypeError("promotion settings must be an object")
    settings = dict(settings_value)
    if "preset" in spec:
        if set(spec) != {"preset"}:
            raise ValueError("promotion preset cannot also name a Python program")
        if spec["preset"] == MINIMIZE_METRIC_PROMOTION_PRESET:
            allowed = {
                "primary",
                "require_improvement",
                "minimum_improvement",
                "tie_breakers",
                "guardrails",
            }
            if not set(settings).issubset(allowed):
                raise ValueError("minimize_metric settings contain unknown fields")
            primary = _metric_reference_from_config(
                settings.get(
                    "primary",
                    {"distribution": "base", "statistic": "mean"},
                ),
                field_name="primary",
            )
            require_improvement = settings.get("require_improvement", True)
            if not isinstance(require_improvement, bool):
                raise TypeError("promotion require_improvement must be bool")
            minimum_improvement = settings.get("minimum_improvement", 0.0)
            tie_breakers = _metric_reference_sequence_from_config(
                settings.get("tie_breakers", ()),
                field_name="tie_breakers",
            )
            guardrails = _metric_reference_sequence_from_config(
                settings.get("guardrails", ()),
                field_name="guardrails",
            )
            return MinimizeMetricPromotion(
                primary=primary,
                require_improvement=require_improvement,
                minimum_improvement=minimum_improvement,
                tie_breakers=tie_breakers,
                guardrails=guardrails,
                settings=settings,
            )
        if spec["preset"] != DEFAULT_PROMOTION_PRESET:
            raise ValueError(f"unknown promotion preset: {spec['preset']!r}")
        allowed = {
            "relative_margin",
            "guardrail_distributions",
            "require_base_improvement",
        }
        if not set(settings).issubset(allowed):
            raise ValueError("default promotion settings contain unknown fields")
        relative_margin = settings.get("relative_margin", 0.05)
        if (
            isinstance(relative_margin, bool)
            or not isinstance(relative_margin, Real)
        ):
            raise TypeError("promotion relative_margin must be numeric")
        guardrail_distributions = settings.get(
            "guardrail_distributions",
            ("guided", "guided_perturbed", "original"),
        )
        if isinstance(guardrail_distributions, (str, bytes)):
            raise TypeError("promotion guardrail_distributions must be a sequence")
        try:
            guardrail_distributions = tuple(guardrail_distributions)
        except TypeError as exc:
            raise TypeError(
                "promotion guardrail_distributions must be a sequence"
            ) from exc
        require_base_improvement = settings.get(
            "require_base_improvement", True
        )
        if not isinstance(require_base_improvement, bool):
            raise TypeError("promotion require_base_improvement must be bool")
        return BaseImprovementWithRobustnessGuardrails(
            relative_margin=float(relative_margin),
            guardrail_distributions=guardrail_distributions,
            require_base_improvement=require_base_improvement,
            settings=settings,
        )
    if set(spec) != {"program", "entrypoint"}:
        raise ValueError(
            "custom promotion needs exactly program, entrypoint, and optional settings"
        )
    raw_path = spec["program"]
    entrypoint = spec["entrypoint"]
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("custom promotion program path cannot be empty")
    if not isinstance(entrypoint, str) or not entrypoint.isidentifier():
        raise ValueError("custom promotion entrypoint must be a Python identifier")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        if config_dir is None:
            raise ValueError("relative promotion program needs a config directory")
        path = config_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"promotion program does not exist: {path}")
    module_spec = importlib.util.spec_from_file_location(
        f"pde_user_promotion_{abs(hash(path))}", path
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot load promotion program: {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    function = getattr(module, entrypoint, None)
    if not callable(function):
        raise AttributeError(
            f"promotion program has no callable {entrypoint!r}: {path}"
        )

    def configured_function(incumbent_digest, reports, context):
        configured_context = PromotionContext(
            round_index=context.round_index,
            settings={**dict(context.settings), **settings},
        )
        return function(incumbent_digest, reports, configured_context)

    return FunctionPromotionProgram(
        configured_function,
        program_id=f"python:{path.name}:{entrypoint}",
    )


def promotion_metric_references(
    specification: Mapping[str, object],
) -> tuple[PromotionMetricReference, ...]:
    """Return the built-in metrics whose hidden distributions must exist.

    A custom Python PromotionProgram is intentionally opaque to the compiler.
    Built-in presets are validated here so a bad hidden-suite recipe fails in
    ``plan`` rather than after an Agent round has completed.
    """

    spec = dict(specification)
    if "preset" not in spec:
        return ()
    program = build_promotion_program(spec)
    if isinstance(program, MinimizeMetricPromotion):
        return (program.primary,) + program.tie_breakers + program.guardrails
    if isinstance(program, BaseImprovementWithRobustnessGuardrails):
        # Optional robustness distributions in the legacy preset retain their
        # historical skip-if-absent behavior. Base is its only required metric.
        return (PromotionMetricReference("base", "mean"),)
    return ()


def validate_promotion_decision(
    incumbent_digest: str | None,
    reports: Sequence[PromotionCandidateReport],
    decision: PromotionDecision,
) -> None:
    """Enforce the framework-wide boundary around user promotion code."""

    if not isinstance(decision, PromotionDecision):
        raise TypeError("PromotionProgram must return PromotionDecision")
    prepared_reports = tuple(reports)
    report_by_digest = {row.candidate_digest: row for row in prepared_reports}
    if len(report_by_digest) != len(prepared_reports):
        raise ValueError("official promotion reports contain duplicate candidates")
    if incumbent_digest is not None and incumbent_digest not in report_by_digest:
        raise ValueError("official promotion reports do not contain the incumbent")
    ranking_by_digest = {
        row.candidate_digest: row
        for row in decision.ranking
    }
    ranking_digests = set(ranking_by_digest)
    if ranking_digests != set(report_by_digest):
        raise ValueError("PromotionProgram ranking must cover every official report")
    if decision.selected_digest not in report_by_digest:
        raise ValueError("PromotionProgram selected an unknown candidate")
    selected_report = report_by_digest[decision.selected_digest]
    if not selected_report.technically_eligible:
        raise ValueError(
            "PromotionProgram cannot select a technically invalid candidate"
        )
    if not ranking_by_digest[decision.selected_digest].eligible:
        raise ValueError(
            "PromotionProgram cannot select a ranking-ineligible candidate"
        )
    actually_promoted = (
        incumbent_digest is None
        or decision.selected_digest != incumbent_digest
    )
    if decision.promoted != actually_promoted:
        raise ValueError("PromotionProgram promoted flag disagrees with its selection")


def metric_summary(
    *, case_count: int, mean: float, cvar: float, maximum: float
) -> PromotionMetricSummary:
    return PromotionMetricSummary(case_count, mean, cvar, maximum)


__all__ = [
    "DEFAULT_PROMOTION_PRESET",
    "MINIMIZE_METRIC_PROMOTION_PRESET",
    "BaseImprovementWithRobustnessGuardrails",
    "MinimizeMetricPromotion",
    "PromotionMetricReference",
    "FunctionPromotionProgram",
    "build_promotion_program",
    "promotion_metric_references",
    "validate_promotion_decision",
    "metric_summary",
]
