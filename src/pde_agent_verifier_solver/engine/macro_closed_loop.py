"""Legacy formal macro protocol retained for historical experiment replay.

New product runs use :class:`CampaignController`. This module remains executable
only so existing formal experiments can be reproduced while their subprocess,
GPU, timeout, bundle, and recovery capabilities are migrated behind the
canonical AgentExecutionService boundary. Do not use it as a new PDE
composition root.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from math import ceil, isfinite
from numbers import Real
from pathlib import Path
from statistics import fmean
from typing import Callable, Mapping, Protocol, Sequence
from types import MappingProxyType

from pde_agent_verifier_solver.sdk import (
    Case,
    PublicProblemBundle,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    GenerationKind,
    RecoverableSamplingRejection,
)
from pde_agent_verifier_solver.sdk.guidance import (
    CohortMetrics,
    FrozenGuidanceSnapshot,
    GuidanceCohortKind,
    GuidanceUnsupported,
    SeparatedGuidanceMetrics,
)

from pde_agent_verifier_solver.sdk.execution_boundary import (
    ExecutionAssurance,
    require_execution_assurance,
)
from pde_agent_verifier_solver.sdk.promotion import (
    PromotionCandidateReport,
    PromotionContext,
    PromotionDecision,
    PromotionMetricSummary,
    PromotionProgram,
)
from pde_agent_verifier_solver.sdk.macro_loop import (
    CandidateRef,
    CandidateSubmission,
    CohortKind,
    MacroLoopPlugin,
    SampleRequest,
)

from .development_session import (
    DevelopmentDrawKind,
    DevelopmentDrawRequest,
    DevelopmentSession,
)
from .macro_evaluator import (
    ControllerMacroEvaluator,
    DevelopmentMacroEvaluator,
    MacroBatchEvaluation,
    MacroBatchStreamAccumulator,
    PreparedMacroCohort,
)
from .macro_query_plan import (
    ControllerMacroQueryPlan,
    controller_query_plan_seed,
)
from .guidance_controller import (
    DevelopmentGuidanceOracle,
    FrozenGuidanceBoundary,
    GuidanceLifecycleController,
    GuidanceProtocolConfig,
    GuidanceRewardFeedback,
    GuidanceRewardRow,
    MacroGuidanceBackend,
)

from .selection import (
    EvidenceStage,
    MacroCandidateScore,
    PairedObjectiveMode,
    PairedObjectivePolicy,
    PairedComparison,
    PromotionResult,
    RobustAggregate,
    aggregate_score,
    compare_perturbed_non_regression,
    paired_objective_score,
    paired_weighted_composite,
    split_paired_score,
    paired_compare,
)

from .promotion import (
    BaseImprovementWithRobustnessGuardrails,
    validate_promotion_decision,
)


@dataclass(frozen=True)
class MacroAgentSpec:
    agent_id: str
    name: str
    prompt: str = (
        "Improve the incumbent model using only the public cases and the "
        "budgeted macro evaluator. Finalize one immutable candidate."
    )

    def __post_init__(self) -> None:
        if not self.agent_id.strip() or not self.name.strip():
            raise ValueError("Agent id and name cannot be empty")


_DEFAULT_RELEASE_CASES = object()
_OFFICIAL_COMMON_ROOT_BATCH_SIZE = 1
_DEVELOPMENT_RELEASE_QUERY_PROFILE = "development"
_CANDIDATE_EVALUATION_QUERY_PROFILE = "terminal"


@dataclass(frozen=True)
class CompetitiveSearchPolicy:
    """PDE-neutral leaderboard and handoff policy for broad Agent search.

    The four search-completion fields are retained only so older campaign
    manifests can still be loaded. The open-search controller deliberately
    does not enforce them or expose them to an Agent as finish requirements.
    Candidate quality on the common cohort is the only checkpoint-selection
    objective. This policy never defines a PDE, perturbation, observable, or
    macroscopic quantity.
    """

    min_distinct_screened_candidates: int = 0
    min_full_stage_candidates: int = 0
    min_released_roots: int = 0
    require_final_full_stage: bool = False
    leaderboard_limit: int = 8
    cross_share_after_round: int = 1

    def __post_init__(self) -> None:
        integer_fields = {
            "min_distinct_screened_candidates": (
                self.min_distinct_screened_candidates
            ),
            "min_full_stage_candidates": self.min_full_stage_candidates,
            "min_released_roots": self.min_released_roots,
            "leaderboard_limit": self.leaderboard_limit,
            "cross_share_after_round": self.cross_share_after_round,
        }
        for name, value in integer_fields.items():
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(
                    f"{name} must be a nonnegative integer"
                )
        if not isinstance(self.require_final_full_stage, bool):
            raise TypeError("require_final_full_stage must be bool")


@dataclass(frozen=True)
class MacroLoopConfig:
    name: str
    round_count: int
    agents: tuple[MacroAgentSpec, ...]
    development_case_count: int
    selection_case_count: int
    evidence_stages: tuple[EvidenceStage, ...]
    terminal_case_count: int
    agent_case_credits: int
    seed_base: int = 0
    seed_stride: int = 10_000
    development_pool_multiplier: int = 1
    development_screen_representative_limit: int = 3
    development_release_root_limit: int = 0
    development_screen_root_stages: tuple[int, ...] = ()
    development_screen_root_credit_limit: int = 0
    development_screen_confidence_z: float = 1.96
    development_screen_continuation_max_mean_ratio: float | None = 1.05
    cvar_tail_fraction: float = 0.1
    terminal_confidence_z: float = 1.645
    terminal_min_win_rate: float = 0.5
    terminal_require_cvar_no_regression: bool = True
    allow_development_truth: bool = False
    invalid_error: float = 1.0e30
    candidate_compute_budget: float | None = None
    candidate_query_seconds_per_point_limit: float | None = None
    execution_assurance: ExecutionAssurance = ExecutionAssurance.REFERENCE_IN_PROCESS
    paired_perturbations: bool = False
    paired_objective_policy: PairedObjectivePolicy = PairedObjectivePolicy()
    paired_sampling_max_attempts: int = 16
    search_policy: CompetitiveSearchPolicy = CompetitiveSearchPolicy()
    guidance_protocol: GuidanceProtocolConfig | None = None
    promotion_program: PromotionProgram = BaseImprovementWithRobustnessGuardrails()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("macro loop name cannot be empty")
        if not isinstance(self.search_policy, CompetitiveSearchPolicy):
            raise TypeError("search_policy must be a CompetitiveSearchPolicy")
        if self.guidance_protocol is not None and not isinstance(
            self.guidance_protocol, GuidanceProtocolConfig
        ):
            raise TypeError("guidance_protocol must be GuidanceProtocolConfig or None")
        promotion_id = getattr(self.promotion_program, "program_id", None)
        if not isinstance(promotion_id, str) or not promotion_id.strip():
            raise TypeError("promotion_program must provide a nonempty program_id")
        if not callable(getattr(self.promotion_program, "select", None)):
            raise TypeError("promotion_program must provide select(...)")
        positive_integer_fields = {
            "round_count": self.round_count,
            "development_case_count": self.development_case_count,
            "selection_case_count": self.selection_case_count,
            "terminal_case_count": self.terminal_case_count,
            "agent_case_credits": self.agent_case_credits,
            "development_pool_multiplier": self.development_pool_multiplier,
            "paired_sampling_max_attempts": self.paired_sampling_max_attempts,
        }
        for field_name, value in positive_integer_fields.items():
            _require_integer(field_name, value, minimum=1)
        screen_stages = tuple(self.development_screen_root_stages)
        object.__setattr__(
            self, "development_screen_root_stages", screen_stages
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for value in screen_stages
        ) or any(
            current <= previous
            for previous, current in zip(screen_stages, screen_stages[1:])
        ):
            raise ValueError(
                "development screen root stages must strictly increase"
            )
        _require_integer(
            "development_release_root_limit",
            self.development_release_root_limit,
            minimum=0,
        )
        _require_integer(
            "development_screen_root_credit_limit",
            self.development_screen_root_credit_limit,
            minimum=0,
        )
        _require_integer(
            "development_screen_representative_limit",
            self.development_screen_representative_limit,
            minimum=0,
        )
        if self.development_screen_representative_limit > 3:
            raise ValueError(
                "development screen representative limit cannot exceed 3"
            )

        if not isfinite(float(self.development_screen_confidence_z)) or (
            self.development_screen_confidence_z < 0.0
        ):
            raise ValueError(
                "development screen confidence_z must be finite and nonnegative"
            )
        continuation_ratio = (
            self.development_screen_continuation_max_mean_ratio
        )
        if continuation_ratio is not None and (
            not isfinite(float(continuation_ratio))
            or float(continuation_ratio) <= 0.0
        ):
            raise ValueError(
                "development screen continuation ratio must be positive or None"
            )
        _require_integer("seed_base", self.seed_base, minimum=0)
        _require_integer("seed_stride", self.seed_stride, minimum=4)
        if not self.agents:
            raise ValueError("at least one Agent is required")
        ids = tuple(agent.agent_id for agent in self.agents)
        if len(set(ids)) != len(ids):
            raise ValueError("Agent ids must be unique")
        if not self.evidence_stages:
            raise ValueError("at least one evidence stage is required")
        for stage in self.evidence_stages:
            _require_integer("evidence stage case_count", stage.case_count, minimum=1)
        counts = tuple(stage.case_count for stage in self.evidence_stages)
        if any(current <= previous for previous, current in zip(counts, counts[1:])):
            raise ValueError("evidence stage counts must be strictly increasing")
        if (
            not isfinite(float(self.cvar_tail_fraction))
            or not 0.0 < self.cvar_tail_fraction <= 1.0
        ):
            raise ValueError("cvar_tail_fraction must lie in (0, 1]")
        if (
            not isfinite(float(self.terminal_confidence_z))
            or self.terminal_confidence_z < 0.0
        ):
            raise ValueError("terminal_confidence_z must be finite and nonnegative")
        if (
            self.terminal_case_count == 1
            and self.terminal_confidence_z > 0.0
        ):
            raise ValueError("positive-z terminal evidence requires at least two cases")
        if not 0.0 <= self.terminal_min_win_rate <= 1.0:
            raise ValueError("terminal_min_win_rate must lie in [0, 1]")
        if not isinstance(self.terminal_require_cvar_no_regression, bool):
            raise TypeError("terminal_require_cvar_no_regression must be bool")
        if not isfinite(float(self.invalid_error)) or self.invalid_error <= 0.0:
            raise ValueError("invalid_error must be finite and positive")
        _validated_compute_budget(self.candidate_compute_budget)
        query_limit = _validated_query_seconds_per_point_limit(
            self.candidate_query_seconds_per_point_limit
        )
        if query_limit is not None and self.candidate_compute_budget is None:
            raise ValueError(
                "formal point-query eligibility also requires a solve budget"
            )
        if not isinstance(self.execution_assurance, ExecutionAssurance):
            raise TypeError("execution_assurance must be an ExecutionAssurance")
        if not isinstance(self.allow_development_truth, bool):
            raise TypeError("allow_development_truth must be bool")
        if not isinstance(self.paired_perturbations, bool):
            raise TypeError("paired_perturbations must be bool")
        if not isinstance(self.paired_objective_policy, PairedObjectivePolicy):
            raise TypeError("paired_objective_policy must be a PairedObjectivePolicy")
        if (
            not self.paired_perturbations
            and self.paired_objective_policy.mode
            is not PairedObjectiveMode.WEIGHTED_COMPOSITE
        ):
            raise ValueError(
                "a generated-primary perturbation guardrail needs paired perturbations"
            )
        if self.paired_perturbations:
            paired_counts = {
                "development_case_count": self.development_case_count,
                "selection_case_count": self.selection_case_count,
                "terminal_case_count": self.terminal_case_count,
            }
            for field_name, value in paired_counts.items():
                if value % 2:
                    raise ValueError(
                        f"{field_name} must be even with paired perturbations"
                    )
            if any(stage.case_count % 2 for stage in self.evidence_stages):
                raise ValueError(
                    "evidence stage case_count must be even with paired perturbations"
                )
            if any(
                stage.confidence_z > 0.0 and stage.case_count < 4
                for stage in self.evidence_stages
            ):
                raise ValueError(
                    "positive-confidence paired evidence needs at least two pairs"
                )
            if self.terminal_confidence_z > 0.0 and self.terminal_case_count < 4:
                raise ValueError(
                    "positive-confidence terminal evidence needs at least two pairs"
                )
            if self.seed_stride < 8:
                raise ValueError(
                    "seed_stride must be at least 8 with paired perturbations"
                )

        strong_development = bool(screen_stages)
        initial_release_roots = (
            self.development_case_count // 2
            if self.paired_perturbations
            else self.development_case_count
        )
        if strong_development:
            if self.development_release_root_limit < initial_release_roots:
                raise ValueError(
                    "development release credits cannot fund the initial roots"
                )
            if (
                self.development_screen_root_credit_limit
                < screen_stages[0]
            ):
                raise ValueError(
                    "development screen credits cannot fund the first stage"
                )
            if self.development_pool_multiplier != 1:
                raise ValueError(
                    "fresh development screens cannot use a fixed hard-mining pool"
                )
            if self.allow_development_truth:
                raise ValueError(
                    "strong development uses completed-answer release, not fixed-board truth"
                )
        elif (
            self.development_release_root_limit != 0
            or self.development_screen_root_credit_limit != 0
        ):
            raise ValueError(
                "development root credits require screen_root_stages"
            )
        search = self.search_policy
        if search.cross_share_after_round > self.round_count:
            raise ValueError("cross-share round cannot exceed campaign rounds")


@dataclass(frozen=True)
class SearchLeaderboardEntry:
    """Sanitized common-cohort standing carried to the next checkpoint."""

    rank: int
    candidate: CandidateRef
    agent_id: str
    eligible: bool
    mean: float | None
    cvar: float | None
    maximum: float | None
    generated_mean: float | None = None
    perturbed_mean: float | None = None
    guardrail_passed: bool | None = None
    roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_integer("leaderboard rank", self.rank, minimum=1)
        if not isinstance(self.candidate, CandidateRef):
            raise TypeError("leaderboard candidate must be a CandidateRef")
        if not isinstance(self.agent_id, str) or not self.agent_id.strip():
            raise ValueError("leaderboard agent_id cannot be empty")
        if not isinstance(self.eligible, bool):
            raise TypeError("leaderboard eligible must be bool")
        for name in (
            "mean",
            "cvar",
            "maximum",
            "generated_mean",
            "perturbed_mean",
        ):
            value = getattr(self, name)
            if value is not None and not isfinite(float(value)):
                raise ValueError(f"leaderboard {name} must be finite or None")
        if self.guardrail_passed is not None and not isinstance(
            self.guardrail_passed, bool
        ):
            raise TypeError("leaderboard guardrail outcome must be bool or None")
        roles = tuple(self.roles)
        if any(not isinstance(role, str) or not role for role in roles):
            raise ValueError("leaderboard roles must be nonempty strings")
        object.__setattr__(self, "roles", roles)


@dataclass(frozen=True)
class SearchCompletionAudit:
    """Aggregate-only result of applying one Agent finish contract."""

    agent_id: str
    distinct_screened_candidates: int
    full_stage_candidates: int
    released_roots: int
    final_candidate_sha256: str | None
    final_completed_root_count: int
    sufficient: bool
    failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id.strip():
            raise ValueError("search audit agent_id cannot be empty")
        for name in (
            "distinct_screened_candidates",
            "full_stage_candidates",
            "released_roots",
            "final_completed_root_count",
        ):
            _require_integer(name, getattr(self, name), minimum=0)
        if self.final_candidate_sha256 is not None:
            if (
                len(self.final_candidate_sha256) != 64
                or any(
                    char not in "0123456789abcdef"
                    for char in self.final_candidate_sha256
                )
            ):
                raise ValueError("search audit final candidate digest is invalid")
        if not isinstance(self.sufficient, bool):
            raise TypeError("search sufficiency must be bool")
        failures = tuple(self.failures)
        if any(not isinstance(item, str) or not item for item in failures):
            raise ValueError("search audit failures must be nonempty strings")
        object.__setattr__(self, "failures", failures)


def evaluate_search_completion(
    session: DevelopmentSession,
    *,
    final_candidate_sha256: str | None,
    agent_id: str,
    policy: CompetitiveSearchPolicy,
) -> SearchCompletionAudit:
    """Return observational search accounting without imposing a finish gate.

    Legacy policy minima are intentionally ignored. Only aggregate session
    accounting is inspected; no private case identity, target, or per-case
    score enters the result. A nominated checkpoint remains admissible even
    when the Agent did not screen it first.
    """

    if not isinstance(session, DevelopmentSession):
        raise TypeError("session must be a DevelopmentSession")
    if not isinstance(policy, CompetitiveSearchPolicy):
        raise TypeError("policy must be a CompetitiveSearchPolicy")

    by_digest = {
        row.candidate.ref.bundle_sha256: row
        for row in session.screened_candidate_summaries
    }
    full_stage = session.screen_root_stages[-1]
    distinct_count = len(by_digest)
    full_stage_count = sum(
        row.completed_root_count >= full_stage
        for row in by_digest.values()
    )
    final_completed = (
        by_digest[final_candidate_sha256].completed_root_count
        if final_candidate_sha256 in by_digest
        else 0
    )
    released_roots = session.release_charged_root_credits


    return SearchCompletionAudit(
        agent_id=agent_id,
        distinct_screened_candidates=distinct_count,
        full_stage_candidates=full_stage_count,
        released_roots=released_roots,
        final_candidate_sha256=final_candidate_sha256,
        final_completed_root_count=final_completed,
        sufficient=True,
        failures=(),
    )


@dataclass(frozen=True)
class MacroAgentContext:
    round_index: int
    agent: MacroAgentSpec
    prompt: str
    cases: tuple[Case, ...]
    evaluator: DevelopmentMacroEvaluator | DevelopmentSession
    incumbent: CandidateRef
    prior_handoff: RoundHandoff | None
    prior_leaderboard: tuple[SearchLeaderboardEntry, ...] = ()
    shared_candidates: tuple[CandidateRef, ...] = ()
    search_policy: CompetitiveSearchPolicy = CompetitiveSearchPolicy()
    public_problem_bundle: PublicProblemBundle | None = None
    public_session_factory: Callable[[Case], object] | None = None
    public_session_methods: tuple[str, ...] = ()
    development_solution_session_factory: Callable[[Case], object] | None = None
    development_solution_session_methods: tuple[str, ...] = ()
    guidance_oracle: DevelopmentGuidanceOracle | None = None


@dataclass(frozen=True)
class MacroAgentOutcome:
    candidate: CandidateSubmission
    observations: tuple[str, ...] = ()


class MacroAgentProgram(Protocol):
    def run(self, context: MacroAgentContext) -> MacroAgentOutcome: ...


@dataclass(frozen=True)
class PublishedMetricAggregate:
    """Public aggregate of one objective-allowlisted per-case metric."""

    mean: float
    cvar: float
    max: float

    def __post_init__(self) -> None:
        if any(
            not isfinite(float(value))
            for value in (self.mean, self.cvar, self.max)
        ):
            raise ValueError("published metric aggregates must be finite")


@dataclass(frozen=True)
class PublishedMetricSummary:
    """A complete-cohort summary with no case ids or per-case values."""

    case_count: int
    aggregates: Mapping[str, PublishedMetricAggregate] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_integer("published metric case_count", self.case_count, minimum=1)
        aggregates = dict(self.aggregates)
        if any(not isinstance(name, str) or not name.strip() for name in aggregates):
            raise ValueError("published metric names must be nonempty strings")
        if any(
            not isinstance(value, PublishedMetricAggregate)
            for value in aggregates.values()
        ):
            raise TypeError("published metrics must contain aggregate values")
        object.__setattr__(self, "aggregates", MappingProxyType(aggregates))


@dataclass(frozen=True)
class PairedPublishedMetricSummary:
    """Allowlisted summaries for two candidates on one exact paired prefix."""

    case_count: int
    candidate: PublishedMetricSummary
    incumbent: PublishedMetricSummary

    def __post_init__(self) -> None:
        _require_integer(
            "paired published metric case_count",
            self.case_count,
            minimum=1,
        )
        if (
            self.candidate.case_count != self.case_count
            or self.incumbent.case_count != self.case_count
        ):
            raise ValueError("paired published metric counts must agree")
        if set(self.candidate.aggregates) != set(self.incumbent.aggregates):
            raise ValueError("paired published metric names must agree")


@dataclass(frozen=True)
class PairedObjectiveAggregates:
    """ID-free generated, perturbed, and weighted-composite summaries."""

    generated: RobustAggregate
    perturbed: RobustAggregate
    composite: RobustAggregate


@dataclass(frozen=True)
class PerturbedGuardrailSummary:
    """Public-safe projection of one perturbed non-regression check."""

    root_count: int
    candidate_mean: float
    incumbent_mean: float
    mean_excess: float
    standard_error: float
    confidence_lower: float
    confidence_z: float
    practical_relative_margin: float
    passed: bool
    failures: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_integer("guardrail root_count", self.root_count, minimum=1)
        if any(
            not isfinite(float(value))
            for value in (
                self.candidate_mean,
                self.incumbent_mean,
                self.mean_excess,
                self.standard_error,
                self.confidence_lower,
                self.confidence_z,
                self.practical_relative_margin,
            )
        ):
            raise ValueError("guardrail summary values must be finite")
        if not isinstance(self.passed, bool):
            raise TypeError("guardrail passed must be bool")
        failures = tuple(self.failures)
        if any(not isinstance(item, str) or not item for item in failures):
            raise ValueError("guardrail failures must be nonempty strings")
        object.__setattr__(self, "failures", failures)


@dataclass(frozen=True)
class FinalistEvaluation:
    candidate: CandidateRef
    batch: MacroBatchEvaluation
    score: MacroCandidateScore
    aggregate: RobustAggregate | None
    published_metrics: PublishedMetricSummary | None = None
    objective_aggregates: PairedObjectiveAggregates | None = None
    perturbed_guardrail: PerturbedGuardrailSummary | None = None



@dataclass(frozen=True)
class GuidanceCandidateEvaluation:
    candidate: CandidateRef
    guided_batch: MacroBatchEvaluation
    guided_score: MacroCandidateScore
    guided_aggregate: RobustAggregate | None
    separated_metrics: SeparatedGuidanceMetrics | None


@dataclass(frozen=True)
class GuidanceCohortEvaluation:
    """Base and guided metrics stay separate; this record has no scalar rank."""

    snapshot: FrozenGuidanceSnapshot
    cohort_id: str
    cohort_sha256: str
    campaign_guided_generation_wall_seconds: float
    candidates: tuple[GuidanceCandidateEvaluation, ...]

@dataclass(frozen=True)
class RoundFeedback:
    previous_incumbent_sha256: str
    selected_sha256: str
    promoted: bool
    decision_reason: str
    evidence_case_count: int
    paired_mean_improvement: float | None
    paired_confidence_lower: float | None
    paired_win_rate: float | None
    confidence_group_size: int = 1
    perturbed_guardrail_passed: bool | None = None
    perturbed_guardrail_confidence_lower: float | None = None

    def __post_init__(self) -> None:
        _require_integer("evidence_case_count", self.evidence_case_count, minimum=0)
        _require_integer("confidence_group_size", self.confidence_group_size, minimum=1)
        if self.evidence_case_count % self.confidence_group_size:
            raise ValueError("evidence cases must divide into confidence groups")
        if self.perturbed_guardrail_passed is not None and not isinstance(
            self.perturbed_guardrail_passed, bool
        ):
            raise TypeError("perturbed guardrail outcome must be bool or None")
        if (self.perturbed_guardrail_passed is None) != (
            self.perturbed_guardrail_confidence_lower is None
        ):
            raise ValueError(
                "guardrail outcome and confidence bound must appear together"
            )
        if self.perturbed_guardrail_confidence_lower is not None and not isfinite(
            float(self.perturbed_guardrail_confidence_lower)
        ):
            raise ValueError("perturbed guardrail confidence bound must be finite")


@dataclass(frozen=True)
class RoundHandoff:
    """Promotion-lineage commitment passed between in-process rounds.

    Search policy, completion audits, and leaderboard state are deliberately
    audited by ``MacroRoundResult`` and the published result rather than this
    digest. A future checkpoint-resume path must therefore bind that search
    state explicitly instead of reconstructing a round from this record alone.
    """

    round_index: int
    selected: CandidateRef
    promoted: bool
    challenger: CandidateRef | None
    decision_reason: str
    feedback: RoundFeedback
    previous_handoff_sha256: str | None
    handoff_sha256: str


@dataclass(frozen=True)
class MacroRoundResult:
    round_index: int
    development_case_ids: tuple[str, ...]
    candidate_refs: tuple[CandidateRef, ...]
    agent_failures: Mapping[str, str]
    finalist_evaluations: tuple[FinalistEvaluation, ...]
    challenger: CandidateRef | None
    promotion: PromotionResult | None
    selected: CandidateRef
    handoff: RoundHandoff
    promotion_published_metrics: tuple[PairedPublishedMetricSummary, ...] = ()
    candidate_portfolio_roles: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    promotion_challenger_objective_aggregates: PairedObjectiveAggregates | None = None
    promotion_incumbent_objective_aggregates: PairedObjectiveAggregates | None = None
    promotion_perturbed_guardrail: PerturbedGuardrailSummary | None = None
    leaderboard: tuple[SearchLeaderboardEntry, ...] = ()
    search_completion: Mapping[str, SearchCompletionAudit] = field(
        default_factory=dict
    )
    guidance_evaluation: GuidanceCohortEvaluation | None = None
    promotion_reports: tuple[PromotionCandidateReport, ...] = ()
    promotion_decision: PromotionDecision | None = None


@dataclass(frozen=True)
class TerminalEvaluation:
    cohort_id: str
    cohort_sha256: str
    cold_start: MacroBatchEvaluation
    selected: MacroBatchEvaluation
    cold_start_score: MacroCandidateScore
    selected_score: MacroCandidateScore
    comparison: PairedComparison | None
    cold_start_published_metrics: PublishedMetricSummary | None = None
    selected_published_metrics: PublishedMetricSummary | None = None
    paired_published_metrics: PairedPublishedMetricSummary | None = None
    cold_start_objective_aggregates: PairedObjectiveAggregates | None = None
    selected_objective_aggregates: PairedObjectiveAggregates | None = None
    perturbed_guardrail: PerturbedGuardrailSummary | None = None
    guidance_evaluation: GuidanceCohortEvaluation | None = None

    @property
    def supports_selected_over_cold_start(self) -> bool:
        """Whether untouched terminal evidence supports the selected model."""

        primary_passed = self.comparison is not None and self.comparison.passed
        guardrail_passed = (
            self.perturbed_guardrail is None
            or self.perturbed_guardrail.passed
        )
        return primary_passed and guardrail_passed


@dataclass(frozen=True)
class SamplingAttemptAudit:
    """Controller-private replay evidence for one paired cohort draw."""

    round_index: int
    cohort: CohortKind
    attempt_index: int
    maximum_attempts: int
    base_seed: int
    perturbed_seed: int
    root_case_ids: tuple[str, ...]
    accepted: bool
    rejection_audit: Mapping[str, object] | None = None


@dataclass(frozen=True)
class MacroAgentDevelopmentResult:
    """Small process-safe result of one Agent's private development phase.

    Hidden answers, prepared cohorts, evaluator objects and executable programs
    never belong in the transport form of this record. A process executor may
    reconstruct ``CandidateSubmission`` objects only after the controller has
    re-attested the returned immutable bundles.

    ``portfolio`` is a legacy field name: it carries zero or one official
    nomination, never the Agent's multi-checkpoint private archive.
    """

    agent_id: str
    portfolio: tuple[tuple[CandidateSubmission, tuple[str, ...]], ...]
    development_case_ids: tuple[str, ...]
    sampled_case_ids: tuple[str, ...]
    sampling_attempts: tuple[SamplingAttemptAudit, ...]
    search_completion: SearchCompletionAudit | None = None
    failure: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id.strip():
            raise ValueError("development result Agent id cannot be empty")
        portfolio = tuple(self.portfolio)
        if len(portfolio) > 1:
            raise ValueError(
                "each Agent may return at most one official nomination per round"
            )
        for candidate, roles in portfolio:
            if not isinstance(candidate, CandidateSubmission):
                raise TypeError(
                    "development portfolio must contain CandidateSubmission values"
                )
            role_values = tuple(roles)
            if any(not isinstance(role, str) or not role for role in role_values):
                raise ValueError("development portfolio roles must be nonempty")
        for name, values in (
            ("development_case_ids", self.development_case_ids),
            ("sampled_case_ids", self.sampled_case_ids),
        ):
            rows = tuple(values)
            if any(not isinstance(value, str) or not value for value in rows):
                raise ValueError(f"{name} must contain nonempty strings")
            if len(set(rows)) != len(rows):
                raise ValueError(f"{name} must be unique")
        if (
            self.search_completion is not None
            and self.search_completion.agent_id != self.agent_id
        ):
            raise ValueError("search completion Agent identity changed")
        if self.failure is not None and (
            not isinstance(self.failure, str) or not self.failure
        ):
            raise ValueError("development failure must be a nonempty string")
        object.__setattr__(self, "portfolio", portfolio)
        object.__setattr__(
            self, "development_case_ids", tuple(self.development_case_ids)
        )
        object.__setattr__(self, "sampled_case_ids", tuple(self.sampled_case_ids))
        object.__setattr__(self, "sampling_attempts", tuple(self.sampling_attempts))


class MacroAgentDevelopmentExecutor(Protocol):
    """Run one same-round Agent batch behind a round synchronization barrier."""

    def run_agents(
        self,
        *,
        round_index: int,
        agents: tuple[MacroAgentSpec, ...],
        incumbent: CandidateSubmission,
        prior_handoff: RoundHandoff | None,
        prior_leaderboard: tuple[SearchLeaderboardEntry, ...],
        shared_candidates: tuple[CandidateRef, ...],
        guidance_oracles: Mapping[
            str, DevelopmentGuidanceOracle
        ] | None = None,
    ) -> tuple[MacroAgentDevelopmentResult, ...]: ...


@dataclass(frozen=True)
class MacroLoopResult:
    """Controller-private audit result, not an Agent/public serialization."""

    # Batches below include hidden case scores and private target commitments.

    name: str
    plugin_id: str
    generator_id: str
    generator_sha256: str
    macro_contract_id: str
    execution_assurance: ExecutionAssurance
    formal_claims_allowed: bool
    cold_start: CandidateRef
    selected: CandidateRef
    rounds: tuple[MacroRoundResult, ...]
    terminal: TerminalEvaluation
    public_result_kind: str = "macro_target_rom"
    public_result_task: str = "macro_quantity_matching"
    published_metric_names: tuple[str, ...] = ()
    paired_perturbations: bool = False
    paired_objective_policy: PairedObjectivePolicy = PairedObjectivePolicy()
    promotion_program_id: str = ""
    sampling_attempts: tuple[SamplingAttemptAudit, ...] = ()


class MacroClosedLoopRunner:
    """Legacy equation-neutral controller for history/control-arm replay.

    New product runs use :class:`CampaignController`. This implementation is
    retained only until its formal execution capabilities have moved behind
    the canonical :class:`AgentExecutionService` boundary.
    """

    def __init__(
        self,
        *,
        legacy_macro_reference: bool = False,
        config: MacroLoopConfig,
        plugin: MacroLoopPlugin,
        cold_start: CandidateSubmission,
        agents: Mapping[str, MacroAgentProgram],
        cohort_store_root: Path | None = None,
        development_executor: MacroAgentDevelopmentExecutor | None = None,
        guidance_backend: MacroGuidanceBackend | None = None,
    ) -> None:
        if legacy_macro_reference is not True:
            raise PermissionError(
                "MacroClosedLoopRunner is disabled for product runs; use "
                "CampaignController. Pass legacy_macro_reference=True only "
                "for an explicit history/control-arm reproduction."
            )
        self.config = config
        self.plugin = plugin
        self.cold_start = cold_start
        self.agents = dict(agents)
        self.development_executor = development_executor
        bound_guidance_backend = plugin.generator.guidance_backend
        if (
            guidance_backend is not None
            and bound_guidance_backend is not None
            and guidance_backend is not bound_guidance_backend
        ):
            raise ValueError(
                "runner and Generator bind different guidance backends"
            )
        resolved_guidance_backend = (
            guidance_backend
            if guidance_backend is not None
            else bound_guidance_backend
        )
        self.guidance_backend = resolved_guidance_backend
        self._guidance_lifecycle = None
        if config.guidance_protocol is not None:
            if resolved_guidance_backend is None:
                raise GuidanceUnsupported(
                    "guidance protocol enabled without a guidance backend"
                )
            self._guidance_lifecycle = GuidanceLifecycleController(
                config=config.guidance_protocol,
                backend=resolved_guidance_backend,
            )
        if development_executor is not None and not callable(
            getattr(development_executor, "run_agents", None)
        ):
            raise TypeError("development_executor must provide run_agents")
        if cohort_store_root is None:
            self._cohort_store_root = None
        else:
            store_root = Path(cohort_store_root).expanduser().resolve()
            if store_root == Path(store_root.anchor):
                raise ValueError("cohort_store_root must be a specific directory")
            self._cohort_store_root = store_root
        self._candidate_ids: dict[str, str] = {
            cold_start.ref.candidate_id: cold_start.ref.bundle_sha256
        }
        self._candidate_digests: dict[str, str] = {
            cold_start.ref.bundle_sha256: cold_start.ref.candidate_id
        }
        self._seen_case_ids: set[str] = set()
        self._cohort_plans: dict[
            tuple[int, CohortKind], tuple[str, int, str]
        ] = {}
        self._active_development_sessions: set[DevelopmentSession] = set()
        self._prepared_cohorts: dict[
            tuple[int, CohortKind], PreparedMacroCohort
        ] = {}
        self._sampling_attempts: list[SamplingAttemptAudit] = []
        self._sampled_case_id_order: list[str] = []
        self._started = False

    @property
    def sampling_attempts(self) -> tuple[SamplingAttemptAudit, ...]:
        """Return controller-private sampling audit accumulated so far."""

        return tuple(self._sampling_attempts)

    def run(self) -> MacroLoopResult:
        try:
            result = self._run_once()
        except BaseException as exc:
            try:
                self._drain_owned_resources()
            except BaseException as cleanup_exc:
                exc.add_note(
                    "macro-loop resource cleanup also failed: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
            raise
        self._drain_owned_resources()
        return result

    def _run_once(self) -> MacroLoopResult:
        if self._started:
            raise RuntimeError("a MacroClosedLoopRunner instance is single-use")
        boundary = require_execution_assurance(self.config.execution_assurance)
        self._started = True
        plugin_identity = (
            self.plugin.plugin_id,
            self.plugin.generator.generator_id,
            self.plugin.generator.artifact_sha256,
            self.plugin.objective.matcher.contract_id,
            self._published_metric_names(),
        )
        self._validate_runtime()
        self._precompute_cohorts()
        incumbent = self.cold_start
        prior_handoff: RoundHandoff | None = None
        round_results: list[MacroRoundResult] = []
        prior_leaderboard: tuple[SearchLeaderboardEntry, ...] = ()
        shared_candidates: tuple[CandidateRef, ...] = ()

        for round_index in range(1, self.config.round_count + 1):
            round_guidance = self._train_guidance_boundary(
                source_round_index=round_index - 1,
                target_round_index=round_index,
                incumbent=incumbent,
            )
            round_guidance_oracles = (
                {
                    spec.agent_id: round_guidance.development_oracle(
                        spec.agent_id
                    )
                    for spec in self.config.agents
                }
                if round_guidance is not None
                else {}
            )
            strong_development = bool(
                self.config.development_screen_root_stages
            )
            development = (
                None
                if strong_development
                else self._development_cohort(round_index, incumbent)
            )
            development_case_ids: list[str] = (
                [] if development is None else list(development.case_ids)
            )

            outcomes: list[MacroAgentOutcome] = []
            failures: dict[str, str] = {}
            search_completion: dict[str, SearchCompletionAudit] = {}
            portfolio_roles: dict[str, tuple[str, ...]] = {}
            round_candidate_ids: dict[str, str] = {}
            if self.development_executor is None:
                external_development = ()
            else:
                try:
                    external_development = self.development_executor.run_agents(
                        round_index=round_index,
                        agents=self.config.agents,
                        incumbent=incumbent,
                        prior_handoff=prior_handoff,
                        prior_leaderboard=prior_leaderboard,
                        shared_candidates=shared_candidates,
                        **(
                            {"guidance_oracles": round_guidance_oracles}
                            if round_guidance_oracles
                            else {}
                        ),
                    )
                finally:
                    _close_guidance_oracles(round_guidance_oracles.values())
            local_agents = (
                self.config.agents if self.development_executor is None else ()
            )
            for spec in local_agents:
                audit: SearchCompletionAudit | None = None
                if strong_development:
                    evaluator = self._strong_development_session(
                        round_index=round_index,
                        agent=spec,
                        incumbent=incumbent,
                    )
                    initial_root_count = (
                        self.config.development_case_count // 2
                        if self.config.paired_perturbations
                        else self.config.development_case_count
                    )
                    try:
                        release = evaluator.release(initial_root_count)
                    except BaseException as exc:
                        try:
                            self._close_development_session(evaluator)
                            oracle = round_guidance_oracles.get(spec.agent_id)
                            if oracle is not None:
                                oracle.close()
                        except BaseException as cleanup_exc:
                            exc.add_note(
                                "strong-development session cleanup also failed: "
                                f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                            )
                        raise
                    cases = release.cases
                    development_case_ids.extend(
                        case.case_id
                        for case in evaluator._controller_release_cases()
                    )
                    del release
                else:
                    assert development is not None
                    evaluator = DevelopmentMacroEvaluator(
                        development,
                        case_credit_limit=self.config.agent_case_credits,
                        allow_finite_truth=self.config.allow_development_truth,
                        compute_budget=self.config.candidate_compute_budget,
                        query_seconds_per_point_limit=(
                            self.config.candidate_query_seconds_per_point_limit
                        ),
                        safe_metric_names=self._agent_visible_metrics(),
                        public_session_factory=self.plugin.public_session_factory,
                        public_session_methods=self.plugin.public_session_methods,
                    )
                    cases = development.public_cases()
                context = MacroAgentContext(
                    round_index=round_index,
                    agent=spec,
                    prompt=self._agent_prompt(spec),
                    cases=cases,
                    evaluator=evaluator,
                    prior_leaderboard=prior_leaderboard,
                    shared_candidates=shared_candidates,
                    search_policy=_open_search_policy(
                        self.config.search_policy
                    ),
                    incumbent=incumbent.ref,
                    prior_handoff=prior_handoff,
                    public_problem_bundle=self.plugin.public_problem_bundle,
                    public_session_factory=self.plugin.public_session_factory,
                    guidance_oracle=round_guidance_oracles.get(spec.agent_id),
                    public_session_methods=self.plugin.public_session_methods,
                    development_solution_session_factory=(
                        self.plugin.development_solution_session_factory
                    ),
                    development_solution_session_methods=(
                        self.plugin.development_solution_session_methods
                    ),
                )
                outcome: MacroAgentOutcome | None = None
                try:
                    outcome = self.agents[spec.agent_id].run(context)
                except (KeyboardInterrupt, GeneratorExit):
                    raise
                except BaseException as exc:
                    failures[spec.agent_id] = f"agent_failed:{type(exc).__name__}"
                finally:
                    if strong_development:
                        final_digest = (
                            outcome.candidate.ref.bundle_sha256
                            if outcome is not None
                            else None
                        )
                        audit = evaluate_search_completion(
                            evaluator,
                            final_candidate_sha256=final_digest,
                            agent_id=spec.agent_id,
                            policy=self.config.search_policy,
                        )
                        search_completion[spec.agent_id] = audit
                        self._close_development_session(evaluator)
                    oracle = round_guidance_oracles.get(spec.agent_id)
                    if oracle is not None:
                        oracle.close()
                    del context
                    del cases

                portfolio = (
                    _contract_safe_screened_portfolio(
                        evaluator,
                        final=(outcome.candidate if outcome is not None else None),
                        representative_limit=(
                            self.config.development_screen_representative_limit
                        ),
                        audit=audit,
                        policy=self.config.search_policy,
                    )
                    if strong_development
                    else (
                        ((outcome.candidate, ("agent_nomination",)),)
                        if outcome is not None
                        else ()
                    )
                )
                for candidate, roles in portfolio:
                    digest = candidate.ref.bundle_sha256
                    existing_id = round_candidate_ids.get(digest)
                    if existing_id is not None:
                        portfolio_roles[existing_id] = tuple(
                            dict.fromkeys(
                                (*portfolio_roles[existing_id], *roles)
                            )
                        )
                        continue
                    try:
                        self._validate_candidate(
                            candidate.ref,
                            spec,
                            round_index,
                            incumbent.ref,
                        )
                    except (KeyboardInterrupt, GeneratorExit):
                        raise
                    except BaseException as exc:
                        failures.setdefault(
                            spec.agent_id, f"agent_failed:{type(exc).__name__}"
                        )
                        continue
                    round_candidate_ids[digest] = candidate.ref.candidate_id
                    outcomes.append(MacroAgentOutcome(candidate))
                    portfolio_roles[candidate.ref.candidate_id] = roles

            if self.development_executor is not None:
                self._merge_external_development(
                    round_index=round_index,
                    incumbent=incumbent,
                    results=external_development,
                    outcomes=outcomes,
                    failures=failures,
                    search_completion=search_completion,
                    portfolio_roles=portfolio_roles,
                    round_candidate_ids=round_candidate_ids,
                    development_case_ids=development_case_ids,
                )

            if development is not None:
                self._discard_prepared_cohort(
                    round_index, CohortKind.DEVELOPMENT
                )
            try:
                (
                    incumbent_evaluation,
                    finalists,
                ) = self._evaluate_common_cohort(
                    round_index,
                    outcomes,
                    incumbent,
                )
            finally:
                self._discard_prepared_cohort(round_index, CohortKind.SELECTION)
            guidance_evaluation = None
            if (
                round_guidance is not None
                and incumbent_evaluation is not None
            ):
                base_rows = (incumbent_evaluation, *finalists)
                base_evidence = {
                    row.candidate.candidate_id: (
                        row.batch.cohort_id,
                        (
                            row.objective_aggregates.generated
                            if row.objective_aggregates is not None
                            else row.aggregate
                        ),
                    )
                    for row in base_rows
                }
                guidance_evaluation = self._evaluate_guided_common_cohort(
                    boundary=round_guidance,
                    cohort_id=f"round-{round_index}-guided-hidden",
                    sample_count=(
                        round_guidance.config.official_round_case_count
                    ),
                    submissions=(
                        incumbent,
                        *(outcome.candidate for outcome in outcomes),
                    ),
                    base_evidence=base_evidence,
                )
            leaderboard_rows = (
                (incumbent_evaluation, *finalists)
                if incumbent_evaluation is not None
                else ()
            )
            leaderboard_roles = dict(portfolio_roles)
            if incumbent_evaluation is not None:
                leaderboard_roles.setdefault(
                    incumbent.ref.candidate_id,
                    ("incumbent",),
                )
            leaderboard = _search_leaderboard(
                leaderboard_rows,
                roles=leaderboard_roles,
                limit=self.config.search_policy.leaderboard_limit,
                incumbent_digest=incumbent.ref.bundle_sha256,
            )
            (
                selected,
                challenger_submission,
                promotion,
                promotion_published_metrics,
                promotion_challenger_aggregates,
                promotion_incumbent_aggregates,
                promotion_guardrail,
                promotion_decision,
                promotion_reports,
            ) = self._rank_common_cohort(
                round_index=round_index,
                incumbent=incumbent,
                outcomes=outcomes,
                incumbent_evaluation=incumbent_evaluation,
                finalists=finalists,
                guidance_evaluation=guidance_evaluation,
            )
            reason = (
                promotion.reason
                if promotion is not None
                else "no_eligible_finalist"
            )
            handoff = self._handoff(
                round_index=round_index,
                incumbent=incumbent.ref,
                selected=selected.ref,
                challenger=(
                    challenger_submission.ref
                    if challenger_submission is not None
                    else None
                ),
                promotion=promotion,
                reason=reason,
                previous_handoff=prior_handoff,
                perturbed_guardrail=promotion_guardrail,
            )
            result = MacroRoundResult(
                round_index=round_index,
                development_case_ids=tuple(development_case_ids),
                candidate_refs=tuple(outcome.candidate.ref for outcome in outcomes),
                agent_failures=MappingProxyType(dict(failures)),
                finalist_evaluations=finalists,
                challenger=(
                    challenger_submission.ref
                    if challenger_submission is not None
                    else None
                ),
                promotion=promotion,
                selected=selected.ref,
                handoff=handoff,
                promotion_published_metrics=promotion_published_metrics,
                candidate_portfolio_roles=MappingProxyType(
                    dict(portfolio_roles)
                ),
                promotion_challenger_objective_aggregates=(
                    promotion_challenger_aggregates
                ),
                promotion_incumbent_objective_aggregates=(
                    promotion_incumbent_aggregates
                ),
                promotion_perturbed_guardrail=promotion_guardrail,
                leaderboard=leaderboard,
                search_completion=MappingProxyType(dict(search_completion)),
                guidance_evaluation=guidance_evaluation,
                promotion_reports=promotion_reports,
                promotion_decision=promotion_decision,
            )
            round_results.append(result)
            incumbent = selected
            prior_handoff = handoff
            prior_leaderboard = leaderboard
            share_after = self.config.search_policy.cross_share_after_round
            shared_candidates = (
                tuple(
                    entry.candidate
                    for entry in leaderboard
                    if entry.eligible
                )
                if share_after and round_index >= share_after
                else ()
            )

        current_plugin_identity = (
            self.plugin.plugin_id,
            self.plugin.generator.generator_id,
            self.plugin.generator.artifact_sha256,
            self.plugin.objective.matcher.contract_id,
            self._published_metric_names(),
        )
        if current_plugin_identity != plugin_identity:
            raise RuntimeError("macro-loop plugin identity changed during the run")
        terminal_guidance = self._train_guidance_boundary(
            source_round_index=self.config.round_count,
            target_round_index=self.config.round_count + 1,
            incumbent=incumbent,
        )
        try:
            terminal = self._terminal_evaluation(incumbent)
        finally:
            self._discard_prepared_cohort(
                self.config.round_count + 1,
                CohortKind.TERMINAL,
            )
        if terminal_guidance is not None:
            cold_base_aggregate = (
                terminal.cold_start_objective_aggregates.generated
                if terminal.cold_start_objective_aggregates is not None
                else (
                    aggregate_score(
                        terminal.cold_start_score,
                        cvar_tail_fraction=self.config.cvar_tail_fraction,
                    )
                    if terminal.cold_start_score.eligible
                    else None
                )
            )
            selected_base_aggregate = (
                terminal.selected_objective_aggregates.generated
                if terminal.selected_objective_aggregates is not None
                else (
                    aggregate_score(
                        terminal.selected_score,
                        cvar_tail_fraction=self.config.cvar_tail_fraction,
                    )
                    if terminal.selected_score.eligible
                    else None
                )
            )
            terminal_submissions = (
                (self.cold_start,)
                if incumbent.ref.bundle_sha256
                == self.cold_start.ref.bundle_sha256
                else (self.cold_start, incumbent)
            )
            terminal_base_evidence = {
                self.cold_start.ref.candidate_id: (
                    terminal.cohort_id,
                    cold_base_aggregate,
                )
            }
            if len(terminal_submissions) == 2:
                terminal_base_evidence[incumbent.ref.candidate_id] = (
                    terminal.cohort_id,
                    selected_base_aggregate,
                )
            terminal = replace(
                terminal,
                guidance_evaluation=self._evaluate_guided_common_cohort(
                    boundary=terminal_guidance,
                    cohort_id="terminal-final-solver-guided-hidden",
                    sample_count=(
                        terminal_guidance.config.official_terminal_case_count
                    ),
                    submissions=terminal_submissions,
                    base_evidence=terminal_base_evidence,
                ),
            )
        terminal_plugin_identity = (
            self.plugin.plugin_id,
            self.plugin.generator.generator_id,
            self.plugin.generator.artifact_sha256,
            self.plugin.objective.matcher.contract_id,
            self._published_metric_names(),
        )
        if terminal_plugin_identity != plugin_identity:
            raise RuntimeError("macro-loop plugin identity changed during terminal")
        return MacroLoopResult(
            name=self.config.name,
            plugin_id=plugin_identity[0],
            generator_id=plugin_identity[1],
            generator_sha256=plugin_identity[2],
            macro_contract_id=plugin_identity[3],
            execution_assurance=boundary.assurance,
            formal_claims_allowed=boundary.formal_claims_allowed,
            cold_start=self.cold_start.ref,
            selected=incumbent.ref,
            rounds=tuple(round_results),
            terminal=terminal,
            public_result_kind=self.plugin.public_result_kind,
            public_result_task=self.plugin.public_result_task,
            published_metric_names=plugin_identity[4],
            paired_perturbations=self.config.paired_perturbations,
            paired_objective_policy=self.config.paired_objective_policy,
            promotion_program_id=self.config.promotion_program.program_id,
            sampling_attempts=tuple(self._sampling_attempts),
        )

    def _validate_runtime(self) -> None:
        if (
            self.config.paired_perturbations
            and getattr(self.plugin.generator, "supports_perturbations", False)
            is not True
        ):
            raise ValueError("case source does not support paired perturbations")
        if (
            self.development_executor is not None
            and not self.config.development_screen_root_stages
        ):
            raise ValueError(
                "external development workers require strong-development stages"
            )
        required = {spec.agent_id for spec in self.config.agents}
        if self.development_executor is None:
            missing = required - set(self.agents)
            if missing:
                raise ValueError(
                    f"no Agent program was provided for: {sorted(missing)}"
                )
        _require_integer(
            "cold_start round_index", self.cold_start.ref.round_index, minimum=0
        )
        if self.cold_start.ref.round_index != 0:
            raise ValueError("fresh-study cold_start must have round_index 0")
        if self.cold_start.ref.parent_sha256 is not None:
            raise ValueError("fresh-study cold_start cannot have a parent")

    def _precompute_cohorts(self) -> None:
        """Freeze formal request/seed plans without manufacturing answers."""

        if self._cohort_plans or self._prepared_cohorts:
            raise RuntimeError("controller cohort plans were already prepared")

        plans: dict[tuple[int, CohortKind], tuple[str, int, str]] = {}
        pool_count = (
            self.config.development_case_count
            * self.config.development_pool_multiplier
        )
        for round_index in range(1, self.config.round_count + 1):
            development_id = (
                f"round-{round_index}-development"
                if self.config.development_pool_multiplier == 1
                else f"round-{round_index}-development-pool"
            )
            requests = [
                (
                    CohortKind.SELECTION,
                    self.config.selection_case_count,
                    f"round-{round_index}-selection",
                    _CANDIDATE_EVALUATION_QUERY_PROFILE,
                ),
            ]
            if not self.config.development_screen_root_stages:
                requests.insert(
                    0,
                    (
                        CohortKind.DEVELOPMENT,
                        pool_count,
                        development_id,
                        _DEVELOPMENT_RELEASE_QUERY_PROFILE,
                    ),
                )
            for cohort_kind, count, cohort_id, profile in requests:
                plans[(round_index, cohort_kind)] = (
                    cohort_id, count, profile
                )

        terminal_round = self.config.round_count + 1
        plans[(terminal_round, CohortKind.TERMINAL)] = (
            "terminal-holdout",
            self.config.terminal_case_count,
            _CANDIDATE_EVALUATION_QUERY_PROFILE,
        )
        self._cohort_plans = plans

    def _prepared_cohort(
        self,
        round_index: int,
        cohort_kind: CohortKind,
    ) -> PreparedMacroCohort:
        key = (round_index, cohort_kind)
        prepared = self._prepared_cohorts.get(key)
        if prepared is not None:
            return prepared
        try:
            cohort_id, count, profile = self._cohort_plans[key]
        except KeyError as exc:
            raise RuntimeError(
                "requested controller cohort was not precomputed"
            ) from exc
        records = self._sample(round_index, cohort_kind, count)
        prepared = self._prepare_cohort(
            cohort_id=cohort_id,
            records=records,
            profile=profile,
        )
        if key in self._prepared_cohorts:
            prepared.discard()
            raise RuntimeError("controller cohort was prepared more than once")
        self._prepared_cohorts[key] = prepared
        return prepared

    def _discard_prepared_cohort(
        self, round_index: int, cohort_kind: CohortKind
    ) -> None:
        key = (round_index, cohort_kind)
        prepared = self._prepared_cohorts.get(key)
        if prepared is None:
            return
        prepared.discard()
        if self._prepared_cohorts.get(key) is prepared:
            del self._prepared_cohorts[key]

    def _drain_prepared_cohorts(self) -> None:
        failures: list[BaseException] = []
        for key, prepared in tuple(self._prepared_cohorts.items()):
            try:
                prepared.discard()
            except BaseException as exc:
                failures.append(exc)
            else:
                if self._prepared_cohorts.get(key) is prepared:
                    del self._prepared_cohorts[key]
        if failures:
            first = failures[0]
            for failure in failures[1:]:
                first.add_note(
                    "additional prepared-cohort release failure: "
                    f"{type(failure).__name__}: {failure}"
                )
            raise first

    def _close_development_session(self, session: DevelopmentSession) -> None:
        """Close one owned session and forget it only after successful cleanup."""

        session.close()
        self._active_development_sessions.discard(session)

    def _drain_development_sessions(self) -> None:
        failures: list[BaseException] = []
        for session in tuple(self._active_development_sessions):
            try:
                self._close_development_session(session)
            except BaseException as exc:
                failures.append(exc)
        if failures:
            first = failures[0]
            for failure in failures[1:]:
                first.add_note(
                    "additional development-session release failure: "
                    f"{type(failure).__name__}: {failure}"
                )
            raise first

    def _drain_owned_resources(self) -> None:
        """Drain every runner-owned session and cohort even after failures."""

        failures: list[tuple[str, BaseException]] = []
        for label, drain in (
            ("development-session", self._drain_development_sessions),
            ("prepared-cohort", self._drain_prepared_cohorts),
        ):
            try:
                drain()
            except BaseException as exc:
                failures.append((label, exc))
        if failures:
            _first_label, first = failures[0]
            for label, failure in failures[1:]:
                first.add_note(
                    f"additional macro-loop {label} cleanup failure: "
                    f"{type(failure).__name__}: {failure}"
                )
            raise first

    def run_agent_development(
        self,
        *,
        round_index: int,
        agent: MacroAgentSpec,
        incumbent: CandidateSubmission,
        prior_handoff: RoundHandoff | None = None,
        prior_leaderboard: tuple[SearchLeaderboardEntry, ...] = (),
        shared_candidates: tuple[CandidateRef, ...] = (),
        guidance_oracle: DevelopmentGuidanceOracle | None = None,
    ) -> MacroAgentDevelopmentResult:
        """Run exactly one strong-development phase in a worker-owned runner."""

        if self.development_executor is not None:
            raise RuntimeError("a development worker cannot delegate recursively")
        if not self.config.development_screen_root_stages:
            raise RuntimeError(
                "process Agent workers require fresh strong-development sessions"
            )
        if agent not in self.config.agents:
            raise ValueError("development worker Agent is outside its config")
        self._validate_runtime()
        try:
            try:
                result = self._run_agent_development(
                    round_index=round_index,
                    agent=agent,
                    incumbent=incumbent,
                    prior_handoff=prior_handoff,
                    prior_leaderboard=prior_leaderboard,
                    shared_candidates=shared_candidates,
                    guidance_oracle=guidance_oracle,
                )
            except BaseException as exc:
                try:
                    self._drain_owned_resources()
                except BaseException as cleanup_exc:
                    exc.add_note(
                        "Agent development worker cleanup also failed: "
                        f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                    )
                raise
            self._drain_owned_resources()
            return result
        finally:
            if guidance_oracle is not None:
                guidance_oracle.close()

    def _run_agent_development(
        self,
        *,
        round_index: int,
        agent: MacroAgentSpec,
        incumbent: CandidateSubmission,
        prior_handoff: RoundHandoff | None,
        prior_leaderboard: tuple[SearchLeaderboardEntry, ...],
        shared_candidates: tuple[CandidateRef, ...],
        guidance_oracle: DevelopmentGuidanceOracle | None,
    ) -> MacroAgentDevelopmentResult:
        sampled_start = len(self._sampled_case_id_order)
        attempts_start = len(self._sampling_attempts)
        evaluator = self._strong_development_session(
            round_index=round_index,
            agent=agent,
            incumbent=incumbent,
        )
        initial_root_count = (
            self.config.development_case_count // 2
            if self.config.paired_perturbations
            else self.config.development_case_count
        )
        try:
            release = evaluator.release(initial_root_count)
        except BaseException as exc:
            try:
                self._close_development_session(evaluator)
            except BaseException as cleanup_exc:
                exc.add_note(
                    "strong-development session cleanup also failed: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
            raise
        cases = release.cases
        development_case_ids = tuple(
            case.case_id
            for case in evaluator._controller_release_cases()
        )
        del release
        context = MacroAgentContext(
            round_index=round_index,
            agent=agent,
            prompt=self._agent_prompt(agent),
            cases=cases,
            evaluator=evaluator,
            prior_leaderboard=prior_leaderboard,
            shared_candidates=shared_candidates,
            search_policy=_open_search_policy(
                self.config.search_policy
            ),
            incumbent=incumbent.ref,
            prior_handoff=prior_handoff,
            public_problem_bundle=self.plugin.public_problem_bundle,
            public_session_factory=self.plugin.public_session_factory,
            public_session_methods=self.plugin.public_session_methods,
            development_solution_session_factory=(
                self.plugin.development_solution_session_factory
            ),
            development_solution_session_methods=(
                self.plugin.development_solution_session_methods
            ),
            guidance_oracle=guidance_oracle,
        )
        outcome: MacroAgentOutcome | None = None
        failure: str | None = None
        audit: SearchCompletionAudit | None = None
        try:
            outcome = self.agents[agent.agent_id].run(context)
        except (KeyboardInterrupt, GeneratorExit):
            raise
        except BaseException as exc:
            failure = f"agent_failed:{type(exc).__name__}"
        finally:
            final_digest = (
                outcome.candidate.ref.bundle_sha256
                if outcome is not None
                else None
            )
            audit = evaluate_search_completion(
                evaluator,
                final_candidate_sha256=final_digest,
                agent_id=agent.agent_id,
                policy=self.config.search_policy,
            )
            self._close_development_session(evaluator)
            del context
            del cases

        portfolio = _contract_safe_screened_portfolio(
            evaluator,
            final=(outcome.candidate if outcome is not None else None),
            representative_limit=(
                self.config.development_screen_representative_limit
            ),
            audit=audit,
            policy=self.config.search_policy,
        )
        return MacroAgentDevelopmentResult(
            agent_id=agent.agent_id,
            portfolio=portfolio,
            development_case_ids=development_case_ids,
            sampled_case_ids=tuple(
                self._sampled_case_id_order[sampled_start:]
            ),
            sampling_attempts=tuple(self._sampling_attempts[attempts_start:]),
            search_completion=audit,
            failure=failure,
        )

    def _merge_external_development(
        self,
        *,
        round_index: int,
        incumbent: CandidateSubmission,
        results: tuple[MacroAgentDevelopmentResult, ...],
        outcomes: list[MacroAgentOutcome],
        failures: dict[str, str],
        search_completion: dict[str, SearchCompletionAudit],
        portfolio_roles: dict[str, tuple[str, ...]],
        round_candidate_ids: dict[str, str],
        development_case_ids: list[str],
    ) -> None:
        rows = tuple(results)
        by_agent = {row.agent_id: row for row in rows}
        expected = {agent.agent_id for agent in self.config.agents}
        if len(by_agent) != len(rows) or set(by_agent) != expected:
            raise RuntimeError(
                "development executor must return exactly one result per Agent"
            )

        for agent in self.config.agents:
            row = by_agent[agent.agent_id]
            if len(row.portfolio) > 1:
                raise RuntimeError(
                    "development worker returned multiple official nominations"
                )
            sampled_ids = tuple(row.sampled_case_ids)
            sampled_set = set(sampled_ids)
            duplicates = sampled_set & self._seen_case_ids
            if duplicates:
                raise RuntimeError(
                    "development workers reused controller case ids: "
                    f"{sorted(duplicates)}"
                )
            if not set(row.development_case_ids).issubset(sampled_set):
                raise RuntimeError(
                    "development worker initial cases are absent from its sample audit"
                )
            for attempt in row.sampling_attempts:
                if attempt.round_index != round_index or attempt.cohort not in {
                    CohortKind.DEVELOPMENT,
                    CohortKind.DEVELOPMENT_SCREEN,
                }:
                    raise RuntimeError(
                        "development worker sampling audit changed round or cohort"
                    )
            self._seen_case_ids.update(sampled_set)
            self._sampled_case_id_order.extend(sampled_ids)
            self._sampling_attempts.extend(row.sampling_attempts)
            development_case_ids.extend(row.development_case_ids)
            if row.failure is not None:
                failures[agent.agent_id] = row.failure
            if row.search_completion is not None:
                search_completion[agent.agent_id] = row.search_completion

            for candidate, roles in row.portfolio:
                digest = candidate.ref.bundle_sha256
                existing_id = round_candidate_ids.get(digest)
                if existing_id is not None:
                    portfolio_roles[existing_id] = tuple(
                        dict.fromkeys((*portfolio_roles[existing_id], *roles))
                    )
                    continue
                try:
                    self._validate_candidate(
                        candidate.ref,
                        agent,
                        round_index,
                        incumbent.ref,
                    )
                except (KeyboardInterrupt, GeneratorExit):
                    raise
                except BaseException as exc:
                    failures.setdefault(
                        agent.agent_id, f"agent_failed:{type(exc).__name__}"
                    )
                    continue
                round_candidate_ids[digest] = candidate.ref.candidate_id
                outcomes.append(MacroAgentOutcome(candidate))
                portfolio_roles[candidate.ref.candidate_id] = tuple(roles)

    def _strong_development_session(
        self,
        *,
        round_index: int,
        agent: MacroAgentSpec,
        incumbent: CandidateSubmission,
    ) -> DevelopmentSession:
        agent_token = hashlib.sha256(
            agent.agent_id.encode("utf-8")
        ).hexdigest()[:12]

        def fresh_cohort(
            request: DevelopmentDrawRequest,
        ) -> PreparedMacroCohort:
            cohort_kind = (
                CohortKind.DEVELOPMENT
                if request.kind is DevelopmentDrawKind.RELEASE
                else CohortKind.DEVELOPMENT_SCREEN
            )
            members_per_root = 2 if self.config.paired_perturbations else 1
            namespace = (
                f"agent-{agent_token}"
                f"\0{request.kind.value}"
                f"\0draw-{request.draw_index}"
            )
            records = self._sample(
                round_index,
                cohort_kind,
                request.root_count * members_per_root,
                seed_namespace=namespace,
            )
            return self._prepare_cohort(
                cohort_id=(
                    f"round-{round_index}-agent-{agent_token}-"
                    f"development-{request.kind.value}-"
                    f"draw-{request.draw_index}"
                ),
                records=records,
                profile=(
                    _DEVELOPMENT_RELEASE_QUERY_PROFILE
                    if request.kind is DevelopmentDrawKind.RELEASE
                    else _CANDIDATE_EVALUATION_QUERY_PROFILE
                ),
            )

        session = DevelopmentSession(
            incumbent=incumbent,
            cohort_factory=fresh_cohort,
            screen_root_stages=(
                self.config.development_screen_root_stages
            ),
            screen_root_credit_limit=(
                self.config.development_screen_root_credit_limit
            ),
            release_root_credit_limit=(
                self.config.development_release_root_limit
            ),
            compute_budget=self.config.candidate_compute_budget,
            query_seconds_per_point_limit=(
                self.config.candidate_query_seconds_per_point_limit
            ),
            safe_metric_names=self._agent_visible_metrics(),
            cvar_tail_fraction=self.config.cvar_tail_fraction,
            confidence_z=self.config.development_screen_confidence_z,
            continuation_max_mean_ratio=(
                self.config.development_screen_continuation_max_mean_ratio
            ),
            paired_objective_policy=self.config.paired_objective_policy,
            public_session_factory=self.plugin.public_session_factory,
            public_session_methods=self.plugin.public_session_methods,
            open_search=True,
        )
        self._active_development_sessions.add(session)
        return session

    def _development_cohort(
        self,
        round_index: int,
        incumbent: CandidateSubmission,
    ) -> PreparedMacroCohort:
        mining = self._prepared_cohort(round_index, CohortKind.DEVELOPMENT)
        if self.config.development_pool_multiplier == 1:
            return mining

        batch = self._controller_evaluator(mining).evaluate(
            incumbent.ref.candidate_id,
            incumbent.program,
        )
        error_by_id = {
            score.case_id: (
                self.config.invalid_error
                if score.primary_error is None
                else float(score.primary_error)
            )
            for score in batch.scores
        }
        status_by_id = {
            score.case_id: score.status for score in batch.scores
        }

        if self.config.paired_perturbations:
            groups = tuple(
                zip(batch.case_ids[::2], batch.case_ids[1::2])
            )
            selected_group_count = self.config.development_case_count // 2
        else:
            groups = tuple((case_id,) for case_id in batch.case_ids)
            selected_group_count = self.config.development_case_count

        ordered_groups = sorted(
            groups,
            key=lambda group: (
                all(status_by_id[case_id] == "valid" for case_id in group),
                -max(error_by_id[case_id] for case_id in group),
                -fmean(error_by_id[case_id] for case_id in group),
                group[0],
            ),
        )
        selected_ids = tuple(
            case_id
            for group in ordered_groups[:selected_group_count]
            for case_id in group
        )
        return mining._subset(
            cohort_id=f"round-{round_index}-development",
            case_ids=selected_ids,
        )

    def _train_guidance_boundary(
        self,
        *,
        source_round_index: int,
        target_round_index: int,
        incumbent: CandidateSubmission,
    ) -> FrozenGuidanceBoundary | None:
        """Train once from a disjoint base pool and freeze for the target round."""

        lifecycle = self._guidance_lifecycle
        protocol = self.config.guidance_protocol
        if lifecycle is None or protocol is None:
            return None
        if target_round_index != source_round_index + 1:
            raise ValueError("guidance target round must follow source round")

        seed_label = (
            "macro-guidance-feedback-v1"
            f"\0{self.config.seed_base}"
            f"\0{source_round_index}"
            f"\0{target_round_index}"
            f"\0{incumbent.ref.bundle_sha256}"
        )
        feedback_seed = int.from_bytes(
            hashlib.sha256(seed_label.encode("utf-8")).digest()[:8],
            "big",
        ) & ((1 << 63) - 1)
        records = self._sample_response(
            SampleRequest(
                round_index=source_round_index,
                cohort=CohortKind.GUIDANCE_FEEDBACK,
                seed=feedback_seed,
                count=protocol.reward_feedback_case_count,
                kind=GenerationKind.BASE,
            )
        )
        ids = tuple(record.case.case_id for record in records)
        if len(ids) != len(set(ids)):
            self._release_records(records)
            raise ValueError("guidance feedback case ids must be unique")
        duplicates = set(ids) & self._seen_case_ids
        if duplicates:
            self._release_records(records)
            raise ValueError(
                "guidance feedback reused controller case ids: "
                f"{sorted(duplicates)}"
            )
        self._seen_case_ids.update(ids)
        self._sampled_case_id_order.extend(ids)

        cohort = self._prepare_cohort(
            cohort_id=(
                f"guidance-feedback-source-{source_round_index}-"
                f"target-{target_round_index}"
            ),
            records=records,
            profile=_CANDIDATE_EVALUATION_QUERY_PROFILE,
        )
        try:
            batch = self._controller_evaluator(cohort).evaluate(
                incumbent.ref.candidate_id,
                incumbent.fresh_program(),
            )
            public_cases = cohort._controller_cases()
            rows = tuple(
                GuidanceRewardRow(
                    case,
                    (
                        self.config.invalid_error
                        if score.primary_error is None
                        else float(score.primary_error)
                    ),
                    score.status,
                )
                for case, score in zip(
                    public_cases,
                    batch.scores,
                    strict=True,
                )
            )
            matcher_label = (
                "macro-guidance-primary-error-v2"
                f"\0query_profile={_CANDIDATE_EVALUATION_QUERY_PROFILE}"
                f"\0{self.plugin.objective.matcher.contract_id}"
            )
            feedback = GuidanceRewardFeedback(
                source_round_index=source_round_index,
                target_round_index=target_round_index,
                target_solver_sha256=incumbent.ref.bundle_sha256,
                base_generator_sha256=self.plugin.generator.artifact_sha256,
                reward_matcher_definition_sha256=hashlib.sha256(
                    matcher_label.encode("utf-8")
                ).hexdigest(),
                rows=rows,
            )
            training_seed_label = (
                "macro-guidance-training-v1"
                f"\0{self.config.seed_base}"
                f"\0{source_round_index}"
                f"\0{target_round_index}"
                f"\0{incumbent.ref.bundle_sha256}"
            )
            training_seed = int.from_bytes(
                hashlib.sha256(training_seed_label.encode("utf-8")).digest()[:8],
                "big",
            ) & ((1 << 63) - 1)
            return lifecycle.train_boundary(
                feedback,
                controller_training_seed=training_seed,
            )
        finally:
            cohort.discard()

    def _evaluate_common_cohort(
        self,
        round_index: int,
        outcomes: Sequence[MacroAgentOutcome],
        incumbent: CandidateSubmission,
    ) -> tuple[
        FinalistEvaluation | None,
        tuple[FinalistEvaluation, ...],
    ]:
        """Evaluate every nomination and the incumbent on one hidden panel.

        The query contains common physical scoring coordinates. It does not
        prescribe a candidate's internal discretization: each ROM may choose
        its own dimensions/grid and must interpolate or otherwise answer at the
        requested points within the candidate wall-clock budget.
        """

        if not outcomes:
            return None, ()
        submissions = (
            incumbent,
            *(outcome.candidate for outcome in outcomes),
        )
        batches = self._evaluate_official_common_cohort(
            round_index=round_index,
            cohort_kind=CohortKind.SELECTION,
            submissions=submissions,
        )
        raw_scores = tuple(
            self._selection_score(submission.ref, batch)
            for submission, batch in zip(submissions, batches)
        )
        incumbent_raw_score = raw_scores[0]

        def evaluation(
            submission: CandidateSubmission,
            batch: MacroBatchEvaluation,
            raw_score: MacroCandidateScore,
            *,
            guardrail: PerturbedGuardrailSummary | None,
        ) -> FinalistEvaluation:
            score = self._objective_score(raw_score)
            return FinalistEvaluation(
                candidate=submission.ref,
                batch=batch,
                score=score,
                aggregate=(
                    aggregate_score(
                        score,
                        cvar_tail_fraction=self.config.cvar_tail_fraction,
                    )
                    if score.eligible
                    else None
                ),
                published_metrics=self._published_metric_summary(batch),
                objective_aggregates=self._objective_aggregates(raw_score),
                perturbed_guardrail=guardrail,
            )

        incumbent_evaluation = evaluation(
            incumbent,
            batches[0],
            incumbent_raw_score,
            guardrail=None,
        )
        guardrail_enabled = self._perturbed_guardrail_enabled()
        finalists = tuple(
            evaluation(
                submission,
                batch,
                raw_score,
                guardrail=(
                    self._guardrail_summary(raw_score, incumbent_raw_score)
                    if guardrail_enabled
                    else None
                ),
            )
            for submission, batch, raw_score in zip(
                submissions[1:],
                batches[1:],
                raw_scores[1:],
            )
        )
        return incumbent_evaluation, finalists

    def _evaluate_guided_common_cohort(
        self,
        *,
        boundary: FrozenGuidanceBoundary,
        cohort_id: str,
        sample_count: int,
        submissions: Sequence[CandidateSubmission],
        base_evidence: Mapping[
            str, tuple[str, RobustAggregate | None]
        ],
    ) -> GuidanceCohortEvaluation:
        """Evaluate all submissions on one immutable hidden guided draw.

        Guide training/generation are campaign work. Candidate solve/query
        timing remains inside each returned ``MacroBatchEvaluation`` and is
        therefore never charged to the guidance generation clock.
        Each candidate identity receives one fresh runtime for this official
        guided cohort.
        """

        rows = tuple(submissions)
        if not rows:
            raise ValueError("guided official evaluation needs candidates")
        candidate_ids = tuple(row.ref.candidate_id for row in rows)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("guided official candidates must be unique")
        if set(candidate_ids) != set(base_evidence):
            raise ValueError("guided candidates and base evidence must agree")

        resolved = boundary.official_common_cohort(
            cohort_id,
            sample_count=sample_count,
        )
        case_ids = tuple(record.case.case_id for record in resolved.records)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("official guided case ids must be unique")
        duplicates = set(case_ids) & self._seen_case_ids
        if duplicates:
            raise ValueError(
                "official guided cohort reused controller case ids: "
                f"{sorted(duplicates)}"
            )
        self._seen_case_ids.update(case_ids)
        self._sampled_case_id_order.extend(case_ids)

        release_guided = getattr(
            self.guidance_backend,
            "release_guided_cases",
            None,
        )
        if release_guided is not None and not callable(release_guided):
            raise TypeError("release_guided_cases must be callable or absent")
        cohort = self._prepare_cohort(
            cohort_id=cohort_id,
            records=resolved.records,
            profile=_CANDIDATE_EVALUATION_QUERY_PROFILE,
            release_cases_override=release_guided,
        )
        try:
            batches = self._controller_evaluator(cohort).evaluate_many(
                tuple(
                    (submission.ref.candidate_id, submission.fresh_program())
                    for submission in rows
                )
            )
            candidate_results: list[GuidanceCandidateEvaluation] = []
            for submission, batch in zip(rows, batches, strict=True):
                guided_score = self._selection_score(submission.ref, batch)
                guided_aggregate = (
                    aggregate_score(
                        guided_score,
                        cvar_tail_fraction=self.config.cvar_tail_fraction,
                    )
                    if guided_score.eligible
                    else None
                )
                base_cohort_id, base_aggregate = base_evidence[
                    submission.ref.candidate_id
                ]
                separated = None
                if base_aggregate is not None and guided_aggregate is not None:
                    separated = SeparatedGuidanceMetrics(
                        average_base=CohortMetrics(
                            cohort_id=base_cohort_id,
                            kind=GuidanceCohortKind.AVERAGE_BASE,
                            generator_or_guidance_sha256=(
                                self.plugin.generator.artifact_sha256
                            ),
                            values={
                                "primary_error_mean": base_aggregate.mean,
                                "primary_error_cvar": base_aggregate.cvar,
                                "primary_error_max": base_aggregate.max,
                            },
                        ),
                        guided_adversarial=CohortMetrics(
                            cohort_id=cohort_id,
                            kind=GuidanceCohortKind.GUIDED_ADVERSARIAL,
                            generator_or_guidance_sha256=(
                                boundary.snapshot.snapshot_sha256
                            ),
                            values={
                                "primary_error_mean": guided_aggregate.mean,
                                "primary_error_cvar": guided_aggregate.cvar,
                                "primary_error_max": guided_aggregate.max,
                            },
                        ),
                    )
                candidate_results.append(
                    GuidanceCandidateEvaluation(
                        candidate=submission.ref,
                        guided_batch=batch,
                        guided_score=guided_score,
                        guided_aggregate=guided_aggregate,
                        separated_metrics=separated,
                    )
                )
            cohort_sha256 = cohort.cohort_sha256
        finally:
            cohort.discard()

        return GuidanceCohortEvaluation(
            snapshot=boundary.snapshot,
            cohort_id=cohort_id,
            cohort_sha256=cohort_sha256,
            campaign_guided_generation_wall_seconds=(
                resolved.public.batch.campaign_guided_generation_wall_seconds
            ),
            candidates=tuple(candidate_results),
        )

    def _promotion_reports(
        self,
        *,
        base_rows: Sequence[FinalistEvaluation],
        guidance_evaluation: GuidanceCohortEvaluation | None,
    ) -> tuple[PromotionCandidateReport, ...]:
        """Build the complete, named-distribution input to PromotionProgram."""

        guided_by_digest = {
            row.candidate.bundle_sha256: row
            for row in (
                ()
                if guidance_evaluation is None
                else guidance_evaluation.candidates
            )
        }
        reports = []
        for row in base_rows:
            distributions: dict[str, PromotionMetricSummary] = {}
            objectives = row.objective_aggregates
            if objectives is not None:
                distributions["base"] = _promotion_metric(
                    objectives.generated,
                    len(row.score.errors),
                )
                distributions["base_perturbed"] = _promotion_metric(
                    objectives.perturbed,
                    len(row.score.errors),
                )
            elif row.aggregate is not None:
                distributions["base"] = _promotion_metric(
                    row.aggregate,
                    len(row.score.errors),
                )

            failures = list(row.score.gate_failures)
            technically_eligible = row.score.eligible and row.aggregate is not None
            guided = guided_by_digest.get(row.candidate.bundle_sha256)
            if guidance_evaluation is not None:
                if guided is None or guided.guided_aggregate is None:
                    technically_eligible = False
                    failures.append("guided_evaluation_failed")
                else:
                    distributions["guided"] = _promotion_metric(
                        guided.guided_aggregate,
                        len(guided.guided_score.errors),
                    )

            guardrails = {}
            if row.perturbed_guardrail is not None:
                guardrails["base_perturbed_non_regression"] = (
                    row.perturbed_guardrail.passed
                )
            reports.append(
                PromotionCandidateReport(
                    candidate_id=row.candidate.candidate_id,
                    candidate_digest=row.candidate.bundle_sha256,
                    technically_eligible=technically_eligible,
                    distributions=distributions,
                    guardrails=guardrails,
                    failures=tuple(failures),
                )
            )
        return tuple(reports)

    def _rank_common_cohort(
        self,
        *,
        round_index: int,
        incumbent: CandidateSubmission,
        outcomes: Sequence[MacroAgentOutcome],
        incumbent_evaluation: FinalistEvaluation | None,
        finalists: Sequence[FinalistEvaluation],
        guidance_evaluation: GuidanceCohortEvaluation | None,
    ) -> tuple[
        CandidateSubmission,
        CandidateSubmission | None,
        PromotionResult | None,
        tuple[PairedPublishedMetricSummary, ...],
        PairedObjectiveAggregates | None,
        PairedObjectiveAggregates | None,
        PerturbedGuardrailSummary | None,
        PromotionDecision | None,
        tuple[PromotionCandidateReport, ...],
    ]:
        """Use the configured PromotionProgram after all official reports exist."""

        if incumbent_evaluation is None:
            return incumbent, None, None, (), None, None, None, None, ()

        base_rows = (incumbent_evaluation, *finalists)
        reports = self._promotion_reports(
            base_rows=base_rows,
            guidance_evaluation=guidance_evaluation,
        )
        decision = self.config.promotion_program.select(
            incumbent.ref.bundle_sha256,
            reports,
            PromotionContext(round_index=round_index),
        )
        validate_promotion_decision(
            incumbent.ref.bundle_sha256,
            reports,
            decision,
        )
        actually_promoted = decision.selected_digest != incumbent.ref.bundle_sha256

        submissions = {
            incumbent.ref.bundle_sha256: incumbent,
            **{
                outcome.candidate.ref.bundle_sha256: outcome.candidate
                for outcome in outcomes
            },
        }
        selected = submissions[decision.selected_digest]
        finalist_by_digest = {
            row.candidate.bundle_sha256: row for row in finalists
        }
        if actually_promoted:
            challenger_row = finalist_by_digest[decision.selected_digest]
        else:
            ranked_challengers = sorted(
                (
                    ranking
                    for ranking in decision.ranking
                    if ranking.candidate_digest in finalist_by_digest
                    and ranking.eligible
                    and ranking.rank is not None
                ),
                key=lambda ranking: ranking.rank,
            )
            challenger_row = (
                finalist_by_digest[ranked_challengers[0].candidate_digest]
                if ranked_challengers
                else None
            )

        if challenger_row is None:
            return (
                selected, None, None, (), None, None, None, decision, reports
            )
        challenger = submissions[challenger_row.candidate.bundle_sha256]

        comparisons: tuple[PairedComparison, ...] = ()
        if incumbent_evaluation.score.eligible:
            stage = EvidenceStage(
                case_count=len(challenger_row.score.case_ids),
                confidence_z=0.0,
                min_mean_improvement=0.0,
                min_win_rate=0.0,
                require_cvar_no_regression=False,
            )
            comparisons = (
                paired_compare(
                    challenger_row.score,
                    incumbent_evaluation.score,
                    stage,
                    cvar_tail_fraction=self.config.cvar_tail_fraction,
                    confidence_group_size=self._objective_confidence_group_size(),
                ),
            )

        promotion = PromotionResult(
            incumbent_artifact_id=incumbent.ref.candidate_id,
            incumbent_digest=incumbent.ref.bundle_sha256,
            selected_artifact_id=selected.ref.candidate_id,
            selected_digest=selected.ref.bundle_sha256,
            promoted=decision.promoted,
            reason=decision.reason,
            comparisons=comparisons,
        )
        published_metrics = (
            (
                self._paired_published_metric_summary(
                    challenger_row.batch,
                    incumbent_evaluation.batch,
                    case_count=len(challenger_row.batch.case_ids),
                ),
            )
            if comparisons
            else ()
        )
        return (
            selected,
            challenger,
            promotion,
            published_metrics,
            challenger_row.objective_aggregates,
            incumbent_evaluation.objective_aggregates,
            challenger_row.perturbed_guardrail,
            decision,
            reports,
        )

    def _terminal_evaluation(
        self,
        selected: CandidateSubmission,
    ) -> TerminalEvaluation:
        round_index = self.config.round_count + 1
        if selected.ref.bundle_sha256 == self.cold_start.ref.bundle_sha256:
            (cold_batch,) = self._evaluate_official_common_cohort(
                round_index=round_index,
                cohort_kind=CohortKind.TERMINAL,
                submissions=(self.cold_start,),
            )
            selected_batch = cold_batch
        else:
            cold_batch, selected_batch = (
                self._evaluate_official_common_cohort(
                    round_index=round_index,
                    cohort_kind=CohortKind.TERMINAL,
                    submissions=(self.cold_start, selected),
                )
            )
        cold_raw_score = self._selection_score(
            self.cold_start.ref,
            cold_batch,
        )
        selected_raw_score = (
            cold_raw_score
            if selected.ref.bundle_sha256 == self.cold_start.ref.bundle_sha256
            else self._selection_score(selected.ref, selected_batch)
        )
        cold_score = self._objective_score(cold_raw_score)
        selected_score = (
            cold_score
            if selected_raw_score is cold_raw_score
            else self._objective_score(selected_raw_score)
        )
        comparison: PairedComparison | None = None
        guardrail: PerturbedGuardrailSummary | None = None
        if (
            selected.ref.bundle_sha256 != self.cold_start.ref.bundle_sha256
            and cold_score.eligible
        ):
            terminal_case_count = self._objective_case_count(
                self.config.terminal_case_count
            )
            comparison = paired_compare(
                selected_score,
                cold_score,
                EvidenceStage(
                    case_count=terminal_case_count,
                    confidence_z=self.config.terminal_confidence_z,
                    min_mean_improvement=0.0,
                    min_win_rate=self.config.terminal_min_win_rate,
                    require_cvar_no_regression=(
                        self.config.terminal_require_cvar_no_regression
                    ),
                ),
                cvar_tail_fraction=self.config.cvar_tail_fraction,
                confidence_group_size=self._objective_confidence_group_size(),
            )
        if selected.ref.bundle_sha256 != self.cold_start.ref.bundle_sha256:
            guardrail = self._guardrail_summary(
                selected_raw_score,
                cold_raw_score,
            )
        cold_published = self._published_metric_summary(cold_batch)
        selected_published = (
            cold_published
            if selected_batch is cold_batch
            else self._published_metric_summary(selected_batch)
        )
        paired_published = (
            self._paired_published_metric_summary(
                selected_batch,
                cold_batch,
                case_count=self._objective_member_case_count(
                    comparison.stage.case_count
                ),
            )
            if comparison is not None
            else None
        )
        return TerminalEvaluation(
            cohort_id=cold_batch.cohort_id,
            cohort_sha256=cold_batch.cohort_sha256,
            cold_start=cold_batch,
            selected=selected_batch,
            cold_start_score=cold_score,
            selected_score=selected_score,
            comparison=comparison,
            cold_start_published_metrics=cold_published,
            selected_published_metrics=selected_published,
            paired_published_metrics=paired_published,
            cold_start_objective_aggregates=(
                self._objective_aggregates(cold_raw_score)
            ),
            selected_objective_aggregates=(
                self._objective_aggregates(selected_raw_score)
            ),
            perturbed_guardrail=guardrail,
        )

    def _selection_score(
        self,
        candidate: CandidateRef,
        batch: MacroBatchEvaluation,
    ) -> MacroCandidateScore:
        if batch.candidate_id != candidate.candidate_id:
            raise ValueError("macro batch candidate identity changed")
        valid = True
        errors: list[float] = []
        gate_failures: list[str] = []
        for row in batch.scores:
            if row.primary_error is None:
                valid = False
                errors.append(float(self.config.invalid_error))
            else:
                errors.append(float(row.primary_error))
            if row.status == "rejected":
                gate_failures.extend(
                    f"{row.case_id}:{failure}"
                    for failure in (row.failures or ("macro_gate_rejected",))
                )
            elif row.status == "invalid":
                valid = False
                gate_failures.extend(
                    f"{row.case_id}:{failure}"
                    for failure in (row.failures or ("invalid_prediction",))
                )
        query_limit = self.config.candidate_query_seconds_per_point_limit
        if query_limit is not None:
            timings = batch.case_timings
            if len(timings) != len(batch.scores):
                valid = False
                for row in batch.scores:
                    failure = f"{row.case_id}:candidate_two_phase_required"
                    if failure not in gate_failures:
                        gate_failures.append(failure)
            else:
                for timing in timings:
                    failure_code = None
                    if not timing.formal_timing_eligible:
                        failure_code = "candidate_two_phase_required"
                    elif (
                        self.config.candidate_compute_budget is not None
                        and timing.candidate_solve_wall_seconds
                        > self.config.candidate_compute_budget
                    ):
                        failure_code = "candidate_solve_budget_exceeded"
                    elif (
                        timing.candidate_query_wall_seconds_per_point is None
                        or timing.candidate_query_wall_seconds_per_point
                        > query_limit
                    ):
                        failure_code = "candidate_query_budget_exceeded"
                    if failure_code is not None:
                        valid = False
                        failure = f"{timing.case_id}:{failure_code}"
                        if failure not in gate_failures:
                            gate_failures.append(failure)
        return MacroCandidateScore(
            artifact_id=candidate.candidate_id,
            artifact_digest=candidate.bundle_sha256,
            target_sha256=batch.target_sha256,
            contract_id=batch.contract_id,
            case_ids=batch.case_ids,
            errors=tuple(errors),
            valid=valid,
            root_ids=batch.root_ids,
            gate_failures=tuple(gate_failures),
        )

    def _objective_reduces_pairs(self) -> bool:
        return self.config.paired_perturbations

    def _objective_score(
        self,
        raw_score: MacroCandidateScore,
    ) -> MacroCandidateScore:
        if not self._objective_reduces_pairs():
            return raw_score
        return paired_objective_score(
            raw_score,
            self.config.paired_objective_policy,
        )

    def _objective_case_count(self, member_case_count: int) -> int:
        if not self._objective_reduces_pairs():
            return member_case_count
        if member_case_count % 2:
            raise ValueError("paired objective evidence cannot split a root")
        return member_case_count // 2

    def _objective_member_case_count(self, objective_case_count: int) -> int:
        return (
            2 * objective_case_count
            if self._objective_reduces_pairs()
            else objective_case_count
        )


    def _objective_confidence_group_size(self) -> int:
        if self._objective_reduces_pairs():
            return 1
        return 2 if self.config.paired_perturbations else 1

    def _objective_aggregates(
        self,
        raw_score: MacroCandidateScore,
    ) -> PairedObjectiveAggregates | None:
        if not self.config.paired_perturbations:
            return None
        split = split_paired_score(raw_score)
        composite = paired_weighted_composite(
            raw_score,
            base_weight=self.config.paired_objective_policy.base_weight,
            perturbed_weight=(
                self.config.paired_objective_policy.perturbed_weight
            ),
        )
        return PairedObjectiveAggregates(
            generated=aggregate_score(
                split.base,
                cvar_tail_fraction=self.config.cvar_tail_fraction,
            ),
            perturbed=aggregate_score(
                split.perturbed,
                cvar_tail_fraction=self.config.cvar_tail_fraction,
            ),
            composite=aggregate_score(
                composite,
                cvar_tail_fraction=self.config.cvar_tail_fraction,
            ),
        )

    def _perturbed_guardrail_enabled(self) -> bool:
        return (
            self.config.paired_perturbations
            and self.config.paired_objective_policy.mode
            is PairedObjectiveMode.PRIMARY_WITH_PERTURBED_GUARDRAIL
        )

    def _guardrail_summary(
        self,
        candidate: MacroCandidateScore,
        incumbent: MacroCandidateScore,
    ) -> PerturbedGuardrailSummary | None:
        if not self._perturbed_guardrail_enabled():
            return None
        private = compare_perturbed_non_regression(
            candidate,
            incumbent,
            policy=self.config.paired_objective_policy,
        )
        failures = tuple(
            dict.fromkeys(
                _public_guardrail_failure_code(item)
                for item in private.failures
            )
        )
        return PerturbedGuardrailSummary(
            root_count=len(private.root_ids),
            candidate_mean=private.candidate_mean,
            incumbent_mean=private.incumbent_mean,
            mean_excess=private.mean_excess,
            standard_error=private.standard_error,
            confidence_lower=private.confidence_lower,
            confidence_z=private.confidence_z,
            practical_relative_margin=private.practical_relative_margin,
            passed=private.passed,
            failures=failures,
        )

    def _release_records(self, records: Sequence[CaseRecord]) -> None:
        release = getattr(self.plugin.generator, "release_cases", None)
        if not callable(release):
            return
        case_ids = tuple(
            dict.fromkeys(
                record.case.case_id
                for record in records
                if isinstance(record, CaseRecord)
                and isinstance(record.case.case_id, str)
                and record.case.case_id.strip()
            )
        )
        if case_ids:
            release(case_ids)

    def _sample(
        self,
        round_index: int,
        cohort: CohortKind,
        count: int,
        *,
        seed_namespace: str | None = None,
        request_settings: Mapping[str, object] | None = None,
    ) -> tuple[CaseRecord, ...]:
        _require_integer("sample round_index", round_index, minimum=0)
        _require_integer("sample count", count, minimum=1)
        if not isinstance(cohort, CohortKind):
            raise ValueError("sample cohort must be a CohortKind")
        if seed_namespace is not None and (
            not isinstance(seed_namespace, str) or not seed_namespace.strip()
        ):
            raise ValueError("sample seed_namespace must be nonempty or None")
        if request_settings is None:
            settings: Mapping[str, object] = {}
        elif not isinstance(request_settings, Mapping):
            raise TypeError("sample request_settings must be a mapping or None")
        else:
            settings = dict(request_settings)
        if self.config.paired_perturbations and count % 2:
            raise ValueError("paired sampling needs an even member count")
        if self.config.paired_perturbations:
            records = self._sample_paired_with_retries(
                round_index=round_index,
                cohort=cohort,
                root_count=count // 2,
                seed_namespace=seed_namespace,
                request_settings=settings,
            )
        else:
            seed = self._paired_sampling_attempt_seed(
                round_index,
                cohort,
                GenerationKind.BASE,
                0,
                seed_namespace=seed_namespace,
            )
            _require_integer("sample seed", seed, minimum=0)
            records = self._sample_response(SampleRequest(
                round_index=round_index,
                cohort=cohort,
                seed=seed,
                count=count,
                settings=settings,
                kind=GenerationKind.BASE,
            ))

        ids = tuple(record.case.case_id for record in records)
        batch_ids = set(ids)
        if len(batch_ids) != len(ids):
            self._release_records(records)
            raise ValueError("case ids must be unique within a controller cohort")
        duplicates = batch_ids & self._seen_case_ids
        if duplicates:
            self._release_records(records)
            raise ValueError(
                "case ids must be unique across controller cohorts: "
                f"{sorted(duplicates)}"
            )
        self._seen_case_ids.update(batch_ids)
        self._sampled_case_id_order.extend(ids)
        return records

    def _sample_paired_with_retries(
        self,
        *,
        round_index: int,
        cohort: CohortKind,
        root_count: int,
        seed_namespace: str | None = None,
        request_settings: Mapping[str, object] | None = None,
    ) -> tuple[CaseRecord, ...]:
        maximum_attempts = self.config.paired_sampling_max_attempts
        attempts: list[SamplingAttemptAudit] = []
        for attempt_index in range(maximum_attempts):
            base_seed = self._paired_sampling_attempt_seed(
                round_index,
                cohort,
                GenerationKind.BASE,
                attempt_index,
                seed_namespace=seed_namespace,
            )
            perturbed_seed = self._paired_sampling_attempt_seed(
                round_index,
                cohort,
                GenerationKind.PERTURBED,
                attempt_index,
                seed_namespace=seed_namespace,
            )
            roots: tuple[CaseRecord, ...] = ()
            try:
                roots = self._sample_response(SampleRequest(
                    round_index=round_index,
                    cohort=cohort,
                    seed=base_seed,
                    count=root_count,
                    settings=(
                        {} if request_settings is None else request_settings
                    ),
                    kind=GenerationKind.BASE,
                ))
                children = self._sample_response(SampleRequest(
                    round_index=round_index,
                    cohort=cohort,
                    seed=perturbed_seed,
                    count=root_count,
                    settings=(
                        {} if request_settings is None else request_settings
                    ),
                    kind=GenerationKind.PERTURBED,
                    anchors=tuple(record.case for record in roots),
                ))
            except RecoverableSamplingRejection as exc:
                self._release_records(roots)
                audit = SamplingAttemptAudit(
                    round_index=round_index,
                    cohort=cohort,
                    attempt_index=attempt_index,
                    maximum_attempts=maximum_attempts,
                    base_seed=base_seed,
                    perturbed_seed=perturbed_seed,
                    root_case_ids=tuple(
                        record.case.case_id for record in roots
                    ),
                    accepted=False,
                    rejection_audit=MappingProxyType(dict(exc.audit)),
                )
                attempts.append(audit)
                self._sampling_attempts.append(audit)
                continue
            except BaseException:
                self._release_records(roots)
                raise

            for root, child in zip(roots, children):
                if child.case.parent_case_id != root.case.case_id:
                    self._release_records((*roots, *children))
                    raise ValueError(
                        "perturbed cases must preserve anchor order and lineage"
                    )
            audit = SamplingAttemptAudit(
                round_index=round_index,
                cohort=cohort,
                attempt_index=attempt_index,
                maximum_attempts=maximum_attempts,
                base_seed=base_seed,
                perturbed_seed=perturbed_seed,
                root_case_ids=tuple(record.case.case_id for record in roots),
                accepted=True,
            )
            attempts.append(audit)
            self._sampling_attempts.append(audit)
            return tuple(
                record
                for pair in zip(roots, children)
                for record in pair
            )

        failure_audit = {
            "schema": "paired_sampling_exhaustion_v1",
            "round_index": round_index,
            "cohort": cohort.value,
            "maximum_attempts": maximum_attempts,
            "attempts": [
                {
                    "attempt_index": row.attempt_index,
                    "base_seed": row.base_seed,
                    "perturbed_seed": row.perturbed_seed,
                    "root_case_ids": list(row.root_case_ids),
                    "rejection_audit": dict(row.rejection_audit or {}),
                }
                for row in attempts
            ],
        }
        raise RecoverableSamplingRejection(
            "paired case sampling exhausted its fresh-draw attempt budget",
            audit=failure_audit,
        )

    def _sample_response(
        self,
        request: SampleRequest,
    ) -> tuple[CaseRecord, ...]:
        records = tuple(self.plugin.generator.sample(request))
        if len(records) != request.count:
            self._release_records(records)
            raise ValueError(
                f"Generator returned {len(records)} cases for a request "
                f"of {request.count}"
            )
        try:
            for record in records:
                if not isinstance(record, CaseRecord):
                    raise TypeError("Generator responses must contain CaseRecord values")
                self._validate_sampled_case(record.case, request)
        except BaseException:
            self._release_records(records)
            raise
        return records

    def _validate_sampled_case(
        self,
        case: Case,
        request: SampleRequest,
    ) -> None:
        if not isinstance(case, Case):
            raise TypeError("CaseRecord.case must be a Case")
        if not isinstance(case.case_id, str) or not case.case_id.strip():
            raise ValueError("Generator case ids must be nonempty strings")
        if not isinstance(case.source, CaseSource):
            raise ValueError("Generator case source must be a CaseSource")
        if request.kind is GenerationKind.PERTURBED:
            if case.source is not CaseSource.PERTURBED:
                raise ValueError("perturbed requests require PERTURBED case source")
            anchor_ids = {anchor.case_id for anchor in request.anchors}
            if case.parent_case_id not in anchor_ids:
                raise ValueError("perturbed cases must name a requested anchor")
        elif self.config.paired_perturbations:
            if case.source is CaseSource.PERTURBED:
                raise ValueError("clean requests cannot return PERTURBED cases")
            if case.parent_case_id is not None:
                raise ValueError("clean cases cannot declare a parent")
        expected_visibility = (
            CaseVisibility.DEVELOPMENT
            if request.cohort is CohortKind.DEVELOPMENT
            else CaseVisibility.HIDDEN
        )
        if case.visibility is not expected_visibility:
            raise ValueError(
                f"{request.cohort.value} cases require "
                f"{expected_visibility.value} visibility"
            )
        if case.round_index is not None:
            _require_integer("case round_index", case.round_index, minimum=0)
            if case.round_index != request.round_index:
                raise ValueError("Generator case round does not match the request")

        declared_cohorts: list[object] = []
        if isinstance(case.public_data, Mapping):
            for key in ("cohort", "cohort_kind", "controller_cohort"):
                if key in case.public_data:
                    declared_cohorts.append(case.public_data[key])
        for tag in case.tags:
            if not isinstance(tag, str):
                raise ValueError("case tags must be strings")
            for prefix in ("cohort:", "cohort="):
                if tag.startswith(prefix):
                    declared_cohorts.append(tag[len(prefix) :].strip())
        for declared in declared_cohorts:
            declared_value = (
                declared.value if isinstance(declared, CohortKind) else declared
            )
            if declared_value != request.cohort.value:
                raise ValueError("Generator case cohort does not match the request")

    def _evaluate_official_common_cohort(
        self,
        *,
        round_index: int,
        cohort_kind: CohortKind,
        submissions: Sequence[CandidateSubmission],
    ) -> tuple[MacroBatchEvaluation, ...]:
        """Evaluate selection/terminal as a bounded root-major stream.

        Every candidate consumes each exact root (or clean/perturbed root
        pair) before that manufactured answer is released. Only sanitized
        scores, timings, root ids, and row commitments survive between
        chunks. One fresh runtime per candidate is retained across all chunks
        in this formal cohort. Official transient cases and targets are always
        memory-backed,
        even when development cohorts have an explicit private store.
        """

        if cohort_kind not in {CohortKind.SELECTION, CohortKind.TERMINAL}:
            raise ValueError(
                "official common streaming is limited to selection and terminal"
            )
        rows = tuple(submissions)
        if not rows:
            raise ValueError("official common evaluation needs candidates")
        try:
            cohort_id, member_count, profile = self._cohort_plans[
                (round_index, cohort_kind)
            ]
        except KeyError as exc:
            raise RuntimeError(
                "requested controller cohort was not precomputed"
            ) from exc
        members_per_root = 2 if self.config.paired_perturbations else 1
        if member_count % members_per_root:
            raise ValueError("official cohort cannot split a paired root")
        total_root_count = member_count // members_per_root
        matcher = self.plugin.objective.matcher
        release_cases = getattr(self.plugin.cases, "release_cases", None)
        release_cases = release_cases if callable(release_cases) else None
        plan = ControllerMacroQueryPlan(
            cohort_id=cohort_id,
            profile=profile,
            controller_seed=controller_query_plan_seed(
                seed_base=self.config.seed_base,
                cohort_id=cohort_id,
                profile=profile,
            ),
        )
        candidate_entries = tuple(
            (submission.ref.candidate_id, submission.fresh_program())
            for submission in rows
        )
        accumulator = MacroBatchStreamAccumulator(
            cohort_id=cohort_id,
            contract_id=matcher.contract_id,
            candidate_ids=tuple(row[0] for row in candidate_entries),
        )

        def consume_chunk(root_offset: int, root_count: int) -> None:
            records = self._sample(
                round_index,
                cohort_kind,
                root_count * members_per_root,
                seed_namespace=(
                    "official-common-stream-v1"
                    f"\0{cohort_id}"
                    f"\0root-offset-{root_offset}"
                ),
                request_settings={
                    "logical_cohort_id": cohort_id,
                    "logical_root_offset": root_offset,
                    "logical_root_count": total_root_count,
                },
            )
            cohort = PreparedMacroCohort.prepare(
                cohort_id=cohort_id,
                records=records,
                matcher=matcher,
                query=lambda case: plan.build(self.plugin.objective, case),
                private_store_root=None,
                release_cases=release_cases,
            )
            try:
                accumulator.consume(
                    cohort,
                    self._controller_evaluator(cohort).evaluate_many(
                        candidate_entries
                    ),
                )
            finally:
                cohort.discard()

        for root_offset in range(
            0,
            total_root_count,
            _OFFICIAL_COMMON_ROOT_BATCH_SIZE,
        ):
            consume_chunk(
                root_offset,
                min(
                    _OFFICIAL_COMMON_ROOT_BATCH_SIZE,
                    total_root_count - root_offset,
                ),
            )
        batches = accumulator.finalize()
        if any(len(batch.scores) != member_count for batch in batches):
            raise RuntimeError("official streamed cohort member count changed")
        return batches

    def _controller_evaluator(
        self,
        cohort: PreparedMacroCohort,
    ) -> ControllerMacroEvaluator:
        return ControllerMacroEvaluator(
            cohort,
            compute_budget=self.config.candidate_compute_budget,
            query_seconds_per_point_limit=(
                self.config.candidate_query_seconds_per_point_limit
            ),
            public_session_factory=self.plugin.public_session_factory,
            public_session_methods=self.plugin.public_session_methods,
        )

    def _agent_visible_metrics(self) -> tuple[str, ...]:
        raw = getattr(self.plugin.objective, "agent_visible_metrics", ())
        if isinstance(raw, (str, bytes)):
            raise ValueError("Agent-visible metrics must be a sequence of names")
        names = tuple(raw)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("Agent-visible metric names must be nonempty strings")
        if len(names) != len(set(names)):
            raise ValueError("Agent-visible metric names must be unique")
        return names

    def _published_metric_names(self) -> tuple[str, ...]:
        raw = getattr(self.plugin.objective, "published_metric_names", ())
        if isinstance(raw, (str, bytes)):
            raise ValueError("published metrics must be a sequence of names")
        names = tuple(raw)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("published metric names must be nonempty strings")
        if len(names) != len(set(names)):
            raise ValueError("published metric names must be unique")
        return names

    def _published_metric_summary(
        self,
        batch: MacroBatchEvaluation,
        *,
        case_count: int | None = None,
    ) -> PublishedMetricSummary:
        count = len(batch.scores) if case_count is None else case_count
        _require_integer("published metric case_count", count, minimum=1)
        if count > len(batch.scores):
            raise ValueError("published metric prefix exceeds the evaluated cohort")
        scores = batch.scores[:count]
        aggregates: dict[str, PublishedMetricAggregate] = {}
        for name in self._published_metric_names():
            if not all(name in score.metrics for score in scores):
                continue
            root_values = _root_grouped_metric_values(
                tuple(float(score.metrics[name]) for score in scores),
                root_ids=batch.root_ids,
                case_count=count,
            )
            aggregates[name] = _aggregate_published_metric(
                root_values,
                cvar_tail_fraction=self.config.cvar_tail_fraction,
            )
        return PublishedMetricSummary(count, aggregates)

    def _paired_published_metric_summary(
        self,
        candidate: MacroBatchEvaluation,
        incumbent: MacroBatchEvaluation,
        *,
        case_count: int,
    ) -> PairedPublishedMetricSummary:
        if (
            candidate.cohort_sha256 != incumbent.cohort_sha256
            or candidate.target_sha256 != incumbent.target_sha256
            or candidate.contract_id != incumbent.contract_id
            or candidate.case_ids != incumbent.case_ids
        ):
            raise ValueError("published metric batches must be paired exactly")
        candidate_summary = self._published_metric_summary(
            candidate,
            case_count=case_count,
        )
        incumbent_summary = self._published_metric_summary(
            incumbent,
            case_count=case_count,
        )
        shared_names = tuple(
            name
            for name in self._published_metric_names()
            if (
                name in candidate_summary.aggregates
                and name in incumbent_summary.aggregates
            )
        )
        return PairedPublishedMetricSummary(
            case_count=case_count,
            candidate=PublishedMetricSummary(
                case_count,
                {name: candidate_summary.aggregates[name] for name in shared_names},
            ),
            incumbent=PublishedMetricSummary(
                case_count,
                {name: incumbent_summary.aggregates[name] for name in shared_names},
            ),
        )

    def _prepare_cohort(
        self,
        *,
        cohort_id,
        records,
        profile,
        release_cases_override=_DEFAULT_RELEASE_CASES,
    ):
        plan = ControllerMacroQueryPlan(
            cohort_id=cohort_id,
            profile=profile,
            controller_seed=controller_query_plan_seed(
                seed_base=self.config.seed_base,
                cohort_id=cohort_id,
                profile=profile,
            ),
        )
        if release_cases_override is _DEFAULT_RELEASE_CASES:
            release_cases = getattr(self.plugin.generator, "release_cases", None)
            release_cases = release_cases if callable(release_cases) else None
        else:
            release_cases = release_cases_override
            if release_cases is not None and not callable(release_cases):
                raise TypeError("release_cases_override must be callable or None")
        return PreparedMacroCohort.prepare(
            cohort_id=cohort_id,
            records=records,
            matcher=self.plugin.objective.matcher,
            query=lambda case: plan.build(self.plugin.objective, case),
            private_store_root=self._cohort_store_root,
            release_cases=release_cases,
        )

    def _paired_sampling_attempt_seed(
        self,
        round_index: int,
        cohort: CohortKind,
        kind: GenerationKind,
        attempt_index: int,
        *,
        seed_namespace: str | None = None,
    ) -> int:
        _require_integer(
            "paired sampling attempt_index",
            attempt_index,
            minimum=0,
        )
        if kind is GenerationKind.BASE:
            variant_offset = 0
        elif kind is GenerationKind.PERTURBED:
            variant_offset = 4
        else:
            raise ValueError("macro loop only samples BASE or PERTURBED cases")
        if seed_namespace is None:
            if attempt_index == 0:
                return self._seed(round_index, cohort, kind)
            label = (
                "macro-paired-fresh-resample-v1"
                f"\0{self.config.seed_base}"
                f"\0{round_index}"
                f"\0{cohort.value}"
                f"\0retry-{attempt_index}"
            )
        else:
            label = (
                "macro-development-fresh-draw-v1"
                f"\0{self.config.seed_base}"
                f"\0{round_index}"
                f"\0{cohort.value}"
                f"\0{seed_namespace}"
                f"\0attempt-{attempt_index}"
            )
        base_seed = int.from_bytes(
            hashlib.sha256(label.encode("utf-8")).digest()[:8],
            "big",
        ) % ((1 << 63) - 4)
        return base_seed + variant_offset

    def _seed(
        self,
        round_index: int,
        cohort: CohortKind,
        kind: GenerationKind,
    ) -> int:
        if cohort is CohortKind.DEVELOPMENT_SCREEN:
            raise ValueError(
                "development screens require a fresh seed namespace"
            )
        offsets = {
            CohortKind.DEVELOPMENT: 0,
            CohortKind.SELECTION: 1,
            CohortKind.PROMOTION: 2,
            CohortKind.TERMINAL: 3,
        }
        if kind is GenerationKind.BASE:
            variant_offset = 0
        elif kind is GenerationKind.PERTURBED:
            variant_offset = 4
        else:
            raise ValueError("macro loop only samples BASE or PERTURBED cases")
        return (
            self.config.seed_base
            + round_index * self.config.seed_stride
            + offsets[cohort]
            + variant_offset
        )

    def _validate_candidate(
        self,
        candidate: CandidateRef,
        agent: MacroAgentSpec,
        round_index: int,
        incumbent: CandidateRef,
    ) -> None:
        _require_integer(
            "candidate round_index", candidate.round_index, minimum=1
        )
        if candidate.round_index != round_index:
            raise ValueError("candidate round identity does not match the request")
        if candidate.agent_id != agent.agent_id:
            raise ValueError("candidate Agent identity does not match the request")
        if candidate.parent_sha256 != incumbent.bundle_sha256:
            raise ValueError("candidate parent must be the exact incumbent bundle")
        known = self._candidate_ids.get(candidate.candidate_id)
        if known is not None:
            if known != candidate.bundle_sha256:
                raise ValueError("candidate id was reused with different content")
            raise ValueError("candidate id was submitted more than once")
        canonical_id = self._candidate_digests.get(candidate.bundle_sha256)
        if canonical_id is not None:
            raise ValueError(
                f"candidate bundle was already submitted as {canonical_id!r}"
            )
        self._candidate_ids[candidate.candidate_id] = candidate.bundle_sha256
        self._candidate_digests[candidate.bundle_sha256] = candidate.candidate_id

    def _agent_prompt(self, spec: MacroAgentSpec) -> str:
        matcher = self.plugin.objective.matcher
        if not self.config.paired_perturbations:
            pairing_note = ""
        elif self._perturbed_guardrail_enabled():
            pairing_note = (
                " Development evidence mixes generated and smooth off-support "
                "questions. Distribution membership, pairing, and evaluation "
                "order are controller-private. Generated-distribution performance "
                "is the primary objective; off-support performance must pass an "
                "independent practical non-regression guardrail."
            )
        else:
            pairing_note = (
                " Development evidence mixes generated and smooth off-support "
                "questions. Distribution membership, pairing, and evaluation "
                "order are controller-private. The objective combines both "
                "distributions with configured weights over independent "
                "statistical roots."
            )
        screening_note = (
            " Candidate development screens use fresh private roots and return "
            "aggregate-only feedback; completed released answers are the only "
            "development truths visible to the Agent. Use "
            "load_reference_public_workspace and read search_contract, "
            "prior_leaderboard, and prior_candidates from the returned public "
            "workspace; the two prior-round fields are empty until the "
            "controller makes a cross-round handoff available."
            if self.config.development_screen_root_stages
            else ""
        )
        evaluation_note = (
            " Candidate discretization is entirely candidate-owned: dimensions, "
            "coordinates, basis, and interpolation are not fixed by the protocol. "
            "The target two-phase ABI is prepare_solution(case, solve_budget) "
            "without hidden scoring points, followed by "
            "solution.evaluate(points_query); solve and query latency are separate "
            "eligibility limits. "
            "predict_macros(case, query, budget=None) is the legacy one-shot "
            "compatibility adapter and never supplies formal timing assurance."
        )
        search_note = (
            " Use the development numerical oracle and private checkpoint archive "
            "as often as useful; there is no minimum usage or finish milestone. "
            "Nominate the single strongest immutable checkpoint as this Agent's "
            "one official entry for the round. Private mean/CVaR/max archive "
            "representatives do not receive extra leaderboard entries. The "
            "controller ranks the incumbent and one nomination per Agent on the "
            "same common hidden physical-point cohort, using only equation-owned "
            "ROM quality among eligible candidates."
        )
        return (
            f"Equation-owned objective: {matcher.info.title}. "
            "The controller uses only sanitized per-case MacroMatchResult values "
            "for the common-cohort leaderboard and terminal report."
            f"{pairing_note}{screening_note}{evaluation_note}{search_note}\n\n"
            f"{spec.prompt}"
        )

    def _handoff(
        self,
        *,
        round_index: int,
        incumbent: CandidateRef,
        selected: CandidateRef,
        challenger: CandidateRef | None,
        promotion: PromotionResult | None,
        reason: str,
        previous_handoff: RoundHandoff | None,
        perturbed_guardrail: PerturbedGuardrailSummary | None,
    ) -> RoundHandoff:
        comparison = (
            promotion.comparisons[-1]
            if promotion is not None and promotion.comparisons
            else None
        )
        promoted = bool(promotion and promotion.promoted)
        feedback = RoundFeedback(
            previous_incumbent_sha256=incumbent.bundle_sha256,
            selected_sha256=selected.bundle_sha256,
            promoted=promoted,
            decision_reason=reason,
            evidence_case_count=(
                comparison.stage.case_count if comparison is not None else 0
            ),
            paired_mean_improvement=(
                comparison.mean_improvement if comparison is not None else None
            ),
            paired_confidence_lower=(
                comparison.confidence_lower if comparison is not None else None
            ),
            paired_win_rate=(
                comparison.win_rate if comparison is not None else None
            ),
            confidence_group_size=self._objective_confidence_group_size(),
            perturbed_guardrail_passed=(
                perturbed_guardrail.passed
                if perturbed_guardrail is not None
                else None
            ),
            perturbed_guardrail_confidence_lower=(
                perturbed_guardrail.confidence_lower
                if perturbed_guardrail is not None
                else None
            ),
        )
        previous_digest = (
            previous_handoff.handoff_sha256
            if previous_handoff is not None
            else None
        )
        digest = _round_handoff_digest(
            run_name=self.config.name,
            plugin_id=self.plugin.plugin_id,
            generator_id=self.plugin.generator.generator_id,
            generator_sha256=self.plugin.generator.artifact_sha256,
            macro_contract_id=self.plugin.objective.matcher.contract_id,
            round_index=round_index,
            incumbent=incumbent,
            selected=selected,
            challenger=challenger,
            promoted=promoted,
            decision_reason=reason,
            feedback=feedback,
            previous_handoff_sha256=previous_digest,
            evidence_target_sha256=(
                comparison.target_sha256 if comparison is not None else None
            ),
            evidence_contract_id=(
                comparison.contract_id if comparison is not None else None
            ),
        )
        return RoundHandoff(
            round_index=round_index,
            selected=selected,
            promoted=promoted,
            challenger=challenger,
            decision_reason=reason,
            feedback=feedback,
            previous_handoff_sha256=previous_digest,
            handoff_sha256=digest,
        )


def verify_round_handoff(
    handoff: RoundHandoff,
    *,
    run_name: str,
    plugin_id: str,
    generator_id: str,
    generator_sha256: str,
    macro_contract_id: str,
    incumbent: CandidateRef,
    evidence_target_sha256: str | None = None,
    evidence_contract_id: str | None = None,
) -> bool:
    """Verify a promotion-lineage handoff using private evidence commitments.

    Candidate bundle URIs are deliberately excluded so an attested bundle may
    move between trusted stores without changing its identity. Every other
    CandidateRef provenance field is bound by the digest. Competitive-search
    policy, completion audits, and leaderboard state are outside this
    promotion-only commitment and must be audited from ``MacroRoundResult``.
    """

    if not isinstance(handoff, RoundHandoff):
        raise TypeError("handoff must be a RoundHandoff")
    expected = _round_handoff_digest(
        run_name=run_name,
        plugin_id=plugin_id,
        generator_id=generator_id,
        generator_sha256=generator_sha256,
        macro_contract_id=macro_contract_id,
        round_index=handoff.round_index,
        incumbent=incumbent,
        selected=handoff.selected,
        challenger=handoff.challenger,
        promoted=handoff.promoted,
        decision_reason=handoff.decision_reason,
        feedback=handoff.feedback,
        previous_handoff_sha256=handoff.previous_handoff_sha256,
        evidence_target_sha256=evidence_target_sha256,
        evidence_contract_id=evidence_contract_id,
    )
    return handoff.handoff_sha256 == expected


def _round_handoff_digest(
    *,
    run_name: str,
    plugin_id: str,
    generator_id: str,
    generator_sha256: str,
    macro_contract_id: str,
    round_index: int,
    incumbent: CandidateRef,
    selected: CandidateRef,
    challenger: CandidateRef | None,
    promoted: bool,
    decision_reason: str,
    feedback: RoundFeedback,
    previous_handoff_sha256: str | None,
    evidence_target_sha256: str | None,
    evidence_contract_id: str | None,
) -> str:
    if (evidence_target_sha256 is None) != (evidence_contract_id is None):
        raise ValueError(
            "evidence target digest and contract must either both be set or both be absent"
        )
    payload = {
        "schema": (
            "macro-round-handoff-v3"
            if feedback.perturbed_guardrail_passed is not None
            else (
                "macro-round-handoff-v1"
                if feedback.confidence_group_size == 1
                else "macro-round-handoff-v2"
            )
        ),
        "run_name": run_name,
        "plugin_id": plugin_id,
        "generator_id": generator_id,
        "generator_sha256": generator_sha256,
        "macro_contract_id": macro_contract_id,
        "previous_handoff_sha256": previous_handoff_sha256,
        "round_index": round_index,
        "promoted": promoted,
        "decision_reason": decision_reason,
        "incumbent": _candidate_ref_commitment(incumbent),
        "selected": _candidate_ref_commitment(selected),
        "challenger": _candidate_ref_commitment(challenger),
        "evidence_target_sha256": evidence_target_sha256,
        "evidence_contract_id": evidence_contract_id,
        "feedback": _round_feedback_commitment(feedback),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _candidate_ref_commitment(
    candidate: CandidateRef | None,
) -> Mapping[str, object] | None:
    if candidate is None:
        return None
    return {
        "candidate_id": candidate.candidate_id,
        "bundle_sha256": candidate.bundle_sha256,
        "round_index": candidate.round_index,
        "agent_id": candidate.agent_id,
        "parent_sha256": candidate.parent_sha256,
    }


def _round_feedback_commitment(feedback: RoundFeedback) -> Mapping[str, object]:
    payload = {
        "previous_incumbent_sha256": feedback.previous_incumbent_sha256,
        "selected_sha256": feedback.selected_sha256,
        "promoted": feedback.promoted,
        "decision_reason": feedback.decision_reason,
        "evidence_case_count": feedback.evidence_case_count,
        "paired_mean_improvement": feedback.paired_mean_improvement,
        "paired_confidence_lower": feedback.paired_confidence_lower,
        "paired_win_rate": feedback.paired_win_rate,
    }
    if feedback.confidence_group_size != 1:
        payload["confidence_group_size"] = feedback.confidence_group_size
    if feedback.perturbed_guardrail_passed is not None:
        payload["perturbed_guardrail_passed"] = (
            feedback.perturbed_guardrail_passed
        )
        payload["perturbed_guardrail_confidence_lower"] = (
            feedback.perturbed_guardrail_confidence_lower
        )
    return payload


def _public_guardrail_failure_code(failure: str) -> str:
    if failure.startswith("candidate_gate:"):
        return "candidate_gate_failure"
    if failure.startswith("incumbent_gate:"):
        return "incumbent_gate_failure"
    return failure


def _aggregate_published_metric(
    values: Sequence[float],
    *,
    cvar_tail_fraction: float,
) -> PublishedMetricAggregate:
    """Aggregate finite allowlisted values without assuming metric sign."""

    materialized = tuple(float(value) for value in values)
    if not materialized or any(not isfinite(value) for value in materialized):
        raise ValueError("published metrics must be a nonempty finite sequence")
    if (
        not isfinite(float(cvar_tail_fraction))
        or not 0.0 < float(cvar_tail_fraction) <= 1.0
    ):
        raise ValueError("published metric CVaR fraction must lie in (0, 1]")
    tail_count = max(1, ceil(len(materialized) * float(cvar_tail_fraction)))
    upper_tail = sorted(materialized, reverse=True)[:tail_count]
    return PublishedMetricAggregate(
        mean=fmean(materialized),
        cvar=fmean(upper_tail),
        max=max(materialized),
    )

def _root_grouped_metric_values(
    values: Sequence[float],
    *,
    root_ids: Sequence[str],
    case_count: int,
) -> tuple[float, ...]:
    """Give each complete manufactured root equal statistical weight."""

    if len(values) != case_count or len(root_ids) < case_count:
        raise ValueError("published metric values must align with root lineage")
    if (
        case_count < len(root_ids)
        and root_ids[case_count - 1] == root_ids[case_count]
    ):
        raise ValueError("published metric prefix cannot split a root group")
    ordered_roots: list[str] = []
    grouped: dict[str, list[float]] = {}
    for root_id, value in zip(root_ids[:case_count], values):
        if root_id not in grouped:
            ordered_roots.append(root_id)
            grouped[root_id] = []
        grouped[root_id].append(float(value))
    return tuple(fmean(grouped[root_id]) for root_id in ordered_roots)


def _leaderboard_eligible(row: FinalistEvaluation) -> bool:
    return bool(
        row.score.eligible
        and row.aggregate is not None
        and (
            row.perturbed_guardrail is None
            or row.perturbed_guardrail.passed
        )
    )


def _promotion_metric(
    aggregate: RobustAggregate,
    case_count: int,
) -> PromotionMetricSummary:
    return PromotionMetricSummary(
        case_count=case_count,
        mean=aggregate.mean,
        cvar=aggregate.cvar,
        maximum=aggregate.max,
    )


def _common_cohort_rank_key(
    row: FinalistEvaluation,
    *,
    incumbent_digest: str | None,
) -> tuple[object, ...]:
    aggregate = row.aggregate
    incumbent_tie_priority = (
        0
        if (
            incumbent_digest is not None
            and row.candidate.bundle_sha256 == incumbent_digest
        )
        else 1
    )
    return (
        not _leaderboard_eligible(row),
        aggregate.mean if aggregate is not None else float("inf"),
        aggregate.cvar if aggregate is not None else float("inf"),
        aggregate.max if aggregate is not None else float("inf"),
        incumbent_tie_priority,
        row.candidate.bundle_sha256,
        row.candidate.candidate_id,
    )


def _search_leaderboard(
    finalists: Sequence[FinalistEvaluation],
    *,
    roles: Mapping[str, tuple[str, ...]],
    limit: int,
    incumbent_digest: str | None = None,
) -> tuple[SearchLeaderboardEntry, ...]:
    """Rank common fresh-cohort aggregates without exposing case-level data."""

    _require_integer("leaderboard limit", limit, minimum=0)
    if limit == 0:
        return ()


    ordered = sorted(
        finalists,
        key=lambda row: _common_cohort_rank_key(
            row,
            incumbent_digest=incumbent_digest,
        ),
    )
    entries: list[SearchLeaderboardEntry] = []
    for rank, row in enumerate(ordered[:limit], start=1):
        aggregate = row.aggregate
        objectives = row.objective_aggregates
        entries.append(
            SearchLeaderboardEntry(
                rank=rank,
                candidate=row.candidate,
                agent_id=row.candidate.agent_id,
                eligible=_leaderboard_eligible(row),
                mean=(aggregate.mean if aggregate is not None else None),
                cvar=(aggregate.cvar if aggregate is not None else None),
                maximum=(aggregate.max if aggregate is not None else None),
                generated_mean=(
                    objectives.generated.mean
                    if objectives is not None
                    else (aggregate.mean if aggregate is not None else None)
                ),
                perturbed_mean=(
                    objectives.perturbed.mean
                    if objectives is not None
                    else None
                ),
                guardrail_passed=(
                    row.perturbed_guardrail.passed
                    if row.perturbed_guardrail is not None
                    else None
                ),
                roles=tuple(roles.get(row.candidate.candidate_id, ())),
            )
        )
    return tuple(entries)


def _screened_portfolio(
    session: DevelopmentSession,
    *,
    final: CandidateSubmission | None,
    representative_limit: int,
    minimum_completed_root_count: int = 0,
) -> tuple[tuple[CandidateSubmission, tuple[str, ...]], ...]:
    """Choose at most one official nomination from a private Agent archive."""

    _require_integer(
        "minimum_completed_root_count",
        minimum_completed_root_count,
        minimum=0,
    )
    _require_integer(
        "representative_limit",
        representative_limit,
        minimum=0,
    )
    if final is not None:
        return ((final, ("agent_nomination",)),)
    if representative_limit == 0:
        return ()

    summaries = tuple(
        row
        for row in session.screened_candidate_summaries
        if row.eligible
        and row.completed_root_count >= minimum_completed_root_count
    )
    generated_primary = (
        session.paired_objective_policy.mode
        is PairedObjectiveMode.PRIMARY_WITH_PERTURBED_GUARDRAIL
        and any(row.generated_loss_mean is not None for row in summaries)
    )
    fields = (
        ("generated_loss_mean", "loss_cvar", "loss_max")
        if generated_primary
        else ("loss_mean", "loss_cvar", "loss_max")
    )
    available = tuple(
        row
        for row in summaries
        if all(getattr(row, field, None) is not None for field in fields)
    )
    if not available:
        return ()
    recovered = min(
        available,
        key=lambda row: tuple(
            float(getattr(row, field)) for field in fields
        )
        + (-row.completed_root_count, row.candidate.ref.bundle_sha256),
    )
    return ((recovered.candidate, ("recovered_checkpoint",)),)


def _contract_safe_screened_portfolio(
    session: DevelopmentSession,
    *,
    final: CandidateSubmission | None,
    representative_limit: int,
    audit: SearchCompletionAudit | None,
    policy: CompetitiveSearchPolicy,
) -> tuple[tuple[CandidateSubmission, tuple[str, ...]], ...]:
    """Return one explicit nomination or one deterministic recovery asset."""

    if audit is None:
        raise RuntimeError(
            "strong development requires observational search accounting"
        )
    if not isinstance(policy, CompetitiveSearchPolicy):
        raise TypeError("policy must be a CompetitiveSearchPolicy")
    return _screened_portfolio(
        session,
        final=final,
        representative_limit=representative_limit,
    )


def _close_guidance_oracles(
    oracles: Sequence[DevelopmentGuidanceOracle],
) -> None:
    """Close every Agent-scoped guide, preserving the first cleanup error."""

    failures: list[BaseException] = []
    seen: set[int] = set()
    for oracle in oracles:
        if id(oracle) in seen:
            continue
        seen.add(id(oracle))
        try:
            oracle.close()
        except BaseException as exc:
            failures.append(exc)
    if failures:
        first = failures[0]
        for extra in failures[1:]:
            first.add_note(
                "additional guidance cleanup failure: "
                f"{type(extra).__name__}: {extra}"
            )
        raise first


def _open_search_policy(
    policy: CompetitiveSearchPolicy,
) -> CompetitiveSearchPolicy:
    """Strip legacy finish milestones before exposing policy to an Agent."""

    if not isinstance(policy, CompetitiveSearchPolicy):
        raise TypeError("policy must be a CompetitiveSearchPolicy")
    return replace(
        policy,
        min_distinct_screened_candidates=0,
        min_full_stage_candidates=0,
        min_released_roots=0,
        require_final_full_stage=False,
    )


def _validated_compute_budget(value: float | None) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError("candidate_compute_budget must be finite and positive")
    return float(value)


def _validated_query_seconds_per_point_limit(
    value: float | None,
) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(
            "candidate_query_seconds_per_point_limit must be finite and positive"
        )
    return float(value)


def _require_integer(name: str, value: object, *, minimum: int) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
