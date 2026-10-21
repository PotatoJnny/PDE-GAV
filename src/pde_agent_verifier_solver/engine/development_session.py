"""Fresh, aggregate-only development interaction for macroscopic ROM Agents.

PDE plugins own sampling, perturbations, and matching.  This synchronous
session owns fresh draws, digest-scoped paired evaluation, and observational
resource accounting.  The unified open-search mode has no promotion stages or
lifetime query quota: every request receives fresh evidence and the configured
root limits are per-request capacity guards.  The staged mode remains only for
loading and reproducing older campaign manifests.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
from functools import wraps
import json
from math import isfinite
from random import SystemRandom
from statistics import fmean
from typing import Mapping, Protocol, Sequence

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    CandidateSubmission,
    Case,
)

from .macro_evaluator import (
    ControllerMacroEvaluator,
    MacroBatchEvaluation,
    PreparedMacroCohort,
    SanitizedMacroCaseScore,
    _snapshot_value,
)
from .selection import (
    EvidenceStage,
    MacroCandidateScore,
    PairedObjectiveMode,
    PairedObjectivePolicy,
    aggregate,
    compare_perturbed_non_regression,
    paired_compare,
    paired_objective_score,
    paired_weighted_composite,
    split_paired_score,
)


class DevelopmentSessionError(RuntimeError):
    """Raised when a request violates the development-session contract."""


class DevelopmentCreditExceeded(DevelopmentSessionError):
    """Raised before a request would exceed its independent root budget."""

def _release_new_cohorts_on_failure(method):
    """Return newly acquired cohort ownership if an operation raises."""

    @wraps(method)
    def guarded(self, *args, **kwargs):
        existing = set(self._active_cohorts)
        try:
            return method(self, *args, **kwargs)
        except BaseException as exc:
            for cohort in tuple(self._active_cohorts - existing):
                try:
                    self._discard_cohort(cohort)
                except BaseException as cleanup_exc:
                    exc.add_note(
                        "development cohort cleanup also failed: "
                        f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                    )
            raise

    return guarded


class DevelopmentDrawKind(str, Enum):
    """Kinds of fresh development draws."""

    RELEASE = "release"
    SCREEN = "screen"


@dataclass(frozen=True)
class DevelopmentDrawRequest:
    """Fresh-cohort factory request; counts are independent clean roots."""

    kind: DevelopmentDrawKind
    draw_index: int
    root_count: int
    candidate_sha256: str | None = None
    cumulative_screen_root_count: int | None = None


class DevelopmentCohortFactory(Protocol):
    """Equation-owned callback that materializes one requested fresh cohort."""

    def __call__(self, request: DevelopmentDrawRequest) -> PreparedMacroCohort: ...


@dataclass(frozen=True)
class DevelopmentScreenResponse:
    """Aggregate-only Agent response; no case or target material is retained."""

    candidate_sha256: str
    incumbent_sha256: str
    completed_root_count: int
    member_case_count: int
    completed_stage_root_counts: tuple[int, ...]
    _aggregates: Mapping[str, object]
    charged_root_credits: int
    cumulative_charged_root_credits: int
    remaining_root_credits: int
    cached: bool = False
    protocol_lifetime_limit: bool = True
    per_request_root_limit: int | None = None

    def public_payload(self) -> Mapping[str, object]:
        """Return the complete allowlisted Agent-facing payload."""

        aggregates = _snapshot_value(self._aggregates)
        return {
            "schema": "macro-development-screen-response-v2",
            "candidate_sha256": self.candidate_sha256,
            "incumbent_sha256": self.incumbent_sha256,
            "completed_root_count": self.completed_root_count,
            "member_case_count": self.member_case_count,
            "completed_stage_root_counts": list(self.completed_stage_root_counts),
            **aggregates,
            "accounting": {
                "charged_root_credits": self.charged_root_credits,
                "cumulative_charged_root_credits": self.cumulative_charged_root_credits,
                "remaining_root_credits": self.remaining_root_credits,
                "cached": self.cached,
                "protocol_lifetime_limit": self.protocol_lifetime_limit,
                "per_request_root_limit": self.per_request_root_limit,
            },
            "private_case_ids_included": False,
            "per_case_errors_included": False,
            "finite_truth_included": False,
            "official_promotion_result": False,
        }


@dataclass(frozen=True)
class DevelopmentScreenedCandidate:
    """Controller-facing summary of one immutable screened digest."""

    candidate: CandidateSubmission
    completed_root_count: int
    eligible: bool
    loss_mean: float | None
    loss_cvar: float | None
    loss_max: float | None
    paired_mean_improvement: float | None
    generated_loss_mean: float | None = None
    perturbed_loss_mean: float | None = None
    composite_loss_mean: float | None = None
    perturbed_guardrail_passed: bool | None = None


@dataclass(frozen=True)
class DevelopmentRelease:
    """Completed public cases and answers charged to the release ledger."""

    release_index: int
    root_count: int
    member_case_count: int
    cohort_id: str
    cohort_sha256: str
    contract_id: str
    cases: tuple[Case, ...]
    queries: tuple[AnswerQuery, ...]
    answers: tuple[AnswerView, ...]
    charged_root_credits: int
    cumulative_charged_root_credits: int
    remaining_root_credits: int


@dataclass(frozen=True)
class DevelopmentDrawAudit:
    """Controller-private commitment for one actual fresh draw."""

    draw_index: int
    kind: DevelopmentDrawKind
    root_count: int
    member_case_count: int
    cohort_id: str
    cohort_sha256: str


@dataclass(frozen=True)
class DevelopmentLedgerEntry:
    """One independent release or screen accounting record."""

    request_index: int
    charged_root_credits: int
    draw: DevelopmentDrawAudit | None
    candidate_sha256: str | None = None
    requested_root_count: int | None = None
    previous_root_count: int = 0
    new_root_count: int = 0
    cached: bool = False


@dataclass(frozen=True)
class _Stage:
    member_roots: tuple[str, ...]
    roots: tuple[str, ...]
    candidate: MacroBatchEvaluation
    incumbent: MacroBatchEvaluation


@dataclass
class _CandidateState:
    candidate: CandidateSubmission
    stages: list[_Stage] = field(default_factory=list)
    responses: dict[int, DevelopmentScreenResponse] = field(default_factory=dict)


class DevelopmentSession:
    """Thin synchronous controller for public releases and private screens.

    All budgets and evidence stages count independent clean roots. Perturbation
    children remain paired with their clean parent and do not consume another
    root credit.
    """

    def __init__(
        self,
        *,
        incumbent: CandidateSubmission,
        cohort_factory: DevelopmentCohortFactory,
        screen_root_stages: Sequence[int],
        screen_root_credit_limit: int,
        release_root_credit_limit: int,
        compute_budget: float | None = None,
        query_seconds_per_point_limit: float | None = None,
        safe_metric_names: Sequence[str] = (),
        cvar_tail_fraction: float = 0.1,
        confidence_z: float = 1.96,
        continuation_max_mean_ratio: float | None = 1.05,
        paired_objective_policy: PairedObjectivePolicy = PairedObjectivePolicy(),
        public_session_factory=None,
        public_session_methods: Sequence[str] = (),
        open_search: bool = False,
        release_order_rng: object | None = None,
    ) -> None:
        stages = tuple(screen_root_stages)
        metrics = tuple(safe_metric_names)
        if not isinstance(incumbent, CandidateSubmission) or not callable(
            cohort_factory
        ):
            raise TypeError("incumbent and cohort_factory are required")
        if (
            not stages
            or any(not _positive_int(value) for value in stages)
            or any(b <= a for a, b in zip(stages, stages[1:]))
        ):
            raise ValueError("screen_root_stages must strictly increase")
        if not _positive_int(screen_root_credit_limit):
            raise ValueError("screen_root_credit_limit must be positive")
        if not _nonnegative_int(release_root_credit_limit):
            raise ValueError("release_root_credit_limit must be nonnegative")
        if (
            isinstance(safe_metric_names, (str, bytes))
            or any(not isinstance(name, str) or not name.strip() for name in metrics)
            or len(set(metrics)) != len(metrics)
        ):
            raise ValueError("safe_metric_names must be unique nonempty strings")
        if not isfinite(cvar_tail_fraction) or not 0 < cvar_tail_fraction <= 1:
            raise ValueError("cvar_tail_fraction must lie in (0, 1]")
        if not isfinite(confidence_z) or confidence_z < 0:
            raise ValueError("confidence_z must be finite and nonnegative")
        if continuation_max_mean_ratio is not None and (
            not isfinite(continuation_max_mean_ratio)
            or continuation_max_mean_ratio <= 0
        ):
            raise ValueError("continuation_max_mean_ratio must be positive or None")
        if not isinstance(paired_objective_policy, PairedObjectivePolicy):
            raise TypeError("paired_objective_policy must be a PairedObjectivePolicy")
        if not isinstance(open_search, bool):
            raise TypeError("open_search must be bool")
        if public_session_factory is not None and not callable(
            public_session_factory
        ):
            raise TypeError("public session factory must be callable")
        if isinstance(public_session_methods, (str, bytes)):
            raise TypeError("public session methods must be a sequence")
        session_methods = tuple(public_session_methods)
        if any(
            not isinstance(name, str)
            or not name.isidentifier()
            or name.startswith("_")
            for name in session_methods
        ) or len(session_methods) != len(set(session_methods)):
            raise ValueError(
                "public session methods must be unique public identifiers"
            )
        if (public_session_factory is None) != (not session_methods):
            raise ValueError(
                "public session factory and explicit method allowlist must align"
            )
        order_rng = SystemRandom() if release_order_rng is None else release_order_rng
        order_shuffle = getattr(order_rng, "shuffle", None)
        if not callable(order_shuffle):
            raise TypeError("release_order_rng must provide shuffle")

        self.incumbent = incumbent
        self.cohort_factory = cohort_factory
        self.screen_root_stages = stages
        self.screen_root_credit_limit = screen_root_credit_limit
        self.release_root_credit_limit = release_root_credit_limit
        self.compute_budget = compute_budget
        self.query_seconds_per_point_limit = query_seconds_per_point_limit
        self.safe_metric_names = metrics
        self.cvar_tail_fraction = float(cvar_tail_fraction)
        self.confidence_z = float(confidence_z)
        self.continuation_max_mean_ratio = continuation_max_mean_ratio
        self.paired_objective_policy = paired_objective_policy
        self.public_session_factory = public_session_factory
        self.public_session_methods = session_methods
        self.open_search = open_search
        self._release_order_shuffle = order_shuffle
        self._screen_charged = self._release_charged = self._draw_count = 0
        self._screen_requests = self._release_requests = 0
        self._seen_roots: set[str] = set()
        self._states: dict[str, _CandidateState] = {}
        self._screen_ledger: list[DevelopmentLedgerEntry] = []
        self._release_ledger: list[DevelopmentLedgerEntry] = []
        self._latest_release: DevelopmentRelease | None = None
        self._latest_release_cohort: PreparedMacroCohort | None = None
        self._latest_release_controller_cases: tuple[Case, ...] | None = None
        self._active_cohorts: set[PreparedMacroCohort] = set()

    @property
    def screen_charged_root_credits(self) -> int:
        return self._screen_charged

    @property
    def screen_remaining_root_credits(self) -> int:
        if self.open_search:
            return self.screen_root_credit_limit
        return self.screen_root_credit_limit - self._screen_charged

    @property
    def release_charged_root_credits(self) -> int:
        return self._release_charged

    @property
    def release_remaining_root_credits(self) -> int:
        if self.open_search:
            return self.release_root_credit_limit
        return self.release_root_credit_limit - self._release_charged

    # Compatibility aliases: the legacy context calls them cases; units are roots.
    @property
    def charged_case_credits(self) -> int:
        return self.screen_charged_root_credits

    @property
    def remaining_case_credits(self) -> int:
        return self.screen_remaining_root_credits

    @property
    def screen_ledger(self) -> tuple[DevelopmentLedgerEntry, ...]:
        return tuple(self._screen_ledger)

    @property
    def release_ledger(self) -> tuple[DevelopmentLedgerEntry, ...]:
        return tuple(self._release_ledger)

    @property
    def screened_candidates(self) -> tuple[CandidateSubmission, ...]:
        return tuple(
            state.candidate
            for _, state in sorted(self._states.items())
        )

    @property
    def screened_candidate_summaries(
        self,
    ) -> tuple[DevelopmentScreenedCandidate, ...]:
        """Return latest aggregate evidence without private cases or truth."""

        summaries: list[DevelopmentScreenedCandidate] = []
        for digest, state in sorted(self._states.items()):
            if not state.responses:
                continue
            response = state.responses[max(state.responses)]
            candidate = response._aggregates.get("candidate")
            loss = (
                candidate.get("loss")
                if isinstance(candidate, Mapping)
                else None
            )
            paired = response._aggregates.get("paired")
            distributions = (
                candidate.get("distributions")
                if isinstance(candidate, Mapping)
                else None
            )
            generated = (
                distributions.get("generated")
                if isinstance(distributions, Mapping)
                else None
            )
            perturbed = (
                distributions.get("perturbed")
                if isinstance(distributions, Mapping)
                else None
            )
            composite = (
                distributions.get("composite")
                if isinstance(distributions, Mapping)
                else None
            )
            guardrail = response._aggregates.get("perturbed_guardrail")
            mean = _finite_aggregate_value(loss, "mean")
            cvar = _finite_aggregate_value(loss, "cvar")
            maximum = _finite_aggregate_value(loss, "max")
            paired_mean = _finite_aggregate_value(paired, "mean_improvement")
            guardrail_passed = (
                bool(guardrail.get("passed"))
                if isinstance(guardrail, Mapping)
                and isinstance(guardrail.get("passed"), bool)
                else None
            )
            guardrail_required = (
                self.paired_objective_policy.mode
                is PairedObjectiveMode.PRIMARY_WITH_PERTURBED_GUARDRAIL
                and response.completed_root_count >= self.screen_root_stages[-1]
            )
            eligible = bool(
                isinstance(candidate, Mapping)
                and candidate.get("eligible") is True
                and mean is not None
                and cvar is not None
                and maximum is not None
                and paired_mean is not None
                and (not guardrail_required or guardrail_passed is True)
            )
            if state.candidate.ref.bundle_sha256 != digest:
                raise DevelopmentSessionError(
                    "screened candidate digest registry changed"
                )
            summaries.append(
                DevelopmentScreenedCandidate(
                    candidate=state.candidate,
                    completed_root_count=response.completed_root_count,
                    eligible=eligible,
                    loss_mean=mean,
                    loss_cvar=cvar,
                    loss_max=maximum,
                    paired_mean_improvement=paired_mean,
                    generated_loss_mean=_distribution_loss_mean(generated),
                    perturbed_loss_mean=_distribution_loss_mean(perturbed),
                    composite_loss_mean=_distribution_loss_mean(composite),
                    perturbed_guardrail_passed=guardrail_passed,
                )
            )
        return tuple(summaries)


    @property
    def latest_release(self) -> DevelopmentRelease | None:
        if self._latest_release is None:
            return None
        return _release_copy(self._latest_release)

    def discard_latest_release(self) -> None:
        """Drop completed answers after their immutable manifest is published."""

        cohort = self._latest_release_cohort
        if cohort is not None:
            self._discard_cohort(cohort)
        self._latest_release_cohort = None
        self._latest_release_controller_cases = None
        self._latest_release = None

    def close(self) -> None:
        """Release all controller-private cohorts owned by this session."""

        failures: list[BaseException] = []
        for cohort in tuple(self._active_cohorts):
            try:
                self._discard_cohort(cohort)
            except BaseException as exc:
                failures.append(exc)
        if self._latest_release_cohort not in self._active_cohorts:
            self._latest_release_cohort = None
            self._latest_release_controller_cases = None
            self._latest_release = None
        if failures:
            first = failures[0]
            for failure in failures[1:]:
                first.add_note(
                    "additional cohort release failure: "
                    f"{type(failure).__name__}: {failure}"
                )
            raise first

    def _released(self) -> DevelopmentRelease:
        release = self.latest_release
        if release is None:
            raise DevelopmentSessionError(
                "call release() before building Agent context"
            )
        return release


    @property
    def public_cases(self) -> tuple[Case, ...]:
        return self._released().cases

    def _controller_release_cases(self) -> tuple[Case, ...]:
        """Return issued Cases for trusted controller/runtime binding only."""

        self._released()
        controller_cases = self._latest_release_controller_cases
        if controller_cases is None:
            raise DevelopmentSessionError(
                "released controller cohort is unavailable"
            )
        return tuple(_snapshot_value(case) for case in controller_cases)

    @property
    def public_queries(self) -> tuple[AnswerQuery, ...]:
        return self._released().queries

    @property
    def public_answers(self) -> tuple[AnswerView, ...]:
        return self._released().answers

    @property
    def cohort_id(self) -> str:
        return self._released().cohort_id

    @property
    def cohort_sha256(self) -> str:
        return self._released().cohort_sha256

    @property
    def contract_id(self) -> str:
        return self._released().contract_id

    @_release_new_cohorts_on_failure
    def release(self, root_count: int) -> DevelopmentRelease:
        """Release completed answers from fresh roots in a private fresh order.

        In open-search mode the configured release limit is a per-request
        capacity guard, not a lifetime allowance. Only the latest release is
        retained, bounding answer residency independently of request count.
        """

        if not _positive_int(root_count):
            raise ValueError("release root_count must be positive")
        remaining = self.release_remaining_root_credits
        if root_count > remaining:
            raise DevelopmentCreditExceeded(
                (
                    f"release request needs {root_count} roots but the per-request "
                    f"capacity is {remaining}"
                    if self.open_search
                    else f"release needs {root_count} root credits but only {remaining} remain"
                )
            )
        self.discard_latest_release()
        request = self._draw(DevelopmentDrawKind.RELEASE, root_count)
        cohort = self._fresh(request)
        member_roots, roots = _cohort_roots(cohort)
        public_cases = cohort.public_cases()
        public_queries = cohort.public_queries()
        public_answers = tuple(
            _snapshot_value(row.target.values) for row in cohort._prepared
        )
        controller_cases = cohort._controller_cases()
        order = list(range(len(public_cases)))
        self._release_order_shuffle(order)
        if sorted(order) != list(range(len(public_cases))):
            raise RuntimeError("release_order_rng returned an invalid permutation")
        self._claim(roots, root_count)
        self._release_charged += root_count
        self._release_requests += 1
        release = DevelopmentRelease(
            self._release_requests,
            root_count,
            len(member_roots),
            cohort.cohort_id,
            cohort.cohort_sha256,
            cohort.contract_id,
            tuple(public_cases[index] for index in order),
            tuple(public_queries[index] for index in order),
            tuple(public_answers[index] for index in order),
            root_count,
            self._release_charged,
            self.release_remaining_root_credits,
        )
        self._latest_release = release
        self._latest_release_cohort = cohort
        self._latest_release_controller_cases = tuple(
            controller_cases[index] for index in order
        )
        self._release_ledger.append(
            DevelopmentLedgerEntry(
                self._release_requests,
                root_count,
                _audit(request, cohort, len(member_roots)),
                new_root_count=root_count,
            )
        )
        return _release_copy(release)

    @_release_new_cohorts_on_failure
    def screen(
        self,
        candidate: CandidateSubmission,
        *,
        requested_root_count: int,
    ) -> DevelopmentScreenResponse:
        """Screen one attested digest on fresh incumbent-paired evidence.

        Open-search requests are independent fresh draws. The requested count
        is this call's size; repeated calls for one immutable digest accumulate
        evidence without cache, stage, continuation, or lifetime-quota rules.
        """

        if not isinstance(candidate, CandidateSubmission):
            raise TypeError("screen candidate must be a CandidateSubmission")
        digest = candidate.ref.bundle_sha256
        if digest == self.incumbent.ref.bundle_sha256:
            raise DevelopmentSessionError("incumbent cannot screen against itself")
        if candidate.ref.parent_sha256 != self.incumbent.ref.bundle_sha256:
            raise DevelopmentSessionError(
                "screen candidate parent must be the session incumbent"
            )
        if self.open_search:
            return self._screen_open(
                candidate,
                requested_root_count=requested_root_count,
            )
        if requested_root_count not in self.screen_root_stages:
            raise ValueError("requested_root_count is not a configured stage")

        state = self._states.get(digest)
        if state is not None and requested_root_count in state.responses:
            self._screen_requests += 1
            response = replace(
                state.responses[requested_root_count],
                charged_root_credits=0,
                cumulative_charged_root_credits=self._screen_charged,
                remaining_root_credits=self.screen_remaining_root_credits,
                cached=True,
            )
            self._screen_ledger.append(
                DevelopmentLedgerEntry(
                    self._screen_requests,
                    0,
                    None,
                    digest,
                    requested_root_count,
                    requested_root_count,
                    cached=True,
                )
            )
            return response

        previous = 0 if state is None else max(state.responses, default=0)
        next_index = 0 if previous == 0 else self.screen_root_stages.index(previous) + 1
        if (
            next_index >= len(self.screen_root_stages)
            or requested_root_count != self.screen_root_stages[next_index]
        ):
            raise DevelopmentSessionError(
                "screen evidence must advance by exactly one configured stage"
            )
        if previous and self.continuation_max_mean_ratio is not None:
            payload = state.responses[previous].public_payload()
            candidate_mean = _loss_mean(payload["candidate"])
            incumbent_mean = _loss_mean(payload["incumbent"])
            paired = payload.get("paired")
            confidence_upper = _finite_aggregate_value(
                paired,
                "confidence_upper",
            )
            if candidate_mean is None or incumbent_mean is None:
                raise DevelopmentSessionError(
                    "candidate did not clear the continuation mean-ratio gate"
                )
            allowed_excess = incumbent_mean * (
                self.continuation_max_mean_ratio - 1.0
            )
            point_estimate_is_worse = (
                candidate_mean
                > incumbent_mean * self.continuation_max_mean_ratio
            )
            uncertainty_confirms_regression = (
                confidence_upper is None
                or confidence_upper < -allowed_excess
            )
            if point_estimate_is_worse and uncertainty_confirms_regression:
                raise DevelopmentSessionError(
                    "candidate did not clear the continuation mean-ratio gate"
                )

        new_roots = requested_root_count - previous
        remaining = self.screen_remaining_root_credits
        if new_roots > remaining:
            raise DevelopmentCreditExceeded(
                f"screen needs {new_roots} root credits but only {remaining} remain"
            )
        request = self._draw(
            DevelopmentDrawKind.SCREEN,
            new_roots,
            candidate_sha256=digest,
            cumulative_screen_root_count=requested_root_count,
        )
        cohort = self._fresh(request)
        member_roots, roots = _cohort_roots(cohort)
        self._claim(roots, new_roots)
        evaluator = ControllerMacroEvaluator(
            cohort,
            compute_budget=self.compute_budget,
            query_seconds_per_point_limit=(
                self.query_seconds_per_point_limit
            ),
            public_session_factory=self.public_session_factory,
            public_session_methods=self.public_session_methods,
        )
        screened = candidate if state is None else state.candidate
        candidate_batch, incumbent_batch = evaluator.evaluate_pair(
            screened.ref.candidate_id,
            screened.program,
            self.incumbent.ref.candidate_id,
            self.incumbent.program,
        )
        _validate_pair(candidate_batch, incumbent_batch)
        if state is None:
            state = _CandidateState(candidate)
            self._states[digest] = state
        state.stages.append(
            _Stage(member_roots, roots, candidate_batch, incumbent_batch)
        )
        self._screen_charged += new_roots
        self._screen_requests += 1
        response = DevelopmentScreenResponse(
            digest,
            self.incumbent.ref.bundle_sha256,
            requested_root_count,
            sum(len(stage.member_roots) for stage in state.stages),
            self.screen_root_stages[: len(state.stages)],
            _aggregates(
                state.stages,
                state.candidate,
                self.incumbent,
                self.safe_metric_names,
                self.cvar_tail_fraction,
                self.confidence_z,
                self.paired_objective_policy,
            ),
            new_roots,
            self._screen_charged,
            self.screen_remaining_root_credits,
        )
        state.responses[requested_root_count] = response
        self._screen_ledger.append(
            DevelopmentLedgerEntry(
                self._screen_requests,
                new_roots,
                _audit(request, cohort, len(member_roots)),
                digest,
                requested_root_count,
                previous,
                new_roots,
            )
        )
        self._discard_cohort(cohort)
        return response

    def _screen_open(
        self,
        candidate: CandidateSubmission,
        *,
        requested_root_count: int,
    ) -> DevelopmentScreenResponse:
        """Evaluate one arbitrary-size fresh draw and accumulate by digest."""

        if not _positive_int(requested_root_count):
            raise ValueError("requested_root_count must be positive")
        if requested_root_count > self.screen_root_credit_limit:
            raise DevelopmentCreditExceeded(
                "screen request needs "
                f"{requested_root_count} roots but the per-request capacity is "
                f"{self.screen_root_credit_limit}"
            )
        digest = candidate.ref.bundle_sha256
        state = self._states.get(digest)
        if state is not None and state.candidate.ref != candidate.ref:
            raise DevelopmentSessionError(
                "screened candidate identity changed for an immutable digest"
            )
        previous = sum(len(stage.roots) for stage in state.stages) if state else 0
        completed = previous + requested_root_count
        request = self._draw(
            DevelopmentDrawKind.SCREEN,
            requested_root_count,
            candidate_sha256=digest,
            cumulative_screen_root_count=completed,
        )
        cohort = self._fresh(request)
        member_roots, roots = _cohort_roots(cohort)
        self._claim(roots, requested_root_count)
        evaluator = ControllerMacroEvaluator(
            cohort,
            compute_budget=self.compute_budget,
            query_seconds_per_point_limit=(
                self.query_seconds_per_point_limit
            ),
            public_session_factory=self.public_session_factory,
            public_session_methods=self.public_session_methods,
        )
        screened = candidate if state is None else state.candidate
        candidate_batch, incumbent_batch = evaluator.evaluate_pair(
            screened.ref.candidate_id,
            screened.program,
            self.incumbent.ref.candidate_id,
            self.incumbent.program,
        )
        _validate_pair(candidate_batch, incumbent_batch)
        if state is None:
            state = _CandidateState(candidate)
            self._states[digest] = state
        state.stages.append(
            _Stage(member_roots, roots, candidate_batch, incumbent_batch)
        )
        self._screen_charged += requested_root_count
        self._screen_requests += 1
        completed_counts: list[int] = []
        running = 0
        for stage in state.stages:
            running += len(stage.roots)
            completed_counts.append(running)
        response = DevelopmentScreenResponse(
            digest,
            self.incumbent.ref.bundle_sha256,
            completed,
            sum(len(stage.member_roots) for stage in state.stages),
            tuple(completed_counts),
            _aggregates(
                state.stages,
                state.candidate,
                self.incumbent,
                self.safe_metric_names,
                self.cvar_tail_fraction,
                self.confidence_z,
                self.paired_objective_policy,
            ),
            requested_root_count,
            self._screen_charged,
            self.screen_root_credit_limit,
            cached=False,
            protocol_lifetime_limit=False,
            per_request_root_limit=self.screen_root_credit_limit,
        )
        state.responses[completed] = response
        self._screen_ledger.append(
            DevelopmentLedgerEntry(
                self._screen_requests,
                requested_root_count,
                _audit(request, cohort, len(member_roots)),
                digest,
                requested_root_count,
                previous,
                requested_root_count,
            )
        )
        self._discard_cohort(cohort)
        return response

    def _draw(
        self,
        kind: DevelopmentDrawKind,
        root_count: int,
        **stage: object,
    ) -> DevelopmentDrawRequest:
        self._draw_count += 1
        return DevelopmentDrawRequest(kind, self._draw_count, root_count, **stage)

    def _fresh(self, request: DevelopmentDrawRequest) -> PreparedMacroCohort:
        cohort = self.cohort_factory(request)
        if not isinstance(cohort, PreparedMacroCohort):
            raise TypeError("cohort factory must return PreparedMacroCohort")
        self._active_cohorts.add(cohort)
        return cohort

    def _discard_cohort(self, cohort: PreparedMacroCohort) -> None:
        cohort.discard()
        self._active_cohorts.discard(cohort)

    def _claim(self, roots: Sequence[str], expected: int) -> None:
        if len(roots) != expected:
            raise DevelopmentSessionError("cohort returned the wrong root count")
        if set(roots) & self._seen_roots:
            raise DevelopmentSessionError("release/screen draws reused roots")
        self._seen_roots.update(roots)


def _cohort_roots(
    cohort: PreparedMacroCohort,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    # Pairing lineage remains controller-private.
    cases = cohort._controller_cases()
    clean = {case.case_id for case in cases if case.parent_case_id is None}
    members: list[str] = []
    ordered: list[str] = []
    previous = None
    for case in cases:
        root = case.parent_case_id or case.case_id
        if root not in clean or (root in ordered and root != previous):
            raise DevelopmentSessionError("invalid or noncontiguous root lineage")
        if root not in ordered:
            ordered.append(root)
        members.append(root)
        previous = root
    return tuple(members), tuple(ordered)


def _validate_pair(
    candidate: MacroBatchEvaluation,
    incumbent: MacroBatchEvaluation,
) -> None:
    left = (
        candidate.cohort_id,
        candidate.cohort_sha256,
        candidate.target_sha256,
        candidate.contract_id,
        candidate.case_ids,
    )
    right = (
        incumbent.cohort_id,
        incumbent.cohort_sha256,
        incumbent.target_sha256,
        incumbent.contract_id,
        incumbent.case_ids,
    )
    if left != right:
        raise DevelopmentSessionError("candidate and incumbent are not paired")


def _scores(
    stages: Sequence[_Stage],
    side: str,
) -> tuple[tuple[SanitizedMacroCaseScore, ...], tuple[str, ...]]:
    scores: list[SanitizedMacroCaseScore] = []
    roots: list[str] = []
    for stage in stages:
        batch = stage.candidate if side == "candidate" else stage.incumbent
        if len(batch.scores) != len(stage.member_roots):
            raise DevelopmentSessionError("score/root alignment changed")
        scores.extend(batch.scores)
        roots.extend(stage.member_roots)
    return tuple(scores), tuple(roots)


def _root_means(values: Sequence[float], roots: Sequence[str]) -> tuple[float, ...]:
    grouped: dict[str, list[float]] = {}
    for root, value in zip(roots, values):
        grouped.setdefault(root, []).append(float(value))
    return tuple(fmean(values) for values in grouped.values())


def _summary(
    stages: Sequence[_Stage],
    side: str,
    safe_metrics: Sequence[str],
    tail: float,
) -> Mapping[str, object]:
    scores, roots = _scores(stages, side)
    statuses = tuple(score.status for score in scores)
    loss = None
    if all(score.primary_error is not None for score in scores):
        loss = _robust(
            _root_means(tuple(float(score.primary_error) for score in scores), roots),
            tail,
        )
    metrics = {
        name: _robust(
            _root_means(tuple(float(score.metrics[name]) for score in scores), roots),
            tail,
        )
        for name in safe_metrics
        if all(name in score.metrics for score in scores)
    }
    return {
        "root_count": len(dict.fromkeys(roots)),
        "member_case_count": len(scores),
        "eligible": all(status == "valid" for status in statuses),
        "status_counts": {
            status: statuses.count(status)
            for status in ("valid", "rejected", "invalid")
        },
        "loss": loss,
        "metrics": metrics,
    }


def _selection_score(
    stages: Sequence[_Stage],
    submission: CandidateSubmission,
    side: str,
) -> MacroCandidateScore | None:
    scores, roots = _scores(stages, side)
    if any(score.primary_error is None for score in scores):
        return None
    batches = tuple(
        stage.candidate if side == "candidate" else stage.incumbent for stage in stages
    )
    contracts = {batch.contract_id for batch in batches}
    if len(contracts) != 1:
        raise DevelopmentSessionError("macro contract changed across fresh stages")
    target = hashlib.sha256(
        json.dumps(
            [batch.target_sha256 for batch in batches], separators=(",", ":")
        ).encode()
    ).hexdigest()
    failures = tuple(
        failure
        for score in scores
        if score.status == "rejected"
        for failure in (score.failures or ("macro_gate_rejected",))
    )
    return MacroCandidateScore(
        submission.ref.candidate_id,
        submission.ref.bundle_sha256,
        target,
        next(iter(contracts)),
        tuple(score.case_id for score in scores),
        tuple(float(score.primary_error) for score in scores),
        all(score.status != "invalid" for score in scores),
        failures,
        roots,
    )


def _score_views(
    score: MacroCandidateScore,
    policy: PairedObjectivePolicy,
) -> Mapping[str, MacroCandidateScore | None]:
    """Build one-score-per-root objective views without equation knowledge."""

    if score.root_ids is None:
        return {
            "generated": score,
            "perturbed": None,
            "composite": score,
            "active": score,
        }
    arities: list[int] = []
    for root_id in score.root_ids:
        if not arities or root_id != score.root_ids[sum(arities) - 1]:
            arities.append(1)
        else:
            arities[-1] += 1
    if all(arity == 1 for arity in arities):
        return {
            "generated": score,
            "perturbed": None,
            "composite": score,
            "active": score,
        }
    if any(arity != 2 for arity in arities):
        raise DevelopmentSessionError(
            "paired objective roots must have exactly base and perturbed members"
        )
    split = split_paired_score(score)
    composite = paired_weighted_composite(
        score,
        base_weight=policy.base_weight,
        perturbed_weight=policy.perturbed_weight,
    )
    return {
        "generated": split.base,
        "perturbed": split.perturbed,
        "composite": composite,
        "active": paired_objective_score(score, policy),
    }


def _score_summary(
    score: MacroCandidateScore | None,
    tail: float,
) -> Mapping[str, object] | None:
    if score is None:
        return None
    return {
        "root_count": len(score.errors),
        "loss": _robust(score.errors, tail),
    }


def _objective_summary(
    stages: Sequence[_Stage],
    submission: CandidateSubmission,
    side: str,
    safe_metrics: Sequence[str],
    tail: float,
    policy: PairedObjectivePolicy,
) -> tuple[Mapping[str, object], Mapping[str, MacroCandidateScore | None]]:
    summary = dict(_summary(stages, side, safe_metrics, tail))
    raw_score = _selection_score(stages, submission, side)
    if raw_score is None:
        summary["distributions"] = {
            "generated": None,
            "perturbed": None,
            "composite": None,
        }
        return summary, {}
    views = _score_views(raw_score, policy)
    active = views["active"]
    if active is None:
        raise DevelopmentSessionError("paired objective has no active score")
    summary["loss"] = _robust(active.errors, tail)
    summary["active_distribution"] = (
        "generated"
        if policy.mode is PairedObjectiveMode.PRIMARY_WITH_PERTURBED_GUARDRAIL
        else "composite"
    )
    summary["distributions"] = {
        name: _score_summary(views[name], tail)
        for name in ("generated", "perturbed", "composite")
    }
    return summary, views


def _paired_feedback(
    candidate: MacroCandidateScore | None,
    incumbent: MacroCandidateScore | None,
    confidence_z: float,
    tail: float,
) -> Mapping[str, object] | None:
    if candidate is None or incumbent is None or not incumbent.eligible:
        return None
    root_count = len(candidate.errors)
    z = confidence_z if root_count > 1 else 0.0
    comparison = paired_compare(
        candidate,
        incumbent,
        EvidenceStage(
            case_count=root_count,
            confidence_z=z,
            min_win_rate=0.0,
            require_cvar_no_regression=False,
        ),
        cvar_tail_fraction=tail,
    )
    return {
        "root_count": comparison.confidence_sample_count,
        "mean_improvement": comparison.mean_improvement,
        "standard_error": (
            0.0
            if z == 0.0
            else (
                comparison.confidence_upper - comparison.confidence_lower
            )
            / (2.0 * z)
        ),
        "confidence_lower": comparison.confidence_lower,
        "confidence_upper": comparison.confidence_upper,
        "win_rate": comparison.win_rate,
        "official_promotion_result": False,
    }


def _guardrail_feedback(
    candidate: MacroCandidateScore | None,
    incumbent: MacroCandidateScore | None,
    policy: PairedObjectivePolicy,
) -> Mapping[str, object]:
    if candidate is None or incumbent is None:
        return {
            "passed": False,
            "failures": ["paired_perturbation_evidence_unavailable"],
            "official_promotion_result": False,
        }
    comparison = compare_perturbed_non_regression(
        candidate,
        incumbent,
        policy=policy,
    )
    return {
        "root_count": len(comparison.root_ids),
        "candidate_mean": comparison.candidate_mean,
        "incumbent_mean": comparison.incumbent_mean,
        "mean_excess": comparison.mean_excess,
        "standard_error": comparison.standard_error,
        "confidence_lower": comparison.confidence_lower,
        "confidence_z": comparison.confidence_z,
        "practical_relative_margin": comparison.practical_relative_margin,
        "passed": comparison.passed,
        "failures": list(comparison.failures),
        "official_promotion_result": False,
    }


def _aggregates(
    stages: Sequence[_Stage],
    candidate_submission: CandidateSubmission,
    incumbent_submission: CandidateSubmission,
    safe_metrics: Sequence[str],
    tail: float,
    confidence_z: float,
    policy: PairedObjectivePolicy,
) -> Mapping[str, object]:
    candidate, candidate_views = _objective_summary(
        stages,
        candidate_submission,
        "candidate",
        safe_metrics,
        tail,
        policy,
    )
    incumbent, incumbent_views = _objective_summary(
        stages,
        incumbent_submission,
        "incumbent",
        safe_metrics,
        tail,
        policy,
    )
    paired_by_distribution = {
        name: _paired_feedback(
            candidate_views.get(name),
            incumbent_views.get(name),
            confidence_z,
            tail,
        )
        for name in ("generated", "perturbed", "composite")
    }
    active_name = (
        "generated"
        if policy.mode is PairedObjectiveMode.PRIMARY_WITH_PERTURBED_GUARDRAIL
        else "composite"
    )
    guardrail = None
    if policy.mode is PairedObjectiveMode.PRIMARY_WITH_PERTURBED_GUARDRAIL:
        guardrail = _guardrail_feedback(
            _selection_score(stages, candidate_submission, "candidate"),
            _selection_score(stages, incumbent_submission, "incumbent"),
            policy,
        )
    return {
        "objective": {
            "mode": policy.mode.value,
            "base_weight": policy.normalized_base_weight,
            "perturbed_weight": policy.normalized_perturbed_weight,
            "generated_primary": (
                policy.mode
                is PairedObjectiveMode.PRIMARY_WITH_PERTURBED_GUARDRAIL
            ),
            "perturbed_role": (
                "non_regression_guardrail"
                if policy.mode
                is PairedObjectiveMode.PRIMARY_WITH_PERTURBED_GUARDRAIL
                else "weighted_objective"
            ),
            "guardrail_relative_margin": policy.guardrail_relative_margin,
            "guardrail_confidence_z": policy.guardrail_confidence_z,
        },
        "candidate": candidate,
        "incumbent": incumbent,
        "paired": paired_by_distribution[active_name],
        "paired_by_distribution": paired_by_distribution,
        "perturbed_guardrail": guardrail,
    }


def _robust(values: Sequence[float], tail: float) -> Mapping[str, float]:
    result = aggregate(values, cvar_tail_fraction=tail)
    return {"mean": result.mean, "cvar": result.cvar, "max": result.max}


def _loss_mean(summary: object) -> float | None:
    if not isinstance(summary, Mapping) or not isinstance(summary.get("loss"), Mapping):
        return None
    value = summary["loss"].get("mean")
    return float(value) if isinstance(value, (int, float)) else None


def _distribution_loss_mean(summary: object) -> float | None:
    if not isinstance(summary, Mapping):
        return None
    return _loss_mean(summary)


def _finite_aggregate_value(summary: object, field: str) -> float | None:
    if not isinstance(summary, Mapping):
        return None
    value = summary.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
    ):
        return None
    return float(value)


def _audit(
    request: DevelopmentDrawRequest,
    cohort: PreparedMacroCohort,
    members: int,
) -> DevelopmentDrawAudit:
    return DevelopmentDrawAudit(
        request.draw_index,
        request.kind,
        request.root_count,
        members,
        cohort.cohort_id,
        cohort.cohort_sha256,
    )


def _release_copy(release: DevelopmentRelease) -> DevelopmentRelease:
    return replace(
        release,
        cases=tuple(_snapshot_value(value) for value in release.cases),
        queries=tuple(_snapshot_value(value) for value in release.queries),
        answers=tuple(_snapshot_value(value) for value in release.answers),
    )


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
