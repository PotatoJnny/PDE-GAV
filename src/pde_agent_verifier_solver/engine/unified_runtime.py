"""The unified, Agent-adaptive product loop used by the panel.

Both development strategies use ``RoundCaseSampler``.  In adaptive runs the
Agent calls it while working; in a fixed-recipe ablation the Controller calls
the same service before the Agent starts.  Official Cases are fresh and common
to every candidate in a round.
"""

from __future__ import annotations

from .agent_prompt import build_agent_context_prompt

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import json
import logging
import os
from pathlib import Path
import traceback
from math import ceil, isfinite
from random import Random
from statistics import fmean
import sys
from threading import Lock
from typing import Callable, Mapping, Sequence

from pde_agent_verifier_solver.sdk import (
    Artifact,
    ArtifactEvaluation,
    Case,
    CaseFeedback,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    GenerationPurpose,
    GeneratorFeedback,
    GeneratorState,
    HiddenSuiteSpec,
    EvaluationPhase,
    PairedSampleBatch,
    PairedSampleRequest,
    PerturbCasesRequest,
    PromotionCandidateReport,
    PromotionContext,
    PromotionMetricSummary,
    SampleBatch,
    SampleCasesRequest,
    SampleSource,
    SamplingStrategy,
    TerminalCohortSpec,
    TerminalCandidateReport,
    TerminalGuidanceSource,
)

from pde_agent_verifier_solver.sdk.evaluation_devices import bind_evaluation_device_pool

from .agent_execution import (
    AgentExecutionRequest,
    AgentExecutionTimeout,
    AgentInfrastructureError,
    execute_reserved_agent_batch,
    reserve_agent_batch,
)
from .campaign import CaseCounts, RoundPlan
from .case_sampler import CaseReservationConflict, RoundCaseSampler
from .candidate_execution import CandidateExecutionInfrastructureError
from .evaluator import AgentEvaluator, HiddenVerifier
from .memory import (
    AgentRoundLog,
    CampaignMemoryLog,
    CandidateHistory,
)
from .promotion import (
    build_promotion_program,
    promotion_metric_references,
    validate_promotion_decision,
)
from .round_recovery import RoundDevelopmentRecovery
from .campaign_continuation import (CampaignContinuation, CampaignPauseControl, CampaignPaused, RestoredCaseLease)
from .seeds import SeedLedger
from .runtime import (
    AgentContext,
    AgentOutcome,
    CampaignResult,
    CampaignRuntimeSupport,
    RoundResult,
    _unique_agent_nominations,
)


_LOG = logging.getLogger(__name__)


class HiddenEvaluationInfrastructureError(CandidateExecutionInfrastructureError):
    """A hidden comparison stopped because Controller evaluation failed."""


class _InvalidCandidateMetrics(ValueError):
    """A scored candidate produced nonfinite or negative primary errors."""


class IncumbentEvaluationUnavailable(RuntimeError):
    """The current comparison cannot rank its required incumbent."""


def _evaluation_error_chain(error: BaseException) -> tuple[BaseException, ...]:
    pending = [error]
    seen: set[int] = set()
    errors: list[BaseException] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        errors.append(current)
        pending.extend(
            item for item in (current.__cause__, current.__context__)
            if item is not None
        )
        for attribute in ("_candidate_cleanup_errors", "_pde_session_cleanup_errors"):
            pending.extend(getattr(current, attribute, ()))
    return tuple(errors)


def _has_evaluation_infrastructure_error(error: BaseException) -> bool:
    return any(
        isinstance(item, CandidateExecutionInfrastructureError)
        for item in _evaluation_error_chain(error)
    )


_DISTRIBUTIONS = (
    "original",
    "base",
    "guided",
    "base_perturbed",
    "guided_perturbed",
)


def _public_agent_failure_code(error: BaseException) -> str:
    """Classify Agent failures without copying private exception details."""

    if isinstance(error, AgentExecutionTimeout):
        return "agent_timeout"
    if isinstance(error, AgentInfrastructureError):
        return "agent_infrastructure_failure"
    return "agent_execution_failed"


def _public_round_abort_code(error: BaseException) -> str:
    """Return a stable cross-round code rather than an arbitrary class name."""

    if isinstance(error, CandidateExecutionInfrastructureError):
        return "hidden_evaluation_infrastructure_failure"
    if isinstance(error, IncumbentEvaluationUnavailable):
        return "incumbent_evaluation_unavailable"
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        return "controller_interrupted"
    if isinstance(error, (AgentExecutionTimeout, AgentInfrastructureError)):
        return _public_agent_failure_code(error)
    return "round_execution_failed"


class _InteractiveSampler:
    """Register each successful batch with this Agent's Evaluator."""

    def __init__(
        self,
        inner: RoundCaseSampler,
        evaluator: AgentEvaluator,
        allowed_sources: Sequence[SampleSource],
        *,
        starter_only_sources: Sequence[SampleSource] = (),
        allow_guidance: bool,
        allow_perturbations: bool,
    ) -> None:
        self._inner = inner
        self._evaluator = evaluator
        self._allowed_sources = frozenset(allowed_sources)
        self._starter_only_sources = frozenset(starter_only_sources)
        if not self._starter_only_sources.issubset(self._allowed_sources):
            raise ValueError(
                "starter-only sources must be included in allowed sources"
            )
        self._starter_open = True
        self._allow_guidance = allow_guidance
        self._allow_perturbations = allow_perturbations
        self._distribution_by_case: dict[str, str] = {}
        self._counts = {name: 0 for name in _DISTRIBUTIONS}

    @property
    def counts(self) -> Mapping[str, int]:
        return dict(self._counts)

    def seal_starter(self) -> None:
        """Close Controller-only starter sources before Agent execution."""

        self._starter_open = False

    def _require_source(self, source: SampleSource) -> None:
        if source not in self._allowed_sources:
            raise PermissionError(
                f"{source.value} sampling is not enabled in this round"
            )
        if not self._starter_open and source in self._starter_only_sources:
            raise PermissionError(
                f"{source.value} sampling is available only in the "
                "Controller-provided starter set"
            )

    def sample_cases(self, request: SampleCasesRequest) -> SampleBatch:
        self._require_source(request.source)
        if request.source is SampleSource.GUIDED and not self._allow_guidance:
            raise PermissionError("guided sampling is disabled for this Run")
        batch = self._inner.sample_cases(request)
        name = request.source.value
        projected = self._register(
            batch,
            {case.case_id: name for case in batch.cases},
        )
        return SampleBatch(projected, batch.remaining)

    def sample_paired_cases(
        self,
        request: PairedSampleRequest,
    ) -> PairedSampleBatch:
        self._require_source(request.source)
        if request.source is SampleSource.GUIDED and not self._allow_guidance:
            raise PermissionError("guided sampling is disabled for this Run")
        if not self._allow_perturbations:
            raise PermissionError("perturbation is disabled for this Run")
        batch = self._inner.sample_paired_cases(request)
        root_name = request.source.value
        kinds = {
            **{case.case_id: root_name for case in batch.roots},
            **{
                case.case_id: f"{root_name}_perturbed"
                for case in batch.children
            },
        }
        projected = self._register(batch, kinds)
        root_count = len(batch.roots)
        return PairedSampleBatch(
            roots=projected[:root_count],
            children=projected[root_count:],
            remaining=batch.remaining,
            # Seeds, raw ids and retry diagnostics stay Controller-private.
            attempts=(),
        )

    def perturb_cases(self, request: PerturbCasesRequest) -> SampleBatch:
        if not self._allow_perturbations:
            raise PermissionError("perturbation is disabled for this Run")
        controller_parent_ids = (
            self._evaluator._controller_case_ids_for_candidate_ids(
                request.parent_case_ids
            )
        )
        parent_kinds: dict[str, str] = {}
        for candidate_id, controller_id in zip(
            request.parent_case_ids,
            controller_parent_ids,
        ):
            root_kind = self._distribution_by_case.get(candidate_id)
            if root_kind not in ("base", "guided"):
                raise ValueError("only issued base or guided roots can be perturbed")
            parent_kinds[controller_id] = f"{root_kind}_perturbed"
        batch = self._inner.perturb_cases(
            PerturbCasesRequest(controller_parent_ids)
        )
        kinds = {
            case.case_id: parent_kinds[str(case.parent_case_id)]
            for case in batch.cases
        }
        projected = self._register(batch, kinds)
        return SampleBatch(projected, batch.remaining)

    def _register(
        self,
        batch: SampleBatch | PairedSampleBatch,
        kinds: Mapping[str, str],
    ) -> tuple[Case, ...]:
        controller_ids = tuple(case.case_id for case in batch.cases)
        if set(kinds) != set(controller_ids):
            raise RuntimeError("sample distribution labels do not match the batch")
        projected = self._evaluator._candidate_cases_for_controller_ids(
            controller_ids
        )
        for candidate, controller_id in zip(projected, controller_ids):
            name = kinds[controller_id]
            self._distribution_by_case[candidate.case_id] = name
            self._counts[name] += 1
        return projected


class _FixedSamplerView:
    """Explain why dynamic draws are unavailable in a fixed-recipe ablation."""

    def sample_cases(self, request: SampleCasesRequest) -> SampleBatch:
        del request
        raise RuntimeError("this ablation uses the Controller's fixed Case recipe")

    def sample_paired_cases(
        self,
        request: PairedSampleRequest,
    ) -> PairedSampleBatch:
        del request
        raise RuntimeError("this ablation uses the Controller's fixed Case recipe")

    def perturb_cases(self, request: PerturbCasesRequest) -> SampleBatch:
        del request
        raise RuntimeError("this ablation uses the Controller's fixed Case recipe")


class CampaignController(CampaignRuntimeSupport):
    """The sole Generator-Agent-Verifier Campaign state machine."""

    def run(
        self,
        *,
        recovery: RoundDevelopmentRecovery | None = None,
        continuation: CampaignContinuation | None = None,
        pause_control: CampaignPauseControl | None = None,
    ) -> CampaignResult:
        """Execute the Campaign or continue its saved, completed scientific work.

        ``continuation`` preserves this run's history, Cases and completed work.
        The separate ``recovery`` input remains a first-round replay into a new
        run with empty memory/history; it is not a continuation.
        """
        if recovery is not None and not isinstance(recovery, RoundDevelopmentRecovery):
            raise TypeError("recovery must be RoundDevelopmentRecovery or None")
        if continuation is not None and not isinstance(continuation, CampaignContinuation):
            raise TypeError("continuation must be CampaignContinuation or None")
        if pause_control is not None and not isinstance(pause_control, CampaignPauseControl):
            raise TypeError("pause_control must be CampaignPauseControl or None")
        if continuation is not None and recovery is not None:
            raise ValueError("continuation and first-round replay are mutually exclusive")
        self._pause_control = pause_control
        self._continuation_state = {} if continuation is None else continuation.state
        self._round_development_recovery = recovery
        self._loaded_generator_state = None
        self._case_samplers: list[RoundCaseSampler] = []
        self._pending_feedback_samplers: list[RoundCaseSampler] = []
        self._terminal_case_samplers: list[RoundCaseSampler] = []
        self._terminal_hidden_case_ids: set[str] = set()
        self._initial_guidance_hidden_case_ids: set[str] = set()
        self._development_case_ids: set[str] = set()
        self._development_case_owners: dict[str, tuple[int, CaseSource]] = {}
        self._promotion_hidden_case_ids: set[str] = set()
        self._case_registry_lock = Lock()
        self._generator_operation_lock = Lock()
        self._next_candidate_sequence = 0
        self._seeds = SeedLedger(self.campaign.master_seed)
        self._campaign_memory = CampaignMemoryLog()
        self._candidate_history = CandidateHistory()
        self._active_round_plan: RoundPlan | None = None
        self._active_agent_logs: tuple[AgentRoundLog, ...] = ()
        self._active_round_memory_written = False
        self._restored_sampling_attempts = ()
        self._next_case_lease = 0
        if continuation is not None:
            self._restore_continuation_controller(continuation.controller_state)
            self.events.emit("run_resumed", "Continuing saved Campaign work", payload={"phase": self._continuation_state.get("phase")})
        try:
            try:
                return self._run_campaign()
            except CampaignPaused as paused:
                self.events.emit("run_paused", "Campaign saved for the next allocation", round_index=(self._active_round_plan.round_index if self._active_round_plan else None), payload={"phase": self._continuation_state.get("phase"), "stage": paused.stage})
                paused.continuation = self._save_continuation()
                raise
            except BaseException as error:
                try:
                    self._preserve_aborted_round(error)
                except BaseException as memory_error:
                    error.add_note(
                        "Campaign memory preservation also failed: "
                        f"{type(memory_error).__name__}: {memory_error}"
                    )
                raise
        finally:
            active_error = sys.exc_info()[1]
            try:
                self._close_case_samplers(self._case_samplers)
            except BaseException as cleanup_error:
                if active_error is None:
                    raise
                active_error.add_note(
                    "Campaign Case cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
                if isinstance(active_error, CampaignPaused):
                    raise CandidateExecutionInfrastructureError(
                        "Campaign pause could not confirm Case cleanup"
                    ) from cleanup_error

    @staticmethod
    def _close_case_samplers(
        samplers: Sequence[RoundCaseSampler],
    ) -> None:
        errors: list[BaseException] = []
        seen: set[int] = set()
        for sampler in samplers:
            identity = id(sampler)
            if identity in seen:
                continue
            seen.add(identity)
            try:
                sampler.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            first = errors[0]
            for additional in errors[1:]:
                first.add_note(
                    "Another Case cohort cleanup failed: "
                    f"{type(additional).__name__}: {additional}"
                )
            raise first

    def _case_lease_for(self, lease_id):
        if lease_id is None:
            return None
        return next((sampler for sampler in self._case_samplers
                     if getattr(sampler, "_continuation_id", None) == lease_id), None)

    def _save_continuation(self):
        control = getattr(self, "_pause_control", None)
        if control is None:
            return None
        leases = []
        for sampler in self._case_samplers:
            identifiers = sampler.generated_case_ids
            if not identifiers:
                continue
            if not hasattr(sampler, "_continuation_id"):
                sampler._continuation_id = self._next_case_lease
                self._next_case_lease += 1
            leases.append({"lease_id": sampler._continuation_id,
                           "generator_state": sampler._generator_state,
                           "case_ids": tuple(identifiers),
                           "pending_feedback": sampler in self._pending_feedback_samplers,
                           "terminal": sampler in self._terminal_case_samplers})
        controller = {
            "master_seed": self.campaign.master_seed,
            "next_candidate_sequence": self._next_candidate_sequence,
            "next_case_lease": self._next_case_lease,
            "seed_ledger": self._seeds.snapshot(),
            "memory": self._campaign_memory.snapshot(),
            "candidate_history": self._candidate_history.snapshot(),
            "events": self.events.snapshot(),
            "leases": tuple(leases),
            "sampling_attempts": tuple(self._restored_sampling_attempts) + tuple(
                attempt for sampler in self._case_samplers for attempt in sampler.sampling_attempts),
        }
        for name in ("terminal_hidden_case_ids", "initial_guidance_hidden_case_ids", "development_case_ids",
                     "development_case_owners", "promotion_hidden_case_ids"):
            value = getattr(self, "_" + name)
            controller[name] = dict(value) if isinstance(value, dict) else set(value)
        continuation = CampaignContinuation(self._continuation_state, controller)
        try:
            control.save(continuation)
        except Exception as error:
            failure = CandidateExecutionInfrastructureError("Controller could not persist the Campaign continuation")
            failure._hidden_evaluation_stage = "save_campaign_continuation"
            raise failure from error
        return continuation

    def _checkpoint_boundary(self, stage, *, required_seconds=0.0):
        control = getattr(self, "_pause_control", None)
        if control is None:
            return
        continuation = self._save_continuation()
        try:
            requested = control.should_pause(stage, float(required_seconds))
        except Exception as error:
            raise CandidateExecutionInfrastructureError("Controller could not check the allocation deadline") from error
        if requested:
            raise CampaignPaused(continuation, stage)

    def _restore_continuation_controller(self, saved):
        if saved.get("master_seed") != self.campaign.master_seed:
            raise ValueError("Campaign continuation master seed differs")
        self._next_candidate_sequence = saved["next_candidate_sequence"]
        self._next_case_lease = saved["next_case_lease"]
        self._seeds._entries = dict(saved["seed_ledger"])
        self._campaign_memory._rounds = list(saved["memory"].rounds)
        self._candidate_history._nodes = list(saved["candidate_history"].nodes)
        self._candidate_history._artifact_to_node = {
            node.artifact_id: node.node_id for node in saved["candidate_history"].nodes}
        self.events._events = list(saved["events"])
        self._restored_sampling_attempts = tuple(saved["sampling_attempts"])
        for name in ("terminal_hidden_case_ids", "initial_guidance_hidden_case_ids", "development_case_ids",
                     "development_case_owners", "promotion_hidden_case_ids"):
            value = saved[name]
            setattr(self, "_" + name, dict(value) if isinstance(value, dict) else set(value))
        for row in saved["leases"]:
            sampler = RestoredCaseLease(self.generator, row["generator_state"], row["case_ids"], self._generator_operation_lock)
            sampler._continuation_id = row["lease_id"]
            self._case_samplers.append(sampler)
            if row["pending_feedback"]:
                self._pending_feedback_samplers.append(sampler)
            if row["terminal"]:
                self._terminal_case_samplers.append(sampler)

    def _run_campaign(self) -> CampaignResult:
        self._validate_agents()
        state = self._continuation_state
        if state.get("phase") in {"base_training_running", "development_running", "guidance_running", "terminal_guidance_running"}:
            raise RuntimeError("Campaign continuation cannot repeat an interrupted Agent or Guidance training budget; operator review is required")
        if state.get("phase") == "complete":
            return state["result"]
        if not state:
            initial_candidate = (self._freeze_initial_candidate(self.campaign.initial_candidate)
                                 if self.campaign.initial_candidate is not None else None)
            self.events.emit("run_started", f"Started {self.campaign.name}")
            generator_settings = {"basis_size": self.campaign.generator.basis_size,
                                  **self.campaign.generator.settings,
                                  "master_seed": self.campaign.master_seed,
                                  "controller_seed": self._seeds.derive("generator", "base")}
            checkpoint = self.campaign.generator.checkpoint
            if checkpoint is None:
                state["phase"] = "base_training_running"
                self._save_continuation()
                checkpoint = self.generator.train(self.dataset, generator_settings,
                                                  self._generator_progress(None, "base_training"))
            else:
                self.events.emit("generator_checkpoint_selected", "Using the existing Generator checkpoint selected by the user",
                                 payload={"generator_id": checkpoint.generator_id})
            generator_state = self.generator.load(checkpoint, generator_settings,
                                                  self._generator_progress(None, "base_loading"))
            self._loaded_generator_state = generator_state
            self.events.emit("base_ready", "Base Generator is ready")
            if initial_candidate is not None:
                self._candidate_history.add_candidate(initial_candidate)
            state.update(phase="bootstrap", initial_candidate=initial_candidate,
                         generator_settings=generator_settings, generator_state=generator_state,
                         terminal_hidden={}, terminal_generator_states={}, original_hidden_offset=0,
                         incumbent=(ArtifactEvaluation(initial_candidate, float("inf"), {}, ())
                                    if initial_candidate is not None else None),
                         feedback=None, development_original_offset=0, round_results=[], round_data={},
                         hidden_suites={}, hidden_progress={})
            self._save_continuation()
        initial_candidate = state["initial_candidate"]
        generator_settings = state["generator_settings"]
        generator_state = state["generator_state"]
        self._loaded_generator_state = generator_state
        terminal_spec = self.campaign.terminal_evaluation
        if terminal_spec is None:
            raise RuntimeError("compiled Campaign has no terminal evaluation")
        terminal_hidden = state["terminal_hidden"]
        terminal_generator_states = state["terminal_generator_states"]
        original_hidden_offset = state["original_hidden_offset"]
        incumbent, feedback = state["incumbent"], state["feedback"]
        development_original_offset = state["development_original_offset"]
        round_results = state["round_results"]
        memory, candidate_history = self._campaign_memory, self._candidate_history
        promotion = build_promotion_program(self.campaign.promotion)
        recovery = self._round_development_recovery
        recovered_development = (self._restore_round_development(recovery, self.campaign.rounds[0], initial_candidate)
                                 if recovery is not None else None)
        if state["phase"] == "bootstrap":
            for cohort in terminal_spec.cohorts:
                if cohort.guidance_source is not TerminalGuidanceSource.BASE:
                    continue
                if any(key.startswith(cohort.cohort_id + "/") for key in terminal_hidden):
                    continue
                original_hidden_offset = self._seal_terminal_cohort(
                    terminal_hidden, cohort, generator_state, self.campaign.rounds[0],
                    original_hidden_offset, generator_settings, terminal_generator_states, generation_round_index=0)
                state["original_hidden_offset"] = original_hidden_offset
                self._checkpoint_boundary("terminal_cohort_sealed")
            if incumbent is not None and self.campaign.generator.guidance_enabled and self._round_needs_guidance(self.campaign.rounds[0]):
                feedback = self._initial_guidance_feedback(incumbent.artifact, generator_state, self.campaign.rounds[0], generator_settings)
            state.update(feedback=feedback, phase="round_prepare")
            # Completed feedback now owns its ordinary pending Case lease.
            # Raw hidden truth is only retained while that evaluation can resume.
            state["hidden_suites"].clear()
            state["hidden_progress"].clear()
            self._checkpoint_boundary("bootstrap_completed")

        for plan in self.campaign.rounds:
            if plan.round_index <= len(round_results):
                continue
            self._active_round_plan = plan
            data = state["round_data"]
            self._active_agent_logs = data.get("agent_logs", ())
            self._active_round_memory_written = False
            policy = plan.development_sampling
            if state["phase"] == "round_prepare":
                if not data.get("started"):
                    self.events.emit("round_started", f"Round {plan.round_index} started", round_index=plan.round_index,
                                     payload={"sampling_strategy": policy.strategy.value, "sample_limit": policy.max_cases_per_agent_per_round})
                    data["started"] = True
                if feedback is not None and self.campaign.generator.guidance_enabled and not data.get("guidance_ready"):
                    self._checkpoint_boundary("before_guidance")
                    guidance_settings = {**generator_settings, "controller_seed": self._seeds.derive("generator", "guidance", plan.round_index)}
                    state["phase"] = "guidance_running"
                    self._save_continuation()
                    generator_state = self.generator.update_guidance(generator_state, feedback, guidance_settings,
                                                                    self._generator_progress(plan.round_index, "guidance_training"))
                    self._loaded_generator_state = generator_state
                    self._close_case_samplers(self._pending_feedback_samplers)
                    self._pending_feedback_samplers.clear()
                    state["generator_state"] = generator_state
                    data["guidance_ready"] = True
                    state["phase"] = "round_prepare"
                    self.events.emit("guidance_ready", "Guidance is trained once and frozen for this round", round_index=plan.round_index)
                    self._checkpoint_boundary("guidance_ready")
                for cohort in terminal_spec.cohorts:
                    if cohort.guidance_source is TerminalGuidanceSource.ROUND and cohort.round_index == plan.round_index:
                        if any(key.startswith(cohort.cohort_id + "/") for key in terminal_hidden):
                            continue
                        original_hidden_offset = self._seal_terminal_cohort(
                            terminal_hidden, cohort, generator_state, plan, original_hidden_offset,
                            generator_settings, terminal_generator_states, generation_round_index=plan.round_index)
                        state["original_hidden_offset"] = original_hidden_offset
                        self._checkpoint_boundary("terminal_cohort_sealed")
                state["phase"] = "development_ready"
                self._save_continuation()
            recovering_development = recovery is not None and plan.round_index == recovery.round_index
            if state["phase"] == "development_ready":
                self._checkpoint_boundary("before_development", required_seconds=plan.agent_budget_seconds +
                                          (self._pause_control.agent_start_margin_seconds if self._pause_control else 0))
                state["phase"] = "development_running"
                self._save_continuation()
                if recovering_development:
                    outcomes, agent_logs, agent_counts, development_original_offset = recovered_development
                    self.events.emit("round_development_recovered", "Frozen Agent submissions and notes restored; development was not rerun",
                                     round_index=plan.round_index, payload={**recovery.audit_metadata, "artifact_ids": [a.artifact_id for a in recovery.artifacts], "agent_case_counts": agent_counts})
                else:
                    outcomes, agent_logs, agent_counts, development_original_offset = self._develop_round(
                        plan, generator_state, generator_settings, incumbent, memory, candidate_history, development_original_offset)
                data.update(outcomes=outcomes, agent_logs=agent_logs, agent_counts=agent_counts)
                self._active_agent_logs = agent_logs
                state.update(development_original_offset=development_original_offset, phase="development_done")
                self._checkpoint_boundary("development_completed")
            outcomes, agent_logs, agent_counts = data["outcomes"], data["agent_logs"], data["agent_counts"]
            nominations = _unique_agent_nominations(outcomes)
            if not nominations and incumbent is None:
                raise RuntimeError(f"round {plan.round_index} has no incumbent or submitted Candidate")
            if state["phase"] == "development_done":
                self._record_submitted_candidates(candidate_history, tuple(nomination.artifact for nomination in nominations))
                hidden, original_hidden_offset, official_sampler = self._hidden_suite(
                    generator_state, plan, plan.round_hidden, GenerationPurpose.PROMOTION, original_hidden_offset, generator_settings)
                self._reserve_hidden_case_ids(tuple(record.case.case_id for records in hidden.values() for record in records), purpose="Official")
                data.update(hidden=hidden, official_lease_id=getattr(official_sampler, "_continuation_id", None))
                state.update(original_hidden_offset=original_hidden_offset, phase="official_ready")
                self._checkpoint_boundary("official_sampled")
            hidden = data["hidden"]
            official_sampler = self._case_lease_for(data.get("official_lease_id"))
            artifacts = [nomination.artifact for nomination in nominations]
            if incumbent is not None:
                artifacts.append(incumbent.artifact)
            if state["phase"] == "official_ready":
                if recovering_development:
                    verification = recovery.verify_official(plan, hidden, tuple(artifacts))
                    if not isinstance(verification, Mapping) or verification.get("status") != "passed":
                        raise ValueError("recovery cohort verification did not explicitly pass")
                    self.events.emit("round_recovery_official_verified", "Recovered Official input checks passed; see verification coverage",
                                     round_index=plan.round_index, payload={**recovery.audit_metadata, "verification": dict(verification)})
                reports, evaluations = self._official_reports(artifacts, hidden, seed_namespace=f"official_round:{plan.round_index}")
                data.update(reports=reports, evaluations=evaluations)
                state["phase"] = "official_done"
                state["hidden_suites"].clear()
                state["hidden_progress"].pop((f"official_round:{plan.round_index}", EvaluationPhase.OFFICIAL.value), None)
                self._checkpoint_boundary("official_completed")
            reports, evaluations = data["reports"], data["evaluations"]
            reports_by_id = {
                report.candidate_digest: report for report in reports
            }
            for artifact in artifacts:
                if not candidate_history.needs_official_result(artifact.artifact_id):
                    continue
                report = reports_by_id[artifact.artifact_id]
                candidate_history.record_official_result(
                    artifact.artifact_id,
                    report.distributions,
                    report.failures,
                    technically_eligible=report.technically_eligible,
                )

            incumbent_digest = (
                incumbent.artifact.artifact_id
                if incumbent is not None
                else None
            )
            if (
                incumbent_digest is not None
                and not reports_by_id[incumbent_digest].technically_eligible
            ):
                self.events.emit(
                    "promotion_blocked",
                    "Promotion needs a complete current Official evaluation of the incumbent",
                    round_index=plan.round_index,
                    payload={
                        "reason_code": "incumbent_evaluation_unavailable",
                        "candidate_id": incumbent_digest,
                    },
                )
                raise IncumbentEvaluationUnavailable(
                    "Promotion stopped: the incumbent has no complete eligible "
                    "evaluation on this round's Official Cases; no selection was "
                    "made and no prior score was reused"
                )
            eligible = tuple(report for report in reports if report.technically_eligible)
            if not eligible:
                raise RuntimeError("every program failed the official round evaluation")
            decision = promotion.select(
                incumbent_digest,
                reports,
                PromotionContext(round_index=plan.round_index),
            )
            validate_promotion_decision(incumbent_digest, reports, decision)
            selected_artifact_id = decision.selected_digest
            selected_by_distribution = evaluations[selected_artifact_id]
            primary_name = self._primary_distribution(selected_by_distribution)
            selected = self._evaluation_for_distribution(
                selected_by_distribution, primary_name,
            )

            round_candidate_evaluations: list[ArtifactEvaluation] = []
            for nomination in nominations:
                by_distribution = evaluations.get(
                    nomination.artifact.artifact_id, {},
                )
                report = reports_by_id[nomination.artifact.artifact_id]
                if primary_name in report.distributions:
                    evaluation = self._evaluation_for_distribution(
                        by_distribution, primary_name,
                    )
                    round_candidate_evaluations.append(evaluation)
            candidate_evaluations = tuple(round_candidate_evaluations)

            hidden_cases = {
                record.case.case_id: record.case
                for records in hidden.values()
                for record in records
            }
            selected_case_results = tuple(
                result
                for evaluation in selected_by_distribution.values()
                for result in evaluation.cases
            )
            feedback = GeneratorFeedback(
                source_round_index=plan.round_index,
                target_round_index=plan.round_index + 1,
                selected_program_id=selected.artifact.artifact_id,
                selected_program=selected.artifact,
                cases=tuple(
                    CaseFeedback(
                        case=hidden_cases[result.case_id],
                        primary_error=result.primary_error,
                        metrics=result.metrics,
                    )
                    for result in selected_case_results
                ),
            )
            if self._pending_feedback_samplers:
                raise RuntimeError("previous Guidance feedback cohort was not released")
            if official_sampler is not None:
                self._pending_feedback_samplers.append(official_sampler)
            if not self.campaign.generator.guidance_enabled:
                self._close_case_samplers(self._pending_feedback_samplers)
                self._pending_feedback_samplers.clear()
            hidden_metrics = self._hidden_metrics(evaluations, reports=reports)
            # Promotion becomes durable only after all feedback and metric work
            # for this round succeeds. An earlier exception remains an aborted
            # round and cannot label a Candidate as selected.
            candidate_history.mark_selected(selected.artifact)
            incumbent = selected
            controller_markdown = self._controller_round_markdown(
                reports,
                decision,
                hidden_suite_size=sum(len(records) for records in hidden.values()),
            )
            if recovering_development:
                controller_markdown += (
                    "\n\nRecovery provenance: frozen first-round Agent notes and "
                    "submissions were imported; development was not rerun. "
                    "Earlier development feedback in those notes is historical "
                    "and may reflect the source run's timing defect. All Official "
                    "results and selection above were recomputed after cohort "
                    "verification; no old Official result or later-round memory "
                    "was imported.\n"
                    f"- Source manifest SHA-256: {recovery.source_manifest_sha256}\n"
                    "- Development case counts: historical source-run counts."
                )
            self._append_campaign_memory_round(
                plan,
                agent_logs,
                controller_markdown,
                message=(
                    "Agent logs and Official results were added to Campaign memory"
                ),
            )
            round_results.append(
                RoundResult(
                    round_index=plan.round_index,
                    case_mix=plan.case_mix_summary(),
                    candidate_evaluations=candidate_evaluations,
                    selected=selected,
                    hidden_suite_size=sum(len(records) for records in hidden.values()),
                    agent_case_counts=agent_counts,
                    hidden_distribution_metrics=hidden_metrics,
                    promotion_reports=reports,
                    promotion_decision=decision,
                )
            )
            self.events.emit(
                "round_finished",
                f"Round {plan.round_index} selected {selected.artifact.name}",
                round_index=plan.round_index,
                payload={
                    "artifact_id": selected.artifact.artifact_id,
                    "promotion_reason": decision.reason,
                    "agent_case_counts": agent_counts,
                    "hidden_distributions": hidden_metrics,
                },
            )
            self._active_round_plan = None
            self._active_agent_logs = ()
            self._active_round_memory_written = False

            state.update(incumbent=incumbent, feedback=feedback, generator_state=generator_state,
                         original_hidden_offset=original_hidden_offset, development_original_offset=development_original_offset,
                         phase="round_prepare", round_data={})
            self._checkpoint_boundary("round_finished")

        if incumbent is None:
            raise RuntimeError("Campaign completed without a program")
        if state["phase"] == "round_prepare":
            state["phase"] = "terminal_prepare"
        final_cohorts = tuple(cohort for cohort in terminal_spec.cohorts if cohort.guidance_source is TerminalGuidanceSource.FINAL)
        if state["phase"] == "terminal_prepare":
            if final_cohorts and not state.get("final_guidance_ready"):
                if feedback is None:
                    raise RuntimeError("final-guided terminal cohort has no S_R feedback")
                self._checkpoint_boundary("before_terminal_guidance")
                guidance_settings = {**generator_settings, "controller_seed": self._seeds.derive("generator", "terminal_final_guidance")}
                state["phase"] = "terminal_guidance_running"
                self._save_continuation()
                generator_state = self.generator.update_guidance(generator_state, feedback, guidance_settings,
                                                                 self._generator_progress(None, "terminal_guidance_training"))
                self._loaded_generator_state = generator_state
                self._close_case_samplers(self._pending_feedback_samplers)
                self._pending_feedback_samplers.clear()
                state.update(generator_state=generator_state, final_guidance_ready=True, phase="terminal_prepare")
                self.events.emit("terminal_guidance_ready", "Final Guidance is frozen for terminal diagnostics only")
                self._checkpoint_boundary("terminal_guidance_ready")
            if not final_cohorts:
                self._close_case_samplers(self._pending_feedback_samplers)
                self._pending_feedback_samplers.clear()
            for cohort in final_cohorts:
                if any(key.startswith(cohort.cohort_id + "/") for key in terminal_hidden):
                    continue
                original_hidden_offset = self._seal_terminal_cohort(
                    terminal_hidden, cohort, generator_state, self.campaign.rounds[-1], original_hidden_offset,
                    generator_settings, terminal_generator_states, generation_round_index=len(self.campaign.rounds) + 1)
                state["original_hidden_offset"] = original_hidden_offset
                self._checkpoint_boundary("terminal_cohort_sealed")
            state["phase"] = "terminal_ready"
            self._checkpoint_boundary("terminal_sampled")
        terminal_hidden_ids = tuple(record.case.case_id for records in terminal_hidden.values() for record in records)
        if len(terminal_hidden_ids) != len(set(terminal_hidden_ids)):
            raise ValueError("the terminal evaluation contains duplicate Case ids")
        terminal_artifacts_by_id = {}
        for round_result in round_results:
            artifact = round_result.selected.artifact
            terminal_artifacts_by_id.setdefault(artifact.artifact_id, artifact)
        if state["phase"] == "terminal_ready":
            terminal_reports, terminal_evaluations = self._official_reports(
                tuple(terminal_artifacts_by_id.values()), terminal_hidden, phase=EvaluationPhase.TERMINAL,
                seed_namespace="terminal_evaluation", generator_states_by_distribution=terminal_generator_states)
            state.update(terminal_reports=terminal_reports, terminal_evaluations=terminal_evaluations, phase="terminal_done")
            state["hidden_suites"].clear()
            state["hidden_progress"].clear()
            self._checkpoint_boundary("terminal_completed")
        terminal_reports, terminal_evaluations = state["terminal_reports"], state["terminal_evaluations"]
        self._close_case_samplers(self._terminal_case_samplers)
        self._terminal_case_samplers.clear()
        terminal_report_by_id = {
            report.candidate_digest: report for report in terminal_reports
        }
        terminal_round_reports = tuple(
            terminal_report_by_id[round_result.selected.artifact.artifact_id]
            for round_result in round_results
        )
        terminal_report = terminal_round_reports[-1]
        if not terminal_report.technically_eligible:
            raise RuntimeError("selected program failed terminal evaluation")
        terminal_by_distribution = terminal_evaluations[incumbent.artifact.artifact_id]
        terminal_primary = self._terminal_distribution_key(
            terminal_spec.ranking.cohort_id,
            terminal_spec.ranking.distribution,
        )
        if terminal_primary not in terminal_report.distributions:
            raise RuntimeError(
                "terminal evaluation did not produce the configured "
                f"{terminal_primary} distribution"
            )
        terminal_selected = self._evaluation_for_distribution(
            terminal_by_distribution, terminal_primary,
        )
        terminal_ranking = tuple(
            (
                getattr(
                    report.distributions[terminal_primary],
                    terminal_spec.ranking.statistic,
                ),
                round_result.round_index,
            )
            for round_result, report in zip(round_results, terminal_round_reports)
            if report.technically_eligible
            and terminal_primary in report.distributions
        )
        best_terminal_round = (
            min(terminal_ranking)[1] if terminal_ranking else None
        )
        terminal_metrics = {
            name: {
                "case_count": summary.case_count,
                "mean": summary.mean,
                "cvar": summary.cvar,
                "maximum": summary.maximum,
            }
            for name, summary in terminal_report.distributions.items()
        }
        result = CampaignResult(
            campaign_name=self.campaign.name,
            selected=terminal_selected,
            rounds=tuple(round_results),
            memory=memory.snapshot(),
            candidate_history=candidate_history.snapshot(),
            terminal_distribution_metrics=terminal_metrics,
            terminal_round_reports=terminal_round_reports,
            best_terminal_round=best_terminal_round,
            seed_ledger=self._seeds.snapshot(),
            sampling_attempts=tuple(self._restored_sampling_attempts) + tuple(
                attempt
                for sampler in self._case_samplers
                for attempt in sampler.sampling_attempts
            ),
        )
        self.events.emit(
            "run_finished",
            f"Run finished with {terminal_selected.artifact.name}",
            payload={
                "artifact_id": terminal_selected.artifact.artifact_id,
                "terminal_distributions": terminal_metrics,
                "terminal_artifact_ids": tuple(
                    report.candidate_digest for report in terminal_round_reports
                ),
                "best_terminal_round": best_terminal_round,
            },
        )
        state["phase"] = "complete"
        state["result"] = result
        self._save_continuation()
        return result

    def _prepare_agent_request(
        self,
        plan: RoundPlan,
        agent,
        evaluator: AgentEvaluator,
        sampler,
        incumbent: ArtifactEvaluation | None,
        memory: CampaignMemoryLog,
        candidate_history: CandidateHistory,
    ) -> AgentExecutionRequest:
        policy = plan.development_sampling
        interactive_sources = tuple(
            source
            for source in policy.allowed_sources
            if source not in policy.starter_only_sources
        )
        agent_plan = (
            plan
            if interactive_sources == policy.allowed_sources
            else replace(
                plan,
                development_sampling=replace(
                    policy,
                    allowed_sources=interactive_sources,
                    starter_only_sources=(),
                ),
            )
        )
        context = AgentContext(
            round_plan=agent_plan,
            agent=agent,
            prompt=build_agent_context_prompt(
                agent.prompt, self.task.agent_brief(self.campaign.task_settings),
            ),
            evaluator=evaluator,
            candidate_solve_budget_seconds=getattr(
                self.candidate_execution, "case_hard_wall_seconds", None,
            ),
            incumbent=(
                self._agent_visible_copy(incumbent.artifact)
                if incumbent is not None
                else None
            ),
            memory=memory.snapshot(),
            candidate_history=candidate_history.snapshot(),
            seed=self._seeds.derive(
                "agent",
                plan.round_index,
                agent.agent_id,
            ),
            sampler=sampler,
            public_problem_bundle=self.public_problem_bundle,
            execution_progress=lambda state, detail: self._agent_execution_progress(
                plan, agent, state, detail,
            ),
        )
        return AgentExecutionRequest(
            self.agents[agent.agent_id],
            context,
            plan.agent_budget_seconds,
        )

    def _agent_execution_progress(self, plan, agent, state, detail):
        if state not in {"recovering", "resumed"}:
            return
        message = (f"{agent.name} is waiting for the model service; saved work will resume"
                   if state == "recovering" else f"{agent.name} resumed the same research session")
        self.events.emit(
            "agent_" + state, message, round_index=plan.round_index,
            payload={"agent_id": agent.agent_id, **detail},
        )

    def _restore_round_development(
        self,
        recovery: RoundDevelopmentRecovery,
        plan: RoundPlan,
        incumbent: Artifact | None,
    ) -> tuple[
        tuple[AgentOutcome, ...],
        tuple[AgentRoundLog, ...],
        dict[str, Mapping[str, int]],
        int,
    ]:
        """Validate Controller-owned snapshots before any recovered run work."""
        if recovery.round_index != plan.round_index:
            raise ValueError("recovery round does not match the Campaign's first round")
        expected_agents = tuple(agent.agent_id for agent in plan.agents)
        if tuple(log.agent_id for log in recovery.agent_logs) != expected_agents:
            raise ValueError("recovery notes must match the Campaign Agent order exactly")
        if any(
            log.agent_name != agent.name
            for log, agent in zip(recovery.agent_logs, plan.agents)
        ):
            raise ValueError("recovery notes do not match the Campaign Agent names")
        if set(recovery.agent_case_counts) != set(expected_agents):
            raise ValueError("recovery case counts must cover exactly the Campaign Agents")
        case_limit = plan.development_sampling.max_cases_per_agent_per_round
        if case_limit is not None and any(
            sum(counts.values()) > case_limit
            for counts in recovery.agent_case_counts.values()
        ):
            raise ValueError("recovery case counts exceed the Campaign development limit")

        artifacts = recovery.artifacts
        expected_ids = tuple(
            f"candidate-r{plan.round_index}-{index:04d}"
            for index in range(1, len(artifacts) + 1)
        )
        if tuple(item.artifact_id for item in artifacts) != expected_ids:
            raise ValueError("recovery requires the original contiguous Controller Candidate ids")
        by_id = {item.artifact_id: item for item in artifacts}
        incumbent_id = incumbent.artifact_id if incumbent is not None else None
        agent_positions = []
        for artifact in artifacts:
            if artifact.kind is not self.task.kind:
                raise ValueError("recovery Candidate kind does not match the Task")
            if artifact.round_index != plan.round_index:
                raise ValueError("recovery Candidate round does not match the recovered round")
            if artifact.agent_id not in expected_agents:
                raise ValueError("recovery Candidate Agent is not in the Campaign")
            agent_positions.append(expected_agents.index(artifact.agent_id))
            parent = artifact.parent_artifact_id
            if parent is not None and parent != incumbent_id:
                if parent not in by_id or by_id[parent].agent_id != artifact.agent_id:
                    raise ValueError(
                        "recovery Candidate parent must be the incoming incumbent "
                        "or a Candidate in the same Agent portfolio"
                    )
            seen = set()
            cursor = artifact.artifact_id
            while cursor in by_id:
                if cursor in seen:
                    raise ValueError("recovery Candidate lineage contains a cycle")
                seen.add(cursor)
                cursor = by_id[cursor].parent_artifact_id
        if agent_positions != sorted(agent_positions):
            raise ValueError("recovery Candidate order differs from Campaign Agent order")

        frozen = tuple(
            replace(
                artifact,
                program=deepcopy(artifact.program),
                metadata=deepcopy(dict(artifact.metadata)),
            )
            for artifact in artifacts
        )
        outcomes = tuple(
            AgentOutcome(
                artifacts=tuple(
                    artifact for artifact in frozen if artifact.agent_id == log.agent_id
                ),
                round_log=log.markdown,
                execution_status=log.status,
            )
            for log in recovery.agent_logs
        )
        # This fresh run has no imported history or selection. Reserve prior
        # development ids solely to preserve cross-phase overlap checks.
        self._development_case_ids.update(recovery.development_case_sources)
        self._development_case_owners.update(
            {
                case_id: (plan.round_index, source)
                for case_id, source in recovery.development_case_sources.items()
            }
        )
        self._next_candidate_sequence = len(frozen)
        counts = {
            agent_id: dict(values)
            for agent_id, values in recovery.agent_case_counts.items()
        }
        original_offset = (
            plan.original_count
            if plan.development_sampling.strategy is SamplingStrategy.FIXED_RECIPE
            else max((row["original"] for row in counts.values()), default=0)
        )
        return outcomes, recovery.agent_logs, counts, original_offset

    def _develop_round(
        self,
        plan: RoundPlan,
        generator_state: GeneratorState,
        generator_settings: Mapping[str, object],
        incumbent: ArtifactEvaluation | None,
        memory: CampaignMemoryLog,
        candidate_history: CandidateHistory,
        development_original_offset: int,
    ) -> tuple[
        tuple[AgentOutcome, ...],
        tuple[AgentRoundLog, ...],
        dict[str, Mapping[str, int]],
        int,
    ]:
        """Run the ordinary development path shared by all non-recovered rounds."""
        policy = plan.development_sampling
        development_sampler_start = len(self._case_samplers)
        requests: list[AgentExecutionRequest] = []
        agent_counts: dict[str, Mapping[str, int]] = {}
        interactive_samplers: list[tuple[object, _InteractiveSampler]] = []
        if policy.strategy is SamplingStrategy.FIXED_RECIPE:
            visible, counts = self._fixed_development(
                plan,
                generator_state,
                generator_settings,
                original_offset=development_original_offset,
            )
            development_original_offset += plan.original_count
            workspace_records = visible
            for agent in plan.agents:
                evaluator = AgentEvaluator(
                    workspace_records,
                    self.evaluation,
                    self.campaign.evaluator,
                    self.campaign.task_settings,
                    seed=self._seeds.derive(
                        "development_evaluation",
                        plan.round_index,
                        agent.agent_id,
                    ),
                    plan_seed_provider=self._development_plan_seed_provider(
                        plan.round_index,
                        agent.agent_id,
                    ),
                    generator=self.generator,
                    generator_state=generator_state,
                    session_lock=self._generator_operation_lock,
                    candidate_execution=(self.candidate_execution.for_development(agent.agent_id)
                        if callable(getattr(self.candidate_execution, "for_development", None))
                        else self.candidate_execution),
                )
                requests.append(
                    self._prepare_agent_request(
                        plan,
                        agent,
                        evaluator,
                        _FixedSamplerView(),
                        incumbent,
                        memory,
                        candidate_history,
                    )
                )
                agent_counts[agent.agent_id] = counts
        else:
            for agent in plan.agents:
                evaluator = AgentEvaluator(
                    (),
                    self.evaluation,
                    self.campaign.evaluator,
                    self.campaign.task_settings,
                    seed=self._seeds.derive(
                        "development_evaluation",
                        plan.round_index,
                        agent.agent_id,
                    ),
                    plan_seed_provider=self._development_plan_seed_provider(
                        plan.round_index,
                        agent.agent_id,
                    ),
                    generator=self.generator,
                    generator_state=generator_state,
                    session_lock=self._generator_operation_lock,
                    candidate_execution=(self.candidate_execution.for_development(agent.agent_id)
                        if callable(getattr(self.candidate_execution, "for_development", None))
                        else self.candidate_execution),
                )
                inner = RoundCaseSampler(
                    agent_id=agent.agent_id,
                    round_index=plan.round_index,
                    max_cases=policy.max_cases_per_agent_per_round,
                    strategy=policy.strategy,
                    dataset=self.dataset,
                    generator=self.generator,
                    generator_state=generator_state,
                    seed=self._adaptive_seed(plan.round_index, agent.agent_id),
                    original_offset=development_original_offset,
                    generation_settings=generator_settings,
                    generator_lock=self._generator_operation_lock,
                    record_reserver=(
                        lambda records,
                        evaluator=evaluator,
                        round_index=plan.round_index:
                        self._register_development_records(
                            records,
                            round_index=round_index,
                            evaluator=evaluator,
                        )
                    ),
                )
                self._case_samplers.append(inner)
                sampler = _InteractiveSampler(
                    inner,
                    evaluator,
                    policy.allowed_sources,
                    starter_only_sources=policy.starter_only_sources,
                    allow_guidance=self.campaign.generator.guidance_enabled,
                    allow_perturbations=(
                        self.campaign.generator.perturbations_enabled
                    ),
                )
                self._sample_counts(sampler, policy.starter)
                sampler.seal_starter()
                requests.append(
                    self._prepare_agent_request(
                        plan,
                        agent,
                        evaluator,
                        sampler,
                        incumbent,
                        memory,
                        candidate_history,
                    )
                )
                interactive_samplers.append((agent, sampler))

        # Every Agent context is complete before the service preflight.
        # The service starts all turns as one ordered batch; only after all
        # raw results settle does the Controller validate and assign ids.
        round_development_samplers = tuple(
            self._case_samplers[development_sampler_start:]
        )
        try:
            outcomes, agent_logs = self._execute_round_agents(
                plan,
                tuple(requests),
                incumbent,
            )
            self._active_agent_logs = agent_logs
        finally:
            self._close_case_samplers(round_development_samplers)
        if policy.strategy is not SamplingStrategy.FIXED_RECIPE:
            for agent, sampler in interactive_samplers:
                agent_counts[agent.agent_id] = sampler.counts
            # D0 is a shared development pool, so Agents in one round may
            # inspect the same original slice. Advance by the largest slice
            # consumed by any Agent before the next round; this prevents
            # accidental cross-round replay without multiplying D0 usage by
            # the number of Agents.
            development_original_offset += max(
                (
                    counts.get("original", 0)
                    for counts in agent_counts.values()
                ),
                default=0,
            )
        return outcomes, agent_logs, agent_counts, development_original_offset

    def _execute_round_agents(
        self,
        plan: RoundPlan,
        requests: Sequence[AgentExecutionRequest],
        incumbent: ArtifactEvaluation | None,
    ) -> tuple[tuple[AgentOutcome, ...], tuple[AgentRoundLog, ...]]:
        # Validation and GPU reservation both happen before the UI can observe
        # an Agent as started, and the lease spans every worker cleanup path.
        reservation = reserve_agent_batch(self.agent_execution, requests)
        prepared = reservation.requests_for(self.agent_execution)
        started: list[AgentExecutionRequest] = []
        try:
            for request in prepared:
                started.append(request)
                agent = request.context.agent
                self.events.emit(
                    "agent_started",
                    f"{agent.name} started",
                    round_index=plan.round_index,
                    payload={
                        "agent_id": agent.agent_id,
                        "budget_seconds": plan.agent_budget_seconds,
                    },
                )
            results = execute_reserved_agent_batch(
                self.agent_execution,
                reservation,
            )
        except BaseException as exc:
            for request in started:
                self._emit_agent_failed(plan, request.context.agent, exc)
            raise
        finally:
            reservation.close()

        outcomes: list[AgentOutcome] = []
        agent_logs: list[AgentRoundLog] = []
        fatal: BaseException | None = None
        for request, result in zip(prepared, results):
            agent = request.context.agent
            if result.error is not None:
                self._emit_agent_failed(plan, agent, result.error)
                agent_logs.append(
                    self._failed_agent_round_log(plan, agent, result.error)
                )
                if isinstance(result.error, AgentInfrastructureError) or not isinstance(
                    result.error,
                    Exception,
                ):
                    fatal = fatal or result.error
                continue
            try:
                outcome = result.outcome
                if not isinstance(outcome, AgentOutcome):
                    raise TypeError("AgentProgram.run must return AgentOutcome")
                outcome = self._freeze_agent_outcome(
                    outcome,
                    plan,
                    agent.agent_id,
                    incumbent.artifact if incumbent is not None else None,
                )
            except BaseException as exc:
                self._emit_agent_failed(plan, agent, exc)
                raw_outcome = result.outcome
                round_log = (
                    raw_outcome.round_log
                    if isinstance(raw_outcome, AgentOutcome)
                    else ""
                )
                agent_logs.append(
                    self._failed_agent_round_log(
                        plan,
                        agent,
                        exc,
                        round_log=round_log,
                    )
                )
                if isinstance(exc, AgentInfrastructureError) or not isinstance(
                    exc,
                    Exception,
                ):
                    fatal = fatal or exc
                continue
            self._emit_agent_finished(plan, agent, outcome)
            outcomes.append(outcome)
            agent_logs.append(self._outcome_agent_round_log(plan, agent, outcome))
        if fatal is not None:
            # A peer infrastructure failure aborts the round, but it must not
            # erase another Agent's already validated Controller snapshots.
            nominations = _unique_agent_nominations(outcomes)
            self._record_submitted_candidates(
                self._candidate_history,
                tuple(nomination.artifact for nomination in nominations),
            )
            setattr(fatal, "agent_logs", tuple(agent_logs))
            raise fatal
        return tuple(outcomes), tuple(agent_logs)

    def _append_campaign_memory_round(
        self,
        plan: RoundPlan,
        agent_logs: Sequence[AgentRoundLog],
        controller_markdown: str,
        *,
        message: str,
        status: str = "completed",
    ) -> None:
        round_log = self._campaign_memory.append_round(
            plan.round_index,
            agent_logs,
            controller_markdown,
            status=status,
        )
        self._active_round_memory_written = True
        history = self._candidate_history.snapshot()
        self.events.emit(
            "campaign_memory_updated",
            message,
            round_index=plan.round_index,
            payload={
                "round_memory": {
                    "round": round_log.round_index,
                    "status": round_log.status,
                    "agents": [
                        {
                            "agent_id": log.agent_id,
                            "agent_name": log.agent_name,
                            "status": log.status,
                            "markdown": log.markdown,
                        }
                        for log in round_log.agent_logs
                    ],
                    "controller_markdown": round_log.controller_markdown,
                    "merged_markdown": round_log.merged_markdown,
                },
                "candidate_history": [
                    {
                        "id": node.node_id,
                        "parent": node.parent_node_id,
                        "round": node.round_index,
                        "agent_id": node.agent_id,
                        "artifact_id": node.artifact_id,
                        "title": node.title,
                        "summary": node.summary,
                        "official_distributions": {
                            name: {
                                "case_count": summary.case_count,
                                "mean": summary.mean,
                                "cvar": summary.cvar,
                                "maximum": summary.maximum,
                                "metrics": dict(summary.metrics),
                            }
                            for name, summary in node.official_distributions.items()
                        },
                        "failures": node.failures,
                        "status": node.status,
                        "selected": node.selected,
                    }
                    for node in history.nodes
                ],
            },
        )

    @staticmethod
    def _record_submitted_candidates(
        candidate_history: CandidateHistory,
        artifacts: Sequence[Artifact],
    ) -> None:
        """Add one accepted portfolio without losing forward parent references."""

        pending = list(artifacts)
        while pending:
            remaining: list[Artifact] = []
            progressed = False
            for artifact in pending:
                parent = artifact.parent_artifact_id
                if parent is not None and not candidate_history.contains(parent):
                    remaining.append(artifact)
                    continue
                candidate_history.add_candidate(artifact)
                progressed = True
            if not progressed:
                raise RuntimeError(
                    "accepted Candidate lineage could not be ordered in history"
                )
            pending = remaining

    def _preserve_aborted_round(self, error: BaseException) -> None:
        plan = self._active_round_plan
        if plan is None or self._active_round_memory_written:
            return
        logs = self._active_agent_logs
        recovered = getattr(error, "agent_logs", ())
        if not logs and isinstance(recovered, tuple) and all(
            isinstance(log, AgentRoundLog) for log in recovered
        ):
            logs = recovered
        self._append_campaign_memory_round(
            plan,
            logs,
            (
                "#### Round aborted\n"
                f"- The round stopped before all round work completed.\n"
                f"- Reason code: {_public_round_abort_code(error)}"
            ),
            message=(
                f"Aborted round and {len(logs)} Agent log(s) preserved"
                if logs
                else "Round aborted before Agent logs were available"
            ),
            status="aborted",
        )

    @staticmethod
    def _outcome_agent_round_log(plan, agent, outcome: AgentOutcome) -> AgentRoundLog:
        if outcome.round_log.strip():
            markdown = outcome.round_log
        else:
            observations = (
                "\n".join(f"- {observation}" for observation in outcome.observations)
                or "- No separate experiment notes were returned."
            )
            candidates = (
                "\n".join(
                    f"- {artifact.name} ({artifact.artifact_id})"
                    for artifact in outcome.nominations
                )
                or "- No Candidate submitted."
            )
            markdown = (
                f"# Round {plan.round_index} — {agent.name}\n\n"
                "## Experiments and observations\n\n"
                f"{observations}\n\n"
                "## Final submission\n\n"
                f"{candidates}\n\n"
                "## Handoff\n\n"
                "Continue from the evidence above; no separate handoff was written."
            )
        return AgentRoundLog(
            round_index=plan.round_index,
            agent_id=agent.agent_id,
            agent_name=agent.name,
            markdown=markdown,
            status=outcome.execution_status,
        )

    @staticmethod
    def _failed_agent_round_log(
        plan,
        agent,
        error: BaseException,
        *,
        round_log: str = "",
    ) -> AgentRoundLog:
        recovered = round_log or getattr(error, "round_log", "")
        status = (
            "timed_out"
            if isinstance(error, AgentExecutionTimeout)
            else "failed"
        )
        markdown = recovered if recovered.strip() else (
            f"# Round {plan.round_index} — {agent.name}\n\n"
            "## Execution status\n\n"
            f"The Agent ended with reason code {_public_agent_failure_code(error)}.\n\n"
            "## Handoff\n\n"
            "No Agent-authored round log was recovered."
        )
        return AgentRoundLog(
            round_index=plan.round_index,
            agent_id=agent.agent_id,
            agent_name=agent.name,
            markdown=markdown,
            status=status,
        )

    @staticmethod
    def _controller_round_markdown(
        reports: Sequence[PromotionCandidateReport],
        decision,
        *,
        hidden_suite_size: int,
    ) -> str:
        lines = [f"- Official cohort: {hidden_suite_size} Cases.", ""]
        for report in reports:
            lines.append(f"#### Candidate {report.candidate_digest}")
            lines.append(
                "- Technical status: "
                + ("eligible" if report.technically_eligible else "ineligible")
            )
            for distribution in _DISTRIBUTIONS:
                summary = report.distributions.get(distribution)
                if summary is None:
                    continue
                lines.append(
                    f"- {distribution}: n={summary.case_count}, "
                    f"mean={summary.mean:.8g}, CVaR={summary.cvar:.8g}, "
                    f"max={summary.maximum:.8g}"
                )
                for metric_name, metric_value in sorted(summary.metrics.items()):
                    lines.append(
                        f"  - {metric_name}: {metric_value:.8g}"
                    )
            for name, passed in sorted(report.guardrails.items()):
                lines.append(
                    f"- Guardrail {name}: {'passed' if passed else 'failed'}"
                )
            for failure in report.failures:
                lines.append(f"- Failure: {failure}")
            lines.append("")
        lines.append("#### Promotion decision")
        for ranking in decision.ranking:
            rank = ranking.rank if ranking.rank is not None else "not ranked"
            lines.append(
                f"- {ranking.candidate_digest}: rank={rank}, "
                f"eligible={'yes' if ranking.eligible else 'no'}; "
                f"{ranking.reason}"
            )
        lines.extend(
            (
                f"- Selected: {decision.selected_digest}",
                f"- Changed incumbent: {'yes' if decision.promoted else 'no'}",
                f"- Reason: {decision.reason}",
            )
        )
        return "\n".join(lines).rstrip()

    def _emit_agent_failed(self, plan: RoundPlan, agent, exc: BaseException) -> None:
        reason_code = _public_agent_failure_code(exc)
        self.events.emit(
            "agent_failed",
            f"{agent.name} failed ({reason_code})",
            round_index=plan.round_index,
            payload={"agent_id": agent.agent_id, "reason_code": reason_code},
        )

    def _emit_agent_finished(
        self,
        plan: RoundPlan,
        agent,
        outcome: AgentOutcome,
    ) -> None:
        summary = (
            f"{agent.name} submitted no Candidate"
            if not outcome.nominations
            else f"{agent.name} returned {outcome.artifact.name}"
            if len(outcome.nominations) == 1
            else f"{agent.name} returned {len(outcome.nominations)} candidates"
        )
        if outcome.execution_status != "completed":
            summary += f" after the turn {outcome.execution_status.replace('_', ' ')}"
        self.events.emit(
            "agent_finished",
            summary,
            round_index=plan.round_index,
            payload={
                "agent_id": agent.agent_id,
                "execution_status": outcome.execution_status,
                "artifact_id": (
                    outcome.artifact.artifact_id
                    if outcome.artifact is not None
                    else None
                ),
                "artifact_ids": tuple(
                    artifact.artifact_id for artifact in outcome.nominations
                ),
                "nomination_count": len(outcome.nominations),
            },
        )

    def _freeze_initial_candidate(self, artifact: Artifact) -> Artifact:
        """Create the Controller-owned S0 snapshot used by the state machine."""

        if artifact.kind is not self.task.kind:
            raise ValueError("initial_candidate kind does not match the selected Task")
        try:
            program = deepcopy(artifact.program)
            metadata = deepcopy(dict(artifact.metadata))
        except Exception as exc:
            raise TypeError(
                "initial_candidate program and metadata must support snapshotting"
            ) from exc
        return replace(
            artifact,
            artifact_id="candidate-s0",
            parent_artifact_id=None,
            program=program,
            metadata={
                **metadata,
                "submitted_artifact_id": artifact.artifact_id,
            },
        )

    def _freeze_agent_outcome(
        self,
        outcome: AgentOutcome,
        plan: RoundPlan,
        agent_id: str,
        incumbent: Artifact | None,
    ) -> AgentOutcome:
        """Assign ordinary Controller ids and snapshot every new submission."""

        snapshots: list[tuple[Artifact, object, Mapping[str, object]]] = []
        for artifact in outcome.nominations:
            # Returning the public copy of the current incumbent means an
            # explicit zero-submission/keep decision. Mutating that copy cannot
            # overwrite the Controller's private snapshot.
            if (
                incumbent is not None
                and artifact.artifact_id == incumbent.artifact_id
            ):
                continue
            if artifact.kind is not self.task.kind:
                raise ValueError("Candidate kind does not match the selected Task")
            if artifact.round_index != plan.round_index:
                raise ValueError("Candidate round_index does not match the active round")
            if artifact.agent_id != agent_id:
                raise ValueError("Candidate agent_id does not match the active Agent")
            try:
                program = deepcopy(artifact.program)
                metadata = deepcopy(dict(artifact.metadata))
            except Exception as exc:
                raise TypeError(
                    "Candidate program and metadata must support snapshotting"
                ) from exc
            snapshots.append((artifact, program, metadata))

        # An Agent portfolio is accepted atomically. Validate and snapshot every
        # nomination before consuming any Controller sequence id, then build all
        # immutable Candidate records against a local range.
        submitted_ids = tuple(artifact.artifact_id for artifact, _, _ in snapshots)
        if len(submitted_ids) != len(set(submitted_ids)):
            raise ValueError("an Agent portfolio contains duplicate Candidate ids")
        first_sequence = self._next_candidate_sequence + 1
        canonical_ids = {
            submitted_id: f"candidate-r{plan.round_index}-{first_sequence + offset:04d}"
            for offset, submitted_id in enumerate(submitted_ids)
        }
        incumbent_id = incumbent.artifact_id if incumbent is not None else None
        canonical_parents: dict[str, str | None] = {}
        for artifact, _, _ in snapshots:
            parent = artifact.parent_artifact_id
            if parent is None:
                canonical_parent = None
            elif parent == incumbent_id:
                canonical_parent = parent
            elif parent in canonical_ids:
                canonical_parent = canonical_ids[parent]
            else:
                raise ValueError(
                    "Candidate parent must be the incumbent or another Candidate "
                    "in the same Agent portfolio"
                )
            canonical_parents[canonical_ids[artifact.artifact_id]] = canonical_parent

        # A same-portfolio parent may appear later in submission order, but the
        # lineage must still be a DAG before any Controller id is consumed.
        candidate_ids = frozenset(canonical_parents)
        for candidate_id in candidate_ids:
            seen: set[str] = set()
            cursor: str | None = candidate_id
            while cursor in candidate_ids:
                if cursor in seen:
                    raise ValueError("an Agent portfolio contains cyclic Candidate lineage")
                seen.add(cursor)
                cursor = canonical_parents[cursor]

        frozen: list[Artifact] = []
        for artifact, program, metadata in snapshots:
            canonical_id = canonical_ids[artifact.artifact_id]
            frozen.append(
                replace(
                    artifact,
                    artifact_id=canonical_id,
                    parent_artifact_id=canonical_parents[canonical_id],
                    program=program,
                    metadata={
                        **metadata,
                        "submitted_artifact_id": artifact.artifact_id,
                    },
                )
            )
        self._next_candidate_sequence += len(frozen)
        return AgentOutcome(
            frozen[0] if frozen else None,
            outcome.observations,
            artifacts=tuple(frozen[1:]),
            round_log=outcome.round_log,
            execution_status=outcome.execution_status,
        )

    @staticmethod
    def _agent_visible_copy(artifact: Artifact) -> Artifact:
        """Keep the Controller snapshot private from a cooperative in-process Agent."""

        try:
            return replace(
                artifact,
                program=deepcopy(artifact.program),
                metadata=deepcopy(dict(artifact.metadata)),
            )
        except Exception as exc:
            raise TypeError("selected Candidate cannot be copied for Agent handoff") from exc

    def _round_needs_guidance(self, plan: RoundPlan) -> bool:
        policy = plan.development_sampling
        counts = (
            policy.fixed_recipe
            if policy.strategy is SamplingStrategy.FIXED_RECIPE
            else policy.starter
        )
        return bool(
            SampleSource.GUIDED in policy.allowed_sources
            or (counts is not None and (counts.guided or counts.guided_perturbed))
            or plan.round_hidden.guided
            or plan.round_hidden.guided_perturbed
            or any(
                cohort.guidance_source is TerminalGuidanceSource.ROUND
                and cohort.round_index == plan.round_index
                for cohort in self.campaign.terminal_evaluation.cohorts
            )
        )

    def _initial_guidance_feedback(
        self,
        artifact: Artifact,
        generator_state,
        plan: RoundPlan,
        generator_settings: Mapping[str, object],
    ) -> GeneratorFeedback:
        """Evaluate S0 on a private cohort distinct from H1 and terminal."""

        base_count = max(1, plan.round_hidden.base)
        hidden, _, feedback_sampler = self._hidden_suite(
            generator_state,
            plan,
            HiddenSuiteSpec(base=base_count),
            GenerationPurpose.GUIDANCE_FEEDBACK,
            0,
            generator_settings,
            seed_namespace="initial_guidance_feedback",
            generation_round_index=0,
        )
        case_ids = tuple(
            record.case.case_id
            for records in hidden.values()
            for record in records
        )
        state = getattr(self, "_continuation_state", {})
        if not state.get("initial_guidance_reserved"):
            self._reserve_hidden_case_ids(case_ids, purpose="Initial Guidance")
            state["initial_guidance_reserved"] = True
        reports, evaluations = self._official_reports(
            (artifact,),
            hidden,
            seed_namespace="initial_guidance_feedback",
        )
        if not reports[0].technically_eligible:
            raise RuntimeError("initial_candidate failed Guidance feedback evaluation")
        if feedback_sampler is not None:
            self._pending_feedback_samplers.append(feedback_sampler)
        hidden_cases = {
            record.case.case_id: record.case
            for records in hidden.values()
            for record in records
        }
        rows = tuple(
            result
            for evaluation in evaluations[artifact.artifact_id].values()
            for result in evaluation.cases
        )
        return GeneratorFeedback(
            source_round_index=0,
            target_round_index=1,
            selected_program_id=artifact.artifact_id,
            selected_program=artifact,
            cases=tuple(
                CaseFeedback(
                    case=hidden_cases[result.case_id],
                    primary_error=result.primary_error,
                    metrics=result.metrics,
                )
                for result in rows
            ),
        )

    def _seal_terminal_cohort(
        self,
        terminal_hidden: dict[str, tuple[CaseRecord, ...]],
        cohort: TerminalCohortSpec,
        generator_state,
        plan: RoundPlan,
        original_offset: int,
        generator_settings: Mapping[str, object],
        generator_states: dict[str, GeneratorState],
        *,
        generation_round_index: int,
    ) -> int:
        hidden, next_original_offset, terminal_sampler = self._hidden_suite(
            generator_state,
            plan,
            cohort.suite,
            GenerationPurpose.TERMINAL,
            original_offset,
            generator_settings,
            seed_namespace=f"terminal:{cohort.cohort_id}",
            generation_round_index=generation_round_index,
        )
        if terminal_sampler is not None:
            self._terminal_case_samplers.append(terminal_sampler)
        cohort_case_ids = tuple(
            record.case.case_id
            for records in hidden.values()
            for record in records
        )
        self._reserve_hidden_case_ids(
            cohort_case_ids,
            purpose="Terminal",
        )
        existing_ids = {
            record.case.case_id
            for records in terminal_hidden.values()
            for record in records
        }
        for distribution, records in hidden.items():
            key = self._terminal_distribution_key(cohort.cohort_id, distribution)
            if key in terminal_hidden:
                raise ValueError(f"duplicate terminal result key: {key}")
            overlap = existing_ids.intersection(record.case.case_id for record in records)
            if overlap:
                raise ValueError(
                    f"terminal Case {sorted(overlap)[0]!r} appears in multiple cohorts"
                )
            terminal_hidden[key] = records
            generator_states[key] = generator_state
            existing_ids.update(record.case.case_id for record in records)
        self.events.emit(
            "terminal_cohort_sealed",
            f"Sealed terminal cohort {cohort.cohort_id}",
            round_index=(
                cohort.round_index
                if cohort.guidance_source is TerminalGuidanceSource.ROUND
                else None
            ),
            payload={
                "cohort_id": cohort.cohort_id,
                "guidance_source": cohort.guidance_source.value,
                "case_count": cohort.suite.case_count,
            },
        )
        return next_original_offset

    def _terminal_distribution_key(self, cohort_id: str, distribution: str) -> str:
        if cohort_id == distribution == "overall":
            return "overall"
        terminal = self.campaign.terminal_evaluation
        if (
            terminal is not None
            and len(terminal.cohorts) == 1
            and cohort_id == "base_model"
        ):
            return distribution
        return f"{cohort_id}/{distribution}"

    def _fixed_development(
        self,
        plan: RoundPlan,
        generator_state,
        generator_settings: Mapping[str, object],
        *,
        original_offset: int,
    ) -> tuple[tuple[CaseRecord, ...], Mapping[str, int]]:
        policy = plan.development_sampling
        if policy is None or policy.fixed_recipe is None:
            raise RuntimeError("fixed strategy has no recipe")
        evaluator = AgentEvaluator(
            (),
            self.evaluation,
            self.campaign.evaluator,
            self.campaign.task_settings,
            seed=self._seeds.derive(
                "fixed_recipe_registration",
                plan.round_index,
            ),
            generator=self.generator,
            generator_state=generator_state,
            session_lock=self._generator_operation_lock,
            candidate_execution=self.candidate_execution,
        )
        inner = RoundCaseSampler(
            agent_id="controller-fixed-recipe",
            round_index=plan.round_index,
            max_cases=policy.max_cases_per_agent_per_round,
            strategy=SamplingStrategy.FIXED_RECIPE,
            dataset=self.dataset,
            generator=self.generator,
            generator_state=generator_state,
            seed=self._seeds.derive("fixed_recipe", plan.round_index),
            original_offset=original_offset,
            generation_settings=generator_settings,
            generator_lock=self._generator_operation_lock,
            record_reserver=(
                lambda records,
                evaluator=evaluator,
                round_index=plan.round_index:
                self._register_development_records(
                    records,
                    round_index=round_index,
                    evaluator=evaluator,
                )
            ),
        )
        self._case_samplers.append(inner)
        sampler = _InteractiveSampler(
            inner,
            evaluator,
            policy.allowed_sources,
            allow_guidance=self.campaign.generator.guidance_enabled,
            allow_perturbations=self.campaign.generator.perturbations_enabled,
        )
        self._sample_counts(sampler, policy.fixed_recipe)
        records = [
            inner.record_for(case.case_id)
            for case in inner.issued_cases
        ]
        if policy.hide_case_types:
            Random(
                self._seeds.derive("fixed_recipe_shuffle", plan.round_index)
            ).shuffle(records)
        return tuple(records), sampler.counts

    def _register_development_records(
        self,
        records: Sequence[CaseRecord],
        *,
        round_index: int,
        evaluator: AgentEvaluator | None = None,
    ) -> None:
        """Reserve Development ids without misclassifying same-round D0 sharing."""

        case_ids = tuple(record.case.case_id for record in records)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("development batch contains duplicate Case ids")
        with self._case_registry_lock:
            for record in records:
                case_id = record.case.case_id
                previous = self._development_case_owners.get(case_id)
                if previous is None:
                    continue
                previous_round, previous_source = previous
                same_round_original = (
                    previous_round == round_index
                    and previous_source is CaseSource.INITIAL
                    and record.case.source is CaseSource.INITIAL
                )
                if not same_round_original:
                    raise CaseReservationConflict(
                        "development Case "
                        f"{case_id!r} was already used for an earlier "
                        "Development batch",
                        retained_generated_case_ids=(
                            (case_id,)
                            if previous_source is not CaseSource.INITIAL
                            else ()
                        ),
                    )
            private_registries = {
                "Initial Guidance": self._initial_guidance_hidden_case_ids,
                "Official": self._promotion_hidden_case_ids,
                "Terminal": self._terminal_hidden_case_ids,
            }
            for label, registered in private_registries.items():
                overlap = registered.intersection(case_ids)
                if not overlap:
                    continue
                raise CaseReservationConflict(
                    "development Case "
                    f"{sorted(overlap)[0]!r} was already reserved for {label}"
                )
            if evaluator is not None:
                evaluator.register(records)
            self._development_case_ids.update(case_ids)
            for record in records:
                self._development_case_owners.setdefault(
                    record.case.case_id,
                    (round_index, record.case.source),
                )

    def _reserve_hidden_case_ids(
        self,
        case_ids: Sequence[str],
        *,
        purpose: str,
    ) -> None:
        """Reserve one private cohort against every other Campaign phase."""

        prepared = tuple(case_ids)
        if len(prepared) != len(set(prepared)):
            raise ValueError(f"{purpose} cohort contains duplicate Case ids")
        registries = {
            "Development": self._development_case_ids,
            "Initial Guidance": self._initial_guidance_hidden_case_ids,
            "Official": self._promotion_hidden_case_ids,
            "Terminal": self._terminal_hidden_case_ids,
        }
        with self._case_registry_lock:
            target = registries[purpose]
            for label, registered in registries.items():
                if label == purpose:
                    continue
                overlap = registered.intersection(prepared)
                if not overlap:
                    continue
                raise ValueError(
                    f"{purpose} Case {sorted(overlap)[0]!r} was already used "
                    f"for {label}"
                )
            repeated = target.intersection(prepared)
            if repeated:
                raise ValueError(
                    f"{purpose} Case {sorted(repeated)[0]!r} was already used for "
                    f"an earlier {purpose} cohort"
                )
            target.update(prepared)

    @staticmethod
    def _sample_counts(sampler: _InteractiveSampler, counts: CaseCounts) -> None:
        if counts.original:
            sampler.sample_cases(
                SampleCasesRequest(SampleSource.ORIGINAL, counts.original)
            )
        for source, clean_count, paired_count in (
            (SampleSource.BASE, counts.base, counts.base_perturbed),
            (SampleSource.GUIDED, counts.guided, counts.guided_perturbed),
        ):
            if paired_count:
                sampler.sample_paired_cases(
                    PairedSampleRequest(source, paired_count)
                )
            unpaired_count = clean_count - paired_count
            if unpaired_count:
                sampler.sample_cases(SampleCasesRequest(source, unpaired_count))

    def _hidden_suite(self, generator_state, plan, spec, purpose, original_offset,
                      generator_settings, *, seed_namespace=None, generation_round_index=None):
        state = getattr(self, "_continuation_state", None)
        key = (purpose.value, seed_namespace, generation_round_index, plan.round_index)
        cache = state.setdefault("hidden_suites", {}) if state is not None else {}
        if key in cache:
            hidden, offset, lease_id = cache[key]
            return hidden, offset, self._case_lease_for(lease_id)
        result = self._hidden_suite_uncached(generator_state, plan, spec, purpose, original_offset,
                                            generator_settings, seed_namespace=seed_namespace,
                                            generation_round_index=generation_round_index)
        hidden, offset, sampler = result
        if sampler is not None:
            sampler._continuation_id = getattr(self, "_next_case_lease", 0)
            self._next_case_lease = sampler._continuation_id + 1
        cache[key] = (hidden, offset, getattr(sampler, "_continuation_id", None))
        # The surrounding phase commits its ownership/registry before pausing.
        return result

    def _hidden_suite_uncached(
        self,
        generator_state,
        plan: RoundPlan,
        spec: HiddenSuiteSpec,
        purpose: GenerationPurpose,
        original_offset: int,
        generator_settings: Mapping[str, object],
        *,
        seed_namespace: str | None = None,
        generation_round_index: int | None = None,
    ) -> tuple[
        dict[str, tuple[CaseRecord, ...]],
        int,
        RoundCaseSampler | None,
    ]:
        hidden: dict[str, tuple[CaseRecord, ...]] = {}
        if spec.original:
            end = original_offset + spec.original
            if end > len(self.dataset.hidden_cases):
                available = len(self.dataset.hidden_cases) - original_offset
                raise ValueError(
                    f"official suite needs {spec.original} original Cases, "
                    f"but only {available} fresh Cases remain"
                )
            hidden["original"] = self.dataset.hidden_cases[original_offset:end]
            original_offset = end
        generated_count = spec.case_count - spec.original
        if not generated_count:
            return hidden, original_offset, None
        if spec.guided and not self.campaign.generator.guidance_enabled:
            raise PermissionError("guided hidden Cases are disabled for this Run")
        if (
            spec.base_perturbed or spec.guided_perturbed
        ) and not self.campaign.generator.perturbations_enabled:
            raise PermissionError("perturbed hidden Cases are disabled for this Run")
        request_round_index = (
            generation_round_index
            if generation_round_index is not None
            else (0 if purpose is GenerationPurpose.TERMINAL else plan.round_index)
        )
        namespace = seed_namespace or purpose.value
        inner = RoundCaseSampler(
            agent_id=f"controller-{namespace}",
            round_index=request_round_index,
            max_cases=generated_count,
            strategy=SamplingStrategy.FIXED_RECIPE,
            dataset=self.dataset,
            generator=self.generator,
            generator_state=generator_state,
            seed=self._seeds.derive(
                "hidden_sampling",
                namespace,
                request_round_index,
                plan.round_index,
            ),
            generation_settings=generator_settings,
            visibility=CaseVisibility.HIDDEN,
            purpose=purpose,
            seed_namespace=namespace,
            generation_progress=self._generator_progress(
                request_round_index,
                f"{purpose.value}_sampling",
            ),
            generator_lock=self._generator_operation_lock,
        )
        self._case_samplers.append(inner)
        for name, source, clean_count, paired_count in (
            ("base", SampleSource.BASE, spec.base, spec.base_perturbed),
            ("guided", SampleSource.GUIDED, spec.guided, spec.guided_perturbed),
        ):
            roots: list[CaseRecord] = []
            children: list[CaseRecord] = []
            if paired_count:
                paired = inner.sample_paired_cases(
                    PairedSampleRequest(source, paired_count)
                )
                roots.extend(inner.record_for(case.case_id) for case in paired.roots)
                children.extend(
                    inner.record_for(case.case_id) for case in paired.children
                )
            unpaired_count = clean_count - paired_count
            if unpaired_count:
                unpaired = inner.sample_cases(
                    SampleCasesRequest(source, unpaired_count)
                )
                roots.extend(
                    inner.record_for(case.case_id) for case in unpaired.cases
                )
            if roots:
                hidden[name] = tuple(roots)
            if children:
                hidden[f"{name}_perturbed"] = tuple(children)
        return hidden, original_offset, inner

    def _adaptive_seed(self, round_index: int, agent_id: str) -> int:
        return self._seeds.derive("development_sampling", round_index, agent_id)

    def _development_plan_seed_provider(self, round_index: int, agent_id: str):
        """Record every private development EvaluationPlan seed by call."""

        def provide(call_index: int, case_index: int, case_id: str) -> int:
            del case_id
            return self._seeds.derive(
                "evaluation_plan",
                "development",
                round_index,
                agent_id,
                "call",
                call_index,
                "case_position",
                case_index,
            )

        return provide

    def _hidden_plan_seed_provider(
        self,
        seed_namespace: str,
        phase: EvaluationPhase,
        hidden: Mapping[str, Sequence[CaseRecord]],
    ):
        """Derive replay-stable plan seeds from cohort positions, not plugin ids."""

        positions = {
            record.case.case_id: (distribution, case_index)
            for distribution, records in hidden.items()
            for case_index, record in enumerate(records)
        }
        if len(positions) != sum(len(records) for records in hidden.values()):
            raise RuntimeError("hidden cohort contains duplicate Case ids")

        # Evaluation programs may explicitly pair hidden parent/child queries.
        # The default preserves each distribution's allocated seeds, including
        # evaluation banks frozen before this optional capability was added.
        paired_positions = positions
        evaluation = getattr(self, "evaluation", None)
        if getattr(evaluation, "pair_hidden_plan_seeds", False) is True:
            parents = {
                record.case.case_id: getattr(record.case, "parent_case_id", None)
                for records in hidden.values() for record in records
            }
            paired_positions = {}
            for case_id in positions:
                root, seen = case_id, set()
                while parents[root] in positions:
                    if root in seen:
                        raise RuntimeError("hidden cohort contains cyclic Case lineage")
                    seen.add(root)
                    root = parents[root]
                paired_positions[case_id] = positions[root]

        def provide(_case_index: int, case_id: str) -> int:
            try:
                distribution, position = paired_positions[case_id]
            except KeyError as exc:
                raise RuntimeError("hidden evaluation requested an unknown Case") from exc
            return self._seeds.derive(
                "evaluation_plan",
                seed_namespace,
                phase.value,
                "distribution",
                distribution,
                "case_position",
                position,
            )

        return provide

    def _official_reports(
        self,
        artifacts: Sequence[Artifact],
        hidden: Mapping[str, Sequence[CaseRecord]],
        *,
        phase: EvaluationPhase = EvaluationPhase.OFFICIAL,
        seed_namespace: str = "official_evaluation",
        generator_states_by_distribution: Mapping[str, GeneratorState] | None = None,
    ) -> tuple[
        tuple[PromotionCandidateReport | TerminalCandidateReport, ...],
        dict[str, dict[str, ArtifactEvaluation]],
    ]:
        # A configured Backend may own isolated numerical workers instead of
        # the in-process plugin device pool. Never activate both in one run.
        execution = getattr(self, "agent_execution", None)
        backend_scope = getattr(
            getattr(self, "candidate_execution", None), "hidden_evaluation_scope", None,
        )
        if callable(backend_scope):
            if getattr(execution, "evaluation_device_map", None):
                raise ValueError(
                    "hidden_evaluation_scope and evaluation_device_map cannot both be configured"
                )
            with bind_evaluation_device_pool(None), backend_scope(
                cases=tuple(record.case for records in hidden.values() for record in records),
                unavailable_gpu_devices=tuple(getattr(execution, "quarantined_gpu_devices", ())),
            ):
                return self._official_reports_in_scope(
                    artifacts, hidden, phase=phase, seed_namespace=seed_namespace,
                    generator_states_by_distribution=generator_states_by_distribution,
                )
        # Agent execution has returned and released its batch reservation before
        # this existing hidden-evaluation boundary. Bootstrap and Terminal also
        # enter here without an active Agent. The service verifies that boundary.
        reserve = getattr(getattr(self, "agent_execution", None), "reserve_evaluation_devices", None)
        scope = nullcontext(None)
        if callable(reserve):
            agent_ids = dict.fromkeys(
                agent.agent_id for plan in self.campaign.rounds for agent in plan.agents
            )
            scope = reserve(tuple(self.agents[agent_id] for agent_id in agent_ids))
        plan = getattr(self, "_active_round_plan", None)
        round_index = None if plan is None else plan.round_index
        pool = None
        try:
            with scope as pool:
                if pool is not None:
                    self.events.emit(
                        "evaluation_devices_borrowed",
                        "Idle Agent GPUs are available for Controller evaluation",
                        round_index=round_index,
                        payload={"phase": phase.value, "devices": pool.devices},
                    )
                with bind_evaluation_device_pool(pool):
                    return self._official_reports_in_scope(
                        artifacts, hidden, phase=phase, seed_namespace=seed_namespace,
                        generator_states_by_distribution=generator_states_by_distribution,
                    )
        finally:
            if pool is not None:
                clean = pool.cleanup_confirmed
                self.events.emit(
                    "evaluation_devices_returned" if clean else "evaluation_devices_quarantined",
                    "Evaluation GPU resources released" if clean else "Evaluation GPU cleanup could not be confirmed",
                    round_index=round_index,
                    payload={"phase": phase.value, "devices": pool.devices, "cleanup_confirmed": clean},
                )

    def _official_reports_in_scope(
        self,
        artifacts: Sequence[Artifact],
        hidden: Mapping[str, Sequence[CaseRecord]],
        *,
        phase: EvaluationPhase = EvaluationPhase.OFFICIAL,
        seed_namespace: str = "official_evaluation",
        generator_states_by_distribution: Mapping[str, GeneratorState] | None = None,
    ) -> tuple[
        tuple[PromotionCandidateReport | TerminalCandidateReport, ...],
        dict[str, dict[str, ArtifactEvaluation]],
    ]:
        generator_state = self._loaded_generator_state
        if generator_state is None:
            raise RuntimeError(
                "hidden evaluation requires a loaded GeneratorState"
            )
        if generator_states_by_distribution is None:
            owning_states = {name: generator_state for name in hidden}
        else:
            owning_states = dict(generator_states_by_distribution)
            if set(owning_states) != set(hidden):
                raise RuntimeError(
                    "hidden distributions and owning Generator states differ"
                )
        plan_seed_provider = self._hidden_plan_seed_provider(
            seed_namespace,
            phase,
            hidden,
        )
        verifiers = {
            name: HiddenVerifier(
                self.evaluation,
                self.campaign.evaluator.default_query,
                self.campaign.task_settings,
                phase=phase,
                seed=self._seeds.derive(seed_namespace, phase.value),
                plan_seed_provider=plan_seed_provider,
                generator=self.generator,
                generator_state=owning_states[name],
                session_lock=self._generator_operation_lock,
                candidate_execution=self.candidate_execution,
            )
            for name in hidden
        }
        control = getattr(self, "_pause_control", None)
        state = getattr(self, "_continuation_state", None)
        comparison = (state.setdefault("hidden_progress", {}).setdefault((seed_namespace, phase.value), {})
                      if control is not None and state is not None else {})
        def before_batch():
            required = float(getattr(self.candidate_execution, "case_hard_wall_seconds", 0) or 0)
            self._checkpoint_boundary("before_hidden_batch", required_seconds=required + control.hidden_start_margin_seconds)

        reports: list[PromotionCandidateReport | TerminalCandidateReport] = []
        all_evaluations: dict[str, dict[str, ArtifactEvaluation]] = {}
        for artifact in artifacts:
            evaluations: dict[str, ArtifactEvaluation] = {}
            summaries: dict[str, PromotionMetricSummary] = {}
            failures: list[str] = []
            for name, records in hidden.items():
                stage = "evaluate"
                distribution_cache = comparison.setdefault(name, {}) if control is not None else {}
                prior_failure = artifact.artifact_id in distribution_cache.get("failed_artifacts", ())
                if prior_failure:
                    failures.append(f"{name}:evaluation_failed")
                    continue
                try:
                    if control is None:
                        evaluation = verifiers[name].evaluate(artifact, records)
                    else:
                        evaluation = verifiers[name].evaluate_incremental(
                            artifact, records, progress=distribution_cache,
                            before_batch=before_batch, checkpoint=self._save_continuation,
                            batch_size=control.hidden_batch_size)
                    stage = "metric_summary"
                    summary = self._metric_summary(evaluation)
                except CampaignPaused:
                    raise
                except Exception as exc:
                    stage = getattr(exc, "_hidden_evaluation_stage", stage)
                    infrastructure_failure = (
                        (stage == "metric_summary" and not isinstance(exc, _InvalidCandidateMetrics))
                        or _has_evaluation_infrastructure_error(exc)
                    )
                    self._record_hidden_evaluation_failure(
                        artifact, phase, name, stage, exc,
                        infrastructure_failure=infrastructure_failure,
                    )
                    if infrastructure_failure:
                        raise HiddenEvaluationInfrastructureError(
                            "Hidden evaluation stopped because the Controller "
                            "evaluation service failed; untested candidates are "
                            "not classified as solver failures"
                        ) from exc
                    # Memory receives a stable code; the original exception and
                    # traceback remain in Controller-private diagnostics.
                    failures.append(f"{name}:evaluation_failed")
                    if control is not None:
                        distribution_cache.setdefault("failed_artifacts", set()).add(artifact.artifact_id)
                        self._save_continuation()
                    continue
                evaluations[name] = evaluation
                summaries[name] = summary
            # Derived summaries never enter the per-source evaluations. That
            # mapping is also the source of next-round Guidance feedback.
            if not failures and self._overall_requested(phase):
                try:
                    summaries["overall"] = self._metric_summary(
                        self._overall_evaluation(evaluations)
                    )
                except Exception as exc:
                    self._record_hidden_evaluation_failure(
                        artifact, phase, "overall", "overall_summary", exc,
                        infrastructure_failure=True,
                    )
                    raise HiddenEvaluationInfrastructureError(
                        "Hidden evaluation stopped because Controller score "
                        "aggregation failed"
                    ) from exc
            all_evaluations[artifact.artifact_id] = evaluations
            report_type = (
                TerminalCandidateReport
                if phase is EvaluationPhase.TERMINAL
                else PromotionCandidateReport
            )
            report = report_type(
                candidate_id=artifact.artifact_id,
                candidate_digest=artifact.artifact_id,
                technically_eligible=not failures,
                distributions=summaries,
                failures=tuple(failures),
            )
            reports.append(report)
            # Preserve completed nominations even if a later candidate hits an
            # infrastructure failure. Bootstrap and Terminal do not write the
            # round's Official history or reuse their scores for Promotion.
            history = getattr(self, "_candidate_history", None)
            if (
                phase is EvaluationPhase.OFFICIAL
                and seed_namespace.startswith("official_round:")
                and history is not None
                and history.needs_official_result(artifact.artifact_id)
            ):
                history.record_official_result(
                    artifact.artifact_id, report.distributions, report.failures,
                    technically_eligible=report.technically_eligible,
                )
        return tuple(reports), all_evaluations

    def _record_hidden_evaluation_failure(
        self,
        artifact: Artifact,
        phase: EvaluationPhase,
        distribution: str,
        stage: str,
        error: Exception,
        *,
        infrastructure_failure: bool,
    ) -> None:
        """Keep raw errors out of Agent memory without throwing evidence away."""

        plan = getattr(self, "_active_round_plan", None)
        reason = (
            "hidden_evaluation_infrastructure_failure"
            if infrastructure_failure else "candidate_evaluation_failed"
        )
        diagnostic = {
            "phase": phase.value,
            "round_index": None if plan is None else plan.round_index,
            "candidate_id": artifact.artifact_id,
            "distribution": distribution,
            "stage": stage,
            "reason_code": reason,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": "".join(traceback.format_exception(
                type(error), error, error.__traceback__,
            )),
            # Python 3.10 has no exception notes; attached cleanup errors still
            # need their own durable details in addition to the primary stack.
            "related_exceptions": [
                {
                    "exception_type": type(related).__name__,
                    "exception_message": str(related),
                    "traceback": "".join(traceback.format_exception(
                        type(related), related, related.__traceback__,
                    )),
                }
                for related in _evaluation_error_chain(error)[1:]
            ],
            # This can include an earlier attempt of the same candidate. Name
            # that boundary explicitly; no new worker audit may exist at all.
            "available_execution_ids_for_candidate_phase": sorted({
                audit.execution_id
                for audit in getattr(self.candidate_execution, "audits", ())
                if audit.artifact_id == artifact.artifact_id
                and audit.phase == phase.value
            }),
        }
        _LOG.error(
            "Hidden evaluation failure: phase=%s round=%s candidate=%s "
            "distribution=%s stage=%s reason=%s execution_ids=%s",
            phase.value, diagnostic["round_index"], artifact.artifact_id,
            distribution, stage, reason,
            diagnostic["available_execution_ids_for_candidate_phase"],
            exc_info=(type(error), error, error.__traceback__),
        )
        private_root = getattr(self.candidate_execution, "root", None)
        if private_root is not None:
            try:
                path = Path(private_root) / "hidden_evaluation_failures.jsonl"
                descriptor = os.open(
                    path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600,
                )
                with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(diagnostic, sort_keys=True) + "\n")
                    handle.flush()
            except Exception:
                # The original traceback is already in the run's stderr log.
                _LOG.exception("Could not persist hidden evaluation diagnostic")
        events = getattr(self, "events", None)
        if events is not None:
            events.emit(
                "hidden_evaluation_failed",
                "Hidden evaluation interrupted by a Controller service failure"
                if infrastructure_failure else "Candidate hidden evaluation failed",
                round_index=diagnostic["round_index"],
                payload={
                    "phase": phase.value,
                    "candidate_id": artifact.artifact_id,
                    "distribution": distribution,
                    "stage": stage,
                    "reason_code": reason,
                },
            )

    def _overall_requested(self, phase: EvaluationPhase) -> bool:
        if phase is EvaluationPhase.TERMINAL:
            return self.campaign.terminal_evaluation.ranking.distribution == "overall"
        return any(
            reference.distribution == "overall"
            for reference in promotion_metric_references(self.campaign.promotion)
        )

    @staticmethod
    def _overall_evaluation(
        evaluations: Mapping[str, ArtifactEvaluation],
    ) -> ArtifactEvaluation:
        """Pool real per-Case errors once, without averaging source summaries.

        Every Case has equal weight. A balanced mixture is therefore declared
        through its source counts, not inferred from source names. Named Task
        metrics retain their per-source meanings and are not averaged here.
        """
        if not evaluations or "overall" in evaluations:
            raise ValueError("overall evaluation requires nonempty real source evaluations")
        first = next(iter(evaluations.values()))
        if any(
            row.artifact.artifact_id != first.artifact.artifact_id
            for row in evaluations.values()
        ):
            raise ValueError("overall evaluation cannot combine different candidates")
        cases = tuple(case for row in evaluations.values() for case in row.cases)
        if not cases or len({case.case_id for case in cases}) != len(cases):
            raise ValueError("overall evaluation requires distinct nonempty Cases")
        return ArtifactEvaluation(
            artifact=first.artifact,
            primary_error=fmean(case.primary_error for case in cases),
            metrics={},
            cases=cases,
        )

    @classmethod
    def _evaluation_for_distribution(
        cls,
        evaluations: Mapping[str, ArtifactEvaluation],
        distribution: str,
    ) -> ArtifactEvaluation:
        if distribution == "overall":
            return cls._overall_evaluation(evaluations)
        return evaluations[distribution]

    @staticmethod
    def _metric_summary(evaluation: ArtifactEvaluation) -> PromotionMetricSummary:
        values = sorted(
            (row.primary_error for row in evaluation.cases),
            reverse=True,
        )
        if any(not isfinite(value) or value < 0 for value in values) or any(
            not isfinite(value) for value in evaluation.metrics.values()
        ):
            raise _InvalidCandidateMetrics("Candidate evaluation metrics are invalid")
        tail_count = max(1, ceil(0.2 * len(values)))
        return PromotionMetricSummary(
            case_count=len(values),
            mean=fmean(values),
            cvar=fmean(values[:tail_count]),
            maximum=values[0],
            metrics=evaluation.metrics,
        )

    def _requested_primary_distribution(self) -> str:
        settings = self.campaign.promotion.get("settings", {})
        primary = settings.get("primary", {}) if isinstance(settings, Mapping) else {}
        requested = (
            primary.get("distribution", "base")
            if isinstance(primary, Mapping)
            else "base"
        )
        return str(requested)

    def _primary_distribution(
        self,
        evaluations: Mapping[str, ArtifactEvaluation],
    ) -> str:
        requested = self._requested_primary_distribution()
        if requested == "overall" or requested in evaluations:
            return requested
        return next(iter(evaluations))

    @staticmethod
    def _hidden_metrics(
        evaluations: Mapping[str, Mapping[str, ArtifactEvaluation]],
        *,
        reports: Sequence[PromotionCandidateReport] = (),
    ) -> Mapping[str, Mapping[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for artifact_id, by_distribution in evaluations.items():
            result[artifact_id] = {
                name: {
                    "case_count": len(evaluation.cases),
                    "mean": evaluation.primary_error,
                    "maximum": max(row.primary_error for row in evaluation.cases),
                }
                for name, evaluation in by_distribution.items()
            }
        for report in reports:
            summary = report.distributions.get("overall")
            if summary is not None:
                result[report.candidate_digest]["overall"] = {
                    "case_count": summary.case_count,
                    "mean": summary.mean,
                    "cvar": summary.cvar,
                    "maximum": summary.maximum,
                }
        return result


# Compatibility import for downstream callers. It is an alias, not a second
# implementation or a PDE-dependent routing choice.
UnifiedCampaignRunner = CampaignController


__all__ = ["CampaignController", "UnifiedCampaignRunner"]
