"""Shared runtime contracts for the sole Campaign Controller."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from threading import Lock
from typing import Callable, Mapping, Protocol, Sequence
from warnings import warn

from pde_agent_verifier_solver.sdk import (
    Artifact,
    ArtifactEvaluation,
    Case,
    CaseSampler,
    GeneratorProgram,
    PairedSamplingAttemptAudit,
    PreparedDataset,
    ProblemProgram,
    PromotionCandidateReport,
    PromotionDecision,
    PublicProblemBundle,
    TaskProgram,
    TerminalCandidateReport,
    resolve_evaluation_program,
)

from .agent_execution import (
    AgentExecutionService,
    InProcessAgentExecutionService,
)
from .candidate_execution import CandidateExecutionService
from .campaign import AgentSpec, CompiledCampaign, RoundPlan
from .evaluator import AgentEvaluator
from .memory import CampaignMemorySnapshot, CandidateHistorySnapshot


@dataclass(frozen=True)
class RunEvent:
    sequence: int
    event_type: str
    message: str
    round_index: int | None = None
    payload: Mapping[str, object] = field(default_factory=dict)


def _copy_run_event(event: RunEvent) -> RunEvent:
    """Return an event whose JSON-shaped payload has no shared references."""

    return RunEvent(
        sequence=event.sequence,
        event_type=event.event_type,
        message=event.message,
        round_index=event.round_index,
        payload=deepcopy(dict(event.payload)),
    )


class EventStream:
    def __init__(self, history: Sequence[RunEvent] = ()) -> None:
        if any(not isinstance(event, RunEvent) or event.sequence != index
               for index, event in enumerate(history, 1)):
            raise ValueError("event history must have contiguous sequences starting at one")
        self._events: list[RunEvent] = [_copy_run_event(event) for event in history]
        self._subscribers: list[Callable[[RunEvent], None]] = []
        self._lock = Lock()

    def subscribe(self, callback: Callable[[RunEvent], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def emit(
        self,
        event_type: str,
        message: str,
        *,
        round_index: int | None = None,
        payload: Mapping[str, object] | None = None,
    ) -> RunEvent:
        stored_payload = deepcopy(dict(payload)) if payload is not None else {}
        with self._lock:
            event = RunEvent(
                sequence=len(self._events) + 1,
                event_type=event_type,
                message=message,
                round_index=round_index,
                payload=stored_payload,
            )
            self._events.append(event)
            subscribers = tuple(self._subscribers)
        for callback in subscribers:
            callback(_copy_run_event(event))
        return _copy_run_event(event)

    def snapshot(self) -> tuple[RunEvent, ...]:
        with self._lock:
            return tuple(_copy_run_event(event) for event in self._events)


@dataclass(frozen=True)
class AgentContext:
    round_plan: RoundPlan
    agent: AgentSpec
    prompt: str
    evaluator: AgentEvaluator
    incumbent: Artifact | None
    memory: CampaignMemorySnapshot
    candidate_history: CandidateHistorySnapshot = field(
        default_factory=lambda: CandidateHistorySnapshot(()),
        kw_only=True,
    )
    seed: int = 0
    sampler: CaseSampler | None = None
    public_problem_bundle: PublicProblemBundle | None = None
    # Actual per-Case worker cap, including preparation and query; a Task
    # may independently choose a smaller preparation-only budget.
    candidate_solve_budget_seconds: float | None = field(default=None, kw_only=True)
    execution_progress: Callable[[str, Mapping[str, object]], None] | None = field(
        default=None, kw_only=True, compare=False, repr=False,
    )

    def __post_init__(self) -> None:
        budget = self.candidate_solve_budget_seconds
        if budget is not None and (
            isinstance(budget, bool) or not isinstance(budget, Real)
            or not isfinite(float(budget)) or float(budget) <= 0.0
        ):
            raise ValueError("candidate_solve_budget_seconds must be finite and positive")
        if not isinstance(self.memory, CampaignMemorySnapshot):
            raise TypeError("AgentContext memory must be a CampaignMemorySnapshot")
        if not isinstance(self.candidate_history, CandidateHistorySnapshot):
            raise TypeError(
                "AgentContext candidate_history must be a CandidateHistorySnapshot"
            )
        if (
            self.public_problem_bundle is not None
            and not isinstance(self.public_problem_bundle, PublicProblemBundle)
        ):
            raise TypeError(
                "public_problem_bundle must be a PublicProblemBundle or None"
            )

    @property
    def cases(self) -> tuple[Case, ...]:
        """Return the Cases currently available through the Evaluator.

        Adaptive Agents can sample while they work. Keeping a second tuple in
        the context would therefore become stale after the first online draw.
        """

        return self.evaluator.cases


@dataclass(frozen=True)
class AgentOutcome:
    artifact: Artifact | None = None
    observations: tuple[str, ...] = ()
    artifacts: tuple[Artifact, ...] = field(default_factory=tuple, kw_only=True)
    round_log: str = field(default="", kw_only=True)
    execution_status: str = field(default="completed", kw_only=True)

    def __post_init__(self) -> None:
        if self.artifact is not None and not isinstance(self.artifact, Artifact):
            raise TypeError("AgentOutcome.artifact must be an Artifact or None")
        if isinstance(self.observations, str):
            raise TypeError("AgentOutcome.observations must be a sequence of strings")
        observations = tuple(self.observations)
        if any(not isinstance(observation, str) for observation in observations):
            raise TypeError("AgentOutcome.observations must contain only strings")
        artifacts = tuple(self.artifacts)
        if any(not isinstance(artifact, Artifact) for artifact in artifacts):
            raise TypeError("AgentOutcome.artifacts must contain only Artifacts")
        if not isinstance(self.round_log, str):
            raise TypeError("AgentOutcome.round_log must be Markdown text")
        if self.execution_status not in {"completed", "failed", "timed_out"}:
            raise ValueError("AgentOutcome.execution_status is invalid")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "artifacts", artifacts)

    @property
    def nominations(self) -> tuple[Artifact, ...]:
        """Return every submitted Candidate without imposing a count limit."""

        if self.artifact is None:
            return self.artifacts
        return (self.artifact, *self.artifacts)


@dataclass(frozen=True)
class _ArtifactNomination:
    artifact: Artifact
    observations: tuple[str, ...]


def _unique_agent_nominations(
    outcomes: Sequence[AgentOutcome],
    *,
    excluded_artifact_ids: Sequence[str] = (),
) -> tuple[_ArtifactNomination, ...]:
    """Flatten Controller-identified Agent portfolios without a count limit.

    Canonical Controller ids are unique before this helper is called. A
    duplicate therefore indicates an internal identity bug and fails closed
    instead of silently dropping a submitted Candidate.
    """

    excluded = frozenset(excluded_artifact_ids)
    unique: dict[str, _ArtifactNomination] = {}
    for outcome in outcomes:
        for artifact in outcome.nominations:
            if artifact.artifact_id in excluded:
                continue
            if artifact.artifact_id in unique:
                raise ValueError(
                    "Controller assigned a duplicate Candidate id: "
                    f"{artifact.artifact_id}"
                )
            unique[artifact.artifact_id] = _ArtifactNomination(
                artifact,
                outcome.observations,
            )
    return tuple(unique.values())


class AgentProgram(Protocol):
    def run(self, context: AgentContext) -> AgentOutcome: ...


@dataclass(frozen=True)
class RoundResult:
    round_index: int
    case_mix: Mapping[str, int]
    candidate_evaluations: tuple[ArtifactEvaluation, ...]
    selected: ArtifactEvaluation
    hidden_suite_size: int
    agent_case_counts: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    hidden_distribution_metrics: Mapping[str, Mapping[str, object]] = field(
        default_factory=dict
    )
    promotion_reports: tuple[PromotionCandidateReport, ...] = ()
    promotion_decision: PromotionDecision | None = None


@dataclass(frozen=True)
class CampaignResult:
    """Completed run with Terminal reports aligned to every selected round.

    ``selected`` is always the terminal evaluation of the final round's S_R.
    ``terminal_round_reports[i]`` evaluates ``rounds[i].selected`` on the one
    common Terminal specification, even when an earlier selection scores best.
    """

    campaign_name: str
    selected: ArtifactEvaluation
    rounds: tuple[RoundResult, ...]
    memory: CampaignMemorySnapshot
    candidate_history: CandidateHistorySnapshot = field(
        default_factory=lambda: CandidateHistorySnapshot(()),
        kw_only=True,
    )
    terminal_distribution_metrics: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict
    )
    terminal_round_reports: tuple[TerminalCandidateReport, ...] = ()
    best_terminal_round: int | None = None
    seed_ledger: Mapping[str, int] = field(default_factory=dict)
    sampling_attempts: tuple[PairedSamplingAttemptAudit, ...] = ()


class CampaignRuntimeSupport:
    """Dependency composition shared by the one Campaign state machine."""

    def __init__(
        self,
        campaign: CompiledCampaign,
        dataset: PreparedDataset,
        problem: ProblemProgram,
        task: TaskProgram,
        generator: GeneratorProgram,
        agents: Mapping[str, AgentProgram],
        events: EventStream | None = None,
        agent_execution: AgentExecutionService | None = None,
        public_problem_bundle: PublicProblemBundle | None = None,
        candidate_execution: CandidateExecutionService | None = None,
    ) -> None:
        self.campaign = campaign
        self.dataset = dataset
        self.problem = problem
        self.task = task
        self.evaluation = resolve_evaluation_program(
            task,
            {
                **campaign.task_settings,
                "default_query": campaign.evaluator.default_query,
            },
        )
        self.generator = generator
        self.agents = agents
        if (
            public_problem_bundle is not None
            and not isinstance(public_problem_bundle, PublicProblemBundle)
        ):
            raise TypeError(
                "public_problem_bundle must be a PublicProblemBundle or None"
            )
        self.public_problem_bundle = public_problem_bundle
        self.events = events or EventStream()
        self.agent_execution = (
            agent_execution
            if agent_execution is not None
            else InProcessAgentExecutionService()
        )
        self.candidate_execution = candidate_execution

    def _validate_agents(self) -> None:
        required = {
            agent.agent_id
            for plan in self.campaign.rounds
            for agent in plan.agents
        }
        missing = required - set(self.agents)
        if missing:
            raise ValueError(f"no Agent program was provided for: {sorted(missing)}")

    def _generator_progress(
        self,
        round_index: int | None,
        stage: str,
    ) -> Callable[[str, float], None]:
        def report(message: str, fraction: float) -> None:
            self.events.emit(
                "generator_progress",
                message,
                round_index=round_index,
                payload={
                    "stage": stage,
                    "fraction": max(0.0, min(1.0, fraction)),
                },
            )

        return report


class CampaignRunner(CampaignRuntimeSupport):
    """Deprecated name that delegates to the sole CampaignController.

    This class contains no lifecycle of its own. It remains importable only so
    older integrations enter the exact same Controller while they rename their
    composition code.
    """

    def run(self) -> CampaignResult:
        warn(
            "CampaignRunner is deprecated; use CampaignController",
            DeprecationWarning,
            stacklevel=2,
        )
        from .unified_runtime import CampaignController

        return CampaignController(
            campaign=self.campaign,
            dataset=self.dataset,
            problem=self.problem,
            task=self.task,
            generator=self.generator,
            agents=self.agents,
            events=self.events,
            agent_execution=self.agent_execution,
            public_problem_bundle=getattr(self, "public_problem_bundle", None),
            candidate_execution=getattr(self, "candidate_execution", None),
        ).run()


__all__ = [
    "AgentContext",
    "AgentOutcome",
    "AgentProgram",
    "CampaignResult",
    "CampaignRunner",
    "CampaignRuntimeSupport",
    "EventStream",
    "RoundResult",
    "RunEvent",
]
