"""Public surface for the sole multi-round Campaign pipeline."""

from __future__ import annotations

from importlib import import_module
from warnings import warn

from .agent_execution import (
    AgentBatchReservation,
    AgentCapabilityFacade,
    AgentCapabilityLease,
    AgentExecutionAudit,
    AgentExecutionError,
    AgentInfrastructureError,
    AgentExecutionRequest,
    AgentExecutionResult,
    AgentExecutionService,
    AgentExecutionTimeout,
    AgentGPUConflict,
    BatchAgentExecutionService,
    CommandAgentProgram,
    InProcessAgentExecutionService,
    SubprocessAgentExecutionService,
    execute_agent_batch,
    execute_reserved_agent_batch,
    reserve_agent_batch,
    validate_agent_batch,
    write_agent_round_log,
)
from .agent_capability import (
    AgentCapabilityClientError,
    ControllerAgentCapabilityClient,
    ControllerAgentCapabilityFacade,
)
from .campaign import (
    AgentSpec,
    CampaignCompiler,
    CampaignRecipe,
    CaseCounts,
    DevelopmentSamplingPolicy,
    EvaluatorPolicy,
    GeneratorRecipe,
    RoundOverride,
    RoundPlan,
    RoundSettings,
)
from .campaign_job import (
    CampaignJobActiveError,
    CampaignJobAdapter,
    CampaignJobError,
    CampaignJobExecutionError,
    CampaignJobResources,
    CampaignJobSpec,
    CampaignJobSubmissionError,
    CampaignJobTimeout,
)
from .candidate_execution import (
    CandidateExecutionAudit,
    CandidateExecutionBatchTimeout,
    CandidateExecutionCancelled,
    CandidateExecutionError,
    CandidateExecutionInfrastructureError,
    CandidateExecutionService,
    CandidateExecutionTimeout,
    InProcessCandidateExecutionService,
    PythonTaskWorkerSpec,
    SubprocessCandidateExecutionService,
)
from .case_sampler import RoundCaseSampler
from .evaluator import AgentEvaluator, EvaluationRequest, HiddenVerifier
from .evaluation_resources import HiddenEvaluationResourceScope
from .memory import (
    AgentRoundLog,
    CampaignMemoryLog,
    CampaignMemorySnapshot,
    CampaignRoundLog,
    CandidateHistory,
    CandidateHistorySnapshot,
    agent_round_log_template,
)
from .promotion import (
    DEFAULT_PROMOTION_PRESET,
    MINIMIZE_METRIC_PROMOTION_PRESET,
    BaseImprovementWithRobustnessGuardrails,
    FunctionPromotionProgram,
    MinimizeMetricPromotion,
    PromotionMetricReference,
    build_promotion_program,
)
from .python_candidate import (
    PythonCandidateCodec,
    PythonCandidateError,
    PythonCandidateProgram,
)
from .runtime import CampaignResult, CampaignRunner, EventStream
from .seeds import SeedLedger
from .unified_runtime import CampaignController, UnifiedCampaignRunner


__all__ = [
    "AgentEvaluator",
    "AgentBatchReservation",
    "AgentRoundLog",
    "AgentCapabilityFacade",
    "AgentCapabilityClientError",
    "AgentCapabilityLease",
    "AgentExecutionAudit",
    "AgentExecutionError",
    "AgentInfrastructureError",
    "AgentExecutionRequest",
    "AgentExecutionResult",
    "AgentExecutionService",
    "AgentExecutionTimeout",
    "AgentGPUConflict",
    "AgentSpec",
    "BatchAgentExecutionService",
    "BaseImprovementWithRobustnessGuardrails",
    "CampaignCompiler",
    "CampaignController",
    "CampaignMemoryLog",
    "CampaignMemorySnapshot",
    "CampaignRoundLog",
    "CampaignJobActiveError",
    "CampaignJobAdapter",
    "CampaignJobError",
    "CampaignJobExecutionError",
    "CampaignJobResources",
    "CampaignJobSpec",
    "CampaignJobSubmissionError",
    "CampaignJobTimeout",
    "CampaignRecipe",
    "CampaignResult",
    "CandidateHistory",
    "CandidateHistorySnapshot",
    "CandidateExecutionAudit",
    "CandidateExecutionBatchTimeout",
    "CandidateExecutionCancelled",
    "CandidateExecutionError",
    "CandidateExecutionInfrastructureError",
    "CandidateExecutionService",
    "CandidateExecutionTimeout",
    "CaseCounts",
    "CommandAgentProgram",
    "ControllerAgentCapabilityClient",
    "ControllerAgentCapabilityFacade",
    "DEFAULT_PROMOTION_PRESET",
    "DevelopmentSamplingPolicy",
    "EvaluationRequest",
    "EvaluatorPolicy",
    "EventStream",
    "FunctionPromotionProgram",
    "GeneratorRecipe",
    "HiddenVerifier",
    "HiddenEvaluationResourceScope",
    "InProcessAgentExecutionService",
    "InProcessCandidateExecutionService",
    "MINIMIZE_METRIC_PROMOTION_PRESET",
    "MinimizeMetricPromotion",
    "PromotionMetricReference",
    "PythonCandidateCodec",
    "PythonCandidateError",
    "PythonCandidateProgram",
    "PythonTaskWorkerSpec",
    "RoundCaseSampler",
    "RoundOverride",
    "RoundPlan",
    "RoundSettings",
    "SeedLedger",
    "SubprocessAgentExecutionService",
    "SubprocessCandidateExecutionService",
    "agent_round_log_template",
    "build_promotion_program",
    "execute_agent_batch",
    "execute_reserved_agent_batch",
    "reserve_agent_batch",
    "validate_agent_batch",
    "write_agent_round_log",
]


# Historical formal experiments still import these names. They remain lazy,
# deprecated compatibility surfaces and are deliberately absent from __all__;
# importing the canonical engine no longer assembles a second state machine.
_LEGACY_MACRO_EXPORTS = {
    "CompetitiveSearchPolicy": ".macro_closed_loop",
    "FinalistEvaluation": ".macro_closed_loop",
    "MacroAgentContext": ".macro_closed_loop",
    "MacroAgentOutcome": ".macro_closed_loop",
    "MacroAgentProgram": ".macro_closed_loop",
    "MacroAgentSpec": ".macro_closed_loop",
    "MacroClosedLoopRunner": ".macro_closed_loop",
    "MacroLoopConfig": ".macro_closed_loop",
    "MacroLoopResult": ".macro_closed_loop",
    "MacroRoundResult": ".macro_closed_loop",
    "PairedPublishedMetricSummary": ".macro_closed_loop",
    "PublishedMetricAggregate": ".macro_closed_loop",
    "PublishedMetricSummary": ".macro_closed_loop",
    "RoundFeedback": ".macro_closed_loop",
    "RoundHandoff": ".macro_closed_loop",
    "SearchCompletionAudit": ".macro_closed_loop",
    "SearchLeaderboardEntry": ".macro_closed_loop",
    "TerminalEvaluation": ".macro_closed_loop",
    "evaluate_search_completion": ".macro_closed_loop",
    "verify_round_handoff": ".macro_closed_loop",
    "DevelopmentCreditExceeded": ".development_session",
    "DevelopmentDrawKind": ".development_session",
    "DevelopmentDrawRequest": ".development_session",
    "DevelopmentRelease": ".development_session",
    "DevelopmentScreenResponse": ".development_session",
    "DevelopmentSession": ".development_session",
    "DevelopmentSessionError": ".development_session",
    "CaseCreditExceeded": ".macro_evaluator",
    "ControllerMacroEvaluator": ".macro_evaluator",
    "DevelopmentMacroCaseResult": ".macro_evaluator",
    "DevelopmentMacroEvaluator": ".macro_evaluator",
    "DevelopmentMacroResponse": ".macro_evaluator",
    "MacroBatchEvaluation": ".macro_evaluator",
    "PreparedMacroCohort": ".macro_evaluator",
    "SanitizedMacroCaseScore": ".macro_evaluator",
    "EvidenceStage": ".selection",
    "MacroCandidateScore": ".selection",
    "PairedComparison": ".selection",
    "PairedObjectiveMode": ".selection",
    "PairedObjectivePolicy": ".selection",
    "PairedScoreSplit": ".selection",
    "PerturbedGuardrailComparison": ".selection",
    "PromotionResult": ".selection",
    "RobustAggregate": ".selection",
    "aggregate": ".selection",
    "aggregate_score": ".selection",
    "compare_perturbed_non_regression": ".selection",
    "paired_compare": ".selection",
    "paired_objective_score": ".selection",
    "paired_weighted_composite": ".selection",
    "select_or_keep": ".selection",
    "split_paired_score": ".selection",
}


def __getattr__(name: str):
    module_name = _LEGACY_MACRO_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    warn(
        f"engine.{name} belongs to the legacy formal macro surface; "
        "new product runs use CampaignController",
        DeprecationWarning,
        stacklevel=2,
    )
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
