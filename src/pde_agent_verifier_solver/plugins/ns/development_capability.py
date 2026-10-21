"""Keep NS development tools independent of hidden evaluation capabilities."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pde_agent_verifier_solver.engine.agent_capability import (
    ControllerAgentCapabilityFacade,
)
from pde_agent_verifier_solver.engine.agent_execution import (
    AgentCapabilityLease,
    CommandAgentProgram,
)
from pde_agent_verifier_solver.engine.runtime import AgentContext
from pde_agent_verifier_solver.sdk import (
    CaseSampler,
    PairedSampleBatch,
    PairedSampleRequest,
    PerturbCasesRequest,
    SampleBatch,
    SampleCasesRequest,
    SampleSource,
)


class NSDevelopmentSamplerView:
    """Restrict Agent draws before the Controller consumes any Case budget.

    The underlying sampler still owns allowed sources, parent access, quotas,
    and truth access. Rejecting a disabled paired draw here prevents creating
    roots before a hidden-capable Generator rejects children.
    """

    def __init__(
        self,
        sampler: CaseSampler,
        *,
        guidance_enabled: bool,
        perturbations_enabled: bool,
    ) -> None:
        for value in (guidance_enabled, perturbations_enabled):
            if not isinstance(value, bool):
                raise TypeError("NS development permissions must be booleans")
        self._sampler = sampler
        self._guidance_enabled = guidance_enabled
        self._perturbations_enabled = perturbations_enabled

    def _require_source(self, source: SampleSource) -> None:
        if source is SampleSource.GUIDED and not self._guidance_enabled:
            raise PermissionError("NS development guidance is disabled")

    def _require_perturbations(self) -> None:
        if not self._perturbations_enabled:
            raise PermissionError("NS development perturbation is disabled")

    def sample_cases(self, request: SampleCasesRequest) -> SampleBatch:
        self._require_source(request.source)
        return self._sampler.sample_cases(request)

    def sample_paired_cases(self, request: PairedSampleRequest) -> PairedSampleBatch:
        self._require_source(request.source)
        self._require_perturbations()
        return self._sampler.sample_paired_cases(request)

    def perturb_cases(self, request: PerturbCasesRequest) -> SampleBatch:
        self._require_perturbations()
        return self._sampler.perturb_cases(request)


class NSDevelopmentCapabilityFacade:
    """Use the canonical Agent RPC with the run's development permissions."""

    def __init__(self, *, guidance_enabled: bool, perturbations_enabled: bool) -> None:
        for value in (guidance_enabled, perturbations_enabled):
            if not isinstance(value, bool):
                raise TypeError("NS development permissions must be booleans")
        self._guidance_enabled = guidance_enabled
        self._perturbations_enabled = perturbations_enabled
        self._inner = ControllerAgentCapabilityFacade()

    def open(
        self,
        context: AgentContext,
        workspace: Path,
        execution_id: str,
        program: CommandAgentProgram,
    ) -> AgentCapabilityLease:
        if context.sampler is not None:
            context = replace(
                context,
                sampler=NSDevelopmentSamplerView(
                    context.sampler,
                    guidance_enabled=self._guidance_enabled,
                    perturbations_enabled=self._perturbations_enabled,
                ),
            )
        return self._inner.open(context, workspace, execution_id, program)
