"""Bind an existing scientific backend to the single product Campaign.

PDE factories supply a plugin, a frozen checkpoint, and prepared data. This
module only composes those objects with the existing Compiler and Controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

from pde_agent_verifier_solver.engine.agent_execution import (
    AgentExecutionService,
    InProcessAgentExecutionService,
)
from pde_agent_verifier_solver.engine.campaign import CampaignCompiler, CampaignRecipe
from pde_agent_verifier_solver.engine.candidate_execution import (
    CandidateExecutionService,
    InProcessCandidateExecutionService,
)
from pde_agent_verifier_solver.engine.runtime import AgentProgram, CampaignResult, EventStream
from pde_agent_verifier_solver.engine.unified_runtime import CampaignController
from pde_agent_verifier_solver.plugins.registry import PDEPlugin
from pde_agent_verifier_solver.sdk import GeneratorCheckpoint, PreparedDataset


@dataclass(frozen=True)
class BackendRunBinding:
    plugin: PDEPlugin
    checkpoint: GeneratorCheckpoint
    dataset: PreparedDataset
    report: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.plugin, PDEPlugin):
            raise TypeError("backend binding requires a PDEPlugin")
        if not isinstance(self.checkpoint, GeneratorCheckpoint):
            raise TypeError("backend binding requires a GeneratorCheckpoint")
        if self.checkpoint.generator_id != self.plugin.generator.info.program_id:
            raise ValueError("backend checkpoint belongs to a different Generator")
        if not isinstance(self.dataset, PreparedDataset):
            raise TypeError("backend binding requires a PreparedDataset")

    def controller(
        self,
        recipe: CampaignRecipe,
        agents: Mapping[str, AgentProgram],
        *,
        agent_execution: AgentExecutionService | None = None,
        candidate_execution: CandidateExecutionService | None = None,
        local_development: bool = False,
        task_id: str | None = None,
        events: EventStream | None = None,
    ) -> CampaignController:
        if recipe.generator.checkpoint is not None and recipe.generator.checkpoint is not self.checkpoint:
            raise ValueError("recipe checkpoint differs from the bound backend checkpoint")
        if agent_execution is None:
            if not local_development:
                raise ValueError("provide an Agent execution service or select local_development=True")
            agent_execution = InProcessAgentExecutionService()
        if isinstance(agent_execution, InProcessAgentExecutionService) and not local_development:
            raise ValueError("in-process Agent execution requires local_development=True")
        if not local_development and (
            candidate_execution is None
            or isinstance(candidate_execution, InProcessCandidateExecutionService)
        ):
            raise ValueError(
                "formal backend runs require an explicit non-controller Candidate "
                "execution service; in-process Candidates require local_development=True"
            )
        if candidate_execution is not None:
            if not callable(getattr(candidate_execution, "execute_cases", None)):
                raise TypeError("candidate_execution must implement execute_cases(...)")
            invocation_process = getattr(
                candidate_execution, "candidate_invocation_process", None,
            )
            if not isinstance(invocation_process, str) or not invocation_process:
                raise TypeError(
                    "candidate_execution must declare candidate_invocation_process"
                )
            if not local_development and invocation_process == "controller":
                raise ValueError(
                    "Controller-process Candidate execution requires "
                    "local_development=True"
                )
        recipe = replace(recipe, generator=replace(recipe.generator, checkpoint=self.checkpoint))
        compiled = CampaignCompiler().compile(recipe, self.plugin.generator.capabilities)
        from .service import _validate_case_inventory
        _validate_case_inventory(compiled, self.dataset)
        return CampaignController(
            campaign=compiled,
            dataset=self.dataset,
            problem=self.plugin.problem,
            task=self.plugin.task(task_id),
            generator=self.plugin.generator,
            agents=agents,
            agent_execution=agent_execution,
            candidate_execution=candidate_execution,
            public_problem_bundle=self.plugin.public_problem_bundle,
            events=events,
        )

    def run(self, recipe: CampaignRecipe, agents: Mapping[str, AgentProgram], **kwargs) -> CampaignResult:
        return self.controller(recipe, agents, **kwargs).run()

    def panel(self, app, agents, *, default_query, task_id=None):
        """Expose the loaded backend through the existing Setup/Run/Result UI."""
        from .service import ProgramBundle
        from pde_agent_verifier_solver.panel.controller import PanelController
        return PanelController(
            app, ProgramBundle.from_plugin(self.plugin, agents, task_id),
            prepared_dataset=self.dataset,
            generator_checkpoint=self.checkpoint,
            default_query=default_query,
        )
