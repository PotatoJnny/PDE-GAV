"""Assemble the public Navier--Stokes plugin."""

from __future__ import annotations

from pde_agent_verifier_solver.plugins.registry import PDEPlugin
from .generator import NSStructuredGenerator
from .problem import NSProblem
from .public_problem import ns_public_problem_bundle
from .tasks import NSSolverTask


def _build_ns_legacy_staged_adapter():
    """Load the historical compiler only after an explicit legacy lookup."""

    from pde_agent_verifier_solver.plugins.registry import (
        LegacyStagedCampaignAdapter,
    )

    from .campaign import (
        NSCampaignConfig,
        NSCampaignCompiler,
        load_ns_campaign_config,
        preflight_ns_campaign,
    )

    compiler = NSCampaignCompiler()

    def compile_legacy_staged(config: NSCampaignConfig):
        return compiler.compile(config, legacy_staged_intent=True)

    return LegacyStagedCampaignAdapter(
        load_config=load_ns_campaign_config,
        preflight=preflight_ns_campaign,
        compile=compile_legacy_staged,
    )


def build_ns_plugin(*, task_id: str = "solver") -> PDEPlugin:
    """Return one fresh canonical plugin value for the registry and panel."""

    if task_id == "experimental_mms_solver":
        return build_ns_experimental_mms_plugin()
    if task_id != "solver":
        raise ValueError(f"unknown NS task {task_id!r}")
    # Task selection alone cannot replace a plugin's Problem/Generator/oracles.
    # The experimental solver therefore uses the explicit complete factory above.
    generator = NSStructuredGenerator()
    return PDEPlugin(
        plugin_id="ns",
        title="Navier--Stokes",
        description=(
            "Find a velocity-pressure solver with a structured "
            "continuous Generator."
        ),
        aliases=("navier_stokes", "navier-stokes"),
        problem=NSProblem(),
        generator=generator,
        tasks={"solver": NSSolverTask()},
        default_task="solver",
        public_problem_bundle=ns_public_problem_bundle(
            guidance_supported=generator.capabilities.guidance
        ),
        legacy_staged_campaign_factory=_build_ns_legacy_staged_adapter,
    )


def build_ns_experimental_mms_plugin() -> PDEPlugin:
    """Bind the new NS task to its data preparation and Generator, within the same product."""
    from .experimental_mms import ExperimentalMMSDatasetRegistry
    from .experimental_mms_project import ExperimentalMMSProblem
    from .tasks.experimental_mms import NSExperimentalMMSSolverTask

    backend = ExperimentalMMSDatasetRegistry()
    return PDEPlugin(
        plugin_id="ns", title="Navier--Stokes", description="Experimental-data NS solver",
        aliases=("navier_stokes", "navier-stokes"),
        problem=ExperimentalMMSProblem(backend), generator=NSStructuredGenerator(),
        tasks={"experimental_mms_solver": NSExperimentalMMSSolverTask()},
        default_task="experimental_mms_solver",
        public_problem_bundle=ns_public_problem_bundle(guidance_supported=True, boundary_velocity_trace=True),
    )
