"""Built-in PDE programs and the small public plugin registry."""

from .coefficient_flow import (
    BackendGeneratorProgram,
    CoefficientFlowGenerator,
)
from .registry import (
    LegacyStagedCampaignAdapter,
    PDEPlugin,
    built_in_plugins,
    campaign_plugin,
    get_plugin,
    legacy_staged_plugin,
)
from .toy import (
    ToyPoissonGenerator,
    ToyPoissonProblem,
    ToyResearchAgent,
    ToySineSolver,
    ToySolverTask,
)

__all__ = [
    "BackendGeneratorProgram",
    "CoefficientFlowGenerator",
    "LegacyStagedCampaignAdapter",
    "PDEPlugin",
    "ToyPoissonGenerator",
    "ToyPoissonProblem",
    "ToyResearchAgent",
    "ToySineSolver",
    "ToySolverTask",
    "built_in_plugins",
    "get_plugin",
    "legacy_staged_plugin",
]


def __getattr__(name: str):
    if name == "CampaignAdapter":
        from .registry import CampaignAdapter

        return CampaignAdapter
    raise AttributeError(name)
