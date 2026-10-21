"""Assemble the GKW plugin without coupling to an unstable campaign runtime."""

from __future__ import annotations

from pde_agent_verifier_solver.plugins.registry import PDEPlugin

from .generator import GKWAnswerFirstGenerator
from .macro_matching import GKWMacroMatcher
from .problem import GKWProblem
from .public_problem import gkw_public_problem_bundle
from .tasks import GKWObservableTask


def build_gkw_plugin(backend=None) -> PDEPlugin:
    matcher = GKWMacroMatcher()
    return PDEPlugin(
        plugin_id="gkw",
        title="GKW electrostatic gyrokinetics",
        description=(
            "Find a model for four complex macroscopic projections of a "
            "continuous five-dimensional ion-state answer."
        ),
        aliases=("gyrokinetic", "fusion-gkw", "fusion_gkw"),
        problem=GKWProblem(),
        generator=GKWAnswerFirstGenerator(backend),
        tasks={"observable_model": GKWObservableTask(matcher)},
        default_task="observable_model",
        public_problem_bundle=gkw_public_problem_bundle(),
        legacy_staged_campaign=None,
        macro_matcher=matcher,
    )
