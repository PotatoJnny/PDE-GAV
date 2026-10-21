"""Assemble the public VPFP plugin and its three task choices."""

from __future__ import annotations

from pde_agent_verifier_solver.plugins.registry import PDEPlugin
from .generator import VPFPContinuousGenerator
from .macro_matching import VPFPMacroMatcher
from .problem import VPFPProblem
from .public_problem import vpfp_public_problem_bundle
from .tasks import (
    VPFPObservableTask,
    VPFPReducedDynamicsTask,
    VPFPStatisticsTask,
)
from ..plasma_transport_settings import configured_matcher


def _build_vpfp_legacy_staged_adapter():
    """Load the historical compiler only after an explicit legacy lookup."""

    from pde_agent_verifier_solver.plugins.registry import (
        LegacyStagedCampaignAdapter,
    )

    from .campaign import (
        VPFPCampaignConfig,
        VPFPCampaignCompiler,
        load_vpfp_campaign_config,
        preflight_vpfp_campaign,
    )

    compiler = VPFPCampaignCompiler()

    def compile_legacy_staged(config: VPFPCampaignConfig):
        return compiler.compile(config, legacy_staged_intent=True)

    return LegacyStagedCampaignAdapter(
        load_config=load_vpfp_campaign_config,
        preflight=preflight_vpfp_campaign,
        compile=compile_legacy_staged,
    )


def build_vpfp_plugin(*, metric_profile=None, metric_calibration=None) -> PDEPlugin:
    """Return one fresh canonical VPFP plugin value for the registry and panel."""

    macro_matcher = configured_matcher(
        {"metric_profile": metric_profile, "metric_calibration": metric_calibration},
        VPFPMacroMatcher(), "vpfp")
    return PDEPlugin(
        plugin_id="vpfp",
        title="Vlasov--Poisson--Fokker--Planck",
        description=(
            "Find a macroscopic model, explicit reduced dynamics, or "
            "task-selected statistics for 2D2V VPFP."
        ),
        aliases=(
            "vlasov_poisson_fokker_planck",
            "vlasov-poisson-fokker-planck",
        ),
        problem=VPFPProblem(),
        generator=VPFPContinuousGenerator(),
        tasks={
            "observable_model": VPFPObservableTask(macro_matcher),
            "reduced_dynamics": VPFPReducedDynamicsTask(
                VPFPMacroMatcher() if getattr(macro_matcher, "calibration", None) is not None else macro_matcher),
            "statistics": VPFPStatisticsTask(),
        },
        default_task="observable_model",
        public_problem_bundle=vpfp_public_problem_bundle(),
        legacy_staged_campaign_factory=_build_vpfp_legacy_staged_adapter,
        macro_matcher=macro_matcher,
    )
