"""VPFP product programs with lazy legacy compatibility attributes.

Importing the canonical plugin registry must not assemble VPFP's historical
campaign or Macro runner.  Every package export is therefore resolved only
when a caller asks for it; the legacy names remain available for explicit
reproduction imports but are absent from the canonical star-export surface.
"""

from __future__ import annotations

from importlib import import_module
from warnings import warn


_EXPORTS = {
    "VPFPOnlineModelSpec": ".canonical_binding",
    "build_vpfp_online_binding": ".canonical_binding",
    "LegacyMacroObservableAdapter": ".canonical_binding",
    "VPFPConditionalLatentDistribution": ".learned_latent",
    "VPFPLearnedBatchGenerator": ".learned_latent",
    "VPFPAnchoredPerturbationBackend": ".perturbation_backend",
    "VPFPContinuousGenerator": ".generator",
    "VPFPContinuousMacroTruth": ".problem",
    "VPFP_DEVELOPMENT_SOLUTION_SESSION_METHODS": ".public_problem",
    "VPFPGeneratorBackend": ".generator",
    "VPFPHoldInitialMacroProgram": ".macro_programs",
    "VPFPLegacyStructuredRuntime": ".perturbation_backend",
    "VPFPMacroMatcher": ".macro_matching",
    "VPFPMacroTruth": ".problem",
    "VPFP_MAX_ORACLE_POINTS_PER_CALL": ".public_problem",
    "VPFPObservableTask": ".tasks",
    "VPFP_OPERATOR_ID": ".public_problem",
    "VPFP_PUBLIC_SESSION_METHODS": ".public_problem",
    "VPFPProblem": ".problem",
    "VPFPPhysicalRuntime": ".perturbation_backend",
    "VPFPReducedDynamicsTask": ".tasks",
    "VPFPRuntimeCase": ".perturbation_backend",
    "VPFPStatisticsTask": ".tasks",
    "load_vpfp_case_record": ".problem",
    "vpfp_public_problem_bundle": ".public_problem",
}

_LEGACY_EXPORTS = {
    "VPFPCampaignCompiler": ".campaign",
    "VPFPCampaignConfig": ".campaign",
    "load_vpfp_campaign_config": ".campaign",
    "preflight_vpfp_campaign": ".campaign",
    "VPFPLegacyObservableAdapter": ".legacy_adapter",
    "VPFPLegacyRunner": ".legacy_adapter",
    "bind_vpfp_macro_loop": ".macro_loop",
    "bind_vpfp_release_bank_reference": ".macro_loop",
    "vpfp_native_macro_query": ".macro_loop",
    "VPFPReleaseBankCaseSource": ".release_bank",
    "make_vpfp_hold_initial_cold_start": ".macro_programs",
}

__all__ = [
    "VPFPOnlineModelSpec",
    "build_vpfp_online_binding",
    "LegacyMacroObservableAdapter",
    "VPFPConditionalLatentDistribution",
    "VPFPLearnedBatchGenerator",
    "VPFPAnchoredPerturbationBackend",
    "VPFPContinuousGenerator",
    "VPFPContinuousMacroTruth",
    "VPFP_DEVELOPMENT_SOLUTION_SESSION_METHODS",
    "VPFPGeneratorBackend",
    "VPFPHoldInitialMacroProgram",
    "VPFPLegacyStructuredRuntime",
    "VPFPMacroMatcher",
    "VPFPMacroTruth",
    "VPFP_MAX_ORACLE_POINTS_PER_CALL",
    "VPFPObservableTask",
    "VPFP_OPERATOR_ID",
    "VPFP_PUBLIC_SESSION_METHODS",
    "VPFPProblem",
    "VPFPPhysicalRuntime",
    "VPFPReducedDynamicsTask",
    "VPFPRuntimeCase",
    "VPFPStatisticsTask",
    "load_vpfp_case_record",
    "vpfp_public_problem_bundle",
]


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    legacy = False
    if module_name is None:
        module_name = _LEGACY_EXPORTS.get(name)
        legacy = module_name is not None
    if module_name is None:
        raise AttributeError(name)
    if legacy:
        warn(
            f"plugins.vpfp.{name} belongs to a legacy campaign/Macro surface; "
            "canonical runs use CampaignController",
            DeprecationWarning,
            stacklevel=2,
        )
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
