"""Navier--Stokes product plugin.

The package is intentionally lazy.  Importing the canonical registry must not
load the historical staged compiler or Macro binding merely because NS is a
built-in PDE.
"""

from __future__ import annotations

from importlib import import_module
from warnings import warn


_EXPORTS = {
    "build_ns_experimental_mms_plugin": ".product",
    "ContinuousPIVReconstruction": ".experimental_mms_project",
    "ExperimentalMMSProblem": ".experimental_mms_project",
    "save_continuous_mms_dataset": ".experimental_mms_project",
    "assemble_experimental_mms": ".experimental_mms",
    "NSExperimentalMMSSolverTask": ".tasks.experimental_mms",

    "CanonicalStructuredNSBundle": ".canonical_backend",
    "CanonicalStructuredNSBackend": ".canonical_backend",
    "CanonicalNSPIVAssembly": ".experimental_piv",
    "CompletedNSFieldAdapter": ".datasets",
    "NSDatasetCoefficientGeneratorBackend": ".field_generator",
    "NSExperimentalPIVGenerator": ".experimental_piv",
    "NSPointCloudTruth": ".datasets",
    "assemble_completed_piv": ".experimental_piv",
    "ContinuousNSAnswer": ".continuous_backend",
    "LegacyStructuredNSSmoothPerturber": ".continuous_backend",
    "StructuredNSMMSAnswer": ".continuous_backend",
    "StructuredNSManufacturedCaseSource": ".continuous_backend",
    "StructuredNSPerturbationResult": ".continuous_backend",
    "StructuredNSBaseState": ".canonical_backend",
    "build_canonical_structured_ns_generator": ".canonical_backend",
    "NSGeneratorBackend": ".generator",
    "NSStructuredGenerator": ".generator",
    "NSVelocityMatcher": ".macro_matching",
    "NS_VELOCITY_FIELDS": ".macro_matching",
    "NS_VELOCITY_MATCH_CONTRACT": ".macro_matching",
    "NSFieldTruth": ".datasets",
    "NSProblem": ".problem",
    "NS_DEVELOPMENT_SOLUTION_SESSION_METHODS": ".public_problem",
    "NS_OPERATOR_ID": ".public_problem",
    "NS_PUBLIC_SESSION_METHODS": ".public_problem",
    "ns_public_problem_bundle": ".public_problem",
    "LegacyStructuredNSFlowSampler": ".structured_flow_runtime",
    "StructuredNSArtifactBackend": ".structured_flow_runtime",
    "build_structured_ns_backend": ".structured_flow_runtime",
    "NS_STRUCTURED_GUIDANCE_BACKEND_ID": ".structured_guidance_backend",
    "StructuredNSGuidanceConfig": ".structured_guidance_backend",
    "StructuredNSMacroGuidanceBackend": ".structured_guidance_backend",
    "NSSolverTask": ".tasks",
}

_LEGACY_EXPORTS = {
    "NSCampaignCompiler": ".campaign",
    "NSCampaignConfig": ".campaign",
    "load_ns_campaign_config": ".campaign",
    "preflight_ns_campaign": ".campaign",
    "bind_ns_macro_loop": ".macro_loop",
    "NS_VELOCITY_POINT_COUNTS": ".macro_loop",
    "ns_native_velocity_query": ".macro_loop",
    "ns_velocity_point_query": ".macro_loop",
}

__all__ = [
    "build_ns_experimental_mms_plugin", "ContinuousPIVReconstruction",
    "ExperimentalMMSProblem", "save_continuous_mms_dataset", "assemble_experimental_mms",
    "NSExperimentalMMSSolverTask",

    "CanonicalStructuredNSBundle",
    "CanonicalStructuredNSBackend",
    "CanonicalNSPIVAssembly",
    "CompletedNSFieldAdapter",
    "NSDatasetCoefficientGeneratorBackend",
    "NSExperimentalPIVGenerator",
    "NSPointCloudTruth",
    "assemble_completed_piv",
    "ContinuousNSAnswer",
    "LegacyStructuredNSFlowSampler",
    "LegacyStructuredNSSmoothPerturber",
    "NSFieldTruth",
    "NSGeneratorBackend",
    "NSVelocityMatcher",
    "NS_DEVELOPMENT_SOLUTION_SESSION_METHODS",
    "NS_OPERATOR_ID",
    "NSProblem",
    "NS_PUBLIC_SESSION_METHODS",
    "NS_STRUCTURED_GUIDANCE_BACKEND_ID",
    "NSSolverTask",
    "NSStructuredGenerator",
    "NS_VELOCITY_FIELDS",
    "NS_VELOCITY_MATCH_CONTRACT",
    "StructuredNSArtifactBackend",
    "StructuredNSGuidanceConfig",
    "StructuredNSMacroGuidanceBackend",
    "StructuredNSMMSAnswer",
    "StructuredNSManufacturedCaseSource",
    "StructuredNSPerturbationResult",
    "StructuredNSBaseState",
    "build_canonical_structured_ns_generator",
    "build_structured_ns_backend",
    "ns_public_problem_bundle",
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
            f"plugins.ns.{name} belongs to a legacy campaign/Macro surface; "
            "canonical runs use CampaignController",
            DeprecationWarning,
            stacklevel=2,
        )
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
