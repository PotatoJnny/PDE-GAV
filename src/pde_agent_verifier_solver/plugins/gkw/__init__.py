"""GKW product programs with lazy legacy compatibility attributes.

The canonical registry imports this package without constructing GKW's old
Macro runner.  Historical adapter, Macro-loop, and evidence-bridge names stay
available through explicit imports and are loaded only when requested.
"""

from __future__ import annotations

from importlib import import_module
from warnings import warn


_EXPORTS = {
    "GKWManufacturedCase": ".manufactured_backend",
    "GKWManufacturedGeneratorBackend": ".manufactured_backend",
    "GKWManufacturingAudit": ".manufactured_backend",
    "GKWManufacturingRuntime": ".manufactured_backend",
    "GKWAnswerFirstGenerator": ".generator",
    "GKWMacroMatcher": ".macro_matching",
    "GKWObservableTask": ".tasks",
    "GKWProblem": ".problem",
    "GKWReleasedMacroTruth": ".problem",
    "GKW_DEVELOPMENT_SOLUTION_SESSION_METHODS": ".public_problem",
    "GKW_PUBLIC_SESSION_METHODS": ".public_problem",
    "GKW_SOLUTION_SAMPLING_MODES": ".selected_torch_runtime",
    "GKWSelectedTorchRuntime": ".selected_torch_runtime",
    "GKW_LEGACY_COMPUTE_BUDGET_SCHEMA": ".contract",
    "GKW_LEGACY_MACRO_OUTPUT_CONTRACT": ".contract",
    "GKW_MACRO_FIELDS": ".contract",
    "GKW_MACRO_MATCH_CONTRACT": ".contract",
    "GKW_STABLE_SOURCE_COMMIT": ".contract",
    "build_gkw_plugin": ".product",
    "gkw_public_problem_bundle": ".public_problem",
}

_LEGACY_EXPORTS = {
    "GKWLegacyObservableAdapter": ".legacy_adapter",
    "GKWLegacyRuntimeRunner": ".legacy_adapter",
    "bind_gkw_macro_loop": ".macro_loop",
    "gkw_native_macro_query": ".macro_loop",
    "GKW_LEGACY_MACRO_BRIDGE_SCHEMA": ".legacy_macro_bridge",
    "evaluate_gkw_legacy_macro_match": ".legacy_macro_bridge",
    "validate_gkw_legacy_macro_evidence": ".legacy_macro_bridge",
}

__all__ = [
    "GKWManufacturedCase",
    "GKWManufacturedGeneratorBackend",
    "GKWManufacturingAudit",
    "GKWManufacturingRuntime",
    "GKWAnswerFirstGenerator",
    "GKWMacroMatcher",
    "GKWObservableTask",
    "GKWProblem",
    "GKWReleasedMacroTruth",
    "GKW_DEVELOPMENT_SOLUTION_SESSION_METHODS",
    "GKW_PUBLIC_SESSION_METHODS",
    "GKW_SOLUTION_SAMPLING_MODES",
    "GKWSelectedTorchRuntime",
    "GKW_LEGACY_COMPUTE_BUDGET_SCHEMA",
    "GKW_LEGACY_MACRO_OUTPUT_CONTRACT",
    "GKW_MACRO_FIELDS",
    "GKW_MACRO_MATCH_CONTRACT",
    "GKW_STABLE_SOURCE_COMMIT",
    "build_gkw_plugin",
    "gkw_public_problem_bundle",
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
            f"plugins.gkw.{name} belongs to a legacy adapter/Macro surface; "
            "canonical runs use CampaignController",
            DeprecationWarning,
            stacklevel=2,
        )
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
