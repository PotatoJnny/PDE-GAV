"""Label-only compatibility for the qualified, pre-mode-field GKW deployment.

This is not a fallback for an unknown sampler. The public factory requires the
exact clean physical checkout, its exact imported classes, and the startup-bound
production qualification. Sampling and decoding remain physical-package calls.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
from pathlib import Path
import subprocess
from types import MappingProxyType
from typing import Callable, Mapping


LEGACY_LOCAL_EDIT_PHYSICAL_REVISION = "80632f11a39b8577124ffa652556af1521d9b992"
LEGACY_LOCAL_EDIT_ADAPTER_IDENTITY = MappingProxyType({
    "schema": "gkw_qualified_80632f_local_edit_labels_and_bridge_v2",
    "physical_revision": LEGACY_LOCAL_EDIT_PHYSICAL_REVISION,
    "sampling_mode": "local_edit",
    "changes_sampling_math": False,
})
_PACKAGE = "agent_verifier.fusion_simulation_first"
_MISSING = object()


@dataclass(frozen=True)
class _PinnedTypes:
    service: type
    sampler: type
    config: type
    codes: type
    qualification: type
    certificate: type
    qualification_token: object
    sample: Callable
    symbols: Mapping[str, object] | None = None
    observation: type | None = None
    resident: type | None = None


# Every slot consumed by GKWSelectedTorchRuntime is bound to the same verified
# physical checkout. Only capture's explicit new metadata keyword is adapted.
_SYMBOL_BINDINGS = {
    "prepare_cases": ("gkw_generated_pde_bridge", "prepare_generated_bridge_cases"),
    "streaming_bridge": ("gkw_online_bridge", "GKWStreamingProjectedWindowBridge"),
    "selected_generator_schema": ("gkw_selected_generator", "GKW_SELECTED_GENERATOR_SCHEMA"),
    "validated_production_qualification": ("gkw_perturbation_qualification", "GKWValidatedProductionQualification"),
    "qualification_clean_maximum_attempts": ("gkw_perturbation_qualification", "GKW_QUALIFICATION_CLEAN_MAXIMUM_ATTEMPTS"),
    "qualification_clean_retry_seed": ("gkw_perturbation_qualification", "gkw_qualification_clean_retry_seed"),
    "retention_policy": ("gkw_online_bridge", "GKWOnlineRetentionPolicy"),
    "build_macro_truth": ("gkw_macro_contract", "build_structured_gkw_macro_truth"),
    "build_perturbation_macro_truth": ("gkw_perturbed_macro_truth", "build_gkw_qualified_perturbation_macro_truth"),
    "oracle_limits": ("gkw_numerical_oracle", "GKWOracleLimits"),
    "oracle_protocol": ("gkw_numerical_oracle", "gkw_numerical_oracle_protocol"),
    "capture_resident_answer": ("gkw_generated_pde_bridge", "capture_gkw_resident_answer"),
    "build_qualified_perturbed_observation": ("gkw_generated_pde_bridge", "build_gkw_qualified_perturbed_observation_from_resident"),
    "build_oracle_backend_from_resident": ("gkw_generated_pde_bridge", "build_gkw_numerical_oracle_backend_from_resident_answer"),
}


def _verified_pinned_types(root: Path, revision: str) -> _PinnedTypes:
    if revision != LEGACY_LOCAL_EDIT_PHYSICAL_REVISION:
        raise ValueError("legacy GKW adapter requires physical revision 80632f")
    observed = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    )
    if observed != revision or dirty.strip():
        raise ValueError("legacy GKW adapter requires the clean pinned physical checkout")

    def module(name):
        value = importlib.import_module(f"{_PACKAGE}.{name}")
        expected = root / "src" / "agent_verifier" / "fusion_simulation_first" / f"{name}.py"
        filename = getattr(value, "__file__", None)
        if filename is None or Path(filename).resolve() != expected.resolve():
            raise ValueError("legacy GKW class module is outside the verified physical root")
        return value

    generator = module("gkw_selected_generator")
    service = module("gkw_selected_generator_service")
    qualification = module("gkw_perturbation_qualification")
    perturbation = module("gkw_smooth_perturbation")

    def defined_type(owner, name):
        value = getattr(owner, name)
        if not isinstance(value, type) or value.__module__ != owner.__name__:
            raise TypeError("legacy GKW class identity differs from the pinned module")
        return value

    sampler = defined_type(generator, "GKWSelectedGeneratorRuntime")
    if Path(sampler.sample.__code__.co_filename).resolve() != Path(generator.__file__).resolve():
        raise TypeError("legacy GKW sampling implementation is not the pinned module method")
    return _PinnedTypes(
        service=defined_type(service, "GKWSelectedOnlineGeneratorService"),
        sampler=sampler,
        config=defined_type(generator, "GKWSelectedGeneratorConfig"),
        codes=defined_type(generator, "GKWSelectedGeneratedCodes"),
        qualification=defined_type(qualification, "GKWValidatedProductionQualification"),
        certificate=defined_type(perturbation, "GKWQualifiedProductionPerturbationConfig"),
        qualification_token=qualification._VALIDATED_PRODUCTION_QUALIFICATION_TOKEN,
        sample=sampler.sample,
        symbols=MappingProxyType({
            name: getattr(module(module_name), member)
            for name, (module_name, member) in _SYMBOL_BINDINGS.items()
        }),
        observation=defined_type(module("gkw_generated_pde_bridge"), "GKWTransientCaseObservation"),
        resident=defined_type(module("gkw_generated_pde_bridge"), "GKWProductionResidentAnswer"),
    )


def _require_missing_or_local_edit(value: object, name: str) -> None:
    observed = getattr(value, name, _MISSING)
    if observed is not _MISSING and observed != "local_edit":
        raise ValueError(f"legacy GKW {name} conflicts with qualified local_edit")


@dataclass(frozen=True)
class _ConfigView:
    _raw: object

    @property
    def sampling_mode(self):
        _require_missing_or_local_edit(self._raw, "sampling_mode")
        return "local_edit"

    def __getattr__(self, name):
        return getattr(self._raw, name)


@dataclass(frozen=True)
class _CodesView:
    _raw: object

    @property
    def solution_sampling_mode(self):
        _require_missing_or_local_edit(self._raw, "solution_sampling_mode")
        return "local_edit"

    def __getattr__(self, name):
        return getattr(self._raw, name)


@dataclass(frozen=True)
class _SamplerView:
    _raw: object
    config: _ConfigView
    _types: _PinnedTypes

    def sample(self, *args, **kwargs):
        # Fail closed if a trusted controller later replaces the checked method
        # or config. No seeds, tensors, arguments, exceptions, or retry policy
        # are changed by this call.
        if (self._raw.config is not self.config._raw
                or getattr(self._raw.sample, "__func__", None) is not self._types.sample):
            raise TypeError("legacy GKW sampler changed after adaptation")
        _require_missing_or_local_edit(self._raw.config, "sampling_mode")
        result = self._raw.sample(*args, **kwargs)
        if type(result) is not self._types.codes:
            raise TypeError("legacy GKW sampler returned an unrecognized generated-code type")
        _require_missing_or_local_edit(result, "solution_sampling_mode")
        return _CodesView(result)

    def __getattr__(self, name):
        return getattr(self._raw, name)


@dataclass(frozen=True)
class _ServiceView:
    _raw: object
    sampler: _SamplerView
    _qualification: object

    def __getattr__(self, name):
        return getattr(self._raw, name)


def _adapt_service(service, qualification, types: _PinnedTypes):
    """Internal contract seam; tests may supply tiny non-physical fixture types."""
    if type(service) is not types.service or type(service.sampler) is not types.sampler:
        raise TypeError("legacy GKW adapter rejects unknown service or sampler types")
    sampler = service.sampler
    if type(sampler.config) is not types.config:
        raise TypeError("legacy GKW adapter rejects an unknown sampler config type")
    if getattr(sampler.sample, "__func__", None) is not types.sample:
        raise TypeError("legacy GKW adapter rejects a replaced sampling method")
    _require_qualification(qualification, types)
    _require_missing_or_local_edit(sampler.config, "sampling_mode")
    return _ServiceView(service, _SamplerView(sampler, _ConfigView(sampler.config), types), qualification)


def _require_qualification(qualification, types: _PinnedTypes):
    if type(qualification) is not types.qualification:
        raise TypeError("legacy GKW adapter requires the pinned validated qualification type")
    certificate = qualification.certificate
    if (type(certificate) is not types.certificate
            or qualification._token is not types.qualification_token
            or qualification.startup_validation_performed is not True
            or certificate.physical_git_commit != LEGACY_LOCAL_EDIT_PHYSICAL_REVISION
            or certificate.qualification_code_commit != LEGACY_LOCAL_EDIT_PHYSICAL_REVISION):
        raise ValueError("legacy GKW qualification is not startup-bound to revision 80632f")
    # Recheck the physical token's digest invariants without rehashing assets or
    # constructing a replacement qualification authority.
    types.qualification.__post_init__(qualification)


@dataclass(frozen=True)
class _QualifiedLegacyCapture:
    _capture: Callable
    _types: _PinnedTypes
    _qualification: object

    def __call__(self, observation, parent=None, *, solution_sampling_mode=_MISSING):
        _require_qualification(self._qualification, self._types)
        if type(observation) is not self._types.observation:
            raise TypeError("legacy GKW capture requires the exact pinned observation type")
        if solution_sampling_mode is _MISSING and parent is None:
            raise ValueError("legacy GKW clean capture requires explicit local_edit provenance")
        if solution_sampling_mode is not _MISSING and solution_sampling_mode != "local_edit":
            raise ValueError("legacy GKW capture solution_sampling_mode conflicts with local_edit")
        if parent is not None:
            if type(parent) is not self._types.resident:
                raise TypeError("legacy GKW capture requires the exact pinned parent resident type")
            _require_missing_or_local_edit(parent, "solution_sampling_mode")
            if (parent.state is not parent.base_state
                    or parent.qualification_provenance.get("passed") is not True):
                raise ValueError("legacy GKW capture requires its qualified clean parent")
        # 80632f has only the certified local-edit sampler. The new keyword is
        # private mode metadata, not an argument to its physical capture math.
        # Pass the original objects through and leave qualification provenance,
        # storage sharing, copying, and every physical audit entirely unchanged.
        resident = self._capture(observation, parent=parent)
        if type(resident) is not self._types.resident:
            raise TypeError("legacy GKW capture returned an unrecognized resident type")
        _require_missing_or_local_edit(resident, "solution_sampling_mode")
        if (resident.case is not observation.case
                or resident.qualification_provenance.get("passed") is not True
                or (parent is None and resident.state is not resident.base_state)
                or (parent is not None and resident.base_state is not parent.base_state)):
            raise ValueError("legacy GKW capture changed qualified resident lineage")
        return resident


def bind_qualified_80632f_symbols(service, symbols, *, production_qualification):
    """Bind the exact old physical ABI, without rewriting or wrapping residents."""
    if type(service) is not _ServiceView or production_qualification is not service._qualification:
        raise TypeError("legacy GKW symbols require the already-qualified service binding")
    types = service.sampler._types
    _require_qualification(production_qualification, types)
    expected = types.symbols
    if (expected is None or set(expected) != set(_SYMBOL_BINDINGS)
            or types.observation is None or types.resident is None):
        raise TypeError("legacy GKW service has no verified physical symbol binding")
    for name, value in expected.items():
        if getattr(symbols, name, _MISSING) is not value:
            raise TypeError(f"legacy GKW physical symbol identity changed: {name}")
    if (symbols.validated_direct_noise_compatibility is not None
            or symbols.build_direct_noise_perturbed_observation is not None):
        raise TypeError("legacy GKW local-edit symbols reject direct-noise authority")
    return replace(symbols, capture_resident_answer=_QualifiedLegacyCapture(
        expected["capture_resident_answer"], types, production_qualification,
    ))


def adapt_qualified_80632f_local_edit_service(
    service: object, *, physical_root: str | Path, physical_revision: str,
    production_qualification: object,
) -> object:
    """Add only the two missing local-edit labels to the exact qualified pin.

    Call after the existing physical environment/asset qualification succeeds,
    before constructing ``GKWSelectedTorchRuntime``. Keep the original token for
    that runtime and include ``LEGACY_LOCAL_EDIT_ADAPTER_IDENTITY`` in the binding
    identity/report. A public revision string alone never authorizes adaptation.
    """
    types = _verified_pinned_types(Path(physical_root).expanduser().resolve(), physical_revision)
    return _adapt_service(service, production_qualification, types)
