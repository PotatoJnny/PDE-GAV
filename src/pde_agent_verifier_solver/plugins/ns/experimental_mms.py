"""Prepare reconstructed NS fields and bind the ordinary numerical Generator.

Reconstruction ends at the shared continuous coefficient representation. This
module registers its Original Cases and assembles existing frozen PCA/Flow
artifacts; it contains no training, Guidance, or perturbation algorithm.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from pde_agent_verifier_solver.sdk import (
    Case, CaseRecord, CaseSource, CaseVisibility, DatasetInput,
    GeneratorCheckpoint, PreparedDataset,
)
from pde_agent_verifier_solver.sdk.macro_loop import CohortKind
from .continuous_backend import StructuredNSManufacturedCaseSource, StructuredNSMMSAnswer
from .generator import NSStructuredGenerator

_ROLES = ("generator_training", "generator_validation", "development", "final_test", "verifier")
_RUNTIME_SETTINGS = frozenset({
    "device", "integration_steps", "flow_state", "integration_method",
    "cuda_graph_sampling", "perturbation_amplitude", "perturbation_family",
    "spectral_max_mode", "max_points_per_request",
})


def _integer(settings, name, default, *, minimum=1):
    value = settings.get(name, default)
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return int(value)


def _number(settings, name, default):
    value = settings.get(name, default)
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not np.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _field_state(raw):
    from .completed_fields import field_to_state
    return field_to_state(raw)


def _answer(record):
    answer = getattr(record.truth, "_answer", None)
    if not isinstance(answer, StructuredNSMMSAnswer):
        raise ValueError("experimental MMS Cases need the shared structured NS continuous answer")
    return answer.raw_answer


def _coefficient_batch(answers):
    from agent_verifier.neural_generator.structured_ns_robustness import coefficient_tensor_from_field
    return np.stack([coefficient_tensor_from_field(answer).detach().cpu().numpy()
                     for answer in answers]).astype(np.float32)


def coefficient_sha256(values):
    """Digest an ordered [N,2,R,Kx,Ky] corpus in the numerical float32 format."""
    array = np.ascontiguousarray(values, dtype=np.float32)
    if array.ndim != 5 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError("coefficient corpus must be finite [N,2,R,Kx,Ky]")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _contract(answer):
    # Validate the actual shared field rather than relabeling a PIV-only basis.
    _field_state(answer)
    if np.any(answer.coefficients.pressure.detach().cpu().numpy() != 0):
        raise ValueError("experimental reconstruction pressure convention must remain explicitly zero")
    return {
        "domain_bounds": list(answer.domain_bounds),
        "time_interval": [float(answer.t0), float(answer.t1)],
        "viscosity": float(answer.viscosity),
        "geometry": answer.geometry_spec.to_dict(),
        "projection_config": answer.config.to_dict(),
        "continuous_mms_abi": answer.continuous_mms_abi,
        "coefficient_shape": [2, int(answer.config.time_coefficient_count),
                              int(answer.config.spatial_mode_x)+1,
                              int(answer.config.spatial_mode_y)+1],
        "pressure_model": "zero",
    }


def _all_records(dataset):
    return (*dataset.generator_training_cases, *dataset.generator_validation_cases,
            *dataset.development_cases, *dataset.hidden_cases)


def _dataset_digest(dataset):
    roles = (dataset.generator_training_cases, dataset.generator_validation_cases,
             dataset.development_cases, dataset.hidden_cases)
    payload = [[_field_state(_answer(record)) for record in records] for records in roles]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _case_dict(case):
    return {**asdict(case), "source": case.source.value, "visibility": case.visibility.value}


def _case_from_dict(payload):
    values = dict(payload)
    values["source"] = CaseSource(values["source"])
    values["visibility"] = CaseVisibility(values["visibility"])
    values["tags"] = tuple(values.get("tags", ()))
    return Case(**values)


class ExperimentalMMSDatasetRegistry:
    """Original observation-derived Cases before a frozen Generator is chosen."""

    def __init__(self):
        self._records = {}
        self._bound_dataset_sha256 = None
        self.case_source = StructuredNSManufacturedCaseSource(
            generator_id="ns-experimental-reconstructed-originals",
            artifact_sha256=hashlib.sha256(b"ns-shared-structured-originals-v1").hexdigest(),
            clean_sampler=self._no_generator, perturbation_sampler=None,
            boundary_velocity_trace=True,
        )

    @staticmethod
    def _no_generator(seed):
        raise RuntimeError("select the ordinary frozen NS PCA/Flow checkpoints before sampling")

    @property
    def public_session_methods(self):
        return self.case_source.public_session_methods

    @property
    def development_solution_session_methods(self):
        return self.case_source.development_solution_session_methods

    def public_session_factory(self, case):
        return self.case_source.public_session_factory(case)

    def development_solution_session_factory(self, case):
        return self.case_source.development_solution_session_factory(case)

    def bind_dataset(self, dataset):
        digest = _dataset_digest(dataset)
        if self._bound_dataset_sha256 is not None and self._bound_dataset_sha256 != digest:
            raise ValueError("Original Case registry is bound to different reconstructed data")
        for record in _all_records(dataset):
            raw = _answer(record)
            existing = self._records.get(record.case.case_id)
            if existing is not None:
                if existing.case != record.case or _field_state(_answer(existing)) != _field_state(raw):
                    raise ValueError("reconstructed data changed an existing Original Case")
                raw = _answer(existing)
            cohort = (CohortKind.DEVELOPMENT if record.case.visibility is CaseVisibility.DEVELOPMENT
                      else CohortKind.SELECTION)
            self.case_source.register_existing_answer(record.case, raw, cohort=cohort)
            self._records[record.case.case_id] = existing or record
        self._bound_dataset_sha256 = digest


def prepare_experimental_mms_answers(dataset: DatasetInput, split_answers: Mapping[str, Sequence], *,
                                     training_summary=None, backend=None) -> PreparedDataset:
    """Register shared continuous NS fields without exposing coefficients to solvers."""
    backend = backend or ExperimentalMMSDatasetRegistry()
    input_mode = dict(training_summary or {}).get("input_mode", "reconstructed")
    origin = "experimental_measurement" if input_mode == "measured" else "experimental_reconstruction"
    converted, counts, contracts, provenance = {}, {}, [], {}
    unknown = set(split_answers) - set(_ROLES)
    if unknown:
        raise ValueError(f"unknown experimental MMS splits: {sorted(unknown)}")
    for role in _ROLES:
        entries = tuple(split_answers.get(role, ()))
        answers = tuple(entry["answer"] if isinstance(entry, Mapping) else entry for entry in entries)
        counts[role] = len(answers)
        contracts.extend(_contract(answer) for answer in answers)
        if not answers:
            converted[role] = ()
            continue
        records = backend.case_source.issue_answer_first_records(
            answers, round_index=0,
            visibility=CaseVisibility.DEVELOPMENT if role == "development" else CaseVisibility.HIDDEN,
            case_id_prefix=f"experimental-mms-{role}", controller_seed=_ROLES.index(role),
            distribution_lineage={"origin": origin, "split": role},
            case_source=CaseSource.INITIAL,
        )
        converted[role] = records
        for record, entry in zip(records, entries):
            if isinstance(entry, Mapping):
                provenance[record.case.case_id] = {
                    name: entry[name] for name in ("case_id", "source_recording") if name in entry
                }
    if not contracts or any(value != contracts[0] for value in contracts):
        raise ValueError("experimental MMS dataset must use one continuous physical/basis contract")
    summary = {**dict(training_summary or {}), "dataset_format": "ns_experimental_mms_v1",
        "task": "initial_boundary_value_solver", "split_counts": counts,
        "terminal_final_test_cases": counts["final_test"], "official_hidden_cases": counts["verifier"],
        "continuous_contract": contracts[0], "pressure_model": "zero", "source_provenance": provenance}
    prepared = PreparedDataset(dataset, tuple(converted["development"]),
        tuple((*converted["final_test"], *converted["verifier"])), summary,
        tuple(converted["generator_training"]), tuple(converted["generator_validation"]))
    backend.bind_dataset(prepared)
    return prepared


@dataclass(frozen=True)
class ExperimentalMMSAssembly:
    prepared_dataset: PreparedDataset
    generator: NSStructuredGenerator
    checkpoint: GeneratorCheckpoint
    bundle: object

    def generator_recipe(self, *, settings=None, guidance_enabled=True, perturbations_enabled=True):
        from pde_agent_verifier_solver.engine import GeneratorRecipe
        return GeneratorRecipe(basis_size=int(self.bundle.components.flow_sampler.pca.rank),
            guidance_enabled=guidance_enabled, perturbations_enabled=perturbations_enabled,
            settings=dict(settings or {}), checkpoint=self.checkpoint)


def assemble_experimental_mms(prepared_dataset, *, flow_checkpoint, pca_checkpoint,
                              generator_id="ns-experimental-mms", generator_sha256=None,
                              generator_settings=None, guidance_artifact_root=None,
                              guidance_protocol=None, guidance_config=None,
                              development_guidance_enabled=True,
                              development_perturbations_enabled=True):
    """Load the numerical PCA/Flow formats and reuse the complete canonical backend."""
    from .canonical_backend import build_canonical_structured_ns_generator
    settings = dict(generator_settings or {})
    unknown = set(settings) - _RUNTIME_SETTINGS
    if unknown:
        raise ValueError(f"unsupported numerical NS runtime settings: {sorted(unknown)}")
    records = _all_records(prepared_dataset)
    if not records:
        raise ValueError("experimental MMS dataset contains no Original Cases")
    template = _answer(records[0])
    contract = _contract(template)
    if any(_contract(_answer(record)) != contract for record in records):
        raise ValueError("experimental MMS dataset must use one continuous physical/basis contract")
    flow_path, pca_path = Path(flow_checkpoint).resolve(), Path(pca_checkpoint).resolve()
    # Runtime identity includes the physical question contract; Generator
    # training itself depends only on coefficient corpora and their basis.
    identity = {"continuous_contract": contract,
                "flow_sha256": hashlib.sha256(flow_path.read_bytes()).hexdigest(),
                "pca_sha256": hashlib.sha256(pca_path.read_bytes()).hexdigest()}
    if generator_sha256 is None:
        generator_sha256 = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    amplitude = _number(settings, "perturbation_amplitude", None)
    bundle = build_canonical_structured_ns_generator(
        generator_id=generator_id, generator_sha256=generator_sha256,
        flow_checkpoint=flow_path, pca_checkpoint=pca_path,
        time_interval=(template.t0, template.t1), viscosity=template.viscosity,
        geometry_spec=template.geometry_spec.to_dict(),
        device=str(settings.get("device", "cuda")),
        integration_steps=_integer(settings, "integration_steps", 64),
        flow_state=str(settings.get("flow_state", "ema")),
        integration_method=str(settings.get("integration_method", "heun")),
        cuda_graph_sampling=settings.get("cuda_graph_sampling", False),
        perturbation_amplitude=amplitude,
        perturbation_family=str(settings.get("perturbation_family", "spectral_complement")),
        spectral_max_mode=_integer(settings, "spectral_max_mode", 20),
        max_points_per_request=_integer(settings, "max_points_per_request", 4096),
        guidance_artifact_root=guidance_artifact_root,
        guidance_protocol=guidance_protocol, guidance_config=guidance_config,
        development_guidance_enabled=development_guidance_enabled,
        development_perturbations_enabled=development_perturbations_enabled,
    )
    sampler = bundle.components.flow_sampler
    signature = sampler.pca.source_signature
    for label, cohort in (("training", prepared_dataset.generator_training_cases),
                          ("validation", prepared_dataset.generator_validation_cases)):
        if not cohort:
            raise ValueError(f"numerical PCA checkpoint belongs to different reconstructed {label} coefficients")
    if (signature.get("geometry") != template.geometry_spec.to_dict()
            or signature.get("time_interval") != [float(template.t0), float(template.t1)]):
        raise ValueError("numerical PCA checkpoint and reconstructed geometry/time contract differ")
    if sampler._projection.to_dict() != template.config.to_dict():
        raise ValueError("numerical PCA checkpoint and reconstructed projection contract differ")
    for record in records:
        cohort = (CohortKind.DEVELOPMENT if record.case.visibility is CaseVisibility.DEVELOPMENT
                  else CohortKind.SELECTION)
        bundle.backend.case_source.register_existing_answer(record.case, _answer(record), cohort=cohort)
    return ExperimentalMMSAssembly(prepared_dataset, bundle.generator, bundle.checkpoint, bundle)
