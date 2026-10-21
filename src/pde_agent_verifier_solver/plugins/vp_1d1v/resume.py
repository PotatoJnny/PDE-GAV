"""Private segmented-run codec for the pinned Artemis production backend.

Store finite public/truth arrays, compact field coordinates and frozen reward
weights, never a live model, lock, session or CUDA cache. Immutable physical
assets are reloaded by the ordinary binding. CheckpointStore owns global RNG
capture/restoration and calls these hooks only at quiescent stage boundaries.
"""
from __future__ import annotations

from dataclasses import asdict, fields
from enum import Enum
from hashlib import sha256
import importlib
import io
from pathlib import Path
from typing import Mapping
import weakref

from pde_agent_verifier_solver.engine.checkpoint_store import atomic_bytes
from pde_agent_verifier_solver.engine.common_evaluation import json_value
from pde_agent_verifier_solver.sdk import Case, CaseRecord, CaseSource, CaseVisibility, GeneratorCheckpoint, GeneratorState

from .generator import ArtemisBackend
from .guidance import ArtemisRewardGuidance, ArtemisRewardState
from ..plasma_transport_settings import check_guidance_objective


def _backend(generator):
    backend = generator._backend()
    if not isinstance(backend, ArtemisBackend):
        raise TypeError("Artemis resume requires an Artemis backend")
    return backend


def _module(backend):
    return importlib.import_module(type(backend.source.runtime).__module__)


def _identity(backend):
    runtime = backend.source.runtime
    return {
        "checkpoint": asdict(backend.checkpoint_payload),
        **({"scoring_objective": backend.score_calibration.objective_identity}
           if getattr(backend, "score_calibration", None) is not None else {}),
        **({"continuous_macro_truth": True} if getattr(backend, "continuous_macro_truth", False) else {}),
        "runtime_config": asdict(runtime.config),
        "oracle_options": asdict(runtime.oracle_options),
        "sensor_sha256": runtime.sensor_candidate.artifact.sha256,
        "conditions": list(runtime.conditions),
        "generator_id": backend.source.generator_id,
        "source_limits": [backend.source.maximum_equation_relative_residual,
                          backend.source.maximum_poisson_relative_residual,
                          backend.source.maximum_source_cross_backend_relative_l2],
        "guidance_config_sha256": (None if backend.guidance_adapter is None
                                    else backend.guidance_adapter.config.sha256),
    }


def _pack(value):
    import numpy as np
    import torch
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {key: _pack(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_pack(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported Artemis checkpoint data: {type(value).__name__}")


def _arrays(value):
    import torch
    if isinstance(value, torch.Tensor):
        array = value.numpy().copy()
        array.setflags(write=False)
        return array
    if isinstance(value, dict):
        return {key: _arrays(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_arrays(item) for item in value)
    return value


def _put(directory, value):
    import torch
    stream = io.BytesIO()
    torch.save(value, stream)
    data = stream.getvalue()
    digest = sha256(data).hexdigest()
    path = Path(directory) / (digest + ".artemis.pt")
    if path.is_symlink():
        raise ValueError("Artemis checkpoint assets cannot be symlinks")
    if path.exists():
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Artemis checkpoint asset changed")
    else:
        atomic_bytes(path, data)
    return {"asset": path.name, "sha256": digest}


def _get(directory, descriptor):
    import torch
    digest = descriptor["sha256"]
    if (not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or descriptor["asset"] != digest + ".artemis.pt"):
        raise ValueError("invalid Artemis checkpoint asset identity")
    path = Path(directory) / descriptor["asset"]
    if path.is_symlink():
        raise ValueError("Artemis checkpoint assets cannot be symlinks")
    data = path.read_bytes()
    if sha256(data).hexdigest() != digest:
        raise ValueError("Artemis checkpoint asset hash differs")
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


def _cache(backend, directory, kind):
    caches = getattr(backend, "_resume_export_caches", None)
    if caches is None:
        caches = backend._resume_export_caches = {}
    return caches.setdefault((str(Path(directory).resolve()), kind), {})


def _remember(backend, directory, kind, value, descriptor):
    cache = _cache(backend, directory, kind)
    key = id(value)
    def discard(reference):
        if cache.get(key, (None,))[0] is reference:
            cache.pop(key, None)
    cache[key] = (weakref.ref(value, discard), descriptor)
    return descriptor


def _cached(backend, directory, kind, value, build):
    old = _cache(backend, directory, kind).get(id(value))
    if old is not None and old[0]() is value:
        return old[1]
    return _remember(backend, directory, kind, value, _put(directory, build()))


def _case_asset(backend, case, directory):
    return _cached(backend, directory, "case", case,
                   lambda: {item.name: _pack(getattr(case, item.name)) for item in fields(case)})


def _truth_asset(backend, truth, directory):
    if not isinstance(truth, _module(backend).VP1DNativeMacroTruth) or truth.continuous_answer is not None:
        raise TypeError("Artemis resume requires cached native macro truth")
    return _cached(backend, directory, "truth", truth, lambda: {
        "fields": _pack(truth.fields), "t": truth.t, "x": truth.x,
    })


def _read_record(backend, payload, directory):
    row = _arrays(_get(directory, payload["case"]))
    row["source"], row["visibility"] = CaseSource(row["source"]), CaseVisibility(row["visibility"])
    case = Case(**row)
    truth = _module(backend).VP1DNativeMacroTruth(**_get(directory, payload["truth"]))
    _remember(backend, directory, "case", case, payload["case"])
    _remember(backend, directory, "truth", truth, payload["truth"])
    return CaseRecord(case, truth)


def _record_assets(backend, case, truth, directory):
    from .continuous_truth import ArtemisContinuousMacroTruth
    if isinstance(truth, ArtemisContinuousMacroTruth):
        return {**_record_assets(backend, case, truth.native, directory),
                "continuous_manufactured": _manufactured_asset(backend, truth.manufactured, directory)}
    return {"case": _case_asset(backend, case, directory),
            "truth": _truth_asset(backend, truth, directory)}


def _field_asset(backend, field, directory):
    return _cached(backend, directory, "field", field, lambda: {
        "condition_scores": _pack(field._condition_scores), "latent": _pack(field._latent),
    })


def _manufactured_asset(backend, manufactured, directory):
    def build():
        handle = backend.source.runtime._exact_query_handle(manufactured)
        return {"field": _field_asset(backend, handle.field, directory),
                "curvatures": _pack(handle.curvatures),
                "spatial_phase": asdict(handle.spatial_phase), "audit": asdict(manufactured.audit)}
    return _cached(backend, directory, "manufactured", manufactured, build)


def _restore_manufactured(backend, descriptor, record, directory, field_cache):
    import torch
    runtime, module = backend.source.runtime, _module(backend)
    row = _get(directory, descriptor)
    key = row["field"]["sha256"]
    field = field_cache.get(key)
    if field is None:
        coordinates = _get(directory, row["field"])
        field = module.ArtemisCompactFactorizedExactFieldV4(
            runtime.artifact.model, coordinates["condition_scores"], coordinates["latent"],
            jvp_mode=runtime.oracle_options.factorized_jvp_mode,
            trusted_source=runtime.oracle_options.exact_query_execution_mode == "trusted",
            vp_specialized=runtime.oracle_options.exact_query_kernel_mode == "vp_specialized",
        )
        field_cache[key] = field
        _remember(backend, directory, "field", field, row["field"])
    curvature = row["curvatures"]
    if curvature is not None:
        curvature = torch.as_tensor(curvature, dtype=field.dtype, device=field.device).detach()
    audit = module.VP1DManufacturingAudit(**row["audit"])
    handle = module._ExactQueryHandle(runtime._runtime_token, audit.state_sha256, field,
                                     curvature, module._PeriodicSpatialPhase(**row["spatial_phase"]))
    manufactured = module.VP1DManufacturedCase(record.case.case_id, record.case.public_data,
                                              record.truth, handle, audit)
    _remember(backend, directory, "manufactured", manufactured, descriptor)
    return manufactured


class _DeferredRoot:
    """Rebuild only a still-unconsumed parent's generation cache when needed.

    Public queries and cached truth need only the exact field. A later paired
    perturbation needs its original tx/quadrature cache; rebuild that from the
    same saved field without drawing a new latent or altering the case ID.
    """
    def __init__(self, runtime, record, compact, condition):
        self.runtime, self.record, self.compact, self.condition = runtime, record, compact, condition
        self._materialized = None

    @property
    def manufactured(self):
        if self._materialized is None:
            exact = self.compact.hidden_handle
            handle = self.runtime._materialize_parent_from_field(
                condition=self.condition, seed=self.compact.audit.seed, field=exact.field,
                spatial_phase=exact.spatial_phase, state_sha256=exact.state_sha256,
            )
            if handle.case_id != self.record.case.case_id:
                raise ValueError("restored Artemis parent identity changed")
            self._materialized = self.runtime._manufactured_case(handle, parent=None)
        return self._materialized


def export_runtime_state(generator, directory):
    backend = _backend(generator)
    source, guide = backend.source, backend.guidance_adapter
    if guide is not None and not isinstance(guide, ArtemisRewardGuidance):
        raise TypeError("Artemis resume supports the bound reward Guidance adapter")
    with source._lock:
        residents = []
        for row in source._issued_records.values():
            root = source._roots.get(row.case.case_id)
            condition = None if root is None else (
                root.condition if isinstance(root, _DeferredRoot) else root.manufactured.hidden_handle.condition)
            residents.append({**_record_assets(backend, row.case, row.truth, directory),
                              "case_id": row.case.case_id, "cohort": row.cohort.value,
                              "manufactured": _manufactured_asset(backend, row.manufactured, directory),
                              "root_condition": condition})
        result = {"schema": "artemis_runtime_v1", "identity": _identity(backend),
                  "issued_case_ids": sorted(source._issued_case_ids), "residents": residents,
                  "guidance": None}
    if guide is not None:
        with guide._audit_lock:
            result["guidance"] = {
                "coordinates": {key: asdict(value) for key, value in guide._coordinates.items()},
                "last_sampling_report": json_value(guide.last_sampling_report),
                "training_audits": json_value(guide.training_audits),
                "sampling_audits": json_value(guide.sampling_audits),
            }
    return result


def restore_runtime_state(generator, payload, directory):
    backend = _backend(generator)
    if backend._parallel_oracle_pool is not None:
        raise RuntimeError("restore Artemis runtime before starting an evaluation pool")
    if payload.get("schema") != "artemis_runtime_v1" or payload["identity"] != _identity(backend):
        raise ValueError("Artemis checkpoint physical assets, Guidance or execution settings changed")
    source, guide = backend.source, backend.guidance_adapter
    with source._lock:
        if source._issued_records or source._roots or source._issued_case_ids:
            raise ValueError("restore Artemis runtime before issuing any Cases")
        backend._resume_export_caches = {}
        source_module = importlib.import_module(type(source).__module__)
        from pde_agent_verifier_solver.sdk.macro_loop import CohortKind
        issued, roots, descriptors, field_cache = {}, {}, {}, {}
        seen = payload["issued_case_ids"]
        if len(set(seen)) != len(seen) or any(not isinstance(key, str) or not key for key in seen):
            raise ValueError("invalid Artemis issued Case identity set")
        for row in payload["residents"]:
            record = _read_record(backend, row, directory)
            key = record.case.case_id
            if key != row["case_id"] or key in issued or key not in seen:
                raise ValueError("Artemis checkpoint resident identity differs")
            manufactured = _restore_manufactured(backend, row["manufactured"], record, directory, field_cache)
            issued[key] = source_module._IssuedCase(record.case, record.truth, manufactured, CohortKind(row["cohort"]))
            if row["root_condition"] is not None:
                if record.case.parent_case_id is not None or row["root_condition"] not in source.runtime.conditions:
                    raise ValueError("invalid Artemis pending parent")
                roots[key] = _DeferredRoot(source.runtime, record, manufactured, row["root_condition"])
            descriptors[key] = {"case": row["case"], "truth": row["truth"]}
        if (payload["guidance"] is None) != (guide is None):
            raise ValueError("Artemis checkpoint Guidance capability changed")
        coordinates = {}
        if guide is not None:
            coordinates = {key: guide.runtime_api.ArtemisCompactV5GuidanceCoordinateCapsule(**row)
                           for key, row in payload["guidance"]["coordinates"].items()}
            if not set(coordinates) <= set(issued):
                raise ValueError("Artemis feedback coordinates refer to absent Cases")
        source._issued_records, source._roots, source._issued_case_ids = issued, roots, set(seen)
        backend._resume_restored_records = descriptors
        backend._resume_record_asset_ids = {
            (row["case"]["sha256"], row["truth"]["sha256"]): key
            for key, row in descriptors.items()
        }
    if guide is not None:
        with guide._audit_lock:
            guide._coordinates = coordinates
            for name in ("last_sampling_report", "training_audits", "sampling_audits"):
                setattr(guide, name, payload["guidance"][name])


def export_generator_state(generator, state, directory):
    backend = _backend(generator)
    backend._require_checkpoint(state.base)
    result = {"schema": "artemis_generator_state_v1", "base": asdict(state.base),
              "checkpoint": {"generator_id": state.checkpoint.generator_id,
                             "location": state.checkpoint.location, "report": json_value(state.checkpoint.report)},
              "guidance": None}
    if state.guidance is not None:
        value = state.guidance
        if not isinstance(value, ArtemisRewardState) or value.base_sha256 != backend.source.artifact_sha256:
            raise TypeError("Artemis resume requires matching frozen reward Guidance")
        guide = backend.guidance_adapter
        if guide is None or value.model.config.sha256 != guide.config.sha256:
            raise ValueError("Artemis frozen Guidance config differs")
        result["guidance"] = _cached(backend, directory, "guidance", value, lambda: {
            "base_sha256": value.base_sha256, "snapshot_sha256": value.snapshot_sha256,
            "target_solver_id": value.target_solver_id, "report": _pack(value.report),
            "normalization": value.normalization.to_dict(), "config_sha256": guide.config.sha256,
            "model": _pack(value.model.state_dict()), "training": value.model.training,
            "requires_grad": {key: parameter.requires_grad for key, parameter in value.model.named_parameters()},
        })
    return result


def restore_generator_state(generator, payload, directory):
    backend = _backend(generator)
    if payload.get("schema") != "artemis_generator_state_v1" or payload["base"] != asdict(backend.checkpoint_payload):
        raise ValueError("Artemis GeneratorState belongs to another Base")
    descriptor = payload["checkpoint"]
    if descriptor["generator_id"] != generator.info.program_id:
        raise ValueError("Artemis GeneratorState has another Generator ID")
    checkpoint = GeneratorCheckpoint(descriptor["generator_id"], backend.checkpoint_payload,
                                     descriptor["location"], descriptor["report"])
    state = None
    if payload["guidance"] is not None:
        row = _get(directory, payload["guidance"])
        guide = backend.guidance_adapter
        if (guide is None or row["config_sha256"] != guide.config.sha256
                or row["base_sha256"] != backend.source.artifact_sha256):
            raise ValueError("Artemis frozen Guidance identity differs")
        model = guide.api.ArtemisRewardModel(guide.config).to(guide.device)
        model.load_state_dict(row["model"], strict=True)
        model.train(row["training"])
        for key, parameter in model.named_parameters():
            parameter.requires_grad_(row["requires_grad"][key])
        normalization = guide.api.RewardNormalization.from_dict(row["normalization"])
        state = ArtemisRewardState(row["base_sha256"], row["snapshot_sha256"], row["target_solver_id"],
                                   model, normalization, row["report"])
        check_guidance_objective(getattr(guide, "score_calibration", None), state, {})
        _remember(backend, directory, "guidance", state, payload["guidance"])
    return GeneratorState(checkpoint, backend.checkpoint_payload, state)


def export_case_record(generator, record, directory):
    backend = _backend(generator)
    if record.generator_target is not None:
        raise TypeError("Artemis checkpoint does not support Generator training targets")
    return {"schema": "artemis_case_record_v1", **_record_assets(backend, record.case, record.truth, directory)}


def restore_case_record(generator, payload, directory):
    backend = _backend(generator)
    from .continuous_truth import continuous_record
    def restore_truth(record, resident=None):
        descriptor = payload.get("continuous_manufactured")
        if descriptor is None:
            return record
        if not getattr(backend, "continuous_macro_truth", False):
            raise ValueError("continuous Artemis truth requires the matching protocol")
        manufactured = (resident.manufactured if resident is not None else
                        _restore_manufactured(backend, descriptor, record, directory, {}))
        return continuous_record(backend, record, manufactured=manufactured)
    if payload.get("schema") != "artemis_case_record_v1":
        raise ValueError("unknown Artemis Case checkpoint schema")
    identity = getattr(backend, "_resume_record_asset_ids", {}).get(
        (payload["case"]["sha256"], payload["truth"]["sha256"]))
    resident = backend.source._issued_records.get(identity)
    if resident is not None:
        expected = backend._resume_restored_records[identity]
        if expected != {"case": payload["case"], "truth": payload["truth"]}:
            raise ValueError("Artemis Case record differs from restored resident")
        return restore_truth(CaseRecord(resident.case, resident.truth), resident)
    record = _read_record(backend, payload, directory)
    resident = backend.source._issued_records.get(record.case.case_id)
    if resident is not None:
        expected = backend._resume_restored_records[record.case.case_id]
        if expected != {"case": payload["case"], "truth": payload["truth"]}:
            raise ValueError("Artemis Case record differs from restored resident")
        return restore_truth(CaseRecord(resident.case, resident.truth), resident)
    return restore_truth(record)
