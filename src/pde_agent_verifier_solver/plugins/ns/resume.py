"""Controller-private round checkpoints for the canonical structured NS backend.

Only tensors and explicit descriptors cross a process boundary. No live
backend, lock, optimizer, CUDA graph, Oracle session, or callable is pickled.
The caller checkpoints a quiescent round and restores global RNG state last,
after all model construction/loading performed here.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import fields
import hashlib
import io
import os
from pathlib import Path
import secrets
import tempfile
from weakref import WeakKeyDictionary

from pde_agent_verifier_solver.engine.common_evaluation import content_sha256, json_value
from pde_agent_verifier_solver.sdk import CaseRecord, GeneratorCheckpoint, GeneratorState
from .case_bank import _BankTruth, _case, _plan, _reference
from .continuous_backend import StructuredNSMMSAnswer, _NSVelocityTruth

_SCHEMA = "ns_private_round_runtime_v1"


def _parameters(value):
    return {item.name: json_value(getattr(value, item.name))
            for item in fields(value) if item.init}


def _put_bytes(directory, data):
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    digest = hashlib.sha256(data).hexdigest()
    path = root / (digest + ".bin")
    if not path.exists():
        descriptor, temporary = tempfile.mkstemp(prefix=".ns-checkpoint-", dir=root)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
            parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        finally:
            os.unlink(temporary)
    return {"asset": path.name, "sha256": digest}


def _asset(directory, descriptor):
    root = Path(directory).resolve()
    path = (root / descriptor["asset"]).resolve()
    if path.parent != root:
        raise ValueError("NS checkpoint asset escapes its private directory")
    return path


def _put_tensors(directory, payload):
    import torch
    stream = io.BytesIO()
    torch.save(payload, stream)
    return _put_bytes(directory, stream.getvalue())


def _get_tensors(directory, descriptor):
    import torch
    return torch.load(_asset(directory, descriptor), map_location="cpu", weights_only=True)


def _backend(generator):
    from .canonical_backend import CanonicalStructuredNSBackend
    backend = generator.backend
    if not isinstance(backend, CanonicalStructuredNSBackend):
        raise NotImplementedError("round resume currently requires canonical structured NS")
    return backend


def _export_cache(backend, directory):
    root = str(Path(directory).resolve())
    caches = getattr(backend, "_resume_export_caches", None)
    if caches is None:
        caches = backend._resume_export_caches = {}
    return caches.setdefault(root, {"fields": WeakKeyDictionary(), "files": {}})


def _export_field(backend, field, directory):
    cache = _export_cache(backend, directory)["fields"]
    if field not in cache:
        cache[field] = _put_tensors(directory, _field_payload(field))
    return cache[field]


def _export_file(backend, path, directory):
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    cache = _export_cache(backend, directory)["files"]
    if key not in cache:
        cache[key] = _put_bytes(directory, path.read_bytes())
    return cache[key]


def _field_payload(field):
    from agent_verifier.neural_generator.structured_ns_answer_first import ContinuousStructuredNSMMS
    if type(field) is not ContinuousStructuredNSMMS or field.geometry_spec is None:
        raise NotImplementedError("NS resume supports the continuous polygon MMS field")
    return {
        "config": field.config.to_dict(), "geometry": field.geometry_spec.to_dict(),
        "abi": field.continuous_mms_abi, "bounds": list(field.domain_bounds),
        "viscosity": field.viscosity, "t0": field.t0, "t1": field.t1,
        "device_kind": field.coefficients.psi.device.type,
        "psi": field.coefficients.psi.detach().cpu().clone(),
        "pressure": field.coefficients.pressure.detach().cpu().clone(),
    }


def _restore_field(backend, payload):
    from agent_verifier.neural_generator.structured_ns_answer_first import (
        ContinuousStructuredNSMMS, StructuredNSCaseCoefficients, StructuredNSProjectionConfig,
    )
    # CPU replay banks stay on CPU. Live CUDA fields follow the newly allocated
    # Controller's logical device, never the previous job's physical GPU index.
    device = (backend.components.flow_sampler._device
              if payload["device_kind"] == "cuda" else "cpu")
    return ContinuousStructuredNSMMS(
        StructuredNSCaseCoefficients(payload["psi"], payload["pressure"]),
        config=StructuredNSProjectionConfig.from_dict(payload["config"]),
        geometry_spec=payload["geometry"], continuous_mms_abi=payload["abi"],
        domain_bounds=tuple(payload["bounds"]), viscosity=payload["viscosity"],
        t0=payload["t0"], t1=payload["t1"], device=device,
    )


def _flow_identity(sampler):
    metadata = dict(sampler.artifact_metadata)
    parameter = next(sampler._model.parameters())
    metadata.update(device_type=parameter.device.type, parameter_dtype=str(parameter.dtype))
    for key in ("flow_checkpoint", "pca_checkpoint"):
        metadata[key + "_sha256"] = hashlib.sha256(Path(metadata.pop(key)).read_bytes()).hexdigest()
    return metadata


def export_runtime_state(generator, directory):
    backend = _backend(generator)
    source, sampler = backend.case_source, backend.components.flow_sampler
    with source._lock, sampler._lock:
        cache = _export_cache(backend, directory)
        if "base" not in cache:
            # The frozen RF/PCA objects are immutable during canonical runs.
            # Export once per object store, independently of resident Cases.
            pca = sampler.pca
            pca_state = {
                name: getattr(pca, name).detach().cpu().clone() for name in
                ("coordinate_mean", "field_scale", "components", "score_scale",
                 "singular_values", "total_centered_energy")
            }
            pca_state.update(coefficient_shape=list(pca.coefficient_shape),
                             fit_case_count=pca.fit_case_count,
                             source_signature=json_value(pca.source_signature))
            cache["base"] = _put_tensors(directory, {
                "flow": {name: tensor.detach().cpu().clone()
                         for name, tensor in sampler._model.state_dict().items()},
                "pca": pca_state,
            })
            cache["flow_identity"] = _flow_identity(sampler)
        residents = [{"case": json_value(row.case),
                      "field": _export_field(backend, row.raw_answer, directory),
                      "cohort": row.cohort.value,
                      "perturbation_audit": json_value(row.perturbation_audit)}
                     for row in source._issued.values()]
        payload = {
            "schema": _SCHEMA, "base": _parameters(backend.base_state),
            "flow_identity": cache["flow_identity"],
            "case_id_key": source._case_id_key.hex(),
            "maximum_points": source._max_points,
            "development_permissions": [backend.development_guidance_enabled,
                                        backend.development_perturbations_enabled],
            "numerical": cache["base"], "residents": residents,
            "guidance_attempted": [list(pair) for pair in sorted(backend._guidance_attempted)],
            "guidance": None,
        }
    guide = backend.guidance_backend
    if guide is not None:
        with guide._lock:
            assets = dict(getattr(guide, "_resume_artifacts", {}))
            assets.update({str(path): path for path in guide.artifact_root.iterdir() if path.is_file()})
            payload["guidance"] = {
                "config_sha256": guide.config.sha256,
                "attempted": sorted(guide._attempted),
                "artifacts": {uri: _export_file(backend, path, directory)
                              for uri, path in sorted(assets.items())},
                "feedback_sidecars": {
                    digest: {"asset": _export_file(backend, path, directory), "sha256": checksum}
                    for digest, (path, checksum) in guide._feedback_sidecars.items()
                },
            }
    return payload


def restore_runtime_state(generator, payload, directory):
    from pde_agent_verifier_solver.sdk.macro_loop import CohortKind
    backend = _backend(generator)
    sampler, source = backend.components.flow_sampler, backend.case_source
    if payload.get("schema") != _SCHEMA or payload["base"] != _parameters(backend.base_state):
        raise ValueError("NS checkpoint belongs to another Base Generator")
    if payload["flow_identity"] != _flow_identity(sampler):
        raise ValueError("NS checkpoint flow/PCA or sampling settings changed")
    if payload["maximum_points"] != source._max_points or payload["development_permissions"] != [
            backend.development_guidance_enabled, backend.development_perturbations_enabled]:
        raise ValueError("NS checkpoint changes development permissions or query limits")
    key = bytes.fromhex(payload["case_id_key"])
    if len(key) != 32:
        raise ValueError("NS checkpoint case identity key is invalid")
    numerical = _get_tensors(directory, payload["numerical"])
    restored = {}
    for row in payload["residents"]:
        case = _case(row["case"])
        raw = _restore_field(backend, _get_tensors(directory, row["field"]))
        if case.case_id in restored:
            raise ValueError("NS checkpoint duplicates a resident Case")
        restored[case.case_id] = source._resident(case, raw, StructuredNSMMSAnswer(raw),
                                                  CohortKind(row["cohort"]),
                                                  perturbation_audit=row["perturbation_audit"])
    with sampler._lock, source._lock:
        if source._issued:
            # Project preparation can register immutable Originals before Run
            # restoration. Only identical saved Originals may already exist;
            # fresh generated Cases must still fail this continuation boundary.
            from pde_agent_verifier_solver.sdk import CaseSource
            for case_id, resident in source._issued.items():
                saved = restored.get(case_id)
                if (resident.case.source is not CaseSource.INITIAL or saved is None
                        or json_value(resident.case) != json_value(saved.case)):
                    raise ValueError("restore NS runtime before creating any new Cases")
                current_field, saved_field = _field_payload(resident.raw_answer), _field_payload(saved.raw_answer)
                for name in current_field:
                    left, right = current_field[name], saved_field[name]
                    same = resident.raw_answer.torch.equal(left, right) if name in ("psi", "pressure") else left == right
                    if not same:
                        raise ValueError("restored NS Original differs from the prepared field")
        sampler._model.load_state_dict(numerical["flow"])
        sampler._model.eval()
        device = next(sampler._model.parameters()).device
        pca = numerical["pca"]
        sampler.pca = type(sampler.pca)(**{
            name: (value.to(device) if hasattr(value, "to") else
                   tuple(value) if name == "coefficient_shape" else value)
            for name, value in pca.items()
        })
        source._case_id_key = key
        source._issued = restored
        backend._resume_restored_fields = {row["case"]["case_id"]: row["field"] for row in payload["residents"]}
        backend._resume_export_caches = {}
        backend._guidance_attempted = {tuple(pair) for pair in payload["guidance_attempted"]}
    guide_state, guide = payload["guidance"], backend.guidance_backend
    if (guide_state is None) != (guide is None):
        raise ValueError("NS checkpoint Guidance capability changed")
    if guide is not None:
        if guide_state["config_sha256"] != guide.config.sha256:
            raise ValueError("NS checkpoint Guidance settings changed")
        artifacts = {uri: _asset(directory, asset) for uri, asset in guide_state["artifacts"].items()}
        sidecars = {}
        for digest, row in guide_state["feedback_sidecars"].items():
            path = _asset(directory, row["asset"])
            if row["sha256"] != row["asset"]["sha256"]:
                raise ValueError("NS checkpoint feedback sidecar binding changed")
            sidecars[digest] = (path, row["sha256"])
        with guide._lock:
            guide._attempted = set(guide_state["attempted"])
            guide._feedback_sidecars = sidecars
            guide._resume_artifacts = artifacts
            guide._model_cache.clear()


def export_generator_state(generator, state, directory):
    del directory
    backend = _backend(generator)
    backend._require_base_state(state.base)
    result = {"schema": "ns_generator_state_v1", "base": _parameters(state.base),
              "checkpoint": {"generator_id": state.checkpoint.generator_id,
                             "location": state.checkpoint.location,
                             "report": json_value(state.checkpoint.report)}, "guidance": None}
    if state.guidance is not None:
        from pde_agent_verifier_solver.engine.guidance_controller import FrozenGuidanceBoundary
        if not isinstance(state.guidance, FrozenGuidanceBoundary) or state.guidance._official:
            raise NotImplementedError("NS resume expects the canonical guidance sampling boundary")
        result["guidance"] = {"snapshot": _parameters(state.guidance.snapshot),
                              "config": _parameters(state.guidance.config),
                              "snapshot_sha256": state.guidance.snapshot.snapshot_sha256}
    return result


def restore_generator_state(generator, payload, directory):
    del directory
    from pde_agent_verifier_solver.engine.guidance_controller import FrozenGuidanceBoundary, GuidanceProtocolConfig
    from pde_agent_verifier_solver.sdk import FrozenGuidanceSnapshot, GuidanceTrainingBudget, GuidanceTrainingRequest
    backend = _backend(generator)
    if payload.get("schema") != "ns_generator_state_v1" or payload["base"] != _parameters(backend.base_state):
        raise ValueError("NS GeneratorState belongs to another Base")
    descriptor = payload["checkpoint"]
    checkpoint = GeneratorCheckpoint(descriptor["generator_id"], backend.base_state,
                                     descriptor["location"], descriptor["report"])
    boundary = None
    if payload["guidance"] is not None:
        guide = payload["guidance"]
        snapshot_data = dict(guide["snapshot"])
        request = dict(snapshot_data["training_request"])
        request.pop("cadence", None)
        request["training_budget"] = GuidanceTrainingBudget(**request["training_budget"])
        snapshot_data["training_request"] = GuidanceTrainingRequest(**request)
        snapshot = FrozenGuidanceSnapshot(**snapshot_data)
        if snapshot.snapshot_sha256 != guide["snapshot_sha256"]:
            raise ValueError("NS frozen Guidance snapshot identity changed")
        config = dict(guide["config"])
        config["training_budget"] = GuidanceTrainingBudget(**config["training_budget"])
        boundary = FrozenGuidanceBoundary(backend=backend.guidance_backend, snapshot=snapshot,
                    config=GuidanceProtocolConfig(**config), entropy64=lambda: secrets.randbits(63),
                    nonce_factory=lambda: secrets.token_hex(12))
        # Load once before the runner restores global RNG. Future sampling must
        # not unexpectedly consume global RNG by lazily constructing a model.
        backend.guidance_backend.open_guidance(snapshot)
    return GeneratorState(checkpoint, backend.base_state, boundary)


def export_case_record(generator, record, directory):
    backend = _backend(generator)
    answer = getattr(record.truth, "_answer", None)
    if not isinstance(answer, StructuredNSMMSAnswer) or record.generator_target is not None:
        raise NotImplementedError("NS resume Case codec requires a continuous answer")
    payload = {"schema": "ns_continuous_record_v1", "case": json_value(record.case),
               "field": _export_field(backend, answer.raw_answer, directory),
               "fixed": None}
    if isinstance(record.truth, _BankTruth):
        payload["fixed"] = {"plan": json_value(record.truth.plan), "reference": json_value(record.truth.reference)}
    elif not isinstance(record.truth, _NSVelocityTruth):
        raise NotImplementedError("unknown continuous NS truth adapter")
    return payload


def restore_case_record(generator, payload, directory):
    backend = _backend(generator)
    if payload.get("schema") != "ns_continuous_record_v1":
        raise ValueError("unknown NS resume Case schema")
    case = _case(payload["case"])
    resident = backend.case_source._issued.get(case.case_id)
    if resident is not None:
        if (content_sha256(resident.case) != content_sha256(case)
                or getattr(backend, "_resume_restored_fields", {}).get(case.case_id) != payload["field"]):
            raise ValueError("restored Case differs from its runtime resident")
        answer = resident.answer
    else:
        raw = _restore_field(backend, _get_tensors(directory, payload["field"]))
        answer = StructuredNSMMSAnswer(raw)
    fixed = payload["fixed"]
    truth = (_BankTruth(_plan(fixed["plan"]), _reference(fixed["reference"]), answer)
             if fixed is not None else _NSVelocityTruth(answer))
    return CaseRecord(case, truth)


def _mixed_identities(wrapper):
    # Frozen manifest and schedule contents cannot change during a run. Cache
    # their hashes after the first export, not once per evaluation batch.
    cached = getattr(wrapper, "_resume_identity_cache", None)
    if cached is None:
        cached = wrapper._resume_identity_cache = {
            replay.label: {"manifest_sha256": hashlib.sha256(replay.manifest_path.read_bytes()).hexdigest(),
                           "requests_sha256": content_sha256(replay.requests),
                           "contexts_sha256": content_sha256(replay.contexts)}
            for replay in wrapper._all
        }
    return cached


def export_mixed_runtime(wrapper, directory):
    with wrapper._lock:
        return {"schema": "ns_mixed_round_runtime_v1",
                "generator": wrapper._generator.export_runtime_state(directory),
                "bank_case_ids": sorted(wrapper._bank_case_ids),
                "replays": [{"label": replay.label, **_mixed_identities(wrapper)[replay.label],
                             "next_request": replay.next_request,
                             "offsets": dict(replay.offsets), "active": sorted(replay.active),
                             "issued": sorted(replay.issued), "counts": dict(replay.counts),
                             "strata": dict(replay.strata), "loaded": replay.bank is not None}
                            for replay in wrapper._all]}


def restore_mixed_runtime(wrapper, payload, directory):
    if payload.get("schema") != "ns_mixed_round_runtime_v1":
        raise ValueError("unknown NS mixed resume schema")
    if [row["label"] for row in payload["replays"]] != [replay.label for replay in wrapper._all]:
        raise ValueError("NS resume mixed cohort order changed")
    wrapper._generator.restore_runtime_state(payload["generator"], directory)
    with wrapper._lock:
        for replay, row in zip(wrapper._all, payload["replays"], strict=True):
            if (row["requests_sha256"] != content_sha256(replay.requests)
                    or row["contexts_sha256"] != content_sha256(replay.contexts)):
                raise ValueError("NS resume frozen mixed bank or request schedule changed")
            if not 0 <= row["next_request"] <= len(replay.requests):
                raise ValueError("NS resume mixed cursor is invalid")
            if row["loaded"]:
                # _load uses persisted finite references only, never evaluates
                # continuous truth, samples a Case, or reruns a Candidate.
                wrapper._load(replay)
            replay.next_request = row["next_request"]
            replay.offsets = dict(row["offsets"])
            replay.active, replay.issued = set(row["active"]), set(row["issued"])
            replay.counts, replay.strata = Counter(row["counts"]), dict(row["strata"])
            if not replay.active <= replay.issued or not replay.active <= set(wrapper._source._issued):
                raise ValueError("NS resume mixed active Cases are missing")
        wrapper._bank_case_ids = set(payload["bank_case_ids"])
