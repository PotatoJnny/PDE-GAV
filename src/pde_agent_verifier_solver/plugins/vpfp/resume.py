"""Private VPFP continuation assets; never serialize models, pools or sessions.

The ordinary binding reloads frozen physical models. These hooks retain actual
continuous answers and feedback draws, not seed replay or manufactured grids.
CheckpointStore restores process RNG only after these hooks have finished.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields
from hashlib import sha256
import io
import json
from pathlib import Path
from types import MethodType
import weakref

from pde_agent_verifier_solver.engine.checkpoint_store import atomic_bytes
from pde_agent_verifier_solver.sdk import (
    Case, CaseRecord, CaseSource, CaseVisibility, GeneratorCheckpoint, GeneratorState,
)
from .formal_evaluation_bank import _pack, _unpack, _runtime_identity
from .problem import VPFPContinuousMacroTruth


def _backend(generator):
    from .iteration_backend import VPFPTrainingGuidanceBackend
    backend = generator._backend()
    if not isinstance(backend, VPFPTrainingGuidanceBackend):
        raise TypeError("VPFP continuation requires the frozen IC40 training-Guidance backend")
    if backend.runtime.materialize_cases:
        raise ValueError("VPFP continuation requires continuous demand-only answers")
    return backend


def _json_identity(value):
    """Use the CheckpointStore JSON ABI, including tuple-to-array conversion.

    Native EDM config.to_dict() deliberately retains Python tuples. Comparing
    it directly with a JSON-loaded identity would reject unchanged assets.
    Keep the native binding untouched: live sampling still requires its types.
    No default serializer or numeric coercion is allowed here.
    """
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _identity_difference_paths(saved, current, *, limit=8):
    """Bounded field names only; never dump large/private sampler reports."""
    result = []
    def visit(left, right, path):
        if len(result) >= limit or left == right:
            return
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                child = f"{path}.{key}" if path else key
                if key not in left or key not in right:
                    result.append(child)
                else:
                    visit(left[key], right[key], child)
                if len(result) >= limit:
                    break
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                result.append(f"{path}.length")
            for index, (a, b) in enumerate(zip(left, right)):
                visit(a, b, f"{path}[{index}]")
                if len(result) >= limit:
                    break
        else:
            result.append(path or "<root>")
    visit(saved, current, "")
    return tuple(result)


def _require_identity(saved, current, message):
    saved, current = _json_identity(saved), _json_identity(current)
    if saved != current:
        paths = _identity_difference_paths(saved, current)
        raise ValueError(message + "; differing fields (up to 8): " + ", ".join(paths))


def _identity(generator):
    backend = _backend(generator)
    return _json_identity({"binding": generator._identity,
            "runtime": _runtime_identity(backend.runtime),
            "guidance_config": asdict(backend.config),
            "feedback_selection": backend.feedback_selection})


def _put_bytes(directory, data, suffix):
    digest = sha256(data).hexdigest()
    path = Path(directory) / (digest + suffix)
    if path.is_symlink():
        raise ValueError("VPFP continuation assets cannot be symlinks")
    if path.exists():
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("VPFP continuation asset changed")
    else:
        atomic_bytes(path, data)
    return {"asset": path.name, "sha256": digest}


def _asset(directory, descriptor, suffix):
    digest = descriptor["sha256"]
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or descriptor["asset"] != digest + suffix):
        raise ValueError("invalid VPFP continuation asset descriptor")
    path = Path(directory) / descriptor["asset"]
    if path.is_symlink() or sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("VPFP continuation asset integrity mismatch")
    return path


def _put(directory, payload):
    import torch
    stream = io.BytesIO()
    torch.save(_pack(payload), stream)
    return _put_bytes(directory, stream.getvalue(), ".vpfp.pt")


def _get(directory, descriptor):
    import torch
    return _unpack(torch.load(_asset(directory, descriptor, ".vpfp.pt"),
                             map_location="cpu", weights_only=True))


def _cached(backend, directory, kind, value, build):
    caches = getattr(backend, "_resume_export_caches", None)
    if caches is None:
        caches = backend._resume_export_caches = {}
    cache = caches.setdefault((str(Path(directory).resolve()), kind), {})
    old = cache.get(id(value))
    if old is not None and old[0]() is value:
        return old[1]
    descriptor = build()
    _cache_descriptor(backend, directory, kind, value, descriptor)
    return descriptor


def _cache_descriptor(backend, directory, kind, value, descriptor):
    """Remember an already verified immutable asset without serializing it again."""
    caches = getattr(backend, "_resume_export_caches", None)
    if caches is None:
        caches = backend._resume_export_caches = {}
    cache = caches.setdefault((str(Path(directory).resolve()), kind), {})
    key = id(value)
    def discard(reference):
        if cache.get(key, (None,))[0] is reference:
            cache.pop(key, None)
    cache[key] = (weakref.ref(value, discard), descriptor)


def _exact_resident_payload_equal(left, right):
    """Exact persisted-value equality, not byte/alias-layout or numerical tolerance."""
    import numpy as np
    import torch
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return (left.dtype == right.dtype and left.shape == right.shape
                and left.layout == right.layout and torch.equal(left, right))
    if isinstance(left, np.ndarray):
        return (left.dtype == right.dtype and left.shape == right.shape
                and np.array_equal(left, right))
    if isinstance(left, dict):
        return (left.keys() == right.keys()
                and all(_exact_resident_payload_equal(left[key], right[key]) for key in left))
    if isinstance(left, (tuple, list)):
        return (len(left) == len(right)
                and all(_exact_resident_payload_equal(a, b) for a, b in zip(left, right)))
    return type(left) in (str, bool, int, float, type(None)) and left == right


def _case_data(case):
    return {field.name: (getattr(case, field.name).value
            if field.name in ("source", "visibility") else getattr(case, field.name))
            for field in fields(case)}


def _record(runtime, data):
    values = dict(data)
    values.update(source=CaseSource(values["source"]), visibility=CaseVisibility(values["visibility"]))
    case = Case(**values)
    arrays = case.public_data["arrays"]
    return CaseRecord(case, VPFPContinuousMacroTruth(
        case_id=case.case_id, t_axis=tuple(arrays["t"]), x_axis=tuple(arrays["x"]),
        y_axis=tuple(arrays["y"]), evaluate_points=runtime.evaluate_macros,
        evaluate_transport_points=getattr(runtime, "evaluate_transport_macros", None)))


def _resident_asset(backend, resident, directory):
    return _cached(backend, directory, "resident", resident, lambda: _put(directory, {
        "case": _case_data(resident.case), "latent": resident.latent,
        "condition": resident.equation_condition,
    }))


def export_runtime_state(generator, directory):
    backend = _backend(generator)
    runtime, source = backend.runtime, backend.backend
    with backend._lock, runtime._runtime_lock:
        # The Controller checkpoints between completed hidden batches while
        # idle GPU workers may remain alive. Only parent-owned immutable CPU
        # data crosses this boundary; worker pools and prepared caches do not.
        if backend.capture.pending or backend.capture.guide is not None:
            raise RuntimeError("checkpoint cannot interrupt a native sampling transaction")
        residents = {key: _resident_asset(backend, value, directory)
                     for key, value in runtime._resident_cases.items()}
        anchors = {}
        for key, value in source._anchors.items():
            from .perturbation_backend import _LegacyHandle
            if not isinstance(value.private_handle, _LegacyHandle) or key not in residents:
                raise TypeError("VPFP continuation cannot encode an unknown pending anchor")
            anchors[key] = _cached(backend, directory, "anchor", value,
                lambda value=value: _put(directory, {
                    "handle": {f.name: getattr(value.private_handle, f.name)
                               for f in fields(value.private_handle)},
                    "provenance": value.provenance,
                }))
        # Draws are immutable until release. Cache against their resident to
        # avoid rewriting the native tensor bank at every hidden batch boundary.
        draws = {key: _cached(backend, directory, "draw", runtime._resident_cases[key],
                             lambda value=value: _put(directory, value))
                 for key, value in backend._draws.items()}
        return {"schema": "vpfp_runtime_v1", "identity": _identity(generator),
                "residents": residents, "anchors": anchors, "draws": draws,
                "issued_case_ids": sorted(source._issued_case_ids),
                "training_audits": deepcopy(backend.training_audits),
                "sampling_audits": deepcopy(backend.sampling_audits)}


def restore_runtime_state(generator, payload, directory):
    backend = _backend(generator)
    runtime, source = backend.runtime, backend.backend
    if payload.get("schema") != "vpfp_runtime_v1":
        raise ValueError("unknown VPFP continuation runtime schema")
    _require_identity(payload["identity"], _identity(generator),
                      "VPFP continuation physical assets, execution or Guidance recipe changed")
    if (getattr(runtime, "_parallel_oracle_pool", None) is not None or runtime._resident_cases
            or source._anchors or source._issued_case_ids or backend._draws
            or backend.capture.pending or backend.capture.guide is not None):
        raise ValueError("restore VPFP continuation into a fresh inactive binding")
    seen = payload["issued_case_ids"]
    if len(set(seen)) != len(seen) or not set(seen) <= set(payload["residents"]):
        raise ValueError("VPFP continuation issued Case inventory is invalid")
    if not (set(payload["anchors"]) <= set(seen)
            and set(payload["draws"]) <= set(payload["residents"])):
        raise ValueError("VPFP continuation anchor or feedback Case is absent")
    from .perturbation_backend import _Anchor, _LegacyHandle
    registered, records = [], {}
    try:
        for key, descriptor in payload["residents"].items():
            item = _get(directory, descriptor)
            record = _record(runtime, item["case"])
            if record.case.case_id != key:
                raise ValueError("VPFP continuation resident identity changed")
            runtime._register_resident(record, item["latent"], item["condition"])
            registered.append(key)
            records[key] = record
        anchors = {}
        for key, descriptor in payload["anchors"].items():
            row = _get(directory, descriptor)
            if records[key].case.parent_case_id is not None:
                raise ValueError("VPFP continuation child cannot be a pending root")
            anchors[key] = _Anchor(records[key].case, _LegacyHandle(**row["handle"]), tuple(row["provenance"]))
        draws = {key: _get(directory, value) for key, value in payload["draws"].items()}
    except BaseException:
        runtime.release_cases(registered)
        raise
    source._anchors, source._issued_case_ids = anchors, set(seen)
    backend._draws = draws
    backend.training_audits = deepcopy(payload["training_audits"])
    backend.sampling_audits = deepcopy(payload["sampling_audits"])
    backend._resume_records = records
    backend._resume_resident_assets = dict(payload["residents"])
    backend._resume_export_caches = {}
    backend._resume_verified_resident_aliases = set()
    backend._resume_resident_repairs = []
    for key, descriptor in payload["residents"].items():
        _cache_descriptor(backend, directory, "resident", runtime._resident_cases[key], descriptor)


def export_generator_state(generator, state, directory):
    backend = _backend(generator)
    _require_identity(state.base, generator._identity, "VPFP GeneratorState differs from its frozen binding")
    _require_identity(state.checkpoint.payload, generator._identity,
                      "VPFP Generator checkpoint differs from its frozen binding")
    result = {"schema": "vpfp_generator_state_v1", "base": _json_identity(state.base),
              "checkpoint": {"generator_id": state.checkpoint.generator_id,
                  "location": state.checkpoint.location, "report": deepcopy(dict(state.checkpoint.report))},
              "guidance": None}
    if state.guidance is not None:
        from .round_solver_guidance import TrainedRoundGuidance
        value = state.guidance
        if not isinstance(value, TrainedRoundGuidance) or value.sampler.base is not backend.capture.base:
            raise TypeError("VPFP continuation requires matching frozen round Guidance")
        value.sampler.assert_frozen()
        def build():
            path = Path(value.checkpoint_path)
            if path.is_symlink():
                raise ValueError("VPFP Guidance checkpoint cannot be a symlink")
            data = path.read_bytes()
            if sha256(data).hexdigest() != value.checkpoint_sha256:
                raise ValueError("VPFP Guidance checkpoint changed")
            return _put_bytes(directory, data, ".vpfp.guide")
        result["guidance"] = {"checkpoint": _cached(backend, directory, "guidance", value, build),
                              "report": deepcopy(value.report),
                              "feedback_manifest_sha256": value.manifest_sha256}
    return result


def load_frozen_guidance(backend, path, digest, report, feedback_manifest_sha256):
    """Restore any completed round's exact regressor and standardized recipe."""
    from . import solver_error_guidance as legacy
    from .round_solver_guidance import TrainedRoundGuidance
    from .rescaled_solver_error_guidance import RescaledIC40Sampler, RescalingPolicy
    base = backend.capture.base
    if (legacy._module_sha(base.model) != report["base_state_sha256"]
            or legacy._json_sha(base.normalizer.to_payload()) != report["normalizer_sha256"]
            or legacy._json_sha(base.identity) != report["fixed_base_identity_sha256"]):
        raise ValueError("VPFP restored Guidance has a different frozen Base or normalizer")
    guide = legacy.load_solver_error_regressor(path, digest, device=next(base.model.parameters()).device)
    policy = RescalingPolicy.from_dict(dict(report["policy"]))
    sampler = RescaledIC40Sampler(base, guide, policy, max_gradient_norm=1., sigma_min=.002, sigma_max=2.)
    sampler.assert_frozen()
    if legacy._json_sha(sampler.identity) != legacy._json_sha(report["sampler_identity"]):
        raise ValueError("VPFP restored Guidance sampler identity differs")
    return TrainedRoundGuidance(guide, sampler, str(path), digest, feedback_manifest_sha256, deepcopy(report))


def restore_generator_state(generator, payload, directory):
    backend = _backend(generator)
    if payload.get("schema") != "vpfp_generator_state_v1":
        raise ValueError("unknown VPFP continuation GeneratorState schema")
    _require_identity(payload["base"], generator._identity,
                      "VPFP continuation GeneratorState has a different Base")
    item = payload["checkpoint"]
    if item["generator_id"] != generator.info.program_id:
        raise ValueError("VPFP continuation Generator ID changed")
    base = deepcopy(generator._identity)
    checkpoint = GeneratorCheckpoint(item["generator_id"], base, item["location"], item["report"])
    guidance = None
    if payload["guidance"] is not None:
        row = payload["guidance"]
        path = _asset(directory, row["checkpoint"], ".vpfp.guide")
        guidance = load_frozen_guidance(backend, path, row["checkpoint"]["sha256"],
                                       row["report"], row["feedback_manifest_sha256"])
        from ..plasma_transport_settings import check_guidance_objective
        check_guidance_objective(getattr(backend, "score_calibration", None), guidance, {})
    return GeneratorState(checkpoint, base, guidance)


def export_case_record(generator, record, directory):
    backend = _backend(generator)
    if record.generator_target is not None or not isinstance(record.truth, VPFPContinuousMacroTruth):
        raise TypeError("VPFP continuation requires continuous macro truth without a training target")
    resident = backend.runtime._active_resident(record.case)
    return {"schema": "vpfp_case_record_v1", "case_id": record.case.case_id,
            "resident": _resident_asset(backend, resident, directory)}


def restore_case_record(generator, payload, directory):
    backend = _backend(generator)
    if payload.get("schema") != "vpfp_case_record_v1":
        raise ValueError("unknown VPFP continuation Case schema")
    key = payload["case_id"]
    current = getattr(backend, "_resume_resident_assets", {}).get(key)
    if current is None:
        raise ValueError("VPFP Case record differs from its restored continuous resident")
    accepted = payload["resident"]
    if current != accepted:
        # Older checkpoints retained CaseRecord tokens while reserializing the
        # same restored resident. Different pickle alias layouts can change its
        # byte hash. Only independently verified, exactly equal assets qualify.
        pair = (str(Path(directory).resolve()), key, *sorted((
            json.dumps(current, sort_keys=True), json.dumps(accepted, sort_keys=True))))
        verified = backend._resume_verified_resident_aliases
        if pair not in verified:
            if not _exact_resident_payload_equal(_get(directory, current), _get(directory, accepted)):
                raise ValueError("VPFP Case record differs from its restored continuous resident")
            verified.add(pair)
            backend._resume_resident_repairs.append({
                "case_id": key, "previous_sha256": current["sha256"],
                "accepted_sha256": accepted["sha256"],
            })
        # CheckpointStore retains the loaded CaseRecord's original token. Align
        # the next runtime export with that same verified asset, fixing the next
        # checkpoint rather than repeating the fallback at every continuation.
        backend._resume_resident_assets[key] = accepted
        _cache_descriptor(backend, directory, "resident", backend.runtime._resident_cases[key], accepted)
    return backend._resume_records[key]


def install_resume_hooks(generator):
    """Composition-only hook binding; never changes public scientific settings."""
    _backend(generator)
    for name in ("export_runtime_state", "restore_runtime_state", "export_generator_state",
                 "restore_generator_state", "export_case_record", "restore_case_record"):
        setattr(generator, name, MethodType(globals()[name], generator))
    return generator
