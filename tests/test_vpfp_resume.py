"""Tiny continuation codec checks; no physical model loading or GPU work."""
from dataclasses import dataclass
import json
from threading import RLock
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp import resume
from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import _ResidentCase, _Anchor, _LegacyHandle
from pde_agent_verifier_solver.sdk import Case, CaseRecord, CaseSource, CaseVisibility, GeneratorCheckpoint, GeneratorState


@dataclass
class Config:
    steps: int = 2000


def binding(monkeypatch):
    runtime = SimpleNamespace(materialize_cases=False, _resident_cases={}, _parallel_oracle_pool=None,
                              _runtime_lock=RLock(), evaluate_macros=lambda *_: None)
    def register(record, latent, condition):
        if record.case.case_id in runtime._resident_cases:
            raise ValueError("duplicate")
        runtime._resident_cases[record.case.case_id] = _ResidentCase(
            record.case, record.case, latent, np.asarray(condition))
    runtime._register_resident = register
    runtime.release_cases = lambda ids: [runtime._resident_cases.pop(key, None) for key in ids]
    runtime._active_resident = lambda case: runtime._resident_cases[case.case_id]
    backend = SimpleNamespace(runtime=runtime, backend=SimpleNamespace(_anchors={}, _issued_case_ids=set()),
        capture=SimpleNamespace(pending={}, guide=None, base=object()), _draws={}, _lock=RLock(),
        config=Config(), feedback_selection="fixed", training_audits=[], sampling_audits=[])
    generator = SimpleNamespace(_backend=lambda: backend, _identity={"frozen": "base"},
                                info=SimpleNamespace(program_id="vpfp"))
    monkeypatch.setattr(resume, "_backend", lambda g: g._backend())
    monkeypatch.setattr(resume, "_runtime_identity", lambda r: {"frozen_decoder": True})
    return generator, backend


def resident(backend, key="root"):
    case = Case(key, CaseSource.GENERATED, CaseVisibility.HIDDEN,
                {"arrays": {"t": np.array([0., 1.]), "x": np.array([0., 1.]), "y": np.array([0., 1.])}},
                round_index=3)
    record = resume._record(backend.runtime, resume._case_data(case))
    backend.runtime._register_resident(record, torch.arange(5, dtype=torch.float64)[None], np.array([24., 1.]))
    backend.backend._issued_case_ids.add(key)
    backend._draws[key] = {"source": "base", "root_seed": 7,
                           "draw": {"full": torch.arange(3)[None], "root_seeds": [7]}}
    backend.backend._anchors[key] = _Anchor(case, _LegacyHandle(torch.ones(1, 3), torch.ones(1, 5),
                                                              torch.tensor([[24., 1.]])), ("vpfp", "sha", "cohort"))
    return record


def test_runtime_round_trip_keeps_exact_residents_anchors_and_actual_draws(tmp_path, monkeypatch):
    gen, old = binding(monkeypatch); record = resident(old)
    old.training_audits = [{"target_round_index": 3}]
    payload = resume.export_runtime_state(gen, tmp_path)
    newgen, new = binding(monkeypatch)
    resume.restore_runtime_state(newgen, payload, tmp_path)
    assert torch.equal(new.runtime._resident_cases["root"].latent, old.runtime._resident_cases["root"].latent)
    assert new._draws["root"]["root_seed"] == 7
    assert torch.equal(new._draws["root"]["draw"]["full"], old._draws["root"]["draw"]["full"])
    assert new.backend._anchors["root"].case is new._resume_records["root"].case
    assert new.training_audits == [{"target_round_index": 3}]
    descriptor = resume.export_case_record(gen, record, tmp_path)
    assert resume.restore_case_record(newgen, descriptor, tmp_path) is new._resume_records["root"]


def test_export_does_not_rewrite_immutable_latent_bank(tmp_path, monkeypatch):
    gen, backend = binding(monkeypatch); resident(backend)
    first = resume.export_runtime_state(gen, tmp_path)
    def fail(*args):
        pytest.fail("immutable tensors rewritten")
    monkeypatch.setattr(resume, "_put", fail)
    assert resume.export_runtime_state(gen, tmp_path) == first


def alternate_resident_asset(directory, descriptor, change=None):
    """Change serialization order, not values; no physical model is involved."""
    item = resume._get(directory, descriptor)
    item = dict(reversed(tuple(item.items())))
    if change is not None:
        change(item)
    alternate = resume._put(directory, item)
    assert alternate != descriptor
    return alternate


@pytest.mark.parametrize("stale_descriptor", [False, True])
def test_three_checkpoint_round_trips_keep_verified_resident_descriptor(
        tmp_path, monkeypatch, stale_descriptor):
    from pde_agent_verifier_solver.engine.checkpoint_store import CheckpointStore, atomic_json
    gen, backend = binding(monkeypatch)
    resume.install_resume_hooks(gen)
    record = resident(backend)
    backend.backend._anchors.clear()
    backend._draws.clear()
    store = CheckpointStore(tmp_path, identity={"test": "three-resident-restores"}, generator=gen)
    store.save({"record": record})
    manifest = json.loads(store.path.read_bytes())
    original = manifest["generator_runtime"]["residents"]["root"]
    if stale_descriptor:
        # Reproduce the existing broken ABI: runtime reserialized B, while the
        # memoized CaseRecord JSON still names semantically identical asset A.
        manifest["generator_runtime"]["residents"]["root"] = alternate_resident_asset(store.objects, original)
        atomic_json(store.path, manifest)
    def fail(*args):
        pytest.fail("restored resident must not be reserialized")
    monkeypatch.setattr(resume, "_put", fail)
    for index in range(3):
        fresh_gen, fresh = binding(monkeypatch)
        resume.install_resume_hooks(fresh_gen)
        resumed = CheckpointStore(tmp_path, identity=store.identity, generator=fresh_gen)
        restored = resumed.load()
        assert restored["record"] is fresh._resume_records["root"]
        assert torch.equal(fresh.runtime._resident_cases["root"].latent, backend.runtime._resident_cases["root"].latent)
        assert len(fresh._resume_resident_repairs) == int(stale_descriptor and index == 0)
        resumed.save(restored)
        saved = json.loads(resumed.path.read_bytes())
        assert saved["generator_runtime"]["residents"]["root"] == original
        assert fresh._resume_resident_assets["root"] == original


def test_equal_stale_descriptor_comparison_cached_and_export_aligned(tmp_path, monkeypatch):
    gen, backend = binding(monkeypatch); record = resident(backend)
    runtime = resume.export_runtime_state(gen, tmp_path)
    descriptor = resume.export_case_record(gen, record, tmp_path)
    original = descriptor["resident"]
    alternate = alternate_resident_asset(tmp_path, original)
    runtime["residents"]["root"] = alternate
    newgen, new = binding(monkeypatch)
    resume.restore_runtime_state(newgen, runtime, tmp_path)
    restored = resume.restore_case_record(newgen, descriptor, tmp_path)
    assert new._resume_resident_repairs == [{"case_id": "root",
        "previous_sha256": alternate["sha256"], "accepted_sha256": original["sha256"]}]
    def fail(*args):
        pytest.fail("same exact-equality comparison repeated")
    monkeypatch.setattr(resume, "_get", fail)
    for accepted in (alternate, original, alternate, original):
        assert resume.restore_case_record(newgen, {**descriptor, "resident": accepted}, tmp_path) is restored
        assert resume.export_case_record(newgen, restored, tmp_path)["resident"] == accepted
    assert len(new._resume_resident_repairs) == 1


@pytest.mark.parametrize("change", [
    lambda item: item["latent"].add_(1),
    lambda item: item.update(latent=item["latent"].float()),
    lambda item: item.update(latent=item["latent"].reshape(-1)),
    lambda item: item.update(condition=item["condition"] + 1),
    lambda item: item.update(condition=item["condition"].astype(np.float32)),
    lambda item: item.update(condition=item["condition"].reshape(1, 2)),
    lambda item: item["case"].update(case_id="other"),
    lambda item: item["case"].update(round_index=4),
    lambda item: item["case"].update(round_index=3.),
    lambda item: item["case"]["public_data"]["arrays"].update(t=np.array([0., 2.])),
    lambda item: item["case"]["public_data"]["arrays"].update(t=np.array([0., 1.], dtype=np.float32)),
    lambda item: item["case"]["public_data"]["arrays"].update(t=[0., 1.]),
    lambda item: item.update(extra="unapproved"),
], ids=["latent-values", "latent-dtype", "latent-shape", "condition-values", "condition-dtype",
        "condition-shape", "case-id", "metadata-value", "metadata-type", "public-values", "public-dtype",
        "public-container", "extra-field"])
def test_stale_descriptor_requires_exact_persisted_payload(tmp_path, monkeypatch, change):
    gen, backend = binding(monkeypatch); record = resident(backend)
    runtime = resume.export_runtime_state(gen, tmp_path)
    descriptor = resume.export_case_record(gen, record, tmp_path)
    alternate = alternate_resident_asset(tmp_path, descriptor["resident"], change)
    newgen, new = binding(monkeypatch)
    resume.restore_runtime_state(newgen, runtime, tmp_path)
    with pytest.raises(ValueError, match="differs from its restored continuous resident"):
        resume.restore_case_record(newgen, {**descriptor, "resident": alternate}, tmp_path)
    assert new._resume_resident_assets == runtime["residents"]
    assert new._resume_resident_repairs == []


@pytest.mark.parametrize("broken", ["missing", "corrupt"])
def test_stale_descriptor_missing_or_corrupt_asset_is_not_recovered(tmp_path, monkeypatch, broken):
    gen, backend = binding(monkeypatch); record = resident(backend)
    runtime = resume.export_runtime_state(gen, tmp_path)
    descriptor = resume.export_case_record(gen, record, tmp_path)
    alternate = alternate_resident_asset(tmp_path, descriptor["resident"])
    newgen, new = binding(monkeypatch)
    resume.restore_runtime_state(newgen, runtime, tmp_path)
    path = tmp_path / alternate["asset"]
    if broken == "missing":
        path.unlink()
    else:
        path.write_bytes(b"corrupt")
    with pytest.raises((ValueError, FileNotFoundError)):
        resume.restore_case_record(newgen, {**descriptor, "resident": alternate}, tmp_path)
    assert new._resume_resident_repairs == []


@pytest.mark.parametrize("busy", ["pending", "guide"])
def test_export_requires_quiescent_backend(tmp_path, monkeypatch, busy):
    gen, backend = binding(monkeypatch)
    if busy == "pending": backend.capture.pending[7] = object()
    else: backend.capture.guide = object()
    with pytest.raises(RuntimeError): resume.export_runtime_state(gen, tmp_path)


def test_between_batch_export_omits_live_gpu_pool(tmp_path, monkeypatch):
    gen, backend = binding(monkeypatch); resident(backend)
    backend.runtime._parallel_oracle_pool = object()
    payload = resume.export_runtime_state(gen, tmp_path)
    assert "pool" not in payload and "cache" not in payload
    newgen, _ = binding(monkeypatch)
    resume.restore_runtime_state(newgen, payload, tmp_path)


def test_corrupt_assets_rejected_and_partial_restore_released(tmp_path, monkeypatch):
    gen, backend = binding(monkeypatch); resident(backend)
    payload = resume.export_runtime_state(gen, tmp_path)
    (tmp_path/payload["anchors"]["root"]["asset"]).write_bytes(b"changed")
    newgen, new = binding(monkeypatch)
    with pytest.raises(ValueError, match="integrity"):
        resume.restore_runtime_state(newgen, payload, tmp_path)
    assert not new.runtime._resident_cases and not new.backend._issued_case_ids


@pytest.mark.parametrize("field", ["identity", "issued_case_ids", "anchors", "draws"])
def test_unknown_or_incompatible_saved_inventory_rejected(tmp_path, monkeypatch, field):
    gen, backend = binding(monkeypatch); resident(backend)
    payload = resume.export_runtime_state(gen, tmp_path)
    payload[field] = ["missing"] if field == "issued_case_ids" else {"missing": {}}
    newgen, _ = binding(monkeypatch)
    with pytest.raises(ValueError): resume.restore_runtime_state(newgen, payload, tmp_path)


def test_case_descriptor_cannot_substitute_another_resident(tmp_path, monkeypatch):
    gen, backend = binding(monkeypatch); record = resident(backend)
    payload = resume.export_runtime_state(gen, tmp_path)
    descriptor = resume.export_case_record(gen, record, tmp_path)
    newgen, _ = binding(monkeypatch); resume.restore_runtime_state(newgen, payload, tmp_path)
    descriptor["case_id"] = "other"
    with pytest.raises(ValueError): resume.restore_case_record(newgen, descriptor, tmp_path)


def test_generator_base_round_trip_and_instance_hooks(tmp_path, monkeypatch):
    gen, backend = binding(monkeypatch)
    resume.install_resume_hooks(gen)
    state = GeneratorState(GeneratorCheckpoint("vpfp", gen._identity), gen._identity)
    saved = gen.export_generator_state(state, tmp_path)
    restored = gen.restore_generator_state(saved, tmp_path)
    assert restored.base == state.base and restored.base is restored.checkpoint.payload
    assert restored.guidance is None
    saved["base"] = {"wrong": True}
    with pytest.raises(ValueError): gen.restore_generator_state(saved, tmp_path)


def test_guidance_export_preserves_original_checkpoint_bytes_and_manifest(tmp_path, monkeypatch):
    from pde_agent_verifier_solver.plugins.vpfp.round_solver_guidance import TrainedRoundGuidance
    from hashlib import sha256
    gen, backend = binding(monkeypatch)
    data = b"synthetic checkpoint bytes"; path = tmp_path/"fixed-step.pt"; path.write_bytes(data)
    digest = sha256(data).hexdigest()
    guide = TrainedRoundGuidance(object(), SimpleNamespace(base=backend.capture.base, assert_frozen=lambda: None),
                                str(path), digest, "feedback-sha", {"target_round_index": 3})
    state = GeneratorState(GeneratorCheckpoint("vpfp", gen._identity), gen._identity, guide)
    payload = resume.export_generator_state(gen, state, tmp_path)
    assert payload["guidance"]["feedback_manifest_sha256"] == "feedback-sha"
    assert (tmp_path/payload["guidance"]["checkpoint"]["asset"]).read_bytes() == data
    calls = []
    monkeypatch.setattr(resume, "load_frozen_guidance", lambda *a: calls.append(a) or guide)
    assert resume.restore_generator_state(gen, payload, tmp_path).guidance is guide
    assert calls[0][2:] == (digest, {"target_round_index": 3}, "feedback-sha")


def native_binding_identity():
    # Exactly the config constructor/type path in the actual IC40 sampler.
    # This creates no model, tensors or physical trajectories.
    from pde_agent_verifier_solver.plugins.vpfp.initial_conditioned_diffusion import VPFPInitialConditionedEDMConfig
    from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import VPFPStructuredEDMConfig
    config = VPFPInitialConditionedEDMConfig(VPFPStructuredEDMConfig(collision_nu_range=(10., 30.)))
    return {"frozen": "base", "sampling": {"model_config": config.to_dict()}}


def test_actual_native_config_exposes_the_four_json_sequence_type_changes():
    native = native_binding_identity()
    saved = json.loads(json.dumps(native))
    left = native["sampling"]["model_config"]["full_config"]
    right = saved["sampling"]["model_config"]["full_config"]
    assert {key for key in left if left[key] != right[key]} == {
        "temporal_dilation_cycle", "collision_nu_range", "temperature_range", "time_domain"}
    assert native != saved
    assert resume._json_identity(native) == saved
    assert isinstance(left["time_domain"], tuple)  # Canonicalization never mutates native identity.


def test_checkpoint_store_json_round_trip_preserves_native_binding_and_case(tmp_path, monkeypatch):
    from pde_agent_verifier_solver.engine.checkpoint_store import CheckpointStore
    gen, old = binding(monkeypatch)
    gen._identity = native_binding_identity()
    resume.install_resume_hooks(gen)
    record = resident(old)
    state = GeneratorState(GeneratorCheckpoint("vpfp", gen._identity, report=gen._identity), gen._identity)
    store = CheckpointStore(tmp_path, identity={"test": "vpfp-json-runtime"}, generator=gen)
    store.save({"state": state, "record": record})
    manifest = json.loads(store.path.read_bytes())
    serialized = manifest["generator_runtime"]["identity"]["binding"]
    assert isinstance(serialized["sampling"]["model_config"]["full_config"]["time_domain"], list)
    newgen, new = binding(monkeypatch)
    newgen._identity = native_binding_identity()
    resume.install_resume_hooks(newgen)
    # A fresh store exercises BOTH the manifest JSON and GeneratorState JSON,
    # unlike a direct export/restore call with an in-memory Python dictionary.
    restored = CheckpointStore(tmp_path, identity=store.identity, generator=newgen).load()
    assert restored["state"].base == newgen._identity
    assert restored["state"].base is restored["state"].checkpoint.payload
    assert isinstance(restored["state"].base["sampling"]["model_config"]["full_config"]["time_domain"], tuple)
    assert restored["record"] is new._resume_records["root"]
    assert torch.equal(new._draws["root"]["draw"]["full"], old._draws["root"]["draw"]["full"])


@pytest.mark.parametrize("changed, path", [
    ("range", "binding.sampling.model_config.full_config.collision_nu_range[1]"),
    ("steps", "guidance_config.steps"),
    ("runtime", "runtime.frozen_decoder"),
    ("selector", "feedback_selection"),
    ("extra", "binding.extra"),
])
def test_json_identity_fix_does_not_accept_changed_assets_or_recipe(tmp_path, monkeypatch, changed, path):
    gen, old = binding(monkeypatch); gen._identity = native_binding_identity()
    payload = json.loads(json.dumps(resume.export_runtime_state(gen, tmp_path)))
    if changed == "range":
        payload["identity"]["binding"]["sampling"]["model_config"]["full_config"]["collision_nu_range"][1] = 31.
    elif changed == "steps": payload["identity"]["guidance_config"]["steps"] = 3000
    elif changed == "runtime": payload["identity"]["runtime"]["frozen_decoder"] = False
    elif changed == "selector": payload["identity"]["feedback_selection"] = "different-selection"
    else: payload["identity"]["binding"]["extra"] = "unapproved"
    with pytest.raises(ValueError) as error:
        resume.restore_runtime_state(gen, payload, tmp_path)
    assert path in str(error.value)
    assert not old.runtime._resident_cases


def test_json_generator_state_still_rejects_changed_tuple_values(tmp_path, monkeypatch):
    gen, _ = binding(monkeypatch); gen._identity = native_binding_identity()
    state = GeneratorState(GeneratorCheckpoint("vpfp", gen._identity), gen._identity)
    payload = json.loads(json.dumps(resume.export_generator_state(gen, state, tmp_path)))
    payload["base"]["sampling"]["model_config"]["full_config"]["temporal_dilation_cycle"][0] = 9
    with pytest.raises(ValueError, match=r"temporal_dilation_cycle\[0\]"):
        resume.restore_generator_state(gen, payload, tmp_path)


def test_identity_diagnostics_are_bounded_field_names_without_report_values():
    secret = "large-private-report-value" * 1000
    first = {f"field_{i:02d}": secret for i in range(100)}
    second = {key: "changed" for key in first}
    paths = resume._identity_difference_paths(first, second)
    assert len(paths) == 8 and paths[0] == "field_00" and paths[-1] == "field_07"
    with pytest.raises(ValueError) as error:
        resume._require_identity(first, second, "identity changed")
    assert secret not in str(error.value) and len(str(error.value)) < 250


@pytest.mark.parametrize("unsupported", [float("nan"), float("inf"), object()])
def test_identity_canonicalization_rejects_non_json_or_nonfinite_values(unsupported):
    with pytest.raises((TypeError, ValueError)):
        resume._json_identity({"value": unsupported})
