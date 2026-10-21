"""Tiny portability mechanics only; real paired replay is a separate gate."""
from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pytest
import pickle
import torch

from pde_agent_verifier_solver.plugins.vpfp import initial_conditioned_diffusion as original
from pde_agent_verifier_solver.plugins.vpfp import portable_initial_conditioned as portable
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import (
    extract_value_controls, lift_value_controls, _value_split)
from tests.test_vpfp_initial_conditioned_diffusion import data, checkpoint_payload


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def bundle_from(normalizer, *, old=None, checkpoint_sha="a"*64, model_sha="b"*64):
    candidate = normalizer.to_payload()
    return portable.export_frozen_normalizer(candidate if old is None else old, candidate,
        normalizer.state_dict(), source_checkpoint_sha256=checkpoint_sha, model_state_sha256=model_sha)


def load(bundle, device="cpu"):
    return portable.load_frozen_normalizer(bundle, expected_sha256=bundle["sha256"], device=device)


def forbid_eigh(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("portable load must not rerun a spectral solve")
    monkeypatch.setattr(torch.linalg, "eigh", forbidden)
    monkeypatch.setattr(torch.linalg, "eigvalsh", forbidden)
    monkeypatch.setattr(original.VPFPInitialConditionedNormalizer, "from_payload", forbidden)


def assert_bits(left, right):
    assert left.dtype == right.dtype and left.shape == right.shape
    assert torch.equal(left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                       right.detach().cpu().contiguous().reshape(-1).view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("count", [5, 33])
def test_no_eigh_frozen_reload_preserves_every_buffer_and_arithmetic(monkeypatch, dtype, count):
    config, raw, source = data(dtype, count)
    # Exercise the actual T33/DC ~9 regime without making observed residuals
    # a new tolerance. Equality is against the fixed candidate's arithmetic.
    values = extract_value_controls(raw, config.value_config).clone()
    _value_split(values, config.value_config)[1][:, 0, :, 0] += 9.
    raw = lift_value_controls(values, config.value_config)
    original_encoded = source.encode(raw)
    expected = source.decode(original_encoded)
    rho, kinetic = _value_split(values, config.value_config)
    initial = torch.cat((rho[:, 0], kinetic[:, :, 0].flatten(1)), dim=1)
    expected_initial = source.encode_initial(initial)
    bundle = bundle_from(source)
    forbid_eigh(monkeypatch)
    restored = load(bundle)
    assert len(dict(restored.named_buffers())) == 15
    assert not list(restored.parameters())
    for name, value in source.state_dict().items():
        assert_bits(restored.state_dict()[name], value)
    actual_encoded = restored.encode(raw)
    for actual, wanted in zip(actual_encoded, original_encoded):
        assert_bits(actual, wanted)
    assert_bits(restored.decode(actual_encoded), expected)
    assert_bits(restored.encode_initial(initial), expected_initial)
    assert restored.to_payload()["schema"] == portable.SCHEMA
    json.dumps(restored.to_payload(), allow_nan=False)
    assert restored.to_frozen_payload()["sha256"] == bundle["sha256"]
    assert not restored.report()["scientific_admission"]
    with pytest.raises(ValueError, match="compact portable"):
        portable.PortableInitialConditionedNormalizer.from_payload(restored.to_payload())


def test_context_future_isolation_original_lift_and_gradients():
    config, raw, source = data(torch.float32)
    normalizer = load(bundle_from(source))
    context, future = normalizer.encode(raw[:2])
    context, future = context.detach().requires_grad_(), future.detach().requires_grad_()
    full = normalizer.decode((context, future))
    changed = normalizer.decode((context, future + 1.))
    rv, kv = _value_split(extract_value_controls(full, config.value_config), config.value_config)
    rc, kc = _value_split(extract_value_controls(changed, config.value_config), config.value_config)
    assert_bits(rv[:, 0], rc[:, 0])
    assert_bits(kv[:, :, 0], kc[:, :, 0])
    assert_bits(full, lift_value_controls(extract_value_controls(full, config.value_config), config.value_config))
    assert torch.count_nonzero(full[:, config.full_config.rho_dimension:
        config.full_config.rho_dimension+config.full_config.inactive_dim]) == 0
    full.square().mean().backward()
    assert context.grad is not None and torch.isfinite(context.grad).all()
    assert future.grad is not None and torch.isfinite(future.grad).all()
    assert context.grad.abs().sum() > 0 and future.grad.abs().sum() > 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_copy_deepcopy_serialization_and_dtype_guard(tmp_path, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA allocation required")
    _, raw, source = data(torch.float32)
    bundle = bundle_from(source)
    source.to(device)
    expected = source.decode(source.encode(raw.to(device)))
    forbid_eigh(monkeypatch)
    normalizer = load(bundle, device)
    copied = deepcopy(normalizer)
    for item in (normalizer, copied):
        item.assert_frozen(full=True)
        assert_bits(item.decode(item.encode(raw.to(device))), expected)
    copied.to("cpu").to(device)
    assert_bits(copied.decode(copied.encode(raw.to(device))), expected)
    path = tmp_path/"normalizer.pt"
    torch.save(copied.to_frozen_payload(), path)
    reloaded = load(torch.load(path, map_location="cpu", weights_only=True), device)
    reloaded.assert_frozen(full=True)
    assert_bits(reloaded.decode(reloaded.encode(raw.to(device))), expected)
    with pytest.raises(ValueError, match="FP64|frozen"):
        normalizer.float()


@pytest.mark.parametrize("name", ["square_root", "temporal.temporal.square_root", "temporal.parent.group_scales"])
def test_version_zero_replacement_is_rejected_before_arithmetic(name):
    _, raw, source = data(torch.float32)
    normalizer = load(bundle_from(source))
    owner = normalizer
    parts = name.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    replacement = getattr(owner, parts[-1]).clone()
    assert replacement._version == 0
    setattr(owner, parts[-1], replacement)
    with pytest.raises(ValueError, match="identity/version"):
        normalizer.encode(raw[:1])


def test_inplace_metadata_and_unsafe_storage_writes_do_not_export():
    _, raw, source = data(torch.float32)
    normalizer = load(bundle_from(source))
    normalizer.square_root.add_(1.)
    with pytest.raises(ValueError, match="identity/version"):
        normalizer.decode(normalizer.encode(raw[:1]))
    normalizer = load(bundle_from(source))
    expected = normalizer.encode(raw[:1])
    normalizer.temporal.parent.fit_metadata["sample_ids"][0] = 900
    torch.testing.assert_close(normalizer.encode(raw[:1]), expected)
    normalizer = load(bundle_from(source))
    normalizer.square_root.data.add_(1.)
    assert normalizer.to_frozen_payload()
    assert normalizer.to_payload()


@pytest.mark.parametrize("mutation", [
    lambda b: b.update(schema="old"),
    lambda b: b.update(extra=True),
    lambda b: b["candidate_normalizer"]["fit_metadata"].update(context_slopes_or_future_values=True),
    lambda b: b["buffers"].pop("regression"),
    lambda b: b["buffers"].update(extra=torch.ones(1, dtype=torch.float64)),
    lambda b: b["buffers"].update(regression=b["buffers"]["regression"].float()),
    lambda b: b["buffers"].update(regression=b["buffers"]["regression"][:, :1]),
    lambda b: b["buffers"]["square_root"].fill_(float("nan")),
    lambda b: b["buffers"]["square_root"].add_(.01),
])
def test_external_binding_rejects_tampering(mutation):
    _, _, source = data()
    bundle = bundle_from(source)
    mutation(bundle)
    with pytest.raises((ValueError, TypeError)):
        load(bundle)


def test_only_schur_metadata_difference_allowed_and_wrong_inverse_not_absorbed():
    _, _, source = data()
    old, candidate = source.to_payload(), source.to_payload()
    old["fit_metadata"]["schur_operator_sha256"] = "e"*64
    bundle = bundle_from(source, old=old)
    assert bundle["identity"]["old_schur_operator_sha256"] == "e"*64
    assert bundle["identity"]["candidate_schur_operator_sha256"] != "e"*64
    assert not bundle["identity"]["old_sampler_equivalence_certified"]
    changed = deepcopy(candidate)
    changed["fit_metadata"]["sample_ids"][0] = 41
    with pytest.raises(ValueError, match="only the explicitly"):
        portable.export_frozen_normalizer(old, changed, source.state_dict(),
            source_checkpoint_sha256="a"*64, model_state_sha256="b"*64)
    state = {k: v.clone() for k, v in source.state_dict().items()}
    state["inverse_square_root"] *= 1.00001
    changed = deepcopy(candidate)
    changed["fit_metadata"]["schur_operator_sha256"] = original._json_sha({k: state[k].tolist() for k in portable._SCHUR})
    with pytest.raises(ValueError, match="Schur inverse"):
        portable.export_frozen_normalizer(candidate, changed, state,
            source_checkpoint_sha256="a"*64, model_state_sha256="b"*64)


@pytest.mark.parametrize("key,value", [("rho_policy", "covariance"), ("covariance_strength", .5), ("covariance_recipe", "unit_trace_shape")])
def test_scope_is_not_silently_generalized(key, value):
    _, _, source = data()
    candidate = source.to_payload()
    candidate["temporal_normalizer"]["fit_metadata"][key] = value
    with pytest.raises(ValueError, match="only supports"):
        portable.export_frozen_normalizer(candidate, candidate, source.state_dict(),
            source_checkpoint_sha256="a"*64, model_state_sha256="b"*64)


def write_assets(tmp_path, mutate=None):
    config, raw, normalizer, model, checkpoint = checkpoint_payload()
    if mutate is not None:
        mutate(checkpoint)
    path = tmp_path/"original.pt"
    torch.save(checkpoint, path)
    checkpoint_sha = sha256(path.read_bytes()).hexdigest()
    bundle = bundle_from(normalizer, checkpoint_sha=checkpoint_sha,
        model_sha=portable.tensor_state_sha256(checkpoint["model"]))
    target = tmp_path/"portable.pt"
    torch.save(bundle, target)
    kwargs = dict(checkpoint_sha256=checkpoint_sha, portable_bundle_path=target,
        portable_bundle_sha256=sha256(target.read_bytes()).hexdigest(), device="cpu", sample_steps=3, sample_churn=0.)
    return config, raw, model, path, kwargs


def test_explicit_sampler_loads_actual_sha_files_preserves_weights_and_seed_stream(tmp_path, monkeypatch):
    config, raw, model, path, kwargs = write_assets(tmp_path)
    forbid_eigh(monkeypatch)
    sampler = portable.load_portable_initial_conditioned_sampler(path, **kwargs)
    assert isinstance(sampler, original.VPFPInitialConditionedBatchSampler)
    assert sampler.identity["checkpoint_sha256"] == kwargs["checkpoint_sha256"]
    assert sampler.identity["checkpoint_schema"] == portable.SAMPLER_SCHEMA
    assert sampler.identity["latent_dimension"] == config.full_dimension
    for name, value in model.state_dict().items():
        assert_bits(value, sampler.model.state_dict()[name])
    calls = []
    def prior_sample(seeds, conditions=None):
        calls.append(list(seeds))
        return {"initial_controls": torch.ones(len(seeds), config.context_dimension),
                "equation_conditions": np.asarray(conditions, dtype=np.float32)}
    monkeypatch.setattr(sampler.prior, "sample", prior_sample)
    context = sampler.normalizer.encode_initial(torch.ones(1, config.context_dimension))
    theta, seed = [[20., 1.]], 32
    stream = lambda tag: int.from_bytes(sha256(f"vpfp-initial-conditioned-{tag}:{seed}".encode()).digest()[:8], "little") % 2**63
    expected_future = model.sample(context, theta, [stream("future")], steps=3, churn=0.)
    expected = sampler.normalizer.decode(context, expected_future)
    output = sampler.sample_conditions(theta, [seed])
    assert calls == [[stream("prior")]]
    assert_bits(output, expected)
    json.dumps(sampler.identity, allow_nan=False)
    json.dumps(sampler.normalizer.to_payload(), allow_nan=False)


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p["model"]["transform.rho_basis"].add_(.1), "immutable"),
    (lambda p: p["training"].update(train_case_ids=["wrong"]), "case IDs"),
    (lambda p: p["training"].update(normalizer_sha256="d"*64), "training normalizer"),
    (lambda p: p.update(corpus_manifest_sha256="d"*64), "corpus identity"),
    (lambda p: p["model"].pop("initial_input.weight"), "tensor keys"),
])
def test_sampler_rejects_self_consistent_bad_original_metadata_or_geometry(tmp_path, mutation, match):
    _, _, _, path, kwargs = write_assets(tmp_path, mutation)
    if match == "training normalizer":
        assert portable.load_portable_initial_conditioned_sampler(path, **kwargs)
    else:
        with pytest.raises(ValueError, match=match):
            portable.load_portable_initial_conditioned_sampler(path, **kwargs)


def test_invalid_model_still_fails_deserialization_without_checksum(tmp_path, monkeypatch):
    path = tmp_path/"not-a-checkpoint.bin"
    path.write_bytes(b"not a torch object")
    with pytest.raises(pickle.UnpicklingError):
        portable.load_portable_initial_conditioned_sampler(path, checkpoint_sha256="a"*64,
            portable_bundle_path=path, portable_bundle_sha256="b"*64)


def test_runtime_checks_do_not_hash_full_tensor_bytes_per_encode(monkeypatch):
    _, raw, source = data(torch.float32)
    normalizer = load(bundle_from(source))
    monkeypatch.setattr(portable, "tensor_state_manifest", lambda *a: pytest.fail("no full byte copy in runtime encode/decode"))
    normalizer.decode(normalizer.encode(raw[:1]))


def test_explicit_sampler_is_independent_of_ambient_default_dtype(tmp_path):
    _, _, model, path, kwargs = write_assets(tmp_path)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        sampler = portable.load_portable_initial_conditioned_sampler(path, **kwargs)
        assert torch.get_default_dtype() == torch.float64
        assert portable.tensor_state_sha256(sampler.model.state_dict()) == portable.tensor_state_sha256(model.state_dict())
        with pytest.raises(ValueError):
            portable.load_portable_initial_conditioned_sampler(path, **dict(kwargs, sample_steps=1))
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(previous)


def test_portable_historical_checksum_label_is_not_a_loading_gate():
    _, _, source = data()
    bundle = bundle_from(source)
    bundle["source_checkpoint_sha256"] = "c" * 64
    assert load(bundle)
