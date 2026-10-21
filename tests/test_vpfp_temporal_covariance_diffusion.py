"""Small native affine-noise asset tests; not a VPFP distribution audit."""

from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import (
    VPFPStructuredEDMConfig, load_vpfp_diffusion_sampler,
)
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import (
    VPFPValueEDMConfig, VPFPValueControlEDM, VPFPValueSobolevMetric,
    lift_value_controls, extract_value_controls, _value_split,
)
from pde_agent_verifier_solver.plugins.vpfp.temporal_covariance_diffusion import (
    CHECKPOINT_SCHEMA, VPFPTemporalCovarianceNormalizer as Normalizer,
    VPFPTemporalCovarianceEDM as Model, VPFPLowNoiseTemporalCovarianceEDM as LowModel,
    VPFPTemporalCovarianceBatchSampler,
)


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _config():
    return VPFPValueEDMConfig.from_full_config(VPFPStructuredEDMConfig(
        control_count=4, hermite_channels=2, rho_max_mode=1, kinetic_max_mode=1,
        inactive_dim=3, grid_size=3, width=8, depth=1,
        sigma_max=80., log_sigma_mean=-.6, log_sigma_std=1.8))


def _raw(config, count=8, dtype=torch.float64):
    values = torch.randn(count, config.latent_dimension, dtype=dtype,
        generator=torch.Generator().manual_seed(31))
    rho, kinetic = _value_split(values, config)
    kinetic[..., 0] += torch.linspace(3., 1., config.control_count, dtype=dtype)
    return lift_value_controls(values, config)


def _fit(config=None, dtype=torch.float64):
    config = _config() if config is None else config
    return Normalizer.fit(config, _raw(config, dtype=dtype), sample_ids=tuple(range(8)))


def _conditions(config, count=4):
    return torch.tensor([[sum(config.collision_nu_range)/2, sum(config.temperature_range)/2]]*count)


def _payload():
    config = _config()
    normalizer, model = _fit(config), Model(config)
    return {"schema": CHECKPOINT_SCHEMA, "config": config.to_dict(), "model": model.state_dict(),
        "normalizer": normalizer.to_payload(), "base_decoder_sha256": "a"*64,
        "corpus_manifest_sha256": "b"*64, "training": {"step": 1, "native_temporal_pilot": True}}


def _sha_json(value):
    return sha256(json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_native_encode_decode_full_abi_roundtrip_gradient_and_float64_factors(dtype):
    config = _config()
    normalizer = _fit(config, dtype).to("cpu")
    original = _raw(config, count=3, dtype=dtype)
    white = normalizer.encode(original)
    assert white.shape == (3, config.latent_dimension)
    restored = normalizer.decode(white)
    assert restored.shape == original.shape and restored.dtype == dtype
    torch.testing.assert_close(restored, original, rtol=3.e-6 if dtype==torch.float32 else 2.e-12,
                               atol=3.e-6 if dtype==torch.float32 else 2.e-12)
    variable = white.detach().requires_grad_(True)
    normalizer.decode(variable).square().sum().backward()
    assert torch.isfinite(variable.grad).all() and variable.grad.abs().sum() > 0
    assert all(b.dtype == torch.float64 for b in normalizer.buffers())
    assert not list(normalizer.parameters())
    assert normalizer.config.full_dimension == original.shape[1]
    assert normalizer.fit_metadata["native_sampler_asset"] is True
    assert normalizer.fit_metadata["physical_answer_fidelity_certified"] is False
    json.dumps(normalizer.report(), allow_nan=False)


def test_payload_roundtrip_train_provenance_and_only_dc_centering():
    config = _config()
    normalizer = _fit(config)
    restored = Normalizer.from_payload(normalizer.to_payload())
    raw = _raw(config)
    torch.testing.assert_close(restored.encode(raw), normalizer.encode(raw), rtol=0, atol=0)
    assert restored.fit_metadata == normalizer.fit_metadata
    mean_full = restored.decode(torch.zeros(1, config.latent_dimension, dtype=torch.float64))
    mean = restored.parent.encode(mean_full)
    rho, kinetic = _value_split(mean, config)
    assert torch.count_nonzero(rho) == torch.count_nonzero(kinetic[..., 1:]) == 0
    torch.testing.assert_close(kinetic[0, ..., 0], normalizer.temporal.kinetic_dc_mean)
    assert normalizer.parent.fit_metadata["sample_ids"] == normalizer.temporal.fit_metadata["sample_ids"]
    with pytest.raises(ValueError, match="train split"):
        Normalizer.fit(config, raw, sample_ids=tuple(range(8)), split="reference_a")
    with pytest.raises(ValueError, match="distinct"):
        Normalizer.fit(config, raw, sample_ids=(1,)*8)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_default_kinetic_only_whitening_passes_rho_through_bitwise(dtype):
    config = _config()
    raw = _raw(config, dtype=dtype)
    normalizer = Normalizer.fit(config, raw, sample_ids=tuple(range(8)), rho_policy="identity")
    parent = normalizer.parent.encode(raw)
    scoped = normalizer.encode(raw)
    assert torch.equal(scoped[:, :config.rho_dimension], parent[:, :config.rho_dimension])
    expected = normalizer.temporal.encode(parent)
    torch.testing.assert_close(scoped[:, config.rho_dimension:], expected[:, config.rho_dimension:], rtol=0, atol=0)
    arbitrary = torch.randn_like(scoped)
    restored, parent_restored = normalizer.decode(arbitrary), normalizer.parent.decode(arbitrary)
    assert torch.equal(restored[:, :config.full_config.rho_dimension], parent_restored[:, :config.full_config.rho_dimension])
    assert normalizer.fit_metadata["rho_policy"] == "identity"
    assert normalizer.fit_metadata["rho_temporal_statistics"] == "retained_diagnostic_only_not_applied"
    assert _fit(config).rho_policy == "identity"


def test_explicit_rho_covariance_preserves_original_affine_and_scope_is_bound():
    config = _config()
    raw = _raw(config)
    normalizer = Normalizer.fit(config, raw, sample_ids=tuple(range(8)), rho_policy="covariance")
    expected = normalizer.temporal.encode(normalizer.parent.encode(raw))
    torch.testing.assert_close(normalizer.encode(raw), expected, rtol=0, atol=0)
    torch.testing.assert_close(normalizer.decode(expected), raw)
    payload = normalizer.to_payload()
    restored = Normalizer.from_payload(payload)
    assert restored.rho_policy == "covariance"
    assert restored.fit_metadata["rho_temporal_statistics"] == "applied_covariance"
    identity = Normalizer.fit(config, raw, sample_ids=tuple(range(8)), rho_policy="identity")
    assert identity.fit_metadata["temporal_transform_sha256"] == normalizer.fit_metadata["temporal_transform_sha256"]
    assert identity.fit_metadata["normalization_scope_sha256"] != normalizer.fit_metadata["normalization_scope_sha256"]
    with pytest.raises(ValueError, match="rho_policy"):
        Normalizer.fit(config, raw, sample_ids=tuple(range(8)), rho_policy="adaptive")


@pytest.mark.parametrize("model_type", (Model, LowModel))
def test_same_architecture_white_loss_backward_and_old_metric_rejected(model_type):
    config = _config()
    normalizer, model = _fit(config, torch.float32), model_type(config)
    base = VPFPValueControlEDM(config)
    base.load_state_dict(model.state_dict())
    assert list(model.state_dict()) == list(base.state_dict())
    white = normalizer.encode(_raw(config, count=4, dtype=torch.float32))
    sigma = torch.tensor([.003, .03, .3, 3.])
    first, second = torch.Generator().manual_seed(13), torch.Generator().manual_seed(13)
    actual, metrics = model.loss(white, _conditions(config), generator=first, sigma=sigma)
    expected, base_metrics = base.loss(white, _conditions(config), generator=second, sigma=sigma)
    assert torch.equal(actual, expected)
    assert all(torch.equal(metrics[key], base_metrics[key]) for key in metrics)
    assert torch.equal(first.get_state(), second.get_state())
    actual.backward()
    assert torch.isfinite(model.output.weight.grad).all()
    invalid_metric = VPFPValueSobolevMetric(config, normalizer.parent)
    with pytest.raises(ValueError, match="do not apply"):
        model.loss(white, _conditions(config), residual_metric=invalid_metric)
    if model_type is LowModel:
        assert model.training_noise_report()["low_probability"] == .5
        loss, implicit_metrics = model.loss(white, _conditions(config), generator=torch.Generator().manual_seed(11))
        assert torch.isfinite(loss) and torch.all(implicit_metrics["sigma"] >= config.sigma_min)
        assert deepcopy(model).training_noise_report() == model.training_noise_report()


def test_native_low_noise_class_accepts_frozen_middle_low_band_without_explicit_rng_side_effect():
    config = _config()
    model = LowModel(config, low_min=.01, low_max=.3)
    base = Model(config)
    base.load_state_dict(model.state_dict())
    report = model.training_noise_report()
    assert report["low_min"] == .01 and report["low_max"] == .3
    clean = torch.randn(4, config.latent_dimension)
    first, second = torch.Generator().manual_seed(11), torch.Generator().manual_seed(11)
    left, _ = model.loss(clean, _conditions(config), sigma=torch.full((4,), .1), generator=first)
    right, _ = base.loss(clean, _conditions(config), sigma=torch.full((4,), .1), generator=second)
    assert torch.equal(left, right) and torch.equal(first.get_state(), second.get_state())
    implicit, _ = model.loss(clean, _conditions(config), generator=torch.Generator().manual_seed(12))
    implicit.backward()
    assert torch.isfinite(model.output.weight.grad).all()


def test_hash_loader_native_sampler_complete_output_and_batch_replay(tmp_path):
    payload = _payload()
    path = tmp_path / "temporal.pt"
    torch.save(payload, path)
    digest = sha256(path.read_bytes()).hexdigest()
    sampler = load_vpfp_diffusion_sampler(path, digest, steps=3)
    assert isinstance(sampler, VPFPTemporalCovarianceBatchSampler)
    assert type(sampler.model) is Model
    assert sampler.model.checkpoint_schema == CHECKPOINT_SCHEMA
    config = sampler.model.config
    assert sampler.identity["latent_dimension"] == config.full_dimension
    assert sampler.identity["random_dimension"] == config.latent_dimension
    assert sampler.identity["base_decoder_sha256"] == "a"*64
    assert sampler.identity["corpus_manifest_sha256"] == "b"*64
    assert sampler.identity["geometry_binding"]["temporal_transform_sha256"] == sampler.normalizer.fit_metadata["temporal_transform_sha256"]
    assert sampler.identity["geometry_binding"]["rho_policy"] == "identity"
    assert sampler.identity["geometry_binding"]["normalization_scope_sha256"] == sampler.normalizer.fit_metadata["normalization_scope_sha256"]
    assert "whitened" in sampler.identity["sampling"]["noise_space"]
    together, reverse = sampler.generate((21,22)), sampler.generate((22,21))
    assert together["latents"].shape == (2, config.full_dimension)
    assert together["diagnostics"]["pde_solves"] == 0
    np.testing.assert_array_equal(together["latents"], reverse["latents"][::-1])
    extract_value_controls(together["latents"], config)
    with_stale_metadata = load_vpfp_diffusion_sampler(path, "0"*64, steps=3)
    np.testing.assert_array_equal(together["latents"], with_stale_metadata.generate((21,22))["latents"])
    link = tmp_path / "linked.pt"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="non-symlink"):
        load_vpfp_diffusion_sampler(link, digest)


@pytest.mark.parametrize("corruption", ("extra", "model_tensor", "fourier", "training_nan", "source_hash",
    "geometry_hash", "parent_hash", "train_ids", "raw_center_relation", "parent_value_scale", "parent_source_ids",
    "geometry_config", "outer_config", "outer_metadata", "scope_policy", "scope_hash"))
def test_native_loader_rejects_tampering_even_with_new_outer_checkpoint_hash(tmp_path, corruption):
    payload = _payload()
    normal = payload["normalizer"]
    temporal = normal["temporal_transform"]
    if corruption == "extra":
        payload["extra"] = 1
    elif corruption == "model_tensor":
        payload["model"]["output.bias"][0] = float("nan")
    elif corruption == "fourier":
        payload["model"]["transform.rho_basis"][0,0,0] += .01
    elif corruption == "training_nan":
        payload["training"]["loss"] = float("nan")
    elif corruption == "source_hash":
        payload["corpus_manifest_sha256"] = "bad"
    elif corruption == "geometry_hash":
        normal["fit_metadata"]["temporal_transform_sha256"] = "0"*64
    elif corruption == "parent_hash":
        normal["fit_metadata"]["value_normalizer_sha256"] = "0"*64
    elif corruption in ("train_ids", "raw_center_relation", "geometry_config"):
        if corruption == "train_ids":
            temporal["fit_metadata"]["sample_ids"][0] = 99
        elif corruption == "raw_center_relation":
            temporal["kinetic_dc_mean"][0][0] += 1.
        else:
            temporal["config"]["full_config"]["sigma_max"] = 40.
        normal["fit_metadata"]["temporal_transform_sha256"] = _sha_json(temporal)
    elif corruption in ("parent_value_scale", "parent_source_ids"):
        parent = normal["value_normalizer"]
        if corruption == "parent_value_scale":
            parent["group_scales"][0] *= 2
        else:
            parent["fit_metadata"]["parent_normalizer_payload"]["fit_metadata"]["sample_ids"][0] = 99
            parent["fit_metadata"]["parent_normalizer_sha256"] = _sha_json(parent["fit_metadata"]["parent_normalizer_payload"])
        normal["fit_metadata"]["value_normalizer_sha256"] = _sha_json(parent)
    elif corruption == "outer_config":
        payload["config"]["full_config"]["sigma_max"] = 40.
    elif corruption == "outer_metadata":
        normal["fit_metadata"]["physical_answer_fidelity_certified"] = True
    elif corruption == "scope_policy":
        normal["fit_metadata"]["rho_policy"] = "covariance"
    elif corruption == "scope_hash":
        normal["fit_metadata"]["normalization_scope_sha256"] = "0"*64
    path = tmp_path / "corrupt.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_vpfp_diffusion_sampler(path, sha256(path.read_bytes()).hexdigest(), steps=2)


def test_frozen_normalizer_source_and_derived_buffers_cannot_silently_change():
    config = _config()
    normalizer = _fit(config)
    normalizer.temporal.square_root.add_(.01)
    with pytest.raises(ValueError, match="modified"):
        normalizer.decode(torch.zeros(1, config.latent_dimension, dtype=torch.float64))
    normalizer = _fit(config).float()
    with pytest.raises(ValueError, match="float64"):
        normalizer.encode(_raw(config))
    normalizer = _fit(config)
    normalizer.temporal.fit_metadata["center_kinetic_dc"] = False
    assert normalizer.to_payload()["temporal_transform"]["fit_metadata"]["center_kinetic_dc"] is False


@pytest.mark.parametrize("strength", (0., .5, 1.))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_covariance_strength_changes_only_full_rank_noise_factors_not_training_center(strength, dtype):
    config = _config()
    raw = _raw(config, dtype=dtype)
    full = Normalizer.fit(config, raw, sample_ids=range(8))
    mixed = Normalizer.fit(config, raw, sample_ids=range(8), covariance_strength=strength)
    assert mixed.parent.to_payload() == full.parent.to_payload()
    assert mixed.temporal.to_payload() == full.temporal.to_payload()
    assert torch.equal(mixed.temporal.kinetic_dc_mean, full.temporal.kinetic_dc_mean)
    white = mixed.encode(raw)
    torch.testing.assert_close(mixed.decode(white), raw,
        rtol=3.e-6 if dtype == torch.float32 else 2.e-12,
        atol=3.e-6 if dtype == torch.float32 else 2.e-12)
    assert torch.equal(white[:, :config.rho_dimension], mixed.parent.encode(raw)[:, :config.rho_dimension])
    root = full.temporal.square_root if strength == 1 else mixed.mixture_square_root
    identity = torch.eye(config.control_count, dtype=torch.float64)
    expected = (1-strength)*identity + strength*(full.temporal.square_root @ full.temporal.square_root.transpose(-1,-2))
    torch.testing.assert_close(root @ root.transpose(-1,-2), expected, rtol=2.e-12, atol=2.e-12)
    assert torch.linalg.eigvalsh(root @ root.transpose(-1,-2)).min() > 0
    if strength == 0:
        assert torch.equal(root, identity.expand_as(root))
        parent = mixed.parent.encode(raw)
        rho, kinetic = _value_split(parent.double(), config)
        kinetic = kinetic.clone()
        kinetic[..., 0] -= mixed.temporal.kinetic_dc_mean
        expected_white = torch.cat((rho.flatten(1), kinetic.flatten(1)), 1).to(dtype)
        assert torch.equal(white, expected_white)
    restored = Normalizer.from_payload(mixed.to_payload())
    assert restored.covariance_strength == strength
    torch.testing.assert_close(restored.encode(raw), white, rtol=0, atol=0)
    variable = white.detach().requires_grad_(True)
    mixed.decode(variable).square().sum().backward()
    assert torch.isfinite(variable.grad).all() and variable.grad.abs().sum() > 0


def test_default_strength_retains_legacy_payload_and_identity_shape():
    cfg = _config()
    raw = _raw(cfg)
    default = Normalizer.fit(cfg, raw, sample_ids=range(8))
    explicit = Normalizer.fit(cfg, raw, sample_ids=range(8), covariance_strength=1.)
    assert default.to_payload() == explicit.to_payload()
    assert "covariance_strength" not in default.fit_metadata
    assert "mixture_square_root" not in dict(default.named_buffers())
    sampler = VPFPTemporalCovarianceBatchSampler(Model(cfg), default,
        base_decoder_sha256="a"*64, checkpoint_sha256="b"*64, steps=2)
    assert "covariance_strength" not in sampler.identity["geometry_binding"]
    assert sampler.identity["geometry_binding"]["kinetic_policy"] == "covariance"


@pytest.mark.parametrize("strength", (0., .5))
def test_shrunk_native_checkpoint_reloads_exact_geometry_and_replays(tmp_path, strength):
    payload = _payload()
    cfg = VPFPValueEDMConfig.from_dict(payload["config"])
    normalizer = Normalizer.fit(cfg, _raw(cfg), sample_ids=range(8), covariance_strength=strength)
    payload["normalizer"] = normalizer.to_payload()
    path = tmp_path / "mixture.pt"
    torch.save(payload, path)
    sampler = load_vpfp_diffusion_sampler(path, sha256(path.read_bytes()).hexdigest(), steps=2)
    assert sampler.normalizer.covariance_strength == strength
    assert sampler.identity["geometry_binding"]["covariance_strength"] == strength
    assert sampler.identity["geometry_binding"]["kinetic_policy"] == "identity_covariance_mixture"
    assert sampler.identity["sampling"]["covariance_strength"] == strength
    torch.testing.assert_close(sampler.normalizer.encode(_raw(cfg)), normalizer.encode(_raw(cfg)), rtol=0, atol=0)
    forward, reverse = sampler.generate((41,42)), sampler.generate((42,41))
    np.testing.assert_array_equal(forward["latents"], reverse["latents"][::-1])
    assert forward["latents"].shape == (2, cfg.full_dimension)
    assert forward["diagnostics"]["pde_solves"] == 0
    corrupted = deepcopy(payload)
    corrupted["normalizer"]["fit_metadata"]["covariance_strength"] = .25
    torch.save(corrupted, path)
    with pytest.raises(ValueError, match="metadata/bindings"):
        load_vpfp_diffusion_sampler(path, sha256(path.read_bytes()).hexdigest(), steps=2)


@pytest.mark.parametrize("strength", (-.1, 1.1, float("nan"), float("inf"), True, "0.5"))
def test_invalid_covariance_strength_is_rejected(strength):
    full = _fit()
    with pytest.raises(ValueError, match="covariance_strength"):
        Normalizer(full.parent, full.temporal, covariance_strength=strength)


def test_shrinkage_rho_scope_and_mutable_factor_are_rejected():
    full = _fit()
    with pytest.raises(ValueError, match="identity rho_policy"):
        Normalizer(full.parent, full.temporal, covariance_strength=.5, rho_policy="covariance")
    mixed = Normalizer(full.parent, full.temporal, covariance_strength=.5)
    mixed.mixture_square_root.add_(.01)
    with pytest.raises(ValueError, match="modified"):
        mixed.decode(torch.zeros(1, mixed.config.latent_dimension, dtype=torch.float64))
    mixed = Normalizer(full.parent, full.temporal, covariance_strength=.5)
    mixed.covariance_strength = 0.
    with pytest.raises(ValueError, match="covariance_strength changed"):
        mixed.to_payload()


@pytest.mark.parametrize("recipe", ("trace_isotropic", "unit_trace_shape"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_factorial_recipes_separate_floored_energy_and_shape_preserve_mean_rho_rank(recipe, dtype):
    cfg = _config()
    raw = _raw(cfg, dtype=dtype)
    full = Normalizer.fit(cfg, raw, sample_ids=range(8), relative_floor=1.)
    norm = Normalizer(full.parent, full.temporal, covariance_recipe=recipe)
    assert norm.parent.to_payload() == full.parent.to_payload()
    assert norm.temporal.to_payload() == full.temporal.to_payload()
    tau = full.temporal.regularized_eigenvalues.mean(-1)
    # Deliberately large existing floor distinguishes trace(C_reg) from trace(C_raw).
    assert not torch.allclose(tau, full.temporal.eigenvalues.mean(-1))
    root = norm.mixture_square_root
    covariance = root @ root.transpose(-1, -2)
    expected = (tau[:, None, None]*torch.eye(cfg.control_count, dtype=torch.float64)
        if recipe == "trace_isotropic" else
        (full.temporal.square_root @ full.temporal.square_root.transpose(-1, -2))/tau[:, None, None])
    torch.testing.assert_close(covariance, expected, rtol=3.e-12, atol=3.e-12)
    assert torch.linalg.eigvalsh(covariance).min() > 0
    white = norm.encode(raw)
    assert torch.equal(white[:, :cfg.rho_dimension], full.parent.encode(raw)[:, :cfg.rho_dimension])
    torch.testing.assert_close(norm.decode(white), raw, rtol=4.e-6, atol=4.e-6)
    # All recipes decode zero to exactly the same train-only DC center.
    zero = torch.zeros(2, cfg.latent_dimension, dtype=dtype)
    torch.testing.assert_close(norm.decode(zero), full.decode(zero), rtol=0, atol=0)
    variable = white.detach().requires_grad_(True)
    norm.decode(variable).square().sum().backward()
    assert torch.isfinite(variable.grad).all() and variable.grad.abs().sum() > 0
    assert norm.fit_metadata["kinetic_tau"] == tau[1:].tolist()
    assert norm.fit_metadata["post_recipe_floor_or_clipping"] is False
    restored = Normalizer.from_payload(norm.to_payload())
    torch.testing.assert_close(restored.encode(raw), white, rtol=0, atol=0)


@pytest.mark.parametrize("strength", (0., .5, 1.))
def test_explicit_legacy_recipe_retains_payload_bytes_and_native_identity(strength):
    base = _fit()
    old = Normalizer(base.parent, base.temporal, covariance_strength=strength)
    explicit = Normalizer(base.parent, base.temporal, covariance_strength=strength, covariance_recipe="mixture")
    assert _sha_json(old.to_payload()) == _sha_json(explicit.to_payload())
    assert "covariance_recipe" not in old.fit_metadata
    make = lambda normal: VPFPTemporalCovarianceBatchSampler(Model(base.config), normal,
        base_decoder_sha256="a"*64, checkpoint_sha256="b"*64, steps=2).identity
    assert make(old) == make(explicit)


@pytest.mark.parametrize("recipe", ("trace_isotropic", "unit_trace_shape"))
def test_factorial_native_reload_identity_replay_and_binding_tamper(tmp_path, recipe):
    payload = _payload()
    base = Normalizer.from_payload(payload["normalizer"])
    payload["normalizer"] = Normalizer(base.parent, base.temporal, covariance_recipe=recipe).to_payload()
    path = tmp_path / "factorial.pt"
    torch.save(payload, path)
    sampler = load_vpfp_diffusion_sampler(path, sha256(path.read_bytes()).hexdigest(), steps=2)
    assert sampler.identity["geometry_binding"]["covariance_recipe"] == recipe
    assert sampler.identity["sampling"]["covariance_recipe"] == recipe
    assert sampler.identity["geometry_binding"]["post_recipe_floor_or_clipping"] is False
    forward, reverse = sampler.generate((41, 42)), sampler.generate((42, 41))
    np.testing.assert_array_equal(forward["latents"], reverse["latents"][::-1])
    assert np.isfinite(forward["latents"]).all()
    payload["normalizer"]["fit_metadata"]["kinetic_tau"][0] *= 1.01
    torch.save(payload, path)
    with pytest.raises(ValueError, match="metadata/bindings"):
        load_vpfp_diffusion_sampler(path, sha256(path.read_bytes()).hexdigest(), steps=2)
    sampler.normalizer.covariance_recipe = "mixture"
    with pytest.raises(ValueError, match="covariance_recipe changed"):
        sampler.normalizer.to_payload()


@pytest.mark.parametrize("kwargs", ({"covariance_recipe": "unknown"},
    {"covariance_recipe": "trace_isotropic", "covariance_strength": .5},
    {"covariance_recipe": "unit_trace_shape", "rho_policy": "covariance"}))
def test_factorial_recipe_rejects_ambiguous_strength_or_rho_scope(kwargs):
    base = _fit()
    with pytest.raises(ValueError):
        Normalizer(base.parent, base.temporal, **kwargs)


@pytest.mark.parametrize("recipe,strength", [("mixture",0.),("mixture",.5),("mixture",1.),
    ("trace_isotropic",1.),("unit_trace_shape",1.)])
def test_device_copy_preserves_frozen_factor_versions_and_still_rejects_mutation(recipe, strength):
    base = _fit()
    norm = Normalizer(base.parent, base.temporal, covariance_recipe=recipe, covariance_strength=strength)
    raw = _raw(base.config)
    expected = norm.encode(raw)
    payload = _sha_json(norm.to_payload())
    # nn.Module.to(new_device) replaces tensors with fresh version-zero copies.
    # CPU clones reproduce that version behavior without requiring a GPU.
    norm._apply(lambda tensor: tensor.clone())
    torch.testing.assert_close(norm.encode(raw), expected, rtol=0, atol=0)
    torch.testing.assert_close(norm.decode(expected), raw, rtol=2.e-12, atol=2.e-12)
    assert _sha_json(norm.to_payload()) == payload
    next(norm.buffers()).add_(.01)
    with pytest.raises(ValueError, match="modified"):
        norm.encode(raw)
