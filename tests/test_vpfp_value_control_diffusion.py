"""Bounded synthetic math/API tests; not VPFP physical qualification."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import math

import numpy as np
import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import (
    VPFPStructuredEDMConfig, VPFPStructuredEDM, VPFPCoefficientNormalizer, _split,
    load_vpfp_diffusion_sampler,
)
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import (
    CHECKPOINT_SCHEMA, VPFPValueEDMConfig, VPFPValueControlNormalizer,
    VPFPValueControlEDM, VPFPValueFourierTransform, VPFPValueSobolevMetric,
    VPFPValueDiffusionBatchSampler, value_control_slopes,
    extract_value_controls, lift_value_controls,
)


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _config(**changes):
    return VPFPValueEDMConfig.from_full_config(replace(VPFPStructuredEDMConfig(
        control_count=3, hermite_channels=2, rho_max_mode=1, kinetic_max_mode=1,
        inactive_dim=3, grid_size=3, width=8, depth=1), **changes))


def _values(config, count=4, dtype=torch.float64):
    return torch.randn((count, config.latent_dimension), dtype=dtype,
        generator=torch.Generator().manual_seed(91))


def _normalizer(config, dtype=torch.float64):
    return VPFPValueControlNormalizer.fit(config, lift_value_controls(_values(config, dtype=dtype), config),
        sample_ids=(1, 2, 3, 4))


def _conditions(config, count):
    return torch.tensor([[sum(config.collision_nu_range)/2, sum(config.temperature_range)/2]]*count)


def _payload(config):
    model, normalizer = VPFPValueControlEDM(config), _normalizer(config)
    return {"schema": CHECKPOINT_SCHEMA, "config": config.to_dict(), "model": model.state_dict(),
        "normalizer": normalizer.to_payload(), "base_decoder_sha256": "a"*64,
        "corpus_manifest_sha256": "b"*64, "training": {"step": 1, "pilot": True}}


def _save(tmp_path, payload):
    path = tmp_path / "value.pt"
    torch.save(payload, path)
    return path, sha256(path.read_bytes()).hexdigest()


def test_config_production_dimension_and_roundtrip():
    config = VPFPValueEDMConfig.from_full_config(VPFPStructuredEDMConfig())
    assert config.full_dimension == 652672
    assert config.latent_dimension == 326304
    assert config.inactive_dim == 0
    assert config.field_channels == 121
    assert config.rho_dimension == 5544
    assert config.slope_interval == 1/32
    assert VPFPValueEDMConfig.from_dict(config.to_dict()) == deepcopy(config)
    with pytest.raises(ValueError):
        VPFPValueEDMConfig.from_dict({**config.to_dict(), "surprise": 1})
    with pytest.raises(ValueError):
        VPFPValueEDMConfig.from_dict({**config.to_dict(), "schema": "unknown"})


@pytest.mark.parametrize("count", (2, 3, 33))
def test_slope_lift_constant_linear_cubic_and_time_units(count):
    config = _config(control_count=count, time_domain=(2., 5.))
    tau = torch.linspace(0., 1., count, dtype=torch.float64)
    torch.testing.assert_close(value_control_slopes(torch.ones(1, count, 2, dtype=torch.float64), config),
                               torch.zeros(1, count, 2, dtype=torch.float64))
    linear = tau[None, :, None].expand(1, count, 2)
    torch.testing.assert_close(value_control_slopes(linear, config), torch.ones_like(linear))
    cubic = tau**3
    actual = value_control_slopes(cubic[None, :, None], config)[0, :, 0]
    expected = np.gradient(cubic.numpy(), tau.numpy(), edge_order=1)
    torch.testing.assert_close(actual, torch.tensor(expected), atol=2.e-14, rtol=2.e-14)
    # The lift reproduces the reference FD rule, not analytic cubic slopes.
    assert actual[0] > 0
    assert actual[-1] < 3.
    torch.testing.assert_close(value_control_slopes(linear, config),
        value_control_slopes(linear, _config(control_count=count, time_domain=(0., 1.))))


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_lift_extract_bijection_on_encoder_subspace_and_gradients(dtype):
    config = _config()
    reduced = _values(config, dtype=dtype).requires_grad_(True)
    full = lift_value_controls(reduced, config)
    assert full.shape == (4, config.full_dimension)
    torch.testing.assert_close(extract_value_controls(full, config), reduced, rtol=0, atol=0)
    rho, kinetic = _split(full, config.full_config)
    torch.testing.assert_close(rho[:, 1], value_control_slopes(rho[:, 0], config), rtol=0, atol=0)
    torch.testing.assert_close(kinetic[:, :, 1], value_control_slopes(kinetic[:, :, 0], config), rtol=0, atol=0)
    assert torch.count_nonzero(full[:, config.full_config.rho_dimension:
        config.full_config.rho_dimension + config.full_config.inactive_dim]) == 0
    extract_value_controls(full, config).square().sum().backward()
    torch.testing.assert_close(reduced.grad, 2*reduced.detach())
    bad = full.detach().clone()
    bad[0, config.control_count*config.rho_modes] += .01
    with pytest.raises(ValueError, match="lift subspace"):
        extract_value_controls(bad, config)


def test_quantized_fp64_source_slopes_pass_without_loosening_nonzero_constraint():
    config = _config(control_count=33)
    values = _values(config)
    values[:, :config.rho_dimension] += 3.
    full = lift_value_controls(values, config).float()
    extracted = extract_value_controls(full, config)
    torch.testing.assert_close(extracted, values.float(), rtol=0, atol=0)
    # Entirely zero neighbors may not conceal an arbitrary tiny slope.
    full.zero_()
    full[0, config.rho_dimension] = 1.e-12
    with pytest.raises(ValueError, match="lift subspace"):
        extract_value_controls(full, config)


def test_normalizer_train_only_roundtrip_floor_and_subspace_validation():
    config = _config()
    values = _values(config)
    values[:, config.rho_dimension + config.control_count*config.kinetic_modes:] *= 1.e-14
    full = lift_value_controls(values, config)
    normalizer = VPFPValueControlNormalizer.fit(config, full, sample_ids=(1, 2, 3, 4))
    torch.testing.assert_close(normalizer.decode(normalizer.encode(full)), full)
    restored = VPFPValueControlNormalizer.from_payload(normalizer.to_payload())
    torch.testing.assert_close(restored.encode(full), normalizer.encode(full))
    assert normalizer.group_scales[2] >= normalizer.fit_metadata["kinetic_floor"]
    assert normalizer.fit_metadata["slope_subspace_verified"] is True
    with pytest.raises(ValueError, match="train split"):
        VPFPValueControlNormalizer.fit(config, full, sample_ids=(1, 2, 3, 4), split="validation")
    with pytest.raises(ValueError, match="distinct"):
        VPFPValueControlNormalizer.fit(config, full, sample_ids=(1, 1, 3, 4))
    full[0, config.rho_dimension] += 1.
    with pytest.raises(ValueError, match="lift subspace"):
        normalizer.encode(full)


@pytest.mark.parametrize("count", (3, 33))
def test_value_scales_and_encoded_coordinates_match_full_normalizer_bitwise(count):
    config = _config(control_count=count)
    values = _values(config, dtype=torch.float32)
    # Exercise the floor with a nearly absent kinetic channel and rapid
    # changes in the other one, making independent value-only floor fitting
    # a genuinely different operation.
    values[:, config.rho_dimension + config.control_count*config.kinetic_modes:] *= 1.e-12
    full = lift_value_controls(values, config)
    reduced = VPFPValueControlNormalizer.fit(config, full, sample_ids=(1, 2, 3, 4))
    parent = VPFPCoefficientNormalizer.fit(config.full_config, full, sample_ids=(1, 2, 3, 4))
    assert torch.equal(reduced.group_scales, parent.group_scales[:, 0])
    encoded_parent = parent.encode(full)
    rho, kinetic = _split(encoded_parent, config.full_config)
    parent_values = torch.cat((rho[:, 0].flatten(1), kinetic[:, :, 0].flatten(1)), dim=1)
    assert torch.equal(reduced.encode(full), parent_values)
    metadata = reduced.fit_metadata
    assert metadata["floor_policy"] == "inherited_full_representation_for_paired_control"
    assert metadata["parent_normalizer_payload"] == parent.to_payload()
    expected_sha = sha256(json.dumps(parent.to_payload(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert metadata["parent_normalizer_sha256"] == expected_sha


@pytest.mark.parametrize("dtype,tolerance", ((torch.float32, 3.e-6), (torch.float64, 3.e-14)))
def test_value_fourier_view_exact_roundtrip_and_no_slope_channels(dtype, tolerance):
    config = _config()
    transform = VPFPValueFourierTransform(config)
    values = _values(config, dtype=dtype).requires_grad_(True)
    field = transform.to_field(values)
    assert field.shape == (4, 3, 3, 3, 3)
    restored = transform.to_coefficients(field)
    torch.testing.assert_close(restored, values, rtol=tolerance, atol=tolerance)
    restored.square().sum().backward()
    torch.testing.assert_close(values.grad, 2*values.detach(), rtol=2*tolerance, atol=2*tolerance)


def test_true_reduced_edm_noise_loss_heun_and_common_trunk_keys():
    config = _config()
    model = VPFPValueControlEDM(config)
    assert model.input.in_channels == model.output.out_channels == config.field_channels
    full_model = VPFPStructuredEDM(config.full_config)
    assert model.state_dict().keys() == full_model.state_dict().keys()
    for name, tensor in model.state_dict().items():
        if name not in ("input.weight", "output.weight", "output.bias"):
            assert tensor.shape == full_model.state_dict()[name].shape
    clean = _values(config, dtype=torch.float32)
    sigma = torch.tensor([.003, .03, .3, 3.])
    noise = torch.randn_like(clean)
    expected = (clean + sigma[:, None]*noise) / (1+sigma[:, None].square())
    torch.testing.assert_close(model.denoise(clean + sigma[:, None]*noise, sigma, _conditions(config, 4)), expected)
    loss, metrics = model.loss(clean, _conditions(config, 4), sigma=sigma, noise=noise)
    torch.testing.assert_close(loss, metrics["zero_mean_isotropic_baseline_per_case"].mean())
    loss.backward()
    assert torch.isfinite(model.output.weight.grad).all()
    observed_shapes = []
    original_denoise = model.denoise
    def capture(values, *args):
        observed_shapes.append(tuple(values.shape))
        return original_denoise(values, *args)
    model.denoise = capture
    samples = model.sample(_conditions(config, 2), (10, 11), steps=3)
    assert samples.shape == (2, config.latent_dimension)
    assert observed_shapes and all(shape == (1, config.latent_dimension) for shape in observed_shapes)
    torch.testing.assert_close(samples, model.sample(_conditions(config, 2), (11, 10), steps=3).flip(0), rtol=0, atol=0)
    assert deepcopy(VPFPValueControlEDM(config)).config == config


@pytest.mark.parametrize("weights", ((0., 0.), (0., .25), (.2, .25)))
def test_metric_spd_exact_trace_in_reduced_noise_space_and_gradient(weights):
    config = _config()
    metric = VPFPValueSobolevMetric(config, _normalizer(config), spatial_weight=weights[0], temporal_weight=weights[1])
    basis = torch.eye(config.latent_dimension, dtype=torch.float64)
    parts = metric.components(basis)
    for group in ("rho", "kinetic"):
        torch.testing.assert_close(parts[f"{group}_spatial"].sum(), torch.tensor(1., dtype=torch.float64))
        torch.testing.assert_close(parts[f"{group}_temporal"].sum(), torch.tensor(1., dtype=torch.float64))
    torch.testing.assert_close(parts["total"].sum(), torch.tensor(1., dtype=torch.float64))
    assert torch.all(parts["total"] > 0)
    residual = _values(config).requires_grad_(True)
    metric(residual).sum().backward()
    assert torch.isfinite(residual.grad).all()
    assert torch.sum(residual*residual.grad) > 0
    json.dumps(metric.report(), allow_nan=False)


def test_metric_constant_linear_temporal_fields_and_raw_time_units():
    config = _config(time_domain=(2., 5.))
    normalizer = VPFPValueControlNormalizer(config, torch.ones(3),
        fit_metadata={"split": "train", "generated_samples_used_for_fit": False})
    metric = VPFPValueSobolevMetric(config, normalizer)
    values = torch.zeros(1, config.latent_dimension, dtype=torch.float64)
    rho = values[:, :config.rho_dimension].reshape(1, config.control_count, config.rho_modes)
    rho[0, :, 0] = 1.
    assert metric.components(values)["rho_raw_temporal"] < 1.e-25
    rho[0, :, 0] = torch.linspace(0., 1., config.control_count)
    torch.testing.assert_close(metric.components(values)["rho_raw_temporal"], torch.tensor([2/9], dtype=torch.float64))
    shorter = _config(time_domain=(0., 1.))
    shorter_normalizer = VPFPValueControlNormalizer(shorter, torch.ones(3),
        fit_metadata={"split": "train", "generated_samples_used_for_fit": False})
    shorter_metric = VPFPValueSobolevMetric(shorter, shorter_normalizer)
    torch.testing.assert_close(shorter_metric.components(values)["rho_raw_temporal"],
        9*metric.components(values)["rho_raw_temporal"])
    torch.testing.assert_close(shorter_metric(values), metric(values))


def test_zero_weight_loss_bitwise_unchanged_and_fixed_metric_conditional_mean_optimum():
    config = _config()
    model = VPFPValueControlEDM(config)
    clean, noise = _values(config, dtype=torch.float32), _values(config, dtype=torch.float32).roll(1, 0)
    kwargs = {"sigma": torch.ones(4), "noise": noise}
    baseline, base_metrics = model.loss(clean, _conditions(config, 4), **kwargs)
    zero_metric = VPFPValueSobolevMetric(config, _normalizer(config), spatial_weight=0., temporal_weight=0.)
    unchanged, new_metrics = model.loss(clean, _conditions(config, 4), residual_metric=zero_metric, **kwargs)
    assert torch.equal(baseline, unchanged)
    assert all(torch.equal(base_metrics[key], new_metrics[key]) for key in base_metrics)
    metric = VPFPValueSobolevMetric(config, _normalizer(config))
    targets = _values(config)
    prediction = targets.mean(0, keepdim=True).detach().requires_grad_(True)
    metric(prediction-targets).mean().backward()
    assert prediction.grad.abs().max() < 1.e-15
    # The target can equally be conditional FM velocity; no clean-only map.
    velocity_targets = 2*targets + .3
    velocity = velocity_targets.mean(0, keepdim=True).detach().requires_grad_(True)
    metric(velocity-velocity_targets).mean().backward()
    assert velocity.grad.abs().max() < 1.e-15


def test_loader_dispatch_hash_bound_full_native_output_and_no_pde(tmp_path):
    config = _config()
    path, identity = _save(tmp_path, _payload(config))
    sampler = load_vpfp_diffusion_sampler(path, identity, steps=3)
    assert isinstance(sampler, VPFPValueDiffusionBatchSampler)
    assert sampler.identity["latent_dimension"] == config.full_dimension
    assert sampler.identity["random_dimension"] == config.latent_dimension
    assert sampler.identity["base_decoder_sha256"] == "a"*64
    assert sampler.identity["corpus_manifest_sha256"] == "b"*64
    assert sampler.identity["sampling"]["noise_space"] == "independent_value_coefficients_only"
    batch = sampler.generate((11, 12))
    assert batch["latents"].shape == (2, config.full_dimension)
    extract_value_controls(batch["latents"], config)
    assert batch["diagnostics"]["pde_solves"] == 0
    replay = sampler.generate((12, 11))
    np.testing.assert_array_equal(batch["latents"], replay["latents"][::-1])
    with_stale_metadata = load_vpfp_diffusion_sampler(path, "c"*64, steps=3)
    np.testing.assert_array_equal(batch["latents"], with_stale_metadata.generate((11,12))["latents"])
    link = tmp_path / "link.pt"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="non-symlink"):
        load_vpfp_diffusion_sampler(link, identity)


@pytest.mark.parametrize("corruption", ("field", "base_sha", "training_nan", "transform", "model_shape",
    "model_nan", "normalizer_scale", "normalizer_provenance", "config_mismatch",
    "parent_sha", "parent_value_scale", "parent_train_ids", "parent_floor_policy"))
def test_loader_rejects_tampered_value_checkpoint(tmp_path, corruption):
    config = _config()
    payload = _payload(config)
    if corruption == "field":
        payload["extra"] = 1
    elif corruption == "base_sha":
        payload["base_decoder_sha256"] = "invalid"
    elif corruption == "training_nan":
        payload["training"]["loss"] = float("nan")
    elif corruption == "transform":
        payload["model"]["transform.rho_basis"][0, 0, 0] += .1
    elif corruption == "model_shape":
        payload["model"]["output.bias"] = torch.zeros(1)
    elif corruption == "model_nan":
        payload["model"]["output.bias"][0] = float("nan")
    elif corruption == "normalizer_scale":
        payload["normalizer"]["group_scales"][0] = 0.
    elif corruption == "normalizer_provenance":
        payload["normalizer"]["fit_metadata"]["slope_subspace_verified"] = False
    elif corruption == "config_mismatch":
        payload["normalizer"]["config"]["full_config"]["macro_loss_fraction"] = .3
    elif corruption == "parent_sha":
        payload["normalizer"]["fit_metadata"]["parent_normalizer_sha256"] = "0"*64
    elif corruption in ("parent_value_scale", "parent_train_ids"):
        metadata = payload["normalizer"]["fit_metadata"]
        parent = metadata["parent_normalizer_payload"]
        if corruption == "parent_value_scale":
            parent["group_scales"][0][0] *= 2
        else:
            parent["fit_metadata"]["sample_ids"][0] = 99
        metadata["parent_normalizer_sha256"] = sha256(json.dumps(parent, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    elif corruption == "parent_floor_policy":
        payload["normalizer"]["fit_metadata"]["floor_policy"] = "new_value_only_floor"
    path, identity = _save(tmp_path, payload)
    if corruption == "parent_sha":
        assert load_vpfp_diffusion_sampler(path, identity, steps=2)
    else:
        with pytest.raises(ValueError):
            load_vpfp_diffusion_sampler(path, identity, steps=2)
