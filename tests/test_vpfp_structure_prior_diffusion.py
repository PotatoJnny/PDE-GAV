"""Tiny architecture/native-ABI tests, not scientific VPFP validation."""

from copy import deepcopy
from hashlib import sha256
import json
import math

import numpy as np
import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import (
    VPFPStructuredEDMConfig, load_vpfp_diffusion_sampler,
)
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import (
    VPFPValueEDMConfig, lift_value_controls, extract_value_controls,
)
from pde_agent_verifier_solver.plugins.vpfp.temporal_covariance_diffusion import (
    VPFPTemporalCovarianceNormalizer, VPFPTemporalCovarianceEDM,
)
from pde_agent_verifier_solver.plugins.vpfp.structure_prior_diffusion import (
    BRANCH_KINDS, CHECKPOINT_SCHEMAS, VPFPWeakStructurePriorEDM,
    make_structure_prior_model, _spatial_projector,
)


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _config():
    return VPFPValueEDMConfig.from_full_config(VPFPStructuredEDMConfig(
        control_count=4, hermite_channels=3, rho_max_mode=3, kinetic_max_mode=3,
        inactive_dim=3, grid_size=7, width=8, depth=1,
        sigma_max=80., log_sigma_mean=-.6, log_sigma_std=1.8))


def _conditions(config, count=2):
    return torch.tensor([[sum(config.collision_nu_range)/2, sum(config.temperature_range)/2]]*count)


def _values(config, count=2):
    return torch.randn(count, config.latent_dimension, generator=torch.Generator().manual_seed(419))


def _model(kind, config=None):
    torch.manual_seed(371)
    return make_structure_prior_model(_config() if config is None else config, kind)


def _activate_branch(model):
    with torch.no_grad():
        model.structure_branch.output.weight.copy_(torch.randn(
            model.structure_branch.output.weight.shape, generator=torch.Generator().manual_seed(8))*.02)


def _payload(kind):
    config = _config()
    raw = lift_value_controls(_values(config, count=8).double(), config)
    normalizer = VPFPTemporalCovarianceNormalizer.fit(config, raw, sample_ids=list(range(8)),
        covariance_strength=.5, rho_policy="identity", center_kinetic_dc=True)
    model = _model(kind, config)
    _activate_branch(model)
    return {"schema": model.checkpoint_schema, "config": config.to_dict(), "model": model.state_dict(),
        "normalizer": normalizer.to_payload(), "base_decoder_sha256": "a"*64,
        "corpus_manifest_sha256": "b"*64, "training": {"step": 2, "mechanics_fixture": True}}


def _save(tmp_path, payload, name="model.pt"):
    path = tmp_path / name
    torch.save(payload, path)
    return path, sha256(path.read_bytes()).hexdigest()


def test_exact_parameter_matching_original_trunk_initialization_and_zero_start():
    config = _config()
    torch.manual_seed(371)
    original = VPFPTemporalCovarianceEDM(config)
    prior, control = (_model(kind, config) for kind in BRANCH_KINDS)
    for name, value in original.state_dict().items():
        assert torch.equal(value, prior.state_dict()[name])
        assert torch.equal(value, control.state_dict()[name])
    parameters = dict(prior.named_parameters())
    assert parameters.keys() == dict(control.named_parameters()).keys()
    assert all(torch.equal(value, dict(control.named_parameters())[name]) for name, value in parameters.items())
    assert not torch.equal(prior.structure_branch.spatial_projector, control.structure_branch.spatial_projector)
    assert prior.architecture_report()["branch_parameters"] == control.architecture_report()["branch_parameters"]
    values, conditions, sigma = _values(config), _conditions(config), torch.tensor([.1, 2.])
    expected = original.denoise(values, sigma, conditions)
    for model in (prior, control):
        assert torch.equal(model.denoise(values, sigma, conditions), expected)
        assert torch.equal(model.sample(conditions, [17, 18], steps=2), original.sample(conditions, [17, 18], steps=2))
        assert len(model.structure_branch.blocks) == 4
        assert model.structure_branch.input.in_channels == config.hermite_channels + 1
        assert model.structure_branch.output.out_channels == config.hermite_channels + 1


def test_declared_full_layout_parameter_budget():
    config = VPFPValueEDMConfig.from_full_config(VPFPStructuredEDMConfig(hermite_channels=120, width=256, depth=4))
    model = _model("lowmode_coherent", config)
    report = model.architecture_report()
    assert report["base_parameters"] == 1742713
    assert report["branch_parameters"] == 128569
    assert report["total_parameters"] == 1871282
    assert report["unchanged_full_support_trunk"] is True
    assert report["all_Hermite_amplitude_channels_retained"] is True
    assert report["exact_Hermite_transport_physics_imposed"] is False
    assert report["physical_derivative_distribution_loss"] is False
    json.dumps(report, allow_nan=False)


def test_low_spatial_projector_is_fixed_orthogonal_translation_equivariant_with_DC():
    projector = _spatial_projector(13, "lowmode_coherent")
    torch.testing.assert_close(projector, projector.T, rtol=0, atol=0)
    torch.testing.assert_close(projector @ projector, projector, rtol=1.e-6, atol=1.e-7)
    torch.testing.assert_close(projector @ torch.ones(13), torch.ones(13), rtol=0, atol=2.e-7)
    x = torch.arange(13)*2*math.pi/13
    for frequency in (1, 2, 3, 4, 5, 6):
        for values in ((frequency*x).sin(), (frequency*x).cos()):
            expected = values if frequency <= 2 else torch.zeros_like(values)
            torch.testing.assert_close(projector @ values, expected, rtol=0, atol=2.e-6)
    values = torch.randn(13, 13, generator=torch.Generator().manual_seed(4))
    torch.testing.assert_close(projector @ values.roll((3, 2), (-2, -1)) @ projector,
        (projector @ values @ projector).roll((3, 2), (-2, -1)), rtol=1.e-6, atol=2.e-7)
    assert not projector.requires_grad


@pytest.mark.parametrize("kind", BRANCH_KINDS)
def test_same_branch_operation_shapes_without_downsampling(kind):
    model = _model(kind)
    shapes = []
    handles = [module.register_forward_hook(lambda module, inputs, output: shapes.append(tuple(output.shape)))
               for module in (model.structure_branch.input, *model.structure_branch.blocks, model.structure_branch.output)]
    config = model.config
    model.denoise(_values(config), torch.tensor([.2, .3]), _conditions(config))
    for handle in handles:
        handle.remove()
    assert shapes == [(2, 64, 4, 7, 7)]*5 + [(2, 4, 4, 7, 7)]


def test_prior_branch_output_is_low_band_but_complete_denoiser_keeps_high_modes():
    model = _model("lowmode_coherent")
    _activate_branch(model)
    cfg = model.config
    values = _values(cfg)
    sigma, conditions = torch.tensor([.1, 1.]), _conditions(cfg)
    parent = VPFPTemporalCovarianceEDM(cfg)
    parent.load_state_dict({name: value for name, value in model.state_dict().items()
                           if not name.startswith("structure_branch.")})
    field = model.transform.to_field(values)
    branch_output = model.structure_branch(field, sigma, model._conditions(conditions, 2, values))
    torch.testing.assert_close(model.structure_branch.project(branch_output), branch_output, rtol=2.e-6, atol=2.e-7)
    x = torch.arange(cfg.grid_size)*2*math.pi/cfg.grid_size
    high = (3*x).cos()[None, None, None, None, :].expand_as(field)
    torch.testing.assert_close(model.structure_branch(field+high, sigma, model._conditions(conditions, 2, values)),
        branch_output, rtol=3.e-5, atol=2.e-7)
    high_coefficients = model.transform.to_coefficients(high)
    result = model.denoise(high_coefficients, sigma, conditions)
    original = parent.denoise(high_coefficients, sigma, conditions)
    # High-mode output is unchanged, not zeroed by a sampler-state projection.
    original_high = (original*high_coefficients).sum(1)
    assert torch.all(original_high > 0)
    torch.testing.assert_close((result*high_coefficients).sum(1), original_high, rtol=2.e-6, atol=2.e-5)


@pytest.mark.parametrize("kind", BRANCH_KINDS)
def test_single_zero_output_allows_gradient_to_start_then_reach_branch_input(kind):
    model = _model(kind)
    cfg = model.config
    optimizer = torch.optim.SGD(model.structure_branch.parameters(), lr=.01)
    values = _values(cfg)
    loss, _ = model.loss(values, _conditions(cfg), sigma=torch.tensor([.1, .3]), noise=_values(cfg))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.structure_branch.output.weight.grad.abs().sum() > 0
    assert model.structure_branch.output.weight.grad[-1].abs().sum() > 0  # highest retained Hermite channel
    assert model.structure_branch.input.weight.grad.abs().sum() == 0  # one zero final affine is intentional
    optimizer.step()
    model.zero_grad(set_to_none=True)
    loss, _ = model.loss(values, _conditions(cfg), sigma=torch.tensor([.1, .3]), noise=_values(cfg))
    loss.backward()
    assert model.structure_branch.input.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(value.grad).all() for value in model.parameters() if value.grad is not None)


@pytest.mark.parametrize("kind", BRANCH_KINDS)
def test_original_EDM_objective_and_explicit_sigma_do_not_consume_extra_rng(kind):
    model = _model(kind)
    _activate_branch(model)
    cfg = model.config
    clean, noise, sigma = _values(cfg), _values(cfg)*.4, torch.tensor([.01, 1.])
    generator = torch.Generator().manual_seed(17)
    initial = generator.get_state().clone()
    loss, metrics = model.loss(clean, _conditions(cfg), sigma=sigma, noise=noise, generator=generator)
    assert torch.equal(initial, generator.get_state())
    error = (model.denoise(clean+sigma[:, None]*noise, sigma, _conditions(cfg))-clean).square()
    weight = (sigma.square()+cfg.sigma_data**2)/(sigma*cfg.sigma_data).square()
    expected = (cfg.macro_loss_fraction*error[:, :cfg.rho_dimension].mean(1)
                +(1-cfg.macro_loss_fraction)*error[:, cfg.rho_dimension:].mean(1))*weight
    torch.testing.assert_close(loss, expected.mean(), rtol=0, atol=0)
    assert torch.equal(metrics["base_loss_per_case"], metrics["loss_per_case"])
    with pytest.raises(ValueError, match="Sobolev"):
        model.loss(clean, _conditions(cfg), residual_metric=object())


@pytest.mark.parametrize("kind", BRANCH_KINDS)
def test_native_hash_loader_schema_geometry_full_ABI_and_sampling_roundtrip(tmp_path, kind):
    payload = _payload(kind)
    path, digest = _save(tmp_path, payload)
    sampler = load_vpfp_diffusion_sampler(path, digest, steps=2)
    cfg = sampler.model.config
    assert sampler.identity["checkpoint_schema"] == CHECKPOINT_SCHEMAS[kind]
    assert sampler.identity["architecture"]["branch_kind"] == kind
    assert sampler.identity["latent_dimension"] == cfg.full_dimension
    assert sampler.identity["random_dimension"] == cfg.latent_dimension
    assert sampler.identity["checkpoint_sha256"] == digest
    assert sampler.identity["base_decoder_sha256"] == "a"*64
    assert sampler.identity["corpus_manifest_sha256"] == "b"*64
    assert sampler.identity["geometry_binding"]["covariance_strength"] == .5
    assert sampler.normalizer.fit_metadata["sample_ids"] == list(range(8))
    assert sampler.normalizer.rho_policy == "identity"
    assert not sampler.model.training and not any(p.requires_grad for p in sampler.model.parameters())
    report = sampler.identity["architecture"]
    assert sampler.identity["architecture_sha256"] == sha256(json.dumps(report, allow_nan=False,
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    model = _model(kind)
    model.load_state_dict(payload["model"])
    expected = sampler.normalizer.decode(model.sample(_conditions(cfg), [11, 12], steps=2)).detach().numpy()
    actual = sampler.normalizer.decode(sampler.model.sample(_conditions(cfg), [11, 12], steps=2)).detach().numpy()
    np.testing.assert_array_equal(actual, expected)
    batch = sampler.generate([31, 32])
    assert batch["latents"].shape == (2, cfg.full_dimension)
    assert np.isfinite(batch["latents"]).all()
    raw = torch.from_numpy(batch["latents"])
    torch.testing.assert_close(lift_value_controls(extract_value_controls(raw, cfg), cfg), raw, rtol=0, atol=0)
    assert batch["diagnostics"]["pde_solves"] == 0
    with pytest.raises(ValueError, match="SHA256"):
        load_vpfp_diffusion_sampler(path, "0"*64, steps=2)


@pytest.mark.parametrize("corruption", (
    "branch_nan", "branch_dtype", "branch_shape", "projector", "wrong_variant",
    "missing_branch", "extra_tensor", "fourier", "config", "training_nan", "source_sha",
    "normalizer_geometry", "normalizer_source", "extra_payload",
))
def test_rehashed_native_tampering_is_rejected(tmp_path, corruption):
    payload = deepcopy(_payload("lowmode_coherent"))
    state = payload["model"]
    key = "structure_branch.input.weight"
    if corruption == "branch_nan":
        state[key].flatten()[0] = float("nan")
    elif corruption == "branch_dtype":
        state[key] = state[key].double()
    elif corruption == "branch_shape":
        state[key] = state[key][:1]
    elif corruption == "projector":
        state["structure_branch.spatial_projector"][0, 0] += .1
    elif corruption == "wrong_variant":
        payload["schema"] = CHECKPOINT_SCHEMAS["branch_identity_control"]
    elif corruption == "missing_branch":
        del state[key]
    elif corruption == "extra_tensor":
        state["structure_branch.extra"] = torch.zeros(1)
    elif corruption == "fourier":
        state["transform.kinetic_basis"].flatten()[0] += 1
    elif corruption == "config":
        payload["config"]["full_config"]["sigma_max"] = 40.
    elif corruption == "training_nan":
        payload["training"]["loss"] = float("nan")
    elif corruption == "source_sha":
        payload["corpus_manifest_sha256"] = "not-a-sha"
    elif corruption == "normalizer_geometry":
        payload["normalizer"]["fit_metadata"]["normalization_scope_sha256"] = "0"*64
    elif corruption == "normalizer_source":
        payload["normalizer"]["value_normalizer"]["fit_metadata"]["sample_ids"][0] = 91
    else:
        payload["architecture"] = {"claims": "unbound fields are forbidden"}
    path, digest = _save(tmp_path, payload)
    with pytest.raises(ValueError):
        load_vpfp_diffusion_sampler(path, digest, steps=2)


@pytest.mark.parametrize("kind", ("", "coherent", None, 3))
def test_unknown_branch_kind_rejected(kind):
    with pytest.raises(ValueError, match="branch kind"):
        make_structure_prior_model(_config(), kind)


def test_fixed_low_spatial_prior_requires_resolved_grid_and_value_config():
    with pytest.raises(ValueError, match="grid_size"):
        _spatial_projector(3, "lowmode_coherent")
    with pytest.raises(TypeError, match="value-state"):
        VPFPWeakStructurePriorEDM(VPFPStructuredEDMConfig())
