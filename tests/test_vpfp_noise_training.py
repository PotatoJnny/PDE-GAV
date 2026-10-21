"""Tiny synthetic distribution/API tests; no physical scientific validation."""

from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp.noise_training import (
    sample_low_noise_mixture, low_noise_mixture_report, VPFPLowNoiseValueControlEDM,
)
from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import (
    VPFPStructuredEDMConfig, load_vpfp_diffusion_sampler,
)
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import (
    VPFPValueEDMConfig, VPFPValueControlEDM, VPFPValueControlNormalizer,
    VPFPValueSobolevMetric, lift_value_controls,
)


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _config():
    return VPFPValueEDMConfig.from_full_config(VPFPStructuredEDMConfig(
        control_count=3, hermite_channels=2, rho_max_mode=1, kinetic_max_mode=1,
        inactive_dim=2, grid_size=3, width=8, depth=1,
        sigma_max=80., log_sigma_mean=-.6, log_sigma_std=1.8))


def _clean(config, count=4):
    return torch.randn(count, config.latent_dimension, generator=torch.Generator().manual_seed(31))


def _conditions(config, count=4):
    return torch.tensor([[sum(config.collision_nu_range)/2, sum(config.temperature_range)/2]]*count)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("probability", [0., .5, 1.])
def test_mixture_domain_determinism_and_dtype(dtype, probability):
    config, prototype = _config(), torch.empty((), dtype=dtype)
    first = sample_low_noise_mixture(config, 256, generator=torch.Generator().manual_seed(1),
        prototype=prototype, low_probability=probability)
    second = sample_low_noise_mixture(config, 256, generator=torch.Generator().manual_seed(1),
        prototype=prototype, low_probability=probability)
    assert torch.equal(first, second)
    assert first.dtype == dtype and first.shape == (256,)
    assert torch.isfinite(first).all() and torch.all(first >= config.sigma_min) and torch.all(first <= config.sigma_max)
    if probability == 1:
        assert torch.all(first <= .1)


def test_zero_probability_preserves_original_rng_and_sigma_bitwise():
    config = _config()
    first, second = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    sampled = sample_low_noise_mixture(config, 50, generator=first, prototype=torch.empty(()), low_probability=0.)
    expected = (torch.randn(50, generator=second)*config.log_sigma_std+config.log_sigma_mean).exp().clamp(
        config.sigma_min, config.sigma_max)
    assert torch.equal(sampled, expected)
    assert torch.equal(first.get_state(), second.get_state())


def test_distribution_frequencies_and_report_are_the_declared_mixture():
    config = _config()
    report = low_noise_mixture_report(config)
    before = torch.get_rng_state().clone()
    json.dumps(report, allow_nan=False)
    assert torch.equal(before, torch.get_rng_state())
    sampled = sample_low_noise_mixture(config, 100000, generator=torch.Generator().manual_seed(7),
        prototype=torch.empty(()))
    edges = torch.tensor(config.sigma_bin_edges[1:-1])
    observed = torch.bincount(torch.bucketize(sampled, edges), minlength=4)/len(sampled)
    torch.testing.assert_close(observed, torch.tensor(report["sigma_bin_probabilities"]), atol=.005, rtol=0)
    # Original P(sigma < .01)=.01304; half-mixture expectation is .21222.
    assert .205 < float((sampled < .01).float().mean()) < .22
    assert report["full_original_sigma_support"] is True
    assert report["importance_reweighting"] is False
    assert low_noise_mixture_report(config, low_probability=1.)["full_original_sigma_support"] is False


def test_default_low_min_preserves_old_sampling_and_explicit_min_has_same_rng():
    config = _config()
    first, second = torch.Generator().manual_seed(17), torch.Generator().manual_seed(17)
    default = sample_low_noise_mixture(config, 64, generator=first, prototype=torch.empty(()))
    explicit = sample_low_noise_mixture(config, 64, generator=second, prototype=torch.empty(()), low_min=config.sigma_min)
    assert torch.equal(default, explicit) and torch.equal(first.get_state(), second.get_state())
    assert low_noise_mixture_report(config)["low_min"] == config.sigma_min
    assert low_noise_mixture_report(config)["low_min_policy"] == "sigma_min_default"


def test_diagnostic_band_point01_to_point3_frequency_and_full_support():
    config = _config()
    report = low_noise_mixture_report(config, low_min=.01, low_max=.3)
    sampled = sample_low_noise_mixture(config, 100000, generator=torch.Generator().manual_seed(18),
        prototype=torch.empty(()), low_min=.01, low_max=.3)
    observed = torch.bincount(torch.bucketize(sampled, torch.tensor(config.sigma_bin_edges[1:-1])), minlength=4)/len(sampled)
    torch.testing.assert_close(observed, torch.tensor(report["sigma_bin_probabilities"]), atol=.005, rtol=0)
    assert report["low_min"] == .01 and report["low_max"] == .3
    assert report["low_min_policy"] == "explicit_band_lower_bound"
    assert report["full_original_sigma_support"] is True
    assert .003 < float((sampled < .01).float().mean()) < .01
    pure = sample_low_noise_mixture(config, 256, generator=torch.Generator().manual_seed(18),
        prototype=torch.empty(()), low_probability=1., low_min=.01, low_max=.3)
    assert torch.all(pure >= .01) and torch.all(pure <= .3)
    assert low_noise_mixture_report(config, low_probability=1., low_min=.01,
        low_max=config.sigma_max)["full_original_sigma_support"] is False


@pytest.mark.parametrize("change", [
    {"low_probability": -.1}, {"low_probability": 1.1}, {"low_probability": True},
    {"low_probability": float("nan")}, {"low_max": .002}, {"low_max": 81.},
    {"low_max": float("inf")}, {"low_max": True},
    {"low_min": .001}, {"low_min": .1}, {"low_min": .2},
    {"low_min": float("nan")}, {"low_min": True},
])
def test_invalid_mixture_parameters_fail_without_consuming_rng(change):
    generator = torch.Generator().manual_seed(10)
    before = generator.get_state().clone()
    with pytest.raises(ValueError):
        sample_low_noise_mixture(_config(), 4, generator=generator, prototype=torch.empty(()), **change)
    assert torch.equal(before, generator.get_state())


@pytest.mark.parametrize("change", [{"sigma_min": 0.}, {"sigma_max": .001},
    {"log_sigma_std": 0.}, {"log_sigma_mean": float("nan")}, {"sigma_min": True}])
def test_invalid_config_fails_closed(change):
    config = _config()
    fake = SimpleNamespace(**{key:getattr(config, key) for key in (
        "sigma_min", "sigma_max", "log_sigma_mean", "log_sigma_std")})
    for key, value in change.items():
        setattr(fake, key, value)
    with pytest.raises(ValueError):
        sample_low_noise_mixture(fake, 4, generator=torch.Generator(), prototype=torch.empty(()))


@pytest.mark.parametrize("change", [{"sigma_min": 1.e-50}, {"log_sigma_std": 1.e-50},
    {"log_sigma_std": 1.e100}, {"log_sigma_mean": 1.e100}])
def test_finite_but_unrepresentable_parameters_fail_before_rng(change):
    config = _config()
    fake = SimpleNamespace(**{key:getattr(config, key) for key in (
        "sigma_min", "sigma_max", "log_sigma_mean", "log_sigma_std")})
    for key, value in change.items():
        setattr(fake, key, value)
    generator = torch.Generator().manual_seed(9)
    before = generator.get_state().clone()
    with pytest.raises(ValueError, match="representable"):
        sample_low_noise_mixture(fake, 4, generator=generator, prototype=torch.empty(()))
    assert torch.equal(before, generator.get_state())


@pytest.mark.parametrize("batch", [0, -1, True, 1.5])
def test_invalid_batch(batch):
    with pytest.raises(ValueError, match="batch"):
        sample_low_noise_mixture(_config(), batch, generator=torch.Generator(), prototype=torch.empty(()))


@pytest.mark.parametrize("prototype", [torch.empty((), dtype=torch.float16), torch.tensor(1), None])
def test_invalid_prototype(prototype):
    with pytest.raises(ValueError, match="prototype"):
        sample_low_noise_mixture(_config(), 4, generator=torch.Generator(), prototype=prototype)


@pytest.mark.parametrize("provide_noise", [False, True])
@pytest.mark.parametrize("band", [{}, {"low_min": .01, "low_max": .3}])
def test_explicit_sigma_bypasses_mixture_without_any_extra_rng(provide_noise, band):
    config = _config()
    base = VPFPValueControlEDM(config)
    mixture = VPFPLowNoiseValueControlEDM(config, **band)
    mixture.load_state_dict(base.state_dict())
    first, second = torch.Generator().manual_seed(91), torch.Generator().manual_seed(91)
    clean, condition = _clean(config), _conditions(config)
    kwargs = {"sigma": torch.tensor([.003, .03, .3, 3.])}
    if provide_noise:
        kwargs["noise"] = clean.roll(1, 0)
    first_loss, first_metrics = base.loss(clean, condition, generator=first, **kwargs)
    second_loss, second_metrics = mixture.loss(clean, condition, generator=second, **kwargs)
    assert torch.equal(first_loss, second_loss)
    assert all(torch.equal(first_metrics[k], second_metrics[k]) for k in first_metrics)
    assert torch.equal(first.get_state(), second.get_state())


def test_implicit_mixture_matches_direct_sampling_and_preserves_optional_metric():
    config = _config()
    model = VPFPLowNoiseValueControlEDM(config)
    base = VPFPValueControlEDM(config)
    base.load_state_dict(model.state_dict())
    clean = _clean(config)
    normalizer = VPFPValueControlNormalizer.fit(config, lift_value_controls(clean, config), sample_ids=(1,2,3,4))
    metric = VPFPValueSobolevMetric(config, normalizer)
    first, second = torch.Generator().manual_seed(19), torch.Generator().manual_seed(19)
    explicit = sample_low_noise_mixture(config, len(clean), generator=first, prototype=next(base.parameters()))
    expected, expected_metrics = base.loss(clean, _conditions(config), generator=first, sigma=explicit, residual_metric=metric)
    actual, actual_metrics = model.loss(clean, _conditions(config), generator=second, residual_metric=metric)
    assert torch.equal(expected, actual)
    assert all(torch.equal(expected_metrics[k], actual_metrics[k]) for k in expected_metrics)
    assert torch.equal(first.get_state(), second.get_state())
    actual.backward()
    assert torch.isfinite(model.output.weight.grad).all()


def test_loader_restores_original_model_with_identical_inference(tmp_path):
    config = _config()
    model = VPFPLowNoiseValueControlEDM(config)
    with torch.no_grad():
        model.output.weight.normal_(std=.02)
    assert model.state_dict().keys() == VPFPValueControlEDM(config).state_dict().keys()
    assert model.checkpoint_schema == VPFPValueControlEDM.checkpoint_schema
    clean = _clean(config)
    normalizer = VPFPValueControlNormalizer.fit(config, lift_value_controls(clean, config), sample_ids=(1,2,3,4))
    report = model.training_noise_report()
    payload = {"schema": model.checkpoint_schema, "config": config.to_dict(), "model": model.state_dict(),
        "normalizer": normalizer.to_payload(), "base_decoder_sha256": "a"*64,
        "corpus_manifest_sha256": "b"*64, "training": {"step": 1, "training_noise": report}}
    path = tmp_path / "mixture-trained.pt"
    torch.save(payload, path)
    digest = sha256(path.read_bytes()).hexdigest()
    loaded = load_vpfp_diffusion_sampler(path, digest, steps=3)
    assert type(loaded.model) is VPFPValueControlEDM
    assert loaded.identity["training"]["training_noise"] == report
    expected = model.sample(_conditions(config, 2), (10, 11), steps=3)
    actual = loaded.model.sample(_conditions(config, 2), (10, 11), steps=3)
    assert torch.equal(expected, actual)
    assert deepcopy(model).training_noise_report() == report
