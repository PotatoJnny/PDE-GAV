"""CPU synthetic operator tests; not a real-data distribution qualification."""

from dataclasses import replace
import math

import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp.sobolev_matching import VPFPSobolevMetric
from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import (
    VPFPCoefficientNormalizer, VPFPStructuredEDM, VPFPStructuredEDMConfig, _join, _split,
)


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _config(**changes):
    base = VPFPStructuredEDMConfig(control_count=3, hermite_channels=2,
        rho_max_mode=1, kinetic_max_mode=1, inactive_dim=2, grid_size=3, width=8, depth=1)
    return replace(base, **changes)


def _normalizer(config):
    # Deliberately different values/slopes/channel scales; tests must include
    # the inverse normalizer, not silently measure standardized fields.
    scales = torch.arange(1, 2*(config.hermite_channels+1)+1, dtype=torch.float64).reshape(-1, 2) / 3
    return VPFPCoefficientNormalizer(config, scales,
        fit_metadata={"split": "train", "generated_samples_used_for_fit": False})


def _answers(config, count=3, dtype=torch.float64):
    values = torch.randn((count, config.latent_dimension), dtype=dtype,
                         generator=torch.Generator().manual_seed(93))
    values[:, config.rho_dimension:config.rho_dimension+config.inactive_dim] = 0.
    return values


def _active_basis(config):
    indices = torch.cat((torch.arange(config.rho_dimension),
        torch.arange(config.rho_dimension+config.inactive_dim, config.latent_dimension)))
    return torch.eye(config.latent_dimension, dtype=torch.float64)[indices]


def test_exact_cubic_mass_and_derivative_gram_including_cross_terms():
    config = _config(time_domain=(2., 5.))
    metric = VPFPSobolevMetric(config, _normalizer(config))
    h, duration = config.slope_interval, 3.
    expected_mass = h/420 * torch.tensor([
        [156, 54, 22*h, -13*h], [54, 156, 13*h, -22*h],
        [22*h, 13*h, 4*h*h, -3*h*h], [-13*h, -22*h, -3*h*h, 4*h*h]], dtype=torch.float64)
    expected_time = 1/(30*h*duration**2) * torch.tensor([
        [36, -36, 3*h, 3*h], [-36, 36, -3*h, -3*h],
        [3*h, -3*h, 4*h*h, -h*h], [3*h, -3*h, -h*h, 4*h*h]], dtype=torch.float64)
    torch.testing.assert_close(metric.mass_factor.T @ metric.mass_factor, expected_mass, atol=2.e-16, rtol=3.e-14)
    torch.testing.assert_close(metric.time_factor.T @ metric.time_factor, expected_time, atol=2.e-16, rtol=3.e-14)
    assert expected_time[0, 2] != 0  # not a slope-only penalty


@pytest.mark.parametrize("group", ["rho", "kinetic"])
def test_constant_and_linear_physical_time_derivatives(group):
    config = _config(time_domain=(1., 4.))
    normalizer = _normalizer(config)
    metric = VPFPSobolevMetric(config, normalizer)
    raw = torch.zeros((1, config.latent_dimension), dtype=torch.float64)
    rho, kinetic = _split(raw, config)
    target = rho[0] if group == "rho" else kinetic[0, 0]
    target[0, :, 0] = 1.
    constant = metric.components(normalizer.encode(raw))
    assert constant[f"{group}_raw_temporal"].item() < 1.e-27
    target[0, :, 0] = torch.linspace(0., 1., config.control_count)
    target[1, :, 0] = 1.  # d/dtau, NOT d/dt
    linear = metric.components(normalizer.encode(raw))
    expected = 2/9 if group == "rho" else 4/(config.kinetic_modes*config.hermite_channels*9)
    torch.testing.assert_close(linear[f"{group}_raw_temporal"], torch.tensor([expected], dtype=torch.float64))
    # Values with missing slopes are still C1 Hermite, but a different curve.
    target[1, :, 0] = 0.
    no_slopes = metric.components(normalizer.encode(raw))
    assert no_slopes[f"{group}_raw_temporal"] > linear[f"{group}_raw_temporal"]


def _mode_index(maximum, kx, ky, *, constant=False):
    index = int(constant)
    for first in range(-maximum, maximum+1):
        for second in range(-maximum, maximum+1):
            if first > 0 or (first == 0 and second > 0):
                if (first, second) == (kx, ky):
                    return index
                index += 2
    raise AssertionError("missing test mode")


@pytest.mark.parametrize("group", ["rho", "kinetic"])
def test_spatial_single_fourier_mode_has_k_squared_and_physical_length(group):
    config = _config(rho_max_mode=2, kinetic_max_mode=2, grid_size=5)
    normalizer = _normalizer(config)
    metric = VPFPSobolevMetric(config, normalizer, spatial_lengths=(4*math.pi, 2*math.pi))
    energies = []
    for wave in (1, 2):
        raw = torch.zeros((1, config.latent_dimension), dtype=torch.float64)
        rho, kinetic = _split(raw, config)
        target = rho[0] if group == "rho" else kinetic[0, 0]
        target[0, :, _mode_index(2, wave, 0, constant=group == "kinetic")] = 1.
        component = metric.components(normalizer.encode(raw))
        energies.append(component[f"{group}_raw_x"])
        assert component[f"{group}_raw_y"].item() == 0.
        assert component[f"{group}_raw_temporal"].item() < 1.e-27
    expected = .5 if group == "rho" else 1/(config.kinetic_modes*config.hermite_channels)
    torch.testing.assert_close(energies[0], torch.tensor([expected], dtype=torch.float64))
    torch.testing.assert_close(energies[1], 4*energies[0])


def test_trace_normalization_is_exact_and_total_preserves_white_noise_scale():
    config = _config()
    metric = VPFPSobolevMetric(config, _normalizer(config), spatial_weight=.25, temporal_weight=.25)
    # Summing Q(e_i) is the exact trace, no Monte Carlo accuracy assertion.
    components = metric.components(_active_basis(config))
    for group in ("rho", "kinetic"):
        for name in ("base", "spatial", "temporal", "total"):
            torch.testing.assert_close(components[f"{group}_{name}"].sum(), torch.tensor(1., dtype=torch.float64))
    torch.testing.assert_close(components["total"].sum(), torch.tensor(1., dtype=torch.float64))
    report = metric.report()
    assert report["total_scale_divisor"] == 1.5
    assert report["physical_derivative_distribution_matching"] is False
    assert report["prediction_or_batch_dependent_weights"] is False


def test_positive_definite_active_metric_and_exact_autograd():
    config = _config()
    metric = VPFPSobolevMetric(config, _normalizer(config), spatial_weight=.8, temporal_weight=.3)
    basis = _active_basis(config).requires_grad_(True)
    gradients = torch.autograd.grad(metric(basis).sum(), basis)[0]
    active = torch.nonzero(basis.detach().sum(0)).flatten()
    matrix = gradients[:, active] / 2
    torch.testing.assert_close(matrix, matrix.T)
    lower_bound = min(config.macro_loss_fraction/config.rho_dimension,
        (1-config.macro_loss_fraction)/config.kinetic_dimension) / metric.total_scale
    assert torch.linalg.eigvalsh(matrix).min() >= lower_bound - 2.e-16
    values = _answers(config).requires_grad_(True)
    torch.testing.assert_close(metric(values), torch.einsum("bi,ij,bj->b", values[:, active], matrix, values[:, active]))
    gradient = torch.autograd.grad(metric(values).sum(), values)[0]
    torch.testing.assert_close(gradient[:, active], 2*values[:, active] @ matrix)
    assert gradient[:, config.rho_dimension:config.rho_dimension+config.inactive_dim].count_nonzero() == 0


def test_periodic_translation_preserves_spatial_and_temporal_metric():
    config = _config()
    metric = VPFPSobolevMetric(config, _normalizer(config), spatial_lengths=(7., 9.))
    values = _answers(config)
    rho, kinetic = _split(values.clone(), config)
    for data, maximum, constant in ((rho, config.rho_max_mode, False), (kinetic, config.kinetic_max_mode, True)):
        for kx in range(-maximum, maximum+1):
            for ky in range(-maximum, maximum+1):
                if kx > 0 or (kx == 0 and ky > 0):
                    index = _mode_index(maximum, kx, ky, constant=constant)
                    angle = 2*math.pi*(kx*.43/7 + ky*.71/9)
                    cosine, sine = data[..., index].clone(), data[..., index+1].clone()
                    data[..., index] = cosine*math.cos(angle) + sine*math.sin(angle)
                    data[..., index+1] = -cosine*math.sin(angle) + sine*math.cos(angle)
    shifted = _join(rho, kinetic, config)
    original, translated = metric.components(values), metric.components(shifted)
    for key in original:
        torch.testing.assert_close(original[key], translated[key], atol=1.e-12, rtol=1.e-13)


def test_snapshot_is_fixed_across_normalizer_mutation_batching_and_device_dtype():
    config = _config()
    normalizer = _normalizer(config)
    metric = VPFPSobolevMetric(config, normalizer).to("cpu")
    values = _answers(config)
    before = metric(values)
    normalizer.group_scales *= 7
    torch.testing.assert_close(metric(values), before, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat([metric(row[None]) for row in values]), before)
    float_values = values.float().requires_grad_(True)
    float_output = metric(float_values)
    torch.testing.assert_close(float_output.double(), before, rtol=2.e-6, atol=2.e-7)
    float_output.sum().backward()
    assert torch.isfinite(float_values.grad).all()
    assert not list(metric.parameters())


def test_physical_time_and_amplitude_scales_are_reported_before_trace_cancellation():
    config = _config()
    normalizer = _normalizer(config)
    first = VPFPSobolevMetric(config, normalizer)
    longer_config = replace(config, time_domain=(0., 2.))
    longer = VPFPSobolevMetric(longer_config, _normalizer(longer_config))
    scaled = VPFPSobolevMetric(config, normalizer, latent_input_scale=6.)
    values = _answers(config)
    a, b, c = (metric.components(values) for metric in (first, longer, scaled))
    for group in ("rho", "kinetic"):
        torch.testing.assert_close(b[f"{group}_raw_temporal"], a[f"{group}_raw_temporal"]/4)
        torch.testing.assert_close(c[f"{group}_raw_temporal"], a[f"{group}_raw_temporal"]*9)
        torch.testing.assert_close(b[f"{group}_temporal"], a[f"{group}_temporal"])
        torch.testing.assert_close(c[f"{group}_temporal"], a[f"{group}_temporal"])


@pytest.mark.parametrize("target_kind", ["denoiser", "flow_matching_velocity"])
def test_fixed_metric_keeps_conditional_mean_optimum(target_kind):
    config = _config()
    metric = VPFPSobolevMetric(config, _normalizer(config), spatial_weight=.8, temporal_weight=.4)
    # Exact finite conditional target distribution at fixed y, sigma/time, c.
    targets = _answers(config, count=3)
    if target_kind == "flow_matching_velocity":
        targets = targets - _answers(config, count=3).roll(1, 0)
    probabilities = torch.tensor([.2, .3, .5], dtype=torch.float64)
    mean = (probabilities[:, None]*targets).sum(0).requires_grad_(True)
    objective = (probabilities*metric(mean[None]-targets)).sum()
    gradient = torch.autograd.grad(objective, mean)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=2.e-15, rtol=0)
    shifted = mean.detach() + .3*_answers(config, count=1)[0]
    assert (probabilities*metric(shifted[None]-targets)).sum() > objective


def test_noise_denoiser_and_flow_velocity_residual_changes_of_variables():
    config = _config()
    metric = VPFPSobolevMetric(config, _normalizer(config))
    error = _answers(config)
    sigma = torch.tensor([.02, .4, 4.], dtype=torch.float64)
    torch.testing.assert_close(metric(-sigma[:, None]*error), sigma.square()*metric(error))
    # FM path y_t=(1-t)*noise+t*answer => answer=y_t+(1-t)*velocity.
    times = torch.tensor([.1, .4, .9], dtype=torch.float64)
    torch.testing.assert_close(metric((1-times[:, None])*error), (1-times).square()*metric(error))


def test_edm_integration_same_metric_baseline_base_diagnostics_and_gradients():
    config = _config()
    model = VPFPStructuredEDM(config)
    metric = VPFPSobolevMetric(config, _normalizer(config), spatial_weight=.25, temporal_weight=.25)
    clean, noise = _answers(config, dtype=torch.float32), _answers(config, dtype=torch.float32).roll(1, 0)
    conditions = torch.tensor([[.05, 1.]]*len(clean))
    sigma = torch.tensor([.01, .2, 2.])
    loss, metrics = model.loss(clean, conditions, sigma=sigma, noise=noise, residual_metric=metric)
    residual = model.denoise(clean+sigma[:, None]*noise, sigma, conditions)-clean
    weight = (1+sigma.square())/sigma.square()
    expected = metric.components(residual)
    torch.testing.assert_close(loss, (expected["total"]*weight).mean())
    torch.testing.assert_close(metrics["base_loss_per_case"], expected["base"]*weight)
    torch.testing.assert_close(metrics["sobolev_spatial_loss_per_case"], expected["spatial"]*weight)
    torch.testing.assert_close(metrics["sobolev_temporal_loss_per_case"], expected["temporal"]*weight)
    torch.testing.assert_close(metrics["zero_mean_isotropic_baseline_per_case"], metrics["loss_per_case"])
    torch.testing.assert_close(metrics["base_isotropic_baseline_per_case"], metrics["base_loss_per_case"])
    torch.testing.assert_close(metrics["sigma_bin_loss_sums"].sum(), metrics["loss_per_case"].sum())
    loss.backward()
    assert model.output.weight.grad.abs().sum() > 0
    assert torch.isfinite(model.output.weight.grad).all()
    mismatched = VPFPSobolevMetric(replace(config, width=9), _normalizer(replace(config, width=9)))
    with pytest.raises(ValueError, match="config"):
        model.loss(clean, conditions, sigma=sigma, noise=noise, residual_metric=mismatched)


def test_zero_weights_are_bitwise_edm_baseline_and_sampling_is_unchanged():
    config = _config()
    model = VPFPStructuredEDM(config)
    metric = VPFPSobolevMetric(config, _normalizer(config), spatial_weight=0., temporal_weight=0.)
    clean, noise = _answers(config, dtype=torch.float32), _answers(config, dtype=torch.float32).roll(1, 0)
    conditions = torch.tensor([[.05, 1.]]*len(clean))
    sigma = torch.tensor([.01, .2, 2.])
    sampled_before = model.sample(conditions[:1], (918,), steps=2)
    old_loss, old_metrics = model.loss(clean, conditions, sigma=sigma, noise=noise)
    new_loss, new_metrics = model.loss(clean, conditions, sigma=sigma, noise=noise, residual_metric=metric)
    assert torch.equal(old_loss, new_loss)
    assert old_metrics.keys() == new_metrics.keys()
    for key in old_metrics:
        assert torch.equal(old_metrics[key], new_metrics[key])
    old_gradient = torch.autograd.grad(old_loss, model.output.weight)[0]
    new_gradient = torch.autograd.grad(new_loss, model.output.weight)[0]
    assert torch.equal(old_gradient, new_gradient)
    assert torch.equal(sampled_before, model.sample(conditions[:1], (918,), steps=2))


def test_dc_only_kinetic_operator_is_zero_without_nan_and_invalid_inputs_rejected():
    config = _config(kinetic_max_mode=0)
    metric = VPFPSobolevMetric(config, _normalizer(config))
    components = metric.components(_answers(config))
    assert torch.count_nonzero(components["kinetic_spatial"]) == 0
    assert torch.isfinite(components["total"]).all()
    bad = _answers(config)
    bad[0, config.rho_dimension] = 1.
    with pytest.raises(ValueError, match="inactive"):
        metric(bad)
    with pytest.raises(ValueError, match="float32 or float64"):
        metric(_answers(config).half())
    for kwargs in ({"spatial_weight": -1.}, {"temporal_weight": float("nan")},
                   {"spatial_weight": True}, {"spatial_lengths": (1., 0.)},
                   {"latent_input_scale": 0.},
                   {"spatial_weight": 1.e308, "temporal_weight": 1.e308}):
        with pytest.raises(ValueError):
            VPFPSobolevMetric(config, _normalizer(config), **kwargs)


def test_33_control_time_operator_uses_all_values_and_slopes_without_compression():
    config = _config(control_count=33, hermite_channels=1)
    normalizer = _normalizer(config)
    metric = VPFPSobolevMetric(config, normalizer)
    raw = torch.zeros((1, config.latent_dimension), dtype=torch.float64)
    rho, kinetic = _split(raw, config)
    tau = torch.linspace(0., 1., 33, dtype=torch.float64)
    rho[0, 0, :, 0] = tau**3
    rho[0, 1, :, 0] = 3*tau**2
    kinetic[0, 0, 0, :, 0] = tau**3
    kinetic[0, 0, 1, :, 0] = 3*tau**2
    components = metric.components(normalizer.encode(raw))
    # Exact integral_0^1 (3*tau^2)^2 dtau = 9/5. This also checks
    # all-interval endpoint assembly and derivative value/slope correlation.
    torch.testing.assert_close(components["rho_raw_temporal"], torch.tensor([2*9/5], dtype=torch.float64))
    torch.testing.assert_close(components["kinetic_raw_temporal"], torch.tensor([4/config.kinetic_modes*9/5], dtype=torch.float64))


@pytest.mark.parametrize("scale", [1.e-25, 1.e-20, 1.e20])
def test_fp64_valid_scales_cannot_silently_underflow_or_overflow_fp32_metric(scale):
    config = _config()
    normalizer = VPFPCoefficientNormalizer(config,
        torch.full((config.hermite_channels+1, 2), scale, dtype=torch.float64),
        fit_metadata={"split": "train", "generated_samples_used_for_fit": False})
    unit_normalizer = VPFPCoefficientNormalizer(config,
        torch.ones((config.hermite_channels+1, 2), dtype=torch.float64),
        fit_metadata={"split": "train", "generated_samples_used_for_fit": False})
    metric = VPFPSobolevMetric(config, normalizer)
    unit_metric = VPFPSobolevMetric(config, unit_normalizer)
    values = _answers(config)
    # The mathematical normalized metric is invariant to this common scale.
    # FP64 still evaluates it; unsupported FP32 must error, not change W.
    torch.testing.assert_close(metric(values), unit_metric(values))
    with pytest.raises(ValueError, match="finite normal torch.float32"):
        metric(values.float())


def test_callers_casting_metric_buffers_to_fp32_cannot_erase_nonzero_trace():
    config = _config()
    normalizer = VPFPCoefficientNormalizer(config,
        torch.full((config.hermite_channels+1, 2), 1.e-25, dtype=torch.float64),
        fit_metadata={"split": "train", "generated_samples_used_for_fit": False})
    metric = VPFPSobolevMetric(config, normalizer).float()
    assert metric.rho_traces[0] == 0  # The original FP64 operator was NOT DC.
    with pytest.raises(ValueError, match="rho_traces"):
        metric(_answers(config, dtype=torch.float32))


def test_finite_fp32_residual_with_overflowing_raw_energy_fails_closed():
    config = _config()
    metric = VPFPSobolevMetric(config, _normalizer(config))
    values = _answers(config, dtype=torch.float32) * 1.e20
    assert torch.isfinite(values).all()
    with pytest.raises(ValueError, match="energy overflowed torch.float32"):
        metric(values)
    assert torch.isfinite(metric(values.double())).all()


@pytest.mark.parametrize("scale", [1.e-6, 1.e-3, 1., 1.e3, 1.e6])
def test_supported_normalizer_scales_preserve_fp32_values_and_gradients(scale):
    config = _config()
    normalizer = VPFPCoefficientNormalizer(config,
        _normalizer(config).group_scales * scale,
        fit_metadata={"split": "train", "generated_samples_used_for_fit": False})
    metric = VPFPSobolevMetric(config, normalizer)
    values = _answers(config).requires_grad_(True)
    low = values.detach().float().requires_grad_(True)
    expected, actual = metric(values), metric(low)
    torch.testing.assert_close(actual.double(), expected, rtol=3.e-6, atol=3.e-7)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), low)[0].double(),
        torch.autograd.grad(expected.sum(), values)[0], rtol=3.e-5, atol=3.e-7)


def test_disabled_metric_does_not_change_edm_even_if_unused_fp32_scale_is_unsupported():
    config = _config()
    normalizer = VPFPCoefficientNormalizer(config,
        torch.full((config.hermite_channels+1, 2), 1.e20, dtype=torch.float64),
        fit_metadata={"split": "train", "generated_samples_used_for_fit": False})
    metric = VPFPSobolevMetric(config, normalizer, spatial_weight=0., temporal_weight=0.)
    model = VPFPStructuredEDM(config)
    clean, noise = _answers(config, dtype=torch.float32), _answers(config, dtype=torch.float32).roll(1, 0)
    conditions, sigma = torch.tensor([[.05, 1.]]*len(clean)), torch.tensor([.01, .2, 2.])
    original_loss, original_metrics = model.loss(clean, conditions, sigma=sigma, noise=noise)
    new_loss, new_metrics = model.loss(clean, conditions, sigma=sigma, noise=noise, residual_metric=metric)
    assert torch.equal(new_loss, original_loss)
    assert original_metrics.keys() == new_metrics.keys()
    assert all(torch.equal(new_metrics[key], value) for key, value in original_metrics.items())
    assert torch.equal(torch.autograd.grad(new_loss, model.output.weight)[0],
                       torch.autograd.grad(original_loss, model.output.weight)[0])
