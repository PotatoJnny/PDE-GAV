"""Small synthetic covariance tests, never a real-data distribution audit."""

from copy import deepcopy
import json

import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import VPFPStructuredEDMConfig
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import VPFPValueEDMConfig, _value_split
from pde_agent_verifier_solver.plugins.vpfp.noise_geometry import VPFPTemporalCovarianceTransform as Transform, _dct


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config(time=4):
    return VPFPValueEDMConfig.from_full_config(VPFPStructuredEDMConfig(
        control_count=time, hermite_channels=2, rho_max_mode=1, kinetic_max_mode=1,
        inactive_dim=2, grid_size=3, width=8, depth=1))


def values(cfg, count=12, dtype=torch.float64):
    return torch.randn(count, cfg.latent_dimension, dtype=dtype,
                       generator=torch.Generator().manual_seed(727))


def fit(cfg, data=None, **kwargs):
    data = values(cfg) if data is None else data
    return Transform.fit(cfg, data, sample_ids=list(range(len(data))), **kwargs)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("center", (False, True))
def test_encode_decode_are_full_rank_inverses_and_differentiable(dtype, center):
    cfg = config()
    original = values(cfg, dtype=dtype)
    _, kinetic = _value_split(original, cfg)
    kinetic[..., 0] += 8.
    transform = fit(cfg, original, center_kinetic_dc=center).to("cpu")
    input_values = values(cfg, 5, dtype).requires_grad_()
    decoded = transform.decode(transform.encode(input_values))
    torch.testing.assert_close(decoded, input_values, rtol=2.e-6 if dtype == torch.float32 else 1.e-12,
                               atol=2.e-6 if dtype == torch.float32 else 1.e-12)
    assert decoded.dtype == dtype
    decoded.square().sum().backward()
    assert input_values.grad is not None and torch.isfinite(input_values.grad).all()
    assert transform.report()["full_rank"] is True
    assert transform.report()["subspace_or_frequency_removal"] is False
    assert not tuple(transform.parameters())


def test_uncentered_second_moments_match_direct_definition():
    cfg = config()
    data = values(cfg)
    rho, kinetic = _value_split(data, cfg)
    transform = fit(cfg, data, batch_size=3)
    torch.testing.assert_close(transform.second_moment[0], torch.einsum("btm,bsm->ts", rho, rho)/(len(data)*cfg.rho_modes))
    torch.testing.assert_close(transform.second_moment[1:], torch.einsum("bhtm,bhsm->hts", kinetic, kinetic)/(len(data)*cfg.kinetic_modes))
    torch.testing.assert_close(transform.raw_second_moment, transform.second_moment)
    assert torch.count_nonzero(transform.kinetic_dc_mean) == 0


def test_only_kinetic_dc_mean_is_subtracted():
    cfg = config()
    data = values(cfg)
    rho, kinetic = _value_split(data, cfg)
    rho += 4.
    kinetic[..., 0] += 7.
    kinetic[..., 1] += 3.
    transform = fit(cfg, data, center_kinetic_dc=True)
    expected_mean = kinetic[..., 0].mean(0)
    torch.testing.assert_close(transform.kinetic_dc_mean, expected_mean)
    centered = kinetic.clone()
    centered[..., 0] -= expected_mean[None]
    torch.testing.assert_close(transform.second_moment[0], torch.einsum("btm,bsm->ts", rho, rho)/(len(data)*cfg.rho_modes))
    torch.testing.assert_close(transform.second_moment[1:], torch.einsum("bhtm,bhsm->hts", centered, centered)/(len(data)*cfg.kinetic_modes))
    mean_vector = transform.decode(torch.zeros_like(data))
    mean_rho, mean_micro = _value_split(mean_vector, cfg)
    assert torch.count_nonzero(mean_rho) == 0
    assert torch.count_nonzero(mean_micro[..., 1:]) == 0
    torch.testing.assert_close(mean_micro[..., 0], expected_mean[None].expand(len(data), -1, -1))


def test_rank_deficient_and_zero_training_covariance_get_positive_floor():
    cfg = config()
    data = torch.zeros(3, cfg.latent_dimension, dtype=torch.float64)
    rho, kinetic = _value_split(data, cfg)
    rho[:] = 2.
    transform = fit(cfg, data, relative_floor=.1, absolute_floor=.01)
    expected_floor = (.1**2 * transform.second_moment.diagonal(dim1=-2, dim2=-1).mean(-1)).clamp_min(.01**2)
    torch.testing.assert_close(transform.eigenvalue_floor, expected_floor)
    assert torch.all(transform.regularized_eigenvalues > 0)
    assert torch.all(transform.regularized_eigenvalues >= expected_floor[:, None])
    torch.testing.assert_close(transform.decode(transform.encode(values(cfg))), values(cfg), rtol=1.e-11, atol=1.e-11)
    json.dumps(transform.report(), allow_nan=False)


def test_whitening_covariance_trace_and_inverse_noise_sampling():
    cfg = config()
    transform = fit(cfg, center_kinetic_dc=True)
    report = transform.report()
    torch.testing.assert_close(torch.tensor(report["whitened_regularized_noise_trace"]), torch.full((3,), 4.))
    assert report["factor_inverse_max_abs_error"] < 1.e-12
    noise = values(cfg, 1000)
    restored = transform.encode(transform.decode(noise))
    torch.testing.assert_close(restored, noise, rtol=1.e-12, atol=1.e-12)
    assert abs(float(restored.square().mean()) - 1.) < .02


def rotate_pairs(data, cfg):
    rho, kinetic = _value_split(data.clone(), cfg)
    cosine, sine = .6, .8
    for block, start in ((rho, 0), (kinetic, 1)):
        left = block[..., start::2].clone()
        right = block[..., start+1::2].clone()
        block[..., start::2] = cosine*left - sine*right
        block[..., start+1::2] = sine*left + cosine*right
    return torch.cat((rho.flatten(1), kinetic.flatten(1)), dim=1)


def test_translation_pair_preservation_in_fit_and_transform():
    cfg = config()
    data = values(cfg)
    transform = fit(cfg, data, center_kinetic_dc=True)
    translated = fit(cfg, rotate_pairs(data, cfg), center_kinetic_dc=True)
    torch.testing.assert_close(transform.second_moment, translated.second_moment, rtol=1.e-12, atol=1.e-12)
    torch.testing.assert_close(transform.kinetic_dc_mean, translated.kinetic_dc_mean, rtol=0, atol=0)
    target = values(cfg, 3)
    torch.testing.assert_close(transform.encode(rotate_pairs(target, cfg)), rotate_pairs(transform.encode(target), cfg), rtol=1.e-12, atol=1.e-12)
    torch.testing.assert_close(transform.posterior_mean(rotate_pairs(target, cfg), .2), rotate_pairs(transform.posterior_mean(target, .2), cfg), rtol=1.e-12, atol=1.e-12)


def test_dct_and_time_rms_statistics_satisfy_parseval():
    cfg = config()
    transform = fit(cfg)
    report = transform.report()
    for label, moment in (("raw", transform.raw_second_moment), ("centered", transform.second_moment)):
        spectrum = torch.tensor(report[label]["dct_mean_energy"], dtype=torch.float64)
        rms = torch.tensor(report[label]["time_rms"], dtype=torch.float64)
        torch.testing.assert_close(spectrum.sum(-1), moment.diagonal(dim1=-2, dim2=-1).sum(-1))
        torch.testing.assert_close(rms.square(), moment.diagonal(dim1=-2, dim2=-1))
    basis = _dct(4, device="cpu")
    torch.testing.assert_close(basis @ basis.T, torch.eye(4, dtype=torch.float64))


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_gaussian_posterior_matches_analytic_diagonal(dtype):
    cfg = config()
    template = fit(cfg, center_kinetic_dc=True).to_payload()
    diagonal = torch.tensor([[.25, 1., 4., 9.], [1., 2., 3., 4.], [4., 3., 2., 1.]], dtype=torch.float64)
    template["second_moment"] = torch.diag_embed(diagonal).tolist()
    template["raw_second_moment"] = deepcopy(template["second_moment"])
    transform = Transform.from_payload(template)
    noisy = values(cfg, 5, dtype)
    sigma = torch.tensor([0., .1, .5, 2., 8.], dtype=dtype)
    prior_mean = transform.decode(torch.zeros_like(noisy))
    rho, kinetic = _value_split(noisy - prior_mean, cfg)
    weights = diagonal[None] / (diagonal[None] + sigma.double()[:, None, None].square())
    expected = torch.cat(((rho * weights[:, 0, :, None]).flatten(1),
                          (kinetic * weights[:, 1:, :, None]).flatten(1)), dim=1) + prior_mean
    torch.testing.assert_close(transform.posterior_mean(noisy, sigma), expected.to(dtype), rtol=2.e-6 if dtype == torch.float32 else 1.e-12,
                               atol=2.e-6 if dtype == torch.float32 else 1.e-12)


def test_gaussian_posterior_sigma_zero_and_large_limit():
    cfg = config()
    transform = fit(cfg, center_kinetic_dc=True)
    noisy = values(cfg, 3)
    torch.testing.assert_close(transform.posterior_mean(noisy, 0.), noisy, rtol=1.e-12, atol=1.e-12)
    torch.testing.assert_close(transform.posterior_mean(noisy, 1.e200), transform.decode(torch.zeros_like(noisy)), rtol=0, atol=0)


def test_gaussian_posterior_matches_dense_covariance_solve():
    cfg = config()
    transform = fit(cfg, center_kinetic_dc=True)
    noisy = values(cfg, 3)
    sigma = torch.tensor([.1, .8, 3.], dtype=torch.float64)
    mean = transform.decode(torch.zeros_like(noisy))
    rho, kinetic = _value_split(noisy - mean, cfg)
    covariance = transform.square_root @ transform.square_root.transpose(-1, -2)
    expected_rho, expected_kinetic = torch.empty_like(rho), torch.empty_like(kinetic)
    for case in range(len(noisy)):
        for group, inputs, output in [(0, rho[case], expected_rho[case]),
                *((channel+1, kinetic[case, channel], expected_kinetic[case, channel])
                  for channel in range(cfg.hermite_channels))]:
            output.copy_(covariance[group] @ torch.linalg.solve(
                covariance[group] + sigma[case].square()*torch.eye(cfg.control_count, dtype=torch.float64), inputs))
    expected = torch.cat((expected_rho.flatten(1), expected_kinetic.flatten(1)), dim=1) + mean
    torch.testing.assert_close(transform.posterior_mean(noisy, sigma), expected, rtol=1.e-12, atol=1.e-12)


def test_fitting_is_detached_and_does_not_mutate_targets():
    cfg = config()
    data = values(cfg).requires_grad_()
    before = data.detach().clone()
    transform = fit(cfg, data, center_kinetic_dc=True)
    torch.testing.assert_close(data, before, rtol=0, atol=0)
    assert all(not value.requires_grad for value in transform.buffers())
    assert data.grad is None


def test_payload_roundtrip_and_fixed_metadata():
    cfg = config()
    transform = fit(cfg, center_kinetic_dc=True)
    payload = json.loads(json.dumps(transform.to_payload(), allow_nan=False))
    restored = Transform.from_payload(payload)
    torch.testing.assert_close(transform.encode(values(cfg)), restored.encode(values(cfg)), rtol=0, atol=0)
    assert restored.report()["fit_metadata"]["sample_ids"] == list(range(12))
    assert restored.report()["fit_metadata"]["generated_samples_used_for_fit"] is False
    assert restored.report()["fit_metadata"]["reference_samples_used_for_fit"] is False
    assert restored.report()["native_sampler_asset"] is False


@pytest.mark.parametrize("kwargs", [
    {"split": "validation"}, {"batch_size": 9}, {"batch_size": True},
    {"center_kinetic_dc": 1}, {"relative_floor": 0.}, {"relative_floor": 2.},
    {"absolute_floor": 0.}, {"absolute_floor": float("inf")},
])
def test_invalid_fit_settings_rejected(kwargs):
    cfg = config()
    with pytest.raises(ValueError):
        fit(cfg, **kwargs)


@pytest.mark.parametrize("sigma", [-1., float("nan"), [1., 2.]])
def test_invalid_posterior_noise_levels_rejected(sigma):
    cfg = config()
    with pytest.raises(ValueError, match="sigma"):
        fit(cfg).posterior_mean(values(cfg), sigma)


def test_duplicate_ids_or_wrong_dimension_or_nonfinite_cannot_fit():
    cfg = config()
    with pytest.raises(ValueError, match="identities"):
        Transform.fit(cfg, values(cfg), sample_ids=[0]*12)
    with pytest.raises(ValueError):
        fit(cfg, values(cfg)[:, :-1])
    bad = values(cfg)
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError):
        fit(cfg, bad)


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(schema="native_sampler_asset"),
    lambda p: p.update(unexpected=1),
    lambda p: p["fit_metadata"].update(split="reference_a"),
    lambda p: p["fit_metadata"].update(generated_samples_used_for_fit=True),
    lambda p: p["second_moment"][0][0].__setitem__(0, -100.),
    lambda p: p["second_moment"][0][0].__setitem__(1, 100.),
])
def test_bad_payload_rejected(mutation):
    payload = fit(config()).to_payload()
    mutation(payload)
    with pytest.raises(ValueError):
        Transform.from_payload(payload)
