"""Small CPU scientific-regression tests; no trained assets or Solver validation."""
from dataclasses import replace
import importlib
import inspect

import numpy as np
import pytest

from pde_agent_verifier_solver.plugins.vpfp import basic_oracle as basic
from tests.test_vpfp_derivative_features import tiny
from tests.test_vpfp_batched_source import decoder, _queries, _reference, _numpy


@pytest.mark.parametrize("kind", ["cartesian", "sparse", "sparse_shuffled_duplicates"])
@pytest.mark.parametrize("chunk", [1, 7, 8192])
def test_source_matches_pinned_analytic_for_batch_and_query_order(decoder, kind, chunk):
    points = _queries(kind)
    expected = _reference(decoder, points)
    actual = basic.evaluate_source_basic(decoder.model, points, decoder.latents[1],
        equation_condition=decoder.conditions[1], chunk_size=chunk)
    np.testing.assert_allclose(_numpy(actual), expected, rtol=decoder.rtol, atol=decoder.atol)


@pytest.mark.parametrize("field", ["rho", "g"])
@pytest.mark.parametrize("kind", ["cartesian", "sparse_shuffled_duplicates"])
def test_initial_matches_original_full_fields(decoder, field, kind):
    points = _queries(kind)
    points[:, 0] = 0.
    with decoder.model.torch.no_grad():
        expected = decoder.model.fields(points, decoder.latents[1],
            equation_condition=decoder.conditions[1])[field]
        actual = basic.evaluate_initial_basic(decoder.model, points, decoder.latents[1],
            equation_condition=decoder.conditions[1], field_name=field)
    np.testing.assert_allclose(_numpy(actual), _numpy(expected), rtol=decoder.rtol, atol=decoder.atol)


def test_initial_rho_does_not_evaluate_kinetic_or_derivative_basis(decoder, monkeypatch):
    points = _queries("sparse")
    points[:, 0] = 0.
    def forbidden(*args, **kwargs):
        raise AssertionError("rho0 must not evaluate kinetic basis or jets")
    monkeypatch.setattr(basic, "_basis", forbidden)
    monkeypatch.setattr(basic, "_spatial_basis_jet", forbidden)
    actual = basic.evaluate_initial_basic(decoder.model, points, decoder.latents[1],
        equation_condition=decoder.conditions[1], field_name="rho")
    assert actual.shape == (len(points),)


def test_initial_g_deduplicates_space_before_velocity_expansion(decoder, monkeypatch):
    points = _queries("cartesian")
    points[:, 0] = 0.
    observed = []
    original = basic._spatial_basis_jet
    def spy(torch, model, coordinates, **kwargs):
        observed.append(len(coordinates))
        assert kwargs.get("derivatives") is False
        return original(torch, model, coordinates, **kwargs)
    monkeypatch.setattr(basic, "_spatial_basis_jet", spy)
    basic.evaluate_initial_basic(decoder.model, points, decoder.latents[1],
        equation_condition=decoder.conditions[1], field_name="g")
    assert observed == [len(np.unique(points[:, 1:3], axis=0))]


@pytest.mark.parametrize("degree,spatial", [(18, 1), (1, 6)])
def test_large_mode_axis_vectorization_with_tiny_synthetic_state(tiny, degree, spatial):
    cfg = replace(tiny.model.config, hermite_total_degree=degree,
        structured_local_channels=(degree+1)*(degree+2)//2,
        structured_local_spatial_mode=spatial, latent_input_scale=1.)
    cfg = replace(cfg, latent_dim=cfg.structured_latent_layout()["total_latent_dim"])
    model = tiny.api.build_deterministic_direct_model(cfg, device="cpu")
    latent = np.random.default_rng(667).normal(size=cfg.latent_dim) * .01
    points = _queries("sparse")
    analytic = importlib.import_module("agent_verifier.vpfp.analytic_source")
    with model.torch.no_grad():
        expected = analytic.manufacture_standard_vpfp_source_analytic(
            model, points, latent, equation_condition=tiny.conditions[1])["source"]
        actual = basic.evaluate_source_basic(model, points, latent, equation_condition=tiny.conditions[1])
    np.testing.assert_allclose(_numpy(actual), _numpy(expected), rtol=2e-10, atol=2e-11)


def test_no_runtime_tensor_scalar_checks_or_finite_scans(decoder, monkeypatch):
    # Float/bool Tensor extraction would synchronize CUDA even though this
    # tiny fixture is CPU. Freeze/reference validation happens outside kernel.
    def forbidden(*args, **kwargs):
        raise AssertionError("trusted basic kernel must not extract Tensor scalars")
    torch = decoder.model.torch
    monkeypatch.setattr(torch.Tensor, "__bool__", forbidden)
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch, "isfinite", forbidden)
    points = _queries("sparse")
    result = basic.evaluate_source_basic(decoder.model, points, decoder.latents[1],
        equation_condition=decoder.conditions[1])
    assert result.shape == (len(points),)
    points[:, 0] = 0.
    for field in ("rho", "g"):
        result = basic.evaluate_initial_basic(decoder.model, points, decoder.latents[1],
            equation_condition=decoder.conditions[1], field_name=field)
        assert result.shape == (len(points),)


def test_source_matches_independent_nested_autograd_physical_residual(tiny):
    torch, model = tiny.api.torch, tiny.model
    points = np.array([[.133, .21, .34, .41, -.53], [.619, .43, .11, -.7, .2]])
    query = torch.tensor(points, dtype=torch.float64, requires_grad=True)
    fields = model.fields(query, tiny.latents[1], equation_condition=tiny.conditions[1])
    gradient = torch.autograd.grad(fields["f"].sum(), query, create_graph=True)[0]
    vx2 = torch.autograd.grad(gradient[:, 3].sum(), query, retain_graph=True)[0][:, 3]
    vy2 = torch.autograd.grad(gradient[:, 4].sum(), query)[0][:, 4]
    nu, temperature = tiny.conditions[1]
    collision = 2*fields["f"] + query[:, 3]*gradient[:, 3] + query[:, 4]*gradient[:, 4] + float(temperature)*(vx2+vy2)
    expected = (gradient[:, 0] + query[:, 3]*gradient[:, 1] + query[:, 4]*gradient[:, 2]
                + (fields["electric"]*gradient[:, 3:5]).sum(-1) - float(nu)*collision)
    actual = basic.evaluate_source_basic(model, points, tiny.latents[1], equation_condition=tiny.conditions[1])
    np.testing.assert_allclose(_numpy(actual), _numpy(expected), rtol=2e-10, atol=2e-11)


def test_zero_coefficients_keep_original_ground_state_and_floor(decoder):
    points = _queries("sparse")
    actual = basic.evaluate_source_basic(decoder.model, points, decoder.latents[0],
        equation_condition=decoder.conditions[0])
    expected = _reference(decoder, points, latent=decoder.latents[0], condition=decoder.conditions[0])
    np.testing.assert_allclose(_numpy(actual), expected, rtol=decoder.rtol, atol=decoder.atol)
    points[:, 0] = 0.
    for field in ("rho", "g"):
        expected = decoder.model.fields(points, decoder.latents[0], equation_condition=decoder.conditions[0])[field]
        actual = basic.evaluate_initial_basic(decoder.model, points, decoder.latents[0],
            equation_condition=decoder.conditions[0], field_name=field)
        np.testing.assert_allclose(_numpy(actual), _numpy(expected), rtol=decoder.rtol, atol=decoder.atol)


def test_source_only_macros_and_return_are_minimal(tiny):
    query, row, _ = basic._inputs(tiny.model, _queries("sparse"), tiny.latents[1], tiny.conditions[1])
    macro = basic._macro_jet(tiny.model, basic._tensor(tiny.model, query[:, :3]), row)
    assert set(macro) == {"rho", "rho_derivatives", "electric", "coefficients", "coefficient_derivatives"}
    assert "f_gradient" not in inspect.getsource(basic._source_from_amplitudes)


def test_supported_chebyshev_with_distinct_rho_and_local_orders(tiny):
    cfg = replace(tiny.model.config, structured_time_basis="global_chebyshev_v1",
                  structured_rho_time_order=2, structured_local_time_order=3)
    cfg = replace(cfg, latent_dim=cfg.structured_latent_layout()["total_latent_dim"])
    model = tiny.api.build_deterministic_direct_model(cfg, device="cpu")
    latent = np.random.default_rng(668).normal(scale=.01, size=cfg.latent_dim)
    points = _queries("sparse")
    analytic = importlib.import_module("agent_verifier.vpfp.analytic_source")
    expected = analytic.manufacture_standard_vpfp_source_analytic(
        model, points, latent, equation_condition=tiny.conditions[1])["source"]
    actual = basic.evaluate_source_basic(model, points, latent, equation_condition=tiny.conditions[1])
    np.testing.assert_allclose(_numpy(actual), _numpy(expected), rtol=2e-10, atol=2e-11)
    points[:, 0] = 0.
    for field in ("rho", "g"):
        expected = model.fields(points, latent, equation_condition=tiny.conditions[1])[field]
        actual = basic.evaluate_initial_basic(model, points, latent,
            equation_condition=tiny.conditions[1], field_name=field)
        np.testing.assert_allclose(_numpy(actual), _numpy(expected), rtol=2e-10, atol=2e-11)
