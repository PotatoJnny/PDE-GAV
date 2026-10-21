"""Tiny analytic fixtures only: not real-corpus physical calibration.

Tests are intended for a compute allocation. No real checkpoints, PDE solves,
training, reference fitting or generated-answer measurements are performed.
"""
from dataclasses import replace
import json
import math

import numpy as np
import pytest
import torch

from tests.test_vpfp_moving_frame import tiny, constant_macro
from pde_agent_verifier_solver.plugins.vpfp import independent_physics as ip
from pde_agent_verifier_solver.plugins.vpfp import moving_frame as mf


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def setup(*, degree=1, dtype="float64", floor=.12, u=(.4, -.3), covariance=((1.3, .2), (.2, .7))):
    base, value, config, z, condition = tiny(degree=degree, dtype=dtype, floor=floor)
    decoder = mf.MovingFrameDecoder(base, config, value_config=value)
    carrier = {"m": constant_macro(config, u=u, covariance=covariance), "z": z}
    return base, decoder, value, carrier, condition


def probe(time=3.1):
    return np.asarray([[time, .3, .7]], dtype=np.float64)


@pytest.mark.parametrize("dtype,tolerance", [("float32", 3.e-6), ("float64", 2.e-12)])
def test_native_maxwellian_moments_l2_units_and_source_zero(dtype, tolerance):
    base, _, cfg, carrier, condition = setup(dtype=dtype)
    condition[1] = 1.7
    # Change only the physical density scale, not the quadrature prescription.
    base.config = replace(base.config, equation=replace(base.config.equation, background_density=2.3))
    full = mf.lift_value_controls(carrier["z"], cfg)
    rho, theta = 2.3, float(condition[1])
    moments = ip.evaluate_velocity_moments(base, full, condition, probe(), order=4)
    expected = np.asarray([rho, 0., 0., rho*theta, rho*theta, 0., rho*theta, 0., 0.])
    np.testing.assert_allclose(moments["moments"][0], expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(moments["source_moments"], 0., rtol=0, atol=tolerance)
    norms = ip.evaluate_velocity_l2(base, full, condition, probe(), order=6)
    np.testing.assert_allclose(norms["f_l2_squared"], rho*rho/(4*math.pi*theta), rtol=tolerance)
    assert float(norms["source_l2_squared"][0]) <= tolerance*tolerance
    assert norms["quadrature"][0]["carrier_gaussian_temperature"] == theta/2


@pytest.mark.parametrize("u,covariance", [((0., 0.), ((.9, 0.), (0., .9))),
    ((.8, -.4), ((.9, 0.), (0., .9))), ((.4, -.3), ((1.3, .2), (.2, .7)))])
def test_A_shift_anisotropy_exact_source_moments_and_source_L2(u, covariance):
    base, decoder, _, carrier, condition = setup(u=u, covariance=covariance)
    rho, nu, theta = float(base.config.equation.background_density), *map(float, condition)
    u, c = np.asarray(u), np.asarray(covariance)
    k = .5*rho*(u@u+np.trace(c))
    result = ip.evaluate_velocity_moments(decoder, carrier, condition, probe(), order=4, chunk_size=7)
    np.testing.assert_allclose(result["moments"][0, :4], np.r_[rho, rho*u, k], rtol=4.e-12, atol=4.e-12)
    expected_source = np.r_[0., nu*rho*u, 2*nu*(k-theta*rho)]
    np.testing.assert_allclose(result["source_moments"][0], expected_source, rtol=2.e-10, atol=2.e-12)
    # Independent Gaussian fourth-moment identity under the normalized f² law:
    # d~N(0,C/2), collision/f = c0+l.d+d'B d.
    inverse, half = np.linalg.inv(c), c/2
    b, linear = theta*inverse@inverse-inverse, -inverse@u
    c0 = 2-theta*np.trace(inverse)
    mean_square = (c0+np.trace(b@half))**2+2*np.trace(b@half@b@half)+linear@half@linear
    f2 = rho*rho/(4*math.pi*np.sqrt(np.linalg.det(c)))
    norms = ip.evaluate_velocity_l2(decoder, carrier, condition, probe(), order=6, chunk_size=7)
    np.testing.assert_allclose(norms["f_l2_squared"], f2, rtol=5.e-12, atol=5.e-13)
    np.testing.assert_allclose(norms["source_l2_squared"], nu*nu*f2*mean_square, rtol=3.e-10, atol=3.e-13)
    assert norms["quadrature"][0]["map"] == "u+L_target*inverse(L_source)*(w-mu_source)"


def test_highest_N14_mode_has_full_space_N_plus_one_kinetic_moment():
    base, _, cfg, carrier, condition = setup(degree=14, floor=0.)
    controls = carrier["z"][:, cfg.rho_dimension:].reshape(1, 120, 3, 9)
    controls.zero_()
    mode = list(base.hermite_modes).index((14, 0))
    controls[:, mode, :, 0] = 1.
    full = mf.lift_value_controls(carrier["z"], cfg)
    result = ip.evaluate_velocity_moments(base, full, condition, probe(), order=17)
    rho, nu, theta = float(base.config.equation.background_density), *map(float, condition)
    np.testing.assert_allclose(result["moments"][0, :4], [rho, 0., 0., 15*theta*rho], rtol=8.e-12, atol=8.e-12)
    np.testing.assert_allclose(result["source_moments"][0], [0., 0., 0., 28*nu*theta*rho], rtol=3.e-10, atol=5.e-12)
    finer = ip.evaluate_velocity_moments(base, full, condition, probe(), order=19)
    np.testing.assert_allclose(result["moments"], finer["moments"], rtol=2.e-11, atol=2.e-11)
    assert result["quadrature"][0]["nominal_positive_floor_above_dtype_normal"] is None
    coarse_l2 = ip.evaluate_velocity_l2(base, full, condition, probe(), order=32)
    fine_l2 = ip.evaluate_velocity_l2(base, full, condition, probe(), order=34)
    for key in ("f_l2_squared", "source_l2_squared"):
        np.testing.assert_allclose(coarse_l2[key], fine_l2[key], rtol=3.e-10, atol=2.e-12)


def test_native_zero_amplitude_fallback_is_integrated_without_renormalizing_raw():
    base, _, cfg, carrier, condition = setup()
    carrier["z"][:, cfg.rho_dimension:] = 0.
    full = mf.lift_value_controls(carrier["z"], cfg)
    before = full.clone()
    result = ip.evaluate_velocity_moments(base, full, condition, probe(), order=4, include_source=False)
    rho, theta = base.config.equation.background_density, float(condition[1])
    np.testing.assert_allclose(result["moments"][0, :4], [rho, 0., 0., rho*theta], rtol=2.e-12, atol=2.e-12)
    torch.testing.assert_close(full, before, rtol=0, atol=0)


def test_high_Hermite_A_uses_original_carrier_not_target_gaussian():
    base, decoder, cfg, carrier, condition = setup(degree=14, floor=.27)
    controls = carrier["z"][:, cfg.rho_dimension:].reshape(1, 120, 3, 9)
    controls[:, -1, :, 0], controls[:, -2, :, 0] = .7, -.3
    out = ip.evaluate_velocity_moments(decoder, carrier, condition, probe(), order=17, include_source=False)
    rho = float(base.config.equation.background_density)
    np.testing.assert_allclose(out["moments"][0, :4], [rho, .4*rho, -.3*rho, 1.125*rho], rtol=2.e-11, atol=2.e-11)
    assert abs(out["quadrature"][0]["affine_determinant"]-np.sqrt(.87)/float(condition[1])) > .1
    coarse = ip.evaluate_velocity_l2(decoder, carrier, condition, probe(), order=32, include_source=False)
    fine = ip.evaluate_velocity_l2(decoder, carrier, condition, probe(), order=34, include_source=False)
    np.testing.assert_allclose(coarse["f_l2_squared"], fine["f_l2_squared"], rtol=2.e-10, atol=2.e-12)


def test_gauge_scale_sign_and_chunking_preserve_actual_integrals():
    base, decoder, cfg, carrier, condition = setup()
    controls = carrier["z"][:, cfg.rho_dimension:].reshape(1, 3, 3, 9)
    controls[:, 1, :, 0] = .3
    original = ip.evaluate_velocity_moments(decoder, carrier, condition, probe(), order=6, chunk_size=5)
    changed = {key: value.clone() for key, value in carrier.items()}
    changed["z"][:, cfg.rho_dimension:] *= -7.5
    again = ip.evaluate_velocity_moments(decoder, changed, condition, probe(), order=6, chunk_size=32)
    for key in ("moments", "source_moments"):
        np.testing.assert_allclose(original[key], again[key], rtol=4.e-11, atol=4.e-12)
    # Unchanged rho, entire floor and generated target moments are all included.
    assert original["quadrature"][0]["nominal_positive_floor_above_dtype_normal"]


def test_FD_one_sided_endpoints_and_physical_time_units():
    kw = {"time_domain": (2., 5.), "spatial_domain": (2*math.pi, 2*math.pi)}
    for time in (2., 3.1, 5.):
        point = np.asarray([time, .3, .7])
        actual = ip._coordinate_fd(lambda p: np.asarray([p[0]**2, 3*p[0]]), point, 0, .01, **kw)
        np.testing.assert_allclose(actual, [2*time, 3.], rtol=2.e-12, atol=2.e-12)
    # Wrapped periodic queries never leave [0,L), and remain second order.
    visited = []
    def periodic(p):
        visited.append(p.copy())
        return np.sin(p[1])
    point = np.asarray([3., .001, .7])
    derivative = ip._coordinate_fd(periodic, point, 1, .01, **kw)
    assert all(0 <= p[1] < 2*math.pi for p in visited)
    np.testing.assert_allclose(derivative, np.cos(.001), rtol=2.e-5)


def test_time_varying_A_source_moment_checks_real_time_not_control_index():
    base, decoder, _, carrier, condition = setup(u=(0., 0.), covariance=((.9, 0.), (0., .9)))
    u, local, chol = mf.split_macro_controls(carrier["m"], decoder.macro_config)
    u[:, 0] = torch.tensor([.2, .35, .5], dtype=torch.float64)
    carrier["m"] = mf.pack_macro_controls(u, local, chol, decoder.macro_config)
    result = ip.evaluate_velocity_moments(decoder, carrier, condition, probe(3.1), order=4)
    rho, nu = base.config.equation.background_density, float(condition[0])
    np.testing.assert_allclose(result["source_moments"][0], [0., rho*(.1+nu*.31), 0., rho*(.031+nu*.31**2)], rtol=2.e-10, atol=2.e-12)


def test_native_nonuniform_rho_Poisson_and_spatial_mass_ledger():
    base, _, cfg, carrier, condition = setup()
    carrier["z"][:, :cfg.rho_dimension].reshape(1, 3, 8)[:, :, 0] = .15
    full = mf.lift_value_controls(carrier["z"], cfg)
    report = ip.audit_independent_physics(base, full, condition, points_txy=probe(),
        spatial_times=[3.1], spatial_grids=(3, 5), include_l2=False)
    assert report["execution_passed"], report
    row = report["points"][0]
    coarse, fine = row["finite_difference_checks"][-2:]
    for name in ("poisson_divE_minus_integrated_rho", "poisson_minus_laplacian_phi_minus_integrated_rho"):
        assert abs(fine[name]) < abs(coarse[name])+2.e-11
    for entry in report["spatial_mass"]:
        np.testing.assert_allclose(entry["mean_f_minus_background_over_background"], 0., atol=2.e-12)
        assert entry["execution_passed"] and not entry["is_spatial_convergence_certificate"]
    assert report["independent_spatial_mass_measured"]
    assert not report["admitted"] and report["numerically_uncertain"]
    assert "full_velocity_local_source_L2" in report["pending"]
    json.dumps(report, allow_nan=False)


def test_audit_is_independent_of_source_formula_balance_and_retains_failure(monkeypatch):
    base, _, cfg, carrier, condition = setup()
    full = mf.lift_value_controls(carrier["z"], cfg)
    original = ip._Runtime.phase_values
    def corrupted(self, phase, include_source):
        if include_source and float(phase[0, 0]) > 4.:
            raise FloatingPointError("synthetic source failure retained")
        result, disagreement, dtype = original(self, phase, include_source)
        if include_source:
            result[:, 1] += .125*result[:, 0]
        return result, disagreement, dtype
    monkeypatch.setattr(ip._Runtime, "phase_values", corrupted)
    report = ip.audit_independent_physics(base, full, condition, points_txy=np.concatenate((probe(2.), probe(5.))),
        spatial_grids=(), include_l2=False)
    assert report["attempted_point_count"] == 2 and report["completed_point_count"] == 1
    assert report["failed_point_count"] == 1 and not report["execution_passed"]
    assert report["points"][1]["error"]["type"] == "FloatingPointError"
    residual = report["points"][0]["finite_difference_checks"][-1]["balance_residual"][0]
    np.testing.assert_allclose(residual, -.125*base.config.equation.background_density, rtol=1.e-10, atol=2.e-12)
    assert report["points"][0]["numerically_uncertain"]
    json.dumps(report, allow_nan=False)


def test_zero_collision_undefined_normalization_and_frozen_input_mode():
    base, _, cfg, carrier, condition = setup()
    condition[0] = 0.
    full = mf.lift_value_controls(carrier["z"], cfg)
    before, cond_before = full.clone(), condition.clone()
    base.train()
    report = ip.audit_independent_physics(base, full, condition, points_txy=probe(), spatial_grids=(), include_l2=True)
    assert report["execution_passed"], report
    row = report["points"][0]
    assert row["source_L2_relative_to_nu_f_L2"] is None
    assert row["source_L2_collision_background_normalized"] is None
    assert row["source_moments_collision_normalized"] == [None]*4
    assert row["finite_difference_checks"][0]["balance_residual_collision_normalized"] == [None]*4
    torch.testing.assert_close(full, before, rtol=0, atol=0)
    torch.testing.assert_close(condition, cond_before, rtol=0, atol=0)
    assert base.training and not report["admitted"]


def test_rejects_wrong_carrier_low_orders_and_unresolvable_condition():
    base, decoder, cfg, carrier, condition = setup()
    full = mf.lift_value_controls(carrier["z"], cfg)
    with pytest.raises(ValueError, match="complete"):
        ip.evaluate_velocity_moments(decoder, carrier["z"], condition, probe(), order=4)
    with pytest.raises(ValueError, match="below"):
        ip.evaluate_velocity_l2(base, full, condition, probe(), order=4)
    with pytest.raises(ValueError, match="below"):
        ip.evaluate_velocity_moments(base, full, condition, probe(), order=3)
    report = ip.audit_independent_physics(base, full, [-1., .9], points_txy=probe(), spatial_grids=())
    assert not report["execution_passed"] and report["stage"] == "validate"
    assert report["error"]["type"] == "ValueError"
    json.dumps(report, allow_nan=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="tiny CPU/CUDA check needs allocated GPU")
def test_tiny_native_CPU_CUDA_same_physical_nodes_and_integrals():
    base, _, cfg, carrier, condition = setup(dtype="float32")
    full = mf.lift_value_controls(carrier["z"], cfg)
    cpu = ip.evaluate_velocity_moments(base, full, condition, probe(), order=4, chunk_size=7)
    gpu = ip.evaluate_velocity_moments(base.to("cuda"), full, condition, probe(), order=4, chunk_size=7)
    np.testing.assert_allclose(cpu["moments"], gpu["moments"], rtol=5.e-6, atol=5.e-7)
    np.testing.assert_allclose(cpu["source_moments"], gpu["source_moments"], rtol=5.e-5, atol=5.e-7)
