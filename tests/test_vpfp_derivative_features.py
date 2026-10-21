"""Tiny real-decoder CPU checks; not trained assets or PDE validation."""

from dataclasses import replace
from hashlib import sha256
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from pde_agent_verifier_solver.plugins.vpfp import derivative_features as derivative
from pde_agent_verifier_solver.plugins.vpfp.diffusion_audit import AuditFeature


@pytest.fixture
def tiny():
    from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import _legacy_api
    api = _legacy_api()
    torch = api.torch
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    hermite = importlib.import_module("agent_verifier.vpfp.continuous_hermite")
    config = replace(hermite.VPFPContinuousHermiteConfig(), latent_dim=1,
        micro_latent_dim=2, context_dim=4, hidden_width=4, hidden_layers=1,
        rho_max_mode_x=1, rho_max_mode_y=1, rho_positivity_gauge="phase_invariant_group_l1_v2",
        spatial_feature_modes=1, time_feature_modes=1, hermite_total_degree=1,
        coefficient_width=4, coefficient_blocks=1, structured_spectral_latent=True,
        structured_kinetic_global_dim=2, structured_local_channels=3,
        structured_local_spatial_mode=1, structured_time_basis="piecewise_cubic_hermite_v1",
        structured_time_control_count=3, structured_direct_coefficients=True,
        structured_residual_scale=0., time_domain=(0., 1.), dtype="float64")
    config = replace(config, latent_dim=config.structured_latent_layout()["total_latent_dim"])
    model = api.build_deterministic_direct_model(config, device="cpu")
    latents = np.zeros((2, config.latent_dim), dtype=np.float32)
    latents[1] = np.random.default_rng(131).normal(scale=.005, size=config.latent_dim)
    conditions = np.asarray([[.04, 1.], [.06, 1.1]], dtype=np.float32)
    plan = derivative.VPFPDerivativeFeaturePlan(time_fractions=(0., .25, .75, 1.),
        macro_probe_count=2, phase_probe_count=2, chunk_size=3)
    yield SimpleNamespace(model=model, latents=latents, conditions=conditions, plan=plan, api=api)
    torch.set_num_threads(threads)


def _extract(tiny, **kwargs):
    return derivative.extract_vpfp_derivative_features(tiny.model,
        kwargs.pop("latents", tiny.latents), kwargs.pop("conditions", tiny.conditions),
        plan=kwargs.pop("plan", tiny.plan), **kwargs)


def _joined(row, descriptors, quantity, count):
    positions = {item["name"]: index for index, item in enumerate(descriptors)}
    return np.asarray([[row[positions[f"{quantity}.probe{probe}.{name}"]]
                        for name in ("value", "dt", "dx", "dy")] for probe in range(count)])


def _physical_values(model, points, latent, condition, macro):
    with model.torch.no_grad():
        fields = model.fields(points, latent, equation_condition=condition)
        if not macro:
            return {"f": fields["f"].detach().numpy()}
        macros = importlib.import_module("agent_verifier.vpfp.continuous_macros")
        moment = macros.torch_squared_hermite_moments(model.torch, fields["hermite_coefficients"],
            modes=model.hermite_modes, temperature=fields["equation_condition"][:, 1],
            floor_mixture=model.config.velocity_floor_mixture)
        rho = fields["rho"]
        return {name: value.detach().numpy() for name, value in {
            "rho": rho, "momentum_x": rho * moment["mean_velocity"][:, 0],
            "momentum_y": rho * moment["mean_velocity"][:, 1],
            "kinetic_energy": rho * moment["kinetic_energy"]}.items()}


def test_real_decoder_autograd_matches_analytic_f_and_finite_difference_macros(tiny):
    matrix, descriptors, report = _extract(tiny)
    phase = np.asarray(report["phase_points_txyvv"])
    macros = np.column_stack((report["macro_points_txy"], np.zeros((len(report["macro_points_txy"]), 2))))
    analytic = importlib.import_module("agent_verifier.vpfp.analytic_source")
    exact = analytic.manufacture_standard_vpfp_source_analytic(
        tiny.model, phase, tiny.latents[1], equation_condition=tiny.conditions[1])
    f_jet = _joined(matrix[1], descriptors, "f", len(phase))
    np.testing.assert_allclose(f_jet[:, 0], exact["f"].detach().numpy(), rtol=1.e-11, atol=1.e-12)
    np.testing.assert_allclose(f_jet[:, 1:], exact["f_gradient"][:, :3].detach().numpy(), rtol=1.e-10, atol=1.e-11)
    h = 2.e-6
    for macro, points in ((True, macros), (False, phase)):
        base = _physical_values(tiny.model, points, tiny.latents[1], tiny.conditions[1], macro)
        for axis in range(3):
            plus, minus = points.copy(), points.copy()
            plus[:, axis] += h
            minus[:, axis] -= h
            # One-sided second-order stencils at the physical time endpoints;
            # there is no out-of-domain finite difference or periodic time.
            at_start = (points[:, 0] == 0.) if axis == 0 else np.zeros(len(points), dtype=bool)
            at_end = (points[:, 0] == 1.) if axis == 0 else np.zeros(len(points), dtype=bool)
            plus[at_end, axis] = points[at_end, axis]
            minus[at_start, axis] = points[at_start, axis]
            upper = _physical_values(tiny.model, plus, tiny.latents[1], tiny.conditions[1], macro)
            lower = _physical_values(tiny.model, minus, tiny.latents[1], tiny.conditions[1], macro)
            plus2, minus2 = points.copy(), points.copy()
            plus2[at_start, axis] += 2*h
            minus2[at_end, axis] -= 2*h
            upper2 = _physical_values(tiny.model, plus2, tiny.latents[1], tiny.conditions[1], macro)
            lower2 = _physical_values(tiny.model, minus2, tiny.latents[1], tiny.conditions[1], macro)
            for name in base:
                expected = (upper[name] - lower[name]) / (2*h)
                expected[at_start] = (-3*base[name][at_start] + 4*upper[name][at_start] - upper2[name][at_start]) / (2*h)
                expected[at_end] = (3*base[name][at_end] - 4*lower[name][at_end] + lower2[name][at_end]) / (2*h)
                actual = _joined(matrix[1], descriptors, name, len(points))[:, axis+1]
                np.testing.assert_allclose(actual, expected, rtol=2.e-5, atol=2.e-7)


def test_constant_maxwellian_is_independent_of_physical_time_and_space(tiny):
    matrix, descriptors, report = _extract(tiny, latents=tiny.latents[:1], conditions=tiny.conditions[:1])
    phase = np.asarray(report["phase_points_txyvv"])
    for quantity in derivative._QUANTITIES:
        count = len(phase) if quantity == "f" else len(report["macro_points_txy"])
        jet = _joined(matrix[0], descriptors, quantity, count)
        np.testing.assert_allclose(jet[:, 1:], 0., atol=1.e-13)
        expected = {"rho": 1., "momentum_x": 0., "momentum_y": 0., "kinetic_energy": 1.}.get(quantity)
        if quantity == "f":
            expected = np.exp(-.5*np.square(phase[:, 3:]).sum(axis=1))/(2*np.pi)
        np.testing.assert_allclose(jet[:, 0], expected, atol=1.e-13)


def test_physical_time_units_scale_with_domain_length_not_diffusion_sigma(tiny):
    first, descriptors, _ = _extract(tiny, latents=tiny.latents[1:], conditions=tiny.conditions[1:])
    stretched = tiny.api.build_deterministic_direct_model(
        replace(tiny.model.config, time_domain=(2., 5.)), device="cpu")
    second, second_descriptors, report = derivative.extract_vpfp_derivative_features(
        stretched, tiny.latents[1:], tiny.conditions[1:], plan=tiny.plan)
    assert descriptors == second_descriptors
    for index, descriptor in enumerate(descriptors):
        if descriptor["group"] != "physical_value_derivative_joint":
            continue
        expected = first[0, index] / 3. if descriptor["name"].endswith(".dt") else first[0, index]
        assert second[0, index] == pytest.approx(expected, rel=1.e-10, abs=1.e-11)
        if descriptor["name"].endswith(".dt"):
            assert descriptor["unit"].endswith("/time")
    assert "physical_t" in report["time_derivative"]
    assert report["time_domain"] == [2., 5.]


def test_opposite_traveling_waves_have_equal_marginals_but_different_joint_features():
    # Synthetic functional counterexample only, not a manufactured VPFP case.
    x = np.arange(32)*2*np.pi/32
    plus = np.column_stack((3.+np.sin(x), np.cos(x), np.cos(x), np.zeros_like(x)))
    minus = np.column_stack((3.+np.sin(x), -np.cos(x), np.cos(x), np.zeros_like(x)))
    for axis in range(4):
        np.testing.assert_allclose(np.sort(plus[:, axis]), np.sort(minus[:, axis]), atol=1.e-14)
    plan = derivative.VPFPDerivativeFeaturePlan()
    first, description, joint = derivative._joint_features({key: plus for key in derivative._QUANTITIES}, plan)
    second, second_description, _ = derivative._joint_features({key: minus for key in derivative._QUANTITIES}, plan)
    assert description == second_description
    np.testing.assert_allclose(first[joint:], second[joint:], atol=1.e-14)
    assert not np.allclose(first[:joint], second[:joint])
    assert np.mean(plus[:, 1]*plus[:, 2]) > .49
    assert np.mean(minus[:, 1]*minus[:, 2]) < -.49


def test_schema_hash_snapshot_order_and_no_admission(tiny):
    matrix, descriptors, report = _extract(tiny)
    assert matrix.shape[0] == 2
    assert len(descriptors) == matrix.shape[1]
    assert len({row["name"] for row in descriptors}) == len(descriptors)
    assert all(AuditFeature(**item).scale_floor > 0 for item in descriptors)
    assert report["joint_feature_count"] == 4 * (8 + 4*8)
    assert report["joint_component_order"] == ["value", "dt", "dx", "dy"]
    assert report["statistical_unit"] == "one_complete_trajectory_per_row"
    assert not report["admitted"] and not report["high_fidelity_validation"]
    assert not report["sobolev_norm_or_distribution_equivalence_established"]
    assert not report["score_loss_integration"]
    for index, case in enumerate(report["per_case"]):
        assert case["input_answer_sha256"] == sha256(tiny.latents[index].tobytes()+tiny.conditions[index].tobytes()).hexdigest()
    json.dumps(report, allow_nan=False)
    macro_points = np.asarray(report["macro_points_txy"])
    assert set(macro_points[:, 0]) == {0., .25, .75, 1.}
    np.testing.assert_array_equal(macro_points[:2, 1:], macro_points[2:4, 1:])


def test_does_not_lose_derivatives_under_outer_no_grad_or_change_model_state(tiny):
    tiny.model.train(True)
    with tiny.model.torch.no_grad():
        matrix, descriptors, report = _extract(tiny)
    assert tiny.model.training
    assert all(parameter.grad is None for parameter in tiny.model.parameters())
    jet = _joined(matrix[1], descriptors, "rho", len(report["macro_points_txy"]))
    assert np.max(np.abs(jet[:, 1:])) > 1.e-4
    with tiny.model.torch.inference_mode(), pytest.raises(ValueError, match="inference_mode"):
        _extract(tiny)


def test_mutating_original_arrays_does_not_change_inflight_answer(tiny, monkeypatch):
    latents, conditions = tiny.latents[1:].copy(), tiny.conditions[1:].copy()
    expected, expected_names, expected_report = _extract(tiny, latents=latents, conditions=conditions)
    original = tiny.model.fields
    def mutate(*args, **kwargs):
        latents[0] += .05
        conditions[0, 0] = .03
        return original(*args, **kwargs)
    monkeypatch.setattr(tiny.model, "fields", mutate)
    actual, names, report = _extract(tiny, latents=latents, conditions=conditions)
    np.testing.assert_array_equal(actual, expected)
    assert names == expected_names
    assert report["per_case"] == expected_report["per_case"]
    assert report["per_case"][0]["input_answer_sha256"] != sha256(latents[0].tobytes()+conditions[0].tobytes()).hexdigest()


@pytest.mark.parametrize("change", [
    {"time_fractions": (0., .5, 1.)}, {"time_fractions": (1., .5, 0.)},
    {"time_fractions": (0., .25, .25, 1.)}, {"time_fractions": (0., np.nan, 1.)},
    {"time_fractions": (0., .25, 1.1)}, {"time_fractions": (.1, .25, 1.)},
    {"macro_probe_count": 0}, {"phase_probe_count": True}, {"chunk_size": 2.5},
    {"probe_seed": -1}, {"tail_quantiles": (1.,)}, {"tail_quantiles": (.99, .9)},
])
def test_invalid_and_control_node_only_plans_fail(tiny, change):
    with pytest.raises(ValueError):
        _extract(tiny, plan=replace(tiny.plan, **change))


@pytest.mark.parametrize("bad", ["nan", "float32_overflow", "condition_nan", "range", "shape", "empty", "complex"])
def test_invalid_physical_input_fails_before_evaluation(tiny, bad, monkeypatch):
    latents, conditions = tiny.latents.copy(), tiny.conditions.copy()
    if bad == "nan":
        latents[0, 0] = np.nan
    elif bad == "float32_overflow":
        latents = latents.astype(np.float64)
        latents[0, 0] = 1.e100
    elif bad == "condition_nan":
        conditions[0, 0] = np.nan
    elif bad == "range":
        conditions[0, 0] = .09
    elif bad == "shape":
        latents = latents[:, :-1]
    elif bad == "empty":
        latents, conditions = latents[:0], conditions[:0]
    elif bad == "complex":
        latents = latents.astype(complex)
    monkeypatch.setattr(tiny.model, "fields", lambda *args, **kwargs: pytest.fail("invalid input reached decoder"))
    with pytest.raises(ValueError):
        _extract(tiny, latents=latents, conditions=conditions)


def test_disconnected_graph_fails_instead_of_reporting_zero_derivatives(tiny, monkeypatch):
    original = tiny.model.fields
    def detach(*args, **kwargs):
        result = original(*args, **kwargs)
        return {key: value.detach() if hasattr(value, "detach") else value for key, value in result.items()}
    monkeypatch.setattr(tiny.model, "fields", detach)
    with pytest.raises(ValueError, match="disconnected"):
        _extract(tiny)


def test_cuda_requires_slurm_without_initializing_any_gpu(monkeypatch):
    parameter = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Slurm"):
        derivative._require_compute(parameter)
    monkeypatch.setenv("SLURM_JOB_ID", "fixture-no-real-job")
    derivative._require_compute(parameter)


def test_float32_snapshot_hash_matches_the_values_actually_evaluated(tiny):
    latents = tiny.latents[1:].astype(np.float64)
    latents += np.spacing(tiny.latents[1:]).astype(np.float64)*.125
    first, _, report = _extract(tiny, latents=latents, conditions=tiny.conditions[1:])
    second, _, _ = _extract(tiny, latents=latents.astype(np.float32), conditions=tiny.conditions[1:])
    np.testing.assert_array_equal(first, second)
    assert report["per_case"][0]["input_answer_sha256"] == sha256(
        latents[0].astype("<f4").tobytes()+tiny.conditions[1].tobytes()).hexdigest()


def test_nonfinite_gradient_rejected_even_when_point_values_are_finite(tiny, monkeypatch):
    original = tiny.model.fields
    def invalid(points, *args, **kwargs):
        fields = original(points, *args, **kwargs)
        fields["f"] = points[:, 0].sqrt()  # Finite value, infinite derivative at t=0.
        return fields
    monkeypatch.setattr(tiny.model, "fields", invalid)
    with pytest.raises(ValueError, match="nonfinite physical derivative"):
        _extract(tiny)


def test_float32_decoder_agrees_with_exact_coordinate_jet(tiny):
    model = tiny.api.build_deterministic_direct_model(replace(tiny.model.config, dtype="float32"), device="cpu")
    matrix, descriptors, report = derivative.extract_vpfp_derivative_features(
        model, tiny.latents[1:], tiny.conditions[1:], plan=tiny.plan)
    analytic = importlib.import_module("agent_verifier.vpfp.analytic_source")
    phase = np.asarray(report["phase_points_txyvv"], dtype=np.float32)
    exact = analytic.manufacture_standard_vpfp_source_analytic(
        model, phase, tiny.latents[1], equation_condition=tiny.conditions[1])
    actual = _joined(matrix[0], descriptors, "f", len(phase))
    np.testing.assert_allclose(actual[:, 0], exact["f"].detach().numpy(), atol=2.e-6, rtol=2.e-4)
    np.testing.assert_allclose(actual[:, 1:], exact["f_gradient"][:, :3].detach().numpy(), atol=2.e-6, rtol=2.e-4)
    assert report["decoder_config"]["dtype"] == "float32"
    assert "continuous_hermite" in report["physical_module_sha256"]


def test_default_plan_reaches_production_control_interval_interiors(tiny):
    config = replace(tiny.model.config, structured_time_control_count=33)
    plan = derivative.VPFPDerivativeFeaturePlan()
    plan.validate(config)
    interior = np.asarray(plan.time_fractions[1:-1])*32
    assert np.any(np.abs(interior-np.rint(interior)) > .1)
