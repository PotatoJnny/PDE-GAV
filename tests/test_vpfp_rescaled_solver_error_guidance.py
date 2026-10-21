"""Tiny synthetic arithmetic/native-path tests, not physical guidance admission.

Run through the scheduled test gate, never as login-node model evaluation.
"""
from dataclasses import FrozenInstanceError, replace
import json
import math

import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp import rescaled_solver_error_guidance as rs
from pde_agent_verifier_solver.plugins.vpfp import solver_error_guidance as old
from tests.test_vpfp_solver_error_guidance import native


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def vectors(*, dtype=torch.float64, sigma=2.):
    return (torch.full((1, 4), 4., dtype=dtype), torch.zeros((1, 4), dtype=dtype),
            torch.tensor([[1., 0., 0., 0.]], dtype=dtype), torch.tensor([sigma], dtype=dtype))


def test_relative_control_matches_drift_not_denoiser_and_preserves_sigma_squared():
    x, d, g, sigma = vectors()
    guided, record = rs.apply_rescaled_guidance(x, d, g, sigma,
        rs.RescalingPolicy(mode="relative", target_ratio=.1))
    assert record["effective_gain"] == pytest.approx(.2, rel=1e-14)
    torch.testing.assert_close(guided, torch.tensor([[.8, 0., 0., 0.]], dtype=x.dtype), rtol=1e-14, atol=1e-14)
    assert record["base_residual_rms"] == 4. and record["base_drift_rms"] == 2.
    assert record["applied_flow_ratio"] == pytest.approx(.1, rel=1e-14)
    assert record["actual_flow_ratio"] == pytest.approx(.1, rel=1e-14)
    # Reverse integration has decreasing sigma: -sigma*g in the ODE produces
    # a positive movement along g, without a second beta factor.
    base_drift, guided_drift = (x-d)/sigma[:, None], (x-guided)/sigma[:, None]
    assert guided_drift[0, 0] < base_drift[0, 0]
    assert (-.5 * (guided_drift-base_drift))[0, 0] > 0.


def test_fixed_gain_and_relative_cap_are_distinct_controls():
    x = torch.full((1, 4), 2., dtype=torch.float64)
    d, g, sigma = torch.zeros_like(x), torch.full_like(x, .2), x.new_tensor([.5])
    output, record = rs.apply_rescaled_guidance(x, d, g, sigma,
        rs.RescalingPolicy(mode="fixed", fixed_gain=3.))
    torch.testing.assert_close(output, torch.full_like(x, .15), rtol=1e-14, atol=1e-14)
    assert record["effective_gain"] == 3. and not record["flow_capped"]
    assert record["guidance_base_drift_cosine"] == pytest.approx(-1.)
    _, capped = rs.apply_rescaled_guidance(x, d, g, sigma,
        rs.RescalingPolicy(mode="fixed", fixed_gain=30., max_flow_ratio=.03))
    assert capped["flow_capped"] and not capped["gain_capped"]
    assert capped["applied_flow_ratio"] == pytest.approx(.03, rel=1e-14)


def test_requested_relative_gain_has_finite_registered_cap():
    x = torch.ones((1, 4), dtype=torch.float64)
    _, record = rs.apply_rescaled_guidance(x, torch.zeros_like(x), torch.full_like(x, 1e-9),
        x.new_tensor([1.]), rs.RescalingPolicy(mode="relative", gain_max=7.))
    assert record["gain_capped"] and record["effective_gain"] == 7.
    assert record["requested_gain"] == pytest.approx(1e8, rel=1e-13)
    assert record["applied_flow_ratio"] == pytest.approx(7e-9, rel=1e-13)
    assert not record["flow_capped"]


@pytest.mark.parametrize("kind", ["gradient", "base", "gain"])
def test_numerical_degeneracy_skips_without_inflating_base_floor(kind):
    x, d, g, sigma = vectors()
    policy = rs.RescalingPolicy(mode="fixed", fixed_gain=0. if kind == "gain" else 1.)
    if kind == "gradient": g.zero_()
    if kind == "base": d.copy_(x)
    output, record = rs.apply_rescaled_guidance(x, d, g, sigma, policy)
    assert torch.equal(output, d)
    assert record["skip_reason"] == {"gradient": "near_zero_gradient",
        "base": "near_zero_base_drift", "gain": "zero_gain"}[kind]
    assert record["effective_gain"] == 0. and record["actual_changed_coordinates"] == 0
    if kind == "base":
        assert record["base_drift_rms"] == 0. and record["actual_flow_ratio"] is None
    assert not record["numerical_floors_are_reliability_test"]


def test_fp64_statistics_retain_tiny_fp32_gradient_and_no_graph():
    x = torch.ones((1, 4), dtype=torch.float32, requires_grad=True)
    d = torch.zeros_like(x, requires_grad=True)
    g = torch.full_like(x, 1e-30, requires_grad=True)
    output, record = rs.apply_rescaled_guidance(x, d, g, x.new_tensor([1.]), rs.RescalingPolicy())
    assert record["raw_gradient_rms"] == pytest.approx(1e-30, rel=1e-7, abs=0.)
    assert record["raw_gradient_l2_fp64"] == pytest.approx(2e-30, rel=1e-7, abs=0.)
    assert torch.count_nonzero(output) == 4
    assert not output.requires_grad and all(v.grad is None for v in (x, d, g))
    json.dumps(record, allow_nan=False)


def test_fp32_actual_displacement_distinguishes_roundoff_from_proposed_update():
    x = torch.full((1, 4), 1e8+16, dtype=torch.float32)
    d = torch.full_like(x, 1e8)
    output, record = rs.apply_rescaled_guidance(x, d, torch.full_like(x, 1e-4),
        x.new_tensor([1.]), rs.RescalingPolicy(mode="fixed"))
    assert record["applied_delta_denoised_rms"] > 0.
    assert record["applied_flow_ratio"] > 0.
    assert torch.equal(output, d)
    assert record["actual_delta_denoised_rms"] == record["actual_delta_drift_rms"] == 0.
    assert record["actual_flow_ratio"] == 0. and record["actual_changed_coordinates"] == 0


@pytest.mark.parametrize("field", ["state", "denoiser", "gradient", "sigma", "random"])
def test_nonfinite_inputs_fail_closed(field):
    x, d, g, sigma = vectors()
    policy = rs.RescalingPolicy(mode="relative", random_direction=field == "random")
    direction = torch.ones_like(x)
    target = {"state": x, "denoiser": d, "gradient": g, "sigma": sigma, "random": direction}[field]
    target.flatten()[0] = float("nan")
    with pytest.raises((ValueError, FloatingPointError)):
        rs.apply_rescaled_guidance(x, d, g, sigma, policy, random_direction=direction)


def test_random_direction_matches_learned_amplitude_and_scalar_gain_only():
    x, d, g, sigma = vectors()
    policy = rs.RescalingPolicy(mode="relative", target_ratio=.03)
    learned, a = rs.apply_rescaled_guidance(x, d, g, sigma, policy)
    random, b = rs.apply_rescaled_guidance(x, d, g, sigma, replace(policy, random_direction=True),
        random_direction=x.new_tensor([[0., 3., -4., 0.]]))
    assert not torch.equal(learned, random)
    assert a["effective_gain"] == b["effective_gain"]
    assert b["applied_delta_denoised_rms"] == pytest.approx(a["applied_delta_denoised_rms"], rel=1e-14)
    assert b["applied_flow_ratio"] == pytest.approx(.03, rel=1e-14)


@pytest.mark.parametrize("kind", ["gain_cap", "flow_cap", "zero_gradient", "outside_window"])
def test_random_same_state_exponential_score_matches_caps_masks_and_fp32_amplitude(kind):
    x = torch.linspace(1., 2., 16, dtype=torch.float32)[None].requires_grad_(True)
    probability = x[:, 0].sigmoid()
    log_h = ((1.-probability)*math.exp(2.*(.1-.3))
             + probability*math.exp(2.*(.7-.3))).log()
    gradient = torch.autograd.grad(log_h.sum(), x)[0]
    if kind == "zero_gradient": gradient.zero_()
    policy = (rs.RescalingPolicy(mode="fixed", fixed_gain=100., max_flow_ratio=.03)
        if kind == "flow_cap" else rs.RescalingPolicy(mode="relative", gain_max=.3))
    baseline, sigma = torch.zeros_like(x), x.new_tensor([.5])
    enabled = kind != "outside_window"
    _, learned = rs.apply_rescaled_guidance(x, baseline, gradient, sigma, policy, enabled=enabled)
    _, random = rs.apply_rescaled_guidance(x, baseline, gradient, sigma,
        replace(policy, random_direction=True), enabled=enabled,
        random_direction=torch.cos(torch.arange(16, dtype=x.dtype))[None])
    for key in ("effective_gain", "gain_capped", "flow_capped", "gradient_clipped", "skip_reason"):
        assert random[key] == learned[key]
    for key in ("applied_flow_ratio", "actual_flow_ratio", "applied_delta_denoised_rms"):
        assert random[key] == pytest.approx(learned[key], rel=2e-5, abs=2e-7)
    if kind == "gain_cap": assert learned["gain_capped"]
    if kind == "flow_cap": assert learned["flow_capped"]


@pytest.mark.parametrize("kwargs", [{"mode": "unknown"}, {"gain_max": float("inf")},
    {"fixed_gain": -1.}, {"target_ratio": 0.}, {"minimum_gradient_rms": 0.},
    {"random_direction": 1}, {"random_seed": True}, {"random_direction": True},
    {"fixed_gain": 2.}])
def test_policy_rejects_unregistered_or_ambiguous_settings(kwargs):
    with pytest.raises(ValueError): rs.RescalingPolicy(**kwargs)


def test_policy_frozen_json_roundtrip():
    policy = rs.RescalingPolicy(mode="relative", random_direction=True)
    assert rs.RescalingPolicy.from_dict(json.loads(json.dumps(policy.to_dict()))) == policy
    with pytest.raises(FrozenInstanceError): policy.gain_max = 10.
    with pytest.raises(ValueError): rs.RescalingPolicy.from_dict({"mode": "relative"})


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_natural_correction_is_exact_original_native_arithmetic(dtype):
    x, d, g, sigma = vectors(dtype=dtype, sigma=.13)
    g *= 7.123
    norm = torch.linalg.vector_norm(g, dim=1, keepdim=True)
    scale = torch.where(norm > .7, .7/norm.clamp_min(torch.finfo(norm.dtype).tiny), torch.ones_like(norm))
    expected = d + 1. * sigma[:, None].square() * (g.detach()*scale)
    actual, record = rs.apply_rescaled_guidance(x, d, g, sigma, rs.RescalingPolicy(), max_gradient_norm=.7)
    assert torch.equal(actual, expected) and record["gradient_clipped"]


@pytest.mark.parametrize("churn", [0., .1])
def test_natural_is_bitwise_old_sampler_and_reuses_native_heun(monkeypatch, churn):
    base, guide, context, _, theta, _ = native(monkeypatch, churn=churn)
    sampler = old.GuidedIC40Sampler(base, guide, strength=1., sigma_min=.002, sigma_max=2.)
    expected = sampler.sample_future(context, theta, [761, 762])
    original = old.VPFPStructuredEDM.sample
    seen = []
    def capture(view, conditions, seeds, **kwargs):
        seen.append((tuple(seeds), kwargs))
        return original(view, conditions, seeds, **kwargs)
    monkeypatch.setattr(old.VPFPStructuredEDM, "sample", capture)
    scaled = rs.RescaledIC40Sampler(base, guide, rs.RescalingPolicy())
    actual = scaled.sample_future(context, theta, [761, 762])
    assert torch.equal(actual, expected)
    assert [x[0] for x in seen] == [(761,), (762,)]
    assert all(x[1] == {"steps": base.steps, "churn": churn} for x in seen)
    report = scaled.last_report
    assert report["denoise_calls"] == 2 * (2*base.steps-1) == len(report["records"])
    assert report["guided_calls"] == sampler.last_report["guided_calls"]
    assert [x["heun_stage"] for x in report["records"][:5]] == ["predictor", "corrector", "predictor", "corrector", "predictor"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("random_direction", [False, True])
def test_root_local_streams_preserve_single_batch_order_rng_and_context(monkeypatch, random_direction):
    base, guide, context, _, theta, _ = native(monkeypatch, churn=.1)
    saved_context = context.clone()
    base_sha = old._module_sha(base.model)
    guide_sha = old._module_sha(guide)
    scaled = rs.RescaledIC40Sampler(base, guide,
        rs.RescalingPolicy(mode="relative", random_direction=random_direction))
    rng = torch.get_rng_state().clone()
    grouped = scaled.sample_future(context, theta, [661, 662])
    grouped_records = scaled.last_report["records"]
    separate = torch.cat([scaled.sample_future(context[i:i+1], theta[i:i+1], [seed])
        for i, seed in enumerate([661, 662])])
    reverse = scaled.sample_future(context.flip(0), theta.flip(0), [662, 661]).flip(0)
    assert torch.equal(grouped, separate) and torch.equal(grouped, reverse)
    assert torch.equal(torch.get_rng_state(), rng) and torch.equal(context, saved_context)
    assert old._module_sha(base.model) == base_sha and old._module_sha(guide) == guide_sha
    assert all(p.grad is None for p in base.model.parameters())
    assert all(p.grad is None for p in guide.parameters())
    assert len(grouped_records) == 10
    assert not scaled.identity["guidance"]["physical_percentage_claimed"]


def test_sigma_window_skips_guide_forward_and_returns_base_for_those_calls(monkeypatch):
    base, guide, context, _, theta, _ = native(monkeypatch)
    calls = []
    handle = guide.register_forward_hook(lambda _m, _args, _result: calls.append(float(_args[1][0])))
    try:
        scaled = rs.RescaledIC40Sampler(base, guide, rs.RescalingPolicy(mode="relative"),
            sigma_min=.01, sigma_max=1.)
        scaled.sample_future(context[:1], theta[:1], [191])
    finally:
        handle.remove()
    records = scaled.last_report["records"]
    active = [x for x in records if .01 <= x["sigma"] <= 1.]
    skipped = [x for x in records if not .01 <= x["sigma"] <= 1.]
    assert len(calls) == len(active) > 0 and skipped
    assert all(x["skip_reason"] == "outside_sigma_window" and x["actual_changed_coordinates"] == 0 for x in skipped)
    with torch.inference_mode(), pytest.raises(ValueError, match="inference_mode"):
        scaled.sample_future(context[:1], theta[:1], [191])


@pytest.mark.parametrize("tamper", ["policy", "policy_bypass", "guide", "base", "normalizer", "window", "identity"])
def test_frozen_policy_guide_and_base_are_bound(monkeypatch, tamper):
    base, guide, _, _, _, _ = native(monkeypatch)
    scaled = rs.RescaledIC40Sampler(base, guide, rs.RescalingPolicy(mode="relative"))
    if tamper == "policy": scaled.policy = replace(scaled.policy, target_ratio=.03)
    if tamper == "policy_bypass": object.__setattr__(scaled.policy, "gain_max", 10.)
    if tamper == "guide": next(guide.parameters()).data.add_(.01)
    if tamper == "base": next(base.model.parameters()).data.add_(.01)
    if tamper == "normalizer": base.normalizer.regression.data.add_(.01)
    if tamper == "window": scaled.sigma_max = 1.
    if tamper == "identity": scaled.identity["guidance"]["rescaling_policy"]["gain_max"] = 1.
    with pytest.raises(ValueError): scaled.assert_frozen()


class RawPotentialGuide(old.SolverErrorRegressor):
    expected_sampling_strength = 1.
    beta = 2.

    def forward(self, noisy_future_white, sigma, conditions, *, initial_context):
        return {"log_desirability": .001 * noisy_future_white[:, 0]}

    def score_potential(self, prediction):
        return prediction["log_desirability"]

    def runtime_settings(self):
        return {"objective": "synthetic_raw_potential", "beta": 2.,
                "expected_sampling_strength": self.expected_sampling_strength}


def test_explicit_sampling_gain_does_not_change_target_beta_or_old_unit_policy(monkeypatch):
    base, old_guide, context, _, theta, _ = native(monkeypatch)
    guide = RawPotentialGuide(old_guide.config)
    with pytest.raises(ValueError, match="external strength"):
        old.GuidedIC40Sampler(base, guide, strength=2.)
    scaled = rs.RescaledIC40Sampler(base, guide,
        rs.RescalingPolicy(mode="fixed", fixed_gain=4., max_flow_ratio=100.))
    output = scaled.sample_future(context[:1], theta[:1], [182])
    assert torch.isfinite(output).all() and scaled.strength == guide.expected_sampling_strength == 1.
    active = [x for x in scaled.last_report["records"] if x["skip_reason"] is None]
    assert active and all(x["effective_gain"] == 4. for x in active)
    assert all(x["raw_gradient_l2_fp64"] == pytest.approx(.001, rel=1e-7) for x in active)
    assert scaled.identity["guidance"]["gain_semantics"] == "sampling_control_not_target_beta"
