"""Synthetic objective/ABI proofs, not evidence of harder real VPFP cases."""

from copy import deepcopy
from hashlib import sha256
import json
import math

import pytest
import torch
from torch.nn import functional as F

from pde_agent_verifier_solver.plugins.vpfp import calibrated_solver_error_guidance as calibrated
from pde_agent_verifier_solver.plugins.vpfp import exponential_solver_error_guidance as exponential
from pde_agent_verifier_solver.plugins.vpfp import solver_error_guidance as legacy
from tests.test_vpfp_calibrated_solver_error_guidance import _config, _rows


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _make(*, dtype=torch.float64):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(982)
        return exponential.make_exponential_regressor(_config(), parameter_dtype=dtype, beta=2.)


@pytest.fixture(scope="module")
def fitted():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = _make()
        rows = _rows(model.config)
        exponential.fit_exponential_calibration(model, rows, seed=23456)
        return model.eval(), rows
    finally:
        torch.set_num_threads(old)


def _payload(model, rows):
    return exponential.exponential_payload(model, label_manifest_sha256="a" * 64,
        train_root_ids=rows["root_ids"], validation_root_ids=[f"validation-{i:04d}" for i in range(16)],
        training={"label_source": "synthetic_fixture", "step": 0})


def test_positive_regression_optimum_is_mean_exp_not_exp_mean_reward():
    rewards = torch.tensor([.1, .7], dtype=torch.float64)
    errors = (rewards / (1. - rewards)).expm1()
    weights, _ = exponential.exponential_targets(errors, torch.ones(2, dtype=torch.bool),
        prototype=rewards, reward_center=.4, beta=2.)
    torch.testing.assert_close(weights, (2. * (rewards - .4)).exp(), rtol=1e-14, atol=1e-14)
    optimum = weights.mean().log().detach().requires_grad_(True)
    objective = exponential.desirability_mse(optimum.expand(2), weights)
    first = torch.autograd.grad(objective, optimum, create_graph=True)[0]
    second = torch.autograd.grad(first, optimum)[0]
    assert abs(float(first)) < 1e-14 and second > 0.
    assert second == pytest.approx(float(2. * weights.mean().square()), rel=1e-13)
    # The mean-reward plug-in is a different estimator when conditional reward
    # has variance. At that value, the implemented loss still pushes h upward.
    plug_in = (2. * (rewards.mean() - .4)).detach().requires_grad_(True)
    plug_gradient = torch.autograd.grad(exponential.desirability_mse(plug_in.expand(2), weights), plug_in)[0]
    assert plug_gradient < 0.
    assert float(optimum.exp()) > float(plug_in.exp())


def test_joint_target_includes_invalid_zero_without_nonlinear_nan_arithmetic():
    errors = torch.tensor([.1, 1., float("nan"), float("inf"), -1.], dtype=torch.float64)
    valid = torch.tensor([True, True, False, False, False])
    weights, mask = exponential.exponential_targets(errors, valid, prototype=errors, reward_center=.2)
    assert torch.equal(mask, valid) and torch.isfinite(weights).all()
    assert torch.count_nonzero(weights[~valid]) == 0 and torch.all(weights[valid] > 0.)
    assert not weights.requires_grad
    # The joint optimum is P(valid) times the valid-conditional expectation.
    assert weights.mean() == pytest.approx(float(weights[valid].mean() * (2. / 5.)))
    assert weights.mean() < weights[valid].mean()
    all_invalid, _ = exponential.exponential_targets([float("nan"), float("inf")], [False, False],
        prototype=errors, reward_center=.2)
    assert torch.equal(all_invalid, torch.zeros(2, dtype=errors.dtype))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        exponential.exponential_targets([float("nan")], [True], prototype=errors, reward_center=.2)


def test_raw_score_has_no_sigmoid_validity_multiplier_or_second_beta():
    model = _make()
    state = torch.tensor([.3, -.2], dtype=torch.float64, requires_grad=True)
    log_h = 3. * state + .2
    pred = {"log_desirability": log_h, "desirability": log_h.exp(),
            "valid_probability": torch.tensor([.01, .99], dtype=torch.float64)}
    assert model.score_potential(pred) is log_h
    derivative = torch.autograd.grad(model.score_potential(pred).sum(), state)[0]
    assert torch.equal(derivative, torch.full_like(state, 3.))
    pred["valid_probability"] = torch.zeros(2, dtype=torch.float64)
    assert model.score_potential(pred) is log_h
    settings = model.runtime_settings()
    assert settings["beta"] == 2. and settings["beta_applied_in_target_only"]
    assert settings["expected_sampling_strength"] == model.expected_sampling_strength == 1.
    model.expected_sampling_strength = 2.
    with pytest.raises(ValueError, match="already in the target"):
        model.score_potential(pred)


def test_fixed_center_shift_changes_log_h_constant_but_not_exact_score():
    model = _make()
    state = torch.tensor([-.8, .2], dtype=torch.float64, requires_grad=True)
    probability = state.sigmoid()
    logits, gradients = [], []
    for center in (.1, .7):
        h = ((1. - probability) * math.exp(2. * (.2 - center))
             + probability * math.exp(2. * (.8 - center)))
        potential = model.score_potential({"log_desirability": h.log()})
        logits.append(potential)
        gradients.append(torch.autograd.grad(potential.sum(), state, retain_graph=True)[0])
    torch.testing.assert_close(logits[1] - logits[0], torch.full_like(state, -1.2), rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=1e-14, atol=1e-14)
    assert torch.all(gradients[0] > 0.)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_constructor_preserves_combined_initial_parameter_rng_and_precision(dtype):
    torch.manual_seed(451)
    reference = calibrated.make_calibrated_regressor(_config(), parameter_dtype=dtype,
        scale_enabled=True, init_enabled=True)
    state = torch.get_rng_state().clone()
    torch.manual_seed(451)
    model = exponential.make_exponential_regressor(_config(), parameter_dtype=dtype)
    assert torch.equal(torch.get_rng_state(), state)
    assert isinstance(model, legacy.SolverErrorRegressor)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in reference.state_dict().items())
    assert all(parameter.dtype == dtype for parameter in model.parameters())
    assert model.transform.rho_basis.dtype == torch.float64


@pytest.mark.parametrize("beta", [0., 1., True, float("nan")])
def test_unregistered_beta_is_rejected(beta):
    with pytest.raises(ValueError, match="beta=2"):
        exponential.make_exponential_regressor(_config(), parameter_dtype=torch.float64, beta=beta)


def test_fit_reuses_v2_combined_scales_and_small_weights_without_rng_or_data_mutation(fitted):
    model, rows = fitted
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(982)
        reference = calibrated.make_calibrated_regressor(model.config, parameter_dtype=torch.float64,
            scale_enabled=True, init_enabled=True)
    calibrated.fit_calibration(reference, rows, seed=23456)
    for name, parameter in model.named_parameters():
        expected = dict(reference.named_parameters())[name]
        if name == "head.2.bias":
            assert torch.equal(parameter[1], expected[1])
        else:
            assert torch.equal(parameter, expected)
    assert model.calibration_metadata == reference.calibration_metadata
    for name in calibrated.SCALE_NAMES:
        assert torch.equal(getattr(model, name), getattr(reference, name))
    weights, _ = exponential.exponential_targets(rows["errors"], rows["valid"], prototype=rows["future_white"],
        reward_center=model.reward_center, beta=model.beta)
    assert model.exponential_metadata["fit_desirability_mean"] == float(weights.double().mean())
    assert model.head[-1].bias[0] == math.log(float(weights.double().mean()))
    assert torch.count_nonzero(model.head[-1].weight[0]) and torch.count_nonzero(model.head[-1].weight[1])
    fresh = _make()
    original = deepcopy(rows)
    rng_before = torch.get_rng_state().clone()
    metadata = exponential.fit_exponential_calibration(fresh, rows, seed=23456)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert metadata == model.exponential_metadata
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in fresh.state_dict().items())
    for key in ("future_white", "initial_context", "conditions", "valid"):
        assert torch.equal(rows[key], original[key])
    torch.testing.assert_close(rows["errors"], original["errors"], equal_nan=True, rtol=0, atol=0)
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize("change", ["validation", "duplicate", "missing"])
def test_fit_isolates_all_64_registered_training_roots(change):
    model = _make()
    rows = _rows(model.config)
    if change == "validation":
        rows["root_ids"][0] = "validation-0000"
    elif change == "duplicate":
        rows["root_ids"][0] = rows["root_ids"][1]
    else:
        rows["root_ids"].pop()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="fit"):
        exponential.fit_exponential_calibration(model, rows, seed=4)
    assert model.reward_center is None and model.exponential_metadata is None
    assert model.calibration_metadata is None
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


def test_actual_loss_includes_invalid_desirability_and_retains_auxiliary_bce(fitted):
    frozen, rows = fitted
    model = deepcopy(frozen).train()
    x, initial, theta = (rows[key][:2] for key in ("future_white", "initial_context", "conditions"))
    sigma = torch.ones(2, dtype=x.dtype)
    errors, valid = [float("nan"), float("inf")], [False, False]
    prediction = model(x, sigma, theta, initial_context=initial)
    loss, metrics = model.loss(x, theta, initial, errors, valid, sigma=sigma, noise=torch.zeros_like(x))
    expected = prediction["desirability"].square().mean()
    expected_validity = F.binary_cross_entropy_with_logits(prediction["validity_logit"], torch.zeros(2, dtype=x.dtype))
    assert torch.equal(metrics["desirability_loss"], expected)
    assert torch.equal(loss, expected + model.config.validity_weight * expected_validity)
    assert torch.equal(metrics["desirability_target"], torch.zeros(2, dtype=x.dtype))
    assert metrics["invalid_count"] == 2 and metrics["valid_count"] == 0
    loss.backward()
    assert torch.isfinite(model.head[-1].bias.grad).all()
    assert model.head[-1].bias.grad[0] > 0.  # Invalid cases train h toward zero, not an omitted reward row.
    assert "reward" not in prediction and "reward_target" not in metrics


def test_log_score_input_gradient_survives_exp_underflow_and_ignores_validity_head(fitted):
    frozen, rows = fitted
    model = deepcopy(frozen)
    with torch.no_grad():
        model.head[-1].bias[0] = -1000.
    gradients = []
    for validity_bias in (-1000., 1000.):
        with torch.no_grad():
            model.head[-1].bias[1] = validity_bias
        x = rows["future_white"][:2].detach().clone().requires_grad_(True)
        prediction = model(x, torch.ones(2, dtype=x.dtype), rows["conditions"][:2],
                           initial_context=rows["initial_context"][:2])
        assert torch.count_nonzero(prediction["desirability"]) == 0
        gradient = torch.autograd.grad(model.score_potential(prediction).sum(), x)[0]
        assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient)
        gradients.append(gradient)
    assert torch.equal(gradients[0], gradients[1])


def test_strict_payload_roundtrip_loader_and_input_gradient(fitted, tmp_path):
    model, rows = fitted
    payload = _payload(model, rows)
    state = payload.pop("model")
    payload = json.loads(json.dumps(payload, allow_nan=False))
    payload["model"] = state
    path = tmp_path / "synthetic-exponential.pt"
    torch.save(payload, path)
    restored = legacy.load_solver_error_regressor(path, sha256(path.read_bytes()).hexdigest())
    assert isinstance(restored, exponential.ExponentialSolverErrorRegressor)
    assert restored.runtime_settings() == model.runtime_settings()
    assert restored.config == model.config and restored.reward_center == model.reward_center
    assert not restored.training and all(not parameter.requires_grad for parameter in restored.parameters())
    x = rows["future_white"][:2].detach().clone().requires_grad_(True)
    sigma = torch.ones(2, dtype=x.dtype)
    args = (x, sigma, rows["conditions"][:2])
    kwargs = {"initial_context": rows["initial_context"][:2]}
    expected, actual = model(*args, **kwargs), restored(*args, **kwargs)
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    gradient = torch.autograd.grad(restored.score_potential(actual).sum(), x)[0]
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient)
    assert not restored.architecture_report()["scientific_admission"]
    assert not restored.architecture_report()["exact_tilted_sampling_claimed"]
    with pytest.raises(ValueError, match="refitting"):
        exponential.fit_exponential_calibration(restored, rows, seed=4)


def _reseal(payload):
    payload["state_sha256"] = calibrated._state_sha(payload["model"])
    payload["metadata_sha256"] = legacy._json_sha({key: value for key, value in payload.items()
                                                  if key not in ("model", "metadata_sha256")})


@pytest.mark.parametrize("change", ["beta", "center", "second_beta", "target_mean", "scale", "label", "root", "base", "step"])
def test_payload_rejects_objective_scale_or_provenance_tampering(fitted, change):
    model, rows = fitted
    payload = _payload(model, rows)
    if change == "beta":
        payload["exponential"]["beta"] = 3.
    elif change == "center":
        payload["exponential"]["reward_center"] += .01
    elif change == "second_beta":
        payload["recipe"]["expected_sampling_strength"] = 2.
    elif change == "target_mean":
        payload["exponential"]["fit_desirability_mean"] = 0.
    elif change == "scale":
        payload["model"]["future_field_scale"][0] *= 2.
    elif change == "label":
        payload["label_manifest_sha256"] = "b" * 64
    elif change == "root":
        payload["root_partition"]["validation_root_ids"] = [rows["root_ids"][0]]
    elif change == "base":
        payload["config"]["normalizer_sha256"] = "c" * 64
    else:
        payload["training"]["step"] = -1
    _reseal(payload)
    if change == "scale":
        assert exponential.regressor_from_payload(payload)
    else:
        with pytest.raises(ValueError):
            exponential.regressor_from_payload(payload)
