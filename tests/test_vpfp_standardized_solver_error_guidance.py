"""Synthetic standardization and checkpoint proofs, not solver-hardness evidence."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import math

import pytest
import torch
from torch.nn import functional as F

from pde_agent_verifier_solver.plugins.vpfp import calibrated_solver_error_guidance as calibrated
from pde_agent_verifier_solver.plugins.vpfp import exponential_solver_error_guidance as exponential
from pde_agent_verifier_solver.plugins.vpfp import solver_error_guidance as legacy
from pde_agent_verifier_solver.plugins.vpfp import standardized_solver_error_guidance as standardized
from tests.test_vpfp_calibrated_solver_error_guidance import _config, _rows


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _make(objective=standardized.OBJECTIVE, *, dtype=torch.float64, beta_std=.5):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(982)
        return standardized.make_standardized_regressor(_config(), parameter_dtype=dtype,
            beta_std=beta_std, objective=objective)


@pytest.fixture(scope="module")
def fitted():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = _make()
        rows = _rows(model.config)
        standardized.fit_standardized_calibration(model, rows, seed=23456)
        return model.eval(), rows
    finally:
        torch.set_num_threads(previous)


def _target(model, errors, valid, prototype, *, objective=None):
    return standardized.standardized_targets(errors, valid, prototype=prototype,
        fit_reward_mean=model.fit_reward_mean, fit_reward_std=model.fit_reward_std,
        log_fit_mean_exp=model.log_fit_mean_exp, beta_std=model.beta_std,
        objective=model.objective if objective is None else objective,
        error_scale=model.config.error_scale)


def _payload(model, rows):
    return standardized.standardized_payload(model, label_manifest_sha256="a" * 64,
        train_root_ids=rows["root_ids"], validation_root_ids=[f"validation-{i:04d}" for i in range(16)],
        training={"label_source": "synthetic_fixture", "step": 0})


def _reseal(payload):
    payload["state_sha256"] = calibrated._state_sha(payload["model"])
    payload["metadata_sha256"] = legacy._json_sha({key: value for key, value in payload.items()
                                                  if key not in ("model", "metadata_sha256")})


def test_constant_raw_labels_rejected_even_if_transform_has_rounding_variation(monkeypatch):
    model = _make()
    rows = _rows(model.config)
    rows["errors"][rows["valid"]] = .2
    original = legacy.bounded_error_targets
    def noisy_transform(*args, **kwargs):
        rewards, valid = original(*args, **kwargs)
        # Deliberately expose one representable-unit mapping residue. It must
        # not turn a constant-label problem into a usable training scale.
        indices = torch.nonzero(valid).flatten()
        rewards[indices[0]] = torch.nextafter(rewards[indices[1]], rewards.new_tensor(float("inf")))
        return rewards, valid
    monkeypatch.setattr(legacy, "bounded_error_targets", noisy_transform)
    with pytest.raises(ValueError, match="distinct valid solver errors"):
        standardized.fit_standardized_calibration(model, rows, seed=23456)
    assert model.standardized_metadata is None and model.calibration_metadata is None


def test_fit_preserves_historical_log1p_reward_and_all_root_mean_one(fitted):
    model, rows = fitted
    rewards, valid = legacy.bounded_error_targets(rows["errors"], rows["valid"],
        prototype=rows["future_white"], error_scale=model.config.error_scale)
    torch.testing.assert_close(rewards[valid],
        torch.log1p(rows["errors"][valid]) / (1. + torch.log1p(rows["errors"][valid])))
    assert not torch.allclose(rewards[valid], rows["errors"][valid] / (1. + rows["errors"][valid]))
    assert model.fit_reward_mean == pytest.approx(float(rewards[valid].mean()), abs=1.e-15)
    assert model.fit_reward_std == pytest.approx(float(rewards[valid].std(unbiased=False)), abs=1.e-15)
    targets, actual_valid = _target(model, rows["errors"], rows["valid"], rewards)
    assert torch.equal(actual_valid, valid)
    assert targets[~valid].eq(0.).all()
    assert float(targets.mean()) == pytest.approx(1., abs=1.e-14)
    assert float(targets[valid].mean()) == pytest.approx(64. / 63., abs=1.e-14)
    assert model.standardized_metadata["fit_desirability_ess"] == pytest.approx(
        float(targets.sum().square() / targets.square().sum()), rel=1.e-14)
    assert model.beta == model.beta_std / model.fit_reward_std
    assert model.head[-1].bias[0] == 0.
    assert model.standardized_metadata["fit_label_records"][-1]["reward"] is None
    assert model.standardized_metadata["expected_sampling_strength"] is None
    json.dumps(model.standardized_metadata, allow_nan=False)


def test_fixed_fit_normalization_is_invariant_to_batch_size_and_candidate_values(fitted):
    model, rows = fitted
    entire, _ = _target(model, rows["errors"], rows["valid"], rows["future_white"])
    subset, _ = _target(model, rows["errors"][:3], rows["valid"][:3], rows["future_white"])
    assert torch.equal(entire[:3], subset)
    baseline = deepcopy(model.standardized_metadata)
    _target(model, [0., 1.e9], [True, True], rows["future_white"])
    assert model.standardized_metadata == baseline
    assert model.fit_reward_mean == baseline["fit_reward_mean"]


def test_exp_regression_optimum_is_mean_weight_not_mean_log_weight(fitted):
    model, rows = fitted
    weights, _ = _target(model, [.05, 2.], [True, True], rows["future_white"])
    optimum = weights.mean().log().detach().requires_grad_(True)
    gradient = torch.autograd.grad(exponential.desirability_mse(optimum.expand(2), weights), optimum)[0]
    assert abs(float(gradient)) < 1.e-14
    wrong = weights.log().mean().detach().requires_grad_(True)
    gradient = torch.autograd.grad(exponential.desirability_mse(wrong.expand(2), weights), wrong)[0]
    assert gradient < 0.


@pytest.mark.parametrize("objective", [standardized.OBJECTIVE, standardized.MEAN_OBJECTIVE])
def test_score_is_raw_with_matched_mean_first_order_beta_and_no_validity_multiplier(objective):
    model = _make(objective)
    x = torch.tensor([-.2, .3], dtype=torch.float64, requires_grad=True)
    raw = 3. * x + .1
    key = "log_desirability" if objective == standardized.OBJECTIVE else "standardized_reward"
    prediction = {key: raw, "valid_probability": torch.zeros(2)}
    gradient = torch.autograd.grad(model.score_potential(prediction).sum(), x)[0]
    expected = 3. if objective == standardized.OBJECTIVE else 3. * model.beta_std
    assert torch.equal(gradient, torch.full_like(x, expected))
    assert model.expected_sampling_strength is None
    # A lambda belongs to the sampler policy. score_potential does not enforce lambda=1.
    model.expected_sampling_strength = .25
    assert torch.isfinite(model.score_potential(prediction)).all()


@pytest.mark.parametrize("beta_std", [.5, 1.])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_constructor_preserves_calibrated_backbone_parameter_rng_and_dtype(beta_std, dtype):
    torch.manual_seed(711)
    reference = calibrated.make_calibrated_regressor(_config(), parameter_dtype=dtype,
        scale_enabled=True, init_enabled=True)
    rng_after = torch.get_rng_state().clone()
    torch.manual_seed(711)
    model = standardized.make_standardized_regressor(_config(), parameter_dtype=dtype, beta_std=beta_std)
    assert torch.equal(torch.get_rng_state(), rng_after)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in reference.state_dict().items())
    assert all(parameter.dtype == dtype for parameter in model.parameters())
    assert model.transform.rho_basis.dtype == torch.float64


@pytest.mark.parametrize("beta_std", [0., 2., True, float("nan")])
def test_unregistered_beta_rejected(beta_std):
    with pytest.raises(ValueError, match="beta_std"):
        _make(beta_std=beta_std)


def test_mean_and_exp_fit_have_identical_scales_state_and_independent_rng(fitted):
    reference, rows = fitted
    model = _make(standardized.MEAN_OBJECTIVE)
    original = deepcopy(rows)
    rng_before = torch.get_rng_state().clone()
    metadata = standardized.fit_standardized_calibration(model, rows, seed=23456)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert model.calibration_metadata == reference.calibration_metadata
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in reference.state_dict().items())
    for key in ("fit_reward_mean", "fit_reward_std", "log_fit_mean_exp", "effective_beta", "fit_desirability_ess"):
        assert metadata[key] == reference.standardized_metadata[key]
    assert torch.equal(rows["future_white"], original["future_white"])
    torch.testing.assert_close(rows["errors"], original["errors"], equal_nan=True, rtol=0, atol=0)
    assert metadata["invalid_labels"] == "auxiliary_validity_only"


@pytest.mark.parametrize("change", ["validation", "duplicate", "missing", "degenerate", "invalid", "nonfinite"])
def test_bad_fit_fails_atomically_before_parameters_or_statistics_change(change):
    model = _make()
    rows = _rows(model.config)
    if change == "validation":
        rows["root_ids"][0] = "validation-0000"
    elif change == "duplicate":
        rows["root_ids"][0] = rows["root_ids"][1]
    elif change == "missing":
        rows["root_ids"].pop()
    elif change == "degenerate":
        rows["errors"][rows["valid"]] = .2
    elif change == "invalid":
        rows["valid"][:] = False
    else:
        rows["errors"][0] = float("inf")
    state = deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        standardized.fit_standardized_calibration(model, rows, seed=4)
    assert model.calibration_metadata is model.standardized_metadata is None
    assert model.fit_reward_mean is model.fit_reward_std is model.log_fit_mean_exp is model.beta is None
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())


@pytest.mark.parametrize("objective", [standardized.OBJECTIVE, standardized.MEAN_OBJECTIVE])
def test_actual_loss_invalid_semantics_and_raw_head_gradient(fitted, objective):
    frozen, rows = fitted
    model = deepcopy(frozen)
    if objective == standardized.MEAN_OBJECTIVE:
        model = _make(objective)
        standardized.fit_standardized_calibration(model, rows, seed=23456)
    x, theta, context = (rows[key][:2] for key in ("future_white", "conditions", "initial_context"))
    sigma = torch.ones(2, dtype=x.dtype)
    prediction = model(x, sigma, theta, initial_context=context)
    loss, metrics = model.loss(x, theta, context, [float("nan"), float("inf")], [False, False],
        sigma=sigma, noise=torch.zeros_like(x))
    bce = F.binary_cross_entropy_with_logits(prediction["validity_logit"], torch.zeros(2, dtype=x.dtype))
    expected_regression = prediction["desirability"].square().mean() if objective == standardized.OBJECTIVE else 0.
    torch.testing.assert_close(loss, expected_regression + .1 * bce, rtol=0, atol=0)
    assert metrics["invalid_count"] == 2 and metrics["valid_count"] == 0
    loss.backward()
    first_head_gradient = float(model.head[-1].bias.grad[0])
    assert first_head_gradient > 0. if objective == standardized.OBJECTIVE else first_head_gradient == 0.


def test_mean_forward_does_not_exponentiate_or_sigmoid_large_raw_output(fitted):
    frozen, rows = fitted
    model = deepcopy(frozen)
    model.objective = standardized.MEAN_OBJECTIVE
    with torch.no_grad():
        model.head[-1].bias[0] = 1000.
    prediction = model(rows["future_white"][:2], torch.ones(2, dtype=torch.float64), rows["conditions"][:2],
        initial_context=rows["initial_context"][:2])
    assert set(prediction) == {"standardized_reward", "validity_logit", "valid_probability"}
    assert torch.all(prediction["standardized_reward"] > 999.)


def test_targets_reject_overflow_underflow_and_degenerate_scales():
    prototype = torch.zeros(1, dtype=torch.float64)
    for center in (0., 1.):
        with pytest.raises(FloatingPointError, match="clipping"):
            standardized.standardized_targets([1. if center == 0. else 0.], [True], prototype=prototype,
                fit_reward_mean=center, fit_reward_std=1.e-12, log_fit_mean_exp=0., beta_std=1.)
    for std in (0., float("nan")):
        with pytest.raises(ValueError, match="nondegenerate"):
            standardized.standardized_targets([1.], [True], prototype=prototype,
                fit_reward_mean=.1, fit_reward_std=std, log_fit_mean_exp=0., beta_std=1.)


@pytest.mark.parametrize("objective", [standardized.OBJECTIVE, standardized.MEAN_OBJECTIVE])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_payload_roundtrip_strict_dtype_forward_and_input_gradient(objective, dtype, tmp_path):
    model = _make(objective, dtype=dtype)
    rows = _rows(model.config, dtype=dtype)
    standardized.fit_standardized_calibration(model, rows, seed=23456)
    payload = _payload(model, rows)
    state = payload.pop("model")
    payload = json.loads(json.dumps(payload, allow_nan=False))
    payload["model"] = state
    path = tmp_path / "synthetic-standardized.pt"
    torch.save(payload, path)
    restored = legacy.load_solver_error_regressor(path, sha256(path.read_bytes()).hexdigest())
    assert isinstance(restored, standardized.StandardizedSolverErrorRegressor)
    assert restored.runtime_settings() == model.runtime_settings()
    assert not restored.training and all(not parameter.requires_grad for parameter in restored.parameters())
    x = rows["future_white"][:2].clone().requires_grad_(True)
    sigma, theta, context = torch.ones(2, dtype=dtype), rows["conditions"][:2], rows["initial_context"][:2]
    expected, actual = (guide(x, sigma, theta, initial_context=context) for guide in (model, restored))
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    derivative = torch.autograd.grad(restored.score_potential(actual).sum(), x)[0]
    assert torch.isfinite(derivative).all() and torch.count_nonzero(derivative)
    with pytest.raises(ValueError, match="refitting"):
        standardized.fit_standardized_calibration(restored, rows, seed=1)


@pytest.mark.parametrize("change", ["mean", "std", "log_z", "ess", "range", "record", "root", "label",
                                    "config", "beta", "strength", "step", "scale"])
def test_loader_rejects_resealed_inconsistent_statistics_and_bindings(fitted, change):
    model, rows = fitted
    payload = _payload(model, rows)
    metadata = payload["standardized"]
    if change in ("mean", "std", "log_z", "ess"):
        key = {"mean": "fit_reward_mean", "std": "fit_reward_std",
               "log_z": "log_fit_mean_exp", "ess": "fit_desirability_ess"}[change]
        metadata[key] += .01
    elif change == "range":
        metadata["fit_desirability_range"][1] += .01
    elif change == "record":
        metadata["fit_label_records"][0]["reward"] += .01
    elif change == "root":
        metadata["fit_label_records"][0]["root_id"] = "validation-0000"
    elif change == "label":
        metadata["label_manifest_sha256"] = "b" * 64
    elif change == "config":
        payload["config"]["base_checkpoint_sha256"] = "b" * 64
    elif change == "beta":
        payload["recipe"]["beta_std"] = 1.
    elif change == "strength":
        payload["recipe"]["expected_sampling_strength"] = 1.
    elif change == "step":
        payload["training"]["step"] = -1
    else:
        payload["model"]["future_field_scale"][0] += 1.
    _reseal(payload)
    if change == "scale":
        assert standardized.regressor_from_payload(payload)
    else:
        with pytest.raises(ValueError):
            standardized.regressor_from_payload(payload)


def test_payload_rejects_mutated_runtime_fit_statistics(fitted):
    model, rows = fitted
    mutated = deepcopy(model)
    mutated.fit_reward_std *= 2.
    with pytest.raises(ValueError, match="runtime statistics"):
        _payload(mutated, rows)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("objective", [standardized.OBJECTIVE, standardized.MEAN_OBJECTIVE])
def test_native_sampler_identity_names_mean_score_and_allows_separate_gain(monkeypatch, objective, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("allocated CUDA device required for the GPU integration fixture")
    from tests.test_vpfp_solver_error_guidance import native
    base, old_guide, _, _, _, _ = native(monkeypatch, device=device)
    config = replace(old_guide.config, validity_weight=.1)
    parameter = next(base.model.parameters())
    guide = standardized.make_standardized_regressor(config, parameter_dtype=parameter.dtype,
        device=device, beta_std=.5, objective=objective)
    rows = _rows(config, dtype=parameter.dtype)
    standardized.fit_standardized_calibration(guide, rows, seed=23456)
    for strength in (.25, 1., 2.):
        sampler = legacy.GuidedIC40Sampler(base, guide, strength=strength, sigma_min=.002, sigma_max=2.)
        identity = sampler.identity["guidance"]
        expected = ("beta_std*E(standardized_reward|valid,noisy,IC,nu,T)"
                    if objective == standardized.MEAN_OBJECTIVE else "log_joint_valid_exponential_reward_desirability")
        assert identity["utility"] == expected
        assert identity["training_beta_std"] == .5
        assert identity["effective_raw_reward_beta"] == guide.beta
        assert identity["training_objective"] == objective
        assert identity["sampling_gain_separate_from_training_beta"] is True
        assert identity["auxiliary_validity_enters_score"] is False
        assert sampler.strength == strength and guide.expected_sampling_strength is None
        sampler.assert_frozen()
    description = guide.architecture_report()
    assert description["reward_target"] != legacy.SolverErrorRegressor.architecture_report(guide)["reward_target"]
    if objective == standardized.OBJECTIVE:
        assert description["invalid_labels"].startswith("zero_desirability")
