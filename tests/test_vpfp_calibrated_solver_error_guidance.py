"""Small synthetic calibration contracts; no real-model qualification claims."""

from copy import deepcopy
from hashlib import sha256
import json
import math

import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp import calibrated_solver_error_guidance as calibrated
from pde_agent_verifier_solver.plugins.vpfp import solver_error_guidance as legacy
from tests.test_vpfp_initial_conditioned_diffusion import fixture_config


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _config():
    return legacy.SolverErrorGuidanceConfig(fixture_config(), "e" * 64, "f" * 64,
                                            width=4, depth=1, validity_weight=.1)


def _rows(config, *, dtype=torch.float64):
    cfg = config.base_config
    rng = torch.Generator().manual_seed(123456)
    errors = torch.linspace(.05, 2., 64, dtype=torch.float64)
    errors[-1] = float("nan")
    valid = torch.ones(64, dtype=torch.bool)
    valid[-1] = False
    return {"future_white": 12. * torch.randn(64, cfg.latent_dimension, dtype=dtype, generator=rng),
            "initial_context": 30. * torch.randn(64, cfg.context_dimension, dtype=dtype, generator=rng),
            "conditions": torch.tensor([[20., 1.]], dtype=dtype).repeat(64, 1),
            "errors": errors, "valid": valid,
            "root_ids": [f"fit-{index:04d}" for index in range(64)], "excluded_rows": []}


def _make(scale=True, init=True, *, dtype=torch.float64):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(982)
        return calibrated.make_calibrated_regressor(_config(), parameter_dtype=dtype,
            scale_enabled=scale, init_enabled=init)


@pytest.fixture(scope="module")
def fitted():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = _make()
        rows = _rows(model.config)
        calibrated.fit_calibration(model, rows, seed=23456, batch_size=4)
        return model.eval(), rows
    finally:
        torch.set_num_threads(previous)


def _payload(model, rows):
    return calibrated.guidance_payload(model, label_manifest_sha256="a" * 64,
        train_root_ids=rows["root_ids"], validation_root_ids=[f"validation-{index:04d}" for index in range(16)],
        training={"label_source": "synthetic_fixture", "step": 0})


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_00_same_seed_has_exact_v1_initialization_forward_loss_and_gradients(dtype):
    config = _config()
    torch.manual_seed(711)
    old = legacy.make_solver_error_regressor(config, parameter_dtype=dtype)
    after_old = torch.get_rng_state().clone()
    torch.manual_seed(711)
    new = calibrated.make_calibrated_regressor(config, parameter_dtype=dtype, scale_enabled=False, init_enabled=False)
    assert torch.equal(torch.get_rng_state(), after_old)
    assert isinstance(new, legacy.SolverErrorRegressor)
    assert type(new).loss is legacy.SolverErrorRegressor.loss
    assert all(torch.equal(dict(new.named_parameters())[name], value) for name, value in old.named_parameters())
    rows = _rows(config, dtype=dtype)
    calibrated.fit_calibration(new, rows, seed=44)
    assert torch.equal(torch.get_rng_state(), after_old)
    for name in calibrated.SCALE_NAMES:
        assert torch.equal(getattr(new, name), torch.ones_like(getattr(new, name)))
    # Use modest query amplitudes to avoid an uncalibrated sigmoid saturation.
    future = rows["future_white"][:2] / 30.
    initial = rows["initial_context"][:2] / 50.
    sigma = torch.ones(2, dtype=dtype)
    results = []
    for model in (old, new):
        x, context, theta = (value.detach().clone().requires_grad_(True)
                             for value in (future, initial, rows["conditions"][:2]))
        prediction = model(x, sigma, theta, initial_context=context)
        gradients = torch.autograd.grad((prediction["reward"] * prediction["valid_probability"]).sum(),
                                        (x, context, theta, *model.parameters()))
        loss, _ = model.loss(future, theta.detach(), initial, rows["errors"][:2], rows["valid"][:2],
                             sigma=sigma, noise=torch.zeros_like(future))
        results.append((prediction, gradients, loss))
    assert all(torch.equal(results[0][0][key], results[1][0][key]) for key in results[0][0])
    assert all(torch.equal(left, right) for left, right in zip(results[0][1], results[1][1]))
    assert torch.equal(results[0][2], results[1][2])


def test_four_controls_use_fixed_fit_statistics_preserve_rng_and_only_change_declared_initialization():
    rows = _rows(_config())
    original_rows = deepcopy(rows)
    models = {}
    for scale, init in ((False, False), (True, False), (False, True), (True, True)):
        model = _make(scale, init)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        random_state = torch.get_rng_state().clone()
        training_rng = torch.Generator().manual_seed(998)
        training_state = training_rng.get_state().clone()
        metadata = calibrated.fit_calibration(model, rows, seed=3456)
        assert torch.equal(torch.get_rng_state(), random_state)
        assert torch.equal(training_rng.get_state(), training_state)
        assert metadata["split"] == "fit" and metadata["fit_root_ids"] == rows["root_ids"]
        assert metadata["valid_fit_count"] == 63
        assert metadata["validity_bias_probability"] == 63.5 / 65.
        assert metadata["ambient_rng_consumed"] is False and metadata["per_sample_normalization"] is False
        for name in calibrated.SCALE_NAMES:
            buffer = getattr(model, name)
            assert not buffer.requires_grad and name not in dict(model.named_parameters())
            assert torch.all(buffer >= 1.) and torch.isfinite(buffer).all()
            if not scale:
                assert torch.equal(buffer, torch.ones_like(buffer))
        if scale:
            assert torch.any(model.initial_field_scale > 1.) and torch.any(model.future_field_scale > 1.)
        for name, parameter in model.named_parameters():
            if init and name == "head.2.weight":
                assert torch.equal(parameter, before[name] * .01)
                assert torch.count_nonzero(parameter[0]) and torch.count_nonzero(parameter[1])
            elif init and name == "head.2.bias":
                reward_p, valid_p = metadata["reward_bias_probability"], metadata["validity_bias_probability"]
                assert torch.equal(parameter, parameter.new_tensor([math.log(reward_p / (1.-reward_p)),
                                                                   math.log(valid_p / (1.-valid_p))]))
            else:
                assert torch.equal(parameter, before[name])
        json.dumps(metadata, allow_nan=False)
        models[(scale, init)] = model
    for name in calibrated.SCALE_NAMES:
        assert torch.equal(getattr(models[(True, False)], name), getattr(models[(True, True)], name))
    for key in ("future_white", "initial_context", "conditions", "valid"):
        assert torch.equal(rows[key], original_rows[key])
    torch.testing.assert_close(rows["errors"], original_rows["errors"], equal_nan=True, rtol=0, atol=0)
    assert rows["root_ids"] == original_rows["root_ids"]


@pytest.mark.parametrize("change", ["validation_root", "duplicate_root", "split", "missing_root"])
def test_calibration_rejects_nonfit_or_incomplete_root_set_before_mutation(change):
    model = _make()
    rows = _rows(model.config)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    if change == "validation_root":
        rows["root_ids"][0] = "validation-0000"
    elif change == "duplicate_root":
        rows["root_ids"][0] = rows["root_ids"][1]
    elif change == "split":
        rows["split"] = "validation"
    else:
        rows["root_ids"].pop()
    with pytest.raises(ValueError, match="fit"):
        calibrated.fit_calibration(model, rows, seed=3)
    assert model.calibration_metadata is None
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


def test_fixed_buffers_preserve_amplitude_input_gradients_and_survive_optimizer_update(fitted):
    original, rows = fitted
    model = deepcopy(original).train()
    before = {name: getattr(model, name).clone() for name in calibrated.SCALE_NAMES}
    x = (rows["future_white"][:2] / 10.).detach().requires_grad_(True)
    context = rows["initial_context"][:2].detach().clone().requires_grad_(True)
    theta = rows["conditions"][:2].detach().clone().requires_grad_(True)
    sigma = torch.ones(2, dtype=x.dtype)
    observed = []
    hook = model.input.register_forward_pre_hook(lambda _module, args: observed.append(args[0].detach().clone()))
    try:
        prediction = model(x, sigma, theta, initial_context=context)
        model(2. * x, sigma, theta, initial_context=context)
    finally:
        hook.remove()
    assert torch.equal(observed[1], 2. * observed[0])
    derivatives = torch.autograd.grad((prediction["reward"] * prediction["valid_probability"]).sum(), (x, context, theta))
    assert all(torch.isfinite(value).all() and torch.count_nonzero(value) for value in derivatives)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    loss, _ = model.loss(x.detach(), theta.detach(), context.detach(), rows["errors"][:2], rows["valid"][:2],
                         sigma=sigma, noise=torch.zeros_like(x))
    loss.backward()
    optimizer.step()
    assert all(torch.equal(getattr(model, name), value) for name, value in before.items())
    with pytest.raises(ValueError, match="refitting"):
        calibrated.fit_calibration(model, rows, seed=4)


def test_payload_roundtrip_public_loader_dispatch_and_runtime_settings(fitted, tmp_path):
    model, rows = fitted
    payload = _payload(model, rows)
    state = payload.pop("model")
    payload = json.loads(json.dumps(payload, allow_nan=False))
    payload["model"] = state
    path = tmp_path / "synthetic-calibrated-guide.pt"
    torch.save(payload, path)
    restored = legacy.load_solver_error_regressor(path, sha256(path.read_bytes()).hexdigest())
    assert isinstance(restored, calibrated.CalibratedSolverErrorRegressor)
    assert not restored.training and all(not parameter.requires_grad for parameter in restored.parameters())
    assert restored.runtime_settings() == model.runtime_settings()
    assert restored.config == model.config and not restored.architecture_report()["scientific_admission"]
    x = rows["future_white"][:2].detach().clone().requires_grad_(True)
    sigma = torch.ones(2, dtype=x.dtype)
    expected = model(x, sigma, rows["conditions"][:2], initial_context=rows["initial_context"][:2])
    actual = restored(x, sigma, rows["conditions"][:2], initial_context=rows["initial_context"][:2])
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    gradient = torch.autograd.grad(actual["reward"].sum(), x)[0]
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient)
    settings = restored.runtime_settings()
    restored.scale_enabled = False
    assert restored.runtime_settings() != settings


def _reseal(payload):
    payload["state_sha256"] = calibrated._state_sha(payload["model"])
    payload["metadata_sha256"] = legacy._json_sha({key: value for key, value in payload.items()
                                                  if key not in ("model", "metadata_sha256")})


@pytest.mark.parametrize("change", ["scale_zero", "scale_changed", "recipe", "fit_source", "label", "base",
                                    "fourier", "step", "root", "dtype", "state_hash"])
def test_payload_rejects_tampering_even_when_generic_hashes_are_recomputed(fitted, change):
    model, rows = fitted
    payload = _payload(model, rows)
    if change == "scale_zero":
        payload["model"]["future_field_scale"][0] = 0.
    elif change == "scale_changed":
        payload["model"]["future_field_scale"][0] *= 2.
    elif change == "recipe":
        payload["recipe"]["head_weight_multiplier"] = .02
    elif change == "fit_source":
        payload["calibration"]["split"] = "validation"
    elif change == "label":
        payload["label_manifest_sha256"] = "b" * 64
    elif change == "base":
        payload["config"]["base_checkpoint_sha256"] = "c" * 64
    elif change == "fourier":
        payload["model"]["transform.rho_basis"][0, 0, 0] += .1
    elif change == "step":
        payload["training"]["step"] = -1
    elif change == "root":
        payload["root_partition"]["validation_root_ids"] = [rows["root_ids"][0]]
    elif change == "dtype":
        payload["model"]["pooled_scale"] = payload["model"]["pooled_scale"].float()
    else:
        payload["model"]["input.weight"].flatten()[0] += .1
    if change != "state_hash":
        _reseal(payload)
    if change in ("scale_changed", "state_hash"):
        assert calibrated.regressor_from_payload(payload)
    else:
        with pytest.raises(ValueError):
            calibrated.regressor_from_payload(payload)
