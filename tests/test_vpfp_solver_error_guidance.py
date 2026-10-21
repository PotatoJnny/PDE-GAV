"""Synthetic mechanical fixtures, not validation of guidance or solver error.

Run on a scheduled compute node. No real checkpoint, PDE solve or submission.
"""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json

import numpy as np
import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp import initial_conditioned_diffusion as ic
from pde_agent_verifier_solver.plugins.vpfp import solver_error_guidance as sg
from tests.test_vpfp_initial_conditioned_diffusion import checkpoint_payload, fixture_config


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def guide_config(**kwargs):
    return sg.SolverErrorGuidanceConfig(fixture_config(), "e"*64, "f"*64, width=4, depth=1, **kwargs)


def inputs(config, batch=4):
    cfg = config.base_config
    random = torch.Generator().manual_seed(987)
    future = torch.randn(batch, cfg.latent_dimension, generator=random)
    context = torch.randn(batch, cfg.context_dimension, generator=random)
    theta = torch.stack((torch.linspace(12., 28., batch), torch.linspace(.85, 1.15, batch)), 1)
    return future, context, theta


def native(monkeypatch, *, device="cpu", churn=0.):
    torch.manual_seed(451)
    config, raw, _, _, payload = checkpoint_payload()
    base = ic._sampler_from_initial_conditioned_payload(payload, "e"*64, device, steps=3, churn=churn)
    context, future = base.normalizer.encode(raw.to(device)[:2])
    theta = torch.tensor([[15., 1.], [25., 1.]], device=device)
    cfg = sg.SolverErrorGuidanceConfig(config, "e"*64, sg._json_sha(base.normalizer.to_payload()), width=4, depth=1)
    calls = []
    def prior(seeds, conditions=None):
        calls.append(tuple(seeds))
        n = len(seeds)
        if conditions is not None:
            assert isinstance(conditions, np.ndarray) and conditions.dtype == np.float32
        return {"initial_controls": torch.ones(n, config.context_dimension, device=device),
            "equation_conditions": np.array([[20., 1.]]*n, dtype=np.float32) if conditions is None else conditions}
    monkeypatch.setattr(base.prior, "sample", prior)
    return base, sg.SolverErrorRegressor(cfg).to(device), context, future, theta, calls


def test_bounded_actual_error_labels_and_invalid_mask():
    x = torch.zeros(5)
    targets, valid = sg.bounded_error_targets([0., 1., 1.e300, float("nan"), float("inf")],
        [True, True, True, False, False], prototype=x)
    assert torch.isfinite(targets).all() and 0 == targets[0] < targets[1] < targets[2] < 1
    assert targets[1] == pytest.approx(np.log(2)/(1+np.log(2)))
    assert torch.equal(targets[~valid], torch.zeros(2))
    for error, validity in (([-1.], [True]), ([float("nan")], [True]), ([1.], [1])):
        with pytest.raises(ValueError): sg.bounded_error_targets(error, validity, prototype=x)


def test_root_split_is_before_noise_replicates():
    result = sg.validate_root_partition(["r0", "r0", "r1"], ["r2", "r2"])
    assert result["independent_train_root_count"] == 2 and result["independent_validation_root_count"] == 1
    for train, validation in ((["r0"], ["r0"]), ([], ["r1"]), (["r0"], [""])):
        with pytest.raises(ValueError): sg.validate_root_partition(train, validation)


@pytest.mark.parametrize("sigma_kind", ["low", "high"])
def test_complete_white_state_context_conditions_and_sigma_gradients(sigma_kind):
    torch.manual_seed(982)
    cfg = guide_config(); model = sg.SolverErrorRegressor(cfg)
    x, context, theta = inputs(cfg, 2)
    x.requires_grad_(True); context.requires_grad_(True); theta.requires_grad_(True)
    sigma = torch.full((2,), cfg.base_config.sigma_min if sigma_kind == "low" else cfg.base_config.sigma_max)
    prediction = model(x, sigma, theta, initial_context=context)
    derivatives = torch.autograd.grad((prediction["reward"]*prediction["valid_probability"]).sum(), (x, context, theta))
    assert all(torch.isfinite(g).all() and torch.count_nonzero(g) for g in derivatives)
    # Last coordinate belongs to the last retained kinetic channel/mode/time,
    # not a macro-only or low-order surrogate input.
    assert torch.count_nonzero(derivatives[0][:, -1]) and torch.count_nonzero(derivatives[1][:, -1])
    with pytest.raises(ValueError): model(x, sigma, theta, initial_context=context[:, :-1])
    with pytest.raises(ValueError): model(x, torch.zeros(2), theta, initial_context=context)
    with pytest.raises(ValueError): model(x, sigma, [[1., 1.], [25., 1.]], initial_context=context)


def test_real_optimizer_changes_only_guide_and_reduces_synthetic_training_loss():
    torch.manual_seed(882)
    cfg = guide_config(); model = sg.SolverErrorRegressor(cfg)
    x, context, theta = inputs(cfg)
    sigma, noise = torch.full((4,), .15), torch.zeros_like(x)
    errors, valid = [0., .2, 1., 4.], [True]*4
    optimizer = torch.optim.AdamW(model.parameters(), lr=.006, weight_decay=0.)
    first = model.loss(x, theta, context, errors, valid, sigma=sigma, noise=noise)[0].item()
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = model.loss(x, theta, context, errors, valid, sigma=sigma, noise=noise)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    final = model.loss(x, theta, context, errors, valid, sigma=sigma, noise=noise)[0].item()
    assert final < .8*first
    assert metrics["valid_count"] == 4 and metrics["invalid_count"] == 0


def test_invalid_failures_train_classifier_not_large_reward():
    cfg = guide_config(); model = sg.SolverErrorRegressor(cfg)
    x, context, theta = inputs(cfg, 2)
    loss, metrics = model.loss(x, theta, context, [float("nan"), float("inf")], [False, False],
        sigma=torch.ones(2), noise=torch.zeros_like(x))
    assert torch.isfinite(loss) and metrics["reward_loss"] == 0 and metrics["invalid_count"] == 2
    loss.backward()
    assert torch.count_nonzero(model.head[-1].weight.grad[0]) == 0
    assert torch.count_nonzero(model.head[-1].weight.grad[1]) > 0


def snapshot(guide):
    return sg.guidance_payload(guide, label_manifest_sha256="a"*64, train_root_ids=["train-a", "train-b"],
        validation_root_ids=["val-a"], training={"label_source": "synthetic_fixture", "steps": 0})


@pytest.mark.parametrize("tamper", [None, "fourier", "weight", "root", "extra", "binding"])
def test_strict_snapshot_roundtrip_and_reject_tamper(tmp_path, tamper):
    guide = sg.SolverErrorRegressor(guide_config()).eval()
    payload = snapshot(guide)
    # Include a real JSON round trip for all metadata, unlike in-memory-only
    # comparisons that can miss tuple/list serialization bugs.
    state = payload.pop("model"); payload = json.loads(json.dumps(payload)); payload["model"] = state
    if tamper == "fourier": payload["model"]["transform.rho_basis"][0, 0, 0] += .001
    if tamper == "weight": payload["model"]["input.weight"][0, 0, 0, 0, 0] = float("nan")
    if tamper == "root": payload["root_partition"]["validation_root_ids"] = ["train-a"]
    if tamper == "extra": payload["base_weights"] = {}
    if tamper == "binding": payload["config"]["normalizer_sha256"] = "b"*64
    if tamper is not None:
        with pytest.raises(ValueError): sg.regressor_from_payload(payload)
        return
    path = tmp_path/"guide.pt"; torch.save(payload, path)
    restored = sg.load_solver_error_regressor(path, sha256(path.read_bytes()).hexdigest())
    x, context, theta = inputs(guide.config, 2)
    expected = guide(x, torch.ones(2), theta, initial_context=context)
    actual = restored(x, torch.ones(2), theta, initial_context=context)
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    assert all(not p.requires_grad for p in restored.parameters())
    assert sg.load_solver_error_regressor(path, "0"*64)


@pytest.mark.parametrize("churn", [0., .1])
def test_zero_guidance_is_exact_native_sample_with_same_full_initial_context(monkeypatch, churn):
    base, guide, context, _, theta, _ = native(monkeypatch, churn=churn)
    original_context = context.clone()
    sampler = sg.GuidedIC40Sampler(base, guide, strength=0.)
    expected = base.model.sample(context, theta, [901, 902], steps=base.steps, churn=base.churn)
    actual = sampler.sample_future(context, theta, [901, 902])
    assert torch.equal(actual, expected) and torch.equal(context, original_context)
    assert sampler.last_report["guided_calls"] == 0
    assert sampler.last_report["denoise_calls"] == 2*(2*base.steps-1)
    assert sampler.last_report["maximum_denoised_displacement_l2"] == 0


def test_trained_saved_reloaded_guide_changes_real_native_heun_without_base_updates(monkeypatch):
    base, guide, context, future, theta, _ = native(monkeypatch)
    original_base = {k: v.clone() for k, v in base.model.state_dict().items()}
    original_normalizer = deepcopy(base.normalizer.to_payload())
    optimizer = torch.optim.AdamW(guide.parameters(), lr=.003)
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = guide.loss(future, theta, context, [.1, 2.], [True, True],
            sigma=torch.tensor([.1, .3]), noise=torch.zeros_like(future))
        loss.backward(); optimizer.step()
    restored = sg.regressor_from_payload(snapshot(guide))
    sampler = sg.GuidedIC40Sampler(base, restored, strength=2.)
    baseline = base.model.sample(context, theta, [44, 45], steps=3)
    with torch.no_grad():
        guided = sampler.sample_future(context, theta, [44, 45])
    assert torch.isfinite(guided).all() and not torch.equal(guided, baseline)
    assert sampler.last_report["nonzero_gradient_calls"] > 0
    assert sampler.last_report["maximum_denoised_displacement_l2"] > 0
    assert all(torch.equal(original_base[k], v) for k, v in base.model.state_dict().items())
    assert base.normalizer.to_payload() == original_normalizer
    assert all(p.grad is None and not p.requires_grad for p in base.model.parameters())
    restored_full = base.normalizer.decode((context, guided))
    restored_context, _ = base.normalizer.encode(restored_full)
    torch.testing.assert_close(restored_context, context, rtol=2.e-6, atol=2.e-6)


class AnalyticGuide(sg.SolverErrorRegressor):
    """Synthetic increasing utility proves the reverse-score sign, not fidelity."""
    def forward(self, noisy_future_white, sigma, conditions, *, initial_context):
        reward = noisy_future_white[:, 0].sigmoid()
        return {"reward": reward, "validity_logit": torch.zeros_like(reward),
            "valid_probability": torch.ones_like(reward)}


def test_guidance_sign_clip_and_real_sampling_displacement(monkeypatch):
    base, guide, context, _, theta, _ = native(monkeypatch)
    analytic = AnalyticGuide(guide.config)
    sampler = sg.GuidedIC40Sampler(base, analytic, strength=.1, max_gradient_norm=.01)
    baseline = base.model.sample(context, theta, [31, 32], steps=3)
    output = sampler.sample_future(context, theta, [31, 32])
    assert torch.all(output[:, 0] > baseline[:, 0])
    assert torch.equal(output[:, 1:], baseline[:, 1:])
    assert sampler.last_report["clipped_calls"] > 0
    assert sampler.last_report["maximum_applied_gradient_l2"] <= .010001
    with torch.inference_mode(), pytest.raises(ValueError, match="inference_mode"):
        sampler.sample_future(context, theta, [31, 32])


def test_single_draw_native_batch_never_resamples_and_matches_zero_strength(monkeypatch):
    base, guide, _, _, theta, calls = native(monkeypatch)
    draw = sg.draw_frozen_ic40(base, [101, 102], theta)
    assert len(calls) == 1
    batch = sg.native_batch_from_draw(draw)
    assert len(calls) == 1 and np.array_equal(batch["latents"], draw["full"].numpy())
    assert draw["initial_context"].shape[1] == base.model.config.context_dimension
    assert draw["future_white"].shape[1] == base.model.config.latent_dimension
    sampler = sg.GuidedIC40Sampler(base, guide, strength=0.)
    second = sampler.draw([101, 102], theta)
    assert torch.equal(second["full"], draw["full"])
    assert torch.equal(second["future_white"], draw["future_white"])
    assert second["known_prior_seeds"] == draw["known_prior_seeds"]
    assert second["known_prior_seeds"] != second["future_noise_seeds"]
    assert len(calls) == 2
    generated = sampler.batch_from_draw(second)
    assert len(calls) == 2 and np.array_equal(generated["latents"], batch["latents"])
    assert np.array_equal(generated["equation_conditions"], theta.numpy())


@pytest.mark.parametrize("tamper", ["base_data", "normalizer", "guide_data", "settings", "config"])
def test_freeze_rejects_changes_including_data_bypassing_versions(monkeypatch, tamper):
    base, guide, _, _, _, _ = native(monkeypatch)
    sampler = sg.GuidedIC40Sampler(base, guide)
    if tamper == "base_data": next(base.model.parameters()).data.add_(.001)
    if tamper == "normalizer": base.normalizer.regression.data.add_(.001)
    if tamper == "guide_data": next(guide.parameters()).data.add_(.001)
    if tamper == "settings": sampler.strength = 2.
    if tamper == "config": guide.config = replace(guide.config, error_scale=2.)
    sampler.assert_frozen()


def test_wrong_normalizer_or_checkpoint_binding_is_not_accepted(monkeypatch):
    base, guide, _, _, _, _ = native(monkeypatch)
    for change in ({"base_checkpoint_sha256": "a"*64}, {"normalizer_sha256": "b"*64}):
        wrong = sg.SolverErrorRegressor(replace(guide.config, **change))
        with pytest.raises(ValueError, match="bindings differ"): sg.GuidedIC40Sampler(base, wrong)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA allocation required"))])
def test_device_copy_and_full_gradient_path(monkeypatch, device):
    base, guide, context, _, theta, _ = native(monkeypatch, device=device)
    guide = sg.regressor_from_payload(snapshot(guide), device=device)
    sampler = sg.GuidedIC40Sampler(base, guide, strength=.5)
    output = sampler.sample_future(context[:1], theta[:1], [123])
    assert output.device.type == device and torch.isfinite(output).all()
    assert sampler.last_report["nonzero_gradient_calls"] > 0
    sampler.assert_frozen()
    draw = sg.draw_frozen_ic40(base, [456], theta[:1])
    assert torch.equal(draw["equation_conditions"], theta[:1])
    assert draw["full"].device.type == device


def test_explicit_fp32_factory_preserves_original_initialization_and_ambient_dtype(monkeypatch):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(623)
        original = sg.SolverErrorRegressor(guide_config())
        rng_after_original = torch.get_rng_state().clone()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(623)
        actual = sg.make_solver_error_regressor(guide_config(), parameter_dtype=torch.float32)
        assert torch.get_default_dtype() == torch.float64
        assert torch.equal(torch.get_rng_state(), rng_after_original)
        assert all(torch.equal(value, actual.state_dict()[name]) for name, value in original.state_dict().items())
        assert {p.dtype for p in actual.parameters()} == {torch.float32}
        assert {b.dtype for b in actual.transform.buffers()} == {torch.float64}
        with pytest.raises(TypeError):
            sg.make_solver_error_regressor(object(), parameter_dtype=torch.float32)
        assert torch.get_default_dtype() == torch.float64
        with pytest.raises(ValueError):
            sg.make_solver_error_regressor(guide_config(), parameter_dtype=torch.float16)
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("stored_dtype,ambient_dtype", [(torch.float32, torch.float64), (torch.float64, torch.float32)])
def test_checkpoint_dtype_is_verified_not_inherited_from_ambient(tmp_path, stored_dtype, ambient_dtype):
    previous = torch.get_default_dtype()
    try:
        guide = sg.make_solver_error_regressor(guide_config(), parameter_dtype=stored_dtype).eval()
        x, context, theta = inputs(guide.config, 2)
        sigma = torch.ones(2, dtype=stored_dtype)
        expected = guide(x, sigma, theta, initial_context=context)
        path = tmp_path / "guide.pt"
        torch.save(snapshot(guide), path)
        torch.set_default_dtype(ambient_dtype)
        before_rng = torch.get_rng_state().clone()
        actual = sg.load_solver_error_regressor(path, sha256(path.read_bytes()).hexdigest())
        assert torch.get_default_dtype() == ambient_dtype
        assert torch.equal(torch.get_rng_state(), before_rng)
        assert {p.dtype for p in actual.parameters()} == {stored_dtype}
        assert {b.dtype for b in actual.transform.buffers()} == {torch.float64}
        result = actual(x, sigma, theta, initial_context=context)
        assert all(torch.equal(expected[key], result[key]) for key in expected)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("tamper", ["mixed", "half", "integer", "not_tensor", "fixed_dtype", "shape", "nonfinite", "requires_grad"])
def test_checkpoint_dtypes_and_tensor_abi_rejected_before_any_implicit_cast(tamper):
    guide = sg.make_solver_error_regressor(guide_config(), parameter_dtype=torch.float32)
    payload = snapshot(guide)
    state = payload["model"]
    if tamper == "mixed": state["input.weight"] = state["input.weight"].double()
    elif tamper == "half":
        for name in state:
            if not name.startswith("transform."): state[name] = state[name].half()
    elif tamper == "integer": state["input.weight"] = state["input.weight"].to(torch.int64)
    elif tamper == "not_tensor": state["input.weight"] = [0.]
    elif tamper == "fixed_dtype": state["transform.rho_basis"] = state["transform.rho_basis"].float()
    elif tamper == "shape": state["input.weight"] = state["input.weight"][:1]
    elif tamper == "nonfinite": state["input.weight"][0, 0, 0, 0, 0] = float("inf")
    elif tamper == "requires_grad": state["input.weight"].requires_grad_(True)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ValueError): sg.regressor_from_payload(payload)
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(previous)
