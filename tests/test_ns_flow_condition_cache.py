"""Numerical equivalence of frozen-flow conditioning reuse during sampling."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def flow_module():
    path = (Path(__file__).resolve().parents[1] / "legacy/ns/src/agent_verifier/"
            "neural_generator/structured_ns_distribution.py")
    name = "_ns_flow_condition_cache_test_source"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield module
    torch.set_num_threads(old_threads)
    sys.modules.pop(name, None)


def _model(module, diagonal=False):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(81)
        model = module.StructuredCoefficientRectifiedFlow.build(
            module.StructuredCoefficientFlowConfig(
                feature_dim=7, width=24, depth=3, time_width=8,
                full_rank_diagonal_velocity=diagonal,
            )
        )
        # Check nonzero trained-like output and diagonal branches, not the
        # factory's zero-initialized velocity that would mask differences.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.uniform_(-0.15, 0.15)
    return model.eval()


class _OriginalForward:
    """Use the unchanged public forward through the generic sampler path."""
    def __init__(self, model):
        self.config = model.config
        self.model = model

    def __call__(self, values, time):
        return self.model(values, time)


def _guidance(values, time, velocity):
    # Production reward guidance briefly enables autograd inside no_grad flow
    # integration. Preserve that interaction as well as every guidance call.
    with torch.enable_grad():
        differentiable = values.detach().requires_grad_(True)
        reward = (differentiable.sin() * time[:, None]).sum()
        gradient, = torch.autograd.grad(reward, differentiable)
    return gradient * 0.01 + velocity * 0.02


@pytest.mark.parametrize("method", ["euler", "heun"])
@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("guided", [False, True])
@pytest.mark.parametrize("initial", [False, True])
def test_frozen_flow_outputs_and_rng_are_bitwise_identical(
    flow_module, method, diagonal, guided, initial,
):
    model = _model(flow_module, diagonal)
    options = dict(
        count=3, steps=17, device="cpu", method=method,
        guidance=_guidance if guided else None, guidance_scale=0.7 if guided else 0.0,
    )
    if initial:
        options["initial_values"] = torch.arange(21, dtype=torch.float32).reshape(3, 7) / 13
    before_rng = torch.Generator().manual_seed(101)
    after_rng = torch.Generator().manual_seed(101)
    with torch.no_grad():
        before = flow_module._sample_structured_flow(
            _OriginalForward(model), generator=before_rng, **options,
        )
        after = model.sample(generator=after_rng, **options)
    assert torch.equal(before, after)
    assert torch.equal(before_rng.get_state(), after_rng.get_state())


@pytest.mark.parametrize("method", ["euler", "heun"])
def test_reuses_only_repeated_time_conditioning(flow_module, method, monkeypatch):
    model = _model(flow_module, diagonal=True)
    counts = {"time": 0, "block": 0, "diagonal": 0}
    tracked_weights = {
        id(model.time_mlp[0].weight): "time",
        id(model.diagonal_velocity.weight): "diagonal",
        **{id(block.condition.weight): "block" for block in model.blocks},
    }
    original_linear = torch.nn.functional.linear

    def count_linear(values, weight, bias=None):
        name = tracked_weights.get(id(weight))
        if name is not None:
            counts[name] += 1
        return original_linear(values, weight, bias)

    # Observe the tensor operation without adding model hooks: hooks must
    # retain their ordinary per-forward semantics and disable this cache.
    monkeypatch.setattr(torch.nn.functional, "linear", count_linear)
    with torch.no_grad():
        model.sample(2, steps=7, device="cpu", method=method)
    assert counts == {"time": 7, "block": 21, "diagonal": 7}


@pytest.mark.parametrize("target", ["model", "block", "conditioning", "global"])
@pytest.mark.parametrize("pre_hook", [False, True])
def test_hooks_preserve_original_outputs_and_every_forward(flow_module, target, pre_hook):
    model = _model(flow_module, diagonal=True)
    layer = {
        "model": model, "block": model.blocks[0],
        "conditioning": model.blocks[0].condition, "global": model,
    }[target]
    calls = []

    def before(current, inputs):
        if current is layer:
            calls.append(1)
            return (inputs[0] + 0.25, *inputs[1:])

    def after(current, inputs, output):
        if current is layer:
            calls.append(1)
            return output + 0.25

    if target == "global":
        namespace = torch.nn.modules.module
        register = (namespace.register_module_forward_pre_hook if pre_hook
                    else namespace.register_module_forward_hook)
    else:
        register = layer.register_forward_pre_hook if pre_hook else layer.register_forward_hook
    handle = register(before if pre_hook else after)
    try:
        initial = torch.ones((2, 7))
        options = dict(steps=4, device="cpu", initial_values=initial, method="heun")
        with torch.no_grad():
            expected = flow_module._sample_structured_flow(_OriginalForward(model), 2, **options)
            assert len(calls) == 7
            calls.clear()
            actual = model.sample(2, **options)
        assert torch.equal(actual, expected)
        assert len(calls) == 7
    finally:
        handle.remove()


@pytest.mark.parametrize("target", ["model", "block", "conditioning"])
def test_custom_forward_preserves_original_path(flow_module, target, monkeypatch):
    model = _model(flow_module, diagonal=True)
    layer = {"model": model, "block": model.blocks[0],
             "conditioning": model.blocks[0].condition}[target]
    original_forward = layer.forward
    calls = []

    def custom_forward(*args, **kwargs):
        calls.append(1)
        return original_forward(*args, **kwargs) + 0.25

    monkeypatch.setattr(layer, "forward", custom_forward)
    options = dict(steps=4, device="cpu", initial_values=torch.ones((2, 7)), method="heun")
    with torch.no_grad():
        expected = flow_module._sample_structured_flow(_OriginalForward(model), 2, **options)
        assert len(calls) == 7
        calls.clear()
        actual = model.sample(2, **options)
    assert torch.equal(actual, expected)
    assert len(calls) == 7


def test_custom_internal_module_preserves_original_path(flow_module):
    class ShiftedLinear(torch.nn.Linear):
        def __init__(self, original):
            super().__init__(original.in_features, original.out_features)
            self.load_state_dict(original.state_dict())
            self.calls = 0

        def forward(self, values):
            self.calls += 1
            return super().forward(values) + 0.25

    model = _model(flow_module, diagonal=True)
    replacement = ShiftedLinear(model.blocks[0].condition).eval()
    model.blocks[0].condition = replacement
    options = dict(steps=4, device="cpu", initial_values=torch.ones((2, 7)), method="heun")
    with torch.no_grad():
        expected = flow_module._sample_structured_flow(_OriginalForward(model), 2, **options)
        assert replacement.calls == 7
        replacement.calls = 0
        actual = model.sample(2, **options)
    assert torch.equal(actual, expected)
    assert replacement.calls == 7


@pytest.mark.parametrize("training", [False, True])
def test_autograd_and_training_keep_original_forward(flow_module, training):
    model = _model(flow_module, diagonal=True).train(training)
    initial = torch.ones((2, 7), requires_grad=True)
    calls = []
    handle = model.register_forward_hook(lambda *_: calls.append(1))
    try:
        result = model.sample(2, steps=4, device="cpu", initial_values=initial)
        result.sum().backward()
        assert len(calls) == 7
        assert initial.grad is not None and torch.isfinite(initial.grad).all()
        assert model.time_mlp[0].weight.grad is not None
        if training:
            calls.clear()
            with torch.no_grad():
                model.sample(2, steps=4, device="cpu", initial_values=initial)
            assert len(calls) == 7
    finally:
        handle.remove()


def test_cache_does_not_survive_sampling_call_or_checkpoint_reload(flow_module):
    model = _model(flow_module, diagonal=True)
    initial = torch.ones((2, 7))
    with torch.no_grad():
        first = model.sample(2, steps=8, device="cpu", initial_values=initial)
        checkpoint = {key: value.clone() for key, value in model.state_dict().items()}
        checkpoint["time_mlp.0.weight"] += 0.1
        model.load_state_dict(checkpoint)
        expected = flow_module._sample_structured_flow(
            _OriginalForward(model), 2, steps=8, device="cpu", initial_values=initial,
        )
        actual = model.sample(2, steps=8, device="cpu", initial_values=initial)
    assert torch.equal(actual, expected)
    assert not torch.equal(first, actual)
