"""Tiny synthetic observability checks, not actual Guidance training evidence."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch


_PATH = Path(__file__).resolve().parents[1] / "scripts/vpfp_guidance_training_diagnostics.py"
_SPEC = importlib.util.spec_from_file_location("vpfp_guidance_training_diagnostics_test", _PATH)
diagnostics = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diagnostics)


def _plain(value):
    if isinstance(value, dict):
        return all(isinstance(key, str) and _plain(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_plain(item) for item in value)
    return value is None or type(value) in (int, float, bool, str)


def test_tiny_fp32_statistics_retain_nonzero_scale_in_float64():
    tensor = torch.tensor([1e-30, -1e-30], dtype=torch.float32, requires_grad=True)
    result = diagnostics.tensor_stats(tensor)
    amplitude = abs(float(tensor.detach()[0]))
    assert result["shape"] == [2] and result["dtype"] == "torch.float32"
    assert result["count"] == result["finite_count"] == 2
    assert result["all_finite"] and result["nonfinite_count"] == 0
    assert result["rms"] == amplitude and result["rms"] > 0.
    assert result["min"] == -amplitude and result["max"] == amplitude
    assert result["mean"] == 0.
    assert tensor.grad is None and _plain(result)
    json.dumps(result, allow_nan=False)


def test_nonfinite_values_are_counted_not_imputed_into_statistics():
    tensor = torch.tensor([3., -3., float("nan"), float("inf"), -float("inf")])
    result = diagnostics.tensor_stats(tensor)
    assert result["count"] == 5 and result["finite_count"] == 2
    assert result["nonfinite_count"] == 3 and not result["all_finite"]
    assert result["nan_count"] == result["posinf_count"] == result["neginf_count"] == 1
    assert result["statistics_scope"] == "finite_entries_only"
    assert result["mean"] == 0. and result["rms"] == 3.
    assert result["min"] == -3. and result["max"] == 3.
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("values", [[], [float("nan"), float("inf")]])
def test_empty_or_all_nonfinite_statistics_are_null(values):
    result = diagnostics.tensor_stats(torch.tensor(values))
    assert result["finite_count"] == 0
    assert all(result[key] is None for key in ("min", "max", "mean", "rms"))
    assert result["all_finite"] == (len(values) == 0)
    json.dumps(result, allow_nan=False)


def test_scaled_rms_handles_large_finite_float64_without_square_overflow():
    result = diagnostics.tensor_stats(torch.tensor([1e300, -1e300], dtype=torch.float64))
    assert result["rms"] == 1e300 and result["mean"] == 0.
    json.dumps(result, allow_nan=False)


class _TinyCompatibleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(3, 4, dtype=torch.float64)
        self.initial_input = torch.nn.Linear(3, 4, dtype=torch.float64)
        self.blocks = torch.nn.ModuleList([
            torch.nn.Linear(4, 4, dtype=torch.float64), torch.nn.Tanh()])
        self.head = torch.nn.Linear(8, 2, dtype=torch.float64)
        self.fail_before_head = False
        with torch.no_grad():
            for index, parameter in enumerate(self.parameters()):
                parameter.fill_(.02 * (index + 1))

    def forward(self, field, initial_context):
        hidden = self.input(field) + self.initial_input(initial_context)[:, None, :]
        for block in self.blocks:
            hidden = block(hidden)
        if self.fail_before_head:
            raise RuntimeError("synthetic partial forward")
        pooled = torch.cat((hidden.mean(1), hidden.square().mean(1)), dim=1)
        return self.head(pooled)


def _inputs():
    return (torch.arange(18, dtype=torch.float64).reshape(2, 3, 3).div(20).requires_grad_(True),
            torch.ones((2, 3), dtype=torch.float64, requires_grad=True))


def _hook_counts(model):
    return [(len(module._forward_hooks), len(module._forward_pre_hooks)) for module in model.modules()]


def test_probe_keeps_outputs_all_gradients_rng_and_mode_unchanged():
    model = _TinyCompatibleModel().train()
    field, initial = _inputs()
    expected_output = model(field, initial)
    expected_grads = torch.autograd.grad(expected_output.square().sum(),
                                         (field, initial, *model.parameters()))
    observed_field, observed_initial = _inputs()
    old_hooks = _hook_counts(model)
    random_state = torch.get_rng_state().clone()
    probe = diagnostics.ActivationProbe(model)
    with probe:
        output = model(observed_field, observed_initial)
        grads = torch.autograd.grad(output.square().sum(),
                                    (observed_field, observed_initial, *model.parameters()))
        snapshot = probe.snapshot()
    assert torch.equal(output, expected_output)
    assert all(torch.equal(left, right) for left, right in zip(grads, expected_grads))
    assert torch.equal(torch.get_rng_state(), random_state) and model.training
    assert _hook_counts(model) == old_hooks
    assert snapshot["forward_calls"] == 1
    assert set(snapshot["activations"]) == {"input", "initial_input", "blocks.0", "blocks.1", "head"}
    assert snapshot["activations"]["head"]["input"]["shape"] == [2, 8]
    assert snapshot["activations"]["head"]["output"] == diagnostics.tensor_stats(output)
    assert all(set(row) == {"input", "output"} for row in snapshot["activations"].values())
    assert _plain(snapshot) and _plain(probe._activations)
    json.dumps(snapshot, allow_nan=False)


def test_latest_forward_replaces_previous_snapshot_and_snapshot_is_independent():
    model = _TinyCompatibleModel().eval()
    field, initial = _inputs()
    with diagnostics.ActivationProbe(model) as probe:
        assert probe.snapshot() == {"forward_calls": 0, "activations": {}}
        model(field, initial)
        first = probe.snapshot()
        model(field[:1], initial[:1])
        second = probe.snapshot()
        assert second["forward_calls"] == 2
        assert second["activations"]["input"]["input"]["shape"] == [1, 3, 3]
        assert first["activations"]["input"]["input"]["shape"] == [2, 3, 3]
        second["activations"].clear()
        assert probe.snapshot()["activations"]
    closed = probe.snapshot()
    model(field, initial)
    assert probe.snapshot() == closed and not model.training


def test_exception_removes_hooks_and_retains_only_current_partial_forward():
    model = _TinyCompatibleModel()
    field, initial = _inputs()
    old_hooks = _hook_counts(model)
    probe = diagnostics.ActivationProbe(model)
    with pytest.raises(RuntimeError, match="synthetic partial forward"):
        with probe:
            model(field, initial)
            model.fail_before_head = True
            model(field, initial)
    assert _hook_counts(model) == old_hooks
    partial = probe.snapshot()
    assert partial["forward_calls"] == 2
    assert "blocks.1" in partial["activations"] and "head" not in partial["activations"]
    assert _plain(partial)
    model.fail_before_head = False
    with probe:
        model(field, initial)
        assert probe.snapshot()["forward_calls"] == 1
    assert _hook_counts(model) == old_hooks
