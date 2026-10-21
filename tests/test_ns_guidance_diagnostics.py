"""Numerical and synchronization checks for frozen NS reward guidance."""

import importlib
from math import sqrt
from pathlib import Path

import pytest


@pytest.fixture
def guidance(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "legacy" / "ns" / "src")
    )
    module = importlib.import_module(
        "agent_verifier.neural_generator.structured_ns_guidance"
    )

    class QuadraticReward(torch.nn.Module):
        def forward(self, values, time):
            return 0.5 * values.square().sum(dim=1)

    return torch, module.StructuredRewardGradient(
        QuadraticReward(), strength=0.5, max_flow_ratio=0.25
    )


def test_guidance_preserves_capped_updates_and_cumulative_diagnostics(guidance):
    torch, gradient = guidance
    before = gradient.diagnostics()
    assert before["calls"] == before["sample_evaluations"] == 0
    assert before["degenerate_gradient_evaluations"] == 0

    values = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    base = torch.tensor([[1.0, 1.0], [3.0, 4.0]])
    update = gradient(values, torch.zeros(2), base)
    torch.testing.assert_close(
        update, torch.tensor([[0.0, 0.0], [0.75, 1.0]]), rtol=0, atol=0
    )
    first = gradient.diagnostics()
    assert first["degenerate_gradient_evaluations"] == 1

    values = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, -1.0]])
    base = torch.tensor([[1.0, 1.0], [2.0, 2.0], [4.0, -4.0]])
    update = gradient(values, torch.ones(3), base)
    torch.testing.assert_close(
        update, torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, -1.0]]),
        rtol=0, atol=0,
    )
    report = gradient.diagnostics()
    assert report["calls"] == 2
    assert report["sample_evaluations"] == 5
    assert report["degenerate_gradient_evaluations"] == 3
    assert type(report["degenerate_gradient_evaluations"]) is int
    assert report["mean_applied_flow_ratio"] == pytest.approx(0.1)
    assert report["max_applied_flow_ratio"] == pytest.approx(0.25)
    assert report["mean_reward_gradient_rms"] == pytest.approx((sqrt(12.5) + 1) / 5)
    assert report["mean_predicted_normalized_reward"] == pytest.approx(2.7)
    assert gradient.diagnostics() == report  # Reading does not reset the counters.


def test_guidance_stage_reads_only_the_immediate_finiteness_check(guidance):
    torch, gradient = guidance
    values = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    with torch.autograd.profiler.profile(use_cuda=False) as profile:
        gradient(values, torch.zeros(2), torch.ones_like(values))
    scalar_reads = sum(
        event.count for event in profile.key_averages()
        if event.key == "aten::_local_scalar_dense"
    )
    # The safety check still reads one scalar. Diagnostics must not add a
    # second host read on each stage of a hundreds-of-steps sampling request.
    assert scalar_reads == 1
    assert gradient.diagnostics()["degenerate_gradient_evaluations"] == 1


def test_guidance_nonfinite_update_still_fails_before_recording_diagnostics(guidance):
    torch, gradient = guidance
    before = gradient.diagnostics()
    with pytest.raises(FloatingPointError, match="became non-finite"):
        gradient(torch.ones((1, 2)), torch.zeros(1), torch.full((1, 2), float("inf")))
    assert gradient.diagnostics() == before
