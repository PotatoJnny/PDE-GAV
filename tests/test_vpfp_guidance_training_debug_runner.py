"""Small synthetic debug-runner checks, not Guidance training validation."""

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F


_PATH = Path(__file__).resolve().parents[1] / "scripts/run_vpfp_guidance_training_debug.py"
_SPEC = importlib.util.spec_from_file_location("vpfp_guidance_training_debug_runner_test", _PATH)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def test_gradient_summary_preserves_tiny_fp32_norm_and_groups_unused_parameters():
    tiny = 2. ** -100
    named = [("input.weight", torch.nn.Parameter(torch.zeros(2, 2))),
             ("input.bias", torch.nn.Parameter(torch.zeros(2))),
             ("head.weight", torch.nn.Parameter(torch.zeros(2))),
             ("head.bias", torch.nn.Parameter(torch.zeros(1)))]
    gradients = [torch.tensor([[tiny, 0.], [-tiny, 2 * tiny]], dtype=torch.float32),
                 None, torch.zeros(2), torch.tensor([3 * tiny], dtype=torch.float32)]
    result = runner.gradient_summary(named, gradients)
    assert result["l2_fp64"] > 0.
    assert math.isclose(result["l2_fp64"], math.sqrt(15.) * tiny, rel_tol=1e-14, abs_tol=0.)
    assert set(result["groups"]) == {"input", "head"}
    inputs, head = result["groups"]["input"], result["groups"]["head"]
    assert inputs["elements"] == 6 and inputs["nonzero_elements"] == 3
    assert inputs["max_abs"] == 2 * tiny
    assert math.isclose(inputs["l2"], math.sqrt(6.) * tiny, rel_tol=1e-14, abs_tol=0.)
    assert head == {"elements": 3, "nonzero_elements": 1, "max_abs": 3 * tiny, "l2": 3 * tiny}
    assert all(parameter.grad is None for _, parameter in named)
    json.dumps(result, allow_nan=False)


def test_all_unused_gradients_are_zero_without_losing_element_counts():
    named = [("head.bias", torch.nn.Parameter(torch.zeros(2)))]
    result = runner.gradient_summary(named, [None])
    assert result == {"l2_fp64": 0., "groups": {"head": {
        "elements": 2, "nonzero_elements": 0, "max_abs": 0., "l2": 0.}}}


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_diagnostic_gradient_is_explicitly_rejected(value):
    named = [("input.weight", torch.nn.Parameter(torch.zeros(1)))]
    with pytest.raises(FloatingPointError, match="nonfinite diagnostic gradient"):
        runner.gradient_summary(named, [torch.tensor([value])])


class _TwoOutputSharedLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(validity_weight=.1)
        self.input = torch.nn.Linear(2, 3, dtype=torch.float64)
        self.head = torch.nn.Linear(3, 2, dtype=torch.float64)
        self.unused = torch.nn.Parameter(torch.tensor([.7], dtype=torch.float64))
        with torch.no_grad():
            self.input.weight.copy_(torch.tensor([[.1, .3], [-.2, .4], [.5, -.1]]))
            self.input.bias.copy_(torch.tensor([.1, -.2, .3]))
            self.head.weight.copy_(torch.tensor([[.3, -.4, .2], [-.2, .5, .1]]))
            self.head.bias.copy_(torch.tensor([.1, -.1]))

    def forward(self, values):
        logits = self.head(self.input(values))
        return {"reward": logits[:, 0].sigmoid(), "validity_logit": logits[:, 1]}


def _loss(prediction, targets, valid):
    return ((prediction["reward"][valid] - targets[valid]).square().mean()
            + .1 * F.binary_cross_entropy_with_logits(prediction["validity_logit"], valid.double()))


@pytest.mark.parametrize("preexisting_gradients", [False, True])
def test_loss_comparison_preserves_backward_and_reports_already_weighted_validity(preexisting_gradients):
    model = _TwoOutputSharedLinear()
    baseline = deepcopy(model)
    inputs = torch.tensor([[.2, -.5], [.1, .7], [-.4, .3]], dtype=torch.float64)
    targets = torch.tensor([.1, float("nan"), .6], dtype=torch.float64)
    valid = torch.tensor([True, False, True])
    if preexisting_gradients:
        for candidate in (model, baseline):
            for parameter in candidate.parameters():
                parameter.grad = torch.full_like(parameter, .25)
    existing = [None if parameter.grad is None else parameter.grad.clone() for parameter in model.parameters()]
    baseline_prediction = baseline(inputs)
    _loss(baseline_prediction, targets, valid).backward()
    prediction = model(inputs)
    parameters = list(model.parameters())
    unweighted = F.binary_cross_entropy_with_logits(prediction["validity_logit"], valid.double())
    raw_gradients = torch.autograd.grad(unweighted, parameters, retain_graph=True, allow_unused=True)
    raw_l2 = math.sqrt(sum(float(gradient.detach().double().square().sum())
                           for gradient in raw_gradients if gradient is not None))
    result = runner.loss_gradient_comparison(model, prediction, targets, valid)
    # Observing the losses may not populate, clear, or accumulate parameter.grad.
    for parameter, previous in zip(parameters, existing):
        if previous is None:
            assert parameter.grad is None
        else:
            assert torch.equal(parameter.grad, previous)
    assert result["reward"]["l2_fp64"] > 0.
    assert result["weighted_validity"]["l2_fp64"] == pytest.approx(.1 * raw_l2, rel=1e-12, abs=0.)
    assert result["weighted_validity_over_reward"] == pytest.approx(
        .1 * raw_l2 / result["reward"]["l2_fp64"], rel=1e-12, abs=0.)
    assert -1. <= result["cosine"] <= 1.
    assert set(result["reward"]["groups"]) == {"input", "head", "unused"}
    assert result["reward"]["groups"]["unused"]["l2"] == 0.
    json.dumps(result, allow_nan=False)
    # The exact original combined objective still has its graph and produces
    # the same gradients, including accumulation on existing gradients.
    _loss(prediction, targets, valid).backward()
    for observed, expected in zip(model.parameters(), baseline.parameters()):
        if expected.grad is None:
            assert observed.grad is None
        else:
            assert torch.equal(observed.grad, expected.grad)


@pytest.mark.parametrize("job_id,scratch", [(None, None), ("123", None), ("invalid", "/tmp")])
def test_debug_run_refuses_no_slurm_before_assets_subprocess_or_output(tmp_path, monkeypatch, job_id, scratch):
    for key, value in (("SLURM_JOB_ID", job_id), ("SLURM_TMPDIR", scratch)):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    def forbidden(*args, **kwargs):
        pytest.fail("allocation gate must precede assets, subprocesses, and output")

    monkeypatch.setattr(runner, "file_sha", forbidden)
    monkeypatch.setattr(runner, "write", forbidden)
    monkeypatch.setattr(runner.subprocess, "check_output", forbidden)
    output = tmp_path / "must-not-exist"
    args = SimpleNamespace(config=tmp_path / "no-assets.json", output=output, wall_seconds=300.)
    with pytest.raises(ValueError, match="Slurm compute allocation required"):
        runner.run(args)
    assert not output.exists()
