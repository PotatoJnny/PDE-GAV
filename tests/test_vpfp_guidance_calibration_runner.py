"""Synthetic runner checks; not evidence of real Guidance effectiveness."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from run_vpfp_guidance_calibration import (
    ROOT, SIGMAS, SEEDS, VARIANTS, applied_displacement, load_config, noise_seed,
    training_rows_by_stratum, prepare_effect_config)
import run_vpfp_guidance_calibration as runner


def fixture_config(tmp_path):
    cfg = json.loads((ROOT / "configs/vpfp_guidance_calibration_20260908.json").read_text())
    prior = tmp_path / "prior.json"
    prior.write_text("{}")
    cfg["prior_effect"] = {"path": str(prior), "sha256": runner.file_sha(prior)}
    return cfg


def save_config(tmp_path, cfg):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return path


def test_preregistered_inventory_and_independent_seeds(tmp_path):
    cfg = fixture_config(tmp_path)
    assert load_config(save_config(tmp_path, cfg)) == cfg
    assert len(VARIANTS) == 4 and len(SEEDS) == 2 and len(SIGMAS) == 5


@pytest.mark.parametrize("key,value", [
    ("steps", 500), ("steps", 2000.0), ("seeds", [1, 2]), ("variants", ["combined"]),
    ("validation_sigmas", [1.]), ("validation_replicates", 1),
    ("strength", 20), ("bootstrap_replicates", 1000),
    ("calibration_seed", True), ("validation_seed", -1)])
def test_changed_recipe_rejected(tmp_path, key, value):
    cfg = fixture_config(tmp_path); cfg[key] = value
    with pytest.raises(ValueError):
        load_config(save_config(tmp_path, cfg))


def test_calibration_and_validation_stream_may_not_alias(tmp_path):
    cfg = fixture_config(tmp_path); cfg["validation_seed"] = cfg["calibration_seed"]
    with pytest.raises(ValueError, match="independent"):
        load_config(save_config(tmp_path, cfg))


def test_historical_evidence_hash_bound(tmp_path):
    cfg = fixture_config(tmp_path)
    Path(cfg["prior_effect"]["path"]).write_text('{"changed":true}')
    with pytest.raises(ValueError, match="historical"):
        load_config(save_config(tmp_path, cfg))


def test_noising_stream_independent_of_order_and_variant():
    keys = [(r, s, n) for r in ("validation-0000", "validation-0001") for s in SIGMAS for n in range(2)]
    values = {k: noise_seed(17, *k) for k in keys}
    assert values == {k: noise_seed(17, *k) for k in reversed(keys)}
    assert len(set(values.values())) == len(keys)
    assert all(0 <= value < 2**63 for value in values.values())


def test_actual_fp32_effect_not_nonzero_math_gradient():
    baseline = torch.ones((2, 5), dtype=torch.float32)
    gradient = torch.tensor([[1.e-30]*5, [1.e-5]*5], dtype=torch.float32)
    delta, norm = applied_displacement(baseline, gradient, torch.ones(2))
    assert not delta[0].any()
    assert delta[1].any()
    assert gradient[0].any()  # Mathematical nonzero does not imply FP32 sampling change.
    assert norm.dtype == torch.float32


def test_exact_production_clip_multiply_then_add_order():
    base = torch.tensor([[.2, 1., -2.], [4., -5., .6]], dtype=torch.float32)
    gradient = torch.tensor([[3., 4., .1], [.01, -.02, .03]], dtype=torch.float32)
    sigma = torch.tensor([1., .05])
    norm = torch.linalg.vector_norm(gradient, dim=1, keepdim=True)
    scale = torch.where(norm > 1., 1./norm.clamp_min(torch.finfo(norm.dtype).tiny), torch.ones_like(norm))
    applied = gradient.detach()*scale
    expected = (base+2.*sigma[:, None].square()*applied)-base
    observed, _ = applied_displacement(base, gradient, sigma)
    assert torch.equal(observed, expected)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_gradient_is_not_counted_as_effect(value):
    with pytest.raises(FloatingPointError):
        applied_displacement(torch.ones(1, 3), torch.full((1, 3), value), torch.ones(1))


def test_stratum_baselines_use_fit_labels_only():
    roots = [f"fit-{i:04d}" for i in range(64)]
    plan = {"roots": {"fit": [{"slot_id": root, "stratum": i % 8} for i, root in enumerate(roots)],
                      "validation": [{"poison": "must not be read"}]}}
    targets = torch.arange(64, dtype=torch.float64)/64
    result = training_rows_by_stratum({"root_ids": roots}, plan, targets)
    assert result == {str(s): float(targets[s::8].mean()) for s in range(8)}


def test_effect_handoff_freezes_selected_only_and_validates_before_solver(tmp_path, monkeypatch):
    cfg = fixture_config(tmp_path)
    selected = {"checkpoint": {"path": "new.pt", "sha256": "1"*64},
                "training_report": {"path": "new.json", "sha256": "2"*64}}
    template = {"guide": {"path": "old.pt"}, "training_plan": {"path": "old-plan.json"},
                "training_labels": {"path": "old-labels.json"}}
    original = deepcopy(template)
    seen = []
    def verify(path):
        seen.append(json.loads(path.read_text()))
        return None, None, [{"root": i} for i in range(32)], None
    monkeypatch.setattr(runner, "load_effect_config", verify)
    result = prepare_effect_config(template, cfg, selected, tmp_path)
    assert template == original and len(seen) == 1 and result["roots"] == 32
    assert seen[0]["guide"] == selected["checkpoint"]
    assert seen[0]["prior_effect"] == cfg["prior_effect"]
    assert seen[0]["fresh_split_seeds"] == cfg["fresh_split_seeds"]
    assert seen[0]["training_labels"] == template["training_labels"]
    assert result["sha256"] == runner.file_sha(result["path"])
