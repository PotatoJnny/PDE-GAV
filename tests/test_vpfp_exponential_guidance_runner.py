"""Synthetic runner/sampler checks, not VPFP solver-effect evidence."""
from copy import deepcopy
import json

import pytest
import torch

import run_vpfp_exponential_guidance as runner
from pde_agent_verifier_solver.plugins.vpfp import solver_error_guidance as legacy
from pde_agent_verifier_solver.plugins.vpfp import exponential_solver_error_guidance as exponential
from tests.test_vpfp_solver_error_guidance import native


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(tmp_path):
    cfg = json.loads((runner.ROOT / "configs/vpfp_exponential_guidance_20260908.json").read_text())
    prior = tmp_path / "prior.json"; prior.write_text("{}")
    cfg["prior_effects"] = [{"path": str(prior), "sha256": runner.file_sha(prior)}]
    return cfg


def save(tmp_path, cfg):
    path = tmp_path / "config.json"; path.write_text(json.dumps(cfg)); return path


def test_fixed_config(tmp_path):
    cfg = config(tmp_path)
    assert runner.load_config(save(tmp_path, cfg)) == cfg


@pytest.mark.parametrize("key,value", [("steps", 2000.0), ("steps", 1000), ("beta", 4.),
    ("external_strength", 2.), ("seed", 3120260909), ("validation_seed", 1), ("calibration_seed", 1)])
def test_changed_fixed_controls_rejected(tmp_path, key, value):
    cfg = config(tmp_path); cfg[key] = value
    with pytest.raises(ValueError):
        runner.load_config(save(tmp_path, cfg))


def test_prior_hash_and_root_stream_collisions_rejected(tmp_path):
    cfg = config(tmp_path); cfg["fresh_split_seeds"]["fit"] = cfg["seed"]
    with pytest.raises(ValueError, match="independent"):
        runner.load_config(save(tmp_path, cfg))
    cfg = config(tmp_path); cfg["prior_effects"][0]["sha256"] = "0"*64
    with pytest.raises(ValueError, match="evidence"):
        runner.load_config(save(tmp_path, cfg))


def test_effect_build_keeps_reference_and_excludes_both_prior_tests_without_MSE_gate(tmp_path):
    cfg = config(tmp_path)
    template = {"guide": {"path": "old.pt", "sha256": "a"*64},
        "training_report": {"path": "old.json", "sha256": "b"*64},
        "training_labels": {"path": "labels.json", "sha256": "c"*64},
        "prior_effect": {"path": "first.json", "sha256": "d"*64}}
    before = deepcopy(template)
    guide = {"path": "new.pt", "sha256": "e"*64}
    training = {"path": "new.json", "sha256": "f"*64}
    result = runner.build_effect_config(template, cfg, guide, training)
    assert template == before
    assert result["guide"] == guide and result["reference_guide"] == template["guide"]
    assert result["training_report"] == training
    assert result["beta"] == 2. and result["strength"] == 1. and result["reference_strength"] == 2.
    assert result["prior_effect"] == template["prior_effect"] and result["prior_effects"] == cfg["prior_effects"]
    assert result["selection_note"]["regression_metrics_are_diagnostics_only"] is True
    assert result["selection_note"]["primary_endpoint"] == "new_vs_base_actual_solver_error"


def test_full_tensor_hash_detects_equal_norm_different_gradient():
    x, y = torch.tensor([1., 2.]), torch.tensor([-1., 2.])
    assert torch.linalg.vector_norm(x) == torch.linalg.vector_norm(y)
    assert runner.tensor_sha(x) != runner.tensor_sha(y)
    assert runner.tensor_sha(x) == runner.tensor_sha(x.clone())
    assert runner.tensor_sha(x) != runner.tensor_sha(x.double())
    assert runner.tensor_sha(x) != runner.tensor_sha(x[None])


def fitted_native(monkeypatch):
    base, old, context, future, theta, _ = native(monkeypatch)
    guide = exponential.make_exponential_regressor(old.config, parameter_dtype=next(base.model.parameters()).dtype)
    rows = {"future_white": future.repeat(32, 1), "initial_context": context.repeat(32, 1),
        "conditions": theta.repeat(32, 1), "errors": torch.linspace(.01, .08, 64, dtype=torch.float64),
        "valid": torch.ones(64, dtype=torch.bool), "root_ids": [f"fit-{i:04d}" for i in range(64)],
        "excluded_rows": []}
    exponential.fit_exponential_calibration(guide, rows, seed=11)
    base.model.eval().requires_grad_(False)
    return base, guide, context, future, theta


def test_production_sampler_rejects_double_beta_and_binds_runtime(monkeypatch):
    base, guide, context, _, theta = fitted_native(monkeypatch)
    with pytest.raises(ValueError, match="already includes beta"):
        legacy.GuidedIC40Sampler(base, guide, strength=2.)
    sampler = legacy.GuidedIC40Sampler(base, guide, strength=1.)
    assert sampler.identity["guidance"]["beta_applied_once"] is True
    assert sampler.identity["guidance"]["training_beta"] == 2.
    sampler.assert_frozen()
    before = guide.reward_center
    guide.reward_center = before+.01
    with pytest.raises(ValueError, match="frozen"):
        sampler.assert_frozen()


def test_constant_log_h_samples_exact_base_without_reward_fields(monkeypatch):
    base, guide, context, _, theta = fitted_native(monkeypatch)
    with torch.no_grad():
        guide.head[-1].weight[0].zero_()
        guide.head[-1].bias[0].fill_(.03)
    model_sha = legacy._module_sha(base.model)
    zero = legacy.GuidedIC40Sampler(base, guide, strength=0.)
    guided = legacy.GuidedIC40Sampler(base, guide, strength=1.)
    a = zero.sample_future(context[:1], theta[:1], [1234])
    b = guided.sample_future(context[:1], theta[:1], [1234])
    assert torch.equal(a, b)
    assert legacy._module_sha(base.model) == model_sha
    assert guided.last_report["guided_calls"] > 0
    assert guided.last_report["nonzero_gradient_calls"] == 0
    assert guided.last_report["maximum_denoised_displacement_l2"] == 0.


def test_validation_uses_log_h_gradient_and_fingerprints_all_160_records():
    class Guide(torch.nn.Module):
        def forward(self, x, sigma, condition, *, initial_context):
            f = x.sum(1)*.02
            return {"log_desirability": f, "desirability": f.exp(),
                    "valid_probability": torch.ones_like(f)}
        def score_potential(self, prediction):
            return prediction["log_desirability"]
    cache = []
    for sigma in (.002, .05, .2, 1., 2.):
        for replicate in (0, 1):
            for i in range(16):
                cache.append({"roots": [f"validation-{i:04d}"], "strata": [i%8], "start": i,
                    "sigma_value": sigma, "replicate": replicate, "sigma": torch.tensor([sigma]),
                    "noisy": torch.tensor([[.1, .2]]), "conditions": torch.ones(1, 2),
                    "initial": torch.zeros(1, 2), "denoised": torch.ones(1, 2)})
    records, summary = runner.validation_probe(Guide(), cache, torch.ones(16),
        fit_weight_mean=1., check=lambda: None)
    assert len(records) == 160 and summary["independent_roots"] == 16
    assert summary["sigma1_effective_roots"] == 16
    assert summary["selection_or_solver_effect_gate"] is False
    assert len({r["full_gradient_sha256"] for r in records}) == 1
    assert all(r["gradient_l2_fp64"] == pytest.approx((2*.02**2)**.5) for r in records)
