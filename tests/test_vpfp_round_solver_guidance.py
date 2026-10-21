"""Synthetic round/provenance tests; these are not real Solver validation.

Numerical tests run only in the scheduled test allocation. No historical
checkpoint, label pool, Solver execution or on-line PDE solve is used here.
"""
from copy import deepcopy
from hashlib import sha256
import json

import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp import round_solver_guidance as round_guide
from pde_agent_verifier_solver.plugins.vpfp import solver_error_guidance as legacy
from tests.test_vpfp_solver_error_guidance import native


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _identity(**overrides):
    return {"source_round_index": 2, "target_round_index": 3,
        "target_solver_sha256": "a" * 64, "reward_matcher_sha256": "b" * 64,
        "controller_seed": 987654, **overrides}


def _rows(monkeypatch, *, device="cpu"):
    base, _, _, _, _, _ = native(monkeypatch, device=device)
    base.model.eval().requires_grad_(False)
    base.normalizer.eval().requires_grad_(False)
    cfg = base.model.config
    rng = torch.Generator().manual_seed(522)
    result = []
    for index in range(80):
        root = 110000 + index
        initial = torch.randn(1, cfg.context_dimension, generator=rng).to(device)
        future = torch.randn(1, cfg.latent_dimension, generator=rng).to(device)
        condition = torch.tensor([[20., 1.]], device=device)
        stream = lambda tag: int.from_bytes(sha256(f"vpfp-initial-conditioned-{tag}:{root}".encode()).digest()[:8], "little") % 2**63
        with torch.no_grad():
            full = base.normalizer.decode((initial, future))
        result.append({"case_id": f"actual-case-r2-{index}", "root_seed": root,
            "source": "base", "source_round_index": 2, "target_solver_sha256": "a" * 64,
            "valid": True, "error": .015 + (index % 17) * .003,
            "draw": {"initial_context": initial, "future_white": future,
                "full": full, "equation_conditions": condition, "root_seeds": [root],
                "known_prior_seeds": [stream("prior")], "future_noise_seeds": [stream("future")]}})
    return base, result


def _train(monkeypatch, tmp_path, *, config=None, rows=None, base=None, directory="trained", **overrides):
    monkeypatch.setenv("SLURM_JOB_ID", "999001")
    if base is None:
        base, rows = _rows(monkeypatch)
    return round_guide.train_round_solver_guidance(base_sampler=base, rows=rows,
        **_identity(**overrides), output_root=tmp_path / directory,
        config=config or round_guide.RoundGuidanceConfig(steps=2), label_source="synthetic_fixture")


@pytest.mark.parametrize("kwargs", [dict(steps=1), dict(steps=1999), dict(steps=True),
    dict(batch_size=8), dict(fit_count=63), dict(validation_count=17), dict(beta_std=1.),
    dict(target_ratio=.25), dict(max_flow_ratio=.5), dict(wall_seconds=0.), dict(wall_seconds=float("inf"))])
def test_recipe_cannot_silently_change_validated_model_or_budget(kwargs):
    with pytest.raises(ValueError):
        round_guide.RoundGuidanceConfig(**kwargs)


def test_diagnostic_updates_are_explicit_not_formal_completion():
    assert not round_guide.RoundGuidanceConfig().diagnostic_only
    for steps in (2, 3, 4):
        assert round_guide.RoundGuidanceConfig(steps=steps).diagnostic_only
    assert round_guide.RoundGuidanceConfig().wall_seconds == 1800.


def test_actual_roots_preserved_and_local_slots_are_only_calibration_aliases(monkeypatch):
    base, supplied = _rows(monkeypatch)
    prepared, manifest, private = round_guide._prepare_rows(base, supplied, _identity())
    assert len(prepared["fit"]["root_ids"]) == 64 and len(prepared["validation"]["root_ids"]) == 16
    assert prepared["fit"]["root_ids"][0] == "fit-0000"
    assert prepared["validation"]["root_ids"][0] == "validation-0000"
    assert manifest[0]["case_id"] == "actual-case-r2-0"
    assert manifest[64]["root_seed"] == 110064
    assert len({row["root_seed"] for row in manifest}) == 80
    assert not ({row["root_seed"] for row in manifest[:64]} & {row["root_seed"] for row in manifest[64:]})
    old = private[0]["future_white"].clone()
    supplied[0]["draw"]["future_white"].zero_()
    assert torch.equal(private[0]["future_white"], old)
    assert torch.equal(prepared["fit"]["future_white"][0], old[0])


@pytest.mark.parametrize("fault,match", [
    ("count", "80"), ("case", "unique actual feedback case"), ("root", "unique actual physical roots"),
    ("guided", "clean BASE"), ("perturbed", "clean BASE"), ("solver", "target_solver_sha256"),
    ("round", "source_round_index"), ("error", "finite nonnegative actual Solver error"),
    ("imputed", "unavailable"), ("stream", "future_noise_seeds"), ("full", "full answer differs"),
    ("dtype", "native FP32"), ("valid", "explicit boolean")])
def test_feedback_faults_fail_closed_without_drops(monkeypatch, fault, match):
    base, supplied = _rows(monkeypatch)
    if fault == "count": supplied.pop()
    if fault == "case": supplied[1]["case_id"] = supplied[0]["case_id"]
    if fault == "root": supplied[64]["root_seed"] = supplied[0]["root_seed"]
    if fault in ("guided", "perturbed"): supplied[0]["source"] = fault
    if fault == "solver": supplied[0]["target_solver_sha256"] = "c" * 64
    if fault == "round": supplied[0]["source_round_index"] = 1
    if fault == "error": supplied[0]["error"] = float("inf")
    if fault == "imputed": supplied[0].update(valid=False, error=999.)
    if fault == "stream": supplied[0]["draw"]["future_noise_seeds"][0] += 1
    if fault == "full": supplied[0]["draw"]["full"][0, -1] += 1.
    if fault == "dtype": supplied[0]["draw"]["future_white"] = supplied[0]["draw"]["future_white"].double()
    if fault == "valid": supplied[0]["valid"] = 1
    with pytest.raises(ValueError, match=match):
        round_guide._prepare_rows(base, supplied, _identity())


def test_invalid_roots_stay_in_attempt_inventory_with_unavailable_reward(monkeypatch):
    base, supplied = _rows(monkeypatch)
    supplied[7].update(valid=False, error=None)
    prepared, manifest, _ = round_guide._prepare_rows(base, supplied, _identity())
    assert len(manifest) == 80 and manifest[7]["error"] is None and not manifest[7]["valid"]
    assert torch.isnan(prepared["fit"]["errors"][7]) and not prepared["fit"]["valid"][7]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_actual_batch4_draw_survives_single_root_feedback_capture(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires the scheduled CUDA test allocation")
    base, supplied = _rows(monkeypatch, device=device)
    # Unlike the other provenance fixtures, these four rows run the actual
    # native sampler and batch decoder before capture. No reference asset.
    seeds = [row["root_seed"] for row in supplied[:4]]
    actual = legacy.draw_frozen_ic40(base, seeds)
    for index, row in enumerate(supplied[:4]):
        row["draw"] = {
            **{name: actual[name][index:index+1].detach().cpu().clone()
               for name in ("full", "initial_context", "future_white", "equation_conditions")},
            **{name: [int(actual[name][index])]
               for name in ("root_seeds", "known_prior_seeds", "future_noise_seeds")}}
    prepared, manifest, private = round_guide._prepare_rows(base, supplied, _identity())
    assert len(manifest) == 80
    for index in range(4):
        assert torch.equal(private[index]["full"].cpu(), actual["full"][index:index+1].cpu())
        assert torch.equal(prepared["fit"]["future_white"][index:index+1].cpu(), actual["future_white"][index:index+1].cpu())


@pytest.mark.parametrize("overrides", [dict(source_round_index=-1), dict(target_round_index=4),
    dict(controller_seed=True), dict(controller_seed=2**63), dict(target_solver_sha256="bad")])
def test_round_and_solver_provenance_are_required(overrides):
    with pytest.raises(ValueError):
        round_guide._round_identity(**_identity(**overrides))


def test_numerical_training_refuses_login_node(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Slurm"):
        round_guide.train_round_solver_guidance(base_sampler=None, rows=[], **_identity(), output_root=tmp_path / "refused")
    assert not (tmp_path / "refused").exists()


def test_constant_rewards_preserve_failed_round_and_never_make_guide(monkeypatch, tmp_path):
    base, supplied = _rows(monkeypatch)
    for row in supplied:
        row["error"] = .025
    with pytest.raises(ValueError, match="distinct valid solver errors"):
        _train(monkeypatch, tmp_path, base=base, rows=supplied)
    report = json.loads((tmp_path / "trained/training.json").read_text())
    assert not report["execution_passed"] and report["completed_step"] == 0
    assert report["stage"] == "failed" and report["failures"]
    assert (tmp_path / "trained/feedback-manifest.json").is_file()
    assert not list((tmp_path / "trained").glob("fixed-step-*.pt"))


def test_wall_allocation_never_becomes_silent_update_reduction(monkeypatch, tmp_path):
    base, supplied = _rows(monkeypatch)
    with pytest.raises(TimeoutError, match="not reduced"):
        _train(monkeypatch, tmp_path, base=base, rows=supplied,
               config=round_guide.RoundGuidanceConfig(steps=2, wall_seconds=1.e-12))
    report = json.loads((tmp_path / "trained/training.json").read_text())
    assert not report["execution_passed"] and report["target_step"] == 2


def test_real_optimizer_freezes_new_round_head_and_preserves_base_and_rng(monkeypatch, tmp_path):
    base, supplied = _rows(monkeypatch)
    before_base = legacy._module_sha(base.model)
    before_normalizer = deepcopy(base.normalizer.to_payload())
    before_rng = torch.get_rng_state().clone()
    result = _train(monkeypatch, tmp_path, base=base, rows=supplied)
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert legacy._module_sha(base.model) == before_base
    assert base.normalizer.to_payload() == before_normalizer
    assert result.report["execution_passed"] and result.report["diagnostic_only"]
    assert result.report["final_step"] == 2 and result.report["sample_presentations"] == 8
    assert result.report["label_source"] == "synthetic_fixture"
    assert result.report["solver_calls"] == 0 and not result.report["actual_solver_hardness_measured"]
    assert result.report["reload_probe_count"] == 160 and result.report["reload_full_gradient_and_displacement_hashes_exact"]
    assert result.sampler.policy.mode == "relative" and result.sampler.policy.target_ratio == .1
    result.sampler.assert_frozen()
    assert not result.guide.training and not any(p.requires_grad for p in result.guide.parameters())
    payload = torch.load(result.checkpoint_path, weights_only=True)
    assert payload["training"]["target_solver_sha256"] == "a" * 64
    assert payload["training"]["source_round_index"] == 2 and payload["training"]["target_round_index"] == 3
    assert payload["training"]["real_root_seeds"] == [row["root_seed"] for row in supplied]
    assert payload["label_manifest_sha256"] == result.manifest_sha256
    manifest = json.loads((tmp_path / "trained/feedback-manifest.json").read_text())
    assert not manifest["local_slots_are_physical_root_identity"]
    assert manifest["rows"][64]["local_slot"] == "validation-0000"
    assert (tmp_path / "trained/continuation.controller-private.pt").is_file()
    with pytest.raises(FileExistsError):
        _train(monkeypatch, tmp_path, base=base, rows=supplied)


def test_changing_solver_or_feedback_cannot_reuse_old_round_identity(monkeypatch):
    base, supplied = _rows(monkeypatch)
    with pytest.raises(ValueError, match="target_solver_sha256"):
        round_guide._prepare_rows(base, supplied, _identity(target_solver_sha256="c" * 64))
