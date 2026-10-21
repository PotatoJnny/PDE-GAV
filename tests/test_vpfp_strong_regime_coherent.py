"""Mock strong orchestration plus tiny factory controls; never submit a job."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.util.spec_from_file_location("strong_coherent_under_test", ROOT / "scripts/run_vpfp_strong_regime_coherent.py")
driver = importlib.util.module_from_spec(loader)
loader.loader.exec_module(driver)
loader = importlib.util.spec_from_file_location("coherent_prior_test_fixture", Path(__file__).with_name("test_vpfp_initial_conditioned_diffusion.py"))
fixture = importlib.util.module_from_spec(loader)
loader.loader.exec_module(fixture)


@pytest.fixture(autouse=True)
def bounded_threads():
    count = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(count)


def args(tmp_path, *extra):
    return driver.parser().parse_args(["--config", str(tmp_path / "config.json"),
        "--source", str(tmp_path / "source"), "--pilot", str(tmp_path / "pilot"),
        "--output", str(tmp_path / "vpfp-strong-regime-coherent-123"), *extra])


def test_fixed_budgets_and_preflight_cli_contract(tmp_path):
    value = args(tmp_path, "--preflight-only")
    assert value.preflight_only
    assert (value.width, value.depth, value.batch_size, value.learning_rate) == (256, 4, 4, 2.e-4)
    assert value.train_seconds == 2400 and value.maximum_wall_seconds == 8*3600
    assert value.generated_answers == 128 and value.sample_steps == 64
    assert driver.CONTINUATION_STEPS == (30000, 40000) and driver.NEW_STEPS == 20000


def test_covariance_rho_changes_only_normalizer_geometry_not_network_initialization(tmp_path, monkeypatch):
    config, raw, _ = fixture.data(torch.float32)
    prior = fixture.prior_spec(config).to_payload()
    manifest = {"model_spec": prior["model_spec"], "manifest_sha256": prior["corpus_manifest_sha256"], "provenance": prior}
    spec = driver.previous.VPFPOnlineModelSpec.from_mapping(prior["model_spec"])
    arguments = args(tmp_path)
    monkeypatch.setattr(driver.shared, "model_config", lambda *a: config.full_config)
    values = []
    for arm in ("initial_conditioned", driver.RHO_ARM):
        initialization = {}
        cfg, make, fit = driver.factories(arguments, spec, manifest, arm, initialization)
        model = make(cfg)
        norm = fit(cfg, raw, sample_ids=list(range(8)))
        values.append((model, norm, initialization))
    first, second = values
    assert driver.shared.model_state_sha(first[0]) == driver.shared.model_state_sha(second[0])
    assert first[2] == second[2]
    assert first[1].temporal.rho_policy == "identity" and second[1].temporal.rho_policy == "covariance"
    assert torch.equal(first[1].temporal.parent.group_scales, second[1].temporal.parent.group_scales)
    assert torch.equal(first[1].square_root[1:], second[1].square_root[1:])
    assert torch.equal(first[1].regression[1:], second[1].regression[1:])
    assert not torch.equal(first[1].regression[0], second[1].regression[0])


def test_training_orchestrates_four_restores_two_new_runs_and_exact_20k_exports(tmp_path, monkeypatch):
    arguments = args(tmp_path)
    arguments.output.mkdir()
    calls = []
    inputs = {f"seed-{replica}/{arm}": {"last": str(tmp_path / f"last-{replica}-{arm}.pt"),
        "best": str(tmp_path / f"best-{replica}-{arm}.pt"), "last_sha256": "a"*64,
        "best_sha256": "b"*64, "normalizer_sha256": "c"*64}
        for replica in range(2) for arm in driver.old.ARMS}
    def factories(*values):
        values[-1].update(common_trunk_sha256="same", parameter_count=10)
        return SimpleNamespace(), lambda cfg: None, lambda *a: None
    monkeypatch.setattr(driver, "factories", factories)
    monkeypatch.setattr(driver.shared, "_read_training_asset", lambda *a: {
        "schema": "tiny", "config": {}, "model": {}, "normalizer": {}, "base_decoder_sha256": "d"*64,
        "corpus_manifest_sha256": "e"*64, "training": {"step": 20000}, "continuation_state": {"secret": "not_native"}})
    def train(arguments, spec, corpus, output, **factories):
        calls.append((deepcopy(arguments), factories))
        checkpoint = output / "checkpoints"
        checkpoint.mkdir()
        for name in ("best.pt", "last.pt"):
            (checkpoint / name).write_text("tiny saved training state")
        receipts = []
        for step in arguments.checkpoint_steps:
            path = checkpoint / f"fixed-step-{step:06d}.pt"
            path.write_text("tiny fixed state")
            receipts.append({"step": step, "checkpoint": str(path), "sha256": driver.shared.file_sha(path)})
        result = {"final_step": arguments.steps, "stop_reason": "step_limit", "normalizer_sha256": "c"*64,
            "step_snapshots": receipts, "unreached_checkpoint_steps": [], "checkpoint": str(checkpoint / "best.pt"),
            "checkpoint_sha256": driver.shared.file_sha(checkpoint / "best.pt")}
        return checkpoint / "best.pt", result
    monkeypatch.setattr(driver.shared, "train", train)
    training, models, frozen = driver.train_models(arguments, None, {}, inputs, "p"*64, lambda _: None, lambda: None)
    assert len(training) == len(calls) == 6 and len(models) == 14
    for run, hooks in calls:
        assert "normalizer_loader" in hooks
        if run.ablation_metadata["arm"] == driver.RHO_ARM:
            assert run.steps == 20000 and run.checkpoint_steps == (20000,)
            assert not hasattr(run, "resume_checkpoint")
        else:
            assert run.steps == 40000 and run.checkpoint_steps == (30000, 40000)
            assert run.legacy_resume_contract["source_script_sha256"] == driver.LEGACY_TRAINER_SHA
    starts = [entry for entry in models.values() if entry["checkpoint_role"].startswith("exact_20000")]
    assert len(starts) == 4
    for entry in starts:
        payload = torch.load(entry["checkpoint"], weights_only=True)
        assert len(payload) == 7 and "continuation_state" not in payload
        assert payload["training"]["step"] == 20000 and payload["training"]["parent_last_sha256"] == "a"*64
    assert len(frozen) == 26  # fourteen curve states plus six best and six last


@pytest.fixture
def orchestration(tmp_path, monkeypatch):
    arguments = args(tmp_path)
    arguments.source.mkdir()
    arguments.pilot.mkdir()
    corpus = arguments.source / "corpus"
    corpus.mkdir()
    for path in (arguments.config, corpus / "manifest.json", corpus / "seeds.json"):
        path.write_text(json.dumps({"vpfp": {}}))
    manifest = {"manifest_sha256": "m"*64, "model_spec": {}, "split_counts": driver.old.COUNTS,
        "seed_plan": {"path": "seeds.json"}}
    spec = SimpleNamespace(preflight=lambda: None)
    events = []
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path))
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    monkeypatch.setattr(driver.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(driver.old, "SCRATCH", tmp_path)
    monkeypatch.setattr(driver.subprocess, "check_output", lambda *a, **k: "synthetic-revision")
    monkeypatch.setattr(driver.previous.VPFPOnlineModelSpec, "from_mapping", lambda *a: spec)
    monkeypatch.setattr(driver.previous, "read_manifest", lambda *a, **k: manifest)
    monkeypatch.setattr(driver.old, "check_completed_source", lambda *a: {})
    monkeypatch.setattr(driver.old, "check_baseline_recipe", lambda *a: {})
    monkeypatch.setattr(driver.old, "reference_inventory", lambda *a: {})
    monkeypatch.setattr(driver.shared, "source_fingerprints", lambda: {})
    monkeypatch.setattr(driver, "source_inventory", lambda *a: ({"seed-0/cov_base": {"best_step": 19000}}, {}, {}))
    monkeypatch.setattr(driver, "preflight", lambda *a: events.append("train_only_preflight") or {"mechanics": True})
    monkeypatch.setattr(driver.previous, "audit_value_control_corpus", lambda *a, **k: events.append("encoding") or {})
    monkeypatch.setattr(driver.previous, "require_value_control_audit", lambda *a: None)
    monkeypatch.setattr(driver, "train_models", lambda *a: events.append("all_models_frozen") or ({"tiny": {"wall_seconds": 1.}}, {}, {}))
    monkeypatch.setattr(driver, "evaluate_models", lambda *a: events.append("features_opened") or {"tiny": {}})
    cli_args = ["--config", str(arguments.config), "--source", str(arguments.source), "--pilot", str(arguments.pilot),
                "--output", str(arguments.output)]
    return SimpleNamespace(args=arguments, cli_args=cli_args, events=events, tmp_path=tmp_path)


def test_preflight_only_nested_output_has_complete_result_and_never_trains(orchestration):
    run = orchestration
    output = run.tmp_path / "combined-job/strong-preflight"
    argv = run.cli_args[:-1] + [str(output), "--preflight-only"]
    result = driver.main(argv)
    assert result["execution_passed"] and result["preflight_only"]
    assert run.events == ["train_only_preflight"]
    assert result["training"] == {} and result["audits"] == {}
    assert (output / "frozen-plan.json").exists() and (output / "result.json").exists()


def test_full_run_orders_all_freezes_before_feature_reading(orchestration):
    result = driver.main(orchestration.cli_args)
    assert result["execution_passed"] and result["requested_generation_count"] == 1792
    assert orchestration.events == ["train_only_preflight", "encoding", "all_models_frozen", "features_opened"]
    assert result["maximum_gpu_hours"] == 8 and result["generated_cases_per_state"] == 128
    assert result["training_wall_seconds"] == 1.


def test_failed_preflight_preserves_reason_and_never_trains(orchestration, monkeypatch):
    def fail(*args):
        raise RuntimeError("synthetic rho gate failure")
    monkeypatch.setattr(driver, "preflight", fail)
    with pytest.raises(RuntimeError, match="synthetic rho"):
        driver.main(orchestration.cli_args)
    result = json.loads((orchestration.args.output / "result.json").read_text())
    assert not result["execution_passed"] and result["failed_stage"] == "train_only_new_rho_preflight"
    assert result["training"] == {} and "synthetic rho gate failure" in result["error"]


def test_nested_training_output_still_rejected(orchestration):
    argv = orchestration.cli_args[:-1] + [str(orchestration.tmp_path / "nested/full-training")]
    with pytest.raises(SystemExit):
        driver.main(argv)


def test_no_slurm_rejects_before_output(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    value = args(tmp_path)
    with pytest.raises(SystemExit):
        driver.main(["--config", str(value.config), "--source", str(value.source),
            "--pilot", str(value.pilot), "--output", str(value.output)])
    assert not value.output.exists()
