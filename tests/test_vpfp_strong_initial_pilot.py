"""Bounded strong-pilot orchestration fixtures; no real training/data/job."""
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.util.spec_from_file_location("strong_initial_pilot_test", ROOT / "scripts/run_vpfp_strong_initial_pilot.py")
driver = importlib.util.module_from_spec(loader)
loader.loader.exec_module(driver)


def arguments(tmp_path):
    args = driver.parser().parse_args(["--config", str(tmp_path / "config.json"),
        "--source", str(tmp_path / "source"), "--output", str(tmp_path / "vpfp-strong-initial-9001")])
    args.source_revision = "synthetic"
    return args


def conditional_fixture():
    fixture_loader = importlib.util.spec_from_file_location("strong_initial_fixture", Path(__file__).with_name("test_vpfp_initial_conditioned_diffusion.py"))
    fixture = importlib.util.module_from_spec(fixture_loader)
    fixture_loader.loader.exec_module(fixture)
    return fixture


def test_fixed_pilot_budget_and_wrapper_no_submission(tmp_path):
    args = arguments(tmp_path)
    assert (args.steps, args.train_seconds, args.batch_size, args.width, args.depth) == (20000, 2400, 4, 256, 4)
    assert args.learning_rate == 2.e-4 and args.validate_every == 1000
    assert args.sample_steps == 64 and args.generated_answers == 128 and args.checkpoint_seconds == ()
    wrapper = ROOT / "scripts/run_vpfp_strong_initial.slurm"
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    text = wrapper.read_text()
    assert "#SBATCH --time=04:00:00" in text and "--gres=gpu:l40s:1" in text
    assert "#SBATCH --cpus-per-task=8" in text and "#SBATCH --mem=48G" in text
    assert "#SBATCH --job-name=vpfp_strong_initial_gate" in text
    assert text.count("${PDE_GAV_WORKTREES_ROOT}/vpfp-strong-roundtrip-gate") == 1
    assert ".worktrees/vpfp-strong-initial-pilot" not in text
    assert "vpfp-strong-roundtrip-preflight-5259322/result.json" in text
    assert "1c37c6dc2509e6fdaf1bd6bd28bec7ad00bef871274f598f6fdca952b86e3c80" in text
    assert "--seed 1620260906" in text and "run_plasma_canonical_smoke.py" in text
    assert "sbatch " not in text and "scancel" not in text
    assert 'if VPFP_SHA_LINE="$(sha256sum' in text and 'checkpoint_sha_failed:$?' in text
    assert 'if [[ "${VPFP_EXPERIMENT_EXIT}" == 0 ]]' not in text
    outcome = subprocess.run(["bash", str(wrapper)], env={"PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
    assert outcome.returncode == 2 and "Slurm compute" in outcome.stderr


@pytest.mark.parametrize("binding", ["verified", "missing", "symlink", "mismatch", "read_error"])
def test_launcher_requires_pinned_real_preflight_before_gpu(tmp_path, binding):
    # All paths/bytes in this shell test are synthetic. It never reads the
    # pinned real result, starts Python training, or invokes the scheduler.
    checkout, source, results, bins = (tmp_path / name for name in ("tree", "source", "results", "bin"))
    for path in (checkout / "scripts", checkout / "configs", source / "corpus", results, bins):
        path.mkdir(parents=True)
    for path in (checkout / "configs/vpfp_unified_live_strong_collision.json", source / "corpus/manifest.json",
            source / "result.json", source / "frozen-reference-features.json",
            checkout / "scripts/run_vpfp_strong_initial_pilot.py", checkout / "scripts/run_plasma_canonical_smoke.py"):
        path.write_text("synthetic fixture")
    evidence = tmp_path / "preflight-result.json"
    evidence_bytes = b'{"execution_passed": true, "origin": "synthetic test"}'
    if binding != "missing":
        evidence.write_bytes(evidence_bytes + (b"changed" if binding == "mismatch" else b""))
    if binding == "symlink":
        evidence = tmp_path / "preflight-result-link.json"
        evidence.symlink_to(tmp_path / "preflight-result.json")
    calls = tmp_path / "python-calls.txt"
    fake_python = bins / "python"
    fake_python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$CALLS"\nexit 9\n')
    fake_python.chmod(0o700)
    for name in ("git", "module"):
        path = bins / name
        path.write_text("#!/usr/bin/env bash\nexit 0\n")
        path.chmod(0o700)
    fake_sha = bins / "sha256sum"
    fake_sha.write_text('#!/usr/bin/env bash\nif [[ "$SHA_READ_ERROR" == 1 ]]; then exit 6; fi\nexec /usr/bin/sha256sum "$@"\n')
    fake_sha.chmod(0o700)
    script = (ROOT / "scripts/run_vpfp_strong_initial.slurm").read_text()
    script = script.replace(str(ROOT), str(checkout))
    script = script.replace("${PDE_GAV_RUN_ROOT}/vpfp-strong-main-5244223", str(source))
    script = script.replace("${PDE_GAV_RUN_ROOT}/vpfp-strong-roundtrip-preflight-5259322/result.json", str(evidence))
    script = script.replace("1c37c6dc2509e6fdaf1bd6bd28bec7ad00bef871274f598f6fdca952b86e3c80", sha256(evidence_bytes).hexdigest())
    script = script.replace("${PDE_GAV_RUN_ROOT}", str(results))
    script = script.replace("${PDE_GAV_PYTHON}", str(fake_python))
    wrapper = tmp_path / "synthetic-launch.slurm"
    wrapper.write_text(script)
    run = subprocess.run(["bash", str(wrapper)], capture_output=True, text=True, env={
        "PATH": str(bins) + ":/usr/bin:/bin", "SLURM_JOB_ID": "9001", "SLURM_TMPDIR": str(tmp_path),
        "CALLS": str(calls), "SHA_READ_ERROR": str(int(binding == "read_error"))})
    receipt = (results / "vpfp-strong-initial-9001.launch-exit-codes.tsv").read_text()
    if binding == "verified":
        assert run.returncode == 9 and "preflight_binding\t0\n" in receipt
        assert "gpu_preflight\t9\n" in receipt and "experiment\tnot_run\n" in receipt
        assert len(calls.read_text().splitlines()) == 1
    else:
        assert run.returncode == 2 and "preflight_binding\tfailed\n" in receipt
        assert "gpu_preflight\tnot_run\n" in receipt and "experiment\tnot_run\n" in receipt
        assert not calls.exists()
    assert not (results / "vpfp-strong-initial-9001").exists()


def test_native_and_cov_factories_preserve_common_trunk(tmp_path):
    # Reuse the already tested small prior fixture, never load assets.
    fixture = conditional_fixture()
    args = arguments(tmp_path)
    args.width, args.depth = 8, 1
    cfg = fixture.fixture_config()
    prior = fixture.prior_spec(cfg).to_payload()
    manifest = {"model_spec": prior["model_spec"], "manifest_sha256": prior["corpus_manifest_sha256"], "provenance": prior}
    spec = driver.previous.VPFPOnlineModelSpec.from_mapping(prior["model_spec"])
    original = driver.shared.model_config
    driver.shared.model_config = lambda *a: cfg.full_config
    try:
        receipts = {}
        for arm in driver.ARMS:
            receipt = {}
            config, make, fit = driver.factories(args, spec, manifest, arm, receipt)
            make(config)
            receipts[arm] = receipt
        assert receipts["cov_base"]["common_trunk_sha256"] == receipts["initial_conditioned"]["common_trunk_sha256"]
        assert receipts["initial_conditioned"]["parameter_count"] - receipts["cov_base"]["parameter_count"] == 40
    finally:
        driver.shared.model_config = original


@pytest.mark.parametrize("completed", [True, False])
def test_equal_updates_gate_and_all_model_freeze(tmp_path, monkeypatch, completed):
    args = arguments(tmp_path)
    args.output.mkdir()
    calls = []
    def factories(arguments, spec, manifest, arm, initialization):
        initialization.update(common_trunk_sha256=f"seed-{arguments.seed}", parameter_count=10)
        return None, None, None
    monkeypatch.setattr(driver, "factories", factories)
    def train(arguments, spec, corpus, output, **kwargs):
        calls.append(output.name)
        checkpoint = output / "best.pt"
        checkpoint.write_bytes(str(output).encode())
        return checkpoint, {"checkpoint_sha256": driver.shared.file_sha(checkpoint), "best_step": 1000,
            "final_step": 20000 if completed else 19999, "stop_reason": "step_limit" if completed else "training_wall_time_limit",
            "normalizer_sha256": "same-for-" + output.name}
    monkeypatch.setattr(driver.shared, "train", train)
    if not completed:
        with pytest.raises(RuntimeError, match="did not complete 20000"):
            driver.train_models(args, None, {}, "p"*64, lambda _: None, lambda: None)
        assert len(calls) == 1
        assert (args.output / "training-progress.json").is_file()
        assert not (args.output / "frozen-checkpoints.json").exists()
    else:
        training, models, frozen = driver.train_models(args, None, {}, "p"*64, lambda _: None, lambda: None)
        assert calls == ["cov_base", "initial_conditioned", "initial_conditioned", "cov_base"]
        assert len(training) == len(models) == len(frozen) == 4
        assert json.loads((args.output / "frozen-checkpoints.json").read_text())["all_models_completed_updates"] == 20000


def test_baseline_recipe_rejects_half_or_no_center(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    plan = {"model_spec": {"frozen": True}, "covariance_floor": .05, "covariance_rho_policy": "identity",
        "covariance_center": "training_kinetic_DC_mean", "width": 256, "depth": 4, "batch_size": 4, "learning_rate": 2.e-4}
    (source / "frozen-plan.json").write_text(json.dumps(plan))
    result = driver.check_baseline_recipe(source, {"model_spec": plan["model_spec"]})
    assert result["covariance_strength"] == 1.
    plan["covariance_center"] = "none"
    (source / "frozen-plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="centered full covariance"):
        driver.check_baseline_recipe(source, {"model_spec": plan["model_spec"]})


@pytest.mark.parametrize("passed,regime", [(True, "strong"), (False, "strong"), (True, "weak"), (1, "strong")])
def test_completed_source_identity(tmp_path, passed, regime):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"execution_passed": passed, "regime": regime}))
    if passed is True and regime == "strong":
        receipt = driver.check_completed_source(tmp_path)
        assert receipt["result_sha256"] == driver.shared.file_sha(path)
    else:
        with pytest.raises(ValueError, match="completed strong"):
            driver.check_completed_source(tmp_path)


@pytest.mark.parametrize("reject_gate,arms", [(False, None), (False, ("initial_conditioned",)), (True, None)])
def test_real_tiny_preflight_includes_ema_adamw_validation(tmp_path, monkeypatch, reject_gate, arms):
    from pde_agent_verifier_solver.plugins.vpfp.diffusion_data import CorpusSplit
    fixture = conditional_fixture()
    config, raw, _ = fixture.data(torch.float32)
    args = arguments(tmp_path)
    args.device = "cpu"
    args.output.mkdir()
    prior = fixture.prior_spec(config).to_payload()
    manifest = {"model_spec": prior["model_spec"], "manifest_sha256": prior["corpus_manifest_sha256"], "provenance": prior}
    spec = driver.previous.VPFPOnlineModelSpec.from_mapping(prior["model_spec"])
    split = CorpusSplit(raw.numpy(), np.tile(np.array([[20., 1.]], dtype=np.float32), (8, 1)),
        np.arange(8), np.arange(8).astype(str), np.arange(8), manifest)
    monkeypatch.setattr(driver.previous, "load_split", lambda *a, **k: split)
    monkeypatch.setattr(driver.shared, "model_config", lambda *a: config.full_config)
    rho, kinetic = fixture._value_split(fixture.extract_value_controls(raw, config.value_config), config.value_config)
    initial = torch.cat((rho[:1, 0], kinetic[:1, :, 0].flatten(1)), dim=1)
    monkeypatch.setattr(fixture.prior.KnownInitialPrior, "sample", lambda self, seeds, conditions: {
        "initial_controls": initial.expand(len(seeds), -1), "equation_conditions": conditions})
    validation = driver.shared.validation_report
    calls = []
    def checked_validation(ema, normalizer, split, **kwargs):
        assert not ema.training and all(not parameter.requires_grad for parameter in ema.parameters())
        report = validation(ema, normalizer, split, **kwargs)
        calls.append(report)
        return report
    monkeypatch.setattr(driver.shared, "validation_report", checked_validation)
    if reject_gate:
        monkeypatch.setattr(driver, "audit_normalizer_roundtrip", lambda *a: {
            "passed": False, "status": "rejected", "groups": {"rho": {"value_IEEE_budget_passed": False}}})
        with pytest.raises(ValueError, match="cov_base numerical normalization gate rejected"):
            driver.preflight(args, spec, manifest)
        stored = json.loads((args.output / "preflight-cov_base-normalization.json").read_text())
        assert stored["arm"] == "cov_base" and not stored["groups"]["rho"]["value_IEEE_budget_passed"]
        assert not calls  # Reject before any training/EMA validation step.
        return
    reports = driver.preflight(args, spec, manifest, arms=arms)
    expected_arms = driver.ARMS if arms is None else arms
    assert set(reports) == set(expected_arms)
    assert len(calls) == len(expected_arms) and all(report["counts_by_sigma"] == [4]*4 for report in calls)
    assert all(report["adamw_and_ema_update_passed"] for report in reports.values())
    if arms is None:
        assert reports["cov_base"]["common_trunk_sha256"] == reports["initial_conditioned"]["common_trunk_sha256"]
    assert all(report["peak_cuda_allocated_bytes"] is None for report in reports.values())
    assert all(json.loads((args.output / f"preflight-{arm}-normalization.json").read_text())["passed"] for arm in expected_arms)


@pytest.mark.parametrize("arms", [(), ("unknown",), ("cov_base", "cov_base")])
def test_preflight_rejects_invalid_arm_subset_before_loading(arms):
    with pytest.raises(ValueError, match="nonempty distinct subset"):
        driver.preflight(None, None, None, arms=arms)


def test_generation_prior_seeds_exclude_cumulative_history(tmp_path):
    args = arguments(tmp_path)
    (args.source / "corpus").mkdir(parents=True)
    spec = driver.previous.VPFPOnlineModelSpec("strong_fixture", "/unused", "a"*64, "/unused", "b"*64, (10., 30.))
    manifest = {"model_spec": asdict(spec), "seed_plan": {"path": "seeds.json"}}
    plan = {"splits": {"train": [{"equation_seed": 3}]}, "excluded_source_equation_seeds": [4]}
    path = args.source / "corpus/seeds.json"
    path.write_text(json.dumps(plan))
    designs = driver.generation_design(args, manifest)
    assert len(designs["seed-0"]["known_initial_seeds"]) == 128
    assert set(designs["seed-0"]["known_initial_seeds"]).isdisjoint(designs["seed-1"]["known_initial_seeds"])
    plan["excluded_source_equation_seeds"].append(designs["seed-0"]["known_initial_seeds"][0])
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="overlaps source history"):
        driver.generation_design(args, manifest)


def test_conditional_sampling_uses_full_native_api_not_overwriting_t0(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic")
    entry = {"checkpoint": str(checkpoint), "sha256": driver.shared.file_sha(checkpoint), "arm": "initial_conditioned"}
    sampler = object.__new__(driver.VPFPInitialConditionedBatchSampler)
    sampler.identity = {"latent_dimension": 4}
    seen = []
    def sample(theta, seeds):
        seen.append(seeds[0])
        return torch.ones(1, 4)
    sampler.sample_conditions = sample
    sampler.model = SimpleNamespace(sample=lambda *a, **k: pytest.fail("must not call unconditioned model.sample"))
    monkeypatch.setattr(driver.previous, "load_vpfp_diffusion_sampler", lambda *a, **k: sampler)
    monkeypatch.setattr(driver.shared, "assert_checkpoint_corpus", lambda *a: None)
    monkeypatch.setattr(driver.previous, "load_split", lambda *a, **k: SimpleNamespace(sample_ids=[1]))
    descriptors = ({"name": "fixture", "group": "macro"},)
    def extract(decoder, raw, theta, **kwargs):
        return np.ones((128, 1)), descriptors, {"per_case": [
            {"input_answer_sha256": sha256(str(i).encode()).hexdigest(), "input_equation_condition": condition.tolist()}
            for i, condition in enumerate(theta)]}
    monkeypatch.setattr(driver.previous, "extract_vpfp_features", extract)
    monkeypatch.setattr(driver.previous, "extract_vpfp_derivative_features", extract)
    monkeypatch.setattr(driver.previous, "AuditFeature", lambda **kwargs: kwargs)
    monkeypatch.setattr(driver.previous, "audit_diffusion_distribution", lambda *a, **k: {"status": "insufficient_evidence"})
    spec = driver.previous.VPFPOnlineModelSpec("strong_fixture", "/unused", "a"*64, "/unused", "b"*64, (10., 30.))
    output = tmp_path / "audit"
    output.mkdir()
    design = {"conditions": [[20., 1.]] * 128, "sampling_seeds": list(range(128)), "known_initial_seeds": list(range(1000, 1128))}
    matrices, reports = driver.sample_and_audit(args, spec, {"manifest_sha256": "c"*64}, entry, output, None,
        {"physical": {}, "derivative": {}}, {"physical": descriptors, "derivative": descriptors}, design)
    assert seen == list(range(128))
    assert matrices["physical"].shape == (128, 1)
    assert reports["physical"]["admitted"] is False and reports["physical"]["unobserved_source_pairwise_matching"] is False
    assert np.load(output / "generated-coefficients.npy").shape == (128, 4)


def test_missing_slurm_rejected_before_output(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(SystemExit):
        driver.main(["--config", str(args.config), "--source", str(args.source), "--output", str(args.output)])
    assert not args.output.exists()


def test_main_preserves_preflight_failure_and_does_not_train(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    args.source.mkdir()
    args.config.write_text("{}")
    monkeypatch.setenv("SLURM_JOB_ID", "9001")
    monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path))
    monkeypatch.setattr(driver, "SCRATCH", tmp_path)
    monkeypatch.setattr(driver.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(driver.shared, "train", lambda *a, **k: pytest.fail("must not train malformed source"))
    with pytest.raises(KeyError):
        driver.main(["--config", str(args.config), "--source", str(args.source), "--output", str(args.output)])
    result = json.loads((args.output / "result.json").read_text())
    assert result["execution_passed"] is False and result["failure"]["stage"] == "source_preflight"
    assert result["new_pde_solves"] == 0 and result["admitted"] is False
