"""Tiny allocation-wrapper fixtures; no actual Slurm, assets, or reference data."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    loader = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(loaded)
    return loaded


driver = module("strong_roundtrip_preflight_test", ROOT / "scripts/run_vpfp_strong_roundtrip_preflight.py")


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def arguments(tmp_path):
    args = driver.parser().parse_args(["--output", str(tmp_path / "output")])
    args.source, args.config, args.device = tmp_path / "source", tmp_path / "config.json", "cpu"
    return args


def forbid(*args, **kwargs):
    pytest.fail("preflight must not enter formal training or distribution reference pipeline")


def tiny_runtime(tmp_path, monkeypatch):
    """Real factories, normalizers, gradient, EMA, Heun; only assets are tiny."""
    from pde_agent_verifier_solver.plugins.vpfp.diffusion_data import CorpusSplit
    fixture = module("strong_preflight_tiny_conditioned", ROOT / "tests/test_vpfp_initial_conditioned_diffusion.py")
    cfg, raw, conditional = fixture.data(torch.float32)
    args = arguments(tmp_path)
    prior = fixture.prior_spec(cfg).to_payload()
    manifest = {"model_spec": prior["model_spec"], "manifest_sha256": prior["corpus_manifest_sha256"], "provenance": prior}
    spec = driver.previous.VPFPOnlineModelSpec.from_mapping(prior["model_spec"])
    split = CorpusSplit(raw.numpy(), np.tile(np.array([[20., 1.]], dtype=np.float32), (8, 1)),
        np.arange(8), np.arange(8).astype(str), np.arange(8), manifest)
    roles = []
    def load_split(corpus, role, *, verify_hashes):
        assert role in ("train", "validation") and verify_hashes is False
        roles.append(role)
        return split
    monkeypatch.setattr(driver.previous, "load_split", load_split)
    monkeypatch.setattr(driver.shared, "model_config", lambda *a: cfg.full_config)
    rho, kinetic = fixture._value_split(fixture.extract_value_controls(raw, cfg.value_config), cfg.value_config)
    initial = torch.cat((rho[:1, 0], kinetic[:1, :, 0].flatten(1)), dim=1)
    monkeypatch.setattr(fixture.prior.KnownInitialPrior, "sample", lambda self, seeds, conditions: {
        "initial_controls": initial.expand(len(seeds), -1), "equation_conditions": conditions})
    checkpoint = tmp_path / "synthetic-old-normalizer.pt"
    payload = {"schema": "vpfp_temporal_covariance_diffusion_checkpoint_v1", "config": {}, "model": {},
        "normalizer": conditional.temporal.to_payload(), "corpus_manifest_sha256": manifest["manifest_sha256"],
        "base_decoder_sha256": spec.base_decoder_sha256, "training": {}}
    torch.save(payload, checkpoint)
    summary = {"checkpoint": str(checkpoint), "checkpoint_sha256": driver.shared.file_sha(checkpoint),
        "normalizer_sha256": driver._json_sha(payload["normalizer"])}
    monkeypatch.setattr(driver, "freeze_inputs", lambda _: (spec, manifest, summary,
        {str(checkpoint): summary["checkpoint_sha256"]}))
    monkeypatch.setattr(driver.shared, "train", forbid)
    monkeypatch.setattr(driver.pilot, "main", forbid)
    monkeypatch.setattr(driver.pilot, "reference_inventory", forbid)
    monkeypatch.setattr(driver.pilot, "sample_and_audit", forbid)
    monkeypatch.setattr(driver.previous, "extract_vpfp_features", forbid)
    monkeypatch.setattr(driver.previous, "extract_vpfp_derivative_features", forbid)
    return args, spec, manifest, summary, roles


@pytest.mark.parametrize("failed_arm", [None, "cov_base"])
def test_execute_real_tiny_mechanics_and_independent_failure_records(tmp_path, monkeypatch, failed_arm):
    args, _, _, _, roles = tiny_runtime(tmp_path, monkeypatch)
    actual = driver.pilot.preflight
    seen = []
    def per_arm(*args, arms):
        seen.extend(arms)
        if arms == (failed_arm,):
            raise RuntimeError("synthetic first-arm rejection")
        return actual(*args, arms=arms)
    monkeypatch.setattr(driver.pilot, "preflight", per_arm)
    result = driver.execute(args)
    assert result["execution_passed"] is (failed_arm is None)
    assert seen == list(driver.pilot.ARMS)
    assert len(result["reports"]) == 7 and set(roles) == {"train", "validation"}
    assert result["reports"]["mechanics-initial_conditioned"]["adamw_and_ema_update_passed"]
    for arm in driver.pilot.ARMS:
        for role in ("train", "validation"):
            stored = json.loads((args.output / f"old2048-{arm}-{role}.json").read_text())
            assert stored["execution_passed"] and stored["count"] == 8
            assert stored["normalizer_refitted"] is False
            # The test fixture really has eight fit rows, never claim 2048.
            assert stored["normalizer_fit_rows"] == 8
    if failed_arm:
        assert "synthetic first-arm rejection" in result["reports"]["mechanics-cov_base"]["error"]
        assert not result["reports"]["paired-initialization"]["execution_passed"]
    else:
        assert result["reports"]["paired-initialization"]["execution_passed"]
    assert result["formal_training_updates"] == result["new_pde_solves"] == 0
    assert not result["scientific_validation"] and not result["distribution_audit_performed"]
    assert not list(args.output.rglob("*.pt"))
    outputs = json.loads((args.output / "frozen-outputs.json").read_text())
    assert "frozen-outputs.json" not in outputs["sha256"]
    for path, digest in outputs["sha256"].items():
        assert driver.shared.file_sha(args.output / path) == digest
    plan = json.loads((args.output / "frozen-plan.json").read_text())
    assert plan["mechanics_sample_steps"] == 2 and plan["mechanics_optimizer_updates_per_arm"] == 1
    for script in ("run_vpfp_strong_roundtrip_preflight.py", "run_vpfp_strong_roundtrip_preflight.slurm",
            "run_vpfp_strong_initial_pilot.py", "run_vpfp_temporal_training_pilot.py", "run_vpfp_value_control_pilot.py"):
        assert str(ROOT / "scripts" / script) in plan["frozen_code_sha256"]


def test_source_failure_preserves_plan_and_stops_before_models(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    def reject(_):
        raise ValueError("synthetic pinned input SHA mismatch")
    monkeypatch.setattr(driver, "freeze_inputs", reject)
    monkeypatch.setattr(driver.pilot, "preflight", forbid)
    result = driver.execute(args)
    assert not result["execution_passed"] and result["reports"] == {}
    assert "pinned input SHA" in result["error"]
    assert (args.output / "frozen-plan.json").is_file() and (args.output / "frozen-outputs.json").is_file()


@pytest.mark.parametrize("field", ["normalizer_sha256", "checkpoint_schema", "training_identity"])
def test_old_normalizer_rejects_identity_change(tmp_path, monkeypatch, field):
    args, spec, manifest, summary, _ = tiny_runtime(tmp_path, monkeypatch)
    if field == "normalizer_sha256":
        summary[field] = "0"*64
    elif field == "checkpoint_schema":
        payload = torch.load(summary["checkpoint"], weights_only=True)
        payload["schema"] = "wrong"
        torch.save(payload, summary["checkpoint"])
    else:
        actual = driver.previous.load_split
        def altered(*a, **k):
            from dataclasses import replace
            return replace(actual(*a, **k), equation_seeds=np.arange(8)+100)
        monkeypatch.setattr(driver.previous, "load_split", altered)
    with pytest.raises(ValueError, match="differs|differ"):
        driver.frozen_normalizer(args, spec, manifest, summary)


def source_fixture(tmp_path, monkeypatch):
    """Small JSON/file hashes only; no real manifests or coefficient arrays."""
    args = arguments(tmp_path)
    args.source.mkdir()
    spec = driver.previous.VPFPOnlineModelSpec("strong_fixture", str(tmp_path / "base.pt"), "a"*64,
        str(tmp_path / "expanded.pt"), "b"*64, (10., 30.))
    model_spec = json.loads(json.dumps(asdict(spec)))
    models = {}
    for name in ("base", "expanded"):
        path = tmp_path / (name + ".pt")
        path.write_bytes(b"synthetic frozen decoder")
        models[name] = {"path": str(path), "sha256": driver.shared.file_sha(path)}
    monkeypatch.setattr(type(spec), "preflight", lambda _: {"checkpoints": models})
    args.config.write_text(json.dumps({"vpfp": model_spec}))
    manifest = {"split_counts": driver.pilot.COUNTS, "model_spec": model_spec, "manifest_sha256": "c"*64,
        "splits": {}, "provenance": {"physical_module_sha256": {}}, "seed_plan": {"path": "seeds.json"}}
    for role in ("train", "validation"):
        directory = args.source / "corpus" / role
        directory.mkdir(parents=True)
        path = directory / "tiny-bytes.npy"
        path.write_bytes(b"synthetic identity bytes, not a physical answer")
        manifest["splits"][role] = {"files": {"coefficients": {
            "path": f"{role}/tiny-bytes.npy", "sha256": driver.shared.file_sha(path)}}}
    seed_plan = args.source / "corpus/seeds.json"
    seed_plan.write_text("{}")
    manifest["seed_plan"]["sha256"] = driver.shared.file_sha(seed_plan)
    for name in driver.PHYSICAL_MODULES:
        path = tmp_path / "tree/legacy/vpfp/src/agent_verifier/vpfp" / (name + ".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic physical module\n")
        manifest["provenance"]["physical_module_sha256"][name] = driver.shared.file_sha(path)
    monkeypatch.setattr(driver, "ROOT", tmp_path / "tree")
    checkpoint = args.source / "seed-0/cov_base/checkpoints/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic frozen checkpoint bytes")
    summary = {"checkpoint": str(checkpoint), "checkpoint_sha256": driver.shared.file_sha(checkpoint), "normalizer_sha256": "d"*64}
    (args.source / "result.json").write_text(json.dumps({"execution_passed": True, "regime": "strong",
        "training": {"seed-0/cov_base": summary}}))
    (args.source / "frozen-plan.json").write_text(json.dumps({"model_spec": model_spec, "covariance_floor": .05,
        "covariance_rho_policy": "identity", "covariance_center": "training_kinetic_DC_mean",
        "width": 256, "depth": 4, "batch_size": 4, "learning_rate": 2.e-4}))
    (args.source / "corpus/manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(driver, "PINS", {name: driver.shared.file_sha(args.source / name) for name in driver.PINS})
    monkeypatch.setattr(driver.previous, "read_manifest", lambda path, verify_hashes: manifest if verify_hashes is False else forbid())
    return args, manifest


def test_frozen_inputs_include_actual_script_assets_and_physical_source(tmp_path, monkeypatch):
    args, manifest = source_fixture(tmp_path, monkeypatch)
    _, _, _, hashes = driver.freeze_inputs(args)
    assert len([p for p in hashes if "/legacy/vpfp/" in p]) == 5
    assert not any("/corpus/reference_" in p or "/corpus/calibration/" in p for p in hashes)
    assert all(str(args.source / "corpus" / role / "tiny-bytes.npy") in hashes for role in ("train", "validation"))
    path = driver.ROOT / "legacy/vpfp/src/agent_verifier/vpfp/high_fidelity.py"
    path.write_text("# synthetic mutation\n")
    with pytest.raises(ValueError, match="frozen artifact changed"):
        driver.freeze_inputs(args)


@pytest.mark.parametrize("fault", ["no_allocation", "wrong_source", "existing_output", "no_cuda", "dirty_source"])
def test_main_guards_before_execution(tmp_path, monkeypatch, fault):
    output = tmp_path / "vpfp-strong-roundtrip-preflight-9001"
    monkeypatch.setattr(driver, "SOURCE", tmp_path / "source")
    monkeypatch.setattr(driver, "CONFIG", tmp_path / "config.json")
    monkeypatch.setattr(driver, "SCRATCH", tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "9001")
    monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path))
    monkeypatch.setattr(driver.torch.cuda, "is_available", lambda: fault != "no_cuda")
    monkeypatch.setattr(driver.subprocess, "check_output", lambda *a, **k: " M src/changed.py" if fault == "dirty_source" else "")
    monkeypatch.setattr(driver, "execute", forbid)
    source = driver.SOURCE
    if fault == "no_allocation":
        monkeypatch.delenv("SLURM_JOB_ID")
    elif fault == "wrong_source":
        source = tmp_path / "other"
    elif fault == "existing_output":
        output.mkdir()
    with pytest.raises(SystemExit) as error:
        driver.main(["--source", str(source), "--config", str(driver.CONFIG), "--output", str(output)])
    assert error.value.code == 2
    assert not (output / "frozen-plan.json").exists()


def test_slurm_resources_correct_tree_and_missing_allocation():
    path = ROOT / "scripts/run_vpfp_strong_roundtrip_preflight.slurm"
    source = path.read_text()
    subprocess.run(["bash", "-n", str(path)], check=True)
    for text in ("--time=00:20:00", "--gres=gpu:l40s:1", "--cpus-per-task=8", "--mem=48G",
            "--partition=gpubase_l40s_b3", "${PDE_GAV_WORKTREES_ROOT}/vpfp-strong-roundtrip-gate"):
        assert text in source
    assert "vpfp-strong-initial-pilot" not in source and "sbatch " not in source and "scancel" not in source
    assert "run_plasma_canonical_smoke" not in source and "--steps" not in source
    run = subprocess.run(["bash", str(path)], env={"PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
    assert run.returncode == 2 and "Slurm compute" in run.stderr


@pytest.mark.parametrize("gpu_status,preflight_status", [(0, 0), (0, 7), (9, 0)])
def test_mock_wrapper_preserves_failure_receipts_no_followup(tmp_path, gpu_status, preflight_status):
    checkout, old, outputs, bin_dir = (tmp_path / name for name in ("tree", "old", "outputs", "bin"))
    for directory in (checkout / "scripts", checkout / "configs", old / "corpus", outputs, bin_dir):
        directory.mkdir(parents=True)
    for path in (checkout / "configs/vpfp_unified_live_strong_collision.json", old / "corpus/manifest.json",
            old / "result.json", old / "frozen-plan.json", checkout / "scripts/run_vpfp_strong_roundtrip_preflight.py",
            checkout / "scripts/run_vpfp_strong_initial_pilot.py"):
        path.write_text("synthetic fixture")
    python = bin_dir / "python"
    python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$CALLS"\nif [[ "$1" == "-c" ]]; then exit "$GPU_STATUS"; fi\nexit "$PREFLIGHT_STATUS"\n')
    python.chmod(0o700)
    for command in ("git", "module"):
        path = bin_dir / command
        path.write_text("#!/usr/bin/env bash\nexit 0\n")
        path.chmod(0o700)
    source = (ROOT / "scripts/run_vpfp_strong_roundtrip_preflight.slurm").read_text()
    source = source.replace(str(ROOT), str(checkout)).replace(str(driver.SOURCE), str(old))
    source = source.replace(str(driver.SCRATCH), str(outputs)).replace("${PDE_GAV_PYTHON}", str(python))
    script = tmp_path / "synthetic-preflight.slurm"
    script.write_text(source)
    calls = tmp_path / "calls.txt"
    env = {"PATH": str(bin_dir) + ":/usr/bin:/bin", "SLURM_JOB_ID": "9001", "SLURM_TMPDIR": str(tmp_path),
        "GPU_STATUS": str(gpu_status), "PREFLIGHT_STATUS": str(preflight_status), "CALLS": str(calls)}
    run = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    expected = gpu_status or preflight_status
    assert run.returncode == expected
    receipt = (outputs / "vpfp-strong-roundtrip-preflight-9001.launch-exit-codes.tsv").read_text()
    assert f"gpu_preflight\t{gpu_status}\n" in receipt and f"launch\t{expected}\n" in receipt
    assert f"preflight\t{'not_run' if gpu_status else preflight_status}\n" in receipt
    assert len(calls.read_text().splitlines()) == (1 if gpu_status else 2)
