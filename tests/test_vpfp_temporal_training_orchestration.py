"""Mock-only temporal pilot lifecycle tests; no training or real data reads."""

from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp.canonical_binding import VPFPOnlineModelSpec
from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import VPFPStructuredEDMConfig


ROOT = Path(__file__).resolve().parents[1]
loader = importlib.util.spec_from_file_location("temporal_orchestration_under_test", ROOT / "scripts/run_vpfp_temporal_training_pilot.py")
cli = importlib.util.module_from_spec(loader)
loader.loader.exec_module(cli)


@pytest.fixture(autouse=True)
def restore_runtime_settings(monkeypatch):
    before = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
              torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
              torch.backends.cudnn.benchmark)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    yield
    torch.set_num_threads(before[0])
    torch.use_deterministic_algorithms(before[1])
    torch.backends.cuda.matmul.allow_tf32 = before[2]
    torch.backends.cudnn.allow_tf32 = before[3]
    torch.backends.cudnn.benchmark = before[4]


@pytest.fixture
def orchestration(tmp_path, monkeypatch):
    cfg = VPFPStructuredEDMConfig(control_count=3, hermite_channels=2, rho_max_mode=1,
        kinetic_max_mode=1, inactive_dim=3, grid_size=3, width=8, depth=1)
    spec = VPFPOnlineModelSpec("weak_fixture", "base.pt", "a"*64, "expanded.pt", "b"*64,
        cfg.collision_nu_range, cfg.temperature_range, cfg.time_domain)
    manifest = {"model_spec": asdict(spec), "manifest_sha256": "c"*64}
    output, cache, diagnosis = tmp_path/"output", tmp_path/"cache", tmp_path/"diagnosis"
    config_file = tmp_path/"config.json"
    config_file.write_text(json.dumps(asdict(spec)))
    cache.mkdir()
    cache_file = cache/"immutable-feature-snapshot.bin"
    cache_file.write_bytes(b"mock feature snapshot; not physical data")
    diagnosis.mkdir()
    (diagnosis/"result.json").write_text(json.dumps({"execution_passed": True,
        "regime": "weak", "training_performed": False, "new_pde_solves": 0, "admitted": False}))
    (diagnosis/"frozen-plan.json").write_text(json.dumps({"regime": "weak",
        "corpus_manifest_sha256": manifest["manifest_sha256"]}))
    fixture = SimpleNamespace(output=output, cache=cache, cache_file=cache_file,
        diagnosis=diagnosis, manifest=manifest, events=[], train_calls=[], sample_calls=[],
        constraints={"passed": True, "source_manifest_sha256": manifest["manifest_sha256"]},
        train_failure=None, sample_failure=None, after_train=None, preflight_failure=False,
        initial_identity_change=None, normalizer_identity_change=None)
    monkeypatch.setenv("SLURM_JOB_ID", "unit-test-not-compute")
    monkeypatch.setattr(VPFPOnlineModelSpec, "preflight", lambda self: {})
    monkeypatch.setattr(cli.previous, "read_manifest", lambda *a, **k: deepcopy(manifest))
    monkeypatch.setattr(cli.shared, "model_config", lambda *a: cfg)
    monkeypatch.setattr(cli.subprocess, "check_output", lambda *a, **k: "frozen-fixture-revision")
    monkeypatch.setattr(cli.shared, "source_fingerprints", lambda: {"fake_source.py": "f"*64})
    monkeypatch.setattr(cli.previous, "cache_inventory", lambda *a: {str(cache_file): cli.shared.file_sha(cache_file)})
    monkeypatch.setattr(cli.previous, "fixed_reference_rows", lambda *a: {"reference_a": [0, 1], "reference_b": [0, 1]})

    def constraint(*a, **k):
        fixture.events.append("constraint")
        return deepcopy(fixture.constraints)

    def require(report):
        if not report["passed"]:
            raise ValueError("off-subspace corpus")

    def preflight(*a, **k):
        fixture.events.append("preflight")
        if fixture.preflight_failure:
            raise ValueError("synthetic preflight failure")
        return {"synthetic_only": True}

    def train(args, spec, corpus, directory, **kwargs):
        key = str(directory.relative_to(output))
        fixture.events.append("train:"+key)
        fixture.train_calls.append({"key": key, "seed": args.seed,
            "ablation": deepcopy(args.ablation_metadata), "factories": kwargs})
        (directory/"partial-training-receipt.json").write_text(json.dumps({"key": key}))
        if key == fixture.train_failure:
            raise RuntimeError("synthetic training failure")
        checkpoints = directory/"checkpoints"
        checkpoints.mkdir()
        checkpoint = checkpoints/"best.pt"
        checkpoint.write_bytes(("mock:"+key).encode())
        result = {"checkpoint_sha256": cli.shared.file_sha(checkpoint),
            "initial_model_sha256": "same-network-seed-"+str(args.seed),
            "normalizer_sha256": directory.name.split("_")[0]+"-normalizer",
            "common_validation_metric": None}
        if key == fixture.initial_identity_change:
            result["initial_model_sha256"] = "unmatched-model"
        if key == fixture.normalizer_identity_change:
            result["normalizer_sha256"] = "unmatched-normalizer"
        if fixture.after_train:
            fixture.after_train(fixture, key)
        return checkpoint, result

    def load_features(*args):
        assert len(fixture.train_calls) == 8
        frozen = json.loads((output/"frozen-checkpoints.json").read_text())
        assert len(frozen["sha256"]) == 8
        assert frozen["all_arms_selected_before_any_feature_evaluation"] is True
        assert args[3]["source_manifest_sha256"] == manifest["manifest_sha256"]
        fixture.events.append("open_features")
        matrix = np.ones((2, 1))
        return {kind: {role: SimpleNamespace(features=matrix) for role in ("reference_a", "reference_b")}
                for kind in ("physical", "derivative")}, {"physical": (), "derivative": ()}

    def sample(args, spec, checkpoint, expected_sha, directory, *rest):
        key = str(directory.relative_to(output))
        assert "open_features" in fixture.events
        assert cli.shared.file_sha(checkpoint) == expected_sha
        fixture.events.append("sample:"+key)
        fixture.sample_calls.append((key, args.seed))
        (directory/"partial-sample-receipt.json").write_text(json.dumps({"key": key}))
        if key == fixture.sample_failure:
            raise RuntimeError("synthetic sample failure")
        return {kind: np.ones((2, 1)) for kind in ("physical", "derivative")}, {
            kind: {"status": "mock-only", "comparisons": {"generated": {"statistics": {
                "sliced_wasserstein_max": {"effect": 1.}}}}} for kind in ("physical", "derivative")}

    monkeypatch.setattr(cli.previous, "audit_value_control_corpus", constraint)
    monkeypatch.setattr(cli.previous, "require_value_control_audit", require)
    monkeypatch.setattr(cli, "preflight", preflight)
    monkeypatch.setattr(cli.shared, "train", train)
    monkeypatch.setattr(cli.previous, "load_cached_features", load_features)
    monkeypatch.setattr(cli.previous, "load_vpfp_diffusion_sampler", lambda *a, **k: None)
    runtime = SimpleNamespace(base_model=None)
    monkeypatch.setattr(cli.previous, "build_vpfp_online_binding", lambda *a, **k:
        SimpleNamespace(plugin=SimpleNamespace(generator=SimpleNamespace(backend=SimpleNamespace(runtime=runtime)))))
    monkeypatch.setattr(cli.previous, "sample_and_audit", sample)
    monkeypatch.setattr(cli.previous, "build_vpfp_direction_scorecard", lambda **k: {
        "seed": k["seed"], "objective": k["objective"], "synthetic_only": True})
    fixture.argv = ["--config", str(config_file), "--cache", str(cache), "--diagnosis", str(diagnosis),
        "--output", str(output), "--regime", "weak", "--seed", "8", "--device", "cpu"]
    return fixture


def result(fixture):
    return json.loads((fixture.output/"result.json").read_text())


def test_all_eight_freeze_before_features_with_reversed_second_replica(orchestration):
    fixture = orchestration
    assert cli.main(fixture.argv) == 0
    expected = [f"train:seed-0/{arm}" for arm in cli.ARMS] + [f"train:seed-1/{arm}" for arm in reversed(cli.ARMS)]
    assert fixture.events[:11] == ["constraint", "preflight", *expected, "open_features"]
    assert [entry[1] for entry in fixture.sample_calls] == [8]*4+[10008]*4
    assert [entry["seed"] for entry in fixture.train_calls] == [8]*4+[10008]*4
    plan_hashes = {entry["ablation"]["plan_sha256"] for entry in fixture.train_calls}
    assert plan_hashes == {cli.shared.file_sha(fixture.output/"frozen-plan.json")}
    assert all(call["factories"]["training_metric_factory"] is None and
               call["factories"]["validation_metric_factory"] is None for call in fixture.train_calls)
    report = result(fixture)
    assert report["execution_passed"] is True
    assert report["admitted"] is False
    assert report["large_experiment_launched"] is False
    assert report["canonical_campaign_smoke"] == "separate_required_step"
    assert len(report["audits"]) == 8


def test_off_subspace_corpus_saves_failed_gate_before_training(orchestration):
    fixture = orchestration
    fixture.constraints["passed"] = False
    assert cli.main(fixture.argv) == 1
    assert fixture.events == ["constraint"]
    assert (fixture.output/"value-slope-corpus-audit.json").is_file()
    assert result(fixture)["failed_stage"] == "source_constraint_check"
    assert result(fixture)["partial_artifacts_preserved"] is True


def test_source_manifest_drift_after_plan_freeze_is_rejected(orchestration):
    fixture = orchestration
    fixture.constraints["source_manifest_sha256"] = "d"*64
    assert cli.main(fixture.argv) == 1
    assert fixture.events == ["constraint"]
    assert "corpus changed" in result(fixture)["error"]


def test_preflight_failure_preserves_constraint_evidence(orchestration):
    fixture = orchestration
    fixture.preflight_failure = True
    assert cli.main(fixture.argv) == 1
    assert fixture.events == ["constraint", "preflight"]
    assert result(fixture)["failed_stage"] == "full_layout_preflight"
    assert (fixture.output/"value-slope-corpus-audit.json").is_file()


def test_training_failure_preserves_earlier_checkpoints_and_partial_stage(orchestration):
    fixture = orchestration
    fixture.train_failure = "seed-0/cov_base"
    assert cli.main(fixture.argv) == 1
    assert result(fixture)["failed_stage"] == "train_seed-0/cov_base"
    assert result(fixture)["partial_artifacts_preserved"] is True
    assert (fixture.output/"seed-0/value_base/checkpoints/best.pt").is_file()
    assert (fixture.output/"seed-0/value_low/checkpoints/best.pt").is_file()
    assert (fixture.output/"seed-0/cov_base/partial-training-receipt.json").is_file()
    assert "open_features" not in fixture.events
    assert not (fixture.output/"frozen-checkpoints.json").exists()


@pytest.mark.parametrize("field,message", [("initial_identity_change", "parameter identities"),
                                           ("normalizer_identity_change", "different normalization")])
def test_pair_identity_mismatch_prevents_feature_opening(orchestration, field, message):
    fixture = orchestration
    setattr(fixture, field, "seed-0/cov_low")
    assert cli.main(fixture.argv) == 1
    assert len(fixture.train_calls) == 4
    assert message in result(fixture)["error"]
    assert "open_features" not in fixture.events


def test_changed_checkpoint_is_rejected_before_features(orchestration):
    fixture = orchestration
    def corrupt(item, key):
        if key == "seed-1/value_base":
            (item.output/"seed-0/value_base/checkpoints/best.pt").write_bytes(b"unexpected mutation")
    fixture.after_train = corrupt
    assert cli.main(fixture.argv) == 1
    assert len(fixture.train_calls) == 8
    assert "frozen artifact changed" in result(fixture)["error"]
    assert "open_features" not in fixture.events


def test_changed_feature_cache_is_rejected_after_freeze(orchestration):
    fixture = orchestration
    fixture.after_train = lambda item, key: item.cache_file.write_bytes(b"unexpected mutation")
    assert cli.main(fixture.argv) == 1
    assert len(fixture.train_calls) == 8
    assert (fixture.output/"frozen-checkpoints.json").is_file()
    assert "frozen artifact changed" in result(fixture)["error"]
    assert "open_features" not in fixture.events


def test_audit_failure_preserves_all_frozen_models_and_partial_samples(orchestration):
    fixture = orchestration
    fixture.sample_failure = "seed-0/cov_base"
    assert cli.main(fixture.argv) == 1
    assert len(fixture.train_calls) == 8
    assert result(fixture)["failed_stage"] == "frozen_development_audit_seed-0/cov_base"
    assert len(json.loads((fixture.output/"frozen-checkpoints.json").read_text())["sha256"]) == 8
    assert (fixture.output/"seed-0/value_base/partial-sample-receipt.json").is_file()
    assert (fixture.output/"seed-0/cov_base/partial-sample-receipt.json").is_file()


@pytest.mark.parametrize("target,updates", [("result.json", {"execution_passed": False}),
    ("result.json", {"regime": "strong"}), ("frozen-plan.json", {"corpus_manifest_sha256": "d"*64})])
def test_diagnosis_must_be_complete_and_bound_to_same_corpus(orchestration, target, updates):
    fixture = orchestration
    path = fixture.diagnosis/target
    payload = json.loads(path.read_text())
    payload.update(updates)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="completed mechanism diagnostic"):
        cli.main(fixture.argv)
    assert not fixture.output.exists()
    assert not fixture.events


@pytest.mark.parametrize("target", ("result.json", "frozen-plan.json"))
def test_diagnosis_mutation_during_training_cannot_finish_successfully(orchestration, target):
    fixture = orchestration
    def corrupt(item, key):
        if key == "seed-0/value_base":
            path = item.diagnosis/target
            path.write_text(path.read_text()+" ")
    fixture.after_train = corrupt
    assert cli.main(fixture.argv) == 1
    report = result(fixture)
    assert report["execution_passed"] is False
    assert report["admitted"] is False
    assert "frozen artifact changed" in report["error"]
    assert target in report["error"]
    assert (fixture.output/"seed-0/value_base/checkpoints/best.pt").is_file()


def test_final_republished_corpus_manifest_cannot_change_identity(orchestration, monkeypatch):
    fixture = orchestration
    count = 0
    def republished(*args, **kwargs):
        nonlocal count
        count += 1
        manifest = deepcopy(fixture.manifest)
        if count > 1:
            manifest["manifest_sha256"] = "d"*64
        return manifest
    monkeypatch.setattr(cli.previous, "read_manifest", republished)
    assert cli.main(fixture.argv) == 1
    report = result(fixture)
    assert report["execution_passed"] is False
    assert "corpus" in report["error"]
    assert report["admitted"] is False
    assert (fixture.output/"frozen-checkpoints.json").is_file()
