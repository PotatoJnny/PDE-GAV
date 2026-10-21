"""Mock first-stage main training; no real collection, training or GPU work."""

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
loader = importlib.util.spec_from_file_location("vpfp_strong_main_under_test", ROOT / "scripts/run_vpfp_strong_main_experiment.py")
cli = importlib.util.module_from_spec(loader)
loader.loader.exec_module(cli)


@pytest.fixture(autouse=True)
def restore_runtime_settings(monkeypatch):
    old = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
           torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    yield
    torch.set_num_threads(old[0])
    torch.use_deterministic_algorithms(old[1])
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark = old[2:]


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    cfg = VPFPStructuredEDMConfig(control_count=4, hermite_channels=2, rho_max_mode=1,
        kinetic_max_mode=1, inactive_dim=3, grid_size=3, width=8, depth=1)
    spec = VPFPOnlineModelSpec("strong_mock", "base.pt", "a"*64, "expanded.pt", "b"*64,
        cfg.collision_nu_range, cfg.temperature_range, cfg.time_domain)
    source_root, output, config_file = tmp_path / "source-corpus", tmp_path / "output", tmp_path / "config.json"
    source_root.mkdir()
    source = {"manifest_sha256": "c"*64, "model_spec": json.loads(json.dumps(asdict(spec))),
        "base_decoder_sha256": "a"*64, "seed_plan": {"path": "seed_plan.json"}}
    fresh = {**deepcopy(source), "manifest_sha256": "d"*64, "split_counts": dict(cli.COUNTS),
        "source_corpus": {"manifest_sha256": source["manifest_sha256"], "reused_splits": [],
            "regenerated_splits": list(cli.previous.ROLES), "excluded_equation_seed_count": 1792}}
    cli.shared.write_json(config_file, asdict(spec))
    cli.shared.write_json(source_root / "manifest.json", source)
    cli.shared.write_json(source_root / "seed_plan.json", {"synthetic_only": True})
    item = SimpleNamespace(source_root=source_root, output=output, config_file=config_file,
        source=source, fresh=fresh, events=[], train_calls=[], audit_calls=[], scorecard_calls=[],
        preflight_failure=False, collection_failure=False, train_failure=None, audit_failure=None,
        init_mismatch=None, normalizer_mismatch=None, after_train=None, after_audit=None,
        missing_snapshot=False, constraint_passed=True)
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-unit-test-not-compute")
    monkeypatch.setattr(VPFPOnlineModelSpec, "preflight", lambda self: {})
    monkeypatch.setattr(cli.shared, "model_config", lambda *a: cfg)
    monkeypatch.setattr(cli.shared, "source_fingerprints", lambda: {})
    monkeypatch.setattr(cli.subprocess, "check_output", lambda *a, **k: "mock-revision")
    monkeypatch.setattr(cli.previous, "read_manifest", lambda root, **kwargs:
        deepcopy(fresh if Path(root) == output / "corpus" else source))
    monkeypatch.setattr(cli.previous, "load_split", lambda *a, **k: SimpleNamespace(sample_ids=np.arange(2048)))

    def preflight(arguments, spec, corpus):
        item.events.append("preflight")
        assert corpus == source_root and not (output / "corpus").exists()
        if item.preflight_failure:
            raise RuntimeError("synthetic preflight failure")
        return {"synthetic_only": True}

    def collect(source, target, counts, seed, **kwargs):
        item.events.append("collect")
        assert source == source_root and target == output / "corpus"
        assert counts == cli.COUNTS and sum(counts.values()) == 2560 and seed == 5000019
        assert kwargs == {"batch_size": 8, "device": "cpu"}
        plan = cli.scaleup.read_json(output / "frozen-plan.json")
        assert plan["reused_answer_splits"] == [] and plan["previous_audits_read_for_model_selection"] is False
        assert plan["split_counts"]["train"] == 2048
        target.mkdir()
        cli.shared.write_json(target / "partial-collection.json", {"synthetic_only": True})
        if item.collection_failure:
            raise RuntimeError("synthetic collection failure")
        cli.shared.write_json(target / "manifest.json", fresh)

    def train(arguments, spec, corpus, directory, **kwargs):
        key = str(directory.relative_to(output))
        item.events.append("train:" + key)
        assert corpus == output / "corpus" and (output / "new-corpus-receipt.json").exists()
        assert arguments.steps == 1000000 and arguments.train_seconds == 7200
        assert arguments.checkpoint_seconds == (1800., 3600.)
        assert kwargs["training_metric_factory"] is None and kwargs["validation_metric_factory"] is None
        item.train_calls.append({"key": key, "seed": arguments.seed, "factories": kwargs,
            "ablation": deepcopy(arguments.ablation_metadata)})
        cli.shared.write_json(directory / "partial-training.json", {"synthetic_only": True})
        if item.train_failure == key:
            raise RuntimeError("synthetic training failure")
        checkpoint_root = directory / "checkpoints"
        checkpoint_root.mkdir()
        best = checkpoint_root / "best.pt"
        best.write_bytes(("mock best " + key).encode())
        receipts = []
        count = len(arguments.checkpoint_seconds) - int(item.missing_snapshot)
        for index, seconds in enumerate(arguments.checkpoint_seconds[:count]):
            path = checkpoint_root / f"snapshot-{index}.pt"
            path.write_bytes(("mock snapshot " + key + str(index)).encode())
            receipts.append({"requested_seconds": seconds, "actual_elapsed_seconds": seconds + 1.,
                "step": index + 1, "checkpoint": str(path), "sha256": cli.shared.file_sha(path), "validation_score": 1.})
        report = {"checkpoint_sha256": cli.shared.file_sha(best), "best_step": 3, "validation_selection_score": .5,
            "initial_model_sha256": "same-within-seed-" + str(arguments.seed), "normalizer_sha256": directory.name,
            "checkpoint_snapshots": receipts, "unreached_checkpoint_seconds": list(arguments.checkpoint_seconds[count:])}
        if key == item.init_mismatch:
            report["initial_model_sha256"] = "different"
        if key == item.normalizer_mismatch:
            report["normalizer_sha256"] = "different"
        if item.after_train:
            item.after_train(item, key)
        return best, report

    def constraint(corpus, **kwargs):
        item.events.append("constraint")
        assert corpus == output / "corpus" and len(item.train_calls) == 0
        assert not (output / "frozen-checkpoints.json").exists()
        return {"passed": item.constraint_passed, "source_manifest_sha256": fresh["manifest_sha256"]}

    def require(report):
        if not report["passed"]:
            raise ValueError("synthetic encoding constraint failure")

    def features(*args):
        item.events.append("features")
        frozen = cli.scaleup.read_json(output / "frozen-checkpoints.json")
        assert frozen["all_arms_and_snapshots_frozen_before_any_feature_evaluation"] is True
        assert len(frozen["models"]) == (8 if item.missing_snapshot else 12)
        assert all(cli.shared.file_sha(path) == digest for path, digest in frozen["sha256"].items())
        matrix = np.ones((128, 1))
        return {kind: {role: SimpleNamespace(features=matrix) for role in cli.previous.ROLES}
            for kind in ("physical", "derivative")}, {"physical": (), "derivative": ()}, {}

    def audit(arguments, spec, entry, directory, decoder, samples, schemas):
        key = str(directory.relative_to(output))
        item.events.append("audit:" + key)
        item.audit_calls.append((key, arguments.seed))
        assert "features" in item.events and arguments.cache == output
        assert arguments.generated_answers == 128 and arguments.sample_steps == 64
        assert cli.shared.file_sha(entry["checkpoint"]) == entry["sha256"]
        cli.shared.write_json(directory / "partial-audit.json", {"synthetic_only": True})
        if key == item.audit_failure:
            raise RuntimeError("synthetic audit failure")
        if item.after_audit:
            item.after_audit(item, key)
        return {kind: np.ones((128, 1)) for kind in ("physical", "derivative")}, {kind: {"status": "mock-only", "comparisons": {"generated": {
            "statistics": {"sliced_wasserstein_max": {"effect": 1.}}}}} for kind in ("physical", "derivative")}

    def scorecard(**kwargs):
        assert set(kwargs["samples"]) == {"reference_a", "reference_b", "value_base", "cov_base"}
        assert set(kwargs["joint_reports"]) == {"value_base", "cov_base"}
        item.scorecard_calls.append(kwargs["seed"])
        return {"synthetic_only": True}

    monkeypatch.setattr(cli.temporal, "preflight", preflight)
    monkeypatch.setattr(cli, "collect_independent_corpus", collect)
    monkeypatch.setattr(cli.shared, "train", train)
    monkeypatch.setattr(cli.previous, "audit_value_control_corpus", constraint)
    monkeypatch.setattr(cli.previous, "require_value_control_audit", require)
    monkeypatch.setattr(cli.scaleup, "write_reference_features", features)
    monkeypatch.setattr(cli.scaleup, "audit_frozen_model", audit)
    monkeypatch.setattr(cli.previous, "build_vpfp_direction_scorecard", scorecard)
    sampler = SimpleNamespace(identity={"corpus_manifest_sha256": fresh["manifest_sha256"], "base_decoder_sha256": "a"*64},
        normalizer=SimpleNamespace(fit_metadata={"sample_ids": list(range(2048))}))
    monkeypatch.setattr(cli.previous, "load_vpfp_diffusion_sampler", lambda *a, **k: sampler)
    runtime = SimpleNamespace(base_model=None)
    monkeypatch.setattr(cli.previous, "build_vpfp_online_binding", lambda *a, **k:
        SimpleNamespace(plugin=SimpleNamespace(generator=SimpleNamespace(backend=SimpleNamespace(runtime=runtime)))))
    item.argv = ["--config", str(config_file), "--source-corpus", str(source_root), "--output", str(output),
        "--seed", "19", "--device", "cpu"]
    return item


def report(fixture):
    return cli.scaleup.read_json(fixture.output / "result.json")


def test_strong_main_defaults_and_existing_helpers_are_used(fixture):
    arguments = cli.parser().parse_args(fixture.argv)
    assert (arguments.train_answers, arguments.heldout_answers, arguments.replicas) == (2048, 128, 2)
    assert arguments.steps == 1000000 and arguments.train_seconds == 7200
    assert arguments.checkpoint_seconds == (1800., 3600.)
    assert arguments.regime == "strong" and cli.ARMS == ("value_base", "cov_base")
    assert arguments.generated_answers == 128 and arguments.sample_steps == 64


def test_full_new_corpus_then_paired_training_then_frozen_reference_audits(fixture):
    assert cli.main(fixture.argv) == 0
    assert fixture.events[:8] == ["preflight", "collect", "constraint", "train:seed-0/value_base", "train:seed-0/cov_base",
        "train:seed-1/cov_base", "train:seed-1/value_base", "features"]
    assert [call["seed"] for call in fixture.train_calls] == [19, 19, 10019, 10019]
    assert len(fixture.audit_calls) == 12 and [value[1] for value in fixture.audit_calls] == [19]*6 + [10019]*6
    result = report(fixture)
    assert result["execution_passed"] is True and result["new_pde_solves"] == 2560
    assert result["generated_answers"] == 1536 and result["all_learning_curve_thresholds_reached"] is True
    assert result["admitted"] is False and result["automatic_followup_job_submitted"] is False
    plan = cli.scaleup.read_json(fixture.output / "frozen-plan.json")
    assert plan["nominal_total_training_seconds"] == 28800 and plan["reused_answer_splits"] == []
    assert plan["previous_audits_read_for_model_selection"] is False
    assert plan["normalization_fit"] == "each_arm_only_new_2048_training_answers"
    assert set(plan["source_file_sha256"]) >= {"scripts/run_vpfp_scaleup_experiment.py",
        "scripts/run_vpfp_temporal_training_pilot.py", "scripts/run_vpfp_strong_main_experiment.py"}
    assert cli.scaleup.read_json(fixture.output / "training-progress.json")["status"] == "all_models_frozen"
    assert len(cli.scaleup.read_json(fixture.output / "audit-progress.json")["completed_audits"]) == 12
    assert fixture.scorecard_calls == [19, 10019]
    card = cli.scaleup.read_json(fixture.output / "seed-0/physical-effect-scorecard.json")
    assert card["used_for_checkpoint_selection"] is False and card["heuristic_boolean_used_as_gate"] is False


@pytest.mark.parametrize("option,value", [("--train-answers", "512"), ("--heldout-answers", "64"),
    ("--replicas", "1"), ("--generated-answers", "64"), ("--collect-batch-size", "16"),
    ("--collect-batch-size", "3"), ("--batch-size", "9"), ("--checkpoint-seconds", "nan"),
    ("--checkpoint-seconds", "7201"), ("--covariance-floor", "0"), ("--regime", "weak")])
def test_invalid_protocol_rejected_before_output(fixture, option, value):
    with pytest.raises(SystemExit, match="2"):
        cli.main(fixture.argv + [option, value])
    assert not fixture.output.exists() and not fixture.events


def test_real_collection_requires_slurm(fixture, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(SystemExit, match="2"):
        cli.main(fixture.argv)
    assert not fixture.output.exists()


@pytest.mark.parametrize("field,stage", [("preflight_failure", "source_training_rows_preflight"),
    ("collection_failure", "collect_new_independent_five_splits")])
def test_preflight_and_collection_failure_preserve_reason_and_never_train(fixture, field, stage):
    setattr(fixture, field, True)
    assert cli.main(fixture.argv) == 1
    result = report(fixture)
    assert result["failed_stage"] == stage and result["partial_artifacts_preserved"] is True
    assert not fixture.train_calls and result["completed_training_receipt"] is None
    assert (fixture.output / "frozen-plan.json").exists()
    if field == "preflight_failure":
        assert "collect" not in fixture.events


def test_training_failure_preserves_prior_native_snapshots_and_training_receipt(fixture):
    fixture.train_failure = "seed-0/cov_base"
    assert cli.main(fixture.argv) == 1
    result = report(fixture)
    assert result["failed_stage"] == "train_seed-0/cov_base"
    assert "synthetic training failure" in result["error"]
    progress = cli.scaleup.read_json(result["completed_training_receipt"])
    assert list(progress["training"]) == ["seed-0/value_base"]
    assert (fixture.output / "seed-0/value_base/checkpoints/snapshot-0.pt").exists()
    assert "features" not in fixture.events


def test_invalid_new_target_encoding_fails_before_long_training(fixture):
    fixture.constraint_passed = False
    assert cli.main(fixture.argv) == 1
    assert report(fixture)["failed_stage"] == "new_corpus_encoding_check"
    assert fixture.events == ["preflight", "collect", "constraint"] and not fixture.train_calls
    assert (fixture.output / "value-slope-corpus-audit.json").exists()


@pytest.mark.parametrize("field,key,phrase", [("init_mismatch", "seed-0/cov_base", "initial network"),
    ("normalizer_mismatch", "seed-1/cov_base", "normalizers differ")])
def test_paired_initialization_and_reproducible_train_only_normalizer_guard(fixture, field, key, phrase):
    setattr(fixture, field, key)
    assert cli.main(fixture.argv) == 1
    assert phrase in report(fixture)["error"] and "features" not in fixture.events


def test_tampered_earlier_snapshot_cannot_reach_reference_features(fixture):
    def mutate(item, key):
        if key == "seed-1/value_base":
            (item.output / "seed-0/value_base/checkpoints/snapshot-0.pt").write_bytes(b"changed")
    fixture.after_train = mutate
    assert cli.main(fixture.argv) == 1
    assert "frozen artifact changed" in report(fixture)["error"] and "features" not in fixture.events


def test_changed_source_metadata_stops_before_more_training(fixture):
    fixture.after_train = lambda item, key: (item.source_root / "seed_plan.json").write_text("changed")
    assert cli.main(fixture.argv) == 1
    assert "frozen artifact changed" in report(fixture)["error"] and len(fixture.train_calls) == 1


def test_collector_cannot_silently_copy_previous_training(fixture):
    fixture.fresh["source_corpus"]["reused_splits"] = ["train", "validation"]
    assert cli.main(fixture.argv) == 1
    assert "five independent splits" in report(fixture)["error"] and not fixture.train_calls


def test_reference_audit_failure_keeps_models_partial_features_and_previous_audits(fixture):
    fixture.audit_failure = "seed-0/value_base/evaluation/snapshot-00-1800s"
    assert cli.main(fixture.argv) == 1
    result = report(fixture)
    assert result["failed_stage"] == "frozen_audit_seed-0/value_base/snapshot-00-1800s"
    assert "synthetic audit failure" in result["error"]
    assert len(cli.scaleup.read_json(fixture.output / "frozen-checkpoints.json")["models"]) == 12
    assert len(cli.scaleup.read_json(result["completed_audit_receipt"])["completed_audits"]) == 1
    assert (fixture.output / fixture.audit_failure / "partial-audit.json").exists()


def test_unreached_snapshots_remain_unreached_not_fabricated(fixture):
    fixture.missing_snapshot = True
    assert cli.main(fixture.argv) == 0
    assert report(fixture)["all_learning_curve_thresholds_reached"] is False
    assert report(fixture)["generated_answers"] == 1024 and len(fixture.audit_calls) == 8


def test_final_source_mutation_prevents_success(fixture):
    def mutate(item, key):
        if key == "seed-1/value_base/evaluation/snapshot-01-3600s":
            (item.source_root / "manifest.json").write_text("changed")
    fixture.after_audit = mutate
    assert cli.main(fixture.argv) == 1
    assert report(fixture)["failed_stage"] == "final_immutability_checks"
    assert "frozen artifact changed" in report(fixture)["error"]
