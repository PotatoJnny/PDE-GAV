"""Mock scale-up lifecycle and tiny feature I/O; no PDE solves or training."""

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
loader = importlib.util.spec_from_file_location("scaleup_experiment_under_test", ROOT / "scripts/run_vpfp_scaleup_experiment.py")
cli = importlib.util.module_from_spec(loader)
loader.loader.exec_module(cli)


@pytest.fixture(autouse=True)
def restore_runtime_settings(monkeypatch):
    state = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.benchmark)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    yield
    torch.set_num_threads(state[0])
    torch.use_deterministic_algorithms(state[1])
    torch.backends.cuda.matmul.allow_tf32 = state[2]
    torch.backends.cudnn.allow_tf32 = state[3]
    torch.backends.cudnn.benchmark = state[4]


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    cfg = VPFPStructuredEDMConfig(control_count=4, hermite_channels=2, rho_max_mode=1,
        kinetic_max_mode=1, inactive_dim=3, grid_size=3, width=8, depth=1)
    spec = VPFPOnlineModelSpec("weak_mock", "base.pt", "a"*64, "expanded.pt", "b"*64,
        cfg.collision_nu_range, cfg.temperature_range, cfg.time_domain)
    output, cache, pilot = tmp_path / "output", tmp_path / "cache", tmp_path / "pilot"
    (cache / "corpus").mkdir(parents=True)
    pilot.mkdir()
    source = {"model_spec": json.loads(json.dumps(asdict(spec))), "manifest_sha256": "c"*64,
        "base_decoder_sha256": "a"*64, "split_counts": dict(cli.COUNTS)}
    fresh = {**deepcopy(source), "manifest_sha256": "d"*64,
        "source_corpus": {"manifest_sha256": source["manifest_sha256"],
            "copy_method": "bytewise_independent_copy", "excluded_equation_seed_count": 2048}}
    config, selection = tmp_path / "config.json", tmp_path / "selection.json"
    cli.shared.write_json(config, asdict(spec))
    cli.shared.write_json(cache / "corpus/manifest.json", source)
    cli.shared.write_json(pilot / "result.json", {"execution_passed": True, "admitted": False,
        "training": {f"seed-{replica}/{arm}": {} for replica in range(2) for arm in cli.temporal.ARMS}})
    cli.shared.write_json(pilot / "frozen-plan.json", {"schema": "vpfp_temporal_geometry_training_pilot_v1",
        "regime": "weak", "source_corpus_manifest_sha256": source["manifest_sha256"]})
    choice = {"schema": "vpfp_scaleup_candidate_selection_v1", "regime": "weak", "candidate": "cov_base",
        "pilot_result_sha256": cli.shared.file_sha(pilot / "result.json"),
        "pilot_plan_sha256": cli.shared.file_sha(pilot / "frozen-plan.json"),
        "evidence": {"synthetic_fixture_only": "not a scientific candidate endorsement"}, "scientific_admission": False}
    cli.shared.write_json(selection, choice)
    item = SimpleNamespace(output=output, cache=cache, pilot=pilot, config=config, selection=selection,
        source=source, fresh=fresh, spec=spec, events=[], train_calls=[], sample_calls=[],
        train_failure=None, sample_failure=None, collection_failure=False, preflight_failure=False,
        constraint_passed=True, after_train=None, after_features=None, after_sample=None,
        initial_identity_change=None, normalizer_identity_change=None, omit_last_threshold=False)
    monkeypatch.setenv("SLURM_JOB_ID", "mock-only-not-compute")
    monkeypatch.setattr(VPFPOnlineModelSpec, "preflight", lambda self: {})
    monkeypatch.setattr(cli.shared, "model_config", lambda *a: cfg)
    monkeypatch.setattr(cli.shared, "source_fingerprints", lambda: {})
    monkeypatch.setattr(cli.subprocess, "check_output", lambda *a, **k: "frozen-mock-revision")
    monkeypatch.setattr(cli.previous, "read_manifest", lambda path, **k:
        deepcopy(fresh if Path(path) == output / "corpus" else source))
    monkeypatch.setattr(cli.previous, "load_split", lambda *a, **k: SimpleNamespace(sample_ids=np.array([1, 2])))

    def constraint(*args, **kwargs):
        item.events.append("source_constraint")
        return {"passed": item.constraint_passed, "source_manifest_sha256": source["manifest_sha256"]}

    def require(report):
        if not report["passed"]:
            raise ValueError("synthetic source constraint failure")

    def preflight(args, spec, corpus):
        item.events.append("preflight")
        assert corpus == cache / "corpus" and not (output / "corpus").exists()
        if item.preflight_failure:
            raise RuntimeError("synthetic preflight failure")
        return {"synthetic_only": True}

    def collect(source_corpus, corpus, seed, **kwargs):
        item.events.append("collect")
        plan = cli.read_json(output / "frozen-plan.json")
        assert plan["manual_selection"] == cli.read_json(selection)
        assert plan["frozen_input_sha256"][str(selection)] == cli.shared.file_sha(selection)
        assert source_corpus == cache / "corpus" and corpus == output / "corpus"
        assert seed == 5000008 and kwargs == {"batch_size": 8, "device": "cpu"}
        corpus.mkdir()
        cli.shared.write_json(corpus / "partial-collection.json", {"synthetic_only": True})
        if item.collection_failure:
            raise RuntimeError("synthetic collection failure")
        cli.shared.write_json(corpus / "manifest.json", fresh)

    def train(args, spec, corpus, directory, **kwargs):
        key = str(directory.relative_to(output))
        item.events.append("train:" + key)
        assert corpus == output / "corpus" and (output / "fresh-corpus-receipt.json").is_file()
        item.train_calls.append({"key": key, "seed": args.seed, "thresholds": args.checkpoint_seconds,
            "ablation": deepcopy(args.ablation_metadata), "factories": kwargs})
        cli.shared.write_json(directory / "partial-training.json", {"synthetic_only": True})
        if key == item.train_failure:
            raise RuntimeError("synthetic training failure")
        checkpoint_root = directory / "checkpoints"
        checkpoint_root.mkdir()
        best = checkpoint_root / "best.pt"
        best.write_bytes(("mock " + key).encode())
        receipts = []
        count = len(args.checkpoint_seconds) - int(item.omit_last_threshold)
        for index, seconds in enumerate(args.checkpoint_seconds[:count]):
            path = checkpoint_root / f"snapshot-step-{index+1:06d}.pt"
            path.write_bytes(("mock snapshot " + key + str(index)).encode())
            receipts.append({"requested_seconds": seconds, "actual_elapsed_seconds": seconds + 1.,
                "step": index + 1, "checkpoint": str(path), "sha256": cli.shared.file_sha(path), "validation_score": 1.})
        report = {"checkpoint_sha256": cli.shared.file_sha(best), "best_step": 3, "validation_selection_score": .5,
            "initial_model_sha256": "same-init-seed-" + str(args.seed),
            "normalizer_sha256": directory.name.split("_")[0], "checkpoint_snapshots": receipts,
            "unreached_checkpoint_seconds": list(args.checkpoint_seconds[count:])}
        if key == item.initial_identity_change:
            report["initial_model_sha256"] = "different-init"
        if key == item.normalizer_identity_change:
            report["normalizer_sha256"] = "different-normalizer"
        if item.after_train:
            item.after_train(item, key)
        return best, report

    def features(*args):
        assert len(item.train_calls) == 4
        frozen = cli.read_json(output / "frozen-checkpoints.json")
        assert len(frozen["models"]) == 4 * (2 if item.omit_last_threshold else 3)
        assert frozen["all_arms_and_snapshots_frozen_before_any_feature_evaluation"] is True
        assert all(cli.shared.file_sha(path) == digest for path, digest in frozen["sha256"].items())
        item.events.append("features")
        matrix = np.ones((128, 1))
        if item.after_features:
            item.after_features(item)
        return {kind: {role: SimpleNamespace(features=matrix) for role in cli.previous.ROLES}
                for kind in ("physical", "derivative")}, {"physical": (), "derivative": ()}, {}

    def sample(args, spec, checkpoint, expected_sha, directory, *rest):
        key = str(directory.relative_to(output))
        item.events.append("sample:" + key)
        item.sample_calls.append((key, args.seed, args.generated_answers))
        assert args.cache == output and cli.shared.file_sha(checkpoint) == expected_sha
        assert "features" in item.events and args.generated_answers == 128
        cli.shared.write_json(directory / "partial-sample.json", {"synthetic_only": True})
        reports = {kind: {"status": "mock-only", "evidence_role": "development_screen_on_previously_opened_reference",
            "comparisons": {"generated": {"statistics": {"sliced_wasserstein_max": {"effect": 1.}}}}}
            for kind in ("physical", "derivative")}
        for kind, report in reports.items():
            cli.shared.write_json(directory / f"{kind}-distribution-audit.json", report)
        if key == item.sample_failure:
            raise RuntimeError("synthetic audit failure")
        if item.after_sample:
            item.after_sample(item, key)
        return {kind: np.ones((128, 1)) for kind in reports}, reports

    monkeypatch.setattr(cli.previous, "audit_value_control_corpus", constraint)
    monkeypatch.setattr(cli.previous, "require_value_control_audit", require)
    monkeypatch.setattr(cli.temporal, "preflight", preflight)
    monkeypatch.setattr(cli, "collect_fresh_evaluation_corpus", collect)
    monkeypatch.setattr(cli.shared, "train", train)
    monkeypatch.setattr(cli, "write_reference_features", features)
    sampler = SimpleNamespace(identity={"corpus_manifest_sha256": fresh["manifest_sha256"], "base_decoder_sha256": "a"*64},
        normalizer=SimpleNamespace(fit_metadata={"sample_ids": [1, 2]}))
    monkeypatch.setattr(cli.previous, "load_vpfp_diffusion_sampler", lambda *a, **k: sampler)
    runtime = SimpleNamespace(base_model=None)
    monkeypatch.setattr(cli.previous, "build_vpfp_online_binding", lambda *a, **k:
        SimpleNamespace(plugin=SimpleNamespace(generator=SimpleNamespace(backend=SimpleNamespace(runtime=runtime)))))
    monkeypatch.setattr(cli.previous, "sample_and_audit", sample)
    monkeypatch.setattr(cli.previous, "build_vpfp_direction_scorecard", lambda **kwargs: {"synthetic_only": True})
    item.argv = ["--config", str(config), "--cache", str(cache), "--pilot", str(pilot), "--selection", str(selection),
        "--output", str(output), "--regime", "weak", "--seed", "8", "--candidate", "cov_base", "--device", "cpu"]
    return item


def result(fixture):
    return cli.read_json(fixture.output / "result.json")


def test_defaults_are_fixed_two_seed_fresh_reference_scaleup(fixture):
    args = cli.parser().parse_args(fixture.argv)
    assert args.replicas == 2 and args.train_seconds == 1800
    assert args.checkpoint_seconds == (360., 900.) and args.generated_answers == 128
    assert args.sample_steps == 64 and args.permutations == 99 and args.bootstrap_repeats == 49
    assert args.collect_batch_size == 8


def test_all_models_and_snapshots_freeze_before_features_and_repeated_audits_are_labeled(fixture):
    assert cli.main(fixture.argv) == 0
    expected = ["source_constraint", "preflight", "collect", "train:seed-0/value_base", "train:seed-0/cov_base",
        "train:seed-1/cov_base", "train:seed-1/value_base", "features"]
    assert fixture.events[:8] == expected
    assert [call["seed"] for call in fixture.train_calls] == [8, 8, 10008, 10008]
    assert all(call["thresholds"] == (360., 900.) for call in fixture.train_calls)
    assert all(call["factories"]["training_metric_factory"] is None and
               call["factories"]["validation_metric_factory"] is None for call in fixture.train_calls)
    assert len(fixture.sample_calls) == 12 and [row[1] for row in fixture.sample_calls] == [8]*6 + [10008]*6
    report = result(fixture)
    assert report["execution_passed"] is True and report["admitted"] is False and report["blind_test"] is False
    assert report["new_pde_solves"] == 384 and report["generated_answers"] == 1536
    assert report["all_learning_curve_thresholds_reached"] is True
    assert report["automatic_followup_job_submitted"] is False
    plan = cli.read_json(fixture.output / "frozen-plan.json")
    assert plan["planned_generated_answers"] == 1536 and plan["nominal_total_training_seconds"] == 7200
    assert "scripts/run_vpfp_temporal_training_pilot.py" in plan["source_file_sha256"]
    assert "scripts/run_vpfp_value_control_pilot.py" in plan["source_file_sha256"]
    for key, _, _ in fixture.sample_calls:
        audit = cli.read_json(fixture.output / key / "physical-distribution-audit.json")
        assert audit["evidence_role"] == cli.EVIDENCE_ROLE
        assert audit["used_for_checkpoint_selection"] is False
        assert audit["independent_confirmation"] is False and audit["multiple_models_and_timepoints_adjusted"] is False
    assert cli.read_json(fixture.output / "seed-0/physical-effect-scorecard.json")["heuristic_boolean_used_as_gate"] is False


def test_missing_learning_curve_thresholds_are_reported_not_invented(fixture):
    fixture.omit_last_threshold = True
    assert cli.main(fixture.argv) == 0
    assert result(fixture)["all_learning_curve_thresholds_reached"] is False
    assert len(fixture.sample_calls) == 8 and result(fixture)["generated_answers"] == 1024


@pytest.mark.parametrize("field,value", [("candidate", "value_low"), ("regime", "strong"),
    ("pilot_result_sha256", "f"*64), ("pilot_plan_sha256", "f"*64),
    ("scientific_admission", True), ("evidence", {}), ("schema", "other")])
def test_manual_selection_must_be_exact_and_sha_bound_before_output(fixture, field, value):
    selection = cli.read_json(fixture.selection)
    selection[field] = value
    cli.shared.write_json(fixture.selection, selection)
    with pytest.raises(ValueError, match="manual candidate selection"):
        cli.main(fixture.argv)
    assert not fixture.output.exists() and not fixture.events


def test_manual_selection_unknown_fields_rejected(fixture):
    selection = cli.read_json(fixture.selection)
    selection["automatic_admission_override"] = False
    cli.shared.write_json(fixture.selection, selection)
    with pytest.raises(ValueError, match="strict schema"):
        cli.main(fixture.argv)


@pytest.mark.parametrize("target,field,value", [("result.json", "execution_passed", False),
    ("result.json", "training", {}), ("frozen-plan.json", "source_corpus_manifest_sha256", "f"*64)])
def test_pilot_must_be_completed_on_same_corpus(fixture, target, field, value):
    path = fixture.pilot / target
    payload = cli.read_json(path)
    payload[field] = value
    cli.shared.write_json(path, payload)
    selection = cli.read_json(fixture.selection)
    selection["pilot_result_sha256" if target == "result.json" else "pilot_plan_sha256"] = cli.shared.file_sha(path)
    cli.shared.write_json(fixture.selection, selection)
    with pytest.raises(ValueError, match="completed two-seed"):
        cli.main(fixture.argv)
    assert not fixture.output.exists()


@pytest.mark.parametrize("option,value", [("--replicas", "3"), ("--generated-answers", "64"),
    ("--checkpoint-seconds", "nan"), ("--checkpoint-seconds", "1801"),
    ("--covariance-floor", "nan"), ("--batch-size", "9"),
    ("--collect-batch-size", "16"), ("--collect-batch-size", "3")])
def test_invalid_fixed_experiment_settings_rejected_before_output(fixture, option, value):
    with pytest.raises(SystemExit, match="2"):
        cli.main(fixture.argv + [option, value])
    assert not fixture.output.exists()


def test_allocation_required(fixture, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(SystemExit, match="2"):
        cli.main(fixture.argv)
    assert not fixture.output.exists()


@pytest.mark.parametrize("attribute,stage", [("preflight_failure", "full_layout_preflight"),
    ("collection_failure", "collect_fresh_evaluation")])
def test_early_failures_preserve_plan_and_never_train(fixture, attribute, stage):
    setattr(fixture, attribute, True)
    assert cli.main(fixture.argv) == 1
    assert result(fixture)["failed_stage"] == stage and result(fixture)["partial_artifacts_preserved"] is True
    assert not fixture.train_calls and not fixture.sample_calls
    assert (fixture.output / "frozen-plan.json").exists()
    if attribute == "preflight_failure":
        assert "collect" not in fixture.events
    else:
        assert (fixture.output / "corpus/partial-collection.json").exists()


def test_training_failure_preserves_prior_snapshots_and_does_not_open_features(fixture):
    fixture.train_failure = "seed-0/cov_base"
    assert cli.main(fixture.argv) == 1
    assert result(fixture)["failed_stage"] == "train_seed-0/cov_base"
    assert (fixture.output / "seed-0/value_base/checkpoints/snapshot-step-000001.pt").exists()
    assert "features" not in fixture.events and "synthetic training failure" in result(fixture)["error"]


def test_changed_snapshot_rejected_before_any_reference_features(fixture):
    def mutate(item, key):
        if key == "seed-1/value_base":
            (item.output / "seed-0/value_base/checkpoints/snapshot-step-000001.pt").write_bytes(b"changed")
    fixture.after_train = mutate
    assert cli.main(fixture.argv) == 1
    assert "frozen artifact changed" in result(fixture)["error"]
    assert "features" not in fixture.events


def test_changed_manual_selection_cannot_be_evaluated(fixture):
    fixture.after_train = lambda item, key: item.selection.write_text(item.selection.read_text() + " ")
    assert cli.main(fixture.argv) == 1
    assert "frozen artifact changed" in result(fixture)["error"] and "features" not in fixture.events


def test_paired_initialization_must_match(fixture):
    fixture.initial_identity_change = "seed-0/cov_base"
    assert cli.main(fixture.argv) == 1
    assert "initial network parameter identities" in result(fixture)["error"]
    assert len(fixture.train_calls) == 2 and "features" not in fixture.events


def test_value_low_pair_normalization_must_match(fixture):
    selection = cli.read_json(fixture.selection)
    selection["candidate"] = "value_low"
    cli.shared.write_json(fixture.selection, selection)
    fixture.argv[fixture.argv.index("cov_base")] = "value_low"
    fixture.normalizer_identity_change = "seed-0/value_low"
    assert cli.main(fixture.argv) == 1
    assert "different normalizers" in result(fixture)["error"] and "features" not in fixture.events


def test_audit_failure_preserves_frozen_models_and_corrects_partial_evidence_role(fixture):
    fixture.sample_failure = "seed-0/value_base/evaluation/best"
    assert cli.main(fixture.argv) == 1
    assert len(cli.read_json(fixture.output / "frozen-checkpoints.json")["models"]) == 12
    assert result(fixture)["failed_stage"] == "frozen_audit_seed-0/value_base/best"
    assert "synthetic audit failure" in result(fixture)["error"]
    partial = cli.read_json(fixture.output / fixture.sample_failure / "physical-distribution-audit.json")
    assert partial["evidence_role"] == cli.EVIDENCE_ROLE and partial["blind_test"] is False


def test_input_changed_on_last_audit_prevents_success(fixture):
    def mutate(item, key):
        if key == "seed-1/value_base/evaluation/snapshot-01-900s":
            (item.pilot / "result.json").write_text((item.pilot / "result.json").read_text() + " ")
    fixture.after_sample = mutate
    assert cli.main(fixture.argv) == 1
    assert len(fixture.sample_calls) == 12 and "frozen artifact changed" in result(fixture)["error"]


class TinySplit:
    latents = np.ones((2, 3), dtype=np.float32)
    conditions = np.array([[.05, 1.], [.06, 1.1]], dtype=np.float32)
    sample_ids = np.array([1, 2])

    def __len__(self):
        return 2


@pytest.mark.parametrize("corrupt", (False, True))
def test_reference_writer_binds_each_physical_and_derivative_answer_snapshot(tmp_path, monkeypatch, corrupt):
    arguments = SimpleNamespace(output=tmp_path)
    manifest = {"manifest_sha256": "a"*64}
    calls = []

    def extract(decoder, raw, conditions, *, plan):
        calls.append(type(plan).__name__)
        hashes = ["b"*64, "c"*64]
        if corrupt and isinstance(plan, cli.previous.VPFPDerivativeFeaturePlan):
            hashes[0] = "d"*64
        return np.ones((2, 1)), [{"name": "mock"}], {"per_case": [
            {"input_answer_sha256": digest, "input_equation_condition": condition.tolist()}
            for digest, condition in zip(hashes, conditions)]}

    monkeypatch.setattr(cli.previous, "load_split", lambda *a, **k: TinySplit())
    monkeypatch.setattr(cli.previous, "read_manifest", lambda *a, **k: manifest)
    monkeypatch.setattr(cli.previous, "extract_vpfp_features", extract)
    monkeypatch.setattr(cli.previous, "extract_vpfp_derivative_features", extract)
    if corrupt:
        with pytest.raises(ValueError, match="different answer snapshots"):
            cli.write_reference_features(arguments, tmp_path / "corpus", None, manifest)
        assert (tmp_path / "reference-features/physical/train.npy").exists()
        return
    samples, schemas, frozen = cli.write_reference_features(arguments, tmp_path / "corpus", None, manifest)
    assert len(calls) == 10 and len(frozen) == 22
    assert samples["physical"]["train"].answer_sha256 == samples["derivative"]["train"].answer_sha256
    assert schemas["physical"] == schemas["derivative"]
    assert all(cli.shared.file_sha(path) == digest for path, digest in frozen.items())
    assert cli.read_json(tmp_path / "frozen-reference-features.json")["opened_only_after_all_model_hashes_frozen"] is True
