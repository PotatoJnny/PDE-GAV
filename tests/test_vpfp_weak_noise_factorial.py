"""Tiny factorial lifecycle/native API tests, never real Slurm/GPU experiments."""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import VPFPValueEDMConfig, lift_value_controls

ROOT = Path(__file__).resolve().parents[1]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
cli = load("weak_factorial_test", ROOT / "scripts/run_vpfp_weak_noise_factorial.py")
debug_tests = load("weak_factorial_shared_fixture", ROOT / "tests/test_vpfp_weak_regime_debug.py")


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    before = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    torch.set_num_threads(1)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    yield
    torch.set_num_threads(before[0])
    torch.use_deterministic_algorithms(before[1])
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark = before[2:]


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    # Reuse only the small synthetic source/reference bank; all factorial
    # freezing and old-feature identity validation run unmocked below.
    original_reference_loader = cli.debug.load_reference_features
    monkeypatch.setattr(debug_tests, "cli", cli.debug)
    item = debug_tests.fixture.__wrapped__(tmp_path, monkeypatch)
    item.anchors = tmp_path / "anchors"
    item.anchors.mkdir()
    item.argv = ["--config", str(item.config), "--source", str(item.source), "--anchors", str(item.anchors),
        "--output", str(item.output), "--regime", "weak", "--seed", "1320260905", "--device", "cpu"]
    args = cli.parser().parse_args(item.argv)
    cfg, _, _ = cli.factories(args, item.spec, cli.NEW_ARMS[0])
    plan = {key: getattr(args, key) for key in cli.PAIRED_SETTINGS}
    plan.update(schema="vpfp_weak_centered_covariance_strength_debug_v1", regime="weak", blind_test=False,
        source=str(item.source), source_corpus_manifest_sha256=item.manifest["manifest_sha256"],
        source_revision="synthetic-revision", config=str(item.config),
        frozen_source_sha256={str(item.config): cli.shared.file_sha(item.config)}, source_file_sha256={})
    cli.shared.write_json(item.anchors / "frozen-plan.json", plan)
    plan_sha = cli.shared.file_sha(item.anchors / "frozen-plan.json")
    models, bindings, frozen = {}, {}, {}
    item.anchor_bits = {}
    for replica in range(2):
        run = deepcopy(args)
        run.seed += 10000*replica
        _, make, _ = cli.factories(run, item.spec, cli.NEW_ARMS[0])
        bits = {field: "synthetic-only" for field in cli.TRAINING_BITS}
        bits.update(seed=run.seed, precision="float32_ieee_no_amp_no_tf32", residual_metric=None,
            common_validation_metric=None, unique_training_answers=512,
            initial_model_sha256=cli.shared.model_state_sha(make(cfg)), ablation={"plan_sha256": plan_sha})
        item.anchor_bits[replica] = bits
        for arm in cli.ANCHOR_ARMS:
            key = f"seed-{replica}/{arm}"
            directory = item.anchors / key
            (directory / "checkpoints").mkdir(parents=True)
            checkpoint = directory / "checkpoints/best.pt"
            checkpoint.write_bytes(("synthetic-anchor:"+key).encode())
            digest = cli.shared.file_sha(checkpoint)
            models[key+"/best"] = {"checkpoint": str(checkpoint), "sha256": digest,
                "arm": arm, "replica": replica, "checkpoint_role": "training_validation_selected_best", "step": 3}
            frozen[str(checkpoint)] = digest
            native = {"checkpoint_sha256": digest, "corpus_manifest_sha256": item.manifest["manifest_sha256"],
                "base_decoder_sha256": item.manifest["base_decoder_sha256"], "training": deepcopy(bits),
                "model_config": cfg.to_dict(), "sampling": {"steps": 64, "churn": 0.}}
            bindings[key] = {"native_sampler": native, "centered": True, "covariance_strength": cli.debug.STRENGTHS[arm],
                "value_normalizer_sha256": "shared-parent", "temporal_transform_sha256": "shared-mu-C"}
            cli.shared.write_json(directory / "training-summary.json", {"checkpoint_sha256": digest,
                "wall_budget_seconds": 1800, "audit_used_for_selection": False,
                "initial_model_sha256": bits["initial_model_sha256"]})
            evaluation = directory / "evaluation/best"
            evaluation.mkdir(parents=True)
            conditions = cli.shared.balanced_conditions(cfg, 128, run.seed+200)
            np.save(evaluation / "generated-conditions.npy", conditions, allow_pickle=False)
            seeds = list(range(run.seed+300000000, run.seed+300000128))
            cli.shared.write_json(evaluation / "generated-provenance.json", {"sampler": native,
                "checkpoint_sha256": digest, "condition_seed": run.seed+200, "noise_seeds": seeds, "pde_solves": 0,
                "coefficient_sha256": "published-synthetic-coefficient-provenance-not-a-real-answer"})
            for kind, schema in item.schemas.items():
                path = evaluation / f"generated-{kind}.npy"
                np.save(path, np.ones((128, len(schema)))*2, allow_pickle=False)
                feature_plan = cli.previous.VPFPFeaturePlan() if kind == "physical" else cli.previous.VPFPDerivativeFeaturePlan()
                hashes = [f"{index+10000:064x}" for index in range(128)]
                cli.shared.write_json(evaluation / f"generated-{kind}.json", {"feature_sha256": cli.shared.file_sha(path),
                    "root_ids": [f"generated:{digest}:{seed}" for seed in seeds], "answer_sha256": hashes,
                    "physical": {"plan": asdict(feature_plan), "per_case": [{"input_answer_sha256": answer,
                        "input_equation_condition": condition.tolist()} for answer, condition in zip(hashes, conditions)]}})
                cli.shared.write_json(evaluation / f"{kind}-distribution-audit.json", audit_report(digest, item.manifest))
    cli.shared.write_json(item.anchors / "frozen-checkpoints.json", {"models": models, "native_geometry_bindings": bindings,
        "sha256": frozen, "all_arms_and_snapshots_frozen_before_any_feature_evaluation": True})
    cli.shared.write_json(item.anchors / "result.json", {"execution_passed": True, "admitted": False})
    monkeypatch.setattr(cli, "preflight", lambda *a: {"synthetic_only": True})

    def train(run, spec, corpus, directory, **kwargs):
        key = str(directory.relative_to(item.output))
        assert directory.name in cli.NEW_ARMS and corpus == item.source / "corpus"
        item.train_calls.append((key, run.seed, kwargs))
        item.events.append("train:"+key)
        cli.shared.write_json(directory / "partial-training.json", {"synthetic_only": True})
        if key == item.train_failure:
            raise RuntimeError("synthetic training failure")
        (directory / "checkpoints").mkdir()
        checkpoint = directory / "checkpoints/best.pt"
        checkpoint.write_bytes(("synthetic-new:"+key).encode())
        snapshots = []
        for index, seconds in enumerate(run.checkpoint_seconds):
            path = directory / f"checkpoints/snapshot-{index}.pt"
            path.write_bytes(("synthetic-snapshot:"+key+str(index)).encode())
            snapshots.append({"checkpoint": str(path), "sha256": cli.shared.file_sha(path), "step": index+1,
                "requested_seconds": seconds, "actual_elapsed_seconds": seconds+1, "validation_score": 1.})
        if item.after_train:
            item.after_train(item, key)
        return checkpoint, {"checkpoint_sha256": cli.shared.file_sha(checkpoint), "best_step": 3,
            "validation_selection_score": .5, "checkpoint_snapshots": snapshots, "unreached_checkpoint_seconds": []}

    def geometry(args, manifest, checkpoint, digest, arm):
        key = str(checkpoint.parent.parent.relative_to(item.output))
        replica = int(checkpoint.parents[2].name.split("-")[1])
        bound = deepcopy(bindings[f"seed-{replica}/cov_centered_iso"])
        bound["native_sampler"]["checkpoint_sha256"] = digest
        if key == item.initial_identity_change:
            bound["native_sampler"]["training"]["initial_model_sha256"] = "bad-init"
        if key == item.geometry_identity_change:
            bound["temporal_transform_sha256"] = "bad-center"
        bound["covariance_recipe"] = cli.RECIPES[arm]
        return bound

    def references(*args):
        registry = cli.scaleup.read_json(item.output / "frozen-checkpoints.json")
        assert len(item.train_calls) == 4 and len(registry["models"]) == 16
        assert registry["reused_anchor_best_count"] == registry["newly_trained_best_count"] == 4
        item.events.append("open_features")
        return original_reference_loader(*args)

    def sample(run, spec, checkpoint, digest, directory, decoder, cached, schemas):
        key = str(directory.parents[1].relative_to(item.output))
        assert directory.parents[1].name in cli.NEW_ARMS and "open_features" in item.events
        item.sample_calls.append((key, run.seed))
        item.events.append("sample:"+key)
        cli.shared.write_json(directory / "partial-sampling.json", {"synthetic_only": True})
        if key == item.sample_failure:
            raise RuntimeError("synthetic sampling failure")
        np.save(directory / "generated-conditions.npy", cli.shared.balanced_conditions(cfg, 128, run.seed+200), allow_pickle=False)
        return ({kind: np.ones((128, len(schema)))*3 for kind, schema in schemas.items()},
            {kind: audit_report(digest, item.manifest) for kind in schemas})

    def scorecard(**kwargs):
        assert set(kwargs["joint_reports"]) == set(cli.ARMS)
        assert kwargs["baseline_arm"] == "cov_centered_iso"
        return {"synthetic_only": True}
    monkeypatch.setattr(cli.shared, "train", train)
    monkeypatch.setattr(cli, "native_geometry_binding", geometry)
    monkeypatch.setattr(cli.debug, "load_reference_features", references)
    monkeypatch.setattr(cli.previous, "sample_and_audit", sample)
    monkeypatch.setattr(cli.previous, "build_vpfp_direction_scorecard", scorecard)
    return item


def audit_report(digest, manifest):
    return {"checkpoint_sha256": digest, "corpus_manifest_sha256": manifest["manifest_sha256"],
        "used_for_checkpoint_selection": False, "independent_confirmation": False, "admitted": False,
        "status": "synthetic-only", "comparisons": {"reference_null": {"fixture": True},
            "generated": {"statistics": {"sliced_wasserstein_max": {"effect": 1.}}}}}


def test_only_four_new_models_and_audits_all_eight_best_frozen_first(fixture):
    assert cli.main(fixture.argv) == 0
    assert [row[0] for row in fixture.train_calls] == [f"seed-0/{arm}" for arm in cli.NEW_ARMS] + [f"seed-1/{arm}" for arm in reversed(cli.NEW_ARMS)]
    assert [row[1] for row in fixture.train_calls] == [1320260905]*2 + [1320270905]*2
    assert len(fixture.sample_calls) == 4 and fixture.events.index("open_features") == 4
    result = cli.scaleup.read_json(fixture.output / "result.json")
    assert result["generated_answers"] == result["reused_generated_answers"] == 512
    assert result["combined_answers"] == 1024 and len(result["training"]) == len(result["reused_anchor_training"]) == 4
    assert not result["blind_test"] and not result["independent_confirmation"] and not result["admitted"]
    assert not (fixture.output / "seed-0/cov_base/checkpoints").exists()
    plan = cli.scaleup.read_json(fixture.output / "frozen-plan.json")
    assert plan["nominal_total_training_seconds"] == 7200 and plan["planned_new_training_models"] == 4
    for replica in range(2):
        report = cli.scaleup.read_json(fixture.output / f"seed-{replica}/spatial-momentum-diagnostic.json")
        assert set(cli.ARMS).issubset(report["roles"]) and report["descriptive_only"]
        card = cli.scaleup.read_json(fixture.output / f"seed-{replica}/physical-effect-scorecard.json")
        assert len(card["energy_comparisons"]) == len(card["shape_comparisons"]) == 2
        assert card["heuristic_boolean_used_as_gate"] is False


@pytest.mark.parametrize("option,value", [("--seed", "7"), ("--replicas", "1"), ("--train-seconds", "360"),
    ("--generated-answers", "64"), ("--batch-size", "8"), ("--width", "128"), ("--sample-steps", "32"),
    ("--covariance-floor", "nan"), ("--learning-rate", ".001"), ("--checkpoint-seconds", "900")])
def test_no_silent_departure_from_anchor_recipe(fixture, option, value):
    with pytest.raises((SystemExit, ValueError)):
        cli.main(fixture.argv + [option, value])
    assert not fixture.output.exists()


def test_no_allocation_or_fresh_output_violation(fixture, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(SystemExit):
        cli.main(fixture.argv)
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-only")
    with pytest.raises(SystemExit):
        cli.main(fixture.argv + ["--output", str(fixture.anchors / "new")])
    fixture.output.symlink_to(fixture.source / "missing")
    with pytest.raises(SystemExit):
        cli.main(fixture.argv)


@pytest.mark.parametrize("field", ("initial_identity_change", "geometry_identity_change"))
def test_pairing_mismatch_stops_before_any_feature_open(fixture, field):
    setattr(fixture, field, "seed-0/cov_energy_iso")
    assert cli.main(fixture.argv) == 1
    result = cli.scaleup.read_json(fixture.output / "result.json")
    assert result["partial_artifacts_preserved"] and "open_features" not in fixture.events
    assert (fixture.output / "seed-0/cov_energy_iso/checkpoints/best.pt").exists()


def test_anchor_feature_byte_drift_after_freeze_stops_training(fixture):
    def drift(item, key):
        if len(item.train_calls) == 1:
            path = item.anchors / "seed-0/cov_base/evaluation/best/generated-physical.npy"
            np.save(path, np.zeros((128, 15)), allow_pickle=False)
    fixture.after_train = drift
    assert cli.main(fixture.argv) == 1
    assert len(fixture.train_calls) == 1 and "open_features" not in fixture.events


@pytest.mark.parametrize("corruption", ("feature_hash", "answer", "condition", "plan", "audit_checkpoint", "noise_seed"))
def test_reused_anchor_identity_is_checked_not_just_loaded(fixture, corruption):
    directory = fixture.anchors / "seed-0/cov_centered_iso/evaluation/best"
    path = directory / ("physical-distribution-audit.json" if corruption == "audit_checkpoint" else
        "generated-provenance.json" if corruption == "noise_seed" else "generated-physical.json")
    info = cli.scaleup.read_json(path)
    if corruption == "feature_hash": info["feature_sha256"] = "0"*64
    elif corruption == "answer": info["answer_sha256"][0] = "0"*64
    elif corruption == "condition": info["physical"]["per_case"][0]["input_equation_condition"][0] += .01
    elif corruption == "plan": info["physical"]["plan"]["grid_size"] = 9
    elif corruption == "audit_checkpoint": info["checkpoint_sha256"] = "0"*64
    elif corruption == "noise_seed": info["noise_seeds"][0] += 1
    cli.shared.write_json(path, info)
    assert cli.main(fixture.argv) == 1
    result = cli.scaleup.read_json(fixture.output / "result.json")
    assert result["failed_stage"].startswith("reuse_frozen_anchor_features")
    assert not fixture.sample_calls


@pytest.mark.parametrize("failure", ("train", "sample"))
def test_partial_failures_are_preserved(fixture, failure):
    setattr(fixture, failure+"_failure", "seed-0/cov_unit_shape")
    assert cli.main(fixture.argv) == 1
    assert (fixture.output / "seed-0/cov_energy_iso/checkpoints/best.pt").exists()
    assert cli.scaleup.read_json(fixture.output / "result.json")["partial_artifacts_preserved"]


def test_physical_source_changes_are_not_allowed():
    path = cli.VPFP_PREFIX + "diffusion_features.py"
    with pytest.raises(ValueError, match="outside explicit"):
        cli.compare_code(None, {"source_file_sha256": {path: "a"*64}}, {path: "b"*64})


def test_only_unused_native_dispatch_extension_is_allowed(tmp_path, monkeypatch):
    path = cli.VPFP_PREFIX + "structured_diffusion.py"
    original = "def load_vpfp_diffusion_sampler(x):\n    return x\n\ndef physical(x):\n    return x * 2\n"
    extended = original.replace("    return x\n", "    if x == 'separate':\n        from .structure_prior_diffusion import load\n        return load(x)\n    return x\n", 1)
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_text(extended)
    old = {"source_revision": "fixture", "source_file_sha256": {path: sha256(original.encode()).hexdigest()}}
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli.subprocess, "check_output", lambda *a, **k: original)
    report = cli.compare_code(None, old, {path: cli.shared.file_sha(target)})
    assert report["physical_training_value_decoder_and_feature_sources_unchanged"]
    target.write_text(extended.replace("x * 2", "x * 3"))
    with pytest.raises(ValueError, match="semantics changed"):
        cli.compare_code(None, old, {path: cli.shared.file_sha(target)})


def test_real_tiny_new_arm_preflight_and_fixed_init(monkeypatch):
    from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import VPFPStructuredEDMConfig
    full = VPFPStructuredEDMConfig(control_count=3, hermite_channels=2, rho_max_mode=1,
        kinetic_max_mode=1, inactive_dim=3, grid_size=3, width=8, depth=1)
    cfg = VPFPValueEDMConfig.from_full_config(full)
    raw = lift_value_controls(torch.randn(8, cfg.latent_dimension), cfg)
    conditions = cli.shared.balanced_conditions(cfg, 16, 7)[:8]
    split = SimpleNamespace(latents=raw.numpy(), conditions=conditions, sample_ids=np.arange(8))
    monkeypatch.setattr(cli.previous, "load_split", lambda *a, **k: split)
    monkeypatch.setattr(cli.shared, "model_config", lambda *a: full)
    args = SimpleNamespace(seed=1320260905, batch_size=4, device="cpu", covariance_floor=.05)
    reports = cli.preflight(args, None, None)
    assert set(reports) == set(cli.NEW_ARMS) and all(row["execution_passed"] for row in reports.values())
    models, normalizers = [], []
    for arm in cli.NEW_ARMS:
        _, make, fit = cli.factories(args, None, arm)
        models.append(cli.shared.model_state_sha(make(cfg)))
        normalizers.append(fit(cfg, raw, sample_ids=range(8)))
    assert len(set(models)) == 1
    assert normalizers[0].parent.to_payload() == normalizers[1].parent.to_payload()
    assert normalizers[0].temporal.to_payload() == normalizers[1].temporal.to_payload()
