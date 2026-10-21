"""Synthetic weak-debug mechanics only; no scheduler, GPU or PDE collection."""
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
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import VPFPValueEDMConfig, lift_value_controls

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.util.spec_from_file_location("weak_debug_test", ROOT / "scripts/run_vpfp_weak_regime_debug.py")
cli = importlib.util.module_from_spec(loader)
loader.loader.exec_module(cli)


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


def physical_schema():
    return [{"name": f"{field}.time7.{stat}", "group": "fixture", "unit": "fixture", "scale_floor": 1.e-6}
        for field in ("momentum_x", "momentum_y", "kinetic_energy")
        for stat in ("mean", "std", "spectrum_low", "spectrum_middle", "spectrum_high")]


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    cfg = VPFPStructuredEDMConfig(control_count=3, hermite_channels=2, rho_max_mode=1,
        kinetic_max_mode=1, inactive_dim=3, grid_size=3, width=8, depth=1)
    spec = VPFPOnlineModelSpec("weak_fixture", "base.pt", "a"*64, "expanded.pt", "b"*64,
        cfg.collision_nu_range, cfg.temperature_range, cfg.time_domain)
    source, output = tmp_path / "source", tmp_path / "output"
    (source / "corpus").mkdir(parents=True)
    config = tmp_path / "config.json"
    manifest = {"manifest_sha256": "c"*64, "base_decoder_sha256": "a"*64,
        "model_spec": json.loads(json.dumps(asdict(spec))), "split_counts": dict(cli.scaleup.COUNTS)}
    cli.shared.write_json(config, asdict(spec))
    cli.shared.write_json(source / "corpus/manifest.json", manifest)
    cli.shared.write_json(source / "result.json", {"execution_passed": True, "regime": "weak", "admitted": False})
    cli.shared.write_json(source / "frozen-plan.json", {"schema": "vpfp_paired_scaleup_experiment_v1", "regime": "weak"})
    cli.shared.write_json(source / "fresh-corpus-receipt.json", {"corpus_manifest_sha256": manifest["manifest_sha256"],
        "plan_sha256": cli.shared.file_sha(source / "frozen-plan.json")})
    schemas = {"physical": physical_schema(), "derivative": [{"name": "fixture.dt", "group": "fixture", "unit": "fixture", "scale_floor": 1.e-6}]}
    splits, constraint, frozen = {}, {"passed": True, "source_manifest_sha256": manifest["manifest_sha256"], "splits": {}}, {}
    for index, (role, count) in enumerate(manifest["split_counts"].items()):
        ids = np.arange(index*1000, index*1000+count)
        conditions = cli.shared.balanced_conditions(cfg, count, index+7)
        splits[role] = SimpleNamespace(sample_ids=ids, conditions=conditions)
        constraint["splits"][role] = {"input_answer_sha256": [f"{int(seed):064x}" for seed in ids]}
    for kind, schema in schemas.items():
        directory = source / "reference-features" / kind
        directory.mkdir(parents=True)
        plan = cli.previous.VPFPFeaturePlan() if kind == "physical" else cli.previous.VPFPDerivativeFeaturePlan()
        cli.shared.write_json(directory / "schema.json", {"features": schema, "plan": asdict(plan)})
        frozen[str(directory / "schema.json")] = cli.shared.file_sha(directory / "schema.json")
        for role, count in manifest["split_counts"].items():
            matrix = np.ones((count, len(schema)))
            np.save(directory / f"{role}.npy", matrix, allow_pickle=False)
            split = splits[role]
            cli.shared.write_json(directory / f"{role}.json", {"feature_sha256": cli.shared.file_sha(directory / f"{role}.npy"),
                "corpus_manifest_sha256": manifest["manifest_sha256"], "root_ids": [f"equation:{int(seed)}" for seed in split.sample_ids],
                "answer_sha256": constraint["splits"][role]["input_answer_sha256"], "conditions": split.conditions.tolist()})
            frozen.update({str(directory / f"{role}.{suffix}"): cli.shared.file_sha(directory / f"{role}.{suffix}") for suffix in ("npy", "json")})
    cli.shared.write_json(source / "frozen-reference-features.json", {"sha256": frozen,
        "corpus_manifest_sha256": manifest["manifest_sha256"], "opened_only_after_all_model_hashes_frozen": True})
    item = SimpleNamespace(source=source, output=output, config=config, manifest=manifest, spec=spec,
        schemas=schemas, splits=splits, constraint=constraint, events=[], train_calls=[], sample_calls=[],
        train_failure=None, sample_failure=None, initial_identity_change=None, geometry_identity_change=None,
        parent_identity_change=None, after_train=None, geometry_bindings={})
    monkeypatch.setenv("SLURM_JOB_ID", "mock-not-real-compute")
    monkeypatch.setattr(VPFPOnlineModelSpec, "preflight", lambda self: {})
    monkeypatch.setattr(cli.previous, "read_manifest", lambda *a, **k: deepcopy(manifest))
    monkeypatch.setattr(cli.previous, "load_split", lambda path, role, **k: splits[role])
    monkeypatch.setattr(cli.shared, "model_config", lambda *a: cfg)
    monkeypatch.setattr(cli.shared, "source_fingerprints", lambda: {})
    monkeypatch.setattr(cli.subprocess, "check_output", lambda *a, **k: "mock-frozen-revision")
    monkeypatch.setattr(cli.previous, "audit_value_control_corpus", lambda *a, **k: deepcopy(constraint))
    monkeypatch.setattr(cli.previous, "require_value_control_audit", lambda d: None if d["passed"] else (_ for _ in ()).throw(ValueError("off-subspace")))
    monkeypatch.setattr(cli, "preflight", lambda *a: {"synthetic_only": True})

    def train(args, spec, corpus, directory, **kwargs):
        key = str(directory.relative_to(output))
        assert corpus == source / "corpus"
        assert not (output / "corpus").exists()
        item.events.append("train:" + key)
        item.train_calls.append((key, args.seed, args.checkpoint_seconds, kwargs))
        cli.shared.write_json(directory / "partial-training.json", {"synthetic_only": True})
        if key == item.train_failure:
            raise RuntimeError("synthetic training failure")
        checkpoints = directory / "checkpoints"
        checkpoints.mkdir()
        path = checkpoints / "best.pt"
        path.write_bytes(("mock " + key).encode())
        receipts = []
        for index, seconds in enumerate(args.checkpoint_seconds):
            snapshot = checkpoints / f"snapshot-{index}.pt"
            snapshot.write_bytes(("mock snapshot " + key + str(index)).encode())
            receipts.append({"checkpoint": str(snapshot), "sha256": cli.shared.file_sha(snapshot), "step": index+1,
                "requested_seconds": seconds, "actual_elapsed_seconds": seconds+1, "validation_score": 1.})
        report = {"checkpoint_sha256": cli.shared.file_sha(path), "best_step": 3, "validation_selection_score": .5,
            "initial_model_sha256": "different" if key == item.initial_identity_change else f"same-seed-{args.seed}",
            "checkpoint_snapshots": receipts, "unreached_checkpoint_seconds": []}
        if item.after_train:
            item.after_train(item, key)
        return path, report

    def geometry(args, manifest, checkpoint, digest, arm):
        key = str(checkpoint.parent.parent.relative_to(output))
        binding = {"value_normalizer_sha256": "different" if key == item.parent_identity_change else "shared-parent"}
        if arm != "value_base":
            binding.update(temporal_transform_sha256="different" if key == item.geometry_identity_change else "shared-mu-C",
                covariance_strength=cli.STRENGTHS[arm])
        item.geometry_bindings[key] = binding
        return binding

    original_load = cli.load_reference_features
    def load(*args):
        assert len(item.train_calls) == 8
        registry = cli.scaleup.read_json(output / "frozen-checkpoints.json")
        assert len(registry["models"]) == 24 and registry["snapshots_physically_evaluated"] is False
        item.events.append("open_features")
        return original_load(*args)

    def sample(args, spec, checkpoint, digest, directory, decoder, cached, schemas):
        key = str(directory.parent.parent.relative_to(output))
        assert args.cache == source and "open_features" in item.events and directory.name == "best"
        item.events.append("sample:" + key)
        item.sample_calls.append((key, args.seed, args.generated_answers))
        cli.shared.write_json(directory / "partial-sampling.json", {"synthetic_only": True})
        if key == item.sample_failure:
            raise RuntimeError("synthetic sampling failure")
        np.save(directory / "generated-conditions.npy", cli.shared.balanced_conditions(cfg, 128, args.seed+200), allow_pickle=False)
        matrices = {kind: np.ones((128, len(schema)))*2 for kind, schema in schemas.items()}
        reports = {kind: {"status": "mock-only", "comparisons": {"generated": {"statistics": {"sliced_wasserstein_max": {"effect": 1.}}}}} for kind in schemas}
        return matrices, reports

    def scorecard(**kwargs):
        assert kwargs["baseline_arm"] == "cov_centered_iso"
        assert "value_base" not in kwargs["samples"] and "value_base" not in kwargs["joint_reports"]
        return {"synthetic_only": True, "baseline_arm": kwargs["baseline_arm"]}

    monkeypatch.setattr(cli.shared, "train", train)
    monkeypatch.setattr(cli, "native_geometry_binding", geometry)
    monkeypatch.setattr(cli, "load_reference_features", load)
    monkeypatch.setattr(cli.previous, "sample_and_audit", sample)
    monkeypatch.setattr(cli.previous, "build_vpfp_direction_scorecard", scorecard)
    monkeypatch.setattr(cli.previous, "load_vpfp_diffusion_sampler", lambda *a, **k: None)
    monkeypatch.setattr(cli.previous, "build_vpfp_online_binding", lambda *a, **k:
        SimpleNamespace(plugin=SimpleNamespace(generator=SimpleNamespace(backend=SimpleNamespace(runtime=SimpleNamespace(base_model=None))))))
    item.argv = ["--config", str(config), "--source", str(source), "--output", str(output), "--regime", "weak", "--seed", "8", "--device", "cpu"]
    return item


def test_fixed_budget_order_only_eight_best_audits_and_centered_causal_baseline(fixture):
    assert cli.main(fixture.argv) == 0
    order = [f"seed-0/{arm}" for arm in cli.ARMS] + [f"seed-1/{arm}" for arm in reversed(cli.ARMS)]
    assert [row[0] for row in fixture.train_calls] == order
    assert [row[1] for row in fixture.train_calls] == [8]*4+[10008]*4
    assert all(row[2] == (360.,900.) and row[3]["training_metric_factory"] is None
               and row[3]["validation_metric_factory"] is None for row in fixture.train_calls)
    assert len(fixture.sample_calls) == 8 and all(row[2] == 128 for row in fixture.sample_calls)
    assert fixture.events.index("open_features") == 8
    result = cli.scaleup.read_json(fixture.output / "result.json")
    assert result["execution_passed"] is True and result["admitted"] is False and result["blind_test"] is False
    assert result["new_pde_solves"] == 0 and result["generated_answers"] == 1024
    assert result["snapshots_physically_evaluated"] is False
    plan = cli.scaleup.read_json(fixture.output / "frozen-plan.json")
    assert plan["nominal_total_training_seconds"] == 14400 and plan["causal_baseline"] == "cov_centered_iso"
    for seed in (0,1):
        card = cli.scaleup.read_json(fixture.output / f"seed-{seed}/physical-effect-scorecard.json")
        assert card["heuristic_boolean_used_as_gate"] is False
        diagnostic = cli.scaleup.read_json(fixture.output / f"seed-{seed}/spatial-momentum-diagnostic.json")
        assert diagnostic["descriptive_only"] and not diagnostic["new_scientific_pass_gate"]
        assert diagnostic["time_fraction"] == 1.
        assert len(diagnostic["roles"]["cov_base"]["strata"]) == 8


@pytest.mark.parametrize("option,value", [("--regime","strong"), ("--replicas","1"), ("--train-seconds","360"),
    ("--generated-answers","64"), ("--batch-size","9"), ("--steps","0"), ("--sample-steps","1"),
    ("--covariance-floor","nan"), ("--covariance-floor","0"), ("--seed","-1"), ("--checkpoint-seconds","900")])
def test_invalid_budget_rejected_before_output(fixture, option, value):
    with pytest.raises(SystemExit, match="2"):
        cli.main(fixture.argv + [option,value])
    assert not fixture.output.exists() and not fixture.train_calls


def test_no_allocation_and_existing_output_are_rejected(fixture, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(SystemExit, match="2"):
        cli.main(fixture.argv)
    assert not fixture.output.exists()
    monkeypatch.setenv("SLURM_JOB_ID", "mock")
    fixture.output.symlink_to(fixture.source / "missing")
    with pytest.raises(SystemExit, match="2"):
        cli.main(fixture.argv)


@pytest.mark.parametrize("failure", ("train", "sample"))
def test_failure_preserves_prior_and_partial_artifacts(fixture, failure):
    key = "seed-0/cov_shrink_half"
    setattr(fixture, failure + "_failure", key)
    assert cli.main(fixture.argv) == 1
    report = cli.scaleup.read_json(fixture.output / "result.json")
    assert report["partial_artifacts_preserved"] is True
    assert key in report["failed_stage"]
    assert (fixture.output / "seed-0/value_base/checkpoints/best.pt").exists()
    if failure == "train":
        assert "open_features" not in fixture.events
        assert (fixture.output / key / "partial-training.json").exists()
    else:
        assert (fixture.output / key / "evaluation/best/partial-sampling.json").exists()


@pytest.mark.parametrize("field,message", [("initial_identity_change","parameter identities"),
    ("parent_identity_change","parent value units"), ("geometry_identity_change","training means")])
def test_mismatched_center_parent_or_initialization_prevents_opening_refs(fixture, field, message):
    setattr(fixture, field, "seed-0/cov_shrink_half")
    assert cli.main(fixture.argv) == 1
    report = cli.scaleup.read_json(fixture.output / "result.json")
    assert message in report["error"] and "open_features" not in fixture.events


def test_reference_drift_mid_training_is_caught_before_features(fixture):
    def mutate(item, key):
        if key == "seed-0/value_base":
            (item.source / "reference-features/physical/reference_a.npy").write_bytes(b"changed fixture")
    fixture.after_train = mutate
    assert cli.main(fixture.argv) == 1
    assert "frozen artifact changed" in cli.scaleup.read_json(fixture.output / "result.json")["error"]
    assert "open_features" not in fixture.events


@pytest.mark.parametrize("corruption", ("source_result", "registry_escape", "feature_hash", "answer_hash", "conditions", "schema"))
def test_source_reference_receipts_reject_identity_mismatches(fixture, corruption):
    if corruption == "source_result":
        cli.shared.write_json(fixture.source / "result.json", {"execution_passed": False})
    elif corruption == "registry_escape":
        path = fixture.source / "frozen-reference-features.json"
        registry = cli.scaleup.read_json(path)
        registry["sha256"][str(fixture.source.parent / "outside.npy")] = "a"*64
        cli.shared.write_json(path, registry)
    elif corruption == "feature_hash":
        (fixture.source / "reference-features/physical/reference_a.npy").write_bytes(b"bad fixture")
    else:
        path = fixture.source / "reference-features/physical" / ("schema.json" if corruption == "schema" else "reference_a.json")
        info = cli.scaleup.read_json(path)
        if corruption == "answer_hash": info["answer_sha256"][0] = "f"*64
        elif corruption == "conditions": info["conditions"][0][0] += .001
        else: info["plan"]["grid_size"] = 9
        cli.shared.write_json(path, info)
        registry_path = fixture.source / "frozen-reference-features.json"
        registry = cli.scaleup.read_json(registry_path)
        registry["sha256"][str(path)] = cli.shared.file_sha(path)
        cli.shared.write_json(registry_path, registry)
    if corruption in ("source_result", "registry_escape", "feature_hash"):
        with pytest.raises(ValueError):
            cli.main(fixture.argv)
        assert not fixture.output.exists()
    else:
        assert cli.main(fixture.argv) == 1
        assert "sample:" not in " ".join(fixture.events)


def test_spatial_momentum_diagnostic_is_time7_rms_not_mean_or_all_time():
    schema = physical_schema()
    rows = np.ones((8, len(schema)))
    rows[:,0] = np.array([-3.,3.,-3.,3.,-3.,3.,-3.,3.])
    generated = rows.copy()
    generated[:,4] = np.arange(2.,10.)
    matrices = {role: {"physical": value} for role,value in
        (("reference_a",rows), ("reference_b",rows), ("cov_base",generated))}
    spec = SimpleNamespace(collision_nu_range=(.02,.08), temperature_range=(.8,1.2))
    conditions = np.array([[nu,temp] for nu in (.025,.04,.055,.07) for temp in (.9,1.1)])
    report = cli.momentum_diagnostics(matrices,schema,{role:conditions for role in matrices},spec)
    fields = report["roles"]["cov_base"]["fields"]["momentum_x"]
    assert fields["spatial_mean_case_rms"] == 3.  # Signed ensemble mean is zero.
    assert fields["mean_spectrum_high"] == fields["median_spectrum_high"] == 5.5
    assert fields["p95_spectrum_high"] == pytest.approx(np.quantile(np.arange(2.,10.),.95))
    assert fields["count_above_pooled_reference_max_high"] == 8
    assert all(row["count"] == 1 for row in report["roles"]["cov_base"]["strata"])
    assert "kinetic_energy" in report["roles"]["cov_base"]["fields"]
    assert report["admitted"] is False and report["new_scientific_pass_gate"] is False


def test_four_factory_tiny_training_native_reload_and_shared_train_geometry(tmp_path, monkeypatch):
    cfg = VPFPStructuredEDMConfig(control_count=3, hermite_channels=2, rho_max_mode=1,
        kinetic_max_mode=1, inactive_dim=3, grid_size=3, width=8, depth=1)
    value = VPFPValueEDMConfig.from_full_config(cfg)
    class Split:
        def __init__(self, offset):
            self.latents = lift_value_controls(torch.randn(8,value.latent_dimension,
                generator=torch.Generator().manual_seed(offset)),value).numpy()
            self.sample_ids = np.arange(offset,offset+8)
            self.case_ids = list(map(str,self.sample_ids))
            self.conditions = np.tile([.05,1.],(8,1)).astype("float32")
        def __len__(self): return 8
    splits = {"train":Split(1),"validation":Split(21)}
    manifest = {"manifest_sha256":"c"*64,"base_decoder_sha256":"a"*64,"latent_dimension":cfg.latent_dimension}
    monkeypatch.setattr(cli.shared,"model_config",lambda *a:cfg)
    monkeypatch.setattr(cli.shared,"read_manifest",lambda *a,**k:manifest)
    monkeypatch.setattr(cli.shared,"load_split",lambda path,role,**k:splits[role])
    monkeypatch.setattr(cli.previous,"load_split",lambda path,role,**k:splits[role])
    monkeypatch.setenv("SLURM_JOB_ID","synthetic-native-training-fixture")
    args = SimpleNamespace(seed=8,device="cpu",covariance_floor=.05,learning_rate=2.e-4,
        batch_size=2,steps=2,train_seconds=30,validate_every=1,source_revision="synthetic-fixture",source=tmp_path)
    preflight = cli.preflight(args,None,tmp_path)
    assert set(preflight) == set(cli.ARMS)
    assert all(row["execution_passed"] and not row["scientific_validation"] for row in preflight.values())
    assert len({row["parameter_count"] for row in preflight.values()}) == 1
    reports, bindings = {},{}
    for arm in cli.ARMS:
        config,make,fit = cli.factories(args,None,arm)
        directory = tmp_path / arm
        directory.mkdir()
        checkpoint,report = cli.shared.train(args,SimpleNamespace(base_decoder_sha256="a"*64),tmp_path,directory,
            model_config_factory=lambda *a,config=config:config,model_factory=make,normalizer_factory=fit)
        reports[arm] = report
        bindings[arm] = cli.native_geometry_binding(args,manifest,checkpoint,report["checkpoint_sha256"],arm)
        assert report["residual_metric"] is None
        assert report["training_noise_distribution"]["schema"] == "vpfp_original_clamped_lognormal_training_sigma_v1"
    assert len({r["initial_model_sha256"] for r in reports.values()}) == 1
    assert len({r["value_normalizer_sha256"] for r in bindings.values()}) == 1
    assert len({bindings[arm]["temporal_transform_sha256"] for arm in cli.STRENGTHS}) == 1
    assert {arm:bindings[arm]["covariance_strength"] for arm in cli.STRENGTHS} == cli.STRENGTHS
