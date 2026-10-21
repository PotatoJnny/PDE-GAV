"""Synthetic orchestration only: no real model, CUDA, PDE, or scheduler."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

cli = load("structure_pilot_test_cli", "scripts/run_vpfp_weak_structure_prior.py")
old = load("structure_pilot_fixture_reuse", "tests/test_vpfp_weak_regime_debug.py")
runtime = old.runtime


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    original_load = old.cli.load_reference_features
    item = old.fixture.__wrapped__(tmp_path, monkeypatch)
    for name, value in (("weak",old.cli), ("previous",old.cli.previous),
                        ("shared",old.cli.shared), ("scaleup",old.cli.scaleup)):
        monkeypatch.setattr(cli, name, value)
    monkeypatch.setattr(cli, "preflight", lambda *a: {"synthetic_only":True})
    def native(args, manifest, entry, arm):
        key = str(Path(entry["checkpoint"]).parent.parent.relative_to(item.output))
        return {"normalizer_sha256":"different" if key == item.geometry_identity_change else "same-half"}
    monkeypatch.setattr(cli, "native_binding", native)
    def features(*args):
        assert len(item.train_calls) == 6
        registry = cli.scaleup.read_json(item.output / "frozen-checkpoints.json")
        assert len(registry["models"]) == 18
        assert registry["snapshots_physically_evaluated"] is False
        item.events.append("open_features")
        return original_load(*args)
    monkeypatch.setattr(old.cli, "load_reference_features", features)
    def diagnostic(checkpoints, corpus, output, **kwargs):
        assert len(item.train_calls) == 6 and "open_features" in item.events
        assert set(checkpoints) == set(cli.ARMS)
        assert kwargs["sample_count"] == 8 and kwargs["batch_size"] == 2
        assert kwargs["expected_corpus_sha256"] == item.manifest["manifest_sha256"]
        item.events.append("diagnostic")
        output.mkdir()
        cli.shared.write_json(output / "result.json", {"synthetic_only":True})
        return {"execution_passed": not getattr(item, "diagnostic_failure", False), "error":"synthetic diagnostic failure"}
    monkeypatch.setattr(cli, "load_diagnostic_module",
        lambda: SimpleNamespace(run_structure_denoising_diagnostic=diagnostic))
    item.argv += ["--seed","1320260905","--device","cuda"]
    return item


def test_all_six_models_freeze_before_features_and_diagnostics(fixture):
    assert cli.main(fixture.argv) == 0
    order = [f"seed-0/{a}" for a in cli.ARMS] + [f"seed-1/{a}" for a in reversed(cli.ARMS)]
    assert [r[0] for r in fixture.train_calls] == order
    assert [r[1] for r in fixture.train_calls] == [1320260905]*3+[1320270905]*3
    assert all(r[2] == (360.,900.) for r in fixture.train_calls)
    assert fixture.events.index("open_features") == 6
    assert len(fixture.sample_calls) == 6 and fixture.events.count("diagnostic") == 2
    result = cli.scaleup.read_json(fixture.output / "result.json")
    assert result["execution_passed"] is True and result["admitted"] is False
    assert result["blind_test"] is False and result["new_pde_solves"] == 0
    assert result["generated_answers"] == 768
    plan = cli.scaleup.read_json(fixture.output / "frozen-plan.json")
    assert plan["nominal_total_training_seconds"] == 10800
    assert plan["full_trunk_support_preserved"] is True
    assert plan["objective"] == "original_EDM_only"


@pytest.mark.parametrize("option,value", [("--replicas","1"),("--train-seconds","360"),("--generated-answers","64"),
    ("--width","128"),("--depth","3"),("--seed","2"),("--device","cpu"),("--steps","50000"),
    ("--batch-size","8"),("--sample-steps","32"),("--covariance-floor","nan"),("--learning-rate","0.001"),
    ("--checkpoint-seconds","900")])
def test_frozen_recipe_cannot_silently_change(fixture, option, value):
    with pytest.raises(SystemExit, match="2"):
        cli.main(fixture.argv+[option,value])
    assert not fixture.output.exists() and not fixture.train_calls


@pytest.mark.parametrize("kind", ["train", "sample"])
def test_partial_failure_preserved(fixture, kind):
    key = "seed-0/branch_identity_control"
    setattr(fixture, kind+"_failure", key)
    assert cli.main(fixture.argv) == 1
    result = cli.scaleup.read_json(fixture.output / "result.json")
    assert result["partial_artifacts_preserved"] is True and key in result["failed_stage"]
    assert (fixture.output / "seed-0/cov_shrink_half/checkpoints/best.pt").exists()
    if kind == "train":
        assert "open_features" not in fixture.events


def test_geometry_drift_blocks_reference_open(fixture):
    fixture.geometry_identity_change = "seed-0/lowmode_coherent"
    assert cli.main(fixture.argv) == 1
    assert "open_features" not in fixture.events
    assert "normalizers are not identical" in cli.scaleup.read_json(fixture.output / "result.json")["error"]


def test_source_drift_blocks_reference_open(fixture):
    def mutate(item, key):
        if key == "seed-0/cov_shrink_half":
            (item.source / "reference-features/physical/reference_a.npy").write_bytes(b"changed fixture")
    fixture.after_train = mutate
    assert cli.main(fixture.argv) == 1
    assert "open_features" not in fixture.events


def test_no_allocation_rejected(fixture, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(SystemExit, match="2"):
        cli.main(fixture.argv)
    assert not fixture.output.exists()


def test_diagnostic_failure_does_not_report_success(fixture):
    fixture.diagnostic_failure = True
    assert cli.main(fixture.argv) == 1
    result = cli.scaleup.read_json(fixture.output / "result.json")
    assert result["partial_artifacts_preserved"] is True
    assert "denoising diagnostic failed" in result["error"]
