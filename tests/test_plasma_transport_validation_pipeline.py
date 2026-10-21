"""CPU orchestration checks only, with fake child receipts."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_plasma_transport_validation as pipeline
from scripts import calibrate_plasma_transport as calibration_script
from pde_agent_verifier_solver.engine.checkpoint_store import source_identity
from tests.test_plasma_transport_calibration_script import _synthetic_collections


def source_configs(tmp_path):
    root = tmp_path / "configs"
    root.mkdir()
    for family in ("strong", "weak", "artemis"):
        (root / f"{family}-sol.json").write_text(json.dumps({
            "schema": "pde_gav_plasma_gpu_v1", "equation": family}))
    return root


def test_pipeline_refuses_login_node(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="compute allocation"):
        pipeline.run(tmp_path, tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("failure", [False, True])
def test_pipeline_isolates_processes_and_retains_failed_stage(tmp_path, monkeypatch, failure):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    configs = source_configs(tmp_path)
    output = tmp_path / "result"
    calls = []
    def execute(args, **kwargs):
        calls.append(args)
        if failure and len(calls) == 2:
            return SimpleNamespace(returncode=7)
        if "fit" in args:
            Path(args[args.index("--output") + 1]).write_text(json.dumps({
                "objective_hash": "synthetic", "floors": {"test": 1}}))
        if "--calibration" in args:
            target = Path(args[args.index("--output-root") + 1])
            target.mkdir()
            (target / "report.json").write_text(json.dumps({"status": "passed", "private_cases_released": True,
                "objective_identity": {"objective_hash": "synthetic"}}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(pipeline.subprocess, "run", execute)
    if failure:
        with pytest.raises(RuntimeError, match="collect-weak"):
            pipeline.run(configs, output)
        assert len(calls) == 2
    else:
        assert pipeline.run(configs, output)["status"] == "passed"
        assert len(calls) == 7
    report = json.loads((output / "validation.json").read_text())
    assert report["status"] == ("failed" if failure else "passed")
    assert report["old_jobs_modified"] is report["formal_jobs_submitted"] is False
    assert report["stages"][-1]["status"] == report["status"]
    assert (output / "collect-weak.err").exists()


@pytest.fixture
def reusable_reference(tmp_path, monkeypatch):
    """Synthetic independent arrays exercise validation, never model quality."""
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    configs = source_configs(tmp_path)
    old, old_source, current = (tmp_path / name for name in ("old", "old-source", "current"))
    old.mkdir()
    for root in (old_source, current):
        for name, text in (("scripts/calibrate_plasma_transport.py", "unchanged science"),
                           ("src/model.py", "unchanged numerical model"),
                           *[(name, str(root)) for name in pipeline.REFERENCE_REUSE_EXCLUSIONS]):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    monkeypatch.setattr(pipeline, "ROOT", current)
    directories = _synthetic_collections(old)
    references = {family: calibration_script._sha(configs / f"{family}-sol.json")
                  for family in calibration_script.COUNTS}
    for directory in directories:
        path = directory / "collection.json"
        manifest = json.loads(path.read_text())
        manifest["config_sha256"] = references[directory.name]
        manifest["collection_script_sha256"] = calibration_script._sha(old_source / "scripts/calibrate_plasma_transport.py")
        path.write_text(json.dumps(manifest))
    frozen = calibration_script.fit(directories, old / "calibration.json")
    report = {"schema": "plasma_transport_v2_validation_pipeline_v1", "status": "failed",
        "source_root": str(old_source), "source_identity": source_identity(old_source),
        "reference_config_hashes": references, "objective_hash": frozen.objective_hash,
        "calibration_sha256": calibration_script._sha(old / "calibration.json"),
        "stages": [{"name": name, "status": "passed", "exit_code": 0}
                   for name in ("collect-strong", "collect-weak", "collect-artemis", "fit")]}
    (old / "validation.json").write_text(json.dumps(report))
    calls = []

    def execute(args, **kwargs):
        calls.append(args)
        if "fit" in args:
            inputs = [args[i + 1] for i, value in enumerate(args) if value == "--input"]
            calibration_script.fit(inputs, args[args.index("--output") + 1])
        else:
            target = Path(args[args.index("--output-root") + 1])
            target.mkdir()
            (target / "report.json").write_text(json.dumps({"status": "passed", "private_cases_released": True,
                "objective_identity": {"objective_hash": frozen.objective_hash}}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pipeline.subprocess, "run", execute)
    return old, configs, current, calls


def test_explicit_reference_reuse_refits_exactly_and_reruns_all_acceptances(tmp_path, reusable_reference):
    old, configs, current, calls = reusable_reference
    before = {str(p.relative_to(old)): p.read_bytes() for p in old.rglob("*") if p.is_file()}
    target = tmp_path / "new"
    report = pipeline.run(configs, target, reference_validation_root=old)
    assert report["status"] == "passed" and len(calls) == 4
    assert all(row["execution"] == "reused_reference" and row["fresh_draws"] == 0 for row in report["stages"][:3])
    assert [row["name"] for row in report["stages"][3:]] == [
        "fit", "acceptance-strong", "acceptance-weak", "acceptance-artemis"]
    assert report["reference_reuse"]["original_status"] == "failed"
    assert (target / "calibration.json").read_bytes() == (old / "calibration.json").read_bytes()
    assert before == {str(p.relative_to(old)): p.read_bytes() for p in old.rglob("*") if p.is_file()}


def test_reference_reuse_checks_independent_refit_bytes(tmp_path, reusable_reference, monkeypatch):
    old, configs, current, calls = reusable_reference
    original = calibration_script.fit

    def changed_serialization(inputs, output):
        result = original(inputs, output)
        path = Path(output)
        path.write_text(path.read_text() + " ")
        return result

    monkeypatch.setattr(calibration_script, "fit", changed_serialization)
    with pytest.raises(ValueError, match="independent refit differs"):
        pipeline.run(configs, tmp_path / "new", reference_validation_root=old)
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["science", "old_source", "config", "asset", "seed", "calibration", "failed_collect"])
def test_reference_reuse_fails_closed_on_changed_evidence(tmp_path, reusable_reference, change):
    old, configs, current, calls = reusable_reference
    if change == "science":
        (current / "src/model.py").write_text("changed model")
    elif change == "old_source":
        (tmp_path / "old-source/src/model.py").write_text("modified frozen source")
    elif change == "config":
        path = configs / "weak-sol.json"
        path.write_text(path.read_text() + " ")
    elif change == "asset":
        (old / "weak/reference-000.npz").write_bytes(b"changed")
    elif change == "seed":
        path = old / "weak/collection.json"
        manifest = json.loads(path.read_text())
        manifest["records"][0]["request_seed"] += 1
        path.write_text(json.dumps(manifest))
    elif change == "calibration":
        path = old / "calibration.json"
        path.write_text(path.read_text() + " ")
    else:
        path = old / "validation.json"
        report = json.loads(path.read_text())
        report["stages"][1]["status"] = "failed"
        path.write_text(json.dumps(report))
    target = tmp_path / "new"
    with pytest.raises(ValueError):
        pipeline.run(configs, target, reference_validation_root=old)
    assert json.loads((target / "validation.json").read_text())["status"] == "failed"
    assert not any("--calibration" in args for args in calls)


@pytest.fixture
def runtime_repair_reference(tmp_path, reusable_reference):
    old, configs, current, calls = reusable_reference
    original = tmp_path / "old-source"
    sources = {
        "src/pde_agent_verifier_solver/plugins/vpfp/canonical_binding.py":
            "class _BoundVPFPContinuousGenerator:\n    def _check_settings(self, settings):\n        return 1\n\n    def numerical(self):\n        return 7\n",
        "scripts/plasma_gpu_campaign.py": "def compose(config):\n    return 1\n\nCONSTANT = 7\n",
        "scripts/check_plasma_transport.py": "def run_acceptance(config):\n    return 1\n\nCONSTANT = 7\n",
    }
    for name, text in sources.items():
        for root in (original, current):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text if root == original else text.replace("return 1", "return 2"))
    weak = "src/pde_agent_verifier_solver/plugins/vpfp/weak_portable_base.py"
    (current / weak).write_bytes((original / weak).read_bytes())
    path = old / "validation.json"
    report = json.loads(path.read_text())
    report["source_identity"] = source_identity(original)
    path.write_text(json.dumps(report))
    return old, configs, current, calls


def test_runtime_repair_reuse_is_explicit_and_preserves_frozen_calibration(tmp_path, runtime_repair_reference):
    old, configs, current, calls = runtime_repair_reference
    with pytest.raises(ValueError, match="reuse scope"):
        pipeline.run(configs, tmp_path / "without-opt-in", reference_validation_root=old)
    before = {str(p.relative_to(old)): p.read_bytes() for p in old.rglob("*") if p.is_file()}
    target = tmp_path / "new-repair"
    report = pipeline.run(configs, target, reference_validation_root=old, reference_runtime_repair=True)
    assert report["status"] == "passed" and len(calls) == 4
    assert [row["name"] for row in report["stages"][3:]] == [
        "fit", "acceptance-strong", "acceptance-weak", "acceptance-artemis"]
    scope = report["reference_reuse"]["runtime_repair_scope"]
    assert scope["unchanged_outside_listed_function_bodies"] is True
    assert set(scope["files"]) == set(pipeline.REFERENCE_RUNTIME_REPAIR_FUNCTIONS)
    assert all(row["original_sha256"] != row["current_sha256"] for row in scope["files"].values())
    assert (target / "calibration.json").read_bytes() == (old / "calibration.json").read_bytes()
    assert before == {str(p.relative_to(old)): p.read_bytes() for p in old.rglob("*") if p.is_file()}


@pytest.mark.parametrize("change", ["outside_body", "header", "missing_function", "duplicate_function", "decorator", "weak_loader", "calibration_science"])
def test_runtime_repair_rejects_changes_outside_reviewed_bodies(tmp_path, runtime_repair_reference, change):
    old, configs, current, calls = runtime_repair_reference
    path = current / "scripts/plasma_gpu_campaign.py"
    text = path.read_text()
    if change == "outside_body":
        path.write_text(text.replace("CONSTANT = 7", "CONSTANT = 8"))
    elif change == "header":
        path.write_text(text.replace("compose(config)", "compose(config, extra=None)"))
    elif change == "missing_function":
        path.write_text(text.replace("compose(config)", "different(config)"))
    elif change == "duplicate_function":
        path.write_text(text + "\ndef compose(config):\n    return 3\n")
    elif change == "decorator":
        path.write_text("@changed\n" + text)
    elif change == "weak_loader":
        (current / "src/pde_agent_verifier_solver/plugins/vpfp/weak_portable_base.py").write_text("changed")
    else:
        (current / "scripts/calibrate_plasma_transport.py").write_text("changed")
    with pytest.raises(ValueError):
        pipeline.run(configs, tmp_path / "rejected", reference_validation_root=old, reference_runtime_repair=True)
    assert not calls
    assert json.loads((tmp_path / "rejected/validation.json").read_text())["status"] == "failed"


def test_runtime_repair_requires_original_reference_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    with pytest.raises(ValueError, match="original reference validation root"):
        pipeline.run(tmp_path, tmp_path / "new", reference_runtime_repair=True)
    assert not (tmp_path / "new").exists()
