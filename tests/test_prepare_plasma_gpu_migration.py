"""Synthetic importer tests: no private assets, CUDA, Slurm or model calls."""
from copy import deepcopy
from hashlib import sha256
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import prepare_plasma_gpu_migration as api


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


@pytest.fixture
def fake_bundle(monkeypatch, tmp_path):
    import pde_agent_verifier_solver.sdk as sdk
    monkeypatch.setattr(sdk, "inspect_candidate_bundle", lambda path: SimpleNamespace(tree_sha256="a" * 64))
    monkeypatch.setattr(api, "ROOT", tmp_path / "new-repo")
    return tmp_path / "S0"


def config(equation, arm, bundle):
    model, effort = api.ARMS[arm]
    return {"schema": "original-frozen-profile", "campaign_id": "original-" + equation + "-" + arm,
        "round_count": 5, "master_seed": 101,
        "execution": {"model": model, "reasoning_effort": effort,
            "agent_hard_wall_seconds": 14400, "candidate_case_hard_wall_seconds": 900,
            "candidate_case_workers": 16 if equation == "strong" else 8,
            "workers_per_gpu": 4, "development_workers_per_gpu": 4,
            "gpu_execution": {"protocol": "shared_gpu_v1", "candidate_devices": ["3"]}},
        "candidate": {"initial_bundle": str(bundle), "initial_bundle_sha256": "a" * 64},
        "official": {"original": 0, "base": 86, "guided": 85, "base_perturbed": 85},
        "development": {"max_cases_per_agent_per_round": 32, "starter_base": 4},
        "task": {"candidate_budget_mode": "ns_remaining_case_v1", "require_prepared_solution": True},
        "terminal": {"frozen_terminal_rule": [256, 256, 128, 128]},
        "promotion": {"frozen_objective": "overall_mean"},
        "generator": {"settings": {"guidance_maximum_wall_seconds": 1800}},
        "backend": {"base_checkpoint": str(bundle.parent / "assets/strong/models/base.pt"),
            "guidance_config": {"steps": 2000, "fit_count": 64, "validation_count": 16},
            "perturbation_config": {"velocity_over_sqrt_bath": .03},
            "oracle_execution_options": {"variant": "batched_geometry_g0_v3"}},
        "agents": [{"agent_id": "a", "prompt": "frozen A scientific instructions"},
                   {"agent_id": "b", "prompt": "frozen B scientific instructions"}],
        "evaluator": {"development_point_count": 128, "official_point_count": 256, "terminal_point_count": 256}}


@pytest.mark.parametrize("equation", ["strong", "weak"])
@pytest.mark.parametrize("arm", ["sol", "astra"])
def test_transform_preserves_science_and_entire_case_budget(equation, arm, fake_bundle, tmp_path):
    original = config(equation, arm, fake_bundle)
    before = deepcopy(original)
    source = tmp_path / "old-repo"
    path = write(source / "config.json", original)
    result = api.transform_config(original, equation=equation, arm=arm, source_root=source, config_path=path)
    assert original == before
    for key in ("backend", "task", "agents", "development", "terminal", "promotion", "generator"):
        assert result[key] == before[key]
    for key, value in before["execution"].items():
        assert result["execution"][key] == value
    assert result["execution"]["candidate_case_hard_wall_seconds"] == 900
    assert "candidate_compute_budget" not in result["task"]
    assert result["execution"]["transient_resume_max_attempts"] == (3 if equation == "weak" else 1)
    assert result["execution"]["transient_resume_until_deadline"] is True
    assert result["execution"]["native_session_persistence"] is True
    assert result["protocol"]["official_nomination_policy"] == {"mode": "single_primary_v1", "start_round": 1}
    assert result["protocol"]["parallel_evaluation"] == "service_global_oracle_v1"
    assert result["execution"]["oracle_max_workers"] == (12 if equation == "strong" else 8)
    assert result["execution"]["solver_batch_size"] == 1
    assert result["execution"]["gpu_slots_per_device"] == 1
    assert result["official"] == {"base": 86, "guided": 85, "base_perturbed": 85}
    assert result["source_experiment"]["sha256"] == api.file_sha(path)
    assert result["source_experiment"]["historical_campaign_imported"] is False


def test_source_model_mismatch_and_changed_s0_are_rejected(fake_bundle, tmp_path):
    original = config("weak", "sol", fake_bundle)
    path = write(tmp_path / "config.json", original)
    with pytest.raises(ValueError, match="model/effort"):
        api.transform_config(original, equation="weak", arm="astra", source_root=tmp_path, config_path=path)
    original["candidate"]["initial_bundle_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="initial program"):
        api.transform_config(original, equation="weak", arm="sol", source_root=tmp_path, config_path=path)


@pytest.mark.parametrize("arm", ["sol", "astra"])
def test_artemis_preserves_deployed_d5_backend_and_verified_primary_prompts(arm, fake_bundle, tmp_path):
    original = config("artemis", arm, fake_bundle)
    source = tmp_path / "old-repo"
    path = write(source / "configs/source.json", original)
    backend = {"factory_settings": {"device": "cuda", "oracle_settings": {"profile": "D5"},
                                   "reward_guidance_settings": {"hermite_degree": 2, "ridge_alpha": 10}}}
    backend_path = write(source / "configs/plasma_artemis_d5_guided.json", backend)
    rows = {}
    for name in ("a", "b"):
        prompt = source / f"configs/{name}.txt"
        prompt.write_text("Primary " + name + "; entire Case budget 900 seconds, no preparation sublimit.")
        rows[name] = {"path": prompt.name, "sha256": api.file_sha(prompt)}
    suffix = "astra_ultra_" if arm == "astra" else ""
    package = write(source / f"configs/artemis_shared_gpu_{suffix}single_primary.json",
                    {"base_config_sha256": api.file_sha(path), "agent_instructions": rows})
    result = api.transform_config(original, equation="artemis", arm=arm, source_root=source, config_path=path)
    assert result["backend"]["factory_settings"] == backend["factory_settings"]
    assert result["backend"]["source_config"] == {"path": str(backend_path), "sha256": api.file_sha(backend_path)}
    assert result["source_experiment"]["primary_package_sha256"] == api.file_sha(package)
    assert all("900 seconds" in row["prompt"] for row in result["agents"])
    assert result["task"]["official_point_count"] == result["task"]["terminal_point_count"] == 256
    assert result["task"]["development_point_count"] == 128
    (source / "configs/a.txt").write_text("tampered prompt")
    with pytest.raises(ValueError, match="migration input changed"):
        api.transform_config(original, equation="artemis", arm=arm, source_root=source, config_path=path)


@pytest.mark.parametrize("actual,dirty", [("b" * 40, ""), ("a" * 40, " M changed.py")])
def test_changed_source_revision_or_dirty_source_is_rejected(monkeypatch, tmp_path, actual, dirty):
    monkeypatch.setattr(api.subprocess, "check_output", lambda cmd, **kwargs: actual if cmd[-2:] == ["rev-parse", "HEAD"] else dirty)
    with pytest.raises(ValueError, match="frozen clean revision"):
        api.checked_source(tmp_path, "a" * 40)


def fixture_plan(tmp_path):
    rows = []
    for equation in ("strong", "weak", "artemis"):
        for arm in api.ARMS:
            name = equation + "-" + arm
            source = tmp_path / name
            path = write(source / "config.json", {"name": name})
            other = write(source / "source-proof.json", {"frozen": True})
            rows.append({"name": name, "source": str(source), "source_revision": "a" * 40,
                "config": str(path), "files": [{"path": str(p), "sha256": api.file_sha(p)} for p in (path, other)]})
    return write(tmp_path / "plan.json", {"entries": rows}), rows


def patch_preparation(monkeypatch):
    campaign = importlib.import_module("plasma_gpu_campaign")
    calls = []
    monkeypatch.setattr(api, "checked_source", lambda path, revision: Path(path))
    monkeypatch.setattr(api, "transform_config", lambda original, **kwargs: {
        "schema": api.SCHEMA, "equation": kwargs["equation"], "name": original["name"],
        "backend": {"frozen_scientific_setting": "unchanged"}})
    monkeypatch.setattr(campaign, "validate_config", lambda value: calls.append(value["name"]))
    monkeypatch.setattr(campaign, "static_check", lambda value: {"status": "synthetic-static-pass", "cuda_qualified": False})
    return calls


def weak_assets():
    return {"portable_base": {"path": "/synthetic/base.pt", "sha256": "a" * 64},
            "historical_runtime_root": {"path": "/synthetic/old", "revision": "b" * 40},
            "historical_path_map": {"path": "/synthetic/map.json", "sha256": "c" * 64}}


def test_complete_batch_checked_before_outputs_and_no_job_mutation(monkeypatch, tmp_path):
    plan, rows = fixture_plan(tmp_path)
    calls = patch_preparation(monkeypatch)
    output = tmp_path / "prepared"
    report = api.prepare(plan, output, weak_assets=weak_assets())
    assert len(calls) == len(report["arms"]) == 6
    assert report["formal_jobs_submitted"] is report["old_jobs_modified"] is report["cuda_qualified"] is False
    for row in report["arms"]:
        path = Path(row["config"])
        assert api.file_sha(path) == row["sha256"]
        imported = json.loads(path.read_text())
        if row["name"].startswith("weak-"):
            assert all(imported["backend"][key] == value for key, value in weak_assets().items())
    with pytest.raises(FileExistsError):
        api.prepare(plan, output, weak_assets=weak_assets())


def test_nonconfig_pinned_input_corruption_leaves_no_output(monkeypatch, tmp_path):
    plan, rows = fixture_plan(tmp_path)
    patch_preparation(monkeypatch)
    Path(rows[-1]["files"][1]["path"]).write_text("tampered nonconfig input")
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="migration input changed"):
        api.prepare(plan, output, weak_assets=weak_assets())
    assert not output.exists()


def test_incomplete_six_arm_plan_leaves_no_output(monkeypatch, tmp_path):
    plan, rows = fixture_plan(tmp_path)
    write(plan, {"entries": rows[:-1]})
    patch_preparation(monkeypatch)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="six authorized"):
        api.prepare(plan, output, weak_assets=weak_assets())
    assert not output.exists()
