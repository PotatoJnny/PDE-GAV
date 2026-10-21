"""The existing runner loads numerical NS artifacts after Reconstruction."""
from dataclasses import replace
import importlib.util
import json
import pickle
from pathlib import Path

import pytest

from pde_agent_verifier_solver.plugins.ns.experimental_mms_project import (
    ContinuousPIVReconstruction, ExperimentalMMSProblem, save_prepared_mms_dataset,
)
from pde_agent_verifier_solver.plugins.ns.product import build_ns_plugin
from tests.test_ns_experimental_mms_project import _fixture, _METHOD
from tests.test_ns_experimental_mms_backend import _prepared, _artifacts, _settings

ROOT = Path(__file__).resolve().parents[1]


def _runner():
    spec = importlib.util.spec_from_file_location("_mms_entry_runner", ROOT / "scripts/run_canonical_ns_campaign.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(tmp_path):
    prepared = _prepared()
    saved = save_prepared_mms_dataset(prepared, tmp_path / "prepared.json")
    flow, pca = _artifacts(tmp_path / "weights", prepared)
    config = json.loads((ROOT / "configs/ns_experimental_mms_5round.example.json").read_text())
    config["round_count"] = 1
    config["_config_path"] = str(ROOT / "configs/ns_experimental_mms_5round.example.json")
    config["candidate"]["initial_bundle"] = str(ROOT / "configs/candidates/ns_hold_initial_bundle")
    config["dataset"] = {"dataset_id": prepared.dataset.dataset_id, "manifest": str(saved), "settings": {}}
    config["generator"]["artifacts"] = {"flow_checkpoint": str(flow), "pca_checkpoint": str(pca)}
    config["generator"]["runtime"] = _settings()
    config["generator"]["guidance"]["protocol"]["official_sample_settings"].update(flow_state="ema", integration_steps=2)
    config["official"]["original"] = 1
    config["development"]["starter"]["original"] = 1
    config["terminal"]["cohorts"][0]["suite"]["original"] = 1
    return config


def test_new_task_factory_uses_original_generator():
    from pde_agent_verifier_solver.plugins.ns.generator import NSStructuredGenerator
    from pde_agent_verifier_solver.plugins.ns.tasks.experimental_mms import NSExperimentalMMSSolverTask
    from pde_agent_verifier_solver.panel.controller import PanelController
    from pde_agent_verifier_solver.app import ProductApplication, ProgramBundle
    plugin = build_ns_plugin(task_id="experimental_mms_solver")
    assert isinstance(plugin.problem, ExperimentalMMSProblem)
    assert type(plugin.generator) is NSStructuredGenerator
    assert isinstance(plugin.task(), NSExperimentalMMSSolverTask)
    programs = ProgramBundle(plugin.problem, plugin.task(), plugin.generator, {}, plugin.public_problem_bundle)
    panel = PanelController(ProductApplication.local_development(), programs)
    assert panel.setup_page()["task"]["title"] == "Experimental-data NS solver"
    assert "query_boundary_velocity" in {m.name for m in plugin.public_problem_bundle.oracle_methods}
    assert "query_boundary_velocity" not in {m.name for m in build_ns_plugin().public_problem_bundle.oracle_methods}


@pytest.mark.parametrize("settings", [{"spatial_degree": 2}, {"pressure_model": "learned"}, {"flow_steps": 2}])
def test_reconstruction_rejects_downstream_algorithm_settings(tmp_path, settings):
    dataset, _, _ = _fixture(tmp_path)
    source = ExperimentalMMSProblem().prepare_reconstruction_source(dataset, {}, 0)
    with pytest.raises(ValueError, match="unsupported reconstruction"):
        ContinuousPIVReconstruction().reconstruct(source.program_input, settings, 0)


def test_source_rejects_unrelated_profile_settings(tmp_path):
    dataset, _, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="unsupported"):
        ExperimentalMMSProblem().prepare_reconstruction_source(dataset, {"development_cases": 1}, 0)


def test_runner_loads_shared_backend_and_reuses_frozen_originals(tmp_path, monkeypatch):
    from pde_agent_verifier_solver.plugins.ns.canonical_backend import CanonicalStructuredNSBackend
    config = _config(tmp_path)
    runner = _runner()
    config_path = ROOT / "configs/ns_experimental_mms_5round.example.json"
    runner._validate_config(config)
    run_root = tmp_path / "run"
    result = runner._compose_experimental_mms(config, config_path, run_root)
    prepared, _, generator, recipe, _, provenance = result
    assert type(generator.backend) is CanonicalStructuredNSBackend
    assert generator.backend.guidance_backend is not None
    assert recipe.checkpoint.payload == generator.backend.base_state
    before = {p.name: p.read_bytes() for p in (run_root / "controller_private").iterdir()}
    def forbidden(*args, **kwargs):
        raise AssertionError("loading a frozen Run must not reconstruct or train")
    monkeypatch.setattr(ContinuousPIVReconstruction, "reconstruct", forbidden)
    resumed = runner._compose_experimental_mms(config, config_path, run_root)
    assert resumed[5] == provenance
    for role in ("generator_training_cases", "generator_validation_cases", "development_cases", "hidden_cases"):
        assert [r.case for r in getattr(resumed[0], role)] == [r.case for r in getattr(prepared, role)]
    assert before == {p.name: p.read_bytes() for p in (run_root / "controller_private").iterdir()}
    changed = json.loads(json.dumps(config))
    changed["generator"]["runtime"]["integration_steps"] += 1
    with pytest.raises(ValueError, match="frozen inputs"):
        runner._compose_experimental_mms(changed, config_path, run_root)
    Path(config["generator"]["artifacts"]["flow_checkpoint"]).write_bytes(b"changed")
    with pytest.raises(pickle.UnpicklingError):
        runner._compose_experimental_mms(config, config_path, run_root)


def test_runner_rejects_exhausted_originals_before_loading_model(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config["round_count"] = 3
    config["official"]["original"] = 0
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid inventory reached model loading")
    monkeypatch.setattr("pde_agent_verifier_solver.plugins.ns.experimental_mms.assemble_experimental_mms", forbidden)
    with pytest.raises(ValueError, match="Development originals exceed"):
        _runner()._compose_experimental_mms(config, ROOT / "configs/ns_experimental_mms_5round.example.json", tmp_path / "not-started")


def test_preflight_rejects_raw_reconstruction(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config["reconstruction"] = {"method_settings": dict(_METHOD)}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    runner = _runner()
    monkeypatch.setattr(runner, "_gpu_allocation", lambda *a, **k: (("0", "1"), "2"))
    monkeypatch.setattr(runner, "_required_executable", lambda *a: config_path)
    monkeypatch.setattr(runner, "_required_regular_file", lambda *a: config_path)
    with pytest.raises(ValueError, match="preflight requires a saved"):
        runner._execute(["--config", str(config_path), "--run-root", str(tmp_path / "preflight"), "--preflight-only"])
    assert not (tmp_path / "preflight").exists()


def test_hidden_guidance_remains_enabled_when_agent_guidance_is_disabled(tmp_path):
    config = _config(tmp_path)
    config["generator"].update(guidance_enabled=False, perturbations_enabled=False)
    result = _runner()._compose_experimental_mms(config,
        ROOT / "configs/ns_experimental_mms_5round.example.json", tmp_path / "hidden-guidance")
    generator, recipe = result[2], result[3]
    assert recipe.guidance_enabled and recipe.perturbations_enabled
    assert generator.backend.supports_guidance
    assert not generator.backend.development_guidance_enabled
    assert not generator.backend.development_perturbations_enabled


@pytest.mark.parametrize("profile", ["experimental_mms", "experimental_piv"])
def test_shared_continuous_suite_accepts_mms_but_rejects_piv(profile):
    runner = _runner()
    config, _ = runner._load_config(ROOT / "configs/ns_experimental_mms_5round.example.json")
    config["profile"] = profile
    config["shared_evaluation_suite_manifest"] = "shared_continuous_suite.json"
    if profile == "experimental_piv":
        with pytest.raises(ValueError, match="structured numerical profile"):
            runner._validate_config(config, inspect_initial_candidate=False)
    else:
        runner._validate_config(config, inspect_initial_candidate=False)
        assert runner._hidden_evaluation_manifest_key(config) == "shared_evaluation_suite_manifest"
