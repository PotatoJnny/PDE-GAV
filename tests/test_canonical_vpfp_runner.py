"""Configuration/composition checks, not an actual LLM campaign."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("canonical_vpfp_runner", ROOT / "scripts/run_canonical_vpfp_campaign.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def config():
    return RUNNER.load_config(ROOT / "configs/vpfp_strong_canonical_2round.json")


def test_only_canonical_recipe_with_current_selected_solver_feedback():
    from pde_agent_verifier_solver.sdk import SampleSource, TerminalGuidanceSource
    row = config()
    plan = RUNNER.recipe(row, None)
    assert plan.round_count == 2 and len(plan.agents) == 1
    assert plan.default_round.official_evaluation.round.base == 80
    assert plan.default_round.official_evaluation.round.guided == 0
    assert plan.generator.guidance_enabled and plan.generator.perturbations_enabled
    assert SampleSource.GUIDED in plan.default_round.development_sampling.allowed_sources
    assert not plan.default_round.development_sampling.cumulative_fixed_cases
    assert plan.task_settings["require_prepared_solution"] is True
    assert plan.task_settings["official_point_count"] == 256
    assert not plan.evaluator.allow_point_truth and not plan.evaluator.allow_development_solution_session
    assert plan.terminal_evaluation.cohorts[0].guidance_source is TerminalGuidanceSource.BASE
    assert [c.round_index for c in plan.terminal_evaluation.cohorts[1:]] == [1, 2]
    assert all(c.guidance_source is TerminalGuidanceSource.ROUND for c in plan.terminal_evaluation.cohorts[1:])
    assert "guidance_checkpoint" not in row["backend"]


@pytest.mark.parametrize("change,match", [
    (lambda c: c["official"].update(base=64), "fit\\+dev"),
    (lambda c: c["official"].update(guided=16), "clean BASE"),
    (lambda c: c["backend"]["guidance_config"].update(steps=4), "validated"),
    (lambda c: c["task"].update(require_prepared_solution=False), "query-blind"),
    (lambda c: c["backend"].update(device="cpu"), "Controller GPU"),
])
def test_formal_configuration_rejects_silent_recipe_changes(tmp_path, change, match):
    row = deepcopy(config())
    change(row)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match=match):
        RUNNER.load_config(path)


def test_static_validation_does_not_load_base_or_call_model(capsys):
    assert RUNNER.main(["--validate-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "static_configuration_validated"
    assert not report["assets_loaded"] and not report["model_called"]


def test_formal_execution_rejects_login_node(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(SystemExit) as error:
        RUNNER.main([])
    assert error.value.code == 2


def test_historical_recipe_cannot_silently_start_as_submission_default(monkeypatch, capsys):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_TMPDIR", "/tmp")
    with pytest.raises(SystemExit) as error:
        RUNNER.main([])
    assert error.value.code == 2
    assert "historical diagnostic, not the default submission profile" in capsys.readouterr().err


def test_json_persistence_handles_sdk_data_without_serializing_private_program(tmp_path):
    from pde_agent_verifier_solver.sdk import Artifact, TaskKind
    writer = RUNNER.Writer(tmp_path)
    artifact = Artifact("test", TaskKind.OBSERVABLE_MODEL, "name", object(), "summary", "usage", 0, "fixture")
    writer.write("result.json", {"artifact": artifact, "invalid_error": float("inf")})
    value = json.loads((tmp_path / "result.json").read_text())
    assert "program" not in value["artifact"]
    assert value["invalid_error"] == "inf"
