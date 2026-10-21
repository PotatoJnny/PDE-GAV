"""Approved formal VPFP composition; synthetic checks, no LLM or model loads."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

import run_vpfp_strong_formal_campaign as runner

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vpfp_strong_formal_5round_s101.json"


def config():
    return runner.load_config(CONFIG)


def test_approved_science_is_exactly_compiled():
    from pde_agent_verifier_solver.engine.campaign import CampaignCompiler
    from pde_agent_verifier_solver.sdk import Artifact, GeneratorCapabilities, GeneratorCheckpoint, TaskKind, TerminalGuidanceSource
    row = config()
    initial = Artifact("synthetic-s0", TaskKind.OBSERVABLE_MODEL, "Synthetic S0",
                       None, "No execution", "Fixture only", 0, "test")
    plan = runner.recipe(row, initial)
    plan = replace(plan, generator=replace(plan.generator,
                   checkpoint=GeneratorCheckpoint("static-vpfp", {})))
    compiled = CampaignCompiler().compile(plan, GeneratorCapabilities(
        trainable=False, offline=False, online=True, guidance=True, perturbations=True))
    assert len(compiled.rounds) == 5
    assert all(len(round_plan.agents) == 2 for round_plan in compiled.rounds)
    for round_plan in compiled.rounds:
        assert round_plan.agent_budget_seconds == 14400
        assert round_plan.development_sampling.max_cases_per_agent_per_round == 32
        assert round_plan.round_hidden.case_count == 1024
        assert round_plan.round_hidden.base == 342
        assert round_plan.round_hidden.guided == round_plan.round_hidden.base_perturbed == 341
        assert round_plan.round_hidden.guided_perturbed == 0
    terminal = compiled.terminal_evaluation
    assert sum(c.suite.case_count for c in terminal.cohorts) == 768
    assert terminal.cohorts[0].guidance_source is TerminalGuidanceSource.BASE
    assert terminal.cohorts[1].guidance_source is TerminalGuidanceSource.ROUND
    assert terminal.cohorts[1].round_index == 1
    assert terminal.ranking.distribution == "overall"
    assert all(c.guidance_source is not TerminalGuidanceSource.FINAL for c in terminal.cohorts)
    assert compiled.task_settings["official_point_count"] == 256
    assert compiled.evaluator.development_point_count == 128
    assert not compiled.evaluator.allow_point_truth


@pytest.mark.parametrize("change", [
    lambda c: c.update(round_count=2),
    lambda c: c.update(master_seed=42),
    lambda c: c["agents"].pop(),
    lambda c: c["official"].update(base=80),
    lambda c: c["terminal"]["cohorts"][1].update(guidance_source="final_guided"),
    lambda c: c["execution"].update(agent_hard_wall_seconds=1800),
    lambda c: c["execution"].update(agent_active_budget_seconds=1800),
    lambda c: c["execution"].update(candidate_case_workers=28),
    lambda c: c["development"].update(max_cases_per_agent_per_round=512),
    lambda c: c["task"].update(official_point_count=96),
    lambda c: c["backend"]["guidance_config"].update(steps=5000),
    lambda c: c["backend"].update(feedback_selection="largest_reward"),
    lambda c: c["candidate"].update(initial_bundle_sha256="0"*64),
])
def test_scientific_values_cannot_silently_change(tmp_path, change):
    row = deepcopy(config())
    change(row)
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        runner.load_config(path)


def test_historical_config_is_not_accepted():
    with pytest.raises(ValueError, match="historical"):
        runner.load_config(ROOT / "configs/vpfp_strong_canonical_2round.json")


def test_gpu_mapping_is_per_agent_and_separate_from_controller():
    mapping = runner.gpu_mapping(config(), ("GPU-a", "GPU-b"), "GPU-controller")
    assert mapping == {"vpfp-broad-explorer": "GPU-a", "vpfp-critical-refiner": "GPU-b"}


@pytest.mark.parametrize("workers,controller", [
    (("0",), "1"), (("0","0"), "1"), (("0","1"), "0"), (("0","1"), ""),
    (("0,1","2"), "3"),
])
def test_shared_or_missing_gpu_tokens_fail(workers, controller):
    with pytest.raises(ValueError):
        runner.gpu_mapping(config(), workers, controller)


def test_static_validation_never_loads_physical_assets(capsys):
    assert runner.main(["--validate-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["round_count"] == 5 and report["agent_count"] == 2
    assert report["official_count"] == 1024 and report["terminal_count"] == 768
    assert not report["assets_loaded"] and not report["model_called"]


def _preflight():
    return dict(preflight_passed=True, execution_passed=True, source_final_verified=True,
                config_final_verified=True, solver_final_verified=True, base_final_verified=True,
                source_revision="frozen-source", config_sha256=runner.file_sha(CONFIG),
                selected_worker_count=8)


def test_measured_worker_choice_does_not_mutate_scientific_json(tmp_path):
    report = tmp_path / "preflight.json"
    report.write_text(json.dumps(_preflight()))
    before = runner.file_sha(CONFIG)
    selected, _ = runner.validate_preflight(report, CONFIG, "frozen-source")
    assert selected == 8 and runner.file_sha(CONFIG) == before


@pytest.mark.parametrize("key,value", [
    ("preflight_passed", False), ("source_final_verified", False),
    ("config_sha256", "wrong"), ("source_revision", "other"),
    ("selected_worker_count", None), ("selected_worker_count", 28),
])
def test_failed_or_mismatched_preflight_cannot_start(tmp_path, key, value):
    row = _preflight()
    row[key] = value
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        runner.validate_preflight(path, CONFIG, "frozen-source")


def test_real_execution_refuses_login_node(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(SystemExit) as caught:
        runner.main([])
    assert caught.value.code == 2


@pytest.mark.parametrize('device_map', [None, {}, {'GPU-controller':'cuda:0','GPU-a':'cuda:1','GPU-b':'cuda:2'}])
def test_composition_supplies_oracle_workers_for_borrowed_devices(tmp_path, monkeypatch, device_map):
    from types import SimpleNamespace
    from pde_agent_verifier_solver.plugins.vpfp import iteration_backend
    from pde_agent_verifier_solver.sdk import Artifact, TaskKind
    initial = Artifact('s0', TaskKind.OBSERVABLE_MODEL, 'S0', None, 'fixture', 'fixture', 0, 'test')
    binding = SimpleNamespace(plugin=SimpleNamespace(task=lambda _: object()),
                              controller=lambda *args, **kwargs: 'controller')
    monkeypatch.setattr(iteration_backend, 'build_vpfp_strong_iteration_binding', lambda **kwargs: binding)
    monkeypatch.setattr(runner, 'inspect_candidate_bundle', lambda _: SimpleNamespace(tree_sha256=runner.S0_SHA))
    monkeypatch.setattr(runner, 'PythonCandidateCodec', lambda **kwargs: SimpleNamespace(bind_existing=lambda *a, **k: initial))
    monkeypatch.setattr(runner, 'build_agent_programs', lambda *a, **k: {})
    monkeypatch.setattr(runner, 'SubprocessAgentExecutionService', lambda *a, **k: k)
    monkeypatch.setattr(runner, 'SubprocessCandidateExecutionService', lambda *a, **k: k)
    monkeypatch.setattr(runner, 'PythonTaskWorkerSpec', SimpleNamespace(from_task_type=lambda _: 'fixture'))
    _, agents, candidates, _ = runner.compose(config(), CONFIG, tmp_path, events=object(),
        codex_binary=tmp_path/'codex', auth_file=tmp_path/'auth',
        worker_gpus=('GPU-a','GPU-b'), controller_gpu='GPU-controller',
        candidate_workers=8, evaluation_device_map=device_map)
    assert agents['evaluation_device_map'] == device_map
    assert candidates['max_workers'] == 8
    assert candidates['oracle_max_workers'] == (8 if device_map else 1)
    assert candidates['case_hard_wall_seconds'] == config()['execution']['candidate_case_hard_wall_seconds']
