"""GPU runner wiring only: synthetic factories, no model/GPU/Slurm execution."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.engine import GeneratorRecipe
from pde_agent_verifier_solver.engine.runtime import EventStream
from pde_agent_verifier_solver.sdk import GeneratorCapabilities, GeneratorCheckpoint

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "ns": ("run_canonical_ns_campaign.py", "ns_canonical_structured_5round.json"),
    "artemis": ("run_canonical_artemis_campaign.py", "artemis_canonical_5round.json"),
    "strong": ("run_vpfp_strong_formal_campaign.py", "vpfp_strong_formal_5round_s101.json"),
}


@pytest.fixture
def runners(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    result = {}
    for name, (script, _) in FILES.items():
        spec = importlib.util.spec_from_file_location("gpu_runner_" + name, ROOT / "scripts" / script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result[name] = module
    return result


def configuration(name, *, gpu=False):
    path = ROOT / "configs" / FILES[name][1]
    value = json.loads(path.read_text())
    # This suite deliberately exercises the historical CPU vs dedicated GPU paths.
    value["execution"].pop("candidate_device", None)
    value["execution"].pop("gpu_execution", None)
    value["execution"].pop("development_solver_placement", None)
    if name == "ns":
        value["_config_path"] = str(path)
    if gpu:
        value["execution"]["gpu_execution"] = {
            "protocol": "shared_gpu_v1", "candidate_devices": ["3"],
            "agent_sandbox_backend": "bwrap-nvidia-v1", "agent_runtime_sha256": "ab" * 32,
        }
        value["execution"].pop("agent_active_budget_seconds", None)
        if name in ("artemis", "strong"):
            value["task"].pop("candidate_compute_budget")
            value["task"]["candidate_budget_mode"] = "ns_remaining_case_v1"
    return value, path


def validate(name, runner, value, tmp_path):
    if name == "ns":
        runner._validate_config(value, inspect_initial_candidate=False)
    elif name == "artemis":
        runner._validate_frozen_protocol(value)
    else:
        path = tmp_path / "strong.json"
        path.write_text(json.dumps(value))
        assert runner.load_config(path) == value


@pytest.mark.parametrize("name", FILES)
@pytest.mark.parametrize("gpu", [False, True])
def test_existing_and_explicit_gpu_configs_validate_without_changing_science(runners, tmp_path, name, gpu):
    value, _ = configuration(name, gpu=gpu)
    before = deepcopy(value)
    validate(name, runners[name], value, tmp_path)
    assert value == before
    legacy, _ = configuration(name)
    for key in ("agents", "official", "terminal", "generator", "development", "promotion"):
        if key in legacy:
            assert value[key] == legacy[key]
    if name == "artemis":
        assert legacy["task"]["candidate_compute_budget"] == 600
        assert legacy["execution"]["agent_active_budget_seconds"] == 7200
        assert runners[name]._FROZEN_SECTIONS["task"]["candidate_compute_budget"] == 600
    if name == "strong":
        assert legacy["task"]["candidate_compute_budget"] == 180


@pytest.mark.parametrize("name", FILES)
@pytest.mark.parametrize("change", ["case", "agent", "prepare", "active", "protocol"])
def test_gpu_budget_drift_fails_before_assets(runners, tmp_path, name, change):
    value, _ = configuration(name, gpu=True)
    if change == "case":
        value["execution"]["candidate_case_hard_wall_seconds"] = 180
    elif change == "agent":
        value["execution"]["agent_hard_wall_seconds"] = 7200
    elif change == "prepare":
        value.setdefault("task", {})["candidate_compute_budget"] = 180
    elif change == "active":
        value["execution"]["agent_active_budget_seconds"] = 7200
    else:
        value["execution"]["gpu_execution"]["protocol"] = "unknown"
    if name == "ns" and change in ("case", "agent"):
        validate(name, runners[name], value, tmp_path)
    else:
        with pytest.raises(ValueError):
            validate(name, runners[name], value, tmp_path)


@pytest.mark.parametrize("name", ["artemis", "strong"])
def test_budget_only_changes_are_not_admitted_without_gpu_optin(runners, tmp_path, name):
    value, _ = configuration(name, gpu=True)
    value["execution"].pop("gpu_execution")
    with pytest.raises(ValueError):
        validate(name, runners[name], value, tmp_path)


@pytest.mark.parametrize("name", ["artemis", "strong"])
def test_gpu_does_not_relax_frozen_scientific_values(runners, tmp_path, name):
    value, _ = configuration(name, gpu=True)
    value["official"]["base"] += 1
    with pytest.raises(ValueError):
        validate(name, runners[name], value, tmp_path)


def test_ns_gpu_round_budget_is_configurable(runners, tmp_path):
    value, _ = configuration("ns", gpu=True)
    value["round_overrides"] = {"2": {"agent_hard_wall_seconds": 3600}}
    validate("ns", runners["ns"], value, tmp_path)
    value["execution"].pop("gpu_execution")
    validate("ns", runners["ns"], value, tmp_path)


def test_gpu_prompt_has_authoritative_budget_without_rewriting_science(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from run_canonical_codex_agent import _research_prompt
    from pde_agent_verifier_solver.engine.gpu_execution_config import candidate_execution_capabilities
    value, _ = configuration("ns", gpu=True)
    prompt = _research_prompt({
        "round": {"round_index": 1, "active_budget_seconds": None,
            "hard_wall_seconds": 14400, "time_budget_mode": "hard_wall",
            "development_sampling": {"max_cases": 32, "allowed_sources": ["base", "guided"]}},
        "agent": {"prompt": "Scientific role text remains intact."},
        "candidate_execution": {"case_charged_wall_seconds": 900.0},
        "worker_settings": {"candidate_execution": candidate_execution_capabilities(value["execution"])},
    })
    assert "Scientific role text remains intact." in prompt
    assert "900-second per-Case charged-wall" in prompt
    assert "14400 seconds of total elapsed time" in prompt
    assert "No additional preparation" in prompt


def _assert_devices(service, programs, gpu):
    assert service.gpu_devices == (("3",) if gpu else ())
    assert service.case_hard_wall_seconds == 900
    for index, program in enumerate(programs.values()):
        assert program.gpu_devices == (str(index),)
        settings = program.worker_settings
        assert settings["candidate_execution"]["device"] == ("cuda" if gpu else "cpu")
        if gpu:
            assert settings["agent_gpu_execution"] == {
                "protocol": "shared_gpu_v1", "sandbox_backend": "bwrap-nvidia-v1",
                "required": True, "runtime_sha256": "ab" * 32,
            }
            assert settings["candidate_execution"]["cuda_probe_required"] is True
        else:
            assert "agent_gpu_execution" not in settings


@pytest.mark.parametrize("gpu", [False, True])
def test_artemis_real_controller_constructors_use_correct_budget_and_devices(runners, tmp_path, monkeypatch, gpu):
    from tests.test_artemis_canonical import _binding
    from pde_agent_verifier_solver.plugins.vp_1d1v import ArtemisBackend, build_artemis_plugin, product
    runner = runners["artemis"]
    value, path = configuration("artemis", gpu=gpu)
    inputs = runner._static_inputs(value, path)
    binding, source = _binding()
    guide = SimpleNamespace(fit_guidance=lambda *a: pytest.fail("no training"),
                            generate=lambda *a: pytest.fail("no generation"))
    backend = ArtemisBackend(source, physical_revision="synthetic-not-science", guidance_adapter=guide)
    binding = replace(binding, plugin=build_artemis_plugin(backend), checkpoint=backend.checkpoint(),
                      report={"guidance_enabled": True, "perturbations_enabled": True})
    monkeypatch.setattr(product, "load_artemis_backend", lambda **kw: binding)
    monkeypatch.setenv("SLURM_JOB_GPUS", "0,1,2,3")
    controller, execution, provenance = runner._compose(
        config=value, config_path=path, inputs=inputs, run_root=tmp_path,
        codex_binary=tmp_path / "never-called-codex", auth_file=tmp_path / "never-read-auth",
        worker_gpus=("0", "1"), controller_gpu="2", events=EventStream())
    _assert_devices(controller.candidate_execution, controller.agents, gpu)
    expected_budget = 14400 if gpu else 7200
    assert all(plan.agent_budget_seconds == expected_budget for plan in controller.campaign.rounds)
    assert execution._wall_budget(expected_budget) == expected_budget
    assert provenance["agent_protocol"]["effective_hard_wall_seconds"] == expected_budget
    assert provenance["candidate_execution_policy"]["solve_budget_seconds"] == (None if gpu else 600)
    assert controller.evaluation.solve_budget == (None if gpu else 600)
    if gpu:
        assert controller.evaluation.candidate_budget_mode == "ns_remaining_case_v1"
    assert not source.requests and not execution.audits


@pytest.mark.parametrize("gpu", [False, True])
def test_strong_real_service_and_program_constructors(runners, tmp_path, monkeypatch, gpu):
    from pde_agent_verifier_solver.plugins.vpfp import iteration_backend
    from pde_agent_verifier_solver.plugins.vpfp.tasks.observable import VPFPObservableTask
    runner = runners["strong"]
    value, path = configuration("strong", gpu=gpu)
    binding = SimpleNamespace(plugin=SimpleNamespace(task=lambda _: VPFPObservableTask()),
        controller=lambda recipe, programs, **kw: SimpleNamespace(recipe=recipe, programs=programs, **kw))
    monkeypatch.setattr(iteration_backend, "build_vpfp_strong_iteration_binding", lambda **kw: binding)
    monkeypatch.setenv("SLURM_JOB_GPUS", "0,1,2,3")
    controller, agents, candidates, _ = runner.compose(
        value, path, tmp_path, events=EventStream(), codex_binary=tmp_path / "never-called",
        auth_file=tmp_path / "never-read", worker_gpus=("0", "1"), controller_gpu="2", candidate_workers=4)
    _assert_devices(candidates, controller.programs, gpu)
    assert controller.recipe.default_round.agent_budget_seconds == 14400
    assert not agents.audits and not candidates.audits


@pytest.mark.parametrize("gpu", [False, True])
def test_ns_real_service_and_program_constructors_without_science_assets(runners, tmp_path, monkeypatch, gpu):
    runner = runners["ns"]
    value, path = configuration("ns", gpu=gpu)
    generator = SimpleNamespace(capabilities=GeneratorCapabilities(
        trainable=False, offline=False, online=True, guidance=True, perturbations=True))
    recipe = GeneratorRecipe(checkpoint=GeneratorCheckpoint("synthetic", {}),
                             guidance_enabled=True, perturbations_enabled=True)
    monkeypatch.setattr(runner, "_compose_numerical", lambda *a: (
        object(), object(), generator, recipe, {}, {"synthetic_fixture": True}))
    monkeypatch.setenv("SLURM_JOB_GPUS", "0,1,2,3")
    composition = runner._compose(
        config=value, config_path=path, run_root=tmp_path,
        codex_binary=tmp_path / "never-called", auth_file=tmp_path / "never-read",
        worker_gpus=("0", "1"), controller_gpu="2")
    _assert_devices(composition.candidate_execution, composition.agents, gpu)
    assert all(plan.agent_budget_seconds == 14400 for plan in composition.campaign.rounds)
    assert not composition.execution.audits and not composition.candidate_execution.audits


def test_ns_agent_owned_routes_bind_nonzero_allocation_without_mutating_config(runners, tmp_path, monkeypatch):
    runner = runners['ns']
    value, path = configuration('ns', gpu=True)
    value['execution']['development_solver_placement'] = 'agent_gpu'
    original = deepcopy(value)
    generator = SimpleNamespace(capabilities=GeneratorCapabilities(
        trainable=False, offline=False, online=True, guidance=True, perturbations=True))
    recipe = GeneratorRecipe(checkpoint=GeneratorCheckpoint('synthetic', {}),
                             guidance_enabled=True, perturbations_enabled=True)
    monkeypatch.setattr(runner, '_compose_numerical', lambda *a: (
        object(), object(), generator, recipe, {}, {'synthetic_fixture': True}))
    monkeypatch.setenv('SLURM_JOB_GPUS', '17,19,23,29')
    composition = runner._compose(config=value, config_path=path, run_root=tmp_path,
        codex_binary=tmp_path/'never-called', auth_file=tmp_path/'never-read',
        worker_gpus=('17','19'), controller_gpu='23', candidate_gpu_tokens=('29',))
    try:
        service = composition.candidate_execution
        assert service.gpu_devices == ('29',)
        expected = {row['agent_id']:(device,) for row,device in zip(value['agents'],('17','19'))}
        assert service.development_gpu_devices == expected
        for agent, devices in expected.items():
            assert service.for_development(agent).gpu_devices == devices
            assert composition.agents[agent].gpu_devices == devices
        assert value == original
        assert not composition.execution.audits and not service.audits
    finally:
        composition.candidate_execution.close()
