"""Shared GPU is an explicit resource protocol, not a silent default change."""
from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from pde_agent_verifier_solver.engine import agent_tool
from pde_agent_verifier_solver.engine.gpu_execution_config import (
    agent_gpu_worker_settings, candidate_execution_capabilities,
    candidate_gpu_kwargs, gpu_execution_settings, legacy_execution_settings,
    validate_gpu_run_budget,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def script_import_path(monkeypatch):
    # Direct importlib loads must emulate Python's real script-entrypoint path.
    # Keep this scoped to each test instead of mutating global import paths.
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))


def config():
    return {"execution": {"agent_hard_wall_seconds": 14400,
        "candidate_case_hard_wall_seconds": 900,
        "gpu_execution": {"protocol": "shared_gpu_v1", "candidate_devices": ["3"]}},
        "task": {"candidate_budget_mode": "ns_remaining_case_v1"}}


def test_cpu_defaults_do_not_add_execution_arguments_or_native_gpu_claims():
    assert gpu_execution_settings({}) is None
    assert candidate_gpu_kwargs({}, reserved_devices=(), environment={}) == {}
    assert candidate_execution_capabilities({}) == {"device": "cpu", "cuda_available": False,
        "solver_batch_size": 1, "gpu_slots_per_device": 1, "oracle_concurrency_per_device": 1}
    assert agent_gpu_worker_settings({}) == {}


@pytest.mark.parametrize("value", [None, {}, True,
    {"protocol": "gpu", "candidate_devices": ["3"]},
    {"protocol": "shared_gpu_v1", "candidate_devices": []},
    {"protocol": "shared_gpu_v1", "candidate_devices": "3"},
    {"protocol": "shared_gpu_v1", "candidate_devices": ["3", "3"]},
    {"protocol": "shared_gpu_v1", "candidate_devices": ["3,4"]},
    {"protocol": "shared_gpu_v1", "candidate_devices": [" 3"]},
    {"protocol": "shared_gpu_v1", "candidate_devices": [3]},
    {"protocol": "shared_gpu_v1", "candidate_devices": ["3"], "extra": True},
    {"protocol": "shared_gpu_v1", "candidate_devices": ["3"], "agent_sandbox_backend": "full-access"},
])
def test_malformed_or_implicit_gpu_profiles_are_rejected(value):
    with pytest.raises(ValueError):
        gpu_execution_settings({"gpu_execution": value})


def test_dedicated_allocated_devices_and_controller_reordering():
    execution = config()["execution"]
    options = candidate_gpu_kwargs(execution, reserved_devices=("0", "1", "2"),
        environment={"SLURM_JOB_GPUS": "0,1,2,3", "CUDA_VISIBLE_DEVICES": "2,0,1"})
    assert options == {"gpu_devices": ("3",)}
    assert "agent_cuda_qualified" not in candidate_execution_capabilities(execution)
    assert candidate_execution_capabilities(execution)["cuda_probe_required"] is True


@pytest.mark.parametrize("env,reserved", [
    ({}, ()), ({"CUDA_VISIBLE_DEVICES": "0,1,2"}, ()),
    ({"SLURM_JOB_GPUS": "0,1,2", "CUDA_VISIBLE_DEVICES": "3"}, ()),
    ({"CUDA_VISIBLE_DEVICES": "0,1,2,3"}, ("3",)),
    ({"CUDA_VISIBLE_DEVICES": "3,3"}, ()),
])
def test_unallocated_ambiguous_and_shared_device_assignments_fail(env, reserved):
    with pytest.raises(ValueError):
        candidate_gpu_kwargs(config()["execution"], reserved_devices=reserved, environment=env)


def test_native_gpu_backend_is_an_explicit_required_gate_not_a_passed_receipt():
    execution = config()["execution"]
    assert agent_gpu_worker_settings(execution) == {}
    execution["gpu_execution"]["agent_sandbox_backend"] = "legacy-landlock"
    assert agent_gpu_worker_settings(execution) == {"agent_gpu_execution": {
        "protocol": "shared_gpu_v1", "sandbox_backend": "legacy-landlock", "required": True}}
    assert legacy_execution_settings(execution) == {
        "agent_hard_wall_seconds": 14400, "candidate_case_hard_wall_seconds": 900}
    assert "gpu_execution" in execution  # No mutation of frozen source config.


def test_bwrap_runtime_pin_is_preserved_in_worker_request_without_config_mutation():
    execution = config()["execution"]
    execution["gpu_execution"].update(
        agent_sandbox_backend="bwrap-nvidia-v1", agent_runtime_sha256="ab" * 32)
    original = deepcopy(execution)
    assert gpu_execution_settings(execution) == original["gpu_execution"]
    settings = agent_gpu_worker_settings(execution)
    assert settings == {"agent_gpu_execution": {
        "protocol": "shared_gpu_v1", "sandbox_backend": "bwrap-nvidia-v1",
        "required": True, "runtime_sha256": "ab" * 32}}
    assert legacy_execution_settings(execution) == {
        "agent_hard_wall_seconds": 14400, "candidate_case_hard_wall_seconds": 900}
    value = request(True)
    value["worker_settings"].update(settings)
    assert "AGENT_GPU_READINESS.json" in worker()._research_prompt(value)
    assert agent_tool._status(value, ())["agent_gpu_execution"]["sandbox_backend"] == "bwrap-nvidia-v1"
    assert execution == original


def test_bwrap_does_not_require_a_runtime_digest():
    execution = config()["execution"]
    execution["gpu_execution"]["agent_sandbox_backend"] = "bwrap-nvidia-v1"
    assert gpu_execution_settings(execution)["agent_sandbox_backend"] == "bwrap-nvidia-v1"
    assert agent_gpu_worker_settings(execution)["agent_gpu_execution"] == {
        "protocol": "shared_gpu_v1", "sandbox_backend": "bwrap-nvidia-v1", "required": True}


@pytest.mark.parametrize("backend", [None, "legacy-landlock", "bwrap-nvidia-v1"])
def test_old_digest_metadata_does_not_control_backend_selection(backend):
    execution = config()["execution"]
    execution["gpu_execution"]["agent_runtime_sha256"] = "legacy-record-only"
    if backend is not None:
        execution["gpu_execution"]["agent_sandbox_backend"] = backend
    assert gpu_execution_settings(execution).get("agent_sandbox_backend") == backend


@pytest.mark.parametrize("change", ["active", "preparation"])
def test_gpu_profile_requires_shared_ns900_budget(change):
    value = config()
    if change == "case":
        value["execution"]["candidate_case_hard_wall_seconds"] = 180
    elif change == "agent":
        value["execution"]["agent_hard_wall_seconds"] = 7200
    elif change == "active":
        value["execution"]["agent_active_budget_seconds"] = 7200
    else:
        value["task"]["candidate_compute_budget"] = 180
    with pytest.raises(ValueError):
        validate_gpu_run_budget(value)
    value["execution"].pop("gpu_execution")
    validate_gpu_run_budget(value)  # Frozen CPU legacy variants stay accepted.


def worker():
    spec = importlib.util.spec_from_file_location("_shared_gpu_prompt_test", ROOT / "scripts/run_canonical_codex_agent.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def request(gpu=False):
    return {"round": {"round_index": 1, "active_budget_seconds": None,
        "hard_wall_seconds": 14400, "time_budget_mode": "hard_wall",
        "development_sampling": {"max_cases": 32, "allowed_sources": ["base", "guided"]}},
        "agent": {"prompt": "Develop a solver using the public equation."},
        "candidate_execution": {"case_charged_wall_seconds": 900.0},
        "worker_settings": {"candidate_execution": candidate_execution_capabilities(
            config()["execution"] if gpu else {})}}


def test_cpu_prompt_retains_original_device_and_budget_language():
    prompt = worker()._research_prompt(request())
    assert "entire Candidate worker" in prompt
    assert "Formal Candidate evaluations run in CPU-only subprocesses with CUDA\ndisabled." in prompt
    assert "submitted bundle must execute correctly without CUDA." in prompt
    assert "shared_gpu_v1" not in prompt


def test_gpu_prompt_describes_budget_assets_and_no_unverified_research_gpu():
    prompt = worker()._research_prompt(request(True))
    for text in ("shared_gpu_v1", "entire Candidate worker", "cuda:0", "900-second",
                 "14400 seconds", "No additional preparation", "candidate_bundle.json",
                 "trained weights", "relative to __file__", "does not prove",
                 "CPU, GPU, and combined implementations are welcome"):
        assert text in prompt
    assert "Formal Candidate evaluations run in CPU-only" not in prompt


def test_unversioned_cuda_prompt_is_rejected():
    value = request(True)
    value["worker_settings"]["candidate_execution"].pop("protocol")
    with pytest.raises(ValueError, match="versioned"):
        worker()._research_prompt(value)


def test_public_status_reports_configured_candidate_not_qualified_agent_cuda():
    result = agent_tool._status(request(True), ())
    assert result["candidate_execution"]["device"] == "cuda"
    assert result["candidate_execution"]["protocol"] == "shared_gpu_v1"
    assert result["candidate_execution"]["cuda_probe_required"] is True
    assert "agent_cuda_qualified" not in result["candidate_execution"]
    assert "agent_gpu_execution" not in result
    assert result["candidate_execution"]["case_charged_wall_seconds"] == 900.0


def test_native_gpu_prompt_and_status_require_receipt_not_config_as_proof():
    value = request(True)
    execution = config()["execution"]
    execution["gpu_execution"]["agent_sandbox_backend"] = "legacy-landlock"
    value["worker_settings"].update(agent_gpu_worker_settings(execution))
    prompt = worker()._research_prompt(value)
    assert "AGENT_GPU_READINESS.json" in prompt and "must report status=passed" in prompt
    assert "device token alone is not that evidence" in prompt
    status = agent_tool._status(value, ())
    assert status["agent_gpu_execution"] == {
        "protocol": "shared_gpu_v1", "qualification_required": True,
        "sandbox_backend": "legacy-landlock", "readiness_receipt": "AGENT_GPU_READINESS.json"}
    assert "passed" not in status["agent_gpu_execution"]


@pytest.mark.parametrize("script", ["run_canonical_ns_campaign.py",
    "run_vpfp_strong_formal_campaign.py",
    "run_canonical_artemis_campaign.py"])
def test_each_shared_composer_wires_same_explicit_gpu_service_options(script):
    tree = ast.parse((ROOT / "scripts" / script).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "SubprocessCandidateExecutionService"]
    assert len(calls) == 1
    assert any(keyword.arg is None and isinstance(keyword.value, ast.Name)
               and keyword.value.id == "gpu_options" for keyword in calls[0].keywords)
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "validate_gpu_run_budget" for node in ast.walk(tree))


@pytest.mark.parametrize("case_limit,agent_limit", [(180,7200),(300,3600),(900,14400)])
def test_gpu_resources_accept_independent_time_budgets(case_limit, agent_limit):
    value = config()
    value['execution'].update(candidate_case_hard_wall_seconds=case_limit, agent_hard_wall_seconds=agent_limit)
    validate_gpu_run_budget(value)


@pytest.mark.parametrize("value", [True,0,-1,float('nan'),float('inf'),'900'])
def test_gpu_budget_validation_rejects_invalid_numbers(value):
    configuration = config()
    configuration['execution']['candidate_case_hard_wall_seconds'] = value
    with pytest.raises(ValueError): validate_gpu_run_budget(configuration)
