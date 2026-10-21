"""Explicit primary composes with the newer unified runtime, not a replacement."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from pde_agent_verifier_solver.engine.agent_execution import (
    SubprocessAgentExecutionService, UnavailableAgentCapabilityFacade,
)
from tests.test_canonical_codex_agent import agent_context, command_program, write_auth
from tests.test_canonical_codex_resume import _write_resumable_codex
from tests.test_single_primary_prompt import WORKER, request


@pytest.mark.parametrize("model,effort", [("gpt-5.6-sol", "max"), ("gpt-6-astra", "ultra")])
def test_primary_gpu_prompt_retains_unlimited_cases_and_requested_model(model, effort):
    configured = request()
    configured["round"]["development_sampling"]["max_cases"] = None
    configured["worker_settings"].update(
        model=model, reasoning_effort=effort, native_session_persistence=True,
        candidate_execution={"protocol": "shared_gpu_v1", "device": "cuda", "cuda_available": True},
    )
    prompt = WORKER._research_prompt(configured)
    assert WORKER._model_settings(configured) == (model, effort)
    assert WORKER._native_session_persistence(configured) is True
    assert "select-primary N" in prompt
    assert "no cumulative Development Case-count limit" in prompt
    assert "14400 seconds of total elapsed time" in prompt
    assert "900-second per-Case charged-wall" in prompt
    assert "No additional preparation" in prompt
    assert "candidate_bundle.json" in prompt


def test_primary_completion_prompt_does_not_claim_a_deadline():
    configured = request()
    configured["round"].update(time_budget_mode="completion", hard_wall_seconds=None)
    prompt = WORKER._research_prompt(configured)
    assert "There is no Agent development time cap" in prompt
    assert "before completing the stage" in prompt
    assert "normal deadline" not in prompt
    assert "select-primary N" in prompt


@pytest.mark.parametrize("model,effort", [("gpt-5.6-sol", "max"), ("gpt-6-astra", "ultra")])
def test_both_models_keep_native_session_and_original_deadline(tmp_path, model, effort):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    native = _write_resumable_codex(tmp_path, failures=())
    source = native.read_text()
    marker = "write_agent_round_log('# Same research turn"
    assert marker in source
    native.write_text(source.replace(marker,
        "from pde_agent_verifier_solver.engine.agent_execution import select_agent_primary_nomination\n"
        "select_agent_primary_nomination(1)\n" + marker, 1))
    auth, original_auth = write_auth(tmp_path)
    service = SubprocessAgentExecutionService(
        tmp_path / "turns", hard_wall_seconds=20,
        capability_facade=UnavailableAgentCapabilityFacade(),
        poll_interval_seconds=.01, termination_grace_seconds=.5,
    )
    program = replace(command_program(native, auth, runtime, gpu_devices=("0",)),
        worker_settings={
            "model": model, "reasoning_effort": effort,
            "nomination_policy": "single_primary_v1",
            "transient_resume_max_attempts": 1, "native_session_persistence": True,
        })
    # The same durable explicit selection is used by both native model routes.
    outcome = service.execute(program, agent_context(), 20)
    audit = service.audits[-1]
    workspace = Path(audit.workspace)
    starts = [json.loads(line) for line in
              (workspace / "native-attempt-starts.jsonl").read_text().splitlines()]
    recovery = json.loads((workspace / "codex-recovery.json").read_text())
    assert len(starts) == 1
    argv = starts[0]["argv"]
    assert argv[argv.index("--model") + 1] == model
    assert f'model_reasoning_effort="{effort}"' in argv
    assert "--ephemeral" not in argv
    assert recovery["native_session_persistence"] is True
    assert recovery["max_attempts"] == 1
    assert recovery["deadline_monotonic"] > 0
    assert recovery["attempts"][0]["remaining_seconds"] < 20
    assert recovery["attempts"][0]["process_group_cleaned"] is True
    assert audit.nomination_policy == "single_primary_v1"
    assert audit.nomination_count == 1
    assert audit.nomination_journal_entry_count == 1
    assert audit.final_nomination_sequence == 1
    assert outcome.execution_status == "completed"
    assert not list(runtime.iterdir())
    assert auth.read_bytes() == original_auth.encode()
