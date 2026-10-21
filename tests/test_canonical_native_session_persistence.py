"""Native child-session storage without extra attempts, models, auth, or GPUs."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from pde_agent_verifier_solver.engine.agent_execution import (
    SubprocessAgentExecutionService, UnavailableAgentCapabilityFacade,
)
from tests.test_canonical_codex_agent import WORKER, agent_context, command_program, write_auth
from tests.test_canonical_codex_resume import _write_resumable_codex


def test_private_session_setting_is_explicit_and_boolean():
    assert WORKER._native_session_persistence({}) is False
    assert WORKER._native_session_persistence({"worker_settings": {
        "native_session_persistence": True}}) is True
    for value in (None, 0, 1, "true", {}):
        with pytest.raises(ValueError, match="must be a boolean"):
            WORKER._native_session_persistence({"worker_settings": {
                "native_session_persistence": value}})


@pytest.mark.parametrize("failures,expected", [((), "completed"),
                                              (("stream disconnected",), "attempt_limit")])
def test_one_attempt_session_preserves_deadline_cleanup_and_auth(tmp_path, failures, expected):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    native = _write_resumable_codex(tmp_path, failures=failures)
    auth, original_auth = write_auth(tmp_path)
    service = SubprocessAgentExecutionService(
        tmp_path / "turns", hard_wall_seconds=20,
        capability_facade=UnavailableAgentCapabilityFacade(), poll_interval_seconds=.01,
        termination_grace_seconds=.5)
    program = replace(command_program(native, auth, runtime, gpu_devices=("0",)),
        worker_settings={"nomination_policy": "latest", "transient_resume_max_attempts": 1,
                         "native_session_persistence": True, "transient_resume_until_deadline": False})
    outcome = service.execute(program, agent_context(), 20)
    audit = service.audits[-1]
    workspace = Path(audit.workspace)
    recovery = json.loads((workspace / "codex-recovery.json").read_text())
    starts = [json.loads(line) for line in
              (workspace / "native-attempt-starts.jsonl").read_text().splitlines()]
    assert len(starts) == 1 and starts[0]["resumed"] is False
    assert recovery["native_session_persistence"] is True
    assert recovery["max_attempts"] == 1 and recovery["status"] == expected
    assert recovery["deadline_monotonic"] > 0
    assert recovery["attempts"][0]["remaining_seconds"] < 20
    assert recovery["attempts"][0]["backoff_seconds"] == 0
    assert recovery["attempts"][0]["process_group_cleaned"] is True
    assert "--ephemeral" not in starts[0]["argv"]
    sqlite_option = next(item for item in starts[0]["argv"] if item.startswith("sqlite_home="))
    sqlite_path = Path(json.loads(sqlite_option.split("=", 1)[1]))
    assert sqlite_path.is_relative_to(runtime) and not sqlite_path.is_relative_to(workspace)
    assert not list(runtime.iterdir())
    assert auth.read_bytes() == original_auth.encode()
    assert outcome.execution_status == ("completed" if expected == "completed" else "failed")


def test_private_persistence_requires_original_deadline(tmp_path, monkeypatch):
    monkeypatch.delenv("PDE_AGENT_DEADLINE_MONOTONIC", raising=False)
    with pytest.raises(ValueError, match="persistent native sessions require"):
        WORKER._run_codex(workspace=tmp_path, prompt="no model invocation",
            codex_binary=tmp_path / "unused", auth_source=tmp_path / "unused-auth",
            config_source=tmp_path / "unused-config", runtime_root=tmp_path,
            final_message=tmp_path / "unused-final", auth_audit_copy=tmp_path / "unused-audit",
            native_session_persistence=True)
    assert list(tmp_path.iterdir()) == []
