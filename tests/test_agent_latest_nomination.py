"""One explicitly chosen final nomination, without limiting development attempts."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

from pde_agent_verifier_solver.engine.agent_execution import (
    AGENT_NOMINATION_SCHEMA,
    AgentExecutionError,
    CommandAgentProgram,
    SubprocessAgentExecutionService,
    UnavailableAgentCapabilityFacade,
)
from tests.test_agent_execution_service import (
    _agent_context, _decode_toy_nomination, _worker_program,
)


@pytest.mark.parametrize("policy", [None, True, 1, [], {}, "best", "Latest"])
def test_policy_is_rejected_before_worker_start(policy):
    with pytest.raises(ValueError, match="nomination_policy"):
        CommandAgentProgram((sys.executable, "-c", "pass"), _decode_toy_nomination,
                            worker_settings={"nomination_policy": policy})


def _journal(directory, sequence, body):
    path = directory / f"nomination-{sequence:08d}.json"
    path.write_text(json.dumps({"schema": AGENT_NOMINATION_SCHEMA,
        "execution_id": "test-execution", "sequence": sequence, "payload": body}))
    return path


def _load(tmp_path, *, policy="latest", decoder=_decode_toy_nomination):
    service = SubprocessAgentExecutionService(tmp_path / "turns", hard_wall_seconds=2,
        capability_facade=UnavailableAgentCapabilityFacade())
    program = CommandAgentProgram((sys.executable, "-c", "pass"), decoder,
                                 worker_settings={"nomination_policy": policy})
    return service._load_nominations(tmp_path, "test-execution", program,
                                    _agent_context(), tmp_path)


def test_only_final_commitment_is_decoded_and_old_files_remain(tmp_path):
    previous = _journal(tmp_path, 1, {"not_mode": "superseded bundle is never decoded"})
    final = _journal(tmp_path, 2, {"mode": 2})
    original_bytes = [previous.read_bytes(), final.read_bytes()]
    calls = []
    def decode(body, context, workspace):
        calls.append(body)
        return _decode_toy_nomination(body, context, workspace)
    artifacts, rejected = _load(tmp_path, decoder=decode)
    assert len(artifacts) == 1 and rejected == 0
    assert calls == [{"mode": 2}]
    assert [previous.read_bytes(), final.read_bytes()] == original_bytes


@pytest.mark.parametrize("damage", ["invalid_bundle", "invalid_json", "foreign_execution",
                                    "gap", "invalid_schema", "mismatched_sequence", "non_object"])
def test_bad_final_commitment_never_falls_back_or_decodes_old_bundle(tmp_path, damage):
    _journal(tmp_path, 1, {"mode": 1})
    final = _journal(tmp_path, 3 if damage == "gap" else 2,
                     {"missing_mode": True} if damage == "invalid_bundle" else {"mode": 2})
    if damage == "invalid_json":
        final.write_text("{")
    elif damage in {"foreign_execution", "invalid_schema", "mismatched_sequence", "non_object"}:
        payload = json.loads(final.read_text())
        key, value = {"foreign_execution": ("execution_id", "other"),
            "invalid_schema": ("schema", "other"),
            "mismatched_sequence": ("sequence", 1), "non_object": ("payload", [])}[damage]
        payload[key] = value
        final.write_text(json.dumps(payload))
    calls = []
    def decode(body, context, workspace):
        calls.append(body)
        return _decode_toy_nomination(body, context, workspace)
    artifacts, rejected = _load(tmp_path, decoder=decode)
    assert artifacts == () and rejected == 1
    assert calls == ([{"missing_mode": True}] if damage == "invalid_bundle" else [])


def test_damaged_history_blocks_final_decode_without_side_effects(tmp_path):
    _journal(tmp_path, 1, {"mode": 1}).write_text("{")
    _journal(tmp_path, 2, {"mode": 2})
    def decode(*args):
        pytest.fail("an invalid journal must not call a decoder")
    assert _load(tmp_path, decoder=decode) == ((), 1)


def test_latest_empty_journal_is_zero_nomination(tmp_path):
    assert _load(tmp_path) == ((), 0)


@pytest.mark.parametrize("ending", ["completed", "failed", "timeout"])
def test_real_worker_returns_latest_and_preserves_committed_history(tmp_path, ending):
    script = tmp_path / "worker.py"
    script.write_text("\n".join([
        "import os, time",
        "from pathlib import Path",
        "from pde_agent_verifier_solver.engine.agent_execution import (",
        " load_agent_worker_request, submit_agent_nomination, write_agent_round_log, write_agent_worker_response)",
        "request = load_agent_worker_request()",
        "assert request['worker_settings']['nomination_policy'] == 'latest'",
        "write_agent_round_log('# Research log\\nThree attempted versions; chose version one.')",
        # Returning to an earlier version is an explicit final commitment, not a best-score fallback.
        "for sequence, mode in enumerate((1, 2, 1), 1):",
        "    submit_agent_nomination({'mode': mode}, sequence=sequence)",
        # An uncommitted partial file must not supersede the final atomic commit.
        "Path(os.environ['PDE_AGENT_NOMINATIONS'], '.nomination-00000004.partial').write_text('{')",
        "time.sleep(60)" if ending == "timeout" else
        "write_agent_worker_response(status='failed', error='local test failure')" if ending == "failed" else
        "write_agent_worker_response()",
    ]))
    calls = []
    def decode(body, context, workspace):
        calls.append(body)
        return _decode_toy_nomination(body, context, workspace)
    service = SubprocessAgentExecutionService(tmp_path / "turns", hard_wall_seconds=2,
        capability_facade=UnavailableAgentCapabilityFacade(),
        termination_grace_seconds=.1, poll_interval_seconds=.01)
    outcome = service.execute(replace(_worker_program(script,
        worker_settings={"nomination_policy": "latest"}), nomination_decoder=decode),
        _agent_context(), 2)
    assert len(outcome.nominations) == 1
    assert calls == [{"mode": 1}]
    assert outcome.execution_status == {"completed": "completed", "failed": "failed", "timeout": "timed_out"}[ending]
    assert "Three attempted versions" in outcome.round_log
    audit = service.audits[-1]
    assert audit.nomination_count == 1 and audit.rejected_nomination_count == 0
    assert audit.nomination_policy == "latest"
    assert audit.nomination_journal_entry_count == 3
    assert audit.final_nomination_sequence == 3
    saved = json.loads((Path(audit.workspace) / "audit.json").read_text())
    assert saved["nomination_count"] == 1 and saved["nomination_journal_entry_count"] == 3
    assert saved["final_nomination_sequence"] == 3
    assert audit.process_group_cleaned and audit.capability_drained
    assert len(list((Path(audit.workspace) / "nominations").glob("nomination-*.json"))) == 3


def test_real_worker_invalid_final_fails_turn_instead_of_preserving_previous(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text("\n".join([
        "from pde_agent_verifier_solver.engine.agent_execution import submit_agent_nomination, write_agent_worker_response",
        "submit_agent_nomination({'mode': 1}, sequence=1)",
        "submit_agent_nomination({'missing_mode': True}, sequence=2)",
        "write_agent_worker_response()",
    ]))
    service = SubprocessAgentExecutionService(tmp_path / "turns", hard_wall_seconds=3,
        capability_facade=UnavailableAgentCapabilityFacade())
    with pytest.raises(AgentExecutionError, match="invalid nomination"):
        service.execute(_worker_program(script, worker_settings={"nomination_policy": "latest"}),
                        _agent_context(), 2)
    assert service.audits[-1].nomination_count == 0
    assert service.audits[-1].rejected_nomination_count == 1
    assert service.audits[-1].nomination_journal_entry_count == 2
    assert service.audits[-1].final_nomination_sequence is None


def test_agent_can_explicitly_resubmit_an_earlier_frozen_version(tmp_path, monkeypatch):
    from pde_agent_verifier_solver.engine.agent_execution import (
        AGENT_EXECUTION_ID_ENV, AGENT_NOMINATIONS_ENV, AGENT_WORKSPACE_ENV,
    )
    from pde_agent_verifier_solver.engine.agent_tool import _commit_candidate_nomination
    from pde_agent_verifier_solver.sdk import inspect_candidate_bundle

    nominations = tmp_path / "nominations"
    nominations.mkdir()
    source = tmp_path / "solver.py"
    source.write_text("VALUE = 'earlier version'\n")
    for name, value in {
        AGENT_WORKSPACE_ENV: str(tmp_path), AGENT_NOMINATIONS_ENV: str(nominations),
        AGENT_EXECUTION_ID_ENV: "resubmit-earlier-version",
    }.items():
        monkeypatch.setenv(name, value)
    _, earlier = _commit_candidate_nomination("solver.py", sequence=1,
        name="Earlier", summary="first choice", usage="test bundle")
    original_tree = inspect_candidate_bundle(tmp_path / earlier).tree_sha256
    source.write_text("VALUE = 'later version'\n")
    _commit_candidate_nomination("solver.py", sequence=2,
        name="Later", summary="second choice", usage="test bundle")
    final_journal, final = _commit_candidate_nomination(earlier, sequence=3,
        name="Chosen earlier version", summary="explicit final choice", usage="test bundle")
    assert earlier != final
    assert inspect_candidate_bundle(tmp_path / final).tree_sha256 == original_tree
    assert json.loads(final_journal.read_text())["payload"]["bundle_path"] == final
    assert len(list(nominations.glob("nomination-*.json"))) == 3
