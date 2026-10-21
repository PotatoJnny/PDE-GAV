"""Real canonical-worker startup in the baseline CLI, with a local stub Codex."""
from __future__ import annotations

import json
from pathlib import Path
import stat
import sys

import pytest

from pde_agent_verifier_solver.engine.agent_execution import (
    AgentExecutionError, SubprocessAgentExecutionService, UnavailableAgentCapabilityFacade,
)
from pde_agent_verifier_solver.plugins.ns.baselines.cli import run_from_config
from tests.test_canonical_codex_agent import agent_context, command_program
from tests.test_ns_baselines import SOURCE, _write_numerical_data

ROOT = Path(__file__).resolve().parents[1]


def _stub_codex(root, *, nomination_count=1):
    executable = root / "stub-codex.py"
    source = SOURCE.replace("FACTOR", "1.0")
    executable.write_text(f'''#!{sys.executable}
import json
import os
from pathlib import Path
import sys
import time
from pde_agent_verifier_solver.engine.agent_execution import submit_agent_nomination, write_agent_round_log
from pde_agent_verifier_solver.engine.agent_capability import ControllerAgentCapabilityClient

print(json.dumps({{"type": "thread.started"}}), flush=True)
sys.stdin.read()
codex_home = Path(os.environ["CODEX_HOME"])
for _ in range(2000):
    if not (codex_home / "auth.json").exists() and (codex_home / "auth-lifecycle.json").is_file():
        break
    time.sleep(.001)
assert not (codex_home / "auth.json").exists()
request = json.loads(Path(os.environ["PDE_AGENT_REQUEST"]).read_text())
cases = ControllerAgentCapabilityClient.from_request(request).current_cases()
assert len(cases) == 1
for sequence in range(1, {nomination_count} + 1):
    name = f"submission-{{sequence}}.py"
    Path(name).write_text({source!r} if sequence == {nomination_count} else {SOURCE.replace("FACTOR", "0.5")!r})
    submit_agent_nomination({{"bundle_path": name}}, sequence=sequence)
write_agent_round_log("# Local stub Codex\\nCommitted one valid NS solver without a model call.")
Path(sys.argv[sys.argv.index("--output-last-message") + 1]).write_text("Local stub completed.\\n")
Path({str(root)!r}, "entered-" + request["execution_id"] + ".json").write_text(json.dumps({{
    "execution_id": request["execution_id"], "case_count": len(cases),
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "auth_removed": not (codex_home / "auth.json").exists(),
}}))
''')
    executable.chmod(0o700)
    auth = root / "dummy-auth.json"
    auth.write_text('{"test_only": true}\n')
    auth.chmod(0o600)
    return executable, auth


def test_actual_canonical_worker_reproduces_missing_runtime_root_error(tmp_path):
    codex, auth = _stub_codex(tmp_path)
    service = SubprocessAgentExecutionService(tmp_path / "turns", hard_wall_seconds=10,
        capability_facade=UnavailableAgentCapabilityFacade())
    with pytest.raises(AgentExecutionError, match="runtime root must be an existing directory"):
        service.execute(command_program(codex, auth, tmp_path / "missing-runtime",
                                        gpu_devices=("0",)), agent_context(), 5)
    audit = service.audits[-1]
    assert audit.status == "worker_failed"
    assert "runtime root must be an existing directory" in audit.error
    assert audit.nomination_count == 0
    assert not list(tmp_path.glob("entered-*.json"))
    response = json.loads(Path(audit.response_path).read_text())
    assert response["status"] == "failed"


@pytest.mark.parametrize("borrow", [False, True])
@pytest.mark.parametrize("nomination_policy", ["all", "latest"])
def test_baseline_cli_creates_private_runtime_and_runs_two_rounds_of_real_canonical_workers(tmp_path, monkeypatch, borrow, nomination_policy):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    dataset = _write_numerical_data(tmp_path / "data")
    nomination_count = 2 if nomination_policy == "latest" else 1
    codex, auth = _stub_codex(tmp_path, nomination_count=nomination_count)
    initial = tmp_path / "initial.py"
    initial.write_text(SOURCE.replace("FACTOR", "0.0"))
    config = tmp_path / "baseline.json"
    config.write_text(json.dumps({
        "schema": "ns_fixed_data_baseline_v1", "method": "competitive_refinement", "seed": 101,
        "dataset": {"manifest": str(dataset), "development_split": "train", "selection_split": "validation"},
        "initial_candidate": str(initial), "search": {"competition_rounds": 2},
        "agent": {
            "command": ["{python}", "{repo}/scripts/run_canonical_codex_agent.py",
                        "--codex-binary", str(codex), "--auth-file", str(auth),
                        "--runtime-root", "{output}/agent_runtime",
                        "--config", "{repo}/scripts/canonical_codex.config.toml"],
            "hard_wall_seconds": 30,
            "nomination_policy": nomination_policy,
            "roles": [
                {"agent_id": "explorer", "name": "Explorer", "prompt": "Write the local solver.", "gpu_devices": ["0"]},
                {"agent_id": "refiner", "name": "Refiner", "prompt": "Write the local solver.", "gpu_devices": ["1"]},
            ],
        },
        "evaluation": {"development_point_count": 3, "selection_point_count": 4,
                       "candidate_case_hard_wall_seconds": 20, "borrow_agent_gpus": borrow},
    }))
    output = tmp_path / "result"
    result = run_from_config(config, output)
    assert result.report["completed_agent_turns"] == 4
    assert result.report["failed_turns"] == 0
    assert result.report["unique_scored_candidates"] == 2
    assert result.selected.artifact_id != "baseline-s0"
    assert result.selection_error == pytest.approx(0., abs=1e-14)
    runtime = output / "agent_runtime"
    assert runtime.is_dir() and not runtime.is_symlink()
    assert stat.S_IMODE(runtime.stat().st_mode) == 0o700
    assert list(runtime.iterdir()) == []
    audits = json.loads((output / "execution_audits.json").read_text())
    assert len(audits) == 4 and all(row["status"] == "completed" and row["nomination_count"] == 1 for row in audits)
    for row in audits:
        workspace = Path(row["workspace"])
        assert runtime not in workspace.parents and workspace not in runtime.parents
        assert json.loads(Path(row["response_path"]).read_text())["status"] == "completed"
        assert (workspace / "codex-auth-lifecycle.json").is_file()
        request = json.loads(Path(row["request_path"]).read_text())
        assert request["worker_settings"]["nomination_policy"] == nomination_policy
        assert request["worker_settings"]["transient_resume_until_deadline"] is True
        assert len(list((workspace / "nominations").glob("nomination-*.json"))) == nomination_count
        if nomination_policy == "latest":
            assert "Choose exactly one final Candidate" in (workspace / "CANONICAL_AGENT_PROMPT.md").read_text()
        assert request["candidate_execution"]["case_charged_wall_seconds"] == 20
        assert "20-second per-Case charged-wall limit" in (
            workspace / "CANONICAL_AGENT_PROMPT.md"
        ).read_text()
    entered = [json.loads(path.read_text()) for path in tmp_path.glob("entered-*.json")]
    assert len(entered) == 4 and {row["cuda_visible_devices"] for row in entered} == {"0", "1"}
    assert all(row["auth_removed"] for row in entered)
    assert auth.read_text() == '{"test_only": true}\n'

    inputs = json.loads((output / "baseline_input.json").read_text())
    assert inputs["evaluation_device_map"] == ({"0": "cuda:1", "1": "cuda:0"} if borrow else {})
    if borrow:
        events = [json.loads(line) for line in (output / "evaluation_device_events.jsonl").read_text().splitlines()]
        assert len(events) == 4 and all(e["cleanup_confirmed"] for e in events[1::2])


@pytest.mark.parametrize("agent", [
    {"nomination_policy": "best"},
    {"nomination_policy": "latest", "worker_settings": {"nomination_policy": "all"}},
])
def test_baseline_rejects_nomination_policy_before_loading_dataset(tmp_path, agent):
    config = tmp_path / "invalid.json"
    config.write_text(json.dumps({"schema": "ns_fixed_data_baseline_v1",
        "method": "competitive_refinement", "agent": agent,
        "dataset": {"manifest": "missing-on-purpose.json"}}))
    with pytest.raises(ValueError, match="nomination_policy"):
        run_from_config(config, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("agent", [
    {"transient_resume_max_attempts": value} for value in (None, True, 0, -1, 4, 1.5, "3")
] + [{"transient_resume_max_attempts": 3,
      "worker_settings": {"transient_resume_max_attempts": 1}}])
def test_baseline_rejects_invalid_resume_bound_before_loading_dataset(tmp_path, agent):
    config = tmp_path / "invalid-resume.json"
    config.write_text(json.dumps({"schema": "ns_fixed_data_baseline_v1",
        "method": "competitive_refinement", "agent": agent,
        "dataset": {"manifest": "missing-on-purpose.json"}}))
    with pytest.raises(ValueError, match="transient_resume_max_attempts"):
        run_from_config(config, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_fixed_entry_resumes_each_agent_with_original_budget_and_final_submission(tmp_path, monkeypatch):
    from tests.test_canonical_codex_resume import _write_resumable_codex
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    dataset = _write_numerical_data(tmp_path / "data")
    native = _write_resumable_codex(tmp_path)
    auth = tmp_path / "local-auth.json"
    auth.write_text('{"test_only": true}\n')
    auth.chmod(0o600)
    initial = tmp_path / "initial-resume.py"
    initial.write_text(SOURCE.replace("FACTOR", "0.0"))
    config = tmp_path / "fixed-resume.json"
    config.write_text(json.dumps({
        "schema": "ns_fixed_data_baseline_v1", "method": "competitive_refinement", "seed": 101,
        "dataset": {"manifest": str(dataset), "development_split": "train", "selection_split": "validation"},
        "initial_candidate": str(initial),
        "search": {"competition_rounds": 1},
        "agent": {
            "command": ["{python}", "{repo}/scripts/run_canonical_codex_agent.py",
                        "--codex-binary", str(native), "--auth-file", str(auth),
                        "--runtime-root", "{output}/agent_runtime",
                        "--config", "{repo}/scripts/canonical_codex.config.toml"],
            "hard_wall_seconds": 30,
            "nomination_policy": "latest", "transient_resume_max_attempts": 3,
            "roles": [
                {"agent_id": "explorer", "name": "Explorer", "prompt": "Develop the local solver.", "gpu_devices": ["0"]},
                {"agent_id": "refiner", "name": "Refiner", "prompt": "Improve the local solver.", "gpu_devices": ["1"]},
            ],
        },
        "evaluation": {"development_point_count": 3, "selection_point_count": 4,
                       "candidate_case_hard_wall_seconds": 20},
    }))
    output = tmp_path / "result"
    result = run_from_config(config, output)
    assert result.report["completed_agent_turns"] == 2
    assert result.report["failed_turns"] == 0
    assert result.selection_error == pytest.approx(0.0, abs=1e-14)
    audits = json.loads((output / "execution_audits.json").read_text())
    assert len(audits) == 2
    for audit in audits:
        workspace = Path(audit["workspace"])
        request = json.loads(Path(audit["request_path"]).read_text())
        assert request["worker_settings"]["transient_resume_max_attempts"] == 3
        assert audit["hard_wall_seconds"] == 30
        assert audit["nomination_count"] == 1 and audit["nomination_journal_entry_count"] == 2
        assert audit["final_nomination_sequence"] == 2
        recovery = json.loads((workspace / "codex-recovery.json").read_text())
        assert recovery["status"] == "completed" and len(recovery["attempts"]) == 2
        assert recovery["attempts"][0]["controller_cleanup"]["confirmed"]
        assert recovery["attempts"][1]["remaining_seconds"] < recovery["attempts"][0]["remaining_seconds"]
        starts = [json.loads(line) for line in (workspace / "native-attempt-starts.jsonl").read_text().splitlines()]
        assert len({row["thread_id"] for row in starts}) == 1
        assert starts[0]["execution_id"] == starts[1]["execution_id"] == audit["execution_id"]
    assert list((output / "agent_runtime").iterdir()) == []
    assert auth.read_text() == '{"test_only": true}\n'


@pytest.mark.parametrize("agent", [
    {"transient_resume_until_deadline": value} for value in (None, 0, 1, "true")
] + [{"transient_resume_until_deadline": True,
      "worker_settings": {"transient_resume_until_deadline": False}}])
def test_baseline_rejects_invalid_deadline_recovery_before_loading_dataset(tmp_path, agent):
    config = tmp_path / "invalid-deadline-recovery.json"
    config.write_text(json.dumps({"schema": "ns_fixed_data_baseline_v1",
        "method": "competitive_refinement", "agent": agent,
        "dataset": {"manifest": "missing-on-purpose.json"}}))
    with pytest.raises(ValueError, match="transient_resume_until_deadline"):
        run_from_config(config, tmp_path / "output")
    assert not (tmp_path / "output").exists()
