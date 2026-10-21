"""Bounded same-session continuation with local stubs; no remote model calls."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import signal
import subprocess
import sys
import time

import pytest

from pde_agent_verifier_solver.engine.agent_execution import (
    SubprocessAgentExecutionService, UnavailableAgentCapabilityFacade,
)
from tests.test_canonical_codex_agent import WORKER, agent_context, command_program, write_auth
from tests.test_ns_baselines import SOURCE

ROOT = Path(__file__).resolve().parents[1]


# Capacity regressions reconciled from the migrated weak-runtime evidence.
def _supervisor():
    spec = importlib.util.spec_from_file_location("capacity_supervisor", ROOT / "scripts/run_codex_with_ephemeral_auth.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("kind", ["error", "turn.failed"])
@pytest.mark.parametrize("detail", [
    "Selected model is at capacity. Please try a different model.",
    {"message": "Selected model is at capacity. Please try a different model."},
    {"code": "model_capacity_exceeded", "message": "Please try again later."},
])
def test_capacity_error_events_are_retryable_without_changing_model(kind, detail):
    events = _supervisor()._NativeEvents()
    events.feed((json.dumps({"type": kind, "error": detail}) + "\n").encode())
    assert events.retryable is True and events.retry_blocked is False
    assert "capacity" in events.failure.lower()


@pytest.mark.parametrize("rejection", ["Unauthorized", "invalid_api_key", "insufficient_quota",
    "usage_limit_reached", "billing_hard_limit_reached", "content_policy_violation", "permission_denied"])
@pytest.mark.parametrize("order", ["before", "after", "combined"])
def test_capacity_never_overrides_auth_quota_or_policy_rejection(rejection, order):
    events = _supervisor()._NativeEvents()
    capacity = {"type": "error", "message": "Selected model is at capacity. Please try a different model."}
    denied = {"type": "turn.failed", "error": {"code": rejection, "message": rejection}}
    rows = ([denied, capacity] if order == "before" else [capacity, denied])
    if order == "combined":
        rows = [{"type": "turn.failed", "error": {"code": rejection, "message": capacity["message"]}}]
    for row in rows:
        events.feed((json.dumps(row) + "\n").encode())
    events.feed(b'{"type":"turn.failed","error":{"message":"stream disconnected"}}\n')
    assert events.retryable is False and events.retry_blocked is True


@pytest.mark.parametrize("event", [
    {"type": "error", "code": "invalid_api_key", "message": "Selected model is at capacity."},
    {"type": "turn.failed", "error": {"message": "You have exceeded your current quota, please check your plan."}},
])
def test_capacity_is_blocked_by_top_level_auth_code_or_quota_message(event):
    events = _supervisor()._NativeEvents()
    events.feed(b'{"type":"error","message":"Selected model is at capacity."}\n')
    events.feed((json.dumps(event) + "\n").encode())
    assert events.retryable is False and events.retry_blocked is True


def _run_capacity_worker(tmp_path, monkeypatch, scenario="success", *, max_attempts=3, until_deadline=False, failures=2):
    """Exercise the real worker state machine with fake native processes/time."""
    workspace, runtime = tmp_path / "workspace", tmp_path / "runtime"
    workspace.mkdir(); runtime.mkdir()
    auth, original_auth = write_auth(tmp_path)
    config = tmp_path / "config.toml"
    config.write_text('approval_policy = "never"\n')
    final = workspace / "codex-final-message.md"
    audit_path = workspace / "codex-auth-lifecycle.json"
    ledger = workspace / "case-budget.json"
    ledger.write_text(json.dumps({"remaining": 7, "case_ids": ["already-issued-case"]}))
    ledger_bytes = ledger.read_bytes()
    clock, sleeps, calls, settlements = [100.0], [], [], []
    deadline = 120.0 if scenario == "deadline" else 500.0
    thread_id = "11111111-1111-4111-8111-111111111111"
    monkeypatch.setenv("PDE_AGENT_DEADLINE_MONOTONIC", str(deadline))
    def sleep(seconds):
        assert clock[0] + seconds <= deadline
        sleeps.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(WORKER, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep, time_ns=lambda: int(clock[0] * 1e9)))
    monkeypatch.setattr(WORKER, "_codex_environment", lambda **kw: {
        "CODEX_HOME": str(kw["codex_home"]), "HOME": str(kw["agent_home"])})
    def settle(path, original_deadline):
        assert path == workspace and original_deadline == deadline
        assert ledger.read_bytes() == ledger_bytes
        settlements.append(original_deadline)
        return {"confirmed": scenario != "cleanup", "reason": "synthetic_cpu_check"}
    monkeypatch.setattr(WORKER, "_settle_capabilities", settle)
    def native(command, *, input, cwd, env, check):
        attempt = len(calls) + 1
        progress = json.loads((workspace / "codex-recovery.json").read_text())
        assert progress["status"] == "running" and progress["current_attempt"] == attempt
        assert progress["thread_id"] == (None if attempt == 1 else thread_id)
        assert original_auth not in json.dumps(progress)
        home = Path(env["CODEX_HOME"])
        assert cwd == workspace and check is False
        assert (home / "auth.json").read_bytes() == original_auth.encode()
        assert command[command.index("--model") + 1] == "gpt-5.6-sol"
        assert command[command.index("--deadline-monotonic") + 1] == str(deadline)
        assert "--ephemeral" not in command and "--require-descendant-cleanup" in command
        assert (home / "sessions").stat().st_mode & 0o777 == 0o700
        assert (home.parent / "sqlite").stat().st_mode & 0o777 == 0o700
        if attempt > 1:
            assert home == calls[0]["home"]
            assert command[command.index("resume") + 1] == thread_id
            assert command[command.index("--expected-thread-id") + 1] == thread_id
            assert input.decode() == WORKER._RESUME_PROMPT
        if scenario != "missing_session":
            session = home / "sessions" / (thread_id + ".jsonl")
            session.write_text(json.dumps({"type": "session_meta", "payload": {"id": thread_id}}) + "\n")
        events = _supervisor()._NativeEvents()
        events.feed((json.dumps({"type": "thread.started", "thread_id": thread_id}) + "\n").encode())
        failed = scenario != "success" or attempt <= failures
        if failed:
            events.feed(b'{"type":"turn.failed","error":{"message":"Selected model is at capacity. Please try a different model."}}\n')
        else:
            events.feed(b'{"type":"turn.completed"}\n')
            final.write_text("Same-model same-session completion.\n")
        (home / "auth.json").unlink()
        audit_path.write_text(json.dumps({"auth_removed_before_first_event_forwarded": True}))
        home.chmod(0o500)
        Path(command[command.index("--result") + 1]).write_text(json.dumps({
            "failure": events.failure, "thread_id": events.thread_id,
            "retryable": events.retryable, "process_group_cleaned": True}))
        calls.append({"home": home, "command": command})
        clock[0] += 1.0
        return SimpleNamespace(returncode=1 if failed else 0)
    monkeypatch.setattr(WORKER, "subprocess", SimpleNamespace(run=native))
    code, failure = WORKER._run_codex(workspace=workspace, prompt="Original task", codex_binary=tmp_path / "fake-codex",
        auth_source=auth, config_source=config, runtime_root=runtime, final_message=final,
        auth_audit_copy=audit_path, model="gpt-5.6-sol", reasoning_effort="max", resume_max_attempts=max_attempts,
        resume_until_deadline=until_deadline)
    recovery = json.loads((workspace / "codex-recovery.json").read_text())
    assert not list(runtime.iterdir())
    assert auth.read_bytes() == original_auth.encode()
    assert ledger.read_bytes() == ledger_bytes
    assert recovery["deadline_monotonic"] == deadline
    return code, failure, recovery, calls, sleeps, settlements


def test_capacity_same_session_backoff_preserves_model_deadline_and_case_budget(tmp_path, monkeypatch):
    code, failure, recovery, calls, sleeps, settlements = _run_capacity_worker(tmp_path, monkeypatch)
    assert code == 0 and failure is None and recovery["status"] == "completed"
    assert len(calls) == 3 and sleeps == [30.0, 60.0] and len(settlements) == 2
    assert [row["backoff_seconds"] for row in recovery["attempts"]] == [30.0, 60.0, 0.0]
    assert [row["remaining_seconds"] for row in recovery["attempts"]] == [399.0, 368.0, 307.0]
    assert len({row["thread_id"] for row in recovery["attempts"]}) == 1


@pytest.mark.parametrize("scenario,reason,count", [
    ("limit", "attempt_limit", 3), ("deadline", "deadline", 1),
    ("missing_session", "no_persisted_session", 1), ("cleanup", "controller_cleanup_unconfirmed", 1),
])
def test_capacity_retry_stops_at_attempt_deadline_session_and_cleanup_boundaries(tmp_path, monkeypatch, scenario, reason, count):
    code, failure, recovery, calls, sleeps, _ = _run_capacity_worker(tmp_path, monkeypatch, scenario)
    assert code == 1 and failure and recovery["status"] == reason
    assert len(calls) == count
    assert sleeps == ([30.0, 60.0] if scenario == "limit" else [])


def _write_resumable_codex(root, *, failures=("stream disconnected",), persist=True,
                           escaped_children=False, resumed_thread_id=None,
                           hang_after_attempt=None, fail_before_work=False):
    executable = root / "resumable-codex.py"
    executable.write_text(f'''#!{sys.executable}
import json, os, signal, subprocess, sys, time
from pathlib import Path
from uuid import UUID
from pde_agent_verifier_solver.engine.agent_execution import submit_agent_nomination, write_agent_round_log
request = json.loads(Path(os.environ['PDE_AGENT_REQUEST']).read_text())
workspace = Path(os.environ['PDE_AGENT_WORKSPACE'])
codex_home = Path(os.environ['CODEX_HOME'])
thread_id = str(UUID(request['execution_id']))
records = workspace / 'native-attempt-starts.jsonl'
previous = [json.loads(line) for line in records.read_text().splitlines()] if records.exists() else []
attempt = len(previous) + 1
resumed = 'resume' in sys.argv
if resumed:
    assert sys.argv[sys.argv.index('resume') + 1] == thread_id
    assert previous and previous[-1]['execution_id'] == request['execution_id']
    for pid in previous[-1]['escaped_pids']:
        try: os.kill(pid, 0)
        except ProcessLookupError: pass
        else: raise AssertionError('a previous escaped tool is still alive')
    assert len(list((workspace / 'nominations').glob('nomination-*.json'))) == (0 if {fail_before_work!r} else attempt - 1)
assert '--ephemeral' not in sys.argv
if {persist!r}:
    session = codex_home / 'sessions' / ('rollout-' + thread_id + '.jsonl')
    if not session.exists():
        session.write_text(json.dumps({{'type':'session_meta', 'payload':{{'id':thread_id}}}}) + '\\n')
reported_thread = {resumed_thread_id!r} if resumed and {resumed_thread_id!r} is not None else thread_id
print(json.dumps({{'type':'thread.started', 'thread_id':reported_thread}}), flush=True)
for _ in range(2000):
    if not (codex_home / 'auth.json').exists() and (codex_home / 'auth-lifecycle.json').is_file(): break
    time.sleep(.001)
assert not (codex_home / 'auth.json').exists()
assert codex_home.stat().st_mode & 0o777 == 0o500
prompt = sys.stdin.read()
assert 'PDE_AGENT_DEADLINE_MONOTONIC' not in os.environ
escaped_pids = []
if attempt == 1 and {escaped_children!r}:
    child_code = "import json,os,subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True); Path('escaped-pids.json').write_text(json.dumps([os.getpid(),p.pid]));time.sleep(60)"
    subprocess.Popen([sys.executable, '-c', child_code], start_new_session=True)
    for _ in range(2000):
        if (workspace / 'escaped-pids.json').exists(): break
        time.sleep(.001)
    escaped_pids = json.loads((workspace / 'escaped-pids.json').read_text())
with records.open('a') as handle:
    handle.write(json.dumps({{'attempt':attempt,'execution_id':request['execution_id'],
        'thread_id':thread_id,'resumed':resumed,'prompt':prompt,'argv':sys.argv[1:],
        'escaped_pids':escaped_pids,'native_pid':os.getpid(),
        'nomination_count_before':len(list((workspace/'nominations').glob('nomination-*.json')))}}) + '\\n')
if {fail_before_work!r} and attempt <= len({list(failures)!r}):
    print(json.dumps({{'type':'turn.failed','error':{{'message':{list(failures)!r}[attempt-1]}}}}), flush=True)
    raise SystemExit(1)
source = {SOURCE!r}.replace('FACTOR', str(attempt / 2))
filename = 'solver-' + str(attempt) + '.py'
(workspace / filename).write_text(source)
submit_agent_nomination({{'mode':attempt,'suffix':'resume-test','bundle_path':filename}}, sequence=(1 if {fail_before_work!r} else attempt))
write_agent_round_log('# Same research turn\\nCommitted submission ' + str(attempt))
if attempt == {hang_after_attempt!r}:
    time.sleep(60)
failures = {list(failures)!r}
if attempt <= len(failures):
    print(json.dumps({{'type':'turn.failed','error':{{'message':failures[attempt-1]}}}}), flush=True)
    raise SystemExit(1)
Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text('Recovered local turn.\\n')
print(json.dumps({{'type':'turn.completed','usage':{{}}}}), flush=True)
''')
    executable.chmod(0o700)
    return executable


def _run(tmp_path, *, max_attempts=3, wall=20, until_deadline=False, backoff_seconds=None, **stub_options):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    native = _write_resumable_codex(tmp_path, **stub_options)
    auth, original_auth = write_auth(tmp_path)
    service = SubprocessAgentExecutionService(tmp_path / "turns", hard_wall_seconds=wall,
        capability_facade=UnavailableAgentCapabilityFacade(), poll_interval_seconds=.01,
        termination_grace_seconds=.5)
    program = replace(command_program(native, auth, runtime, gpu_devices=("0",)),
        worker_settings={"nomination_policy": "latest", "transient_resume_max_attempts": max_attempts,
                         "transient_resume_until_deadline": until_deadline})
    if backoff_seconds is not None:
        entry = tmp_path / "fast-backoff-worker.py"
        entry.write_text("import sys\n" + f"sys.path.insert(0, {str(ROOT / 'scripts')!r})\n"
            + "import run_canonical_codex_agent as worker\n"
            + f"worker._resume_backoff_seconds = lambda *args: {backoff_seconds!r}\n"
            + "raise SystemExit(worker.main())\n")
        program = replace(program, command=tuple(str(entry) if value == str(ROOT / "scripts/run_canonical_codex_agent.py")
                                                else value for value in program.command))
    outcome = service.execute(program, agent_context(), wall)
    audit = service.audits[-1]
    workspace = Path(audit.workspace)
    recovery = json.loads((workspace / "codex-recovery.json").read_text())
    starts = [json.loads(line) for line in (workspace / "native-attempt-starts.jsonl").read_text().splitlines()]
    assert not list(runtime.iterdir())
    assert auth.read_bytes() == original_auth.encode()
    return outcome, audit, recovery, starts


def test_complete_managed_worker_resumes_same_session_and_kills_escaped_tool_tree(tmp_path):
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        outcome, audit, recovery, starts = _run(tmp_path, escaped_children=True)
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)
    assert outcome.execution_status == "completed"
    assert len(outcome.nominations) == 1 and outcome.nominations[0].metadata["max_resolved_mode"] == 2
    assert audit.nomination_count == 1 and audit.nomination_journal_entry_count == 2
    assert audit.final_nomination_sequence == 2
    assert recovery["status"] == "completed" and len(starts) == 2
    assert starts[0]["execution_id"] == starts[1]["execution_id"] == audit.execution_id
    assert starts[0]["thread_id"] == starts[1]["thread_id"]
    assert starts[0]["resumed"] is False and starts[1]["resumed"] is True
    assert starts[1]["nomination_count_before"] == 1
    assert starts[1]["prompt"] == WORKER._RESUME_PROMPT
    assert starts[0]["prompt"] != starts[1]["prompt"]
    assert recovery["attempts"][0]["backoff_seconds"] == 5
    assert all(row["process_group_cleaned"] for row in recovery["attempts"])
    assert recovery["attempts"][1]["remaining_seconds"] < recovery["attempts"][0]["remaining_seconds"]
    workspace = Path(audit.workspace)
    assert (workspace / "solver-1.py").is_file() and (workspace / "solver-2.py").is_file()
    assert (workspace / "attempt-01-codex-auth-lifecycle.json").is_file()


@pytest.mark.parametrize("failure", ["Invalid prompt: potentially violating our usage policy",
    "Invalid request", "Unauthorized", "unknown native failure", "insufficient_quota"])
def test_explicit_rejections_and_unknown_failure_are_never_replayed(tmp_path, failure):
    outcome, audit, recovery, starts = _run(tmp_path, failures=(failure,))
    assert outcome.execution_status == "failed" and len(starts) == 1
    assert recovery["status"] == "not_retryable"
    assert len(outcome.nominations) == 1 and audit.final_nomination_sequence == 1


@pytest.mark.parametrize("options,reason", [
    ({"persist": False}, "no_persisted_session"),
    ({"wall": 3}, "deadline"),
    ({"max_attempts": 2, "failures": ("stream disconnected", "server_error")}, "attempt_limit"),
])
def test_missing_session_budget_and_attempt_limit_stop_without_fresh_exec(tmp_path, options, reason):
    outcome, audit, recovery, starts = _run(tmp_path, **options)
    assert outcome.execution_status == "failed"
    assert recovery["status"] == reason
    assert len(starts) == (2 if reason == "attempt_limit" else 1)
    assert audit.hard_wall_seconds == options.get("wall", 20)


@pytest.mark.parametrize("kind", ["error", "turn.failed"])
def test_unknown_native_error_blocks_later_transport_error(kind):
    spec = importlib.util.spec_from_file_location("resume_supervisor", ROOT / "scripts/run_codex_with_ephemeral_auth.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for first in ("unrecognized failure", "Invalid prompt", "Unauthorized", "content_policy_violation"):
        events = module._NativeEvents()
        events.feed((json.dumps({"type":kind, "message":first}) + "\n").encode())
        events.feed(b'{"type":"turn.failed","error":{"message":"stream disconnected"}}\n')
        assert events.retryable is False and events.retry_blocked is True


def test_controller_cleanup_waits_for_cancelled_evaluation_before_retry(tmp_path, monkeypatch):
    from pde_agent_verifier_solver.engine.agent_capability import ControllerAgentCapabilityClient
    monkeypatch.setattr(WORKER, "load_agent_worker_request", lambda: {"capabilities":{"evaluation":{"available":True}}})
    rows = [
        [{"request_id":"eval-1", "finished":False, "cleanup_confirmed":False}],
        [{"request_id":"eval-1", "finished":True, "cleanup_confirmed":True}],
    ]
    cancelled = []
    class Client:
        def current_status(self): return (), rows.pop(0)
        def cancel_evaluation(self, request_id, *, wait_seconds): cancelled.append((request_id, wait_seconds))
    monkeypatch.setattr(ControllerAgentCapabilityClient, "from_request", lambda *a, **kw: Client())
    result = WORKER._settle_capabilities(tmp_path, time.monotonic()+2)
    assert result["confirmed"] is True and cancelled == [("eval-1", 0.0)]


def test_failed_controller_cleanup_prevents_continuation(tmp_path, monkeypatch):
    from pde_agent_verifier_solver.engine.agent_capability import ControllerAgentCapabilityClient
    monkeypatch.setattr(WORKER, "load_agent_worker_request", lambda: {"capabilities":{"evaluation":{"available":True}}})
    class Client:
        def current_status(self): return (), [{"request_id":"eval-1", "finished":True, "cleanup_confirmed":False}]
    monkeypatch.setattr(ControllerAgentCapabilityClient, "from_request", lambda *a, **kw: Client())
    result = WORKER._settle_capabilities(tmp_path, time.monotonic()+2)
    assert result["confirmed"] is False and result["reason"] == "cleanup_failed"


@pytest.mark.parametrize("later_id", ["22222222-2222-4222-8222-222222222222", "invalid", None])
def test_session_identity_mismatch_survives_later_turn_completed(later_id):
    spec = importlib.util.spec_from_file_location("resume_identity_supervisor", ROOT / "scripts/run_codex_with_ephemeral_auth.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events = module._NativeEvents()
    for value in ("11111111-1111-4111-8111-111111111111", later_id):
        events.feed((json.dumps({"type":"thread.started", "thread_id":value}) + "\n").encode())
    events.feed(b'{"type":"turn.completed"}\n')
    events.finish()
    assert events.failure == "native resume session identity mismatch"
    assert events.retryable is False and events.retry_blocked is True


def test_resumed_native_wrong_session_is_stopped_before_scientific_work(tmp_path):
    outcome, audit, recovery, starts = _run(tmp_path,
        resumed_thread_id="22222222-2222-4222-8222-222222222222")
    assert outcome.execution_status == "failed" and len(outcome.nominations) == 1
    assert len(starts) == 1 and len(recovery["attempts"]) == 2
    assert "session identity mismatch" in recovery["attempts"][-1]["failure"]
    assert audit.final_nomination_sequence == 1


def test_resume_cannot_extend_original_managed_deadline(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    native = _write_resumable_codex(tmp_path, hang_after_attempt=2)
    auth, _ = write_auth(tmp_path)
    service = SubprocessAgentExecutionService(tmp_path / "turns", hard_wall_seconds=10,
        capability_facade=UnavailableAgentCapabilityFacade(), poll_interval_seconds=.01,
        termination_grace_seconds=2)
    program = replace(command_program(native, auth, runtime, gpu_devices=("0",)),
        worker_settings={"nomination_policy":"latest", "transient_resume_max_attempts":3})
    started = time.monotonic()
    outcome = service.execute(program, agent_context(), 10)
    assert time.monotonic() - started < 14
    assert outcome.execution_status == "timed_out"
    audit = service.audits[-1]
    starts = [json.loads(line) for line in (Path(audit.workspace) / "native-attempt-starts.jsonl").read_text().splitlines()]
    assert len(starts) == 2 and audit.hard_wall_seconds == 10
    assert len(outcome.nominations) == 1 and audit.final_nomination_sequence == 2
    for record in starts:
        with pytest.raises(ProcessLookupError):
            os.kill(record["native_pid"], 0)
    assert not list(runtime.rglob("auth.json"))


@pytest.mark.parametrize("max_attempts", [1, 3])
def test_recovery_continues_beyond_legacy_three_attempts_in_original_budget(tmp_path, monkeypatch, max_attempts):
    code, failure, recovery, calls, sleeps, settlements = _run_capacity_worker(
        tmp_path, monkeypatch, max_attempts=max_attempts, until_deadline=True, failures=5)
    assert code == 0 and failure is None and len(calls) == 6
    assert sleeps == [30, 60, 60, 60, 60]
    assert len(settlements) == 5
    assert recovery["deadline_monotonic"] == 500
    assert len({row["thread_id"] for row in recovery["attempts"]}) == 1
    assert all(row["remaining_seconds"] > 0 for row in recovery["attempts"])


def test_deadline_bounded_capacity_outage_stops_without_resetting_budget(tmp_path, monkeypatch):
    code, failure, recovery, calls, sleeps, _ = _run_capacity_worker(
        tmp_path, monkeypatch, scenario="limit", max_attempts=1, until_deadline=True)
    assert code == 1 and failure and recovery["status"] == "deadline"
    assert len(calls) > 3 and sum(sleeps) + len(calls) == 400
    assert 0 < recovery["attempts"][-1]["backoff_seconds"] <= 60


@pytest.mark.parametrize("detail", ["rate_limit_exceeded", "HTTP 429 Too Many Requests",
    "network error", "temporary failure in name resolution", "error sending request",
    "service unavailable", "HTTP 503", "stream closed before response completed"])
def test_temporary_provider_errors_use_redacted_retry_labels(detail):
    events = _supervisor()._NativeEvents()
    events.feed((json.dumps({"type":"turn.failed", "error":{"message":detail + " PRIVATE_SECRET"}}) + "\n").encode())
    assert events.retryable and not events.retry_blocked
    assert "PRIVATE_SECRET" not in events.failure


def test_retry_policy_defaults_preserve_completion_and_explicit_legacy_limits():
    assert WORKER._resume_attempt_limit({"round":{"time_budget_mode":"hard_wall"}}) == 1
    assert WORKER._resume_attempt_limit({"round":{"time_budget_mode":"completion"}}) == 1
    for value in (1, 2, 3):
        assert WORKER._resume_attempt_limit({"worker_settings":{"transient_resume_max_attempts":value}}) == value
    for value in (None, True, False, 0, -1, 4, 20, 1.5, "3"):
        with pytest.raises(ValueError, match="integer from 1 to 3"):
            WORKER._resume_attempt_limit({"worker_settings":{"transient_resume_max_attempts":value}})


@pytest.mark.parametrize("tool_kind", ["command_execution", "mcp_tool_call"])
def test_silence_watchdog_respects_active_native_tools(tool_kind):
    events = _supervisor()._NativeEvents()
    assert events.response_stalled(10, events.last_activity + 11)
    events.feed((json.dumps({"type":"item.started", "item":{"id":"tool", "type":tool_kind}}) + "\n").encode())
    assert not events.response_stalled(10, events.last_activity + 3600)
    events.feed((json.dumps({"type":"item.completed", "item":{"id":"tool", "type":tool_kind}}) + "\n").encode())
    assert not events.response_stalled(10, events.last_activity + 1)
    assert events.response_stalled(10, events.last_activity + 11)


def _run_native_fault(tmp_path, monkeypatch, fault, *, tool_kind="command_execution"):
    """Real launcher/supervisor processes; native stub needs no scientific imports."""
    workspace, runtime = tmp_path / "workspace", tmp_path / "runtime"
    workspace.mkdir(); runtime.mkdir()
    auth, original_auth = write_auth(tmp_path)
    config = tmp_path / "config.toml"
    config.write_text('approval_policy = "never"\n')
    native = tmp_path / "native.py"
    native.write_text(f'''#!{sys.executable}
import json, os, sys, time
from pathlib import Path
home = Path(os.environ['CODEX_HOME'])
thread_id = '11111111-1111-4111-8111-111111111111'
session = home / 'sessions' / (thread_id + '.jsonl')
if not session.exists():
    session.write_text(json.dumps({{'type':'session_meta','payload':{{'id':thread_id}}}})+'\\n')
resumed = 'resume' in sys.argv
if resumed:
    assert sys.argv[sys.argv.index('resume')+1] == thread_id
    assert Path('research.txt').read_text() == 'preserved research'
else:
    Path('research.txt').write_text('preserved research')
with Path('starts.jsonl').open('a') as stream:
    stream.write(json.dumps({{'pid':os.getpid(),'thread_id':thread_id,'resumed':resumed}})+'\\n')
print(json.dumps({{'type':'thread.started','thread_id':thread_id}}), flush=True)
sys.stdin.read()
for _ in range(500):
    if not (home / 'auth.json').exists(): break
    time.sleep(.001)
assert not (home / 'auth.json').exists()
fault={fault!r}
if fault in ('tool', 'tool_hang', 'tool_then_stall'):
    item={{'id':'tool-1','type':{tool_kind!r},'status':'in_progress'}}
    print(json.dumps({{'type':'item.started','item':item}}),flush=True)
    time.sleep(60 if fault=='tool_hang' else 3)
    item['status']='completed'
    print(json.dumps({{'type':'item.completed','item':item}}),flush=True)
if fault in ('silent','tool_then_stall') and not resumed:
    time.sleep(60)
if fault=='heartbeat':
    for i in range(30):
        print(json.dumps({{'type':'item.updated','item':{{'id':'thought','type':'reasoning','text':str(i)}}}}),flush=True)
        time.sleep(.1)
Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text('Finished original turn.\\n')
print(json.dumps({{'type':'turn.completed'}}),flush=True)
''')
    native.chmod(0o700)
    # Only remove expensive framework imports from the test actor. Production
    # deadline, auth supervisor, liveness parsing and retry loop run unchanged.
    monkeypatch.setattr(WORKER, "_codex_environment", lambda **kw: {
        **{key:os.environ[key] for key in ("PATH", "LD_LIBRARY_PATH") if key in os.environ},
        "CODEX_HOME":str(kw["codex_home"]), "HOME":str(kw["agent_home"])})
    monkeypatch.setattr(WORKER, "_settle_capabilities", lambda *args: {"confirmed":True})
    monkeypatch.setattr(WORKER, "_resume_backoff_seconds", lambda *args: .02)
    started = time.monotonic()
    deadline = started + (3 if fault == "tool_hang" else 15)
    monkeypatch.setenv("PDE_AGENT_DEADLINE_MONOTONIC", str(deadline))
    code, failure = WORKER._run_codex(workspace=workspace, prompt="Original research",
        codex_binary=native, auth_source=auth, config_source=config, runtime_root=runtime,
        final_message=workspace / "last.txt", auth_audit_copy=workspace / "audit.json",
        resume_max_attempts=1, resume_until_deadline=True, native_response_timeout_seconds=2)
    recovery = json.loads((workspace / "codex-recovery.json").read_text())
    starts = [json.loads(line) for line in (workspace / "starts.jsonl").read_text().splitlines()]
    assert auth.read_text() == original_auth and not list(runtime.iterdir())
    assert recovery["deadline_monotonic"] == deadline
    assert all(row["process_group_cleaned"] for row in recovery["attempts"])
    assert (workspace / "research.txt").read_text() == 'preserved research'
    for row in starts:
        with pytest.raises(ProcessLookupError): os.kill(row['pid'], 0)
    return code, failure, recovery, starts, time.monotonic() - started


@pytest.mark.parametrize("fault", ["silent", "tool_then_stall"])
def test_silent_model_after_work_recovers_same_session_with_original_deadline(tmp_path, monkeypatch, fault):
    code, failure, recovery, starts, elapsed = _run_native_fault(tmp_path, monkeypatch, fault)
    assert code == 0 and failure is None and len(starts) == 2
    assert recovery["attempts"][0]["failure"] == "native response stalled"
    assert starts[1]["resumed"] and len({row["thread_id"] for row in starts}) == 1
    assert recovery["attempts"][1]["remaining_seconds"] < recovery["attempts"][0]["remaining_seconds"]
    assert elapsed < 15


@pytest.mark.parametrize("tool_kind", ["command_execution", "mcp_tool_call"])
def test_long_silent_tool_completes_without_watchdog_retry(tmp_path, monkeypatch, tool_kind):
    code, failure, recovery, starts, _ = _run_native_fault(tmp_path, monkeypatch, "tool", tool_kind=tool_kind)
    assert code == 0 and failure is None and len(starts) == 1
    assert recovery["status"] == "completed"


def test_active_tool_still_obeys_original_hard_deadline(tmp_path, monkeypatch):
    code, failure, recovery, starts, elapsed = _run_native_fault(tmp_path, monkeypatch, "tool_hang")
    assert code != 0 and failure == "original Agent deadline reached"
    assert len(starts) == 1 and elapsed < 7
    assert recovery["status"] == "not_retryable"


def test_native_progress_resets_response_silence_clock(tmp_path, monkeypatch):
    code, failure, recovery, starts, _ = _run_native_fault(tmp_path, monkeypatch, "heartbeat")
    assert code == 0 and failure is None and len(starts) == 1
    assert recovery["status"] == "completed"


def test_silent_startup_fails_closed_and_removes_auth(tmp_path):
    runtime = tmp_path / "runtime"
    home = runtime / "codex-home"
    home.mkdir(parents=True)
    auth = home / "auth.json"
    auth.write_text("synthetic-credential")
    result = runtime / "result.json"
    started = time.monotonic()
    process = subprocess.run([sys.executable, str(ROOT / "scripts/run_codex_with_ephemeral_auth.py"),
        "--auth", str(auth), "--audit", str(home / "audit.json"), "--result", str(result),
        "--require-descendant-cleanup", "--deadline-monotonic", str(started + 10),
        "--response-timeout-seconds", "0.2", "--", sys.executable, "-c", "import time;time.sleep(60)"],
        capture_output=True, timeout=8)
    assert process.returncode != 0 and time.monotonic() - started < 5
    payload = json.loads(result.read_text())
    assert payload["thread_id"] is None and payload["retryable"] is False
    assert payload["process_group_cleaned"] and not auth.exists()
    assert "synthetic-credential" not in result.read_text()


@pytest.mark.parametrize("output_first", [False, True])
def test_oversized_tool_completion_releases_silence_watchdog(output_first):
    supervisor = _supervisor()
    events = supervisor._NativeEvents()
    events.feed(b'{"type":"item.started","item":{"id":"tool","type":"command_execution"}}\n')
    output = "untrusted output \"item.completed\" " * supervisor._EVENT_MAX_BYTES
    item = ({"aggregated_output":output,"id":"tool","type":"command_execution","status":"completed"}
            if output_first else {"id":"tool","type":"command_execution","status":"completed","aggregated_output":output})
    raw = (json.dumps({"type":"item.completed","item":item}) + "\n").encode()
    for start in range(0, len(raw), 8192):
        events.feed(raw[start:start + 8192])
        assert len(events.pending) <= supervisor._EVENT_MAX_BYTES
        if events.envelope is not None:
            assert len(events.envelope.token) <= 1024
    assert not events.active_tools
    assert events.response_stalled(900, events.last_activity + 901)


def test_oversized_in_progress_tool_output_cannot_fake_completion():
    supervisor = _supervisor()
    events = supervisor._NativeEvents()
    events.feed(b'{"type":"item.started","item":{"id":"tool","type":"command_execution"}}\n')
    output = '{"type":"item.completed","item":{"id":"tool"}}' * supervisor._EVENT_MAX_BYTES
    raw = (json.dumps({"type":"item.updated","item":{"id":"tool","type":"command_execution",
        "status":"in_progress","aggregated_output":output}}) + "\n").encode()
    for start in range(0, len(raw), 8192): events.feed(raw[start:start + 8192])
    assert events.active_tools == {"tool"}
    assert not events.response_stalled(900, events.last_activity + 3600)


@pytest.mark.parametrize("message", [
    "Selected model at capacity", "Selected model is at capacity. Try again later.",
    "model_capacity_exceeded", "model_overloaded",
])
@pytest.mark.parametrize("kind", ["error", "turn.failed"])
def test_capacity_is_retryable_in_both_native_error_events(message, kind):
    spec = importlib.util.spec_from_file_location("capacity_supervisor", ROOT / "scripts/run_codex_with_ephemeral_auth.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events = module._NativeEvents()
    events.feed((json.dumps({"type": kind, "error": {"message": message}}) + "\n").encode())
    events.feed(b'{"type":"turn.failed"}\n')
    assert events.retryable is True and events.retry_blocked is False
    assert "model at capacity" in events.failure


def test_managed_round_recovery_is_default_and_completion_stages_stay_bounded():
    assert WORKER._resume_until_deadline({}) is True
    assert WORKER._resume_until_deadline({"round": {"time_budget_mode": "hard_wall"}}) is True
    assert WORKER._resume_until_deadline({"round": {"time_budget_mode": "completion"}}) is False
    assert WORKER._resume_until_deadline({"worker_settings": {"transient_resume_until_deadline": False}}) is False
    for value in (None, 0, 1, "true", {}):
        with pytest.raises(ValueError, match="must be a boolean"):
            WORKER._resume_until_deadline({"worker_settings": {"transient_resume_until_deadline": value}})


def test_capacity_resumes_real_worker_despite_legacy_one_attempt_limit(tmp_path):
    outcome, audit, recovery, starts = _run(tmp_path, max_attempts=1, until_deadline=True,
        failures=("Selected model at capacity",), backoff_seconds=.02)
    assert outcome.execution_status == "completed"
    assert len(starts) == 2
    assert starts[1]["resumed"] is True
    assert starts[0]["thread_id"] == starts[1]["thread_id"]
    assert starts[1]["nomination_count_before"] == 1
    assert audit.final_nomination_sequence == 2
    assert recovery["until_deadline"] is True and recovery["max_attempts"] is None
    assert recovery["attempts"][0]["backoff_seconds"] == .02
    assert "model at capacity" in recovery["attempts"][0]["failure"]
    assert "[Agent recovery]" in Path(audit.stderr_path).read_text()


def _run_with_virtual_clock(tmp_path, monkeypatch, *, failures=4, failed_attempt_seconds=1,
                            deadline_seconds=14400, cleanup=True):
    """Exercise the actual recovery loop without a model, GPU, or real backoff."""
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location("virtual_clock_worker", ROOT / "scripts/run_canonical_codex_agent.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    workspace = tmp_path / "workspace"
    runtime_root = tmp_path / "runtime"
    workspace.mkdir()
    runtime_root.mkdir()
    auth, original_auth = write_auth(tmp_path)
    thread_id = "11111111-1111-4111-8111-111111111111"
    clock = {"now": 100.0}
    deadline = clock["now"] + deadline_seconds
    starts, sleeps = [], []
    def sleep(seconds):
        assert seconds >= 0
        sleeps.append(seconds)
        clock["now"] += seconds
    monkeypatch.setattr(worker, "time", SimpleNamespace(monotonic=lambda: clock["now"],
        time_ns=lambda: int(clock["now"] * 1e9), sleep=sleep))
    monkeypatch.setenv("PDE_AGENT_DEADLINE_MONOTONIC", str(deadline))
    monkeypatch.setattr(worker, "_codex_environment", lambda **kwargs: {})
    monkeypatch.setattr(worker, "_settle_capabilities", lambda *args: {"confirmed": cleanup})
    def invoke(command, **kwargs):
        native = command[command.index("--") + 1:]
        assert clock["now"] < deadline
        assert float(command[command.index("--deadline-monotonic") + 1]) == deadline
        assert "--ephemeral" not in native
        assert native[native.index("--model") + 1] == worker.MODEL
        assert starts == [] or native[-3:] == ("resume", thread_id, "-")
        starts.append({"time": clock["now"], "argv": native, "prompt": kwargs["input"]})
        private_auth = Path(command[command.index("--auth") + 1])
        assert private_auth.read_text() == original_auth
        session = private_auth.parent / "sessions" / f"rollout-{thread_id}.jsonl"
        if not session.exists():
            session.write_text(json.dumps({"type": "session_meta", "payload": {"id": thread_id}}) + "\n")
        private_auth.unlink()
        Path(command[command.index("--audit-copy") + 1]).write_text(json.dumps({
            "auth_removed_before_first_event_forwarded": True}))
        failed = len(starts) <= failures
        if failed:
            clock["now"] += min(failed_attempt_seconds, deadline - clock["now"])
        else:
            Path(native[native.index("--output-last-message") + 1]).write_text("Research resumed.\n")
        Path(command[command.index("--result") + 1]).write_text(json.dumps({
            "failure": "native turn.failed: model at capacity" if failed else None,
            "thread_id": thread_id, "retryable": failed, "process_group_cleaned": True}))
        return SimpleNamespace(returncode=1 if failed else 0)
    monkeypatch.setattr(worker, "subprocess", SimpleNamespace(run=invoke))
    code, failure = worker._run_codex(workspace=workspace, prompt="Keep improving this solver.",
        codex_binary=tmp_path / "mock-codex", auth_source=auth,
        config_source=ROOT / "scripts/canonical_codex.config.toml", runtime_root=runtime_root,
        final_message=workspace / "codex-last-message.txt", auth_audit_copy=workspace / "codex-auth-lifecycle.json",
        resume_max_attempts=1, resume_until_deadline=True)
    recovery = json.loads((workspace / "codex-recovery.json").read_text())
    assert recovery["deadline_monotonic"] == deadline
    assert auth.read_text() == original_auth and not list(runtime_root.iterdir())
    return code, failure, recovery, starts, sleeps, clock["now"], deadline


def test_capacity_recovers_after_more_than_three_attempts(tmp_path, monkeypatch):
    code, failure, recovery, starts, sleeps, now, deadline = _run_with_virtual_clock(tmp_path, monkeypatch)
    assert code == 0 and failure is None and len(starts) == 5
    assert recovery["status"] == "completed"
    assert sleeps == [30, 60, 60, 60]
    assert now < deadline
    assert all(row["prompt"].decode() == WORKER._RESUME_PROMPT for row in starts[1:])
    remaining = [row["remaining_seconds"] for row in recovery["attempts"]]
    assert remaining == sorted(remaining, reverse=True)


def test_persistent_capacity_stops_at_original_four_hour_deadline(tmp_path, monkeypatch):
    code, failure, recovery, starts, sleeps, now, deadline = _run_with_virtual_clock(
        tmp_path, monkeypatch, failures=100, failed_attempt_seconds=1000)
    assert code != 0 and len(starts) > 3
    assert recovery["status"] == "deadline"
    assert now == deadline == 100 + 4 * 3600
    assert max(sleeps) <= 60 and all(row["time"] < deadline for row in starts)


def test_last_backoff_uses_only_remaining_budget(tmp_path, monkeypatch):
    code, failure, recovery, starts, sleeps, now, deadline = _run_with_virtual_clock(
        tmp_path, monkeypatch, failures=100, failed_attempt_seconds=1, deadline_seconds=3)
    assert recovery["status"] == "deadline" and len(starts) == 1
    assert sleeps == [2] and now == deadline


def test_deadline_retries_do_not_overlap_unsettled_evaluations(tmp_path, monkeypatch):
    code, failure, recovery, starts, sleeps, now, deadline = _run_with_virtual_clock(
        tmp_path, monkeypatch, cleanup=False)
    assert code != 0 and len(starts) == 1 and sleeps == []
    assert recovery["status"] == "controller_cleanup_unconfirmed"


def test_deadline_recovery_requires_a_deadline_even_for_completion_mode(tmp_path, monkeypatch):
    monkeypatch.delenv("PDE_AGENT_DEADLINE_MONOTONIC", raising=False)
    with pytest.raises(ValueError, match="requires the managed Agent deadline"):
        WORKER._run_codex(workspace=tmp_path, prompt="unused", codex_binary=tmp_path / "unused",
            auth_source=tmp_path / "unused-auth", config_source=tmp_path / "unused-config",
            runtime_root=tmp_path, final_message=tmp_path / "final", auth_audit_copy=tmp_path / "audit",
            resume_until_deadline=True, completion_driven=True)


def test_capacity_on_first_model_request_resumes_before_any_solver_is_submitted(tmp_path):
    outcome, audit, recovery, starts = _run(tmp_path, max_attempts=1, until_deadline=True,
        failures=("Selected model at capacity",), fail_before_work=True, backoff_seconds=.02)
    assert outcome.execution_status == "completed" and len(starts) == 2
    assert starts[0]["nomination_count_before"] == starts[1]["nomination_count_before"] == 0
    assert starts[1]["resumed"] is True
    assert starts[0]["thread_id"] == starts[1]["thread_id"]
    assert audit.final_nomination_sequence == 1
    assert recovery["status"] == "completed"


@pytest.mark.parametrize("message", ["insufficient_quota", "Unauthorized", "Invalid prompt", "content_policy_violation"])
def test_capacity_does_not_override_an_unrecoverable_error(message):
    spec = importlib.util.spec_from_file_location("capacity_terminal_supervisor", ROOT / "scripts/run_codex_with_ephemeral_auth.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events = module._NativeEvents()
    events.feed((json.dumps({"type": "error", "message": message}) + "\n").encode())
    events.feed(b'{"type":"turn.failed","error":{"message":"Selected model at capacity"}}\n')
    assert events.retry_blocked is True and events.retryable is False
