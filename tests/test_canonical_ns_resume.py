"""Runner continuation keeps committed events, execution evidence, and failures."""
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.engine.audit_journal import AuditJournal
from pde_agent_verifier_solver.engine.campaign_continuation import CampaignContinuation, CampaignPaused
from pde_agent_verifier_solver.engine.candidate_execution import CandidateExecutionInfrastructureError
from pde_agent_verifier_solver.engine.runtime import EventStream
from pde_agent_verifier_solver.plugins.ns.official_audit import OfficialAuditGenerator


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("canonical_resume_test", Path(__file__).resolve().parents[1] / "scripts/run_canonical_ns_campaign.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Generator:
    def export_runtime_state(self, directory):
        return {"state": "unchanged"}

    def restore_runtime_state(self, state, directory):
        assert state == {"state": "unchanged"}


class Service:
    def __init__(self, fail=False):
        self.audits = []
        self.closed = False
        self.fail = fail
        self.max_workers = 28

    def close(self):
        self.closed = True
        if self.fail:
            raise RuntimeError("service cleanup was not confirmed")


def _setup(runner, monkeypatch, tmp_path, *, cleanup_failure=False):
    config = {"profile": "structured_numerical", "agents": [{"agent_id": "one"}]}
    config_path = tmp_path / "input.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(runner, "_load_config", lambda _: (dict(config), config_path))
    monkeypatch.setattr(runner, "_validate_config", lambda _: None)
    monkeypatch.setattr(runner, "_gpu_allocation", lambda *_, **__: (("0",), "1"))
    monkeypatch.setattr(runner, "_required_executable", lambda *_: config_path)
    monkeypatch.setattr(runner, "_required_regular_file", lambda *_: config_path)
    monkeypatch.setattr(runner, "_source_state", lambda _: {"segment": len(services)})
    monkeypatch.setattr(runner, "_bind_hidden_evaluation_audit", lambda *_: None)
    monkeypatch.setattr(runner, "_record_cost_report", lambda *_: None)
    monkeypatch.setattr(runner, "_publish_selected_candidate", lambda *_: None)
    import pde_agent_verifier_solver.engine.checkpoint_store as storage
    monkeypatch.setattr(storage, "source_identity", lambda _: "frozen-source")
    services = []

    def compose(**kwargs):
        agent, candidate = Service(cleanup_failure), Service(cleanup_failure)
        services.append((agent, candidate))
        return SimpleNamespace(campaign=SimpleNamespace(master_seed=101), dataset=None,
            problem=None, task=None, generator=Generator(), public_problem_bundle=None,
            agents={}, execution=agent, candidate_execution=candidate,
            provenance={"generator_artifacts": {"pca": {"sha256": "frozen-pca"}}})

    monkeypatch.setattr(runner, "_compose", compose)

    class Controller:
        def __init__(self, **kwargs):
            self.events = kwargs["events"]
            self.agent, self.candidate = kwargs["agent_execution"], kwargs["candidate_execution"]

        def run(self, *, continuation=None, pause_control=None):
            if continuation is not None and continuation.state.get("phase") == "complete":
                self.events._events = list(continuation.controller_state["events"])
                self.events.emit("run_resumed", "Finished work restored")
                return continuation.state["result"]
            index = len(services)
            self.agent.audits.append({"execution_id": f"agent-{index}"})
            self.candidate.audits.append({"execution_id": f"case-{index}", "case_index": 0})
            if continuation is None:
                self.events.emit("run_started", "Started")
                saved = CampaignContinuation({"phase": "official_ready"}, {"events": self.events.snapshot()})
                pause_control.save(saved)
                # The process may produce extra events after its last committed checkpoint.
                self.events.emit("uncommitted", "Not part of committed scientific state")
                raise CampaignPaused(saved, "official")
            self.events._events = list(continuation.controller_state["events"])
            self.events.emit("run_resumed", "Restored")
            pause_control.save(continuation)
            result = SimpleNamespace(memory=SimpleNamespace(markdown="completed-memory"),
                candidate_history={}, selected=SimpleNamespace(artifact=SimpleNamespace(artifact_id="S3")),
                campaign_name="fixture", best_terminal_round=3, rounds=(1, 2, 3))
            pause_control.save(CampaignContinuation({"phase": "complete", "result": result},
                                                   {"events": self.events.snapshot()}))
            return result

    monkeypatch.setattr(runner, "CampaignController", Controller)
    segment = SimpleNamespace(should_pause=lambda *_: False, cancel=Event(), signals=nullcontext)
    args = ["--config", str(config_path), "--run-root", str(tmp_path / "run")]
    return args, segment, services, config


def test_runner_restores_committed_events_and_retains_all_segment_audits(runner, monkeypatch, tmp_path):
    args, segment, services, _ = _setup(runner, monkeypatch, tmp_path)
    root = tmp_path / "run"
    assert runner.main(args, segment_budget=segment) == 75
    initial_metadata = {name: (root / name).read_bytes() for name in
                        ("source_state.json", "input_provenance.json", "resolved_config.json", "compiled_campaign.json")}
    # Also exercise a partially written trailing event from a killed process.
    with (root / "events.jsonl").open("a") as stream:
        stream.write('{"partial":')
    assert runner.main(args + ["--resume"], segment_budget=segment) == 0
    history = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    assert [row["sequence"] for row in history] == [1, 2]
    assert [row["event_type"] for row in history] == ["run_started", "run_resumed"]
    archives = list((root / "segments").glob("*/events_before_resume.jsonl"))
    assert len(archives) == 1 and '"partial"' in archives[0].read_text()
    assert initial_metadata == {name: (root / name).read_bytes() for name in initial_metadata}
    for filename, prefix in (("execution_audits.json", "agent"), ("candidate_execution_audits.json", "case")):
        assert [row["execution_id"] for row in json.loads((root / filename).read_text())] == [f"{prefix}-1", f"{prefix}-2"]
    assert all(service.closed for pair in services for service in pair)
    assert json.loads((root / "status.json").read_text())["status"] == "completed"
    before = (root / "candidate_execution_audits.json").read_bytes()
    assert runner.main(args + ["--resume"], segment_budget=segment) == 0
    assert (root / "candidate_execution_audits.json").read_bytes() == before
    assert services[-1][0].audits == services[-1][1].audits == []


def test_cleanup_failure_closes_both_services_and_overrides_paused_status(runner, monkeypatch, tmp_path):
    args, segment, services, _ = _setup(runner, monkeypatch, tmp_path, cleanup_failure=True)
    with pytest.raises(CandidateExecutionInfrastructureError, match="cleanup"):
        runner._main(args, segment_budget=segment)
    assert all(service.closed for pair in services for service in pair)
    root = tmp_path / "run"
    assert json.loads((root / "status.json").read_text())["status"] == "failed"
    assert len(json.loads((root / "execution_audits.json").read_text())) == 1
    assert (root / "cleanup_failure_traceback.txt").exists()


def test_changed_generator_input_rejects_resume_without_overwriting_initial_evidence(runner, monkeypatch, tmp_path):
    args, segment, services, config = _setup(runner, monkeypatch, tmp_path)
    assert runner._main(args, segment_budget=segment) == 75
    root = tmp_path / "run"
    initial = (root / "resolved_config.json").read_bytes()
    config["changed_setting"] = True
    with pytest.raises(ValueError, match="identity differs"):
        runner._main(args + ["--resume"], segment_budget=segment)
    assert (root / "resolved_config.json").read_bytes() == initial
    assert json.loads((root / "status.json").read_text())["status"] == "failed"
    assert all(service.closed for pair in services for service in pair)


def test_setup_failure_is_persisted_but_existing_unrelated_run_is_not_modified(runner, monkeypatch, tmp_path):
    args, segment, _, _ = _setup(runner, monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "_validate_config", lambda _: (_ for _ in ()).throw(ValueError("invalid setup")))
    with pytest.raises(ValueError, match="invalid setup"):
        runner._main(args, segment_budget=segment)
    root = tmp_path / "run"
    previous = (root / "status.json").read_bytes()
    assert json.loads(previous)["status"] == "failed"
    with pytest.raises(ValueError):
        runner._main(args, segment_budget=segment)
    assert (root / "status.json").read_bytes() == previous


def test_audit_journal_is_incremental_and_recovers_only_truncated_tail(tmp_path):
    path = tmp_path / "audits.jsonl"
    journal = AuditJournal(path, prior_rows=[{"execution_id": "old"}])
    rows = [{"execution_id": "a", "case_index": 0}]
    journal.append(rows)
    first = path.read_bytes()
    journal.append(rows)
    assert path.read_bytes() == first
    rows.append({"execution_id": "a", "case_index": 1})
    journal.append(rows)
    with path.open("ab") as stream:
        stream.write(b'{"execution_id":')
    restored = AuditJournal(path)
    restored.append([{"execution_id": "new"}])
    assert restored.rows == [{"execution_id": "old"}, *rows, {"execution_id": "new"}]
    assert len(list(tmp_path.glob("*.uncommitted-*"))) == 1
    assert path.stat().st_mode & 0o777 == 0o600
    path.write_text('{bad}\n{"execution_id":"valid"}\n')
    with pytest.raises(ValueError, match="corrupt committed"):
        AuditJournal(path)


def test_official_audit_codec_preserves_completed_rounds_and_rejects_changed_bank(tmp_path):
    def wrapper(digest):
        value = object.__new__(OfficialAuditGenerator)
        value._generator, value._bank_sha256 = Generator(), digest
        value._expected, value._seen, value._lock = {1: {}, 2: {}}, set(), Lock()
        return value

    first = wrapper("original")
    first._seen = {1}
    saved = first.export_runtime_state(tmp_path)
    restored = wrapper("original")
    restored.restore_runtime_state(saved, tmp_path)
    assert restored._seen == {1}
    restored._seen.add(2)
    restored.assert_complete()
    with pytest.raises(ValueError, match="bank identity"):
        wrapper("changed").restore_runtime_state(saved, tmp_path)
    with pytest.raises(ValueError, match="completed rounds"):
        wrapper("original").restore_runtime_state({**saved, "seen": [1, 3]}, tmp_path)


def test_selected_publication_is_atomic_and_idempotent(runner, monkeypatch, tmp_path):
    from pde_agent_verifier_solver.sdk import write_candidate_bundle_manifest, inspect_candidate_bundle
    source = tmp_path / "bundle"
    source.mkdir()
    (source / "submission.py").write_text("def prepare_solution(case, solve_budget=None): return None\n")
    write_candidate_bundle_manifest(source, files=["submission.py"])
    digest = inspect_candidate_bundle(source).tree_sha256
    artifact = SimpleNamespace(program=SimpleNamespace(bundle_locator=source, tree_sha256=digest))
    destination = tmp_path / "selected"
    copytree = runner.shutil.copytree

    def interrupted_copy(src, dest, **kwargs):
        Path(dest).mkdir()
        (Path(dest) / "partial").write_text("interrupted")
        raise OSError("copy interrupted")

    monkeypatch.setattr(runner.shutil, "copytree", interrupted_copy)
    with pytest.raises(OSError, match="copy interrupted"):
        runner._publish_selected_candidate(artifact, destination)
    assert not destination.exists()
    monkeypatch.setattr(runner.shutil, "copytree", copytree)
    runner._publish_selected_candidate(artifact, destination)
    runner._publish_selected_candidate(artifact, destination)
    assert inspect_candidate_bundle(destination).tree_sha256 == digest
    (destination / "submission.py").write_text("def prepare_solution(case, solve_budget=None): return 2\n")
    with pytest.raises(FileExistsError, match="identity differs"):
        runner._publish_selected_candidate(artifact, destination)


def test_audit_failed_fsync_does_not_duplicate_uncertain_append(tmp_path, monkeypatch):
    import pde_agent_verifier_solver.engine.audit_journal as module
    journal = AuditJournal(tmp_path / "audit.jsonl")
    monkeypatch.setattr(module.os, "fsync", lambda _: (_ for _ in ()).throw(OSError("sync failed")))
    with pytest.raises(OSError, match="sync failed"):
        journal.append([{"execution_id": "one"}])
    before = journal.path.read_bytes()
    with pytest.raises(RuntimeError, match="ambiguous retry"):
        journal.append([{"execution_id": "one"}])
    assert journal.path.read_bytes() == before
