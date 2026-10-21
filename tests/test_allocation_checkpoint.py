"""Allocation deadlines preserve completed work without renewing Agent time."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import pickle
from pathlib import Path
import signal
import time

import pytest

from pde_agent_verifier_solver.engine.checkpoint_store import (
    CheckpointStore, RunSegmentBudget, atomic_bytes, run_writer_lock,
)
from pde_agent_verifier_solver.engine.candidate_execution import (
    CandidateExecutionCancelled, CandidateExecutionInfrastructureError,
    SubprocessCandidateExecutionService,
)
from pde_agent_verifier_solver.engine import PythonTaskWorkerSpec
from pde_agent_verifier_solver.plugins.ns import NSSolverTask
from pde_agent_verifier_solver.sdk import EvaluationPhase, EvaluationPlan
from tests.test_candidate_execution import _artifact, _case, _query, _evaluation_settings


def test_atomic_checkpoint_keeps_last_commit_on_write_failure(tmp_path, monkeypatch):
    store = CheckpointStore(tmp_path, identity={'config': 1})
    store.save({'round': 1})
    original = Path.replace
    def fail_manifest(self, destination):
        if destination == store.path:
            raise OSError('persistent filesystem unavailable')
        return original(self, destination)
    monkeypatch.setattr(Path, 'replace', fail_manifest)
    with pytest.raises(OSError, match='unavailable'):
        store.save({'round': 2})
    assert CheckpointStore(tmp_path, identity={'config': 1}).load() == {'round': 1}
    assert not list(tmp_path.glob('.*.tmp'))
    with pytest.raises(ValueError, match='identity differs'):
        CheckpointStore(tmp_path, identity={'config': 2}).load()
    manifest = json.loads(store.path.read_text())
    payload = store.objects / manifest['payload']['name']
    payload.write_bytes(b'corrupted')
    with pytest.raises(pickle.UnpicklingError):
        CheckpointStore(tmp_path, identity={'config': 1}).load()


def test_single_writer_and_allocation_reserve_do_not_change_agent_budget(tmp_path):
    with run_writer_lock(tmp_path):
        with pytest.raises(RuntimeError, match='another writer'):
            with run_writer_lock(tmp_path):
                pass
    with run_writer_lock(tmp_path):
        pass
    clock = [100.]
    budget = RunSegmentBudget(43200., monotonic=lambda: clock[0])
    old_signal = signal.getsignal(signal.SIGUSR1)
    try:
        assert not budget.should_pause('Agent', 14400 + 600)
        clock[0] += 30000
        assert budget.should_pause('Agent', 14400 + 600)
        assert not budget.cancel.is_set()  # Do not interrupt an already running turn.
        with budget.signals():
            os.kill(os.getpid(), signal.SIGUSR1)
            assert budget.should_pause('hidden batch', 0)
        assert signal.getsignal(signal.SIGUSR1) == old_signal
    finally:
        budget.close()
    for args in [(600, 600), (float('nan'), 1), (1000, float('nan'))]:
        with pytest.raises(ValueError):
            RunSegmentBudget(args[0], flush_seconds=args[1])


@pytest.mark.parametrize('broken_scratch_cleanup', [False, True])
def test_allocation_cancellation_retains_completed_peer_and_checks_cleanup(tmp_path, monkeypatch, broken_scratch_cleanup):
    from pde_agent_verifier_solver.engine import candidate_execution as module
    from threading import Event
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    monkeypatch.setenv('SLURM_TMPDIR', str(scratch))
    artifact = _artifact(tmp_path)
    query = _query()
    settings = _evaluation_settings(query)
    service = SubprocessCandidateExecutionService(tmp_path / 'execution',
        task=PythonTaskWorkerSpec.from_task_type(NSSolverTask),
        evaluation_settings=settings, max_workers=2, case_hard_wall_seconds=60)
    cancellation = Event()
    service.allocation_cancellation_event = cancellation
    cases = (_case('fast', value=2.), _case('slow', value=3., sleep_seconds=30.))
    plans = tuple(EvaluationPlan(phase=EvaluationPhase.OFFICIAL, query=query, settings=settings) for _ in cases)
    original_remove = module.shutil.rmtree
    def remove(path, *args, **kwargs):
        if broken_scratch_cleanup and cancellation.is_set() and Path(path).parent == scratch:
            raise OSError('scratch cleanup failed')
        return original_remove(path, *args, **kwargs)
    monkeypatch.setattr(module.shutil, 'rmtree', remove)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(service.execute_cases, artifact, cases, plans,
                                     NSSolverTask().create_evaluation(settings))
            deadline = time.monotonic() + 20
            while not any(row.case_id == 'fast' and row.status == 'completed' for row in service.audits):
                assert not future.done(), 'batch ended before allocation cancellation'
                assert time.monotonic() < deadline, 'fast Case did not complete'
                time.sleep(.02)
            cancellation.set()
            expected = CandidateExecutionInfrastructureError if broken_scratch_cleanup else CandidateExecutionCancelled
            with pytest.raises(expected) as caught:
                future.result(timeout=10)
        if broken_scratch_cleanup:
            assert not getattr(caught.value, '_allocation_pause_requested', False)
        else:
            assert caught.value._allocation_pause_requested
            assert set(caught.value._candidate_partial_outputs) == {0}
            assert caught.value._candidate_partial_outputs[0].prediction.fields['velocity_x'] == (2., 2.)
        assert all(row.process_group_cleaned for row in service.audits)
    finally:
        cancellation.set()
        service._executor.shutdown(wait=True, cancel_futures=True)


def test_no_progress_identity_ignores_rng_but_detects_completed_case(tmp_path):
    import random
    store = CheckpointStore(tmp_path, identity={"fixture": 1})
    state = {"phase": "official_ready", "hidden_progress": {"official": {"cases": {"S1": {}}}}}
    store.save(state)
    before = json.loads(store.path.read_text())
    random.random()
    store.save(state)
    second = json.loads(store.path.read_text())
    assert before["progress"] == second["progress"]
    assert before["payload"] != second["payload"]
    state["hidden_progress"]["official"]["cases"]["S1"]["case1"] = 0.1
    store.save(state)
    assert json.loads(store.path.read_text())["progress"] != before["progress"]


def test_generator_cost_resume_retains_completed_calls_and_repairs_crash_tail(tmp_path, monkeypatch):
    from pde_agent_verifier_solver.reporting.costs import MeasuredGenerator
    from pde_agent_verifier_solver.reporting import costs
    path = tmp_path / "generator_operations.jsonl"
    first = MeasuredGenerator(object(), path)
    with first._operation("sampling"):
        pass
    syncs = []
    monkeypatch.setattr(costs.os, "fsync", lambda fd: syncs.append(fd))
    first.flush_operations()
    first.flush_operations()
    assert len(syncs) == 1
    valid = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(b'{"operation_id":')
    resumed = MeasuredGenerator(object(), path, resume=True)
    assert path.read_bytes() == valid
    assert list(tmp_path.glob("*.uncommitted-*"))
    with resumed._operation("sampling"):
        pass
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row['operation_id'] for row in rows] == [1, 1, 2, 2]
