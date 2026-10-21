"""Real NS worker execution resumes the frozen external test program list."""
import importlib.util
import json
from pathlib import Path
from threading import Event, Timer

import pytest

from pde_agent_verifier_solver.engine.candidate_execution import SubprocessCandidateExecutionService
from pde_agent_verifier_solver.plugins.ns.common_evaluation import run_ns_common_evaluation


def fixtures():
    spec = importlib.util.spec_from_file_location('ns_common_fixture', Path(__file__).with_name('test_ns_common_evaluation.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PauseAfterOne:
    def __init__(self):
        self.cancel = Event()
        self.starts = 0
    def should_pause(self, stage, required_seconds=0):
        self.starts += 1
        if self.starts > 1:
            self.cancel.set()
        return self.cancel.is_set()


def test_real_ns_resume_reuses_one_case_and_original_frozen_program(tmp_path):
    helper = fixtures()
    manifest, spec = helper._manifest(tmp_path, include_partial=False)
    spec['budget']['max_workers'] = 1
    helper._write(manifest, spec)
    output = tmp_path / 'resumed'
    first = run_ns_common_evaluation(manifest, output, segment_budget=PauseAfterOne())
    assert first['status'] == 'paused' and first['n_completed_program_cases'] == 1
    audit_path = output / 'execution' / 'candidate_execution_audits.jsonl'
    assert len(audit_path.read_text().splitlines()) == 1
    first_rows = helper._rows(output)
    frozen_before = json.loads((output / 'resolved_manifest.json').read_text())
    # Original source becomes unusable: continuation must keep the frozen copy.
    Path(spec['programs'][0]['bundle'], 'submission.py').write_text('raise RuntimeError("source changed")')
    finished = run_ns_common_evaluation(manifest, output, resume=True)
    assert finished['status'] == 'completed' and finished['n_failed_program_cases'] == 0
    assert len(audit_path.read_text().splitlines()) == 4
    assert helper._rows(output)[:len(first_rows)] == first_rows
    assert json.loads((output / 'resolved_manifest.json').read_text()) == frozen_before
    assert json.loads((output / 'status.json').read_text())['status'] == 'completed'
    assert json.loads((output / 'checkpoint.json').read_text())['status'] == 'completed'
    again = run_ns_common_evaluation(manifest, output, resume=True)
    assert again['groups'] == finished['groups']
    assert len(audit_path.read_text().splitlines()) == 4


def test_real_allocation_cancellation_does_not_create_solver_failure(tmp_path, monkeypatch):
    helper = fixtures()
    manifest, spec = helper._manifest(tmp_path, include_partial=False)
    sleeper = helper._bundle(tmp_path / 'sleeper', sleep=True)
    spec['programs'] = [{**spec['programs'][1], 'bundle': str(sleeper)}]
    spec['budget']['max_workers'] = 1
    helper._write(manifest, spec)
    budget = PauseAfterOne()
    original = SubprocessCandidateExecutionService.execute_cases
    timers = []
    def evaluate(self, *args, **kwargs):
        assert self.allocation_cancellation_event is budget.cancel
        timer = Timer(.2, budget.cancel.set)
        timers.append(timer)
        timer.start()
        return original(self, *args, **kwargs)
    monkeypatch.setattr(SubprocessCandidateExecutionService, 'execute_cases', evaluate)
    output = tmp_path / 'cancelled'
    try:
        paused = run_ns_common_evaluation(manifest, output, segment_budget=budget)
    finally:
        for timer in timers:
            timer.cancel()
            timer.join(timeout=2)
    assert paused['status'] == 'paused' and paused['n_completed_program_cases'] == 0
    assert helper._rows(output) == []
    assert not (output / 'summary.json').exists()
    assert not (output / 'failure.json').exists()
    assert json.loads((output / 'checkpoint.json').read_text())['status'] == 'paused'


def test_ns_resume_rejects_manifest_drift_before_new_execution(tmp_path):
    helper = fixtures()
    manifest, spec = helper._manifest(tmp_path, include_partial=False)
    spec['budget']['max_workers'] = 1
    helper._write(manifest, spec)
    output = tmp_path / 'paused'
    run_ns_common_evaluation(manifest, output, segment_budget=PauseAfterOne())
    audit = output / 'execution' / 'candidate_execution_audits.jsonl'
    before = audit.read_bytes()
    spec['budget']['case_hard_wall_seconds'] = 60
    helper._write(manifest, spec)
    with pytest.raises(ValueError, match='input manifest differs'):
        run_ns_common_evaluation(manifest, output, resume=True)
    assert audit.read_bytes() == before
