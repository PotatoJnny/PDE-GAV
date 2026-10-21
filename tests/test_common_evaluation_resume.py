"""Case-boundary continuation preserves scientific results and failure meaning."""
from dataclasses import replace
import json
from threading import Event
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.engine import common_evaluation as common
from pde_agent_verifier_solver.engine.candidate_execution import (
    CandidateExecutionCancelled, CandidateExecutionInfrastructureError,
)
from pde_agent_verifier_solver.plugins.toy import SineTruth
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, Artifact, Case, CaseRecord, CaseSource, CaseVisibility,
    EvaluationPhase, EvaluationPlan, TaskKind,
)


class Budget:
    def __init__(self):
        self.cancel = Event()
    def should_pause(self, stage, required_seconds=0):
        return self.cancel.is_set()


def inputs(output):
    query = AnswerQuery.at_points(fields=('u',), points=({'x': .25},))
    cases = tuple(common.CommonCase('cohort', 'toy', CaseRecord(
        Case(f'case-{index}', CaseSource.GENERATED, CaseVisibility.HIDDEN, {}), SineTruth(1., 1)),
        EvaluationPlan(EvaluationPhase.TERMINAL, query, seed=index)) for index in range(3))
    programs = tuple(common.CommonProgram(f'method-{index}', 'run', 101, f'artifact-{index}',
        Artifact(f'artifact-{index}', TaskKind.SOLVER, 'test', object(), 'test', 'test', 1, 'agent'),
        str(index) * 64) for index in range(2))
    output.mkdir()
    return dict(task_id='toy', programs=programs, cases=cases, evaluation=None,
                evaluation_settings={}, candidate_execution=None, metrics=('l2', 'linf'),
                output_directory=output, max_workers=1)


def result(case):
    value = int(case.case.case_id.split('-')[-1]) / 10
    return SimpleNamespace(cases=(SimpleNamespace(metrics={'l2': value, 'linf': value + 1}),))


def rows(root):
    return [json.loads(line) for line in (root / 'cases.jsonl').read_text().splitlines()]


def scientific_rows(root):
    return [{key: value for key, value in row.items() if key != 'case_wall_seconds'} for row in rows(root)]


def test_uninterrupted_and_two_resumes_keep_identical_rows_without_reexecution(tmp_path, monkeypatch):
    budget = Budget()
    seen = []
    should_stop = [False]
    class Verifier:
        def __init__(self, *args, **kwargs):
            pass
        def evaluate(self, artifact, records):
            seen.append((artifact.artifact_id, records[0].case.case_id))
            if should_stop[0]:
                budget.cancel.set()
            return result(records[0])
    monkeypatch.setattr(common, 'HiddenVerifier', Verifier)
    uninterrupted = inputs(tmp_path / 'uninterrupted')
    full = common.evaluate_common_programs(**uninterrupted)
    expected_calls = list(seen)
    seen.clear()
    split = inputs(tmp_path / 'split')
    should_stop[0] = True
    paused = common.evaluate_common_programs(**split, segment_budget=budget)
    assert paused['status'] == 'paused' and paused['n_completed_program_cases'] == 1
    assert not (split['output_directory'] / 'summary.json').exists()
    assert len(rows(split['output_directory'])) == 2
    budget.cancel.clear()
    paused = common.evaluate_common_programs(**split, resume=True, segment_budget=budget)
    assert paused['n_completed_program_cases'] == 2
    should_stop[0] = False
    budget.cancel.clear()
    resumed = common.evaluate_common_programs(**split, resume=True, segment_budget=budget)
    assert resumed['status'] == 'completed'
    assert seen == expected_calls
    assert full['groups'] == resumed['groups']
    assert scientific_rows(uninterrupted['output_directory']) == scientific_rows(split['output_directory'])
    assert json.loads((split['output_directory'] / 'checkpoint.json').read_text())['status'] == 'completed'
    seen.clear()
    common.evaluate_common_programs(**split, resume=True, segment_budget=budget)
    assert seen == []


def test_allocation_cancelled_case_is_missing_then_retried_not_failure(tmp_path, monkeypatch):
    budget = Budget()
    calls = []
    class Verifier:
        def __init__(self, *args, **kwargs):
            pass
        def evaluate(self, artifact, records):
            calls.append((artifact.artifact_id, records[0].case.case_id))
            if len(calls) == 2:
                budget.cancel.set()
                raise CandidateExecutionCancelled('allocation ended')
            return result(records[0])
    monkeypatch.setattr(common, 'HiddenVerifier', Verifier)
    kwargs = inputs(tmp_path / 'split')
    paused = common.evaluate_common_programs(**kwargs, segment_budget=budget)
    assert paused['status'] == 'paused' and paused['n_completed_program_cases'] == 1
    assert all(row['status'] == 'ok' for row in rows(kwargs['output_directory']))
    assert not (kwargs['output_directory'] / 'failure.json').exists()
    assert not (kwargs['output_directory'] / 'controller_private').exists()
    budget.cancel.clear()
    finished = common.evaluate_common_programs(**kwargs, resume=True, segment_budget=budget)
    assert finished['n_failed_program_cases'] == 0
    assert calls[1] == calls[2]  # Only interrupted, uncommitted Case is retried.
    assert calls.count(calls[0]) == 1


def test_cancel_with_cleanup_failure_is_terminal_not_pause(tmp_path, monkeypatch):
    budget = Budget()
    class Verifier:
        def __init__(self, *args, **kwargs):
            pass
        def evaluate(self, artifact, records):
            budget.cancel.set()
            error = CandidateExecutionCancelled('allocation ended')
            error._candidate_cleanup_errors = (CandidateExecutionInfrastructureError('cleanup failed'),)
            raise error
    monkeypatch.setattr(common, 'HiddenVerifier', Verifier)
    kwargs = inputs(tmp_path / 'failed')
    with pytest.raises(CandidateExecutionInfrastructureError):
        common.evaluate_common_programs(**kwargs, segment_budget=budget)
    state = json.loads((kwargs['output_directory'] / 'checkpoint.json').read_text())
    assert state['status'] == 'failed'
    with pytest.raises(ValueError, match='safely paused'):
        common.evaluate_common_programs(**kwargs, resume=True, segment_budget=budget)


@pytest.mark.parametrize('change', ['program', 'query', 'truth', 'settings', 'rows'])
def test_resume_rejects_drift_before_any_candidate_call(tmp_path, monkeypatch, change):
    budget = Budget()
    budget.cancel.set()
    kwargs = inputs(tmp_path / 'paused')
    class Verifier:
        def __init__(self, *args, **kwargs):
            pass
        def evaluate(self, artifact, records):
            pytest.fail('candidate must not execute with changed checkpoint inputs')
    monkeypatch.setattr(common, 'HiddenVerifier', Verifier)
    common.evaluate_common_programs(**kwargs, segment_budget=budget)
    if change == 'program':
        kwargs['programs'] = (replace(kwargs['programs'][0], artifact_sha256='changed'), *kwargs['programs'][1:])
    elif change == 'query':
        first = kwargs['cases'][0]
        query = AnswerQuery.at_points(fields=('u',), points=({'x': .5},))
        kwargs['cases'] = (replace(first, plan=replace(first.plan, query=query)), *kwargs['cases'][1:])
    elif change == 'truth':
        first = kwargs['cases'][0]
        kwargs['cases'] = (replace(first, record=CaseRecord(first.record.case, SineTruth(2., 1))), *kwargs['cases'][1:])
    elif change == 'settings':
        kwargs['evaluation_settings'] = {'different': True}
    else:
        path = kwargs['output_directory'] / 'checkpoint.json'
        saved = json.loads(path.read_text())
        saved['completed']['0'] = {'path': 'completed_cases/00000000.json', 'sha256': 'wrong'}
        (kwargs['output_directory'] / 'completed_cases' / '00000000.json').write_text('[]')
        path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='differ|incomplete metric rows'):
        common.evaluate_common_programs(**kwargs, resume=True, segment_budget=budget)


def test_bounded_inflight_cancellation_waits_for_all_started_workers(tmp_path, monkeypatch):
    from threading import Barrier, Lock
    budget = Budget()
    barrier = Barrier(2)
    lock = Lock()
    started = []
    ended = []
    class Verifier:
        def __init__(self, *args, **kwargs):
            pass
        def evaluate(self, artifact, records):
            with lock:
                started.append(records[0].case.case_id)
            barrier.wait(timeout=3)
            budget.cancel.set()
            with lock:
                ended.append(records[0].case.case_id)
            return result(records[0])
    monkeypatch.setattr(common, 'HiddenVerifier', Verifier)
    kwargs = inputs(tmp_path / 'bounded')
    kwargs['max_workers'] = 2
    paused = common.evaluate_common_programs(**kwargs, segment_budget=budget)
    assert paused['status'] == 'paused'
    assert sorted(started) == sorted(ended) == ['case-0', 'case-1']
    assert paused['n_completed_program_cases'] == 2
