"""Development failures retain public scalar evidence without becoming scores."""
from contextlib import contextmanager
import json
import sys
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.engine.agent_capability import (
    AgentCapabilityClientError, ControllerAgentCapabilityClient,
    ControllerAgentCapabilityFacade,
)
from pde_agent_verifier_solver.engine.agent_execution import CommandAgentProgram, AgentExecutionError
from pde_agent_verifier_solver.engine.campaign import EvaluatorPolicy
from pde_agent_verifier_solver.engine.candidate_execution import (
    CandidateExecutionError, CandidateExecutionTimeout, CandidateExecutionBatchTimeout,
    CandidateExecutionCancelled, CandidateExecutionInfrastructureError,
)
from pde_agent_verifier_solver.engine.evaluator import AgentEvaluator, EvaluationRequest
from pde_agent_verifier_solver.engine.python_candidate import PythonCandidateError
from pde_agent_verifier_solver.plugins.toy import (
    ToyPoissonProblem, ToySineSolver, ToySolverEvaluation, ToySolverTask,
)
from pde_agent_verifier_solver.sdk import AnswerQuery, Artifact, DatasetInput, DatasetKind, TaskKind
from pde_agent_verifier_solver.sdk.candidate_bundle import CandidateBundleError

PRIVATE = 'private-truth-path secret-controller-case-id'


def artifact(name):
    return Artifact(name, TaskKind.SOLVER, name, ToySineSolver(2), 'test', 'solve', 1, 'test')


class FailingEvaluation(ToySolverEvaluation):
    def __init__(self, error_type):
        super().__init__(AnswerQuery.grid(fields=('u',), x=17))
        self.error_type = error_type
        self.calls = []

    def evaluate(self, program, case, truth, plan):
        self.calls.append(program.artifact_id)
        if program.artifact_id == 'bad' and self.calls.count('bad') == 2:
            raise self.error_type(PRIVATE)
        return super().evaluate(program, case, truth, plan)


def make_evaluator(error_type=CandidateExecutionTimeout, *, split=False):
    records = ToyPoissonProblem().prepare_dataset(DatasetInput('test', DatasetKind.NUMERICAL)).development_cases[:2]
    evaluation = FailingEvaluation(error_type)
    class Task(ToySolverTask):
        def create_evaluation(self, settings):
            return evaluation
    service = None
    if split:
        evaluation.supports_split_candidate_execution = True
        evaluation.execute_candidate = lambda *args: None
        evaluation.score_candidate = evaluation.evaluate
        # Split score has an additional candidate-output argument.
        evaluation.score_candidate = lambda a, c, t, p, output: evaluation.evaluate(a, c, t, p)
        class Service:
            def execute_cases(self, a, cases, plans, ev, **kwargs):
                if a.artifact_id == 'bad':
                    error = error_type(PRIVATE)
                    error._candidate_partial_outputs = {0: None}
                    raise error
                return tuple(None for _ in cases)
        service = Service()
    evaluator = AgentEvaluator(records, Task(), EvaluatorPolicy(
        AnswerQuery.grid(fields=('u',), x=17), allow_grid_truth=True, allow_point_truth=True,
    ), candidate_execution=service)
    return evaluator, evaluation, records


@contextmanager
def client_for(tmp_path, evaluator, decoder=None, *, expected_cleanup_failure=False):
    context = SimpleNamespace(evaluator=evaluator, sampler=None, cases=evaluator.cases)
    decoder = decoder or (lambda p, c, w: artifact(p['artifact_id']))
    lease = ControllerAgentCapabilityFacade().open(
        context, tmp_path, 'failure-feedback', CommandAgentProgram((sys.executable, '-c', 'pass'), decoder),
    )
    try:
        yield ControllerAgentCapabilityClient(lease.manifest, workspace=tmp_path)
    finally:
        if expected_cleanup_failure:
            with pytest.raises(AgentExecutionError, match='cleanup'):
                lease.close()
        else:
            lease.close()


@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('return_truth', [False, True])
def test_rpc_failure_retains_completed_candidate_and_partial_cases(tmp_path, split, return_truth):
    evaluator, evaluation, records = make_evaluator(split=split)
    ids = tuple(c.case_id for c in evaluator.cases)
    with client_for(tmp_path, evaluator) as client:
        with pytest.raises(AgentCapabilityClientError) as caught:
            client.evaluate_many([{'artifact_id': x} for x in ('good', 'bad', 'unattempted')],
                                 case_ids=ids, return_truth=return_truth)
        error = caught.value
        assert error.code == 'candidate_timeout'
        assert error.http_status == 408
        p = error.details
        assert p['complete'] is False
        assert p['completed_candidates'][0]['artifact_id'] == 'good'
        assert len(p['completed_candidates'][0]['cases']) == 2
        assert p['interrupted_candidate']['index'] == 1
        assert p['interrupted_candidate']['completed_cases'][0]['case_id'] == ids[0]
        assert p['interrupted_candidate']['not_completed_case_ids'] == [ids[1]]
        assert 'primary_error' not in p['interrupted_candidate']
        assert p['unattempted_candidate_indices'] == [2]
        status = client.evaluation_status(error.request_id)
        assert status['state'] == 'failed' and status['finished']
        assert status['evaluation_progress'] == p
        public = json.dumps(p)
        assert PRIVATE not in public
        for record in records:
            assert record.case.case_id not in public
        for token in ('prediction', 'truth', 'points', 'axes'):
            assert token not in public
        assert 'unattempted' not in evaluation.calls
        # An ordinary candidate failure must not poison the next evaluation.
        assert len(client.evaluate({'artifact_id': 'recovered'}, case_ids=ids).cases) == 2


@pytest.mark.parametrize(('error_type', 'code', 'status'), [
    (CandidateExecutionTimeout, 'candidate_timeout', 408),
    (CandidateExecutionBatchTimeout, 'evaluation_batch_timeout', 408),
    (CandidateExecutionError, 'candidate_execution_failed', 422),
    (CandidateExecutionCancelled, 'evaluation_cancelled', 409),
    (CandidateExecutionInfrastructureError, 'evaluation_infrastructure_failed', 500),
    (ValueError, 'evaluation_scoring_failed', 500),
])
def test_rpc_error_classification_does_not_export_private_exception(tmp_path, error_type, code, status):
    evaluator, _, _ = make_evaluator(error_type)
    with client_for(tmp_path, evaluator, expected_cleanup_failure=(error_type is CandidateExecutionInfrastructureError)) as client:
        with pytest.raises(AgentCapabilityClientError) as caught:
            client.evaluate({'artifact_id': 'bad'}, case_ids=tuple(c.case_id for c in evaluator.cases))
        assert caught.value.code == code
        assert caught.value.http_status == status
        assert PRIVATE not in str(caught.value) + json.dumps(caught.value.details)
        assert len(caught.value.details['interrupted_candidate']['completed_cases']) == 1


@pytest.mark.parametrize("bundle_error", [PythonCandidateError, CandidateBundleError])
def test_invalid_bundle_has_actionable_hint_and_does_not_solve(tmp_path, bundle_error):
    evaluator, evaluation, _ = make_evaluator()
    def decode(*args):
        raise bundle_error(PRIVATE)
    with client_for(tmp_path, evaluator, decode) as client:
        with pytest.raises(AgentCapabilityClientError) as caught:
            client.evaluate({'artifact_id': 'bad'}, case_ids=tuple(c.case_id for c in evaluator.cases))
        assert caught.value.code == 'candidate_bundle_invalid'
        assert caught.value.http_status == 422
        assert '.orig' in str(caught.value)
        assert PRIVATE not in str(caught.value)
        assert evaluation.calls == []


def test_python_evaluator_retains_progress_on_original_error():
    evaluator, _, _ = make_evaluator()
    with pytest.raises(CandidateExecutionTimeout) as caught:
        evaluator.evaluate_many((artifact('good'), artifact('bad')),
                                EvaluationRequest(tuple(c.case_id for c in evaluator.cases)))
    assert caught.value.evaluation_progress['completed_candidates'][0]['index'] == 0
    assert 'primary_error' not in caught.value.evaluation_progress['interrupted_candidate']


def test_cli_failure_is_json_and_nonzero(monkeypatch, capsys):
    from pde_agent_verifier_solver.engine import agent_tool
    def failing(argv):
        raise AgentCapabilityClientError('candidate_timeout', 'Solve budget exceeded',
                                         http_status=408, request_id='request-1', details={'complete': False})
    monkeypatch.setattr(agent_tool, '_main', failing)
    assert agent_tool.main(['evaluate']) == 1
    row=json.loads(capsys.readouterr().out)
    assert row['status']=='failed'
    assert row['request_id']=='request-1'
    assert row['error']['code']=='candidate_timeout'
    assert row['evaluation_progress']['complete'] is False
