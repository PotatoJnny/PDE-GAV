"""Real failing NS solver processes must return actionable errors to their Agent."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.engine.agent_capability import (
    AgentCapabilityClientError, ControllerAgentCapabilityClient, ControllerAgentCapabilityFacade,
)
from pde_agent_verifier_solver.engine.agent_execution import CommandAgentProgram
from pde_agent_verifier_solver.engine.campaign import EvaluatorPolicy
from pde_agent_verifier_solver.engine.candidate_diagnostics import (
    candidate_exception_diagnostic, normalize_candidate_diagnostic,
)
from pde_agent_verifier_solver.engine.candidate_execution import (
    CANDIDATE_WORKER_RESPONSE_SCHEMA, PythonTaskWorkerSpec, SubprocessCandidateExecutionService,
    _failed_worker_code,
)
from pde_agent_verifier_solver.engine.evaluator import AgentEvaluator
from pde_agent_verifier_solver.engine.python_candidate import PythonCandidateCodec
from pde_agent_verifier_solver.plugins.ns import NSSolverTask
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, AnswerView, Case, CaseRecord, CaseSource, CaseVisibility,
    DevelopmentCoordinateMode, QueryKind, TaskKind,
)

GOOD = '''from pde_agent_verifier_solver.sdk import AnswerView, QueryKind
class Handle:
    def evaluate(self, query):
        n = len(query.points)
        return AnswerView(kind=QueryKind.POINTS, fields={"velocity_x": (1.0,)*n,
            "velocity_y": (2.0,)*n}, points=query.points, shape=(n,))
def prepare_solution(case, solve_budget=None):
    return Handle()
'''


class Truth:
    def answer(self, query):
        n = len(query.points)
        return AnswerView(kind=QueryKind.POINTS, fields={"velocity_x": (1.0,)*n,
            "velocity_y": (2.0,)*n}, points=query.points, shape=(n,))


@contextmanager
def real_client(tmp_path):
    query = AnswerQuery.at_points(fields=('velocity_x', 'velocity_y'),
        points=({'t': 0.1, 'x': 0.2, 'y': 0.3}, {'t': 0.2, 'x': 0.4, 'y': 0.5}))
    settings = {'default_query': query, 'primary_metric': 'velocity_relative_l2', 'include_pressure': False}
    service = SubprocessCandidateExecutionService(tmp_path/'execution',
        task=PythonTaskWorkerSpec.from_task_type(NSSolverTask), evaluation_settings=settings,
        max_workers=1, case_hard_wall_seconds=30)
    record = CaseRecord(Case('controller-private-case', CaseSource.GENERATED,
        CaseVisibility.DEVELOPMENT, {'equation': 'NS'}), Truth())
    evaluator = AgentEvaluator((record,), NSSolverTask(), EvaluatorPolicy(query,
        development_coordinate_mode=DevelopmentCoordinateMode.USER_FIXED_QUERY),
        settings, candidate_execution=service)
    context = SimpleNamespace(evaluator=evaluator, sampler=None, cases=evaluator.cases,
        round_plan=SimpleNamespace(round_index=1), agent=SimpleNamespace(agent_id='debug-agent'), incumbent=None)
    workspace = tmp_path/'agent'
    workspace.mkdir()
    decoder = PythonCandidateCodec(task_kind=TaskKind.SOLVER).decode_nomination
    lease = ControllerAgentCapabilityFacade().open(context, workspace, 'debug-run',
        CommandAgentProgram((sys.executable, '-c', 'pass'), decoder))
    try:
        yield ControllerAgentCapabilityClient(lease.manifest, workspace=workspace), workspace, evaluator
    finally:
        lease.close()


@pytest.mark.parametrize(('source', 'error_type', 'fragment', 'function'), [
    ('def step():\n    raise ValueError("matrix dimensions 3 and 5 do not match")\n'
     'def prepare_solution(case, solve_budget=None):\n    return step()\n',
     'ValueError', 'matrix dimensions 3 and 5', 'step'),
    ('import pde_gav_dependency_that_does_not_exist\n'
     'def prepare_solution(case, solve_budget=None):\n    return None\n',
     'ModuleNotFoundError', 'pde_gav_dependency_that_does_not_exist', '<module>'),
    ('class Handle:\n    def evaluate(self, query):\n        raise KeyError("missing_velocity")\n'
     'def prepare_solution(case, solve_budget=None):\n    return Handle()\n',
     'KeyError', 'missing_velocity', 'evaluate'),
])
def test_real_worker_exception_reaches_rpc_status_and_cli(tmp_path, monkeypatch, capsys,
                                                        source, error_type, fragment, function):
    from pde_agent_verifier_solver.engine import agent_tool
    with real_client(tmp_path) as (client, workspace, evaluator):
        (workspace/'bad.py').write_text(source)
        (workspace/'good.py').write_text(GOOD)
        ids = tuple(c.case_id for c in evaluator.cases)
        with pytest.raises(AgentCapabilityClientError) as caught:
            client.evaluate_many([{'bundle_path': 'good.py'}, {'bundle_path': 'bad.py'}], case_ids=ids)
        error = caught.value
        assert error.code == 'candidate_execution_failed'
        diagnostic = error.diagnostic
        assert diagnostic['exception_type'] == error_type
        assert fragment in diagnostic['message'] and fragment in str(error)
        assert 'Traceback' in diagnostic['traceback']
        assert fragment in diagnostic['traceback']
        assert any(f['function'] == function and f['line'] > 0 and f['source'] for f in diagnostic['frames'])
        assert diagnostic['case_id'] == ids[0]
        assert 'controller-private-case' not in json.dumps(diagnostic)
        assert len(error.details['completed_candidates']) == 1
        assert 'primary_error' not in error.details['interrupted_candidate']
        status = client.evaluation_status(error.request_id)
        assert status['evaluation_progress']['interrupted_candidate']['diagnostic'] == diagnostic
        def fail_cli(argv):
            raise error
        monkeypatch.setattr(agent_tool, '_main', fail_cli)
        assert agent_tool.main(['evaluate']) == 1
        cli = json.loads(capsys.readouterr().out)
        assert cli['error']['diagnostic'] == diagnostic
        # A failed candidate must not quarantine a healthy execution service.
        assert client.evaluate({'bundle_path': 'good.py'}, case_ids=ids).primary_error == 0.0


def test_candidate_syntax_error_returns_exact_source_location(tmp_path):
    with real_client(tmp_path) as (client, workspace, evaluator):
        (workspace/'bad.py').write_text('def prepare_solution(case, solve_budget=None)\n    return None\n')
        with pytest.raises(AgentCapabilityClientError) as caught:
            client.evaluate({'bundle_path': 'bad.py'}, case_ids=tuple(c.case_id for c in evaluator.cases))
        error = caught.value
        assert error.code == 'candidate_bundle_invalid'
        assert error.diagnostic['exception_type'] == 'SyntaxError'
        frame = error.diagnostic['frames'][-1]
        assert frame['line'] == 1 and 'prepare_solution' in frame['source']
        assert "expected ':'" in error.diagnostic['message']
        assert client.evaluation_status(error.request_id)['evaluation_progress']['diagnostic'] == error.diagnostic


def test_mismatched_worker_envelope_never_exports_diagnostics(tmp_path):
    response = tmp_path/'response.json'
    payload = {'schema': CANDIDATE_WORKER_RESPONSE_SCHEMA, 'execution_id': 'another-request',
        'case_index': 0, 'case_id': 'another-case', 'status': 'failed', 'error_code': 'failed',
        'candidate_diagnostic': {'origin': 'candidate', 'exception_type': 'RuntimeError',
            'message': 'belongs-to-another-request', 'traceback': 'private', 'frames': []}}
    response.write_text(json.dumps(payload))
    diagnostic = {}
    assert _failed_worker_code(response, execution_id='expected', case_index=0, case_id='case',
        diagnostic=diagnostic) == 'candidate_worker_failed_with_invalid_response'
    assert diagnostic == {}


def test_long_exception_is_bounded_and_never_captures_locals(tmp_path):
    private_local_value = 'DO_NOT_EXPORT_FRAME_LOCALS'
    try:
        raise RuntimeError('x' * 60000)
    except RuntimeError as error:
        diagnostic = normalize_candidate_diagnostic(candidate_exception_diagnostic(error, bundle_root=tmp_path))
    assert '[diagnostic truncated]' in diagnostic['message']
    assert len(json.dumps(diagnostic)) < 45000
    assert private_local_value not in json.dumps(diagnostic)


def test_deep_solver_traceback_keeps_actual_error_line(tmp_path):
    def actual_failure():
        raise ValueError("failure at the deepest solver step")
    def recurse(depth):
        if depth:
            return recurse(depth - 1)
        return actual_failure()
    try:
        recurse(50)
    except ValueError as error:
        diagnostic = candidate_exception_diagnostic(error, bundle_root=tmp_path)
    assert len(diagnostic['frames']) == 32
    assert diagnostic['frames'][-1]['function'] == 'actual_failure'
    assert 'raise ValueError' in diagnostic['frames'][-1]['source']
    assert 'in actual_failure' in diagnostic['traceback']
