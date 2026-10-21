"""Synthetic timing contract only; no physical simulation or CUDA calls."""
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.plugins.vpfp.tasks.observable import VPFPObservableTask
from pde_agent_verifier_solver.sdk import EvaluationPhase, EvaluationPlan
from pde_agent_verifier_solver.sdk import solve_budget as clocks
from tests.test_vpfp_split_candidate_execution import CASE, _query, _view, _artifact


def _plan(**settings):
    return EvaluationPlan(EvaluationPhase.OFFICIAL, _query(), settings={
        'candidate_budget_mode': 'ns_remaining_case_v1', **settings})


def test_preparation_longer_than_legacy_cap_uses_same_remaining_case(monkeypatch):
    now, received = [0.], []
    monkeypatch.setattr(clocks, 'monotonic', lambda: now[0])
    monkeypatch.setattr(clocks, 'perf_counter', lambda: now[0])

    class Program:
        def prepare_solution(self, case, solve_budget=None):
            received.append(solve_budget)
            now[0] += 1000.
            clocks._record_oracle_response_wait('prepare', {
                'schema': clocks.ORACLE_WAIT_TIMING_SCHEMA,
                'queue_wait_seconds': 280., 'lock_wait_seconds': 20.}, rpc_wall_seconds=1000.)
            def evaluate(query):
                received.append(clocks.candidate_solve_remaining_seconds())
                now[0] += 20.
                return _view(query)
            return SimpleNamespace(evaluate=evaluate)

    with clocks.bind_candidate_solve_budget(new_case=True, budget_seconds=900.) as meter:
        now[0] = 30.
        VPFPObservableTask().execute_candidate(_artifact(Program()), CASE, _plan())
        assert received == [870., 170.]
        assert clocks._current_candidate_solve_budget_meter() is meter
        assert clocks.candidate_solve_remaining_seconds() == 150.


@pytest.mark.parametrize('budget', [180, 900, False, float('inf')])
def test_shared_budget_rejects_separate_preparation_limit(budget):
    with pytest.raises(ValueError, match='forbids a separate'):
        VPFPObservableTask().execute_candidate(_artifact(object()), CASE,
            _plan(candidate_compute_budget=budget))


def test_exhausted_shared_case_never_enters_preparation(monkeypatch):
    now = [0.]
    monkeypatch.setattr(clocks, 'monotonic', lambda: now[0])
    def forbidden(*args):
        pytest.fail('exhausted Case must not call Candidate')
    with clocks.bind_candidate_solve_budget(new_case=True, budget_seconds=900.):
        now[0] = 901.
        with pytest.raises(TimeoutError, match='exhausted before preparation'):
            VPFPObservableTask().execute_candidate(
                _artifact(SimpleNamespace(prepare_solution=forbidden)), CASE, _plan())


def test_legacy_budget_and_brief_remain_unchanged():
    received = []
    def prepare(case, solve_budget=None):
        received.append(solve_budget)
        return SimpleNamespace(evaluate=_view)
    task = VPFPObservableTask()
    task.execute_candidate(_artifact(SimpleNamespace(prepare_solution=prepare)), CASE,
        EvaluationPlan(EvaluationPhase.OFFICIAL, _query(), settings={'candidate_compute_budget': 180}))
    assert received == [180]
    assert 'when absent this Task passes solve_budget=None' in task.agent_brief({})
    brief = task.agent_brief({'candidate_budget_mode':'ns_remaining_case_v1'})
    assert 'no separate preparation limit' in brief and '900-second charged Case' in brief
    assert 'when absent this Task passes solve_budget=None' not in brief
