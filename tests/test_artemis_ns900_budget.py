"""Shared-case budget mode ported from the deployed Artemis NS900 run.

All large durations below are synthetic clocks, not physics or timing claims.
Legacy preparation-budget tests remain in test_artemis_solve_budget.py.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.plugins.vp_1d1v import ArtemisObservableTask
from pde_agent_verifier_solver.plugins.vp_1d1v import task as task_module
from pde_agent_verifier_solver.sdk import EvaluationPhase, EvaluationPlan
from pde_agent_verifier_solver.sdk import solve_budget as clocks
from tests.test_artemis_canonical import _answer, _artifact, _binding, _query, _request


def _evaluation(**settings):
    return ArtemisObservableTask().create_evaluation({
        "default_query": _query(), "require_prepared_solution": True,
        "candidate_budget_mode": "ns_remaining_case_v1", **settings})


def _case():
    binding, _ = _binding()
    generator = binding.plugin.generator
    return generator.sample(generator.load(binding.checkpoint, {}), _request())[0].case


def test_long_preparation_and_query_share_remaining_case_budget(monkeypatch):
    now, received = [0.], []
    monkeypatch.setattr(clocks, "perf_counter", lambda: now[0])
    monkeypatch.setattr(clocks, "monotonic", lambda: now[0])
    monkeypatch.setattr(task_module, "perf_counter", lambda: now[0])

    class Program:
        def prepare_solution(self, case, solve_budget=None):
            received.append(solve_budget)
            now[0] += 1000.
            clocks._record_oracle_response_wait("prepare", {
                "schema": clocks.ORACLE_WAIT_TIMING_SCHEMA,
                "queue_wait_seconds": 280., "lock_wait_seconds": 20.}, rpc_wall_seconds=1000.)

            def evaluate(query):
                received.append(clocks.candidate_solve_remaining_seconds())
                now[0] += 20.
                return _answer(query)

            return SimpleNamespace(evaluate=evaluate)

    with clocks.bind_candidate_solve_budget(new_case=True, budget_seconds=900.) as meter:
        now[0] = 30.
        output = _evaluation().execute_candidate(
            _artifact(Program()), _case(), EvaluationPlan(EvaluationPhase.OFFICIAL, _query()))
        assert received == [870., 170.]
        assert clocks._current_candidate_solve_budget_meter() is meter
        assert clocks.candidate_solve_remaining_seconds() == 150.
        assert output.metrics["candidate_solve_active_seconds"] == 700.
        assert output.metrics["candidate_query_wall_seconds"] == 20.


@pytest.mark.parametrize("budget", [180, 900, float("inf"), False])
def test_shared_case_forbids_an_extra_prepare_cap(budget):
    with pytest.raises(ValueError, match="forbids a separate"):
        _evaluation(candidate_compute_budget=budget)


def test_shared_case_does_not_reset_exhausted_worker_budget(monkeypatch):
    now = [0.]
    monkeypatch.setattr(clocks, "monotonic", lambda: now[0])

    class Program:
        def prepare_solution(self, case, solve_budget=None):
            raise AssertionError("An exhausted case must not enter preparation")

    with clocks.bind_candidate_solve_budget(new_case=True, budget_seconds=900.):
        now[0] = 901.
        with pytest.raises(TimeoutError, match="exhausted before preparation"):
            _evaluation().execute_candidate(
                _artifact(Program()), _case(), EvaluationPlan(EvaluationPhase.OFFICIAL, _query()))


def test_remaining_hint_is_read_after_lazy_candidate_import(monkeypatch):
    now, received = [0.], []
    monkeypatch.setattr(clocks, "monotonic", lambda: now[0])

    class Program:
        @contextmanager
        def _solve_budget_clock_scope(self):
            now[0] += 25.
            yield ()

        @property
        def prepare_solution(self):
            now[0] += 5.
            return self.prepare

        def prepare(self, case, solve_budget=None):
            received.append(solve_budget)
            return SimpleNamespace(evaluate=_answer)

    with clocks.bind_candidate_solve_budget(new_case=True, budget_seconds=900.):
        now[0] = 30.
        _evaluation().execute_candidate(
            _artifact(Program()), _case(), EvaluationPlan(EvaluationPhase.OFFICIAL, _query()))
    assert received == [840.]


def test_budget_mode_is_explicit_and_legacy_default_is_unchanged():
    task = ArtemisObservableTask()
    legacy = task.create_evaluation({"default_query": _query(), "candidate_compute_budget": 180})
    assert legacy.candidate_budget_mode is None and legacy.solve_budget == 180
    assert "charged preparation time" in task.agent_brief({})
    brief = task.agent_brief({"candidate_budget_mode": "ns_remaining_case_v1"})
    assert "no separate preparation limit" in brief
    assert "900-second charged Case limit" in brief
    assert "preparation does not reset or extend it" in brief
    with pytest.raises(ValueError, match="unknown Artemis"):
        _evaluation(candidate_budget_mode="unsupported")
