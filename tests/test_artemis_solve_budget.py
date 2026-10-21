"""Frozen Candidate deadlines and Backend guards share the charged clock."""

import time

import pytest

from pde_agent_verifier_solver.engine import PythonCandidateCodec
from pde_agent_verifier_solver.plugins.vp_1d1v import ArtemisObservableTask
from pde_agent_verifier_solver.plugins.vp_1d1v import task as task_module
from pde_agent_verifier_solver.sdk import (
    EvaluationPhase, EvaluationPlan, TaskKind, inspect_candidate_bundle,
    write_candidate_bundle_manifest,
)
from pde_agent_verifier_solver.sdk import solve_budget as clocks
from tests.test_artemis_canonical import _answer, _binding, _query, _request


SOURCE = '''from time import perf_counter

def prepare_solution(case, solve_budget=None):
    began = perf_counter()
    advance()
    if cooperative and solve_budget is not None and perf_counter() - began > float(solve_budget):
        raise TimeoutError("legacy candidate preparation budget")
    return make_handle()
'''


@pytest.mark.parametrize("work, succeeds, cooperative", [
    (2., True, True), (4., False, True), (2., True, False), (4., False, False),
])
def test_frozen_legacy_deadline_excludes_wait_not_work(tmp_path, monkeypatch, work, succeeds, cooperative):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "submission.py").write_text(SOURCE)
    write_candidate_bundle_manifest(bundle, files=("submission.py",))
    artifact = PythonCandidateCodec(task_kind=TaskKind.OBSERVABLE_MODEL).bind_existing(
        bundle, storage_root=tmp_path / "frozen", artifact_id="candidate",
        name="Frozen deadline", summary="Budget compatibility", usage="prepare then query",
    )
    module = artifact.program._module()
    original_time = time.perf_counter
    digest = artifact.program.tree_sha256
    now = [0.]
    monkeypatch.setattr(clocks, "perf_counter", lambda: now[0])
    monkeypatch.setattr(task_module, "perf_counter", lambda: now[0])

    def advance():
        now[0] += 100. + work
        clocks._record_oracle_response_wait("trusted-test-request", {
            "schema": clocks.ORACLE_WAIT_TIMING_SCHEMA,
            "queue_wait_seconds": 80., "lock_wait_seconds": 20.,
        }, rpc_wall_seconds=100. + work)

    class Handle:
        def evaluate(self, query):
            assert module.perf_counter is original_time  # preparation scope only
            return _answer(query)

    module.advance = advance
    module.make_handle = Handle
    module.cooperative = cooperative
    binding, _ = _binding()
    generator = binding.plugin.generator
    case = generator.sample(generator.load(binding.checkpoint, {}), _request())[0].case
    evaluation = ArtemisObservableTask().create_evaluation({
        "default_query": _query(), "candidate_compute_budget": 3.,
        "require_prepared_solution": True,
    })
    plan = EvaluationPlan(EvaluationPhase.OFFICIAL, _query())
    if succeeds:
        output = evaluation.execute_candidate(artifact, case, plan)
        assert output.metrics["candidate_solve_wall_seconds"] == 102.
        assert output.metrics["candidate_solve_active_seconds"] == 2.
        assert output.metrics["candidate_platform_wait_seconds"] == 100.
        assert output.metrics["candidate_budget_clock_compat_aliases"] == 1.
        assert output.prediction == _answer(_query())
    else:
        with pytest.raises(TimeoutError, match="legacy candidate|solve budget"):
            evaluation.execute_candidate(artifact, case, plan)
    assert time.perf_counter is original_time
    assert module.perf_counter is original_time
    assert inspect_candidate_bundle(artifact.program.bundle_locator).tree_sha256 == digest
    assert clocks.candidate_solve_timing()["candidate_platform_wait_seconds"] == 0.
