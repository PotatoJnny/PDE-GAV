"""A broken common evaluator must not produce a completed solver comparison."""
import json
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.engine import common_evaluation as common
from pde_agent_verifier_solver.engine.candidate_execution import (
    CandidateExecutionInfrastructureError, CandidateExecutionTimeout,
)
from pde_agent_verifier_solver.plugins.toy import SineTruth
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, Artifact, Case, CaseRecord, CaseSource, CaseVisibility,
    EvaluationPhase, EvaluationPlan, TaskKind,
)


def _inputs():
    artifact = Artifact("program", TaskKind.SOLVER, "Program", object(), "test", "test", 1, "agent")
    query = AnswerQuery.at_points(fields=("u",), points=({"x": .25},))
    programs = tuple(common.CommonProgram(method, "run", 101, artifact.artifact_id, artifact, None)
                     for method in ("first-method", "next-method"))
    cases = tuple(common.CommonCase("held-out", "toy", CaseRecord(
        Case(name, CaseSource.GENERATED, CaseVisibility.HIDDEN, {}), SineTruth(1., 1),
    ), EvaluationPlan(EvaluationPhase.TERMINAL, query)) for name in ("good", "bad", "later"))
    return programs, cases


@pytest.mark.parametrize("kind", ["infrastructure", "timeout", "attached_cleanup"])
def test_common_evaluation_distinguishes_solver_failure_from_service_failure(tmp_path, monkeypatch, kind):
    calls = []
    secret = "private numerical service path"
    class Verifier:
        def __init__(self, *args, **kwargs):
            pass
        def evaluate(self, artifact, records):
            case_id = records[0].case.case_id
            calls.append(case_id)
            if case_id == "bad":
                if kind == "infrastructure":
                    raise CandidateExecutionInfrastructureError(secret)
                error = CandidateExecutionTimeout(secret)
                if kind == "attached_cleanup":
                    error._candidate_cleanup_errors = (CandidateExecutionInfrastructureError(secret),)
                raise error
            return SimpleNamespace(cases=(SimpleNamespace(metrics={"relative_l2": .25}),))
    monkeypatch.setattr(common, "HiddenVerifier", Verifier)
    programs, cases = _inputs()
    kwargs = dict(task_id="toy", programs=programs, cases=cases, evaluation=None,
                  evaluation_settings={}, candidate_execution=None, metrics=("relative_l2",),
                  output_directory=tmp_path, max_workers=1)
    if kind == "timeout":
        summary = common.evaluate_common_programs(**kwargs)
        assert summary["status"] == "completed" and summary["n_failed_program_cases"] == 2
        assert calls == ["good", "bad", "later"] * 2
    else:
        with pytest.raises(CandidateExecutionInfrastructureError, match="no final comparison"):
            common.evaluate_common_programs(**kwargs)
        assert calls == ["good", "bad"]
        assert not (tmp_path / "summary.json").exists()
        failure = json.loads((tmp_path / "failure.json").read_text())
        assert failure["status"] == "interrupted"
        assert failure["n_completed_program_cases"] == 1
        assert secret not in json.dumps(failure)
    public = (tmp_path / "cases.jsonl").read_text()
    assert secret not in public
    rows = [json.loads(line) for line in public.splitlines()]
    assert len(rows) == (6 if kind == "timeout" else 1)
    assert rows[0]["case_id"] == "good" and rows[0]["error"] == .25
    private = (tmp_path / "controller_private" / "common_evaluation_failures.jsonl").read_text()
    assert secret in private and "Traceback" in private
