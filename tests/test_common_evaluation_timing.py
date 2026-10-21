"""Shared cost export accepts both explicit-active and legacy charged-wall Tasks."""

import csv
import json
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.engine import common_evaluation
from pde_agent_verifier_solver.plugins.toy import SineTruth
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, Artifact, Case, CaseRecord, CaseSource, CaseVisibility,
    EvaluationPhase, EvaluationPlan, TaskKind,
)


@pytest.mark.parametrize("timing, expected_solve, expected_wait", [
    ({"candidate_solve_wall_seconds": 7.0, "candidate_solve_active_seconds": 4.0,
      "candidate_platform_wait_seconds": 3.0}, 4.0, 3.0),
    ({"candidate_solve_wall_seconds": 3.0, "candidate_solve_active_seconds": 0.0,
      "candidate_platform_wait_seconds": 3.0}, 0.0, 3.0),
    ({"candidate_solve_wall_seconds": 4.0, "candidate_solve_queue_wait_seconds": 3.0},
     4.0, 3.0),
    ({"candidate_solve_wall_seconds": 4.0, "candidate_solve_queue_wait_seconds": 0.0,
      "candidate_platform_wait_seconds": 3.0}, 4.0, 0.0),
    ({}, None, None),
])
def test_common_export_prefers_explicit_active_without_double_subtraction(
    tmp_path, monkeypatch, timing, expected_solve, expected_wait,
):
    measured = {"relative_l2": 0.25, "candidate_query_wall_seconds": 0.5,
                "candidate_query_queue_wait_seconds": 0.125, **timing}

    class Verifier:
        def __init__(self, *args, **kwargs):
            pass

        def evaluate(self, artifact, records):
            return SimpleNamespace(cases=(SimpleNamespace(metrics=measured),))

    monkeypatch.setattr(common_evaluation, "HiddenVerifier", Verifier)
    query = AnswerQuery.at_points(fields=("u",), points=({"x": 0.25},))
    case = Case("timing-case", CaseSource.GENERATED, CaseVisibility.HIDDEN, {})
    artifact = Artifact("program", TaskKind.SOLVER, "Program", object(),
                        "Test program", "test only", 1, "test-agent")
    summary = common_evaluation.evaluate_common_programs(
        task_id="generic-task",
        programs=(common_evaluation.CommonProgram("method", "run", 101,
                  artifact.artifact_id, artifact, None),),
        cases=(common_evaluation.CommonCase("held-out", "toy",
               CaseRecord(case, SineTruth(1.0, 1)),
               EvaluationPlan(EvaluationPhase.TERMINAL, query)),),
        evaluation=None, evaluation_settings={}, candidate_execution=None,
        metrics=("relative_l2",), output_directory=tmp_path,
    )
    rows = [json.loads(line) for line in (tmp_path / "cases.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["solve_wall_seconds"] == expected_solve
    assert row["solve_queue_wait_seconds"] == expected_wait
    assert row["query_wall_seconds"] == 0.5
    assert row["query_queue_wait_seconds"] == 0.125
    assert row["status"] == "ok" and row["error"] == 0.25
    assert summary["groups"][0]["mean_success"] == 0.25
    assert row["case_wall_seconds"] >= 0
    with (tmp_path / "cases.csv").open() as stream:
        exported = next(csv.DictReader(stream))
    assert exported["solve_wall_seconds"] == ("" if expected_solve is None else str(expected_solve))
    assert exported["solve_queue_wait_seconds"] == ("" if expected_wait is None else str(expected_wait))
