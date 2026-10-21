"""Hidden evaluator errors remain diagnosable without becoming solver scores."""
from dataclasses import replace
import json
import stat
from threading import Lock
from types import SimpleNamespace

import pytest

import pde_agent_verifier_solver.engine.unified_runtime as runtime
from pde_agent_verifier_solver.engine.candidate_execution import (
    CandidateExecutionInfrastructureError, CandidateExecutionTimeout,
)
from pde_agent_verifier_solver.engine.evaluator import HiddenVerifier
from pde_agent_verifier_solver.engine.memory import CandidateHistory
from pde_agent_verifier_solver.engine.runtime import EventStream
from pde_agent_verifier_solver.engine.seeds import SeedLedger
from pde_agent_verifier_solver.panel.controller import build_toy_panel
from pde_agent_verifier_solver.plugins.toy import (
    ToyPoissonProblem, ToySineSolver, ToySolverEvaluation, ToySolverTask,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, Artifact, ArtifactEvaluation, CaseEvaluation, DatasetInput,
    DatasetKind, EvaluationPhase, TaskKind, TerminalMetricSpec,
)


def _artifact(name):
    return Artifact(name, TaskKind.SOLVER, name, ToySineSolver(64), "test", "solve", 1, "test-agent")


def _controller(tmp_path, monkeypatch, operation, *, overall=False):
    class Verifier:
        def __init__(self, *args, **kwargs):
            pass

        def evaluate(self, artifact, records):
            operation(artifact, records)
            return ArtifactEvaluation(
                artifact, .1, {}, (CaseEvaluation(records[0].case.case_id, .1, {}),),
            )

    controller = object.__new__(runtime.CampaignController)
    controller.campaign = SimpleNamespace(
        promotion={"preset": "minimize_metric", "settings": {
            "primary": {"distribution": "overall" if overall else "base", "statistic": "mean"},
        }},
        terminal_evaluation=SimpleNamespace(ranking=TerminalMetricSpec("base", "base")),
        evaluator=SimpleNamespace(default_query=None), task_settings={},
    )
    controller._loaded_generator_state = object()
    controller.evaluation = controller.generator = None
    controller.candidate_execution = SimpleNamespace(root=tmp_path, audits=())
    controller._generator_operation_lock = Lock()
    controller._seeds = SeedLedger(101)
    controller.events = EventStream()
    controller._active_round_plan = SimpleNamespace(round_index=1)
    controller._candidate_history = CandidateHistory()
    monkeypatch.setattr(runtime, "HiddenVerifier", Verifier)
    hidden = {"base": (SimpleNamespace(case=SimpleNamespace(case_id="private-base")),)}
    return controller, hidden


def _failures(tmp_path):
    return [json.loads(line) for line in (tmp_path / "hidden_evaluation_failures.jsonl").read_text().splitlines()]


def test_candidate_timeout_is_retained_privately_and_next_candidate_is_evaluated(tmp_path, monkeypatch, caplog):
    secret = "private-oracle-path hidden-case-72"
    calls = []

    def operation(artifact, records):
        calls.append(artifact.artifact_id)
        if artifact.artifact_id == "slow":
            raise CandidateExecutionTimeout(secret)

    controller, hidden = _controller(tmp_path, monkeypatch, operation)
    controller.candidate_execution.audits = (
        SimpleNamespace(artifact_id="slow", phase="official", execution_id="audit-1"),
        SimpleNamespace(artifact_id="different", phase="official", execution_id="unrelated"),
    )
    reports, evaluations = controller._official_reports_in_scope((_artifact("slow"), _artifact("good")), hidden)
    assert calls == ["slow", "good"]
    assert reports[0].failures == ("base:evaluation_failed",)
    assert reports[1].technically_eligible
    assert evaluations["slow"] == {}
    row, = _failures(tmp_path)
    assert row["exception_type"] == "CandidateExecutionTimeout"
    assert secret in row["traceback"]
    assert secret in caplog.text
    assert row["available_execution_ids_for_candidate_phase"] == ["audit-1"]
    assert row["stage"] == "evaluate"
    assert row["round_index"] == 1
    assert secret not in repr(controller.events.snapshot()) + repr(reports)
    assert stat.S_IMODE((tmp_path / "hidden_evaluation_failures.jsonl").stat().st_mode) == 0o600


@pytest.mark.parametrize("attached", [False, True])
def test_infrastructure_failure_stops_before_untested_s0_and_preserves_completed_candidate(tmp_path, monkeypatch, attached):
    calls = []

    def operation(artifact, records):
        calls.append(artifact.artifact_id)
        if artifact.artifact_id == "failure":
            infra = CandidateExecutionInfrastructureError("private session drain failure")
            if attached:
                timeout = CandidateExecutionTimeout("first candidate failure")
                timeout._candidate_cleanup_errors = (infra,)
                raise timeout
            raise infra

    controller, hidden = _controller(tmp_path, monkeypatch, operation)
    artifacts = tuple(_artifact(name) for name in ("good", "failure", "s0"))
    for artifact in artifacts:
        controller._candidate_history.add_candidate(artifact)
    with pytest.raises(runtime.HiddenEvaluationInfrastructureError) as caught:
        controller._official_reports_in_scope(artifacts, hidden, seed_namespace="official_round:1")
    assert calls == ["good", "failure"]
    assert "private session" not in str(caught.value)
    history = {node.artifact_id: node for node in controller._candidate_history.snapshot().nodes}
    assert history["good"].status == "eligible"
    assert history["good"].official_distributions["base"].mean == .1
    assert history["failure"].status == history["s0"].status == "not_evaluated"
    row, = _failures(tmp_path)
    assert row["reason_code"] == "hidden_evaluation_infrastructure_failure"
    assert "private session drain failure" in json.dumps(row)
    assert runtime._public_round_abort_code(caught.value) == "hidden_evaluation_infrastructure_failure"


@pytest.mark.parametrize("overall", [False, True])
def test_controller_summary_exception_is_infrastructure_and_retains_stage(tmp_path, monkeypatch, overall):
    controller, hidden = _controller(tmp_path, monkeypatch, lambda *args: None, overall=overall)
    def fail(*args):
        raise RuntimeError("private summary implementation failure")
    monkeypatch.setattr(controller, "_overall_evaluation" if overall else "_metric_summary", fail)
    with pytest.raises(runtime.HiddenEvaluationInfrastructureError):
        controller._official_reports_in_scope((_artifact("good"),), hidden)
    row, = _failures(tmp_path)
    assert row["stage"] == ("overall_summary" if overall else "metric_summary")
    assert "private summary implementation failure" in row["traceback"]


@pytest.mark.parametrize("failure_stage", ["make_plan", "truth", "aggregate"])
def test_hidden_preparation_and_aggregate_errors_are_not_candidate_failures(monkeypatch, failure_stage):
    dataset = ToyPoissonProblem().prepare_dataset(DatasetInput("dataset", DatasetKind.NUMERICAL))
    verifier = HiddenVerifier(ToySolverTask(), AnswerQuery.grid(fields=("u",), x=17))
    def fail(*args):
        raise RuntimeError("private controller failure")
    if failure_stage == "make_plan":
        monkeypatch.setattr(verifier, "_plan_for", fail)
    elif failure_stage == "truth":
        monkeypatch.setattr(verifier, "_truth_for", fail)
    else:
        monkeypatch.setattr(verifier.evaluation_program, "aggregate", fail)
    with pytest.raises(CandidateExecutionInfrastructureError) as caught:
        verifier.evaluate(_artifact("good"), dataset.hidden_cases[:1])
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert caught.value._hidden_evaluation_stage == (
        "aggregate_case_scores" if failure_stage == "aggregate" else "prepare_hidden_inputs"
    )


def test_scorer_validation_failure_keeps_exact_stage_and_original_error(monkeypatch):
    dataset = ToyPoissonProblem().prepare_dataset(DatasetInput("dataset", DatasetKind.NUMERICAL))
    class SplitEvaluation(ToySolverEvaluation):
        supports_split_candidate_execution = True
        def execute_candidate(self, *args):
            raise AssertionError("subprocess service should execute candidate")
        def score_candidate(self, *args):
            raise ValueError("candidate output has invalid shape")
    service = SimpleNamespace(execute_cases=lambda *args, **kwargs: (object(),))
    query = AnswerQuery.grid(fields=("u",), x=17)
    verifier = HiddenVerifier(SplitEvaluation(query), query, candidate_execution=service)
    with pytest.raises(ValueError, match="invalid shape") as caught:
        verifier.evaluate(_artifact("bad-shape"), dataset.hidden_cases[:1])
    assert caught.value._hidden_evaluation_stage == "score_candidate"
    assert not isinstance(caught.value, CandidateExecutionInfrastructureError)


def test_s0_without_current_report_blocks_promotion_explicitly(monkeypatch):
    original = HiddenVerifier.evaluate
    def evaluate(self, artifact, records):
        if artifact.agent_id == "user":
            raise CandidateExecutionTimeout("private initial candidate timeout")
        return original(self, artifact, records)
    monkeypatch.setattr(HiddenVerifier, "evaluate", evaluate)
    panel = build_toy_panel()
    launched = panel.create_and_launch({
        "project_name": "Missing incumbent report",
        "round_count": 1, "agents": ["careful-agent"],
        "initial_candidate": replace(_artifact("s0"), agent_id="user", round_index=0),
        "allowed_sources": ["base"], "starter_cases": {"base": 1},
        "generator_settings": {"guidance_enabled": False, "perturbations_enabled": False},
        "round_hidden": {"base": 1}, "terminal_hidden": {"base": 1},
    })
    panel.app.wait(launched["run_id"], timeout=5)
    page = panel.run_page(launched["run_id"])
    assert page["status"] == "failed"
    assert "IncumbentEvaluationUnavailable" in page["error"]
    assert "no prior score was reused" in page["error"]
    assert "incumbent_evaluation_unavailable" in page["memory"]["markdown"]
    assert "private initial candidate timeout" not in page["memory"]["markdown"]
    run = panel.app.get_run(launched["run_id"])
    event = next(row for row in run.events.snapshot() if row.event_type == "promotion_blocked")
    assert event.payload["candidate_id"] == "candidate-s0"
    assert not any(row.event_type == "round_completed" for row in run.events.snapshot())
