"""A fresh Python process continues the original scientific run and budgets."""
from __future__ import annotations

from collections import Counter
from dataclasses import fields, is_dataclass
from enum import Enum
import json
from pathlib import Path
import subprocess
import sys
from collections.abc import Mapping

import pytest

from pde_agent_verifier_solver.engine.campaign import (
    AgentSpec, CampaignCompiler, CampaignRecipe, CaseCounts,
    DevelopmentSamplingPolicy, EvaluatorPolicy, GeneratorRecipe, RoundSettings,
)
from pde_agent_verifier_solver.engine.campaign_continuation import (
    CampaignContinuation, CampaignPauseControl, CampaignPaused,
)
from pde_agent_verifier_solver.engine.checkpoint_store import CheckpointStore
from pde_agent_verifier_solver.engine.unified_runtime import CampaignController
from pde_agent_verifier_solver.plugins.toy import (
    ToyPoissonGenerator, ToyPoissonProblem, ToyResearchAgent, ToySineSolver,
    ToySolverEvaluation, ToySolverTask,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, Artifact, DatasetInput, DatasetKind, HiddenSuiteSpec,
    OfficialEvaluationSpec, SampleSource, TaskKind, TerminalCohortSpec,
    TerminalEvaluationSpec, TerminalGuidanceSource, TerminalMetricSpec,
)


def _record(path, kind, **values):
    with Path(path).open("a") as stream:
        stream.write(json.dumps({"kind": kind, **values}, sort_keys=True) + "\n")


class RecordingGenerator(ToyPoissonGenerator):
    def __init__(self, log):
        self.log = log

    def train(self, *args, **kwargs):
        _record(self.log, "train")
        return super().train(*args, **kwargs)

    def update_guidance(self, state, feedback, settings, progress=None):
        _record(self.log, "guidance", source_round=feedback.source_round_index)
        return super().update_guidance(state, feedback, settings, progress)

    def sample(self, state, request, progress=None):
        records = super().sample(state, request, progress)
        _record(self.log, "sample", ids=[record.case.case_id for record in records])
        return records

    def perturb(self, state, request, progress=None):
        records = super().perturb(state, request, progress)
        _record(self.log, "perturb", ids=[record.case.case_id for record in records])
        return records


class RecordingAgent(ToyResearchAgent):
    def __init__(self, log, step_size):
        super().__init__(step_size)
        self.log = log

    def run(self, context):
        _record(self.log, "agent", round=context.round_plan.round_index,
                agent=context.agent.agent_id, budget=context.round_plan.agent_budget_seconds)
        return super().run(context)


class RecordingEvaluation(ToySolverEvaluation):
    def __init__(self, query, log):
        super().__init__(query)
        self.log = log

    def evaluate(self, artifact, case, truth, plan):
        _record(self.log, "evaluate", artifact=artifact.artifact_id,
                case=case.case_id, phase=plan.phase.value, seed=plan.seed)
        return super().evaluate(artifact, case, truth, plan)


class RecordingTask(ToySolverTask):
    def __init__(self, log):
        self.log = log

    def create_evaluation(self, settings):
        return RecordingEvaluation(settings["default_query"], self.log)


def _controller(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    log = str(root / "work.jsonl")
    problem, generator = ToyPoissonProblem(), RecordingGenerator(log)
    dataset = problem.prepare_dataset(DatasetInput(
        "continuation-fixture", DatasetKind.NUMERICAL,
        settings={"development_cases": 12, "hidden_cases": 20},
    ))
    initial = Artifact(
        "S0", TaskKind.SOLVER, "S0", ToySineSolver(1), "Starting solver", "test",
        round_index=0, agent_id="baseline", metadata={"max_resolved_mode": 1},
    )
    recipe = CampaignRecipe(
        name="continuation-fixture", round_count=3,
        agents=(AgentSpec("careful", "Careful"), AgentSpec("explorer", "Explorer")),
        default_round=RoundSettings(
            agent_budget_seconds=14400,
            development_sampling=DevelopmentSamplingPolicy(
                max_cases_per_agent_per_round=8,
                allowed_sources=(SampleSource.BASE, SampleSource.GUIDED),
                starter=CaseCounts(base=2),
            ),
            official_evaluation=OfficialEvaluationSpec(HiddenSuiteSpec(
                base=5, guided=3, base_perturbed=2, guided_perturbed=1,
            )),
        ),
        generator=GeneratorRecipe(),
        evaluator=EvaluatorPolicy(AnswerQuery.grid(fields=("u",), x=17)),
        master_seed=101, initial_candidate=initial,
        terminal_evaluation=TerminalEvaluationSpec(
            cohorts=(
                TerminalCohortSpec("clean", HiddenSuiteSpec(base=5, base_perturbed=2)),
                TerminalCohortSpec("round1", HiddenSuiteSpec(guided=3, guided_perturbed=1),
                                   TerminalGuidanceSource.ROUND, round_index=1),
                TerminalCohortSpec("final", HiddenSuiteSpec(guided=3, guided_perturbed=1),
                                   TerminalGuidanceSource.FINAL),
            ),
            ranking=TerminalMetricSpec("clean", "base"),
        ),
    )
    campaign = CampaignCompiler().compile(recipe, generator.capabilities)
    controller = CampaignController(
        campaign, dataset, problem, RecordingTask(log), generator,
        {"careful": RecordingAgent(log, 1), "explorer": RecordingAgent(log, 2)},
    )
    return controller


def _normalize(value):
    if is_dataclass(value):
        return {field.name: _normalize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    if isinstance(value, (str, float, int, bool)) or value is None:
        return value
    raise TypeError(type(value))


def _child(root, target):
    import faulthandler
    faulthandler.dump_traceback_later(25, repeat=True)
    root = Path(root)
    controller = _controller(root)
    store = CheckpointStore(root / "checkpoint", identity={"fixture": 1}, generator=controller.generator)
    continuation = store.load() if store.path.exists() else None

    def should_pause(stage, required):
        _record(root / "checks.jsonl", "check", stage=stage, required=required)
        if target == stage:
            return True
        phase = controller._continuation_state.get("phase")
        if target in {"partial_bootstrap", "partial_official", "partial_terminal"} and stage == "before_hidden_batch":
            expected = {"partial_bootstrap": "bootstrap", "partial_official": "official_ready", "partial_terminal": "terminal_ready"}[target]
            if phase != expected:
                return False
            caches = controller._continuation_state.get("hidden_progress", {})
            for (_, evaluation_phase), distributions in caches.items():
                if target == "partial_terminal" and evaluation_phase != "terminal":
                    continue
                if any(len(cases) == 2 for distribution in distributions.values()
                       for cases in distribution.get("cases", {}).values()):
                    return True
        return False

    control = None if target == "plain" else CampaignPauseControl(store.save, should_pause, hidden_batch_size=2)
    try:
        result = controller.run(continuation=continuation, pause_control=control)
    except CampaignPaused as paused:
        payload = {"paused": True, "phase": paused.continuation.state["phase"], "stage": paused.stage}
    else:
        payload = {"paused": False, "result": _normalize(result)}
    payload["events"] = [event.event_type for event in controller.events.snapshot()]
    (root / "outcome.json").write_text(json.dumps(payload, sort_keys=True))


def _run_process(root, target):
    command = [sys.executable, str(Path(__file__).resolve()), str(root), target]
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=60)
    except subprocess.TimeoutExpired as error:
        pytest.fail(str(error) + "\n" + (error.stderr or b"no stack").decode())
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((root / "outcome.json").read_text())


def _work(root):
    return Counter((line["kind"], json.dumps(line, sort_keys=True)) for line in
                   (json.loads(value) for value in (root / "work.jsonl").read_text().splitlines()))


@pytest.fixture(scope="module")
def continuous(tmp_path_factory):
    root = tmp_path_factory.mktemp("continuous")
    return root, _run_process(root, "never")


@pytest.mark.parametrize("pause_at", [
    "partial_bootstrap", "guidance_ready", "development_completed", "official_sampled",
    "partial_official", "official_completed", "round_finished", "terminal_sampled",
    "partial_terminal", "terminal_completed",
])
def test_fresh_process_continuation_is_same_scientific_run(tmp_path, continuous, pause_at):
    reference_root, reference = continuous
    paused = _run_process(tmp_path, pause_at)
    assert paused["paused"]
    assert "run_paused" in paused["events"]
    resumed = _run_process(tmp_path, "never")
    assert not resumed["paused"]
    assert resumed["result"] == reference["result"]
    assert _work(tmp_path) == _work(reference_root), "resume repeated or skipped scientific work"
    calls = [json.loads(line) for line in (tmp_path / "work.jsonl").read_text().splitlines()]
    assert len([call for call in calls if call["kind"] == "agent"]) == 6
    assert "run_resumed" in resumed["events"]
    assert "round_aborted" not in resumed["events"]


def test_continuation_mode_preserves_original_unbatched_result(tmp_path, continuous):
    reference_root, reference = continuous
    plain = _run_process(tmp_path, "plain")
    assert plain["result"] == reference["result"]
    assert _work(tmp_path) == _work(reference_root)


def test_new_agent_requires_full_hard_time_before_any_agent_starts(tmp_path):
    outcome = _run_process(tmp_path, "before_development")
    assert outcome["paused"] and outcome["phase"] == "development_ready"
    work = [json.loads(line) for line in (tmp_path / "work.jsonl").read_text().splitlines()]
    assert not any(call["kind"] == "agent" for call in work)
    checks = [json.loads(line) for line in (tmp_path / "checks.jsonl").read_text().splitlines()]
    assert [call["required"] for call in checks if call["stage"] == "before_development"] == [15000]


@pytest.mark.parametrize("phase", ["base_training_running", "development_running", "guidance_running", "terminal_guidance_running"])
def test_interrupted_agent_or_training_budget_cannot_restart(tmp_path, phase):
    _run_process(tmp_path, "before_development")
    controller = _controller(tmp_path)
    store = CheckpointStore(tmp_path / "checkpoint", identity={"fixture": 1}, generator=controller.generator)
    saved = store.load()
    saved.state["phase"] = phase
    before = _work(tmp_path)
    with pytest.raises(RuntimeError, match="cannot repeat an interrupted"):
        controller.run(continuation=saved, pause_control=CampaignPauseControl(store.save, lambda *_: False))
    assert _work(tmp_path) == before


def test_completed_checkpoint_does_not_repeat_work(tmp_path):
    first = _run_process(tmp_path, "never")
    saved = CheckpointStore(tmp_path / "checkpoint", identity={"fixture": 1}).load()
    assert saved.state["hidden_suites"] == {}
    assert saved.state["hidden_progress"] == {}
    before = _work(tmp_path)
    second = _run_process(tmp_path, "never")
    assert first["result"] == second["result"]
    assert _work(tmp_path) == before


@pytest.mark.parametrize("scoring_failure", [False, True])
def test_parallel_partial_batch_is_scored_saved_and_only_missing_cases_resume(tmp_path, monkeypatch, scoring_failure):
    from dataclasses import replace
    from pde_agent_verifier_solver.engine.candidate_execution import CandidateExecutionCancelled
    from pde_agent_verifier_solver.engine.evaluator import HiddenVerifier
    from pde_agent_verifier_solver.sdk import CandidateEvaluationOutput, CaseEvaluation

    controller = _controller(tmp_path)
    records = controller.dataset.hidden_cases[:5]
    from pde_agent_verifier_solver.plugins.toy import SineTruth
    original_answer = SineTruth.answer
    truth_ids = {id(record.truth): record.case.case_id for record in records}
    truth_calls = Counter()

    def count_truth(self, query):
        if id(self) in truth_ids:
            truth_calls[truth_ids[id(self)]] += 1
        return original_answer(self, query)

    monkeypatch.setattr(SineTruth, "answer", count_truth)
    artifact = controller.campaign.initial_candidate
    query = controller.campaign.evaluator.default_query

    class SplitEvaluation(ToySolverEvaluation):
        supports_split_candidate_execution = True

        def execute_candidate(self, artifact, case, plan):
            return CandidateEvaluationOutput(artifact.program.solve(case, plan.query))

        def score_candidate(self, artifact, case, truth, plan, output):
            if scoring_failure and case.public_data["frequency"] == 7:
                raise ValueError("candidate produced invalid output")
            expected = super().evaluate(artifact, case, truth, plan)
            assert output.prediction == expected.prediction
            return expected

    class PartialService:
        calls = []
        cancel_once = True

        def execute_cases(self, artifact, cases, plans, evaluation, **kwargs):
            if self.cancel_once:
                self.cancel_once = False
                selected = (0, 2)  # Out-of-order success inside the original batch.
                self.calls.extend(cases[index].case_id for index in selected)
                error = CandidateExecutionCancelled("allocation ended")
                error._allocation_pause_requested = True
                error._candidate_partial_outputs = {
                    index: evaluation.execute_candidate(artifact, cases[index], plans[index])
                    for index in selected
                }
                raise error
            self.calls.extend(case.case_id for case in cases)
            return tuple(evaluation.execute_candidate(artifact, case, plan) for case, plan in zip(cases, plans))

    evaluation, service, progress = SplitEvaluation(query), PartialService(), {}
    store = CheckpointStore(tmp_path / "partial", identity={"partial": 1})
    save = lambda: store.save(progress)
    verifier = HiddenVerifier(evaluation, query, seed=812, candidate_execution=service)
    with pytest.raises(ValueError if scoring_failure else CampaignPaused) as caught:
        verifier.evaluate_incremental(artifact, records, progress=progress,
                                      before_batch=lambda: None, checkpoint=save, batch_size=4)
    progress = store.load()
    if scoring_failure:
        assert caught.value._hidden_evaluation_stage == "score_candidate"
        assert not getattr(caught.value, "_allocation_pause_requested", False)
        assert set(progress["cases"][artifact.artifact_id]) == {records[0].case.case_id}
        return
    assert set(progress["cases"][artifact.artifact_id]) == {records[index].case.case_id for index in (0, 2)}
    assert len(progress["plans"]) == len(progress["truths"]) == 4
    resumed = HiddenVerifier(evaluation, query, seed=812, candidate_execution=service)
    actual = resumed.evaluate_incremental(artifact, records, progress=progress,
                                         before_batch=lambda: None, checkpoint=save, batch_size=4)
    assert truth_calls == Counter({record.case.case_id: 1 for record in records})
    expected = HiddenVerifier(ToySolverEvaluation(query), query, seed=812).evaluate(artifact, records)
    assert actual == expected
    assert len(service.calls) == len(set(service.calls)) == 5


def test_checkpoint_failure_is_infrastructure_failure(tmp_path):
    from pde_agent_verifier_solver.engine.candidate_execution import CandidateExecutionInfrastructureError

    controller = _controller(tmp_path)

    def cannot_save(_):
        raise OSError("checkpoint filesystem unavailable")

    with pytest.raises(CandidateExecutionInfrastructureError, match="persist the Campaign") as caught:
        controller.run(pause_control=CampaignPauseControl(cannot_save, lambda *_: False))
    assert isinstance(caught.value.__cause__, OSError)


def test_deadline_callback_failure_is_infrastructure_failure(tmp_path):
    from pde_agent_verifier_solver.engine.candidate_execution import CandidateExecutionInfrastructureError

    controller = _controller(tmp_path)

    def cannot_check(*_):
        raise OSError("allocation clock unavailable")

    with pytest.raises(CandidateExecutionInfrastructureError, match="check the allocation deadline"):
        controller.run(pause_control=CampaignPauseControl(lambda _: None, cannot_check))


@pytest.mark.parametrize("release_fails", [False, True])
def test_paused_campaign_requires_confirmed_case_cleanup(tmp_path, monkeypatch, release_fails):
    from pde_agent_verifier_solver.engine.candidate_execution import CandidateExecutionInfrastructureError

    controller = _controller(tmp_path)
    original_release = controller.generator.release
    release_error = RuntimeError("Case release did not finish")
    released = []

    def release(state, case_ids):
        released.extend(case_ids)
        if release_fails:
            raise release_error
        return original_release(state, case_ids)

    monkeypatch.setattr(controller.generator, "release", release)
    control = CampaignPauseControl(lambda _: None,
        lambda stage, _: stage == "terminal_cohort_sealed")
    expected = CandidateExecutionInfrastructureError if release_fails else CampaignPaused
    with pytest.raises(expected) as caught:
        controller.run(pause_control=control)
    assert released
    if release_fails:
        assert caught.value.__cause__ is release_error
        pause = release_error.__context__
        assert isinstance(pause, CampaignPaused)
        assert any("Case cleanup also failed" in note for note in pause.__notes__)
    else:
        assert caught.value.stage == "terminal_cohort_sealed"


def test_pause_cleanup_attempts_every_distinct_sampler_before_raising():
    attempts = []
    failures = [RuntimeError("first Case cleanup"), RuntimeError("second Case cleanup")]

    class Sampler:
        def __init__(self, index):
            self.index = index

        def close(self):
            attempts.append(self.index)
            raise failures[self.index]

    first, second = Sampler(0), Sampler(1)
    with pytest.raises(RuntimeError) as caught:
        CampaignController._close_case_samplers((first, second, first))
    assert attempts == [0, 1]
    assert caught.value is failures[0]
    assert any("second Case cleanup" in note for note in caught.value.__notes__)


if __name__ == "__main__":
    _child(Path(sys.argv[1]), sys.argv[2])
