"""Artemis macro-only compatibility with the canonical Candidate worker."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from pde_agent_verifier_solver.app import backend_run
from pde_agent_verifier_solver.engine import (
    InProcessCandidateExecutionService,
    PythonCandidateCodec,
    PythonTaskWorkerSpec,
    SubprocessCandidateExecutionService,
)
from pde_agent_verifier_solver.engine import candidate_execution as worker_module
from pde_agent_verifier_solver.engine.campaign import (
    AgentSpec, CampaignRecipe, CaseCounts, DevelopmentSamplingPolicy,
    EvaluatorPolicy, GeneratorRecipe, RoundSettings,
)
from pde_agent_verifier_solver.engine.evaluator import HiddenVerifier
from pde_agent_verifier_solver.plugins.vp_1d1v import (
    ArtemisObservableTask, VP1D_MACRO_FIELDS,
)
from pde_agent_verifier_solver.plugins.vp_1d1v import task as task_module
from pde_agent_verifier_solver.plugins.vp_1d1v.macro_matching import VP1DMacroMatcher
from pde_agent_verifier_solver.sdk import (
    CandidateEvaluationOutput, EvaluationPhase, EvaluationPlan, HiddenSuiteSpec,
    OfficialEvaluationSpec, SampleSource, TaskKind,
    write_candidate_bundle_manifest,
)
from tests.test_artemis_canonical import _answer, _artifact, _binding, _query, _request


_BUNDLE = '''\
import numpy as np
from pde_agent_verifier_solver.sdk import (
    AnswerView, open_public_case_session, open_development_solution_session,
)

class Handle:
    def __init__(self, horizon):
        self.horizon = horizon

    def evaluate(self, query):
        assert all(point["t"] == self.horizon for point in query.points)
        return AnswerView(
            kind=query.kind, points=query.points, shape=(len(query.points),),
            fields={name: tuple(.9 * (i + 1) * (1 + p["t"] + p["x"])
                                for p in query.points)
                    for i, name in enumerate(("M0", "M1", "M2", "M3", "E"))},
        )

def prepare_solution(case, solve_budget=None):
    assert case.round_index is None and case.parent_case_id is None
    assert not case.tags and case.case_id.startswith("candidate-case-")
    arrays = case.public_data["arrays"]
    horizon = case.public_data["equation"]["domain"]["time_interval"][1]
    assert float(arrays["t"][-1]) == horizon
    assert arrays["source"].shape[0] == len(arrays["t"])
    session = open_public_case_session(case)
    np.testing.assert_array_equal(
        session.query_initial_distribution(np.array([[.17, .13]])), [1.])
    np.testing.assert_allclose(
        session.query_source(np.array([[horizon, .17, .13]])), [horizon + .3])
    try:
        session.query_source(np.array([[horizon + .01, .17, .13]]))
    except ValueError as error:
        assert "causal" in str(error)
    else:
        raise AssertionError("future source was accessible")
    try:
        open_development_solution_session(case)
    except RuntimeError:
        pass
    else:
        raise AssertionError("completed-solution oracle was accessible")
    return Handle(horizon)
'''


def _bundle_artifact(tmp_path):
    bundle = tmp_path / "source"
    bundle.mkdir()
    (bundle / "submission.py").write_text(_BUNDLE, encoding="utf-8")
    write_candidate_bundle_manifest(bundle, files=("submission.py",))
    return PythonCandidateCodec(task_kind=TaskKind.OBSERVABLE_MODEL).bind_existing(
        bundle, storage_root=tmp_path / "sealed", artifact_id="artemis-split",
        name="Macro-only Candidate", summary="Causal public oracle regression",
        usage="prepare_solution then evaluate",
    )


def test_split_scoring_preserves_causal_macro_contract_and_handle_lifecycle():
    binding, _source = _binding()
    generator = binding.plugin.generator
    state = generator.load(binding.checkpoint, {})
    record = generator.sample(state, _request())[0]
    query = _query()
    events = []

    class Matcher(VP1DMacroMatcher):
        def prepare_target(self, case, truth, query):
            assert case.public_data["equation"]["domain"]["time_interval"] == (0., .5)
            events.append("score_private_truth")
            return super().prepare_target(case, truth, query)

    class Prepared:
        def prepare_solution(self, case, solve_budget=None):
            assert solve_budget == 10.
            events.append("prepare_without_query")

            class Handle:
                def evaluate(self, query):
                    events.append("query")
                    return _answer(query, .9)

                def close(self):
                    events.append("close")

            return Handle()

    evaluation = ArtemisObservableTask(Matcher()).create_evaluation({
        "default_query": query, "require_prepared_solution": True,
        "candidate_compute_budget": 10.,
    })
    plan = EvaluationPlan(EvaluationPhase.OFFICIAL, query)
    assert evaluation.supports_split_candidate_execution is True
    output = evaluation.execute_candidate(_artifact(Prepared()), record.case, plan)
    assert isinstance(output, CandidateEvaluationOutput)
    assert events == ["prepare_without_query", "query", "close"]
    assert set(output.prediction.fields) == set(VP1D_MACRO_FIELDS)
    assert set(output.metrics) == {
        "candidate_solve_wall_seconds", "candidate_query_wall_seconds",
        "candidate_solve_active_seconds", "candidate_budget_clock_compat_aliases",
        "candidate_oracle_queue_wait_seconds", "candidate_oracle_lock_wait_seconds",
        "candidate_platform_wait_seconds", "candidate_oracle_timed_requests",
        "query_blind_preparation",
    }
    # A scorer cannot invoke the Candidate: this Artifact has no callable ABI.
    result = evaluation.score_candidate(
        _artifact(object()), record.case, record.truth.answer(query), plan, output)
    assert events[-1] == "score_private_truth"
    assert result.primary_error == pytest.approx(.1)
    assert result.macro_match.contract_id == "vp_1d1v_m0_m3_e_match_v1"
    assert result.metrics["query_blind_preparation"] == 1.


def test_split_candidate_preserves_solve_budget_and_closes_handle(monkeypatch):
    binding, _source = _binding()
    generator = binding.plugin.generator
    state = generator.load(binding.checkpoint, {})
    case = generator.sample(state, _request())[0].case
    events = []

    class Prepared:
        def prepare_solution(self, case, solve_budget=None):
            class Handle:
                def evaluate(self, query):
                    return _answer(query)

                def close(self):
                    events.append("closed")

            return Handle()

    clock = iter((0., 2., 2., 2.))
    monkeypatch.setattr(task_module, "perf_counter", lambda: next(clock))
    active_clock = iter((0., 2.))
    monkeypatch.setattr(task_module, "candidate_solve_clock", lambda: next(active_clock))
    evaluation = ArtemisObservableTask().create_evaluation({
        "default_query": _query(), "candidate_compute_budget": 1.,
    })
    with pytest.raises(TimeoutError, match="solve budget"):
        evaluation.execute_candidate(
            _artifact(Prepared()), case, EvaluationPlan(EvaluationPhase.OFFICIAL, _query()))
    assert events == ["closed"]


@pytest.mark.parametrize("phase", [EvaluationPhase.OFFICIAL, EvaluationPhase.TERMINAL])
def test_subprocess_hidden_macros_match_legacy_with_causal_oracle(tmp_path, monkeypatch, phase):
    binding, source = _binding()
    generator = binding.plugin.generator
    state = generator.load(binding.checkpoint, {})
    records = generator.sample(state, replace(_request(), count=2))
    artifact = _bundle_artifact(tmp_path)
    settings = {
        "default_query": _query(), "require_prepared_solution": True,
        "official_point_count": 2, "terminal_point_count": 3,
    }
    requests = []
    serialize = worker_module._worker_request_payload

    def capture(**kwargs):
        # Only public Case/plan, sealed Artifact and trusted Task recipe enter.
        assert not ({"truth", "truths", "record", "records"} & set(kwargs))
        request = serialize(**kwargs)
        requests.append(request)
        return request

    monkeypatch.setattr(worker_module, "_worker_request_payload", capture)
    execution = SubprocessCandidateExecutionService(
        tmp_path / "workers", task=PythonTaskWorkerSpec.from_task_type(ArtemisObservableTask),
        evaluation_settings=settings, max_workers=2, case_hard_wall_seconds=30,
    )
    verifier = HiddenVerifier(
        binding.plugin.task(), _query(), settings, phase=phase,
        generator=generator, generator_state=state, candidate_execution=execution,
    )
    result = verifier.evaluate(artifact, records)
    assert artifact.program._loaded is None  # Candidate imports stay in its worker.
    assert result.primary_error == pytest.approx(.1)
    assert tuple(item.case_id for item in result.cases) == tuple(r.case.case_id for r in records)
    assert len(requests) == len(execution.audits) == 2
    for request in requests:
        assert '"truth"' not in json.dumps(request)
        assert '"truths"' not in json.dumps(request)
    assert all(a.status == "completed" and a.process_group_cleaned
               and not a.controller_truth_shared for a in execution.audits)
    assert source.sessions and all(session.closed for session in source.sessions)
    # The same plans and private truths give the same scientific result with
    # the previous in-process entrypoint; execution timing naturally differs.
    baseline = HiddenVerifier(
        binding.plugin.task(), _query(), settings, phase=phase,
        generator=generator, generator_state=state,
    ).evaluate(replace(artifact, program=deepcopy(artifact.program)), records)
    assert result.primary_error == baseline.primary_error
    for actual, expected in zip(result.cases, baseline.cases):
        assert actual.macro_match == expected.macro_match
        assert actual.prediction == expected.prediction
        for name in expected.metrics:
            if not name.endswith("wall_seconds") and name not in {
                "candidate_solve_active_seconds", "candidate_oracle_queue_wait_seconds",
                "candidate_oracle_lock_wait_seconds", "candidate_platform_wait_seconds",
                "candidate_oracle_timed_requests",
            }:
                assert actual.metrics[name] == expected.metrics[name]


def test_backend_binding_passes_candidate_execution_to_existing_controller(monkeypatch):
    binding, _source = _binding()
    recipe = CampaignRecipe(
        name="Artemis split composition", round_count=1,
        agents=(AgentSpec("agent", "Agent"),),
        default_round=RoundSettings(
            development_sampling=DevelopmentSamplingPolicy(
                max_cases_per_agent_per_round=1, allowed_sources=(SampleSource.BASE,),
                starter=CaseCounts(base=1)),
            official_evaluation=OfficialEvaluationSpec(
                round=HiddenSuiteSpec(base=1), terminal=HiddenSuiteSpec(base=1))),
        generator=GeneratorRecipe(guidance_enabled=False, perturbations_enabled=True),
        evaluator=EvaluatorPolicy(_query(), development_point_count=2),
    )
    monkeypatch.setattr(backend_run, "CampaignController", lambda **kwargs: kwargs)
    execution = InProcessCandidateExecutionService()
    configured = binding.controller(
        recipe, {"agent": object()}, local_development=True,
        candidate_execution=execution,
    )
    assert configured["candidate_execution"] is execution
    assert configured["task"].create_evaluation({"default_query": _query()}).supports_split_candidate_execution
