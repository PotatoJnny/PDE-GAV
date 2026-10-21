from __future__ import annotations

from dataclasses import replace
from threading import Event, Thread
import time

import numpy as np
import pytest

from pde_agent_verifier_solver.engine.campaign import EvaluatorPolicy
from pde_agent_verifier_solver.engine.candidate_case import (
    CANDIDATE_CASE_ID_PREFIX,
    candidate_case_id,
    candidate_facing_case,
)
from pde_agent_verifier_solver.engine.candidate_execution import (
    CandidateExecutionBatchTimeout,
)
from pde_agent_verifier_solver.engine.evaluator import (
    AgentEvaluator,
    EvaluationRequest,
    HiddenVerifier,
    _CaseSessionBindings,
)
from pde_agent_verifier_solver.engine.unified_runtime import _InteractiveSampler
from pde_agent_verifier_solver.plugins.gkw.isolation import snapshot_gkw_case
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Artifact,
    CandidateEvaluationOutput,
    Case,
    CaseAccessKind,
    CaseEvaluation,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    DevelopmentCoordinateMode,
    EvaluationPhase,
    EvaluationPlan,
    GeneratorCheckpoint,
    GeneratorState,
    MacroMatchResult,
    PairedSampleBatch,
    PairedSampleRequest,
    PerturbCasesRequest,
    QueryKind,
    SampleBatch,
    SampleCasesRequest,
    SampleSource,
    TaskKind,
    aggregate_case_evaluations,
    open_development_solution_session,
    open_public_case_session,
)


QUERY = AnswerQuery.grid(fields=("u",), x=1)


def test_candidate_case_ids_are_replay_stable_and_seed_scoped() -> None:
    first = candidate_case_id("controller-private-case", seed=17)

    assert candidate_case_id("controller-private-case", seed=17) == first
    assert candidate_case_id("controller-private-case", seed=18) != first
    assert first.startswith(CANDIDATE_CASE_ID_PREFIX)


class _Truth:
    def answer(self, query: AnswerQuery) -> AnswerView:
        return AnswerView(
            kind=query.kind,
            fields={"u": (1.0,)},
            axes={"x": (0.5,)},
            shape=(1,),
        )


def _case(
    case_id: str,
    *,
    visibility: CaseVisibility = CaseVisibility.DEVELOPMENT,
    parent_case_id: str | None = "private-parent",
) -> Case:
    return Case(
        case_id=case_id,
        source=CaseSource.PERTURBED,
        visibility=visibility,
        public_data={
            "case_id": case_id,
            "cohort": "terminal-private",
            "physical": {"values": [1.0, 2.0]},
            "array": np.asarray([3.0, 4.0]),
            "nested": {"parent_case_id": parent_case_id},
        },
        round_index=7,
        parent_case_id=parent_case_id,
        tags=("terminal", "perturbed"),
    )


def _artifact(program: object) -> Artifact:
    return Artifact(
        artifact_id="candidate",
        kind=TaskKind.SOLVER,
        name="Candidate",
        program=program,
        summary="test",
        usage="test",
        round_index=1,
        agent_id="agent",
    )


class _MutatingCandidate:
    def __init__(self) -> None:
        self.seen: list[Case] = []

    def __call__(self, case: Case) -> float:
        self.seen.append(case)
        case.public_data["physical"]["values"][0] = 999.0
        case.public_data["array"][0] = 999.0
        return 0.0


class _BoundaryEvaluation:
    program_id = "candidate-boundary-evaluation"

    def __init__(self) -> None:
        self.plan_cases: list[Case] = []
        self.evaluation_cases: list[Case] = []
        self.aggregate_case_ids: list[tuple[str, ...]] = []
        self.aggregate_macro_ids: list[tuple[str, ...]] = []

    def make_plan(self, case, phase, settings, seed, request=None):
        self.plan_cases.append(case)
        return EvaluationPlan(
            phase=phase,
            query=request or QUERY,
            seed=seed,
            settings=settings,
        )

    def evaluate(self, artifact, case, truth, plan):
        del truth, plan
        self.evaluation_cases.append(case)
        error = artifact.program(case)
        return CaseEvaluation(
            case_id=case.case_id,
            primary_error=error,
            metrics={"error": error},
            macro_match=MacroMatchResult(
                case_id=case.case_id,
                contract_id="boundary-v1",
                primary_error=error,
            ),
        )

    def aggregate(self, artifact, case_results, plans):
        self.aggregate_case_ids.append(
            tuple(result.case_id for result in case_results)
        )
        self.aggregate_macro_ids.append(
            tuple(result.macro_match.case_id for result in case_results)
        )
        return aggregate_case_evaluations(artifact, case_results, plans)


def _policy(*, development_solution: bool = False) -> EvaluatorPolicy:
    return EvaluatorPolicy(
        QUERY,
        allow_grid_truth=True,
        development_coordinate_mode=DevelopmentCoordinateMode.USER_FIXED_QUERY,
        allow_development_solution_session=development_solution,
    )


def test_agent_evaluator_projects_detaches_and_round_trips_only_opaque_ids() -> None:
    raw = _case("raw-development-case")
    evaluation = _BoundaryEvaluation()
    candidate = _MutatingCandidate()
    evaluator = AgentEvaluator((CaseRecord(raw, _Truth()),), evaluation, _policy())

    first = evaluator.cases[0]
    assert first.case_id.startswith(CANDIDATE_CASE_ID_PREFIX)
    assert first.source is CaseSource.GENERATED
    assert first.visibility is CaseVisibility.DEVELOPMENT
    assert first.round_index is None
    assert first.parent_case_id is None
    assert first.tags == ()
    assert "cohort" not in first.public_data
    assert "parent_case_id" not in first.public_data["nested"]
    assert first.public_data["case_id"] == first.case_id

    first.public_data["physical"]["values"][0] = -10.0
    first.public_data["array"][0] = -10.0
    second = evaluator.cases[0]
    assert second.public_data["physical"]["values"][0] == 1.0
    assert second.public_data["array"][0] == 3.0
    assert raw.public_data["physical"]["values"][0] == 1.0
    assert raw.public_data["array"][0] == 3.0

    with pytest.raises(KeyError, match="not available"):
        evaluator.evaluate(_artifact(candidate), EvaluationRequest((raw.case_id,)))

    response = evaluator.evaluate(
        _artifact(candidate),
        EvaluationRequest((second.case_id,), return_truth=True),
    )

    assert response.cases[0].case_id == second.case_id
    assert evaluation.plan_cases == [raw]
    assert evaluation.evaluation_cases[0].case_id == second.case_id
    assert evaluation.aggregate_case_ids == [(raw.case_id,)]
    assert evaluation.aggregate_macro_ids == [(raw.case_id,)]
    assert raw.public_data["physical"]["values"][0] == 1.0
    assert raw.public_data["array"][0] == 3.0


def test_hidden_verifier_projects_candidate_call_but_reports_controller_id() -> None:
    raw = _case(
        "raw-hidden-case",
        visibility=CaseVisibility.HIDDEN,
    )
    evaluation = _BoundaryEvaluation()
    candidate = _MutatingCandidate()
    verifier = HiddenVerifier(
        evaluation,
        QUERY,
        phase=EvaluationPhase.TERMINAL,
    )

    report = verifier.evaluate(_artifact(candidate), (CaseRecord(raw, _Truth()),))

    assert evaluation.plan_cases == [raw]
    assert candidate.seen[0].case_id.startswith(CANDIDATE_CASE_ID_PREFIX)
    assert candidate.seen[0].visibility is CaseVisibility.DEVELOPMENT
    assert report.cases[0].case_id == raw.case_id
    assert report.cases[0].macro_match.case_id == raw.case_id
    assert raw.public_data["array"][0] == 3.0


class _SplitBoundaryEvaluation(_BoundaryEvaluation):
    supports_split_candidate_execution = True

    def __init__(self) -> None:
        super().__init__()
        self.scored_truths: list[AnswerView] = []

    def execute_candidate(self, artifact, case, plan):
        del artifact, case, plan
        raise AssertionError("Candidate execution must be delegated to the service")

    def score_candidate(self, artifact, case, truth, plan, output):
        del artifact, plan
        self.scored_truths.append(truth)
        value = float(output.prediction.fields["u"][0])
        expected = float(truth.fields["u"][0])
        error = abs(value - expected)
        return CaseEvaluation(
            case_id=case.case_id,
            primary_error=error,
            metrics={"error": error},
            prediction=output.prediction,
            macro_match=MacroMatchResult(
                case_id=case.case_id,
                contract_id="split-boundary-v1",
                primary_error=error,
            ),
        )

    def evaluate(self, artifact, case, truth, plan):
        del artifact, case, truth, plan
        raise AssertionError("split Hidden evaluation must not use evaluate()")


class _RecordingCandidateExecution:
    def __init__(self) -> None:
        self.cases: tuple[Case, ...] = ()
        self.plans: tuple[EvaluationPlan, ...] = ()
        self.methods: tuple[str, ...] = ()
        self.session_results: tuple[object, ...] = ()

    def execute_cases(
        self,
        artifact,
        cases,
        plans,
        evaluation,
        *,
        public_session_methods=(),
        invoke_public_session=None,
        close_public_session=None,
        cancellation_event=None,
        development_batch_deadline=None,
    ):
        del artifact, evaluation
        self.cases = tuple(cases)
        self.plans = tuple(plans)
        self.methods = tuple(public_session_methods)
        self.cancellation_event = cancellation_event
        self.development_batch_deadline = development_batch_deadline
        if self.methods:
            assert invoke_public_session is not None
            assert close_public_session is not None
            self.session_results = (
                invoke_public_session(
                    self.cases[0].case_id,
                    "marker",
                    (":remote-1",),
                    {},
                ),
                invoke_public_session(
                    self.cases[0].case_id,
                    "marker",
                    (":remote-2",),
                    {},
                ),
            )
            close_public_session(self.cases[0].case_id)
        return tuple(
            CandidateEvaluationOutput(
                AnswerView(
                    kind=QueryKind.GRID,
                    fields={"u": (1.0,)},
                    axes={"x": (0.5,)},
                    shape=(1,),
                )
            )
            for _case in self.cases
        )


class _SharedDeadlineCandidateExecution:
    development_batch_hard_wall_seconds = 0.05

    def __init__(self) -> None:
        self.deadlines: list[float | None] = []

    def execute_cases(
        self,
        artifact,
        cases,
        plans,
        evaluation,
        *,
        public_session_methods=(),
        invoke_public_session=None,
        close_public_session=None,
        cancellation_event=None,
        development_batch_deadline=None,
    ):
        del (
            artifact,
            plans,
            evaluation,
            public_session_methods,
            invoke_public_session,
            close_public_session,
            cancellation_event,
        )
        self.deadlines.append(development_batch_deadline)
        if len(self.deadlines) == 1:
            time.sleep(0.08)
            return tuple(
                CandidateEvaluationOutput(
                    AnswerView(
                        kind=QueryKind.GRID,
                        fields={"u": (1.0,)},
                        axes={"x": (0.5,)},
                        shape=(1,),
                    )
                )
                for _case in cases
            )
        if development_batch_deadline != self.deadlines[0]:
            raise AssertionError("Development batch deadline was reset")
        if time.monotonic() < development_batch_deadline:
            raise AssertionError("shared Development batch deadline did not expire")
        raise CandidateExecutionBatchTimeout(
            "Development Candidate evaluation batch exceeded its hard wall"
        )


def test_hidden_split_execution_keeps_truth_in_controller_and_bridges_public_session():
    raw = _case(
        "split-hidden-case",
        visibility=CaseVisibility.HIDDEN,
        parent_case_id=None,
    )
    evaluation = _SplitBoundaryEvaluation()
    execution = _RecordingCandidateExecution()
    generator = _SessionGenerator()
    verifier = HiddenVerifier(
        evaluation,
        QUERY,
        generator=generator,
        generator_state=_generator_state(),
        candidate_execution=execution,
    )

    report = verifier.evaluate(
        _artifact(object()),
        (CaseRecord(raw, _Truth()),),
    )

    assert len(execution.cases) == 1
    assert execution.cases[0].case_id.startswith(CANDIDATE_CASE_ID_PREFIX)
    assert execution.cases[0].case_id != raw.case_id
    assert execution.cases[0].visibility is CaseVisibility.DEVELOPMENT
    assert execution.plans[0].phase is EvaluationPhase.OFFICIAL
    assert execution.methods == ("marker",)
    assert execution.cancellation_event is None
    assert execution.development_batch_deadline is None
    assert execution.session_results == (
        "public_problem:remote-1",
        "public_problem:remote-2",
    )
    assert len(generator.opened) == 1
    assert generator.opened[0][2].close_calls == 1
    assert len(evaluation.scored_truths) == 1
    assert report.primary_error == 0.0
    assert report.cases[0].case_id == raw.case_id


def test_development_split_execution_keeps_truth_in_controller_and_uses_service():
    raw = _case("split-development-case", parent_case_id=None)
    evaluation = _SplitBoundaryEvaluation()
    execution = _RecordingCandidateExecution()
    generator = _SessionGenerator()
    evaluator = AgentEvaluator(
        (CaseRecord(raw, _Truth()),),
        evaluation,
        _policy(),
        generator=generator,
        generator_state=_generator_state(),
        candidate_execution=execution,
    )

    response = evaluator.evaluate(
        _artifact(object()),
        EvaluationRequest((evaluator.cases[0].case_id,), return_truth=True),
    )

    assert len(execution.cases) == 1
    assert execution.plans[0].phase is EvaluationPhase.DEVELOPMENT
    assert execution.methods == ("marker",)
    assert execution.cancellation_event is not None
    assert execution.cancellation_event.is_set() is False
    assert execution.development_batch_deadline is None
    assert execution.session_results == (
        "public_problem:remote-1",
        "public_problem:remote-2",
    )
    assert len(generator.opened) == 1
    assert generator.opened[0][2].close_calls == 1
    assert len(evaluation.scored_truths) == 1
    assert response.primary_error == 0.0
    assert response.cases[0].case_id == evaluator.cases[0].case_id


def test_evaluate_many_shares_one_development_deadline_across_candidates() -> None:
    raw = _case("split-development-shared-deadline", parent_case_id=None)
    execution = _SharedDeadlineCandidateExecution()
    evaluator = AgentEvaluator(
        (CaseRecord(raw, _Truth()),),
        _SplitBoundaryEvaluation(),
        _policy(),
        candidate_execution=execution,
    )
    first = _artifact(object())
    second = replace(first, artifact_id="candidate-two", name="Candidate two")

    with pytest.raises(CandidateExecutionBatchTimeout):
        evaluator.evaluate_many(
            (first, second),
            EvaluationRequest((evaluator.cases[0].case_id,)),
        )

    assert len(execution.deadlines) == 2
    assert execution.deadlines[0] == execution.deadlines[1]


def test_split_development_rejects_solution_session_policy() -> None:
    with pytest.raises(ValueError, match="does not expose the Development"):
        AgentEvaluator(
            (),
            _SplitBoundaryEvaluation(),
            _policy(development_solution=True),
            candidate_execution=_RecordingCandidateExecution(),
        )


def test_controller_case_ids_cannot_use_the_reserved_candidate_prefix() -> None:
    reserved = _case(f"{CANDIDATE_CASE_ID_PREFIX}forged")
    with pytest.raises(ValueError, match="reserved candidate-case prefix"):
        AgentEvaluator(
            (CaseRecord(reserved, _Truth()),),
            _BoundaryEvaluation(),
            _policy(),
        )

    hidden = replace(reserved, visibility=CaseVisibility.HIDDEN)
    verifier = HiddenVerifier(_BoundaryEvaluation(), QUERY)
    with pytest.raises(ValueError, match="reserved candidate-case prefix"):
        verifier.evaluate(
            _artifact(_MutatingCandidate()),
            (CaseRecord(hidden, _Truth()),),
        )


class _Session:
    def __init__(self, value: str, *, close_error: str | None = None) -> None:
        self.value = value
        self.close_error = close_error
        self.close_calls = 0

    def marker(self, suffix: str = "") -> str:
        return f"{self.value}{suffix}"

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise RuntimeError(self.close_error)


class _SessionGenerator:
    def __init__(self, *, fail_close: bool = False) -> None:
        self.fail_close = fail_close
        self.opened: list[tuple[Case, CaseAccessKind, _Session]] = []

    def case_session_methods(self, access: CaseAccessKind) -> tuple[str, ...]:
        del access
        return ("marker",)

    def open_case_session(self, state, case, access):
        del state
        error = f"{access.value} close failed" if self.fail_close else None
        session = _Session(access.value, close_error=error)
        self.opened.append((case, access, session))
        return session


def test_persistent_public_session_close_defers_without_waiting_for_hung_invoke():
    entered = Event()
    release = Event()

    class BlockingSession(_Session):
        def marker(self, suffix: str = "") -> str:
            entered.set()
            if not release.wait(3.0):
                raise RuntimeError("test session was not released")
            return super().marker(suffix)

    class BlockingGenerator(_SessionGenerator):
        def open_case_session(self, state, case, access):
            del state
            session = BlockingSession(access.value)
            self.opened.append((case, access, session))
            return session

    raw = _case("deferred-close-controller-case", parent_case_id=None)
    candidate = candidate_facing_case(raw, id_seed=0)
    generator = BlockingGenerator()
    bindings = _CaseSessionBindings(
        generator,
        _generator_state(),
    )
    session = bindings.persistent_public_session(raw, candidate)
    outcome: dict[str, object] = {}
    invocation = Thread(
        target=lambda: outcome.setdefault(
            "value",
            session.invoke("marker", (":done",), {}),
        )
    )
    invocation.start()
    assert entered.wait(1.0)

    close_started = time.monotonic()
    session.close()
    assert time.monotonic() - close_started < 0.2
    assert invocation.is_alive()
    with pytest.raises(RuntimeError, match="no longer active"):
        session.invoke("marker", (), {})

    release.set()
    invocation.join(2.0)
    assert not invocation.is_alive()
    assert outcome == {"value": "public_problem:done"}
    assert generator.opened[0][2].close_calls == 1
    session.close()
    assert generator.opened[0][2].close_calls == 1


def _generator_state() -> GeneratorState:
    return GeneratorState(
        GeneratorCheckpoint("candidate-boundary-generator", object()),
        object(),
    )


class _SnapshotSessionCandidate:
    def __init__(self) -> None:
        self.snapshot: Case | None = None
        self.method = None

    def __call__(self, case: Case) -> float:
        self.snapshot = snapshot_gkw_case(case)
        session = open_public_case_session(self.snapshot)
        self.method = session.marker
        assert self.method(":ok") == "public_problem:ok"
        return 0.0


def test_gkw_copied_snapshot_uses_bound_raw_case_and_is_revoked() -> None:
    raw = _case("gkw-controller-case", parent_case_id=None)
    generator = _SessionGenerator()
    candidate = _SnapshotSessionCandidate()
    evaluator = AgentEvaluator(
        (CaseRecord(raw, _Truth()),),
        _BoundaryEvaluation(),
        _policy(),
        generator=generator,
        generator_state=_generator_state(),
    )

    evaluator.evaluate(
        _artifact(candidate),
        EvaluationRequest((evaluator.cases[0].case_id,)),
    )

    assert candidate.snapshot is not None
    assert candidate.snapshot.case_id != raw.case_id
    assert generator.opened[0][0] is raw
    assert generator.opened[0][2].close_calls == 1
    with pytest.raises(RuntimeError, match="no longer active"):
        candidate.method()


def test_direct_session_hook_uses_opaque_id_and_policy_allowlists() -> None:
    raw = _case("hook-controller-case", parent_case_id=None)
    generator = _SessionGenerator()
    evaluator = AgentEvaluator(
        (CaseRecord(raw, _Truth()),),
        _BoundaryEvaluation(),
        _policy(),
        generator=generator,
        generator_state=_generator_state(),
    )
    opaque_id = evaluator.cases[0].case_id

    assert evaluator.case_session_methods(CaseAccessKind.PUBLIC_PROBLEM) == (
        "marker",
    )
    assert evaluator.case_session_methods(
        CaseAccessKind.DEVELOPMENT_SOLUTION
    ) == ()
    assert evaluator.invoke_case_session(
        opaque_id,
        CaseAccessKind.PUBLIC_PROBLEM,
        "marker",
        (":rpc",),
    ) == "public_problem:rpc"
    assert generator.opened[0][0] is raw
    assert generator.opened[0][2].close_calls == 1
    with pytest.raises(KeyError, match="not available"):
        evaluator.invoke_case_session(
            raw.case_id,
            CaseAccessKind.PUBLIC_PROBLEM,
            "marker",
        )
    with pytest.raises(PermissionError, match="not authorized"):
        evaluator.invoke_case_session(
            opaque_id,
            CaseAccessKind.DEVELOPMENT_SOLUTION,
            "marker",
        )


class _FailingSessionCandidate:
    def __init__(self) -> None:
        self.methods = []

    def __call__(self, case: Case) -> float:
        self.methods = [
            open_public_case_session(case).marker,
            open_development_solution_session(case).marker,
        ]
        raise ValueError("candidate failed")


def test_candidate_error_survives_all_session_cleanup_failures() -> None:
    raw = _case("cleanup-controller-case", parent_case_id=None)
    generator = _SessionGenerator(fail_close=True)
    candidate = _FailingSessionCandidate()
    evaluator = AgentEvaluator(
        (CaseRecord(raw, _Truth()),),
        _BoundaryEvaluation(),
        _policy(development_solution=True),
        generator=generator,
        generator_state=_generator_state(),
    )

    with pytest.raises(ValueError, match="candidate failed") as caught:
        evaluator.evaluate(
            _artifact(candidate),
            EvaluationRequest((evaluator.cases[0].case_id,)),
        )

    assert len(generator.opened) == 2
    assert all(session.close_calls == 1 for _case, _access, session in generator.opened)
    cleanup_errors = getattr(caught.value, "_pde_session_cleanup_errors")
    assert {str(error) for error in cleanup_errors} == {
        "public_problem close failed",
        "development_solution close failed",
    }
    assert len(caught.value.__notes__) == 2
    for method in candidate.methods:
        with pytest.raises(RuntimeError, match="no longer active"):
            method()


class _SamplerInner:
    def __init__(self, evaluator: AgentEvaluator) -> None:
        self.evaluator = evaluator
        self.perturb_requests: list[PerturbCasesRequest] = []
        self.sequence = 0

    def _register(self, cases: tuple[Case, ...]) -> None:
        self.evaluator.register(tuple(CaseRecord(case, _Truth()) for case in cases))

    def sample_cases(self, request: SampleCasesRequest) -> SampleBatch:
        self.sequence += 1
        case = _case(f"raw-root-{self.sequence}", parent_case_id=None)
        case = replace(case, source=CaseSource.GENERATED)
        self._register((case,))
        return SampleBatch((case,), 5)

    def sample_paired_cases(self, request: PairedSampleRequest) -> PairedSampleBatch:
        self.sequence += 1
        root = replace(
            _case(f"raw-paired-root-{self.sequence}", parent_case_id=None),
            source=CaseSource.GENERATED,
        )
        child = _case(
            f"raw-paired-child-{self.sequence}",
            parent_case_id=root.case_id,
        )
        self._register((root, child))
        return PairedSampleBatch((root,), (child,), 3, ())

    def perturb_cases(self, request: PerturbCasesRequest) -> SampleBatch:
        self.perturb_requests.append(request)
        child = _case(
            f"raw-child-{len(self.perturb_requests)}",
            parent_case_id=request.parent_case_ids[0],
        )
        self._register((child,))
        return SampleBatch((child,), 4)


def test_interactive_sampler_projects_batches_and_translates_opaque_parents() -> None:
    evaluator = AgentEvaluator((), _BoundaryEvaluation(), _policy())
    inner = _SamplerInner(evaluator)
    sampler = _InteractiveSampler(
        inner,
        evaluator,
        (SampleSource.BASE,),
        allow_guidance=False,
        allow_perturbations=True,
    )

    base = sampler.sample_cases(SampleCasesRequest(SampleSource.BASE, 1))
    root = base.cases[0]
    assert root.case_id.startswith(CANDIDATE_CASE_ID_PREFIX)
    assert root.parent_case_id is None
    child = sampler.perturb_cases(PerturbCasesRequest((root.case_id,))).cases[0]
    assert inner.perturb_requests[0].parent_case_ids == ("raw-root-1",)
    assert child.case_id.startswith(CANDIDATE_CASE_ID_PREFIX)
    assert child.parent_case_id is None

    paired = sampler.sample_paired_cases(PairedSampleRequest(SampleSource.BASE, 1))
    assert paired.attempts == ()
    assert paired.roots[0].parent_case_id is None
    assert paired.children[0].parent_case_id is None
    assert all(case.round_index is None for case in paired.cases)
    assert all(not case.tags for case in paired.cases)


def test_interactive_sampler_closes_starter_only_sources_before_agent_use() -> None:
    evaluator = AgentEvaluator((), _BoundaryEvaluation(), _policy())
    sampler = _InteractiveSampler(
        _SamplerInner(evaluator),
        evaluator,
        (SampleSource.ORIGINAL, SampleSource.BASE),
        starter_only_sources=(SampleSource.ORIGINAL,),
        allow_guidance=False,
        allow_perturbations=False,
    )

    starter = sampler.sample_cases(SampleCasesRequest(SampleSource.ORIGINAL, 1))
    assert len(starter.cases) == 1
    sampler.seal_starter()

    with pytest.raises(PermissionError, match="Controller-provided starter"):
        sampler.sample_cases(SampleCasesRequest(SampleSource.ORIGINAL, 1))
    assert len(sampler.sample_cases(SampleCasesRequest(SampleSource.BASE, 1)).cases) == 1


def test_only_projected_paired_batches_may_redact_parent_lineage() -> None:
    raw_root = replace(
        _case("ordinary-root", parent_case_id=None),
        source=CaseSource.GENERATED,
    )
    raw_child = _case("ordinary-child", parent_case_id=None)
    with pytest.raises(ValueError, match="preserve root order and lineage"):
        PairedSampleBatch((raw_root,), (raw_child,), 0, ())

    linked_child = replace(raw_child, parent_case_id=raw_root.case_id)
    projected = PairedSampleBatch(
        (candidate_facing_case(raw_root),),
        (candidate_facing_case(linked_child),),
        0,
        (),
    )
    assert projected.children[0].parent_case_id is None
