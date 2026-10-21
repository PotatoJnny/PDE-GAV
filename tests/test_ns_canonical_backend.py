from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.engine.guidance_controller import (
    GuidanceProtocolConfig,
    GuidanceRewardArtifact,
)
from pde_agent_verifier_solver.engine.campaign import (
    AgentSpec,
    CampaignCompiler,
    CampaignRecipe,
    DevelopmentSamplingPolicy,
    EvaluatorPolicy,
    GeneratorRecipe,
    RoundSettings,
)
from pde_agent_verifier_solver.plugins.ns.canonical_backend import (
    CanonicalStructuredNSBackend,
)
from pde_agent_verifier_solver.plugins.ns.generator import NSStructuredGenerator
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Artifact,
    Case,
    CaseAccessKind,
    CaseFeedback,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    FrozenGuidanceSnapshot,
    GenerationKind,
    GenerationPurpose,
    GenerationRequest,
    GeneratorCheckpoint,
    GeneratorFeedback,
    GeneratorMode,
    GuidanceCapability,
    GuidanceTrainingBudget,
    GuidedAnswerFirstSample,
    HiddenSuiteSpec,
    OfficialEvaluationSpec,
    QueryKind,
    RoundFeedback,
    SampleSource,
    SamplingStrategy,
    TaskKind,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _Truth:
    def answer(self, query: AnswerQuery) -> AnswerView:
        return AnswerView(
            kind=query.kind,
            fields={field: (0.0,) * len(query.points) for field in query.fields},
            points=query.points,
            shape=(len(query.points),),
        )


class _Session:
    def __init__(self, case_id: str) -> None:
        self.case_id = case_id


class _Source:
    generator_id = "ns-test-frozen-generator"
    artifact_sha256 = _digest("ns-test-frozen-generator")
    supports_perturbations = True
    public_session_methods = ("query_forcing", "query_initial_velocity")
    development_solution_session_methods = ("query_completed_velocity",)

    def __init__(self) -> None:
        self.requests = []
        self.issued: dict[str, CaseRecord] = {}
        self.released: list[tuple[str, ...]] = []

    def sample(self, request):
        self.requests.append(request)
        visibility = (
            CaseVisibility.DEVELOPMENT
            if request.cohort.value == "development"
            else CaseVisibility.HIDDEN
        )
        records = []
        for index in range(request.count):
            parent = (
                request.anchors[index]
                if request.kind is GenerationKind.PERTURBED
                else None
            )
            source = (
                CaseSource.PERTURBED
                if parent is not None
                else CaseSource.GENERATED
            )
            case = Case(
                case_id=(
                    f"ns-{request.cohort.value}-{request.seed}-{index}-"
                    f"{request.kind.value}"
                ),
                source=source,
                visibility=visibility,
                public_data={"equation": "NS"},
                round_index=request.round_index,
                parent_case_id=(parent.case_id if parent is not None else None),
            )
            record = CaseRecord(case, _Truth())
            self.issued[case.case_id] = record
            records.append(record)
        return tuple(records)

    def register(self, record: CaseRecord) -> None:
        self.issued[record.case.case_id] = record

    def release_cases(self, case_ids):
        identifiers = tuple(case_ids)
        self.released.append(identifiers)
        for case_id in identifiers:
            self.issued.pop(case_id, None)

    def public_session_factory(self, case):
        assert case.case_id in self.issued
        return _Session(case.case_id)

    def development_solution_session_factory(self, case):
        assert case.case_id in self.issued
        return _Session(case.case_id)


class _GuidedOracle:
    development_solution_methods = ("query_completed_velocity",)

    def __init__(self, backend, snapshot) -> None:
        self.backend = backend
        self.snapshot = snapshot
        self.guidance_capability = backend.guidance_capability
        self.records = {}
        self.closed = False

    def generate_answer_first(self, request):
        self.backend.guided_requests.append(request)
        samples = []
        for index in range(request.spec.sample_count):
            case = Case(
                case_id=(
                    f"ns-guided-{request.controller_seed}-{index}-"
                    f"{request.purpose.value}"
                ),
                source=CaseSource.GENERATED,
                visibility=request.expected_visibility,
                public_data={
                    "distribution_lineage": {
                        "kind": "frozen_reward_guided",
                        "snapshot_sha256": self.snapshot.snapshot_sha256,
                    }
                },
                round_index=self.snapshot.training_request.target_round_index,
            )
            record = CaseRecord(case, _Truth())
            sample_id = f"guided-sample-{request.controller_seed}-{index}"
            self.records[sample_id] = record
            self.backend.source.register(record)
            samples.append(
                GuidedAnswerFirstSample(
                    sample_id=sample_id,
                    snapshot_sha256=request.snapshot_sha256,
                    case=case,
                    reference_answer_sha256=_digest(f"answer-{sample_id}"),
                    public_case_sha256=_digest(f"case-{sample_id}"),
                    answer_to_case_derivation_sha256=_digest("ns-test-derivation"),
                    answer_access_locator=(
                        sample_id
                        if request.expected_visibility is CaseVisibility.DEVELOPMENT
                        else None
                    ),
                )
            )
        return tuple(samples)

    def resolve_case_records(self, batch):
        return tuple(self.records[sample.sample_id] for sample in batch.samples)

    def close(self):
        self.closed = True
        self.backend.source.release_cases(
            tuple(record.case.case_id for record in self.records.values())
        )


class _GuidanceBackend:
    guidance_capability = GuidanceCapability.online("ns-test-guidance-v1")

    def __init__(self, source: _Source, protocol: GuidanceProtocolConfig) -> None:
        self.source = source
        self.config = SimpleNamespace(minimum_cases=2)
        self.guidance_config_sha256 = protocol.guidance_config_sha256
        self.official_sample_settings = protocol.official_sample_settings
        self.feedback = []
        self.training_requests = []
        self.guided_requests = []

    def seal_reward_feedback(self, feedback):
        self.feedback.append(feedback)
        return GuidanceRewardArtifact("memory://ns-feedback", _digest("feedback"))

    def train_guidance(self, request):
        self.training_requests.append(request)
        return FrozenGuidanceSnapshot(
            training_request=request,
            backend_id=self.guidance_capability.backend_id,
            guided_artifact_uri="memory://ns-guide",
            guided_artifact_sha256=_digest("guide"),
            campaign_guidance_training_wall_seconds=0.0,
        )

    def open_guidance(self, snapshot):
        return _GuidedOracle(self, snapshot)


def _protocol() -> GuidanceProtocolConfig:
    return GuidanceProtocolConfig(
        training_budget=GuidanceTrainingBudget(10.0, "one-test-device"),
        guidance_config_sha256=_digest("guidance-config"),
        reward_feedback_case_count=2,
        official_round_case_count=2,
        official_terminal_case_count=2,
        inference_wall_budget_seconds=10.0,
        inference_compute_capacity="one-test-device",
        official_sample_settings={"temperature": 0.0},
    )


def _request(
    *,
    round_index: int,
    seed: int,
    kind: GenerationKind,
    visibility: CaseVisibility,
    purpose: GenerationPurpose,
    count: int = 2,
    anchors=(),
) -> GenerationRequest:
    return GenerationRequest(
        round_index=round_index,
        count=count,
        mode=GeneratorMode.ONLINE,
        kind=kind,
        visibility=visibility,
        seed=seed,
        purpose=purpose,
        anchors=tuple(anchors),
    )


def _artifact(round_index: int = 0) -> Artifact:
    return Artifact(
        artifact_id=f"candidate-{round_index}",
        kind=TaskKind.SOLVER,
        name="candidate",
        program=object(),
        summary="test",
        usage="test",
        round_index=round_index,
        agent_id="agent-a",
    )


def test_canonical_ns_backend_routes_base_perturb_sessions_and_release():
    source = _Source()
    backend = CanonicalStructuredNSBackend(
        SimpleNamespace(case_source=source),
    )
    generator = NSStructuredGenerator(backend)
    state = generator.load(
        GeneratorCheckpoint(generator.info.program_id, backend.base_state),
        {},
    )

    roots = generator.sample(
        state,
        _request(
            round_index=1,
            seed=11,
            kind=GenerationKind.BASE,
            visibility=CaseVisibility.DEVELOPMENT,
            purpose=GenerationPurpose.DEVELOPMENT,
        ),
    )
    children = generator.sample(
        state,
        _request(
            round_index=1,
            seed=12,
            kind=GenerationKind.PERTURBED,
            visibility=CaseVisibility.DEVELOPMENT,
            purpose=GenerationPurpose.DEVELOPMENT,
            anchors=tuple(record.case for record in roots),
        ),
    )

    assert [row.case.parent_case_id for row in children] == [
        row.case.case_id for row in roots
    ]
    assert generator.case_session_methods(CaseAccessKind.PUBLIC_PROBLEM) == (
        "query_forcing",
        "query_initial_velocity",
    )
    assert generator.case_session_methods(CaseAccessKind.DEVELOPMENT_SOLUTION) == (
        "query_completed_velocity",
    )
    assert generator.open_case_session(
        state, roots[0].case, CaseAccessKind.PUBLIC_PROBLEM
    ).case_id == roots[0].case.case_id
    generator.release(state, tuple(row.case.case_id for row in (*roots, *children)))
    assert not source.issued
    with pytest.raises(RuntimeError, match="already-frozen checkpoint"):
        generator.train(None, {})


def test_canonical_ns_backend_freezes_feedback_then_samples_guided_cases():
    source = _Source()
    protocol = _protocol()
    guidance = _GuidanceBackend(source, protocol)
    backend = CanonicalStructuredNSBackend(
        SimpleNamespace(case_source=source),
        guidance_backend=guidance,
        guidance_protocol=protocol,
    )
    generator = NSStructuredGenerator(backend)
    state = generator.load(
        GeneratorCheckpoint(generator.info.program_id, backend.base_state),
        {},
    )
    base = generator.sample(
        state,
        _request(
            round_index=0,
            seed=21,
            kind=GenerationKind.BASE,
            visibility=CaseVisibility.HIDDEN,
            purpose=GenerationPurpose.GUIDANCE_FEEDBACK,
            count=3,
        ),
    )
    child = generator.sample(
        state,
        _request(
            round_index=0,
            seed=22,
            kind=GenerationKind.PERTURBED,
            visibility=CaseVisibility.HIDDEN,
            purpose=GenerationPurpose.GUIDANCE_FEEDBACK,
            count=1,
            anchors=(base[0].case,),
        ),
    )[0]
    guided_decoy = Case(
        "guided-decoy",
        CaseSource.GENERATED,
        CaseVisibility.HIDDEN,
        public_data={"distribution_lineage": {"kind": "frozen_reward_guided"}},
        round_index=0,
    )
    feedback = GeneratorFeedback(
        source_round_index=0,
        target_round_index=1,
        selected_program_id="candidate-0",
        selected_program=_artifact(0),
        cases=tuple(
            CaseFeedback(case, 0.1 + index, {"velocity_relative_l2": 0.1 + index})
            for index, case in enumerate(
                (
                    base[0].case,
                    child.case,
                    guided_decoy,
                    base[1].case,
                    base[2].case,
                )
            )
        ),
    )

    guided_state = generator.update_guidance(
        state,
        feedback,
        {"controller_seed": 31},
    )
    assert [row.case.case_id for row in guidance.feedback[0].rows] == [
        base[0].case.case_id,
        base[1].case.case_id,
        base[2].case.case_id,
    ]
    assert len(guidance.feedback[0].rows) == 3
    guided = generator.sample(
        guided_state,
        _request(
            round_index=1,
            seed=32,
            kind=GenerationKind.GUIDED,
            visibility=CaseVisibility.DEVELOPMENT,
            purpose=GenerationPurpose.DEVELOPMENT,
        ),
    )

    assert len(guided) == 2
    assert all(row.case.visibility is CaseVisibility.DEVELOPMENT for row in guided)
    assert guidance.guided_requests[0].spec.settings == {}

    next_base = generator.sample(
        guided_state,
        _request(
            round_index=1,
            seed=33,
            kind=GenerationKind.BASE,
            visibility=CaseVisibility.HIDDEN,
            purpose=GenerationPurpose.PROMOTION,
            count=2,
        ),
    )
    next_feedback = GeneratorFeedback(
        source_round_index=1,
        target_round_index=2,
        selected_program_id="candidate-1",
        selected_program=_artifact(1),
        cases=tuple(
            CaseFeedback(row.case, 0.2 + index, {"velocity_relative_l2": 0.2 + index})
            for index, row in enumerate(next_base)
        ),
    )
    round_two_state = generator.update_guidance(
        guided_state,
        next_feedback,
        {"controller_seed": 34},
    )
    assert round_two_state.guidance.snapshot.training_request.target_round_index == 2
    assert len(guidance.training_requests) == 2
    with pytest.raises(RuntimeError, match="already attempted"):
        generator.update_guidance(
            state,
            feedback,
            {"controller_seed": 35},
        )
    generator.release(
        round_two_state,
        tuple(
            row.case.case_id
            for row in (*base, child, *guided, *next_base)
        ),
    )
    assert not source.issued


def test_canonical_ns_guidance_rejects_wrong_feedback_scale():
    source = _Source()
    protocol = _protocol()
    backend = CanonicalStructuredNSBackend(
        SimpleNamespace(case_source=source),
        guidance_backend=_GuidanceBackend(source, protocol),
        guidance_protocol=protocol,
    )
    only = source.sample(
        SimpleNamespace(
            round_index=0,
            cohort=SimpleNamespace(value="guidance_feedback"),
            seed=90,
            count=1,
            kind=GenerationKind.BASE,
            anchors=(),
        )
    )[0]
    with pytest.raises(ValueError, match="at least 2"):
        backend.fit_guidance(
            backend.base_state,
            None,
            RoundFeedback(
                round_index=0,
                selected_artifact=_artifact(0),
                cases=(CaseFeedback(only.case, 0.1, {}),),
            ),
            {"controller_seed": 1},
        )


def test_bound_frozen_ns_generator_is_accepted_by_canonical_compiler():
    source = _Source()
    protocol = _protocol()
    backend = CanonicalStructuredNSBackend(
        SimpleNamespace(case_source=source),
        guidance_backend=_GuidanceBackend(source, protocol),
        guidance_protocol=protocol,
    )
    generator = NSStructuredGenerator(backend)
    checkpoint = GeneratorCheckpoint(generator.info.program_id, backend.base_state)
    policy = DevelopmentSamplingPolicy(
        strategy=SamplingStrategy.AGENT_ADAPTIVE,
        max_cases_per_agent_per_round=4,
        allowed_sources=(SampleSource.BASE,),
    )
    recipe = CampaignRecipe(
        name="canonical numerical NS compile",
        round_count=1,
        agents=(AgentSpec("agent", "Agent"),),
        default_round=RoundSettings(
            agent_budget_seconds=10,
            development_sampling=policy,
            official_evaluation=OfficialEvaluationSpec(
                round=HiddenSuiteSpec(base=2),
                terminal=HiddenSuiteSpec(base=2),
            ),
        ),
        generator=GeneratorRecipe(
            guidance_enabled=True,
            perturbations_enabled=False,
            checkpoint=checkpoint,
        ),
        evaluator=EvaluatorPolicy(
            AnswerQuery.at_points(
                fields=("velocity_x", "velocity_y"),
                points=({"t": 0.5, "x": 0.5, "y": 0.5},),
            )
        ),
    )

    compiled = CampaignCompiler().compile(recipe, generator.capabilities)

    assert NSStructuredGenerator().capabilities.online is False
    assert generator.capabilities.online is True
    assert generator.capabilities.guidance is True
    assert generator.capabilities.trainable is False
    assert compiled.generator.checkpoint is checkpoint


@pytest.mark.parametrize("purpose", [GenerationPurpose.PROMOTION, GenerationPurpose.TERMINAL])
def test_canonical_ns_hidden_guided_and_perturbed_survive_disabled_development_tools(purpose):
    source = _Source()
    protocol = _protocol()
    guidance = _GuidanceBackend(source, protocol)
    backend = CanonicalStructuredNSBackend(
        SimpleNamespace(case_source=source),
        guidance_backend=guidance,
        guidance_protocol=protocol,
        development_guidance_enabled=False,
        development_perturbations_enabled=False,
    )
    generator = NSStructuredGenerator(backend)
    state = generator.load(GeneratorCheckpoint(generator.info.program_id, backend.base_state), {})
    feedback_records = generator.sample(state, _request(
        round_index=0, seed=200, kind=GenerationKind.BASE,
        visibility=CaseVisibility.HIDDEN, purpose=GenerationPurpose.GUIDANCE_FEEDBACK,
    ))
    state = generator.update_guidance(state, GeneratorFeedback(
        source_round_index=0, target_round_index=1,
        selected_program_id="candidate-0", selected_program=_artifact(0),
        cases=tuple(CaseFeedback(record.case, 0.5, {"velocity_relative_l2": 0.5}) for record in feedback_records),
    ), {"controller_seed": 201})
    assert generator.capabilities.guidance is True
    assert generator.capabilities.perturbations is True

    hidden_guided = generator.sample(state, _request(
        round_index=1, seed=202, kind=GenerationKind.GUIDED,
        visibility=CaseVisibility.HIDDEN, purpose=purpose,
    ))
    hidden_perturbed = generator.sample(state, _request(
        round_index=1, seed=203, kind=GenerationKind.PERTURBED,
        visibility=CaseVisibility.HIDDEN, purpose=purpose,
        anchors=tuple(record.case for record in hidden_guided),
    ))
    assert len(hidden_guided) == len(hidden_perturbed) == 2
    assert all(record.case.visibility is CaseVisibility.HIDDEN for record in (*hidden_guided, *hidden_perturbed))
    assert [row.case.parent_case_id for row in hidden_perturbed] == [row.case.case_id for row in hidden_guided]

    request_counts = (len(source.requests), len(guidance.guided_requests))
    for kind in (GenerationKind.GUIDED, GenerationKind.PERTURBED):
        with pytest.raises(PermissionError, match="NS development"):
            generator.sample(state, _request(
                round_index=1, seed=204, kind=kind,
                visibility=CaseVisibility.DEVELOPMENT, purpose=GenerationPurpose.DEVELOPMENT,
                anchors=tuple(record.case for record in hidden_guided) if kind is GenerationKind.PERTURBED else (),
            ))
    assert (len(source.requests), len(guidance.guided_requests)) == request_counts
    development_base = generator.sample(state, _request(
        round_index=1, seed=205, kind=GenerationKind.BASE,
        visibility=CaseVisibility.DEVELOPMENT, purpose=GenerationPurpose.DEVELOPMENT,
    ))
    assert len(development_base) == 2
