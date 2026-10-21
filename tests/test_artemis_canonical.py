"""Small interface evidence, not Artemis physical/model qualification."""

from dataclasses import replace
from pathlib import Path
import random
from threading import RLock
from types import SimpleNamespace
from types import MappingProxyType

import numpy as np
import pytest

from pde_agent_verifier_solver.app.backend_run import BackendRunBinding
from pde_agent_verifier_solver.engine.campaign import (
    AgentSpec, CampaignRecipe, CaseCounts, DevelopmentSamplingPolicy,
    EvaluatorPolicy, GeneratorRecipe, RoundSettings,
)
from pde_agent_verifier_solver.engine.evaluator import EvaluationRequest, HiddenVerifier
from pde_agent_verifier_solver.engine.runtime import AgentOutcome
from pde_agent_verifier_solver.plugins.vp_1d1v import (
    ArtemisBackend, ArtemisGenerator, LegacyMacroCandidateAdapter,
    VP1D_MACRO_FIELDS, build_artemis_plugin, causal_horizon_case,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, AnswerView, Artifact, Case, CaseRecord, CaseSource, CaseVisibility,
    CohortKind, DatasetInput, DatasetKind, EvaluationPhase, GenerationKind,
    GenerationPurpose, GenerationRequest, GeneratorMode, HiddenSuiteSpec,
    OfficialEvaluationSpec, SampleSource, open_public_case_session,
    open_development_solution_session,
)


def _answer(query, factor=1.0):
    return AnswerView(kind=query.kind, points=query.points, shape=(len(query.points),),
        fields={name: tuple(factor * (index + 1) * (1 + p["t"] + p["x"]) for p in query.points)
                for index, name in enumerate(VP1D_MACRO_FIELDS)})


class _Truth:
    def answer(self, query):
        return _answer(query)


class _Session:
    def __init__(self, case, solution=False):
        self.case, self.solution, self.closed = case, solution, False

    def query_initial_distribution(self, points):
        assert not self.closed
        return np.ones(len(points))

    def query_source(self, points):
        assert not self.closed
        return np.asarray(points).sum(axis=1)

    def query_completed_macros(self, points):
        assert self.solution and not self.closed
        return np.column_stack([(i + 1) * (1 + np.asarray(points).sum(axis=1)) for i in range(5)])

    def close(self):
        assert not self.closed
        self.closed = True


class _Source:
    artifact_sha256 = "1" * 64
    supports_perturbations = True
    public_oracle_mode = "exact_coordinate"

    def __init__(self):
        self.records, self.requests, self.sessions, self.released = {}, [], [], []

    def sample(self, request):
        self.requests.append(request)
        records = []
        for index in range(request.count):
            identity = f"{request.cohort.value}-{request.kind.value}-{request.seed}-{index}"
            t, x, v = np.array([0., .5, 1.]), np.array([0., .25, .5, .75]), np.array([-1., 0., 1.])
            arrays = {"t": t, "x": x, "v": v, "f0": np.ones((4, 3)), "source": np.zeros((3, 4, 3))}
            for array in arrays.values():
                array.setflags(write=False)
            equation = {"case_id": identity, "equation_version": "vp_1d1v_forced_periodic_v1",
                "domain": {"time_interval": (0., 1.), "x_interval": (0., 1.), "velocity_query_window": (-1., 1.)},
                "oracle_protocol": {"schema": "vp_1d1v_exact_coordinate_oracle_v1"}}
            parent = request.anchors[index] if request.anchors else None
            case = Case(identity, CaseSource.PERTURBED if parent else CaseSource.GENERATED,
                CaseVisibility.DEVELOPMENT if request.cohort is CohortKind.DEVELOPMENT else CaseVisibility.HIDDEN,
                {"equation": equation, "arrays": MappingProxyType(arrays)}, request.round_index,
                parent_case_id=parent.case_id if parent else None)
            record = CaseRecord(case, _Truth())
            self.records[identity] = record
            records.append(record)
        return tuple(records)

    def public_session_factory(self, case):
        assert case is self.records[case.case_id].case
        session = _Session(case)
        self.sessions.append(session)
        return session

    def development_solution_session_factory(self, case):
        if case.visibility is not CaseVisibility.DEVELOPMENT:
            raise PermissionError("development only")
        session = _Session(case, solution=True)
        self.sessions.append(session)
        return session

    def release_cases(self, ids):
        for identity in ids:
            self.records.pop(identity, None)
        self.released.extend(ids)


class _MacroOnly:
    def __init__(self, factor=1., retain_session=True):
        self.factor, self.saved_source = factor, None
        self.retain_session = retain_session

    def predict_observables(self, case, query, budget=None):
        # The output contains only macroscopic fields, while the input exposes
        # the PDE source prefix. No f(t,x,v) prediction is requested anywhere.
        horizon = case.public_data["equation"]["domain"]["time_interval"][1]
        assert all(p["t"] == horizon for p in query.points)
        assert float(case.public_data["arrays"]["t"][-1]) == horizon
        assert case.public_data["arrays"]["source"].shape[0] == len(case.public_data["arrays"]["t"])
        session = open_public_case_session(case)
        if self.retain_session:
            self.saved_source = session.query_source
        np.testing.assert_allclose(session.query_source(np.array([[horizon, .17, .13]])), [horizon + .3])
        with pytest.raises(ValueError, match="causal"):
            session.query_source(np.array([[horizon + .01, .17, .13]]))
        with pytest.raises(RuntimeError):
            open_development_solution_session(case)
        return _answer(query, self.factor)


def _artifact(program, round_index=1):
    return Artifact(f"macro-r{round_index}", build_artemis_plugin().task().kind, "Macros only", program,
                    "No kinetic output", "predict_observables", round_index, "agent")


def _binding():
    source = _Source()
    backend = ArtemisBackend(source, physical_revision="test-double-not-physical-evidence")
    plugin = build_artemis_plugin(backend)
    dataset = plugin.problem.prepare_dataset(DatasetInput("online", DatasetKind.NUMERICAL, settings={"online_only": True}))
    return BackendRunBinding(plugin, backend.checkpoint(), dataset), source


@pytest.mark.parametrize("oracle_settings", [None, {
    "exact_query_tx_reuse": "request", "exact_query_source_mode": "source_only",
    "factorized_jvp_mode": "forward",
}, {"factorized_jvp_mode": "reverse"}])
def test_backend_factory_forwards_explicit_oracle_options_without_changing_default(monkeypatch, oracle_settings):
    from pde_agent_verifier_solver.plugins.vp_1d1v import product

    captured = {}
    def runtime_factory(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)
    source = _Source()
    def source_factory(runtime, **kwargs):
        assert runtime.config.time_count == 9
        assert kwargs["expected_offline_admission_sha256"] == "2" * 64
        return source
    admission = SimpleNamespace(record_sha256="2" * 64, checkpoint_sha256="3" * 64)
    startup = SimpleNamespace(admission=admission)
    loaded = SimpleNamespace(artifact=object(), sensor_candidate=object())
    modules = {
        "artemis_compact_v5_startup": SimpleNamespace(load_artemis_compact_v5_startup_artifacts=lambda *args, **kwargs: loaded),
        "artemis_compact_v5_production_runtime": SimpleNamespace(
            ArtemisCompactV5ProductionRuntimeConfig=SimpleNamespace,
            ArtemisCompactV5OracleOptions=SimpleNamespace,
            VP1DArtemisCompactV5ProductionRuntime=runtime_factory),
        "manufactured_backend": SimpleNamespace(VP1DManufacturedCaseSource=source_factory),
    }
    package = SimpleNamespace(module=modules.__getitem__)
    monkeypatch.setattr(product, "preflight_artemis_backend", lambda **kwargs: (package, startup))
    optional = {} if oracle_settings is None else {"oracle_settings": oracle_settings}
    binding = product.load_artemis_backend(
        physical_checkout="unused", admission_record_path="unused",
        expected_admission_record_sha256=admission.record_sha256,
        science_config_path="unused", expected_science_config_sha256="4" * 64,
        runtime_settings={"time_count": 9}, **optional,
    )
    assert binding.plugin.generator.backend.source is source
    assert binding.report["oracle_optimization_settings"] == (oracle_settings or {})
    if oracle_settings is None:
        assert "oracle_options" not in captured  # The physical default is retained.
        assert binding.report["oracle_optimization_qualification"] == "not_requested"
    else:
        assert vars(captured["oracle_options"]) == oracle_settings
        assert binding.report["oracle_optimization_qualification"] == "experimental_equivalence_pending"


def _query():
    return AnswerQuery.at_points(fields=VP1D_MACRO_FIELDS, points=({"t": .5, "x": .25}, {"t": .5, "x": .5}))


def _request(kind=GenerationKind.BASE, visibility=CaseVisibility.HIDDEN, purpose=GenerationPurpose.SELECTION, anchors=()):
    return GenerationRequest(1, 1, GeneratorMode.ONLINE, kind, visibility,
                             seed=42 if not anchors else 43, purpose=purpose, anchors=anchors)


@pytest.mark.parametrize("phase", [EvaluationPhase.OFFICIAL, EvaluationPhase.TERMINAL])
def test_macro_only_native_match_causal_oracle_and_revocation(phase):
    binding, source = _binding()
    generator = binding.plugin.generator
    state = generator.load(binding.checkpoint, {})
    record = generator.sample(state, _request())[0]
    model = _MacroOnly()
    verifier = HiddenVerifier(binding.plugin.task(), _query(), phase=phase,
                              generator=generator, generator_state=state)
    result = verifier.evaluate(_artifact(model), (record,))
    assert result.primary_error == 0.
    assert result.cases[0].macro_match.contract_id == "vp_1d1v_m0_m3_e_match_v1"
    assert source.sessions and all(s.closed for s in source.sessions)
    with pytest.raises(RuntimeError, match="no longer active"):
        model.saved_source(np.array([[.5, .17, .13]]))
    generator.release(state, (record.case.case_id,))
    assert not source.records


def test_source_mapping_pair_lineage_checkpoint_and_guidance_flags():
    binding, source = _binding()
    generator = binding.plugin.generator
    state = generator.load(binding.checkpoint, {})
    root = generator.sample(state, _request())[0]
    child = generator.sample(state, _request(GenerationKind.PERTURBED, anchors=(root.case,)))[0]
    assert child.case.parent_case_id == root.case.case_id
    assert source.requests[-1].anchors == (root.case,)
    assert generator.capabilities.perturbations and not generator.capabilities.guidance
    with pytest.raises(ValueError, match="fitted guidance"):
        generator.sample(state, _request(GenerationKind.GUIDED))
    with pytest.raises(ValueError, match="checkpoint"):
        generator.backend.generate(object(), None, _request())


def test_causal_projection_detaches_arrays_and_rejects_mixed_horizons():
    binding, _ = _binding()
    generator = binding.plugin.generator
    state = generator.load(binding.checkpoint, {})
    case = generator.sample(state, _request())[0].case
    projected = causal_horizon_case(case, _query())
    assert not np.shares_memory(projected.public_data["arrays"]["source"], case.public_data["arrays"]["source"])
    assert case.public_data["arrays"]["source"].shape[0] == 3
    with pytest.raises(ValueError, match="share one"):
        causal_horizon_case(case, AnswerQuery.at_points(fields=VP1D_MACRO_FIELDS,
            points=({"t": .5, "x": .25}, {"t": 1., "x": .5})))


def test_old_macro_abi_and_query_blind_prepared_solution_are_explicit():
    class Old:
        def predict_macros(self, case, query, budget=None):
            return _answer(query)
    assert LegacyMacroCandidateAdapter(Old()).predict_observables(None, _query()).fields.keys() == set(VP1D_MACRO_FIELDS)
    events = []
    class Prepared:
        def prepare_solution(self, case, solve_budget=None):
            events.append("prepare_without_query")
            class Handle:
                def evaluate(self, query):
                    events.append("evaluate")
                    return _answer(query)
            return Handle()
    binding, _ = _binding()
    generator = binding.plugin.generator
    state = generator.load(binding.checkpoint, {})
    record = generator.sample(state, _request())[0]
    verifier = HiddenVerifier(binding.plugin.task(), _query(), {"require_prepared_solution": True},
                              generator=generator, generator_state=state)
    result = verifier.evaluate(_artifact(Prepared()), (record,))
    assert events == ["prepare_without_query", "evaluate"]
    assert result.primary_error == 0.
    assert result.metrics["query_blind_preparation"] == 1.


@pytest.mark.parametrize("setting,value", [("causal_horizon", "false"), ("require_prepared_solution", 1), ("candidate_compute_budget", float("nan")), ("candidate_compute_budget", -1.)])
def test_native_evaluation_rejects_invalid_settings(setting, value):
    with pytest.raises((TypeError, ValueError)):
        build_artemis_plugin().task().create_evaluation({"default_query": _query(), setting: value})


def test_default_query_snapshot_and_invalid_prepared_handle():
    query = _query()
    task = build_artemis_plugin().task()
    evaluation = task.create_evaluation({"default_query": query})
    query.points[0]["x"] = .75
    assert evaluation.default_query.points[0]["x"] == .25
    class BrokenPrepared:
        def prepare_solution(self, case, solve_budget=None):
            return None
    binding, source = _binding()
    generator = binding.plugin.generator
    state = generator.load(binding.checkpoint, {})
    record = generator.sample(state, _request())[0]
    verifier = HiddenVerifier(evaluation, _query(), generator=generator, generator_state=state)
    with pytest.raises(TypeError, match="handle with evaluate"):
        verifier.evaluate(_artifact(BrokenPrepared()), (record,))
    assert not source.sessions  # No oracle opens eagerly during preparation.


def test_physical_session_failure_propagates_without_array_fallback():
    binding, source = _binding()
    generator = binding.plugin.generator
    state = generator.load(binding.checkpoint, {})
    record = generator.sample(state, _request())[0]
    def fail(case):
        raise RuntimeError("physical runtime closed unexpectedly")
    source.public_session_factory = fail
    verifier = HiddenVerifier(binding.plugin.task(), _query(), generator=generator, generator_state=state)
    with pytest.raises(RuntimeError, match="physical runtime closed unexpectedly"):
        verifier.evaluate(_artifact(_MacroOnly()), (record,))


class _MacroAgent:
    def run(self, context):
        candidate = _artifact(_MacroOnly(factor=1.0 - .1 / context.round_plan.round_index, retain_session=False), context.round_plan.round_index)
        result = context.evaluator.evaluate(candidate, EvaluationRequest(tuple(c.case_id for c in context.cases)))
        assert result.primary_error >= 0.
        candidate.program.saved_source = None
        return AgentOutcome(candidate, ("Evaluated macro-only Candidate",), round_log="Macro-only interface test")


def test_two_round_canonical_campaign_promotes_and_terminal_scores_macro_only():
    binding, source = _binding()
    recipe = CampaignRecipe(
        name="Artemis canonical interface test", round_count=2,
        agents=(AgentSpec("agent", "Macro Agent"),),
        default_round=RoundSettings(
            development_sampling=DevelopmentSamplingPolicy(
                max_cases_per_agent_per_round=2,
                allowed_sources=(SampleSource.BASE,),
                starter=CaseCounts(base=1, base_perturbed=1)),
            official_evaluation=OfficialEvaluationSpec(
                round=HiddenSuiteSpec(base=2, base_perturbed=2),
                terminal=HiddenSuiteSpec(base=2, base_perturbed=2))),
        generator=GeneratorRecipe(guidance_enabled=False, perturbations_enabled=True),
        evaluator=EvaluatorPolicy(_query(), development_point_count=2),
    )
    result = binding.run(recipe, {"agent": _MacroAgent()}, local_development=True)
    assert len(result.rounds) == len(result.terminal_round_reports) == 2
    assert result.rounds[1].selected.primary_error < result.rounds[0].selected.primary_error
    assert result.selected.artifact.kind.value == "observable_model"
    assert result.selected.artifact.round_index == 2
    assert result.selected.primary_error == pytest.approx(.05)
    assert result.memory.rounds
    assert source.sessions and all(session.closed for session in source.sessions)
    assert source.released and not source.records
    assert {request.cohort for request in source.requests} >= {CohortKind.DEVELOPMENT, CohortKind.PROMOTION, CohortKind.TERMINAL}


def test_pinned_reward_math_accepts_canonical_feedback_and_preserves_zero_guidance_capsules():
    pytest.importorskip("torch")
    # The backend deployment uses a separate frozen optimized physical tree.
    checkout = Path(__file__).resolve().parents[2] / "artemis-oracle-opt2-20260907"
    if not checkout.is_dir():
        pytest.skip("pinned external Artemis source is not installed")
    from pde_agent_verifier_solver.plugins.vp_1d1v import ArtemisPhysicalPackage, ArtemisRewardGuidance
    from pde_agent_verifier_solver.sdk import CaseFeedback, RoundFeedback
    from pde_agent_verifier_solver.sdk.macro_loop import SampleRequest

    package = ArtemisPhysicalPackage(checkout)
    capsule_type = package.module("artemis_compact_v5_production_runtime").ArtemisCompactV5GuidanceCoordinateCapsule
    class Runtime:
        conditions = ("a", "b")
        def guidance_coordinate_from_seed(self, seed):
            rng = random.Random(seed)
            return capsule_type.from_unit_uniforms(self.conditions[seed % 2], tuple(rng.random() for _ in range(3)))
        def replay_guidance_feedback_coordinate(self, manufactured):
            return manufactured
    class Source(_Source):
        def __init__(self):
            super().__init__()
            self.runtime, self._roots, self._lock = Runtime(), {}, RLock()
            self.last_capsules = ()
        def issue_guided_answer_first_records(self, capsules, **kwargs):
            self.last_capsules = tuple(capsules)
            return self.sample(SampleRequest(kwargs["round_index"], CohortKind.DEVELOPMENT,
                kwargs["controller_seed"], len(capsules)))
    source = Source()
    guide = ArtemisRewardGuidance(package, source, settings={
        "condition_count": 2, "model_kind": "ou_hermite_ridge_v1",
        "hermite_degree": 1, "crossfit_fold_count": 2, "crossfit_repeat_count": 1,
        "guidance_strength": 0.0, "sampling_steps": 2,
    })
    rows, records = [], []
    for index in range(24):
        record = source.sample(SampleRequest(1, CohortKind.PROMOTION, index, 1))[0]
        capsule = source.runtime.guidance_coordinate_from_seed(index)
        source._roots[record.case.case_id] = SimpleNamespace(manufactured=capsule)
        records.append(record)
        rows.append(CaseFeedback(record.case, .1 + (index % 5) / 5., {}))
    guide.retain_feedback_coordinates(records)
    backend = ArtemisBackend(source, physical_revision="test", guidance_adapter=guide)
    state = guide.fit_guidance(backend.checkpoint_payload, None, RoundFeedback(1, _artifact(_MacroOnly()), tuple(rows)), {"controller_seed": 7})
    assert state.report["base_model_updated"] is False
    request = GenerationRequest(2, 3, GeneratorMode.ONLINE, GenerationKind.GUIDED,
        CaseVisibility.DEVELOPMENT, seed=50, purpose=GenerationPurpose.DEVELOPMENT)
    generated = guide.generate(backend.checkpoint_payload, state, request)
    assert len(generated) == 3
    assert guide.last_sampling_report["exact_raw_capsule_reuse_count"] == 3
    assert guide.last_sampling_report["raw_primitive_source"] == "provided_canonical_base_capsules"
    assert guide.training_audits[0]["feedback_case_count"] == 24
    assert guide.training_audits[0]["target_solver_id"] == state.target_solver_id
    assert guide.sampling_audits[0]["snapshot_sha256"] == state.snapshot_sha256
    assert guide.sampling_audits[0]["count"] == 3
    guide.release_feedback_coordinates(tuple(record.case.case_id for record in records))
    with pytest.raises(ValueError, match="released or unknown"):
        guide.fit_guidance(backend.checkpoint_payload, state, RoundFeedback(1, _artifact(_MacroOnly()), tuple(rows)), {})
    assert len(guide.training_audits) == 1  # Failed fits cannot appear as completed.


def test_nonzero_guidance_configuration_runs_s0_official_and_final_controller_lifecycle():
    """Synthetic physics fixture: engineering lifecycle, not model quality."""
    pytest.importorskip("torch")
    from pde_agent_verifier_solver.plugins.vp_1d1v import ArtemisPhysicalPackage, ArtemisRewardGuidance
    from pde_agent_verifier_solver.sdk import (
        TerminalCohortSpec, TerminalEvaluationSpec, TerminalGuidanceSource,
        TerminalMetricSpec,
    )
    from pde_agent_verifier_solver.sdk.macro_loop import SampleRequest

    checkout = Path(__file__).resolve().parents[2] / "artemis-oracle-opt2-20260907"
    if not checkout.is_dir():
        pytest.skip("pinned external Artemis source is not installed")
    package = ArtemisPhysicalPackage(checkout)
    capsule_type = package.module("artemis_compact_v5_production_runtime").ArtemisCompactV5GuidanceCoordinateCapsule

    class Runtime:
        conditions = ("a", "b")
        def guidance_coordinate_from_seed(self, seed):
            rng = random.Random(seed)
            return capsule_type.from_unit_uniforms(self.conditions[seed % 2], tuple(rng.random() for _ in range(3)))
        def replay_guidance_feedback_coordinate(self, manufactured):
            return manufactured

    class Source(_Source):
        def __init__(self):
            super().__init__()
            self.runtime, self._roots, self._lock = Runtime(), {}, RLock()
        def sample(self, request):
            records = super().sample(request)
            for index, record in enumerate(records):
                if record.case.parent_case_id is None:
                    self._roots[record.case.case_id] = SimpleNamespace(
                        manufactured=self.runtime.guidance_coordinate_from_seed(request.seed + index))
            return records
        def issue_guided_answer_first_records(self, capsules, **kwargs):
            cohort = CohortKind.DEVELOPMENT if kwargs["visibility"] is CaseVisibility.DEVELOPMENT else CohortKind.TERMINAL
            # The legacy physical-source request supports BASE/PERT only.
            # Construct canonical guided records from the fixture arrays;
            # the real adapter also issues guided answers through this method.
            raw = self.sample(SampleRequest(kwargs["round_index"], cohort,
                kwargs["controller_seed"], len(capsules)))
            records = []
            for record, capsule in zip(raw, capsules, strict=True):
                self.records.pop(record.case.case_id)
                self._roots.pop(record.case.case_id)
                identity = f"guided-{record.case.case_id}"
                public = {**record.case.public_data,
                    "equation": {**record.case.public_data["equation"], "case_id": identity},
                    "distribution_lineage": {"kind": "frozen_reward_guided",
                        "snapshot_sha256": kwargs["guide_snapshot_sha256"]}}
                guided = replace(record, case=replace(record.case,
                    case_id=identity, public_data=public))
                self.records[identity] = guided
                self._roots[identity] = SimpleNamespace(manufactured=capsule)
                records.append(guided)
            return tuple(records)
        def release_cases(self, ids):
            ids = tuple(ids)
            for identity in ids:
                self._roots.pop(identity, None)
            super().release_cases(ids)

    source = Source()
    guide = ArtemisRewardGuidance(package, source, settings={
        "condition_count": 2, "model_kind": "ou_hermite_ridge_v1",
        "hermite_degree": 1, "crossfit_fold_count": 2, "crossfit_repeat_count": 1,
        "guidance_strength": 2.0, "sampling_steps": 2,
    })
    backend = ArtemisBackend(source, physical_revision="synthetic-lifecycle-fixture", guidance_adapter=guide)
    plugin = build_artemis_plugin(backend)
    dataset = plugin.problem.prepare_dataset(DatasetInput("online", DatasetKind.NUMERICAL, settings={"online_only": True}))
    binding = BackendRunBinding(plugin, backend.checkpoint(), dataset)
    recipe = CampaignRecipe(
        name="Artemis nonzero-G configuration lifecycle fixture", round_count=2,
        agents=(AgentSpec("agent", "Macro Agent"),),
        default_round=RoundSettings(
            development_sampling=DevelopmentSamplingPolicy(
                max_cases_per_agent_per_round=3,
                allowed_sources=(SampleSource.BASE, SampleSource.GUIDED),
                starter=CaseCounts(base=1, guided=1, guided_perturbed=1)),
            official_evaluation=OfficialEvaluationSpec(round=HiddenSuiteSpec(base=24))),
        generator=GeneratorRecipe(guidance_enabled=True, perturbations_enabled=True),
        evaluator=EvaluatorPolicy(_query(), development_point_count=2),
        initial_candidate=_artifact(_MacroOnly(factor=.5, retain_session=False), 0),
        terminal_evaluation=TerminalEvaluationSpec(
            cohorts=(
                TerminalCohortSpec("base_model", HiddenSuiteSpec(base=2)),
                TerminalCohortSpec("final_guided", HiddenSuiteSpec(guided=2),
                    guidance_source=TerminalGuidanceSource.FINAL),
            ),
            ranking=TerminalMetricSpec("base_model", "base"),
        ),
    )
    result = binding.run(recipe, {"agent": _MacroAgent()}, local_development=True)
    assert len(result.rounds) == 2
    assert guide.config.guidance_strength == 2.0
    audits = guide.training_audits
    assert [row["source_round_index"] for row in audits] == [0, 1, 2]
    assert all(row["previous_model_reused"] is False for row in audits)
    assert all(row["initialization"] == "fresh_fit_from_incumbent_feedback" for row in audits)
    assert [row["target_solver_id"] for row in audits] == [
        "candidate-s0",  # Controller-owned frozen S0 identity, not the author ID.
        result.rounds[0].selected.artifact.artifact_id,
        result.rounds[1].selected.artifact.artifact_id,
    ]
    assert all(row["feedback_case_count"] == 24 for row in audits)
    assert all(row["condition_counts"] == [12, 12] for row in audits)
    assert len({row["controller_seed"] for row in audits}) == 3
    assert len({row["snapshot_sha256"] for row in audits}) == 3
    development = [row for row in guide.sampling_audits if row["purpose"] == "development"]
    assert [(row["round_index"], row["snapshot_sha256"]) for row in development] == [
        (1, audits[0]["snapshot_sha256"]), (2, audits[1]["snapshot_sha256"])]
    terminal = [row for row in guide.sampling_audits if row["purpose"] == "terminal"]
    assert terminal and all(row["snapshot_sha256"] == audits[2]["snapshot_sha256"] for row in terminal)
    assert all(row["wall_seconds"] >= 0 for row in audits + guide.sampling_audits)
    assert not guide._coordinates and not source._roots and not source.records
    assert source.sessions and all(session.closed for session in source.sessions)
    # Reports are JSON-friendly cost/provenance only, never feedback arrays.
    import json
    json.dumps({"training": audits, "sampling": guide.sampling_audits}, allow_nan=False)
