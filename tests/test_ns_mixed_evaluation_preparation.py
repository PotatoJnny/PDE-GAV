"""CPU preparation-to-replay qualification with tiny actual continuous fields."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.engine import (
    AgentSpec, CampaignCompiler, CampaignController, CampaignRecipe,
    DevelopmentSamplingPolicy, EvaluatorPolicy, GeneratorRecipe, RoundSettings,
)
from pde_agent_verifier_solver.engine.common_evaluation import content_sha256
from pde_agent_verifier_solver.engine.guidance_controller import (
    GuidanceProtocolConfig, GuidanceRewardArtifact,
)
from pde_agent_verifier_solver.plugins.ns.canonical_backend import CanonicalStructuredNSBackend
from pde_agent_verifier_solver.plugins.ns.case_bank import load_ns_continuous_case_bank
from pde_agent_verifier_solver.plugins.ns.continuous_backend import (
    LegacyStructuredNSSmoothPerturber, StructuredNSManufacturedCaseSource,
)
from pde_agent_verifier_solver.plugins.ns.generator import NSStructuredGenerator
from pde_agent_verifier_solver.plugins.ns.tasks import NSSolverTask
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, AnswerView, Artifact, CaseVisibility, EvaluationPhase, FrozenGuidanceSnapshot,
    GenerationPurpose, GeneratorCheckpoint, GuidanceCapability, GuidanceTrainingBudget,
    GuidedAnswerFirstSample, HiddenSuiteSpec, OfficialEvaluationSpec, PreparedDataset,
    QueryKind, SampleSource, TaskKind, TerminalCohortSpec, TerminalEvaluationSpec,
    TerminalGuidanceSource, TerminalMetricSpec,
)

ROOT = Path(__file__).resolve().parents[1]


def _preparation():
    spec = importlib.util.spec_from_file_location(
        "test_ns_mixed_preparation_script", ROOT / "scripts/prepare_ns_mixed_evaluation.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


@pytest.fixture
def cpu_field_factory(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(ROOT / "legacy/ns/src"))
    from agent_verifier.neural_generator.continuous_geometry import ContinuousGeometrySpec
    from agent_verifier.neural_generator.structured_ns_answer_first import (
        ContinuousStructuredNSMMS, StructuredNSCaseCoefficients, StructuredNSProjectionConfig,
    )
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    geometry = ContinuousGeometrySpec(domain_bounds=(0., 1., 0., 1.), obstacle_polygons=())
    def make(seed):
        random = torch.Generator().manual_seed(seed)
        return ContinuousStructuredNSMMS(
            StructuredNSCaseCoefficients(
                psi=torch.randn((3, 9), generator=random, dtype=torch.float64) * 1e-3,
                pressure=torch.randn((3, 9), generator=random, dtype=torch.float64) * 1e-3,
            ), config=StructuredNSProjectionConfig(spatial_mode_x=2, spatial_mode_y=2, time_order=2),
            geometry_spec=geometry, domain_bounds=geometry.domain_bounds,
            viscosity=.01, t0=0., t1=.12,
        )
    yield make
    torch.set_num_threads(previous)


class _Reference:
    def __init__(self):
        self.calls = []

    def prepare_solution(self, case, solve_budget=None):
        self.calls.append(case.case_id)
        return self

    def evaluate(self, query):
        return AnswerView(QueryKind.POINTS, {name: (0.,) * len(query.points) for name in query.fields},
                          points=query.points, shape=(len(query.points),))


class _Guidance:
    guidance_capability = GuidanceCapability.online("cpu-test-guidance")

    def __init__(self, source, field_factory, protocol):
        self.source, self.field_factory = source, field_factory
        self.config = SimpleNamespace(minimum_cases=2)
        self.guidance_config_sha256 = protocol.guidance_config_sha256
        self.official_sample_settings = protocol.official_sample_settings
        self.feedback, self.training_requests, self.guided_requests = [], [], []

    def seal_reward_feedback(self, feedback):
        self.feedback.append(feedback)
        return GuidanceRewardArtifact("memory://cpu-reference-feedback", _digest(len(self.feedback)))

    def train_guidance(self, request):
        self.training_requests.append(request)
        return FrozenGuidanceSnapshot(request, self.guidance_capability.backend_id,
            "memory://cpu-guidance", _digest(request.target_round_index),
            campaign_guidance_training_wall_seconds=0.)

    def open_guidance(self, snapshot):
        return _GuidedOracle(self, snapshot)


class _GuidedOracle:
    development_solution_methods = ("query_completed_velocity",)

    def __init__(self, backend, snapshot):
        self.backend, self.snapshot = backend, snapshot
        self.guidance_capability = backend.guidance_capability
        self.records = {}

    def generate_answer_first(self, request):
        self.backend.guided_requests.append(request)
        records = self.backend.source.issue_answer_first_records(
            tuple(self.backend.field_factory(request.controller_seed + i) for i in range(request.spec.sample_count)),
            round_index=self.snapshot.training_request.target_round_index,
            visibility=request.expected_visibility, case_id_prefix="cpu-guided",
            controller_seed=request.controller_seed,
            distribution_lineage={"kind": "frozen_reward_guided", "snapshot_sha256": self.snapshot.snapshot_sha256},
        )
        self.records.update((row.case.case_id, row) for row in records)
        return tuple(GuidedAnswerFirstSample(
            sample_id=row.case.case_id, snapshot_sha256=request.snapshot_sha256, case=row.case,
            reference_answer_sha256=_digest(row.case.case_id), public_case_sha256=content_sha256(row.case),
            answer_to_case_derivation_sha256=_digest("cpu-guided-derivation"),
        ) for row in records)

    def resolve_case_records(self, batch):
        return tuple(self.records[row.sample_id] for row in batch.samples)

    def close(self):
        self.backend.source.release_cases(tuple(self.records))


def _controller(field_factory):
    source = StructuredNSManufacturedCaseSource(
        generator_id="cpu-test-flow", artifact_sha256="a" * 64, clean_sampler=field_factory,
        perturbation_sampler=LegacyStructuredNSSmoothPerturber(amplitude=.01, spectral_max_mode=3),
    )
    protocol = GuidanceProtocolConfig(
        training_budget=GuidanceTrainingBudget(1., "cpu-test-only"), guidance_config_sha256="b" * 64,
        reward_feedback_case_count=2, official_round_case_count=2, official_terminal_case_count=2,
        inference_wall_budget_seconds=1., inference_compute_capacity="cpu-test-only",
        official_sample_settings={},
    )
    guidance = _Guidance(source, field_factory, protocol)
    backend = CanonicalStructuredNSBackend(SimpleNamespace(case_source=source),
        guidance_backend=guidance, guidance_protocol=protocol)
    generator = NSStructuredGenerator(backend)
    query = AnswerQuery.at_points(fields=("velocity_x", "velocity_y"),
                                  points=({"x": .2, "y": .3, "t": .01},))
    reference = Artifact("frozen-reference-s0", TaskKind.SOLVER, "Fixed test reference", _Reference(),
                         "CPU-only preparation reference", "prepare_solution", 0, "user")
    campaign = CampaignCompiler().compile(CampaignRecipe(
        name="CPU mixed preparation", round_count=2, master_seed=101,
        agents=(AgentSpec("never-run", "Never run"),),
        default_round=RoundSettings(
            agent_budget_seconds=1,
            development_sampling=DevelopmentSamplingPolicy(max_cases_per_agent_per_round=2,
                allowed_sources=(SampleSource.BASE,)),
            official_evaluation=OfficialEvaluationSpec(round=_preparation().equal_suite(2))),
        generator=GeneratorRecipe(guidance_enabled=True, perturbations_enabled=True,
            checkpoint=GeneratorCheckpoint(generator.info.program_id, backend.base_state)),
        evaluator=EvaluatorPolicy(query), initial_candidate=reference,
        terminal_evaluation=TerminalEvaluationSpec(
            (TerminalCohortSpec("base", HiddenSuiteSpec(base=2)),), TerminalMetricSpec("base", "base")),
        task_settings={"official_point_count": 4, "terminal_point_count": 4},
    ), generator.capabilities)
    return CampaignController(campaign, PreparedDataset(None, (), ()), None, NSSolverTask(), generator, {}), guidance


@pytest.fixture
def prepared_suite(tmp_path, cpu_field_factory):
    controller, guidance = _controller(cpu_field_factory)
    path = _preparation().prepare(controller, tmp_path / "banks", promotion_per_type=2,
                                  selection_per_type=2, terminal_per_type=2)
    return {"path": path, "suite": json.loads(path.read_text()), "controller": controller,
            "guidance": guidance, "field_factory": cpu_field_factory}


def test_prepare_fixed_reference_two_rounds_has_equal_disjoint_banks(prepared_suite):
    suite, root = prepared_suite["suite"], prepared_suite["path"].parent
    assert suite["status"] == "frozen" and suite["provenance"]["coding_agent_calls"] == 0
    assert [row["round_index"] for row in suite["official_rounds"]] == [1, 2]
    paths = [(suite["fixed_selection_manifest"], "promotion")]
    paths += [(row["manifest"], "promotion") for row in suite["official_rounds"]]
    paths += [(suite["terminal_manifest"], "terminal")]
    seen, identities = set(), set()
    for name, purpose in paths:
        bank = load_ns_continuous_case_bank(root / name, expected_purpose=purpose)
        assert bank.manifest["strata_counts"] == {"base": 2, "guided": 2, "base_perturbed": 1, "guided_perturbed": 1}
        ids = {row.case.case_id for row in bank.records}
        assert len(ids) == 6 and not seen.intersection(ids)
        seen.update(ids)
        for row in bank.records:
            field = row.truth._answer.raw_answer
            identity = content_sha256({"psi": field.coefficients.psi.tolist(), "pressure": field.coefficients.pressure.tolist()})
            assert identity not in identities
            identities.add(identity)
            assert field.coefficients.psi.device.type == "cpu"
            if row.case.parent_case_id:
                assert row.case.parent_case_id in ids
    guidance = prepared_suite["guidance"]
    assert [row.target_round_index for row in guidance.training_requests] == [1, 2, 3]
    assert [row.source_round_index for row in guidance.training_requests] == [0, 1, 2]
    from pde_agent_verifier_solver.plugins.ns.canonical_backend import _candidate_backend_digest
    reference = prepared_suite["controller"].campaign.initial_candidate
    expected_digest = _candidate_backend_digest(reference)
    assert {row.target_solver_sha256 for row in guidance.training_requests} == {expected_digest}
    assert all(len(item.rows) == 2 for item in guidance.feedback)
    assert {item.target_solver_sha256 for item in guidance.feedback} == {expected_digest}
    assert len(reference.program.calls) == 6


def test_prepared_sampling_schedule_and_query_plans_replay_through_controller(prepared_suite, tmp_path):
    from pde_agent_verifier_solver.plugins.ns.mixed_evaluation import SharedNSMixedEvaluationGenerator
    controller, _ = _controller(prepared_suite["field_factory"])
    script = _preparation()
    controller = script._probe(controller)
    counts = script.equal_suite(2)
    base = TerminalCohortSpec("common_base", HiddenSuiteSpec(base=2, base_perturbed=1))
    guided = TerminalCohortSpec("common_guided", HiddenSuiteSpec(guided=2, guided_perturbed=1), TerminalGuidanceSource.FINAL)
    controller.campaign = replace(controller.campaign, terminal_evaluation=TerminalEvaluationSpec(
        (base, guided), TerminalMetricSpec("overall", "overall")))
    settings = {**controller.campaign.task_settings, "default_query": controller.campaign.evaluator.default_query}
    wrapper = SharedNSMixedEvaluationGenerator(controller.generator, prepared_suite["path"],
        tmp_path / "replay.jsonl", controller.campaign.master_seed, settings)
    controller.generator = wrapper
    state = wrapper.load(controller.campaign.generator.checkpoint, {})
    # Replay supplies its own frozen G Cases. No new test Guidance training is needed.
    state = replace(state, guidance=object())
    controller._loaded_generator_state = state
    try:
        controller._hidden_suite(state, controller.campaign.rounds[0], base.suite,
            GenerationPurpose.TERMINAL, 0, {}, seed_namespace="terminal:common_base", generation_round_index=0)
        for plan in controller.campaign.rounds:
            hidden, _, _ = controller._hidden_suite(state, plan, counts, GenerationPurpose.PROMOTION, 0, {})
            reports, _ = controller._official_reports((controller.campaign.initial_candidate,), hidden,
                                                       seed_namespace=f"official_round:{plan.round_index}")
            assert reports[0].technically_eligible
        controller._hidden_suite(state, controller.campaign.rounds[-1], guided.suite,
            GenerationPurpose.TERMINAL, 0, {}, seed_namespace="terminal:common_guided", generation_round_index=3)
        expected = {"base": 2, "guided": 2, "base_perturbed": 1, "guided_perturbed": 1}
        assert wrapper.assert_complete() == {"official:1": expected, "official:2": expected, "terminal": expected}
    finally:
        controller._close_case_samplers(controller._case_samplers)


def _uneven_controller(field_factory):
    controller, guidance = _controller(field_factory)
    official = HiddenSuiteSpec(base=4, guided=3, base_perturbed=3)
    terminal = TerminalEvaluationSpec((
        TerminalCohortSpec('common_base', HiddenSuiteSpec(base=2, base_perturbed=1)),
        TerminalCohortSpec('common_guided', HiddenSuiteSpec(guided=2, guided_perturbed=1), TerminalGuidanceSource.FINAL),
    ), TerminalMetricSpec('overall', 'overall'))
    controller.campaign = replace(controller.campaign,
        rounds=tuple(replace(plan, round_hidden=official) for plan in controller.campaign.rounds),
        terminal_evaluation=terminal)
    return controller, guidance


@pytest.fixture
def uneven_prepared_suite(tmp_path, cpu_field_factory):
    controller, guidance = _uneven_controller(cpu_field_factory)
    path = _preparation().prepare(controller, tmp_path/'uneven_banks', promotion_from_config=True,
        selection_suite=HiddenSuiteSpec(base=4, guided=3, base_perturbed=3), terminal_from_config=True)
    return {'path': path, 'controller': controller, 'guidance': guidance, 'field_factory': cpu_field_factory}


def test_prepare_actual_uneven_counts_terminal_only_and_shared_assets(uneven_prepared_suite):
    root = uneven_prepared_suite['path'].parent
    suite = json.loads((root/'suite.json').read_text())
    terminal_suite = json.loads((root/'terminal_suite.json').read_text())
    comparison = json.loads((root/'comparison_suite.json').read_text())
    assert 'official_rounds' not in terminal_suite
    for key in ('master_seed', 'task_settings_sha256', 'terminal_sampling_requests', 'terminal_evaluation_context'):
        assert terminal_suite[key] == suite[key] == comparison[key]
    selection = load_ns_continuous_case_bank(root/'fixed_selection.json', expected_purpose='promotion')
    assert selection.manifest['strata_counts'] == {'base': 4, 'guided': 3, 'base_perturbed': 3}
    selection_ids = {row.case.case_id for row in selection.records}
    for full, common in zip(suite['official_rounds'], comparison['official_rounds']):
        original = load_ns_continuous_case_bank(root/full['manifest'], expected_purpose='promotion')
        subset = load_ns_continuous_case_bank(root/common['manifest'], expected_purpose='promotion')
        assert len(original.records) == 10 and len(subset.records) == 7
        assert original.manifest['strata_counts'] == {'base': 4, 'guided': 3, 'base_perturbed': 3}
        assert subset.manifest['strata_counts'] == {'base': 4, 'base_perturbed': 3}
        assert common['delegate_kinds'] == ['guided']
        assert all(row['kind'] != 'guided' for row in common['sampling_requests'])
        assert not selection_ids.intersection(row.case.case_id for row in original.records)
        original_rows = {row['case']['case_id']: row for row in original.manifest['cases']}
        assert all(row == original_rows[row['case']['case_id']] for row in subset.manifest['cases'])
        assert not (root/common['manifest']).with_suffix('.cases').exists()
    terminal = load_ns_continuous_case_bank(root/'terminal.json', expected_purpose='terminal')
    assert terminal.manifest['strata_counts'] == {'base': 2, 'guided': 2, 'base_perturbed': 1, 'guided_perturbed': 1}
    assert terminal.provenance['guidance_source_round'] == 2
    assert terminal.provenance['guidance_target_round'] == 3
    assert terminal.provenance['terminal_guidance_reference_artifact_id'] == 'frozen-reference-s0'
    assert {row['round_index'] for row in terminal_suite['terminal_sampling_requests']} == {0, 3}
    terminal_ids = {record.case.case_id for record in terminal.records}
    assert all(row.case.case_id not in terminal_ids for feedback in uneven_prepared_suite['guidance'].feedback for row in feedback.rows)


def test_comparison_controller_reuses_base_parents_and_delegates_actual_guided(uneven_prepared_suite, tmp_path):
    from pde_agent_verifier_solver.plugins.ns.mixed_evaluation import SharedNSMixedEvaluationGenerator
    script = _preparation()
    path = uneven_prepared_suite['path'].parent/'comparison_suite.json'
    runs = []
    for run_index in range(2):
        controller, guidance = _uneven_controller(uneven_prepared_suite['field_factory'])
        controller = script._probe(controller)
        settings = {**controller.campaign.task_settings, 'default_query': controller.campaign.evaluator.default_query}
        wrapper = SharedNSMixedEvaluationGenerator(controller.generator, path,
            tmp_path/f'comparison_run_{run_index}.jsonl', controller.campaign.master_seed, settings)
        controller.generator = wrapper
        state = wrapper.load(controller.campaign.generator.checkpoint, {})
        controller._loaded_generator_state = state
        captured = []
        try:
            base, guided = controller.campaign.terminal_evaluation.cohorts
            terminal_base, _, _ = controller._hidden_suite(state, controller.campaign.rounds[0], base.suite,
                GenerationPurpose.TERMINAL, 0, {}, seed_namespace='terminal:common_base', generation_round_index=0)
            reference = replace(controller.campaign.initial_candidate, artifact_id=f'run-{run_index}-incumbent')
            feedback = controller._initial_guidance_feedback(reference, state, controller.campaign.rounds[0], {})
            for plan in controller.campaign.rounds:
                state = wrapper.update_guidance(state, feedback,
                    {'controller_seed': controller._seeds.derive('generator', 'guidance', plan.round_index)})
                controller._loaded_generator_state = state
                controller._close_case_samplers(controller._pending_feedback_samplers)
                controller._pending_feedback_samplers.clear()
                hidden, _, _ = controller._hidden_suite(state, plan, plan.round_hidden, GenerationPurpose.PROMOTION, 0, {})
                assert set(hidden) == {'base', 'guided', 'base_perturbed'}
                assert [len(hidden[name]) for name in ('base', 'guided', 'base_perturbed')] == [4, 3, 3]
                reports, _ = controller._official_reports((reference,), hidden, seed_namespace=f'official_round:{plan.round_index}')
                assert reports[0].technically_eligible
                assert len(guidance.guided_requests) == plan.round_index
                feedback = script._reference_feedback(controller, reference, hidden, plan.round_index)
                captured.append({name: tuple(record.case for record in records) for name, records in hidden.items()})
            # Frozen final Guided needs no new training against this run's solver.
            terminal_guided, _, _ = controller._hidden_suite(state, controller.campaign.rounds[-1], guided.suite,
                GenerationPurpose.TERMINAL, 0, {}, seed_namespace='terminal:common_guided', generation_round_index=3)
            terminal = {controller._terminal_distribution_key(cohort.cohort_id, name): rows
                        for cohort, hidden in ((base, terminal_base), (guided, terminal_guided))
                        for name, rows in hidden.items()}
            reports, _ = controller._official_reports((reference,), terminal, phase=EvaluationPhase.TERMINAL,
                                                      seed_namespace='terminal_evaluation')
            assert reports[0].technically_eligible
            assert len(guidance.training_requests) == 2
            assert wrapper.assert_complete() == {
                'official:1': {'base': 4, 'base_perturbed': 3},
                'official:2': {'base': 4, 'base_perturbed': 3},
                'terminal': {'base': 2, 'guided': 2, 'base_perturbed': 1, 'guided_perturbed': 1},
            }
            runs.append(captured)
        finally:
            controller._close_case_samplers(controller._case_samplers)
    for left, right in zip(*runs):
        assert left['base'] == right['base']
        assert left['base_perturbed'] == right['base_perturbed']
        assert {case.parent_case_id for case in left['base_perturbed']} <= {case.case_id for case in left['base']}
        assert {case.case_id for case in left['guided']}.isdisjoint(case.case_id for case in right['guided'])


def test_campaign_counts_keep_formal_192_and_terminal_768(cpu_field_factory):
    controller, _ = _uneven_controller(cpu_field_factory)
    official = HiddenSuiteSpec(base=64, guided=64, base_perturbed=64)
    terminal = TerminalEvaluationSpec((
        TerminalCohortSpec('common_base', HiddenSuiteSpec(base=256, base_perturbed=256)),
        TerminalCohortSpec('common_guided', HiddenSuiteSpec(guided=256), TerminalGuidanceSource.FINAL),
    ), TerminalMetricSpec('overall', 'overall'))
    controller.campaign = replace(controller.campaign,
        rounds=tuple(replace(plan, round_hidden=official) for plan in controller.campaign.rounds), terminal_evaluation=terminal)
    rounds, selection, terminal_result, _, _ = _preparation()._preparation_specs(controller,
        promotion_from_config=True, selection_suite=official, terminal_from_config=True)
    assert all(plan.round_hidden.case_count == 192 and plan.round_hidden.guided_perturbed == 0 for plan in rounds)
    assert selection == official and terminal_result is terminal
    assert sum(cohort.suite.case_count for cohort in terminal_result.cohorts) == 768
    assert all(cohort.suite.guided_perturbed == 0 for cohort in terminal_result.cohorts)
