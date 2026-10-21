from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from pde_agent_verifier_solver.engine.campaign import (
    AgentSpec, CampaignCompiler, CampaignRecipe, DevelopmentSamplingPolicy,
    EvaluatorPolicy, GeneratorRecipe, RoundSettings,
)
from pde_agent_verifier_solver.engine.seeds import SeedLedger
from pde_agent_verifier_solver.engine.common_evaluation import content_sha256
from pde_agent_verifier_solver.engine.unified_runtime import CampaignController
from pde_agent_verifier_solver.plugins.ns.canonical_backend import CanonicalStructuredNSBackend
from pde_agent_verifier_solver.plugins.ns.continuous_backend import StructuredNSManufacturedCaseSource
from pde_agent_verifier_solver.plugins.ns.generator import NSStructuredGenerator
from pde_agent_verifier_solver.plugins.ns.official_audit import (
    OfficialAuditGenerator, fingerprint_ns_official_batch, freeze_ns_official_bank,
)
from pde_agent_verifier_solver.plugins.ns.tasks.solver import NSSolverTask
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, CaseVisibility, EvaluationPhase, GenerationKind,
    GenerationPurpose, GenerationRequest, GeneratorCheckpoint, GeneratorMode,
    HiddenSuiteSpec, OfficialEvaluationSpec, PreparedDataset, SampleSource, SamplingStrategy,
)


class _Config:
    def to_dict(self):
        return {"time_modes": 2, "spatial_modes": 1}


class _Geometry:
    def to_dict(self):
        return {"obstacle_polygons": [[[0.1, 0.1], [0.2, 0.1], [0.1, 0.2]]]}


class _Field:
    config = _Config()
    geometry_spec = _Geometry()
    continuous_mms_abi = "test-continuous-ns-v1"
    domain_bounds = (0.0, 1.0, 0.0, 1.0)
    viscosity = 0.01
    t0 = 0.0
    t1 = 0.12

    def __init__(self, seed, shift=0, nonfinite=False):
        value = (seed % 1000) / 1000 + shift
        self.coefficients = SimpleNamespace(psi=np.array([[value], [1.0]]),
                                           pressure=np.array([[0.0], [2.0]]))
        self.nonfinite = nonfinite

    def velocity_pressure(self, points):
        value = self.coefficients.psi[0, 0]
        x, y, t = np.asarray(points).T
        u = value + x + t
        if self.nonfinite:
            u[:] = np.nan
        return u, value + y - t, x * 0

    def quantities(self, points):
        x, y, _t = np.asarray(points).T
        return {"forcing_u": 1.0 + x, "forcing_v": 2.0 + y}


def _controller(*, shift=0, nonfinite=False, seed=101, point_count=4, rounds=2):
    source = StructuredNSManufacturedCaseSource(
        generator_id="test-base", artifact_sha256="a" * 64, perturbation_sampler=None,
        clean_sampler=lambda value: _Field(value, shift=shift, nonfinite=nonfinite),
    )
    backend = CanonicalStructuredNSBackend(SimpleNamespace(case_source=source))
    generator = NSStructuredGenerator(backend)
    query = AnswerQuery.at_points(fields=("velocity_x", "velocity_y"),
                                  points=({"x": 0.5, "y": 0.5, "t": 0.01},))
    recipe = CampaignRecipe(
        name="NS Official audit", round_count=rounds, master_seed=seed,
        agents=(AgentSpec("never-run", "Never run"),),
        default_round=RoundSettings(
            agent_budget_seconds=1,
            development_sampling=DevelopmentSamplingPolicy(
                strategy=SamplingStrategy.AGENT_ADAPTIVE,
                max_cases_per_agent_per_round=2, allowed_sources=(SampleSource.BASE,)),
            official_evaluation=OfficialEvaluationSpec(
                round=HiddenSuiteSpec(base=2), terminal=HiddenSuiteSpec(base=2))),
        generator=GeneratorRecipe(guidance_enabled=False, perturbations_enabled=False,
                                  checkpoint=GeneratorCheckpoint(generator.info.program_id,
                                                                 backend.base_state)),
        evaluator=EvaluatorPolicy(query), task_settings={"official_point_count": point_count},
    )
    campaign = CampaignCompiler().compile(recipe, generator.capabilities)
    controller = CampaignController(campaign, PreparedDataset(None, (), ()),
                                    None, NSSolverTask(), generator, {})
    return controller


def _settings(controller):
    return {**controller.campaign.task_settings,
            "default_query": controller.campaign.evaluator.default_query}


def _request(controller, round_index, *, purpose=GenerationPurpose.PROMOTION):
    return GenerationRequest(
        round_index, 2, GeneratorMode.ONLINE, GenerationKind.BASE, CaseVisibility.HIDDEN,
        seed=SeedLedger(controller.campaign.master_seed).derive(
            "hidden_sampling", "promotion", round_index, round_index), purpose=purpose)


def _wrapper(controller, bank_path, audit_path):
    return OfficialAuditGenerator(controller.generator, bank_path, audit_path,
                                  controller.campaign.master_seed, _settings(controller))


def test_bank_uses_controller_recipe_and_accepts_different_opaque_ids(tmp_path):
    preparer = _controller()
    bank = freeze_ns_official_bank(preparer, tmp_path / "bank.json")
    assert bank["status"] == "frozen" and len(bank["rounds"]) == 2
    assert not preparer.generator.backend.case_source._issued
    assert not hasattr(preparer, "_seeds")
    actual = _controller()
    wrapper = _wrapper(actual, tmp_path / "bank.json", tmp_path / "audit.jsonl")
    state = actual.generator.load(actual.campaign.generator.checkpoint, {})
    actual._seeds = SeedLedger(actual.campaign.master_seed)
    for round_index in (1, 2):
        request = _request(actual, round_index)
        records = wrapper.sample(state, request)
        frozen = bank["rounds"][round_index - 1]
        assert records[0].case.case_id != frozen["cases"][0]["case_id"]
        seed_provider = actual._hidden_plan_seed_provider(
            f"official_round:{round_index}", EvaluationPhase.OFFICIAL, {"base": records})
        for i, record in enumerate(records):
            runtime_seed = seed_provider(i, record.case.case_id)
            runtime_plan = actual.evaluation.make_plan(
                record.case, EvaluationPhase.OFFICIAL, _settings(actual), runtime_seed)
            assert runtime_seed == frozen["cases"][i]["plan_seed"]
            assert len(runtime_plan.query.points) == 4
            assert content_sha256(runtime_plan) == frozen["cases"][i]["plan_sha256"]
        wrapper.release(state, tuple(r.case.case_id for r in records))
    wrapper.assert_complete()
    events = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["bank_bound", "verified", "verified"]


def test_actual_changed_fields_block_before_scoring_and_release_records(tmp_path):
    freeze_ns_official_bank(_controller(), tmp_path / "bank.json")
    altered = _controller(shift=0.1)
    wrapper = _wrapper(altered, tmp_path / "bank.json", tmp_path / "audit.jsonl")
    state = altered.generator.load(altered.campaign.generator.checkpoint, {})
    with pytest.raises(ValueError, match="physical fields"):
        wrapper.sample(state, _request(altered, 1))
    assert not altered.generator.backend.case_source._issued
    assert json.loads((tmp_path / "audit.jsonl").read_text().splitlines()[-1])["event"] == "verification_failed"
    with pytest.raises(ValueError, match="missing completed rounds"):
        wrapper.assert_complete()


@pytest.mark.parametrize("alteration", [{"seed": 102}, {"point_count": 5}])
def test_bank_rejects_changed_seed_or_query_settings(tmp_path, alteration):
    freeze_ns_official_bank(_controller(), tmp_path / "bank.json")
    with pytest.raises(ValueError, match="signature differs"):
        _wrapper(_controller(**alteration), tmp_path / "bank.json", tmp_path / "audit.jsonl")
    assert not (tmp_path / "audit.jsonl").exists()


def test_nonfinite_truth_cannot_be_frozen(tmp_path):
    controller = _controller(nonfinite=True)
    with pytest.raises(ValueError, match="finite"):
        freeze_ns_official_bank(controller, tmp_path / "bank.json")
    assert not (tmp_path / "bank.json").exists()
    assert not controller.generator.backend.case_source._issued


def test_development_forwarded_and_duplicate_official_rejected(tmp_path):
    controller = _controller()
    freeze_ns_official_bank(controller, tmp_path / "bank.json")
    wrapper = _wrapper(controller, tmp_path / "bank.json", tmp_path / "audit.jsonl")
    state = controller.generator.load(controller.campaign.generator.checkpoint, {})
    request = replace(_request(controller, 1), purpose=GenerationPurpose.DEVELOPMENT,
                      visibility=CaseVisibility.DEVELOPMENT, seed=999)
    development = wrapper.sample(state, request)
    assert all(r.case.visibility is CaseVisibility.DEVELOPMENT for r in development)
    wrapper.release(state, tuple(r.case.case_id for r in development))
    official = wrapper.sample(state, _request(controller, 1))
    with pytest.raises(ValueError, match="already sampled"):
        wrapper.sample(state, _request(controller, 1))
    wrapper.release(state, tuple(r.case.case_id for r in official))
    assert not controller.generator.backend.case_source._issued


def test_pressure_coefficients_are_verified_even_when_velocity_reference_matches(tmp_path):
    controller = _controller()
    state = controller.generator.load(controller.campaign.generator.checkpoint, {})
    request = _request(controller, 1)
    records = controller.generator.sample(state, request)
    original = fingerprint_ns_official_batch(records, request, master_seed=101,
                                              task_settings=_settings(controller))
    records[0].truth._answer.raw_answer.coefficients.pressure[0, 0] += 1
    altered = fingerprint_ns_official_batch(records, request, master_seed=101,
                                             task_settings=_settings(controller))
    assert original["cases"][0]["reference_sha256"] == altered["cases"][0]["reference_sha256"]
    assert original["cases"][0]["physics_sha256"] != altered["cases"][0]["physics_sha256"]
    controller.generator.release(state, tuple(r.case.case_id for r in records))
