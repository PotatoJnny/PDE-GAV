"""Native Campaign/Panel contract tests; physical runtime here is a test double."""

from dataclasses import replace
import math

import pytest

from pde_agent_verifier_solver.app.service import ProductApplication
from pde_agent_verifier_solver.engine.campaign import (
    AgentSpec, CampaignRecipe, CaseCounts, DevelopmentSamplingPolicy,
    EvaluatorPolicy, GeneratorRecipe, RoundSettings,
)
from pde_agent_verifier_solver.engine.runtime import AgentOutcome
from pde_agent_verifier_solver.engine.evaluator import EvaluationRequest
from pde_agent_verifier_solver.plugins.gkw.canonical_binding import bind_gkw_runtime, gkw_guidance_features
from pde_agent_verifier_solver.plugins.gkw.contract import GKW_MACRO_FIELDS
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, AnswerView, Artifact, CaseVisibility, GenerationKind,
    GenerationPurpose, GenerationRequest, GeneratorMode, HiddenSuiteSpec,
    OfficialEvaluationSpec, SampleSource, TaskKind,
)
from tests.test_gkw_manufactured_backend import _PhysicalRuntime


QUERY = AnswerQuery.at_points(fields=GKW_MACRO_FIELDS, points=(
    {"t": .75, "s_index": 0, "kx_index": 0, "ky_index": 0},
))


class MacroOnlyCandidate:
    """Deliberately has no solve(), distribution(), h(), or full-state method."""

    def predict_observables(self, case, query, budget=None):
        assert tuple(query.fields) == GKW_MACRO_FIELDS
        return AnswerView(query.kind,
            fields={name: (1., .1) * len(query.points) for name in GKW_MACRO_FIELDS},
            points=query.points, shape=(len(query.points), 2))


class MacroOnlyAgent:
    def run(self, context):
        round_index = context.round_plan.round_index
        artifact = Artifact(f"macro-{round_index}", TaskKind.OBSERVABLE_MODEL,
            "Macro-only candidate", MacroOnlyCandidate(), "Four macro fields only",
            "predict_observables(case, query)", round_index, context.agent.agent_id)
        context.evaluator.evaluate(artifact, EvaluationRequest(tuple(case.case_id for case in context.cases)))
        return AgentOutcome(artifact, ("No kinetic-state prediction is requested.",))


def recipe():
    suite = HiddenSuiteSpec(base=2, base_perturbed=2)
    return CampaignRecipe(
        "GKW native macro-only smoke", 2, (AgentSpec("macro", "Macro agent"),),
        RoundSettings(agent_budget_seconds=10,
            development_sampling=DevelopmentSamplingPolicy(
                max_cases_per_agent_per_round=2,
                allowed_sources=(SampleSource.BASE,), starter=CaseCounts(base=1)),
            official_evaluation=OfficialEvaluationSpec(round=suite, terminal=suite)),
        GeneratorRecipe(guidance_enabled=False, perturbations_enabled=True),
        EvaluatorPolicy(QUERY, development_point_count=2),
    )


def test_macro_only_candidate_completes_two_round_canonical_campaign():
    runtime = _PhysicalRuntime()
    binding = bind_gkw_runtime(runtime)
    result = binding.run(recipe(), {"macro": MacroOnlyAgent()}, local_development=True)
    assert len(result.rounds) == 2
    assert math.isfinite(result.selected.primary_error)
    assert result.selected.artifact.kind is TaskKind.OBSERVABLE_MODEL
    assert set(result.terminal_distribution_metrics) >= {"base", "base_perturbed"}
    assert runtime.clean_seeds and runtime.perturbed_seeds
    assert not runtime.committed_case_ids
    assert runtime.release_batches


@pytest.mark.parametrize("guidance_available", [False, True])
def test_loaded_backend_runs_through_the_same_product_panel(guidance_available):
    runtime = _PhysicalRuntime()
    binding = bind_gkw_runtime(runtime, guidance=guidance_available)
    app = ProductApplication.local_development()
    panel = binding.panel(app, {"macro": MacroOnlyAgent()}, default_query=QUERY)
    setup = panel.setup_page()
    assert setup["backend"]["checkpoint_loaded"]
    assert setup["backend"]["query_fields"] == list(GKW_MACRO_FIELDS)
    assert setup["defaults"]["allowed_sources"] == (["base", "guided"] if guidance_available else ["base"])
    launch = panel.create_and_launch({
        "project_name": "GKW loaded model", "round_count": 2,
        "guidance_enabled": False,
        "agent_budget_seconds": 10, "development_point_count": 2,
        "round_hidden": {"base": 2, "base_perturbed": 2},
        "terminal_evaluation": {"cohorts": [{"id": "base_model",
            "guidance_source": "base", "counts": {"base": 2, "base_perturbed": 2}}],
            "ranking": {"cohort_id": "base_model", "distribution": "base", "statistic": "mean"}},
    })
    app.wait(launch["run_id"], timeout=20)
    page = panel.run_page(launch["run_id"])
    assert page["status"] == "completed", page.get("error")
    assert len(page["rounds"]) == 2
    assert "predict_observables" in page["result"]["usage"]
    assert not runtime.committed_case_ids


def test_bound_generator_fills_identity_and_refuses_conflicting_override():
    runtime = _PhysicalRuntime()
    binding = bind_gkw_runtime(runtime)
    generator = binding.plugin.generator
    state = generator.load(binding.checkpoint, {})
    request = GenerationRequest(1, 1, GeneratorMode.ONLINE, GenerationKind.BASE,
        CaseVisibility.HIDDEN, 191, GenerationPurpose.SELECTION)
    records = generator.sample(state, request)
    assert len(gkw_guidance_features(records[0].case)) == 4
    generator.release(state, tuple(row.case.case_id for row in records))
    with pytest.raises(ValueError, match="conflicts"):
        generator.sample(state, replace(request, settings={"fixed_generator_sha256": "f" * 64}))


def test_binding_requires_explicit_local_agent_execution():
    binding = bind_gkw_runtime(_PhysicalRuntime())
    with pytest.raises(ValueError, match="checkpoint does not match"):
        binding.plugin.generator.load(replace(binding.checkpoint, payload="c" * 64), {})
    with pytest.raises(ValueError, match="execution service"):
        binding.controller(recipe(), {"macro": MacroOnlyAgent()})
    with pytest.raises(ValueError, match="checkpoint differs"):
        binding.controller(replace(recipe(), generator=replace(recipe().generator,
            checkpoint=replace(binding.checkpoint))), {"macro": MacroOnlyAgent()}, local_development=True)
