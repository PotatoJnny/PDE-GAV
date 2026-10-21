"""Explicit pooled Promotion/Terminal scores preserve actual per-Case evidence."""
from dataclasses import replace
from math import ceil
from statistics import fmean

import pytest

from pde_agent_verifier_solver.engine.promotion import build_promotion_program
from pde_agent_verifier_solver.engine.unified_runtime import CampaignController
from pde_agent_verifier_solver.panel.controller import build_toy_panel
from pde_agent_verifier_solver.plugins.toy import ToyPoissonGenerator, ToySineSolver
from pde_agent_verifier_solver.sdk import (
    Artifact, ArtifactEvaluation, CaseEvaluation, HiddenSuiteSpec,
    PromotionCandidateReport, PromotionContext, TaskKind,
    TerminalCohortSpec, TerminalEvaluationSpec, TerminalMetricSpec,
)


def _artifact(name="candidate"):
    return Artifact(
        artifact_id=name, kind=TaskKind.SOLVER, name=name,
        program=ToySineSolver(1), summary="test", usage="test",
        round_index=0, agent_id="initial", metadata={"max_resolved_mode": 1},
    )


def _evaluations(artifact, errors):
    return {
        name: ArtifactEvaluation(
            artifact, fmean(values), {}, tuple(
                CaseEvaluation(f"{name}-{index}", value, {})
                for index, value in enumerate(values)
            ),
        )
        for name, values in errors.items()
    }


def test_overall_reweights_unequal_bp_gp_counts_and_sorts_global_tail():
    # B:G:P = 5:5:(2+3), while four equally weighted source means differ.
    errors = {"base": [0., 0., 0., 10., 10.], "guided": [1.] * 5,
              "base_perturbed": [1.] * 2, "guided_perturbed": [1.] * 3}
    evaluations = _evaluations(_artifact("low-mean"), errors)
    pooled = CampaignController._overall_evaluation(evaluations)
    summary = CampaignController._metric_summary(pooled)
    assert summary.case_count == 15
    assert summary.mean == 2.0
    assert summary.cvar == 7.0  # largest three are 10, 10, 1
    assert summary.mean != fmean(row.primary_error for row in evaluations.values())
    assert summary.cvar != fmean(CampaignController._metric_summary(row).cvar for row in evaluations.values())
    assert set(evaluations) == set(errors)  # no synthetic Cases in the source map

    steady = CampaignController._overall_evaluation(_evaluations(
        _artifact("low-tail"), {key: [2.5] * len(values) for key, values in errors.items()},
    ))
    reports = tuple(PromotionCandidateReport(
        row.artifact.artifact_id, row.artifact.artifact_id, True,
        {"overall": CampaignController._metric_summary(row)},
    ) for row in (pooled, steady))
    for statistic, winner in (("mean", "low-mean"), ("cvar", "low-tail")):
        promotion = build_promotion_program({
            "preset": "minimize_metric",
            "settings": {"primary": {"distribution": "overall", "statistic": statistic}},
        })
        assert promotion.select(None, reports, PromotionContext(1)).selected_digest == winner


@pytest.mark.parametrize("statistic", ["mean", "cvar"])
def test_complete_campaign_overall_terminal_and_feedback_each_case_once(statistic):
    class RecordingGenerator(ToyPoissonGenerator):
        def __init__(self):
            super().__init__()
            self.feedback = []

        def update_guidance(self, state, feedback, settings, progress=None):
            self.feedback.append(feedback)
            return super().update_guidance(state, feedback, settings, progress)

    panel = build_toy_panel()
    generator = RecordingGenerator()
    panel.programs = replace(panel.programs, generator=generator)
    launched = panel.create_and_launch({
        "project_name": "Explicit overall evaluation",
        "round_count": 2,
        "agents": ["careful-agent", "explorer-agent"],
        "initial_candidate": _artifact("s0"),
        "allowed_sources": ["base", "guided"],
        "starter_cases": {"base": 1, "guided": 1},
        "max_sampled_cases_per_agent_per_round": 4,
        "generator_settings": {"guidance_enabled": True, "perturbations_enabled": True},
        "round_hidden": {"base": 5, "guided": 5, "base_perturbed": 2, "guided_perturbed": 3},
        "promotion": {"preset": "minimize_metric", "settings": {
            "primary": {"distribution": "overall", "statistic": statistic},
        }},
        "terminal_evaluation": {
            "cohorts": [
                {"id": "base-bank", "guidance_source": "base", "counts": {"base": 5, "base_perturbed": 2}},
                {"id": "guided-bank", "guidance_source": "final_guided", "counts": {"guided": 5, "guided_perturbed": 3}},
            ],
            "ranking": {"cohort_id": "overall", "distribution": "overall", "statistic": statistic},
        },
    })
    panel.app.wait(launched["run_id"], timeout=10)
    page = panel.run_page(launched["run_id"])
    assert page["status"] == "completed", page.get("error")
    result = panel.app.get_run(launched["run_id"]).result
    assert len(result.selected.cases) == 15
    assert result.selected.artifact.artifact_id == result.rounds[-1].selected.artifact.artifact_id
    expected_terminal_keys = {
        "base-bank/base", "base-bank/base_perturbed",
        "guided-bank/guided", "guided-bank/guided_perturbed", "overall",
    }
    assert set(result.terminal_distribution_metrics) == expected_terminal_keys
    values = sorted(case.primary_error for case in result.selected.cases)
    terminal = result.terminal_distribution_metrics["overall"]
    assert terminal["mean"] == pytest.approx(fmean(values))
    assert terminal["cvar"] == pytest.approx(fmean(values[-ceil(.2 * len(values)):]))
    assert result.best_terminal_round == min(
        (getattr(report.distributions["overall"], statistic), row.round_index)
        for row, report in zip(result.rounds, result.terminal_round_reports, strict=True)
        if report.technically_eligible
    )[1]
    for row in result.rounds:
        assert len(row.selected.cases) == 15
        assert len({case.case_id for case in row.selected.cases}) == 15
        for report in row.promotion_reports:
            if report.technically_eligible:
                assert set(report.distributions) == {"base", "guided", "base_perturbed", "guided_perturbed", "overall"}
                assert report.distributions["overall"].case_count == 15
                assert row.hidden_distribution_metrics[report.candidate_digest]["overall"]["case_count"] == 15
    # Initial S0 feedback uses five Base Cases. The next-round and final
    # Guidance feedback each contain exactly the fifteen actual Official Cases.
    assert [len(feedback.cases) for feedback in generator.feedback] == [5, 15, 15]
    for feedback in generator.feedback:
        ids = [row.case.case_id for row in feedback.cases]
        assert len(ids) == len(set(ids))
    assert page["result"]["terminal_score_distribution"] == "overall"
    assert page["result"]["terminal_distributions"]["overall"]["case_count"] == 15


def test_overall_requires_unique_real_cases_and_one_candidate():
    row = _evaluations(_artifact(), {"base": [1.]})["base"]
    with pytest.raises(ValueError, match="distinct"):
        CampaignController._overall_evaluation({"base": row, "guided": row})
    with pytest.raises(ValueError, match="real source"):
        CampaignController._overall_evaluation({"overall": row})
    other = _evaluations(_artifact("other"), {"guided": [2.]})["guided"]
    with pytest.raises(ValueError, match="different candidates"):
        CampaignController._overall_evaluation({"base": row, "guided": other})


def test_terminal_overall_is_explicit_and_does_not_shadow_real_cohorts():
    with pytest.raises(ValueError, match="reserved"):
        TerminalCohortSpec("overall", HiddenSuiteSpec(base=1))
    for cohort, distribution in (("overall", "base"), ("bank", "overall")):
        with pytest.raises(ValueError, match="both cohort_id and distribution"):
            TerminalMetricSpec(cohort, distribution)
    cohort = TerminalCohortSpec("bank", HiddenSuiteSpec(base=1))
    specification = TerminalEvaluationSpec((cohort,), TerminalMetricSpec("overall", "overall", "cvar"))
    assert specification.cohorts == (cohort,)
    with pytest.raises(ValueError, match="unknown cohort"):
        TerminalEvaluationSpec((cohort,), TerminalMetricSpec("unknown", "base"))


@pytest.mark.parametrize("phase_name", ["official", "terminal"])
def test_failed_source_never_produces_partial_overall_score(monkeypatch, phase_name):
    from threading import Lock
    from types import SimpleNamespace
    import pde_agent_verifier_solver.engine.unified_runtime as runtime
    from pde_agent_verifier_solver.engine.seeds import SeedLedger
    from pde_agent_verifier_solver.sdk import EvaluationPhase

    artifact = _artifact()
    complete = _evaluations(artifact, {"base": [1.], "guided": [2.]})
    hidden = {name: (SimpleNamespace(case=SimpleNamespace(case_id=name)),) for name in complete}

    class Verifier:
        def __init__(self, *args, **kwargs):
            pass

        def evaluate(self, candidate, records):
            source = records[0].case.case_id
            if source == "guided":
                raise FloatingPointError("nonfinite candidate prediction")
            return complete[source]

    controller = object.__new__(CampaignController)
    controller.campaign = SimpleNamespace(
        promotion={"preset": "minimize_metric", "settings": {
            "primary": {"distribution": "overall", "statistic": "cvar"},
        }},
        terminal_evaluation=SimpleNamespace(ranking=TerminalMetricSpec("overall", "overall")),
        evaluator=SimpleNamespace(default_query=None), task_settings={},
    )
    controller._loaded_generator_state = object()
    controller.evaluation = controller.generator = controller.candidate_execution = None
    controller._generator_operation_lock = Lock()
    controller._seeds = SeedLedger(101)
    monkeypatch.setattr(runtime, "HiddenVerifier", Verifier)
    reports, evaluations = controller._official_reports(
        (artifact,), hidden, phase=EvaluationPhase(phase_name),
    )
    assert not reports[0].technically_eligible
    assert reports[0].failures == ("guided:evaluation_failed",)
    assert set(reports[0].distributions) == {"base"}
    assert set(evaluations[artifact.artifact_id]) == {"base"}
