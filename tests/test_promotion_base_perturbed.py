"""Explicit BASE-perturbed promotion uses the existing shared report contract."""

from dataclasses import replace

import pytest

from pde_agent_verifier_solver.engine.promotion import (
    DEFAULT_PROMOTION_PRESET,
    BaseImprovementWithRobustnessGuardrails,
    build_promotion_program,
)
from pde_agent_verifier_solver.sdk import HiddenSuiteSpec
from pde_agent_verifier_solver.sdk.promotion import (
    PROMOTION_DISTRIBUTIONS,
    PromotionCandidateReport,
    PromotionContext,
    PromotionMetricSummary,
)


def _metric(value):
    return PromotionMetricSummary(case_count=64, mean=value, cvar=value, maximum=value)


def _report(name, *, base, perturbed):
    distributions = {"base": _metric(base)}
    if perturbed is not None:
        distributions["base_perturbed"] = _metric(perturbed)
    return PromotionCandidateReport(
        candidate_id=name, candidate_digest=name, technically_eligible=True,
        distributions=distributions,
    )


def _program(from_config):
    if from_config:
        return build_promotion_program({
            "preset": DEFAULT_PROMOTION_PRESET,
            "settings": {
                "relative_margin": 0.05,
                "guardrail_distributions": ["base_perturbed"],
                "require_base_improvement": True,
            },
        })
    return BaseImprovementWithRobustnessGuardrails(
        relative_margin=0.05, guardrail_distributions=("base_perturbed",),
    )


def test_base_perturbed_is_already_a_shared_official_and_promotion_distribution():
    assert "base_perturbed" in PROMOTION_DISTRIBUTIONS
    assert HiddenSuiteSpec(base=64, base_perturbed=64).as_dict()["base_perturbed"] == 64
    assert _program(True).guardrail_distributions == ("base_perturbed",)


@pytest.mark.parametrize("from_config", [False, True])
@pytest.mark.parametrize("perturbed, promoted", [(1.04, True), (1.05, True), (1.050001, False)])
def test_base_improvement_requires_perturbed_mean_within_five_percent(from_config, perturbed, promoted):
    incumbent = _report("incumbent", base=1.0, perturbed=1.0)
    challenger = _report("challenger", base=0.7, perturbed=perturbed)
    decision = _program(from_config).select(
        incumbent.candidate_digest, (incumbent, challenger), PromotionContext(round_index=1),
    )
    assert decision.promoted is promoted
    assert decision.selected_digest == ("challenger" if promoted else "incumbent")
    ranking = next(row for row in decision.ranking if row.candidate_digest == "challenger")
    assert ranking.eligible is promoted
    if not promoted:
        assert ranking.reason == "base_perturbed_relative_margin_failed"
    # Promotion never drops a submitted Candidate or the independent stratum.
    assert {row.candidate_digest for row in decision.ranking} == {"incumbent", "challenger"}
    assert challenger.distributions["base_perturbed"].mean == perturbed


@pytest.mark.parametrize("from_config", [False, True])
@pytest.mark.parametrize("incumbent_perturbed, challenger_perturbed", [(None, 1.0), (1.0, None), (None, None)])
def test_requested_perturbed_guardrail_fails_closed_when_either_report_is_missing(
    from_config, incumbent_perturbed, challenger_perturbed,
):
    incumbent = _report("incumbent", base=1.0, perturbed=incumbent_perturbed)
    challenger = _report("challenger", base=0.7, perturbed=challenger_perturbed)
    decision = _program(from_config).select(
        "incumbent", (incumbent, challenger), PromotionContext(round_index=1),
    )
    assert not decision.promoted
    assert decision.selected_digest == "incumbent"
    ranking = next(row for row in decision.ranking if row.candidate_digest == "challenger")
    assert not ranking.eligible
    assert ranking.reason == "base_perturbed_evaluation_missing"


def test_initial_selection_still_has_no_incumbent_relative_guardrail():
    first = _report("first", base=1.0, perturbed=1.0)
    best = _report("best", base=0.7, perturbed=None)
    decision = _program(True).select(None, (first, best), PromotionContext(round_index=1))
    assert decision.selected_digest == "best"
    assert decision.reason == "initial_candidate_selected_by_base_metrics"


def test_legacy_optional_default_distributions_remain_optional():
    incumbent = _report("incumbent", base=1.0, perturbed=None)
    challenger = _report("challenger", base=0.7, perturbed=None)
    decision = BaseImprovementWithRobustnessGuardrails().select(
        "incumbent", (incumbent, challenger), PromotionContext(round_index=1),
    )
    assert decision.selected_digest == "challenger"
    assert decision.promoted


@pytest.mark.parametrize("statistic", ["mean", "cvar", "maximum"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_perturbed_metrics_still_fail_at_sdk_construction(statistic, value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        replace(_metric(1.0), **{statistic: value})


def test_failed_technical_evaluation_still_prevents_promotion():
    incumbent = _report("incumbent", base=1.0, perturbed=1.0)
    challenger = replace(_report("challenger", base=0.7, perturbed=0.8), technically_eligible=False)
    decision = _program(True).select(
        "incumbent", (incumbent, challenger), PromotionContext(round_index=1),
    )
    assert decision.selected_digest == "incumbent"
    ranking = next(row for row in decision.ranking if row.candidate_digest == "challenger")
    assert ranking.reason == "technical_evaluation_failed"
