from pathlib import Path

import pytest

from pde_agent_verifier_solver.engine.promotion import (
    DEFAULT_PROMOTION_PRESET,
    MINIMIZE_METRIC_PROMOTION_PRESET,
    BaseImprovementWithRobustnessGuardrails,
    FunctionPromotionProgram,
    build_promotion_program,
    validate_promotion_decision,
)
from pde_agent_verifier_solver.sdk.promotion import (
    PromotionCandidateReport,
    PromotionContext,
    PromotionDecision,
    PromotionMetricSummary,
    PromotionRankingEntry,
)


def _metric(mean: float) -> PromotionMetricSummary:
    return PromotionMetricSummary(
        case_count=64,
        mean=mean,
        cvar=mean * 1.2,
        maximum=mean * 1.5,
    )


def _report(
    candidate_id: str,
    digest: str,
    *,
    base: float,
    guided: float,
    perturbed_guardrail: bool = True,
    technically_eligible: bool = True,
) -> PromotionCandidateReport:
    return PromotionCandidateReport(
        candidate_id=candidate_id,
        candidate_digest=digest,
        technically_eligible=technically_eligible,
        distributions={
            "base": _metric(base),
            "guided": _metric(guided),
        },
        guardrails={
            "base_perturbed_non_regression": perturbed_guardrail,
        },
    )


def test_default_reports_every_candidate_but_keeps_incumbent_on_guided_regression():
    incumbent = _report("incumbent", "a" * 64, base=1.0, guided=1.0)
    challenger = _report("challenger", "b" * 64, base=0.7, guided=1.2)

    decision = BaseImprovementWithRobustnessGuardrails().select(
        incumbent.candidate_digest,
        (incumbent, challenger),
        PromotionContext(round_index=1),
    )

    assert decision.selected_digest == incumbent.candidate_digest
    assert not decision.promoted
    assert {row.candidate_digest for row in decision.ranking} == {
        incumbent.candidate_digest,
        challenger.candidate_digest,
    }
    rejected = next(
        row for row in decision.ranking
        if row.candidate_digest == challenger.candidate_digest
    )
    assert not rejected.eligible
    assert rejected.reason == "guided_relative_margin_failed"


def test_default_promotes_best_base_candidate_after_guardrails_pass():
    incumbent = _report("incumbent", "a" * 64, base=1.0, guided=1.0)
    challenger = _report("challenger", "b" * 64, base=0.7, guided=1.02)

    decision = BaseImprovementWithRobustnessGuardrails().select(
        incumbent.candidate_digest,
        (incumbent, challenger),
        PromotionContext(round_index=2),
    )

    assert decision.selected_digest == challenger.candidate_digest
    assert decision.promoted
    assert decision.reason == "base_improved_with_robustness_guardrails"


def test_minimize_base_mean_ignores_unconfigured_guided_and_perturbed_regressions():
    incumbent = _report("incumbent", "a" * 64, base=1.0, guided=1.0)
    challenger = _report(
        "challenger",
        "b" * 64,
        base=0.7,
        guided=9.0,
        perturbed_guardrail=False,
    )
    configured = build_promotion_program(
        {
            "preset": MINIMIZE_METRIC_PROMOTION_PRESET,
            "settings": {
                "primary": {"distribution": "base", "statistic": "mean"},
                "require_improvement": True,
                "minimum_improvement": 0.0,
                "tie_breakers": [],
                "guardrails": [],
            },
        }
    )

    decision = configured.select(
        incumbent.candidate_digest,
        (incumbent, challenger),
        PromotionContext(round_index=1),
    )

    assert decision.selected_digest == challenger.candidate_digest
    assert decision.promoted


def test_minimize_metric_never_selects_a_technically_invalid_challenger():
    incumbent = _report("incumbent", "a" * 64, base=1.0, guided=1.0)
    challenger = _report(
        "invalid",
        "b" * 64,
        base=0.1,
        guided=0.1,
        technically_eligible=False,
    )
    configured = build_promotion_program(
        {"preset": MINIMIZE_METRIC_PROMOTION_PRESET}
    )

    decision = configured.select(
        incumbent.candidate_digest,
        (challenger, incumbent),
        PromotionContext(round_index=1),
    )

    assert decision.selected_digest == incumbent.candidate_digest
    rejected = next(
        row for row in decision.ranking
        if row.candidate_digest == challenger.candidate_digest
    )
    assert not rejected.eligible
    assert rejected.reason == "technical_evaluation_failed"


def test_minimize_metric_initial_selection_ignores_relative_incumbent_rules():
    first = _report("first", "a" * 64, base=1.0, guided=1.0)
    best = _report(
        "best",
        "b" * 64,
        base=0.95,
        guided=9.0,
        perturbed_guardrail=False,
    )
    invalid = _report(
        "invalid",
        "c" * 64,
        base=0.1,
        guided=0.1,
        technically_eligible=False,
    )
    configured = build_promotion_program(
        {
            "preset": MINIMIZE_METRIC_PROMOTION_PRESET,
            "settings": {
                "minimum_improvement": 0.5,
                "guardrails": [
                    {"distribution": "guided", "statistic": "mean"}
                ],
            },
        }
    )

    decision = configured.select(
        None,
        (first, invalid, best),
        PromotionContext(round_index=1),
    )

    assert decision.selected_digest == best.candidate_digest
    assert decision.promoted
    assert decision.reason == "initial_candidate_selected_by_configured_metrics"
    rejected = next(
        row for row in decision.ranking
        if row.candidate_digest == invalid.candidate_digest
    )
    assert not rejected.eligible
    assert rejected.reason == "technical_evaluation_failed"


def test_default_initial_selection_uses_best_complete_base_report_directly():
    first = _report("first", "a" * 64, base=1.0, guided=1.0)
    best = _report(
        "best",
        "b" * 64,
        base=0.7,
        guided=9.0,
    )
    incomplete = PromotionCandidateReport(
        candidate_id="incomplete",
        candidate_digest="c" * 64,
        technically_eligible=True,
        distributions={"guided": _metric(0.1)},
    )

    decision = BaseImprovementWithRobustnessGuardrails().select(
        None,
        (first, incomplete, best),
        PromotionContext(round_index=1),
    )

    assert decision.selected_digest == best.candidate_digest
    assert decision.promoted
    assert decision.reason == "initial_candidate_selected_by_base_metrics"
    rejected = next(
        row for row in decision.ranking
        if row.candidate_digest == incomplete.candidate_digest
    )
    assert not rejected.eligible
    assert rejected.reason == "base_evaluation_missing"


def test_framework_rejects_an_incomplete_custom_promotion_ranking():
    incumbent = _report("incumbent", "a" * 64, base=1.0, guided=1.0)
    challenger = _report("challenger", "b" * 64, base=0.8, guided=1.0)
    decision = PromotionDecision(
        selected_digest=incumbent.candidate_digest,
        promoted=False,
        reason="custom",
        ranking=(
            PromotionRankingEntry(
                candidate_digest=incumbent.candidate_digest,
                rank=1,
                eligible=True,
                reason="custom",
            ),
        ),
    )

    with pytest.raises(ValueError, match="cover every official report"):
        validate_promotion_decision(
            incumbent.candidate_digest,
            (incumbent, challenger),
            decision,
        )


def test_framework_rejects_custom_selection_of_invalid_challenger():
    incumbent = _report("incumbent", "a" * 64, base=1.0, guided=1.0)
    invalid = _report(
        "invalid",
        "b" * 64,
        base=0.1,
        guided=0.1,
        technically_eligible=False,
    )
    decision = PromotionDecision(
        selected_digest=invalid.candidate_digest,
        promoted=True,
        reason="custom",
        ranking=tuple(
            PromotionRankingEntry(
                candidate_digest=report.candidate_digest,
                rank=index,
                eligible=True,
                reason="custom",
            )
            for index, report in enumerate((invalid, incumbent), start=1)
        ),
    )

    with pytest.raises(ValueError, match="technically invalid"):
        validate_promotion_decision(
            incumbent.candidate_digest,
            (incumbent, invalid),
            decision,
        )


def test_framework_rejects_inconsistent_custom_promoted_flag():
    incumbent = _report("incumbent", "a" * 64, base=1.0, guided=1.0)
    challenger = _report("challenger", "b" * 64, base=0.8, guided=1.0)
    decision = PromotionDecision(
        selected_digest=challenger.candidate_digest,
        promoted=False,
        reason="custom",
        ranking=tuple(
            PromotionRankingEntry(
                candidate_digest=report.candidate_digest,
                rank=index,
                eligible=True,
                reason="custom",
            )
            for index, report in enumerate((challenger, incumbent), start=1)
        ),
    )

    with pytest.raises(ValueError, match="promoted flag"):
        validate_promotion_decision(
            incumbent.candidate_digest,
            (incumbent, challenger),
            decision,
        )


def test_framework_validates_initial_selection_and_requires_promoted_true():
    first = _report("first", "a" * 64, base=1.0, guided=1.0)
    best = _report("best", "b" * 64, base=0.8, guided=1.0)
    reports = (first, best)
    valid = BaseImprovementWithRobustnessGuardrails().select(
        None,
        reports,
        PromotionContext(round_index=1),
    )

    validate_promotion_decision(None, reports, valid)

    invalid = PromotionDecision(
        selected_digest=valid.selected_digest,
        promoted=False,
        reason="custom forgot initial promotion semantics",
        ranking=valid.ranking,
    )
    with pytest.raises(ValueError, match="promoted flag"):
        validate_promotion_decision(None, reports, invalid)


def test_framework_rejects_ranking_ineligible_initial_selection():
    report = _report("candidate", "a" * 64, base=1.0, guided=1.0)
    decision = PromotionDecision(
        selected_digest=report.candidate_digest,
        promoted=True,
        reason="custom",
        ranking=(
            PromotionRankingEntry(
                candidate_digest=report.candidate_digest,
                rank=None,
                eligible=False,
                reason="custom rejected it",
            ),
        ),
    )

    with pytest.raises(ValueError, match="ranking-ineligible"):
        validate_promotion_decision(None, (report,), decision)


def test_minimize_metric_keeps_incumbent_when_configured_metrics_tie():
    incumbent = _report("incumbent", "z" * 64, base=1.0, guided=2.0)
    challenger = _report("challenger", "a" * 64, base=1.0, guided=0.1)
    configured = build_promotion_program(
        {
            "preset": MINIMIZE_METRIC_PROMOTION_PRESET,
            "settings": {"require_improvement": False},
        }
    )

    decision = configured.select(
        incumbent.candidate_digest,
        (challenger, incumbent),
        PromotionContext(round_index=1),
    )

    assert decision.selected_digest == incumbent.candidate_digest
    assert not decision.promoted


def test_minimize_metric_uses_cvar_tie_breaker_when_primary_ties():
    incumbent = PromotionCandidateReport(
        candidate_id="incumbent",
        candidate_digest="a" * 64,
        technically_eligible=True,
        distributions={
            "base": PromotionMetricSummary(64, mean=1.0, cvar=1.4, maximum=2.0)
        },
    )
    challenger = PromotionCandidateReport(
        candidate_id="challenger",
        candidate_digest="b" * 64,
        technically_eligible=True,
        distributions={
            "base": PromotionMetricSummary(64, mean=1.0, cvar=1.2, maximum=2.5)
        },
    )
    configured = build_promotion_program(
        {
            "preset": MINIMIZE_METRIC_PROMOTION_PRESET,
            "settings": {
                "require_improvement": False,
                "tie_breakers": [
                    {"distribution": "base", "statistic": "cvar"}
                ],
            },
        }
    )

    decision = configured.select(
        incumbent.candidate_digest,
        (incumbent, challenger),
        PromotionContext(round_index=1),
    )

    assert decision.selected_digest == challenger.candidate_digest
    assert decision.promoted


def test_minimize_metric_enforces_minimum_improvement_and_explicit_guardrail():
    incumbent = _report("incumbent", "a" * 64, base=1.0, guided=1.0)
    small_gain = _report("small-gain", "b" * 64, base=0.95, guided=1.0)
    configured = build_promotion_program(
        {
            "preset": MINIMIZE_METRIC_PROMOTION_PRESET,
            "settings": {
                "minimum_improvement": 0.1,
                "guardrails": [
                    {"distribution": "guided", "statistic": "mean"}
                ],
            },
        }
    )

    decision = configured.select(
        incumbent.candidate_digest,
        (incumbent, small_gain),
        PromotionContext(round_index=1),
    )
    assert decision.selected_digest == incumbent.candidate_digest
    assert decision.reason == "incumbent_kept_minimum_improvement_not_met"

    large_gain_but_regression = _report(
        "guardrail-failure",
        "c" * 64,
        base=0.7,
        guided=1.01,
    )
    decision = configured.select(
        incumbent.candidate_digest,
        (incumbent, large_gain_but_regression),
        PromotionContext(round_index=2),
    )
    assert decision.selected_digest == incumbent.candidate_digest
    rejected = next(
        row for row in decision.ranking
        if row.candidate_digest == large_gain_but_regression.candidate_digest
    )
    assert rejected.reason == "guided.mean_non_regression_failed"


def test_custom_program_receives_all_reports_and_settings(tmp_path: Path):
    program = tmp_path / "my_promotion.py"
    program.write_text(
        "from pde_agent_verifier_solver.sdk import PromotionDecision, "
        "PromotionRankingEntry\n"
        "def select(incumbent_digest, reports, context):\n"
        "    assert context.settings['choose'] == 'last'\n"
        "    chosen = reports[-1].candidate_digest\n"
        "    return PromotionDecision(\n"
        "        selected_digest=chosen,\n"
        "        promoted=chosen != incumbent_digest,\n"
        "        reason='user_rule',\n"
        "        ranking=tuple(\n"
        "            PromotionRankingEntry(\n"
        "                candidate_digest=row.candidate_digest,\n"
        "                rank=index + 1,\n"
        "                eligible=True,\n"
        "                reason='user_rule',\n"
        "            )\n"
        "            for index, row in enumerate(reports)\n"
        "        ),\n"
        "    )\n",
        encoding="utf-8",
    )
    configured = build_promotion_program(
        {
            "program": "my_promotion.py",
            "entrypoint": "select",
            "settings": {"choose": "last"},
        },
        config_dir=tmp_path,
    )
    reports = (
        _report("incumbent", "a" * 64, base=1.0, guided=1.0),
        _report("challenger", "b" * 64, base=0.9, guided=0.9),
    )

    decision = configured.select(
        reports[0].candidate_digest,
        reports,
        PromotionContext(round_index=3),
    )

    assert configured.program_id == "python:my_promotion.py:select"
    assert decision.selected_digest == reports[1].candidate_digest


def test_custom_program_can_select_using_immutable_task_named_metrics(
    tmp_path: Path,
):
    program = tmp_path / "named_metric_promotion.py"
    program.write_text(
        "from pde_agent_verifier_solver.sdk import PromotionDecision, "
        "PromotionRankingEntry\n"
        "def select(incumbent_digest, reports, context):\n"
        "    ordered = sorted(\n"
        "        reports,\n"
        "        key=lambda row: row.distributions['base'].metrics['spectral_error'],\n"
        "    )\n"
        "    chosen = ordered[0].candidate_digest\n"
        "    return PromotionDecision(\n"
        "        selected_digest=chosen,\n"
        "        promoted=chosen != incumbent_digest,\n"
        "        reason='lowest_task_named_metric',\n"
        "        ranking=tuple(\n"
        "            PromotionRankingEntry(\n"
        "                candidate_digest=row.candidate_digest,\n"
        "                rank=index,\n"
        "                eligible=True,\n"
        "                reason='task_named_metric',\n"
        "            )\n"
        "            for index, row in enumerate(ordered, start=1)\n"
        "        ),\n"
        "    )\n",
        encoding="utf-8",
    )
    original_metrics = {"spectral_error": 0.9}
    incumbent_summary = PromotionMetricSummary(
        2,
        mean=0.1,
        cvar=0.1,
        maximum=0.1,
        metrics=original_metrics,
    )
    original_metrics["spectral_error"] = 0.01
    with pytest.raises(TypeError):
        incumbent_summary.metrics["spectral_error"] = 3.0
    reports = (
        PromotionCandidateReport(
            candidate_id="incumbent",
            candidate_digest="a" * 64,
            technically_eligible=True,
            distributions={"base": incumbent_summary},
        ),
        PromotionCandidateReport(
            candidate_id="challenger",
            candidate_digest="b" * 64,
            technically_eligible=True,
            distributions={
                "base": PromotionMetricSummary(
                    2,
                    mean=0.2,
                    cvar=0.2,
                    maximum=0.2,
                    metrics={"spectral_error": 0.2},
                )
            },
        ),
    )
    configured = build_promotion_program(
        {
            "program": "named_metric_promotion.py",
            "entrypoint": "select",
        },
        config_dir=tmp_path,
    )

    decision = configured.select(
        reports[0].candidate_digest,
        reports,
        PromotionContext(round_index=1),
    )

    assert incumbent_summary.metrics == {"spectral_error": 0.9}
    assert decision.selected_digest == reports[1].candidate_digest
    assert decision.reason == "lowest_task_named_metric"


def test_function_promotion_passes_none_to_custom_initial_selector():
    reports = (
        _report("first", "a" * 64, base=1.0, guided=1.0),
        _report("best", "b" * 64, base=0.8, guided=1.0),
    )
    received = []

    def select(incumbent_digest, rows, context):
        received.append(incumbent_digest)
        return PromotionDecision(
            selected_digest=rows[-1].candidate_digest,
            promoted=True,
            reason=f"initial round {context.round_index}",
            ranking=tuple(
                PromotionRankingEntry(
                    candidate_digest=row.candidate_digest,
                    rank=index,
                    eligible=True,
                    reason="custom",
                )
                for index, row in enumerate(rows, start=1)
            ),
        )

    configured = FunctionPromotionProgram(select, program_id="test-initial")
    decision = configured.select(
        None,
        reports,
        PromotionContext(round_index=1),
    )

    assert received == [None]
    assert decision.promoted


def test_canonical_promotion_rejects_replay_distribution_and_guardrail():
    with pytest.raises(ValueError, match="unknown distribution"):
        PromotionCandidateReport(
            candidate_id="legacy",
            candidate_digest="a" * 64,
            technically_eligible=True,
            distributions={"replay": _metric(1.0)},
        )

    with pytest.raises(ValueError, match="unknown guardrail distribution"):
        BaseImprovementWithRobustnessGuardrails(
            guardrail_distributions=("replay",),
        )


@pytest.mark.parametrize(
    "specification",
    (
        {"preset": "unknown", "settings": {}},
        {
            "preset": DEFAULT_PROMOTION_PRESET,
            "settings": {"require_base_improvement": "yes"},
        },
        {
            "preset": MINIMIZE_METRIC_PROMOTION_PRESET,
            "settings": {
                "primary": {"distribution": "base", "statistic": "median"}
            },
        },
        {
            "preset": MINIMIZE_METRIC_PROMOTION_PRESET,
            "settings": {
                "require_improvement": False,
                "minimum_improvement": 0.1,
            },
        },
    ),
)
def test_invalid_promotion_configuration_is_rejected(specification):
    with pytest.raises((TypeError, ValueError)):
        build_promotion_program(specification)
