"""Synthetic ABI/score tests only; these are not physical calibration evidence."""
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.plugins.plasma_transport import (
    TRANSPORT_PROTOCOL_ID, VPFP_TRANSPORT_FIELDS, canonicalize_1d, fit_calibration,
)
from pde_agent_verifier_solver.plugins.plasma_transport_settings import (
    check_feedback_objective, check_guidance_objective, transport_calibration,
)
from pde_agent_verifier_solver.plugins.vpfp.product import build_vpfp_plugin
from pde_agent_verifier_solver.plugins.vpfp.tasks.observable import VPFPObservableTask
from pde_agent_verifier_solver.plugins.vp_1d1v.product import build_artemis_plugin
from pde_agent_verifier_solver.plugins.vp_1d1v.task import ArtemisObservableTask
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, AnswerView, CandidateEvaluationOutput, Case, CaseSource,
    CaseVisibility, EvaluationPhase, EvaluationPlan,
)


def calibration(scale=1.):
    values = canonicalize_1d({"M0": [1.], "M1": [1.], "M2": [2.], "M3": [5.], "E": [scale]})
    return fit_calibration([
        {"family": family, "stratum": "fixture", "root_id": family + "-private-root", "values": values}
        for family in ("strong", "weak", "artemis")
    ], units={"density": 1., "velocity": 1., "electric_field": 1.},
        provenance={"fixture": "synthetic test, not scientific validation", "path": "/private/calibration"})


def settings(query, cal=None):
    return {"metric_profile": TRANSPORT_PROTOCOL_ID,
            "metric_calibration": (cal or calibration()).to_dict(), "default_query": query}


def vpfp_query():
    return AnswerQuery.at_points(fields=VPFP_TRANSPORT_FIELDS,
                                points=({"t": .5, "x": .25, "y": .75},) * 2)


def test_transport_settings_require_explicit_frozen_calibration():
    assert transport_calibration() is None
    with pytest.raises(ValueError, match="requires"):
        transport_calibration(TRANSPORT_PROTOCOL_ID)
    with pytest.raises(ValueError, match="explicit"):
        transport_calibration(None, calibration())


@pytest.mark.parametrize("family", ["vpfp", "artemis"])
def test_public_brief_exposes_score_not_private_reference_metadata(family):
    task = VPFPObservableTask() if family == "vpfp" else ArtemisObservableTask()
    cal = calibration()
    brief = task.agent_brief(settings(vpfp_query(), cal))
    assert TRANSPORT_PROTOCOL_ID in brief and cal.objective_hash in brief
    assert "private-root" not in brief and "/private/calibration" not in brief
    assert "RMS" in brief and "25%" in brief
    if family == "vpfp":
        assert all(name in brief for name in VPFP_TRANSPORT_FIELDS)
        assert "Return only rho, momentum_x" not in brief
    else:
        assert "q_h=0.5*(M3-3*u*M2+2*M0*u^3)" in brief


def test_vpfp_task_scores_same_v2_objective_on_all_phases():
    query, cal = vpfp_query(), calibration()
    task = VPFPObservableTask()
    config = settings(query, cal)
    evaluation = task.create_evaluation(config)
    assert evaluation.program_id.endswith("-v2")
    case = Case("fixture", CaseSource.PUBLIC, CaseVisibility.HIDDEN, {})
    fields = {name: (1., 1.) for name in VPFP_TRANSPORT_FIELDS}
    answer = AnswerView(query.kind, fields=fields, points=query.points, shape=(2,))
    for phase in EvaluationPhase:
        result = evaluation.score_candidate(None, case, answer,
            EvaluationPlan(phase, query, settings=config), CandidateEvaluationOutput(answer))
        assert result.primary_error == 0.
        assert result.metrics[TRANSPORT_PROTOCOL_ID] == 0.
        check_feedback_objective(cal, SimpleNamespace(cases=(result,)))


def test_vpfp_v2_refuses_incomplete_old_query():
    query = AnswerQuery.at_points(fields=("rho", "momentum_x", "momentum_y", "kinetic_energy"),
                                 points=({"t": .5, "x": .2, "y": .3},))
    with pytest.raises(ValueError, match="all ten"):
        VPFPObservableTask().create_evaluation(settings(query))


def test_artemis_raw_abi_gets_same_central_transport_score():
    fields = ("M0", "M1", "M2", "M3", "E")
    query = AnswerQuery.at_points(fields=fields, points=({"t": .5, "x": .2},))
    config = {**settings(query), "causal_horizon": False}
    evaluation = ArtemisObservableTask().create_evaluation(config)
    case = Case("artemis-fixture", CaseSource.PUBLIC, CaseVisibility.HIDDEN, {})
    answer = AnswerView(query.kind, fields=dict(zip(fields, ((1.,), (1.,), (2.,), (5.,), (1.,)))),
                        points=query.points, shape=(1,))
    for phase in EvaluationPhase:
        result = evaluation.score_candidate(None, case, answer,
            EvaluationPlan(phase, query, settings=config), CandidateEvaluationOutput(answer))
        assert result.primary_error == result.metrics[TRANSPORT_PROTOCOL_ID] == 0.


def test_strong_restore_retains_new_and_legacy_truth_callbacks():
    from pde_agent_verifier_solver.plugins.vpfp.resume import _case_data, _record
    import numpy as np
    case = Case("restored", CaseSource.PUBLIC, CaseVisibility.HIDDEN,
                {"arrays": {"t": (0., 1.), "x": (0., 1.), "y": (0., 1.)}})
    calls = []
    def macros(case_id, points):
        calls.append("legacy")
        return {"rho": np.ones(len(points)), "momentum": np.zeros((len(points), 2)),
                "kinetic_energy": np.ones(len(points))}
    def transport(case_id, points):
        values = macros(case_id, points)
        calls[-1] = "transport"
        return {**values, "electric": np.zeros((len(points), 2)),
                "pressure": np.tile((1., 0., 1.), (len(points), 1)),
                "heat_flux": np.zeros((len(points), 2))}
    runtime = SimpleNamespace(evaluate_macros=macros, evaluate_transport_macros=transport)
    record = _record(runtime, _case_data(case))
    v2 = vpfp_query()
    answer = record.truth.answer(v2)
    assert set(answer.fields) == set(VPFP_TRANSPORT_FIELDS) and calls == ["transport"]
    record.truth.answer(AnswerQuery.at_points(fields=("rho", "momentum_x", "momentum_y", "kinetic_energy"),
                                             points=v2.points))
    assert calls == ["transport", "legacy"]


@pytest.mark.parametrize("builder", [build_vpfp_plugin, build_artemis_plugin])
def test_plugin_binding_pins_same_objective_and_rejects_task_override(builder):
    cal = calibration()
    plugin = builder(metric_profile=TRANSPORT_PROTOCOL_ID, metric_calibration=cal)
    task = plugin.task("observable_model")
    assert task.macro_matcher is plugin.macro_matcher
    assert plugin.macro_matcher.objective_hash == cal.objective_hash
    assert "central heat flux" in task.info.description
    assert "-v2" in task.info.program_id
    with pytest.raises(ValueError, match="bound transport objective"):
        task.agent_brief(settings(vpfp_query(), calibration(2.)))


def test_guidance_rejects_old_snapshot_or_different_calibration():
    cal = calibration()
    with pytest.raises(ValueError, match="another scoring objective"):
        check_guidance_objective(cal, SimpleNamespace(report={"reward_matcher_sha256": "0" * 64}), {})
    check_guidance_objective(cal, SimpleNamespace(report={"reward_matcher_sha256": cal.objective_hash}), {})
    with pytest.raises(ValueError, match="bound scoring objective"):
        check_guidance_objective(cal, None, settings(vpfp_query(), calibration(2.)))


def test_guidance_uses_the_promoted_transport_label_and_keeps_invalid_attempts():
    cal = calibration()
    row = SimpleNamespace(primary_error=.2, metrics={TRANSPORT_PROTOCOL_ID: .2})
    check_feedback_objective(cal, SimpleNamespace(cases=(row,)))
    with pytest.raises(ValueError, match="legacy"):
        check_feedback_objective(None, SimpleNamespace(cases=(row,)))
    with pytest.raises(ValueError, match="promoted"):
        check_feedback_objective(cal, SimpleNamespace(cases=(SimpleNamespace(primary_error=.1, metrics=row.metrics),)))
    with pytest.raises(ValueError, match="promoted"):
        check_feedback_objective(cal, SimpleNamespace(cases=(SimpleNamespace(primary_error=.2, metrics={}),)))
    check_feedback_objective(cal, SimpleNamespace(cases=(SimpleNamespace(primary_error=float("nan"), metrics={}),)))
