"""Qualification reports held-out error, uncertainty and evidence gaps."""

from dataclasses import replace

import pytest

from pde_agent_verifier_solver.plugins.guidance_qualification import QualificationScore, qualify_reward_guidance
from pde_agent_verifier_solver.sdk import CaseRecord, GenerationKind
from pde_agent_verifier_solver.sdk.sampling import RecoverableSamplingRejection

from tests.test_reward_guidance_backend import _request, _wrapped


def _qualify(wrapper, guide, **changes):
    options = dict(base_state=None, guide=guide, request=_request(count=2), seeds=tuple(range(32)), incumbent_artifact_id=guide.selected_artifact_id, matcher_sha256="b" * 64, score=lambda row: QualificationScore(row.case.public_data["condition"], physical_valid=True), bootstrap_replicates=200)
    options.update(changes)
    return qualify_reward_guidance(wrapper, **options)


def test_heldout_qualification_reports_positive_lift_and_pool_ess():
    backend, wrapper, guide = _wrapped()
    report = _qualify(wrapper, guide)
    assert not backend.live
    assert report["status"] == "passed_clean_hardness"
    assert report["errors"]["guided"]["mean"] > report["errors"]["base"]["mean"]
    assert report["uncertainty"]["mean_error_difference"][0] > 0
    assert report["errors"]["guided"]["cvar"] >= report["errors"]["guided"]["mean"]
    assert report["freshness"]["training_overlap"] == 0
    assert report["freshness"]["unique_evaluation_cases"] == 128
    assert 1 <= report["pool_weight_ess"]["mean"] <= 4
    assert report["physical_sampling_rejection_rate"] is None
    assert report["full_production_guidance_admission"] is False
    assert report["perturbation_guardrail"] == "not_evaluated"


def test_missing_physical_checks_remain_insufficient_even_with_lift():
    backend, wrapper, guide = _wrapped()
    report = _qualify(wrapper, guide, score=lambda row: QualificationScore(row.case.public_data["condition"]))
    assert not backend.live
    assert report["status"] == "insufficient_evidence"
    assert report["physical_validation"]["guided"]["unknown"] == 64


def test_frozen_incumbent_and_training_identity_required():
    _, wrapper, guide = _wrapped()
    with pytest.raises(ValueError, match="incumbent differs"):
        _qualify(wrapper, guide, incumbent_artifact_id="new-solver")
    with pytest.raises(ValueError, match="training Case identities"):
        _qualify(wrapper, replace(guide, training_case_ids=()))


def test_overlap_rejected_and_resident_cases_released():
    backend, wrapper, guide = _wrapped()
    original_generate = backend.generate
    def repeat_training(base, state, request, progress=None):
        records = original_generate(base, state, request, progress)
        old = records[0]
        case = replace(old.case, case_id=guide.training_case_ids[0])
        backend.live.pop(old.case.case_id)
        backend.live[case.case_id] = case
        return (CaseRecord(case, old.truth),) + records[1:]
    backend.generate = repeat_training
    with pytest.raises(ValueError, match="overlap training"):
        _qualify(wrapper, guide)
    assert not backend.live


def test_physical_rejection_is_reported_without_fabricated_sample_rate():
    backend, wrapper, guide = _wrapped()
    original = wrapper.generate
    def reject_base(base, state, request, progress=None):
        if request.kind is GenerationKind.BASE:
            raise RecoverableSamplingRejection("test physical gate", audit={"constraint": "positivity"})
        return original(base, state, request, progress)
    wrapper.generate = reject_base
    report = _qualify(wrapper, guide, seeds=(1, 2))
    assert not backend.live
    assert report["status"] == "insufficient_evidence"
    assert report["request_rejections"]["base"]["rate"] == 1.0
    assert report["physical_sampling_rejection_rate"] is None
    assert report["uncertainty"] is None
