"""CPU numerical and routing checks, not ARTEMIS model-quality validation."""

from copy import deepcopy
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from pde_agent_verifier_solver.plugins.vp_1d1v import ArtemisPhysicalPackage, ArtemisRewardGuidance


SETTINGS = {"controller_seed": 101, "feedback_selection": "first_80_clean_base_feedback_order",
            "feedback_case_count": 80, "guidance_maximum_wall_seconds": 1800}


@pytest.fixture(scope="module")
def package():
    checkout = Path(os.environ.get("ARTEMIS_TEST_PHYSICAL_CHECKOUT",
        str(Path(__file__).resolve().parents[2] / "artemis-oracle-opt5-20260909")))
    if not checkout.is_dir():
        pytest.skip("pinned external Artemis physical package is not installed")
    return ArtemisPhysicalPackage(checkout, expected_revision="e0cccc649a9ccf48ad69ac2d161c4a6b46010675")


@pytest.fixture
def guide(package):
    capsule_type = package.module("artemis_compact_v5_production_runtime").ArtemisCompactV5GuidanceCoordinateCapsule

    class Runtime:
        conditions = tuple(f"condition-{index}" for index in range(6))

        def guidance_coordinate_from_seed(self, seed):
            rng = random.Random(seed)
            return capsule_type.from_unit_uniforms(self.conditions[seed % 6], tuple(rng.random() for _ in range(3)))

    class Source:
        artifact_sha256 = "base-sha256"
        runtime = Runtime()

        def issue_guided_answer_first_records(self, capsules, **kwargs):
            self.last_capsules = tuple(capsules)
            return tuple(SimpleNamespace(case=SimpleNamespace(case_id=f"guided-{index}", parent_case_id=None))
                         for index in range(len(capsules)))

    return ArtemisRewardGuidance(package, Source(), settings={
        "condition_count": 6, "model_kind": "ou_hermite_ridge_v1", "hermite_degree": 1,
        "reward_transform": "log_with_floor", "sampling_steps": 2, "ridge_alpha": 1.0,
    })


def feedback_fixture(guide, *, condition_indices=None, count=86):
    capsules = []
    for index in range(count):
        condition = index % 6 if condition_indices is None else condition_indices[index]
        capsule = guide.runtime_api.ArtemisCompactV5GuidanceCoordinateCapsule.from_unit_uniforms(
            guide.conditions[condition], (0.02 + 0.96 * index / (count - 1), 0.5, 0.5))
        capsules.append(capsule)
        guide._coordinates[f"base-{index}"] = capsule
    latents, _ = guide._primitive_tensors(capsules)
    rows = tuple(SimpleNamespace(case=SimpleNamespace(case_id=f"base-{index}", parent_case_id=None),
                                 primary_error=float(torch.exp(0.3 * latent[0])))
                 for index, latent in enumerate(latents))
    return SimpleNamespace(cases=rows, round_index=0,
                           selected_artifact=SimpleNamespace(artifact_id="incumbent-s0"))


def fit(guide, feedback, previous=None, settings=None):
    return guide.fit_guidance(SimpleNamespace(artifact_sha256=guide.source.artifact_sha256),
                              previous, feedback, SETTINGS if settings is None else settings)


def test_first_80_base_selection_excludes_guided_and_children_before_labels(guide):
    feedback = feedback_fixture(guide)

    class UnreadableLabel:
        def __init__(self, identity, parent=None):
            self.case = SimpleNamespace(case_id=identity, parent_case_id=parent)

        @property
        def primary_error(self):
            raise AssertionError("excluded label was inspected")

    guided = tuple(UnreadableLabel(f"guided-{index}") for index in range(85))
    children = tuple(UnreadableLabel(f"child-{index}", "base-0") for index in range(85))
    discarded_base = tuple(UnreadableLabel(f"base-{index}") for index in range(80, 86))
    # JSON round-trip matches persisted sampling_audits restored by resume.py.
    guide.sampling_audits = json.loads(json.dumps([
        {"count": 85, "case_ids": [row.case.case_id for row in guided]}]))
    feedback.cases = guided[:40] + feedback.cases[:80] + children + guided[40:] + discarded_base
    state = fit(guide, feedback)
    assert state.report["feedback_case_ids"] == [f"base-{index}" for index in range(80)]
    assert state.report["training_case_ids"] == [f"base-{index}" for index in range(64)]
    assert state.report["validation_case_ids"] == [f"base-{index}" for index in range(64, 80)]
    assert state.report["available_clean_base_count"] == 86
    assert state.report["excluded_guided_root_count"] == state.report["excluded_child_count"] == 85
    assert guide.training_audits[-1]["feedback_case_count"] == 80


def test_holdout_does_not_fit_normalization_or_ridge_and_no_final_refit(guide):
    feedback = feedback_fixture(guide)
    original = fit(guide, feedback)
    changed = deepcopy(feedback)
    for row in changed.cases[64:80]:
        row.primary_error *= 1000.0  # Preserve the gate, change held-out magnitudes.
    updated = fit(guide, changed, original)
    assert original.report["active_condition_indices"] == list(range(6))
    assert updated.report["active_condition_indices"] == original.report["active_condition_indices"]
    assert original.normalization.to_dict() == updated.normalization.to_dict()
    assert all(torch.equal(value, updated.model.state_dict()[key])
               for key, value in original.model.state_dict().items())
    assert original.snapshot_sha256 != updated.snapshot_sha256
    assert original.report["split"]["normalization_fit_count"] == 64
    assert original.report["split"]["validation_labels_used_for_fit"] is False
    assert original.report["fit"]["final_fit_count"] == 64
    assert original.report["fit"]["final_refit"] == "none_after_holdout_activation"
    assert original.report["optimization"]["steps"] == 1


def test_missing_and_underrepresented_conditions_are_valid_zero_guidance(guide):
    conditions = [0, 1] * 32 + [0] * 7 + [1] + [2] * 8
    state = fit(guide, feedback_fixture(guide, condition_indices=conditions, count=80))
    assert state.report["active_condition_indices"] == [0]
    assert state.report["inactive_condition_indices"] == [1, 2, 3, 4, 5]
    assert state.report["split"]["training_count_by_condition"] == [32, 32, 0, 0, 0, 0]
    assert state.report["split"]["validation_count_by_condition"] == [7, 1, 8, 0, 0, 0]
    assert torch.count_nonzero(state.model.hermite_coefficients[2:]) == 0
    raw = torch.tensor([[0.1, 0.2, 0.3]] * 5, dtype=torch.float64)
    moved, labels, diagnostics = guide.api.sample_guided_primitives(
        state.model, state.normalization, 5, 101, guide.config,
        raw_latents=raw, raw_conditions=torch.arange(1, 6))
    assert torch.equal(raw, moved)
    assert torch.equal(labels, torch.arange(1, 6))
    assert diagnostics["raw_primitive_source"] == "provided_canonical_base_capsules"


def test_failed_holdout_gate_leaves_exact_raw_sampling_and_persists_guided_ids(guide):
    feedback = feedback_fixture(guide)
    for row in feedback.cases:
        row.primary_error = 1.0
    state = fit(guide, feedback)
    assert state.report["active_condition_indices"] == []
    request = SimpleNamespace(purpose=SimpleNamespace(value="official"),
                              visibility=SimpleNamespace(value="private"),
                              round_index=1, seed=101, count=6)
    records = guide.generate(SimpleNamespace(), state, request)
    assert guide.last_sampling_report["exact_raw_capsule_reuse_count"] == 6
    assert guide.sampling_audits[-1]["case_ids"] == [record.case.case_id for record in records]
    assert guide.sampling_audits[-1]["target_solver_id"] == "incumbent-s0"
    json.dumps(guide.sampling_audits, allow_nan=False)


def test_every_round_refits_for_current_incumbent_without_reusing_previous_model(guide):
    feedback = feedback_fixture(guide)
    first = fit(guide, feedback)
    feedback.round_index = 1
    feedback.selected_artifact.artifact_id = "winner-r1"
    second = fit(guide, feedback, first)
    assert second.model is not first.model
    assert first.target_solver_id == "incumbent-s0" and second.target_solver_id == "winner-r1"
    assert first.snapshot_sha256 != second.snapshot_sha256
    assert all(audit["previous_model_reused"] is False for audit in guide.training_audits)
    assert all(audit["initialization"] == "fresh_fit_from_incumbent_feedback"
               for audit in guide.training_audits)
    assert all(parameter.requires_grad is False for parameter in second.model.parameters())
    assert second.model.config.sha256 == guide.config.sha256


def test_aligned_fit_honors_wall_cap_without_recording_success(guide, monkeypatch):
    from pde_agent_verifier_solver.plugins.vp_1d1v import guidance
    feedback = feedback_fixture(guide)
    ticks = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(guidance, "perf_counter", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="wall budget"):
        fit(guide, feedback, settings={**SETTINGS, "guidance_maximum_wall_seconds": 1.0})
    assert guide.training_audits == []


def test_legacy_profile_still_calls_unmodified_pinned_fitter_on_all_roots(guide, monkeypatch):
    feedback = feedback_fixture(guide, count=84)
    called = []
    legacy_fit = guide.api.fit_reward_model

    def spy(latents, conditions, rewards, **kwargs):
        called.append(len(rewards))
        return legacy_fit(latents, conditions, rewards, **kwargs)

    monkeypatch.setattr(guide.api, "fit_reward_model", spy)
    state = fit(guide, feedback, settings={"controller_seed": 101})
    assert called == [84]
    assert state.report["sample_count"] == 84
    assert state.report["model_selection_on_crossfit"] is True
    assert "feedback_selection" not in state.report


@pytest.mark.parametrize("problem,error", [
    ("count", "first 80"), ("mode", "first 80"),
    ("short", "at least 80"), ("old_audit", "start a fresh aligned campaign"),
])
def test_aligned_selection_rejects_incompatible_inputs(guide, problem, error):
    feedback = feedback_fixture(guide)
    settings = dict(SETTINGS)
    if problem == "count":
        settings["feedback_case_count"] = 86
    elif problem == "mode":
        settings["feedback_selection"] = "label_sorted"
    elif problem == "short":
        feedback.cases = feedback.cases[:79]
    else:
        guide.sampling_audits = [{"count": 5}]
    with pytest.raises(ValueError, match=error):
        fit(guide, feedback, settings=settings)
    assert guide.training_audits == []
