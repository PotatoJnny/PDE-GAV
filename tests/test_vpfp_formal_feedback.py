"""Mock-only acceptance of the approved mixed Official VPFP feedback recipe.

These tests inspect identities and dispatch, not physical accuracy or training.
"""
from dataclasses import dataclass, replace
from hashlib import sha256
import inspect
import json
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.plugins.vpfp import iteration_backend as integration
from pde_agent_verifier_solver.plugins.vpfp.round_solver_guidance import RoundGuidanceConfig
from pde_agent_verifier_solver.sdk import (
    Artifact, Case, CaseFeedback, CaseSource, CaseVisibility, RoundFeedback, TaskKind,
)


def _backend(tmp_path, *, policy=integration.FORMAL_FEEDBACK_SELECTION, trainer=None):
    base = SimpleNamespace(identity={"checkpoint_sha256": "e" * 64})
    capture = integration._CapturedSampler(base)
    inner = SimpleNamespace(runtime=object(), public_session_methods=(),
                            development_solution_session_methods=())
    calls = []

    def capture_train(**kwargs):
        calls.append(kwargs)
        if trainer is not None:
            return trainer(**kwargs)
        return SimpleNamespace(guide=object(), checkpoint_path="/private/mock.pt",
                               checkpoint_sha256="f" * 64, report={"mock_only": True})

    wrapper = integration.VPFPTrainingGuidanceBackend(
        inner, capture, output_root=tmp_path / "guidance", config=RoundGuidanceConfig(),
        trainer=capture_train, feedback_selection=policy)
    return wrapper, calls


def _feedback(wrapper, *, base_count=342, guided_count=341, perturbed_count=341):
    records = []
    # Interleave distributions to ensure selection uses feedback order, not a
    # coincidental contiguous array slice from today's engine implementation.
    for index in range(max(base_count, guided_count, perturbed_count)):
        for source, count in (("guided", guided_count), ("base", base_count),
                              ("perturbed", perturbed_count)):
            if index >= count:
                continue
            case_id = f"{source}-{index:04d}"
            case = Case(case_id=case_id,
                source=CaseSource.PERTURBED if source == "perturbed" else CaseSource.GENERATED,
                visibility=CaseVisibility.HIDDEN, public_data={}, round_index=2,
                parent_case_id=f"base-{index:04d}" if source == "perturbed" else None)
            records.append(CaseFeedback(case, .01 + index * .001, {}))
            if source != "perturbed":
                root = (100000 if source == "base" else 200000) + index
                wrapper._draws[case_id] = {
                    "source": source, "root_seed": root, "draw": {"mock_root": root}}
    artifact = Artifact(artifact_id="selected-solver", kind=TaskKind.OBSERVABLE_MODEL,
        name="frozen mock Solver", program=SimpleNamespace(tree_sha256="a" * 64),
        summary="mock", usage="mock", round_index=2, agent_id="agent")
    return RoundFeedback(2, artifact, tuple(records))


def _fit(wrapper, feedback):
    return wrapper.fit_guidance({}, None, feedback, {"controller_seed": 101})


def test_mixed_1024_selects_first_80_clean_roots_with_complete_exclusion_audit(tmp_path):
    wrapper, calls = _backend(tmp_path)
    feedback = _feedback(wrapper)
    _fit(wrapper, feedback)
    assert len(feedback.cases) == 1024
    rows = calls[0]["rows"]
    assert [row["case_id"] for row in rows] == [f"base-{index:04d}" for index in range(80)]
    assert [row["root_seed"] for row in rows] == list(range(100000, 100080))
    assert all(row["source"] == "base" for row in rows)
    assert calls[0]["config"] == RoundGuidanceConfig()
    audit = wrapper.training_audits[0]
    report = audit["feedback_selection_report"]
    assert report["available_clean_base_case_count"] == 342
    assert report["selected_clean_base_case_count"] == audit["clean_base_case_count"] == 80
    assert report["selection_uses_rewards"] is False
    assert report["invalid_labels_replace_roots"] is False
    assert len(report["excluded"]) == 944
    assert sum(row["reason"] == "not_clean_base" for row in report["excluded"]) == 682
    assert sum(row["reason"] == "outside_fixed_first_80_clean_base_feedback_order"
               for row in report["excluded"]) == 262
    selected = report["selected"]
    assert selected[0]["feedback_index"] == 1
    assert selected[63]["local_slot"] == "fit-0063"
    assert selected[64]["local_slot"] == "validation-0000"
    assert selected[79]["local_slot"] == "validation-0015"
    selected_ids = {row["case_id"] for row in selected}
    excluded_ids = {row["case_id"] for row in report["excluded"]}
    assert not selected_ids & excluded_ids
    assert selected_ids | excluded_ids == {item.case.case_id for item in feedback.cases}
    manifest_path = tmp_path / "guidance/round-003-feedback-selection.json"
    assert json.loads(manifest_path.read_text()) == report
    assert audit["feedback_selection_manifest"] == {
        "path": str(manifest_path), "sha256": sha256(manifest_path.read_bytes()).hexdigest()}
    assert manifest_path.stat().st_mode & 0o777 == 0o400


def test_selection_identical_despite_reward_changes_and_never_reads_excluded_rewards(tmp_path):
    first, first_calls = _backend(tmp_path / "first")
    first_feedback = _feedback(first)
    _fit(first, first_feedback)
    second, second_calls = _backend(tmp_path / "second")
    second_feedback = _feedback(second)

    class UnreadableReward:
        def __float__(self):
            raise AssertionError("excluded rewards must not be inspected")

    cases = []
    for item in second_feedback.cases:
        index = int(item.case.case_id.rsplit("-", 1)[-1])
        selected = item.case.case_id.startswith("base-") and index < 80
        cases.append(replace(item, primary_error=80. - index if selected else UnreadableReward()))
    _fit(second, replace(second_feedback, cases=tuple(cases)))
    assert first.training_audits[0]["feedback_selection_report"] == second.training_audits[0]["feedback_selection_report"]
    assert [row["case_id"] for row in first_calls[0]["rows"]] == [row["case_id"] for row in second_calls[0]["rows"]]
    assert [row["error"] for row in first_calls[0]["rows"]] != [row["error"] for row in second_calls[0]["rows"]]


@pytest.mark.parametrize("fault", ["too_few", "duplicate_case", "duplicate_root", "duplicate_surplus_root"])
def test_incomplete_or_nonunique_clean_inventory_fails_without_selection_or_training(tmp_path, fault):
    wrapper, calls = _backend(tmp_path)
    feedback = _feedback(wrapper, base_count=79 if fault == "too_few" else 342)
    if fault == "duplicate_case":
        feedback = replace(feedback, cases=feedback.cases + (feedback.cases[1],))
    if fault in ("duplicate_root", "duplicate_surplus_root"):
        position = 1 if fault == "duplicate_root" else 341
        wrapper._draws[f"base-{position:04d}"]["root_seed"] = 100000
    with pytest.raises(ValueError, match="unique actual clean BASE"):
        _fit(wrapper, feedback)
    assert not calls and not wrapper.training_audits
    assert not (tmp_path / "guidance").exists()


def test_invalid_fit_and_validation_keep_slots_and_manifest_survives_trainer_failure(tmp_path):
    def reject_invalid(**kwargs):
        rows = kwargs["rows"]
        assert len(rows) == 80
        assert rows[0]["case_id"] == "base-0000" and rows[0]["error"] is None
        assert rows[64]["case_id"] == "base-0064" and rows[64]["error"] is None
        assert rows[0]["valid"] is rows[64]["valid"] is False
        assert rows[79]["case_id"] == "base-0079"
        raise FloatingPointError("mock existing trainer invalid-feedback outcome")

    wrapper, calls = _backend(tmp_path, trainer=reject_invalid)
    feedback = _feedback(wrapper)

    class SelectedReward:
        def __init__(self, value):
            self.value = value
        def __float__(self):
            assert (tmp_path / "guidance/round-003-feedback-selection.json").exists()
            return self.value

    feedback = replace(feedback, cases=tuple(replace(item,
        primary_error=SelectedReward(float("nan") if item.case.case_id == "base-0000" else
                                     float("inf") if item.case.case_id == "base-0064" else
                                     float(item.primary_error))) for item in feedback.cases))
    with pytest.raises(FloatingPointError, match="existing trainer"):
        _fit(wrapper, feedback)
    assert len(calls) == 1 and not wrapper.training_audits
    report = json.loads((tmp_path / "guidance/round-003-feedback-selection.json").read_text())
    assert report["selected"][64]["case_id"] == "base-0064"
    assert not (tmp_path / "guidance/round-003").exists()


def test_legacy_default_never_silently_truncates_extra_clean_rows(tmp_path):
    def legacy_trainer(**kwargs):
        if len(kwargs["rows"]) != 80:
            raise ValueError("exactly 80 actual clean BASE feedback roots required; no silent filtering")
        return SimpleNamespace(guide=object(), checkpoint_path="/private/mock.pt",
                               checkpoint_sha256="f" * 64, report={"mock_only": True})

    wrapper, calls = _backend(tmp_path, policy=integration.STRICT_FEEDBACK_SELECTION, trainer=legacy_trainer)
    feedback = _feedback(wrapper, base_count=81, guided_count=0, perturbed_count=0)
    with pytest.raises(ValueError, match="exactly 80"):
        _fit(wrapper, feedback)
    assert len(calls[0]["rows"]) == 81
    assert not (tmp_path / "guidance").exists()
    assert inspect.signature(integration.VPFPTrainingGuidanceBackend).parameters["feedback_selection"].default == integration.STRICT_FEEDBACK_SELECTION
    assert inspect.signature(integration.build_vpfp_strong_iteration_binding).parameters["feedback_selection"].default == integration.STRICT_FEEDBACK_SELECTION


def test_formal_initial_342_base_feedback_uses_same_rule(tmp_path):
    wrapper, calls = _backend(tmp_path)
    feedback = _feedback(wrapper, guided_count=0, perturbed_count=0)
    _fit(wrapper, replace(feedback, round_index=0))
    assert len(calls[0]["rows"]) == 80
    assert calls[0]["source_round_index"] == 0 and calls[0]["target_round_index"] == 1
    assert (tmp_path / "guidance/round-001-feedback-selection.json").exists()


def test_unknown_or_mutated_policy_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="feedback_selection"):
        _backend(tmp_path, policy="highest_reward_80")
    wrapper, calls = _backend(tmp_path)
    feedback = _feedback(wrapper)
    wrapper.feedback_selection = integration.STRICT_FEEDBACK_SELECTION
    with pytest.raises(ValueError, match="recipe changed"):
        _fit(wrapper, feedback)
    with pytest.raises(ValueError, match="recipe changed"):
        wrapper.generate({}, None, None)
    assert not calls


def test_factory_binds_selection_policy_into_frozen_identity(monkeypatch, tmp_path):
    from pde_agent_verifier_solver.plugins.vpfp import portable_initial_conditioned as portable
    import torch

    class Frozen:
        def eval(self):
            return self
        def requires_grad_(self, value):
            assert value is False
            return self

    @dataclass(frozen=True)
    class Plugin:
        generator: object

    @dataclass(frozen=True)
    class Binding:
        plugin: object
        checkpoint: object
        report: dict

    inner = SimpleNamespace(runtime=object(), public_session_methods=(),
                            development_solution_session_methods=())
    base = SimpleNamespace(identity={"collision_nu_range": [10., 30.]},
                           model=Frozen(), normalizer=Frozen())
    monkeypatch.setattr(portable, "load_portable_initial_conditioned_sampler", lambda *args, **kwargs: base)
    monkeypatch.setattr(integration, "build_vpfp_online_binding", lambda *args, **kwargs:
        Binding(Plugin(SimpleNamespace(backend=inner)), None, {"binding_sha256": "b" * 64}))
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *args, **kwargs: None)
    kwargs = dict(native_spec={}, base_checkpoint="mock-base", base_checkpoint_sha256="c" * 64,
        portable_codec="mock-codec", portable_codec_sha256="d" * 64,
        output_root=tmp_path, device="cpu")
    legacy = integration.build_vpfp_strong_iteration_binding(**kwargs)
    formal = integration.build_vpfp_strong_iteration_binding(
        **kwargs, feedback_selection=integration.FORMAL_FEEDBACK_SELECTION)
    assert formal.report["feedback_selection"] == integration.FORMAL_FEEDBACK_SELECTION
    assert legacy.report["feedback_selection"] == integration.STRICT_FEEDBACK_SELECTION
    assert formal.report["binding_sha256"] != legacy.report["binding_sha256"]
    assert formal.plugin.generator.backend.feedback_selection == integration.FORMAL_FEEDBACK_SELECTION
    identity = dict(formal.report)
    digest = identity.pop("binding_sha256")
    assert digest == sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    assert formal.checkpoint.payload == formal.report
