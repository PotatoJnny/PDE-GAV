"""Selection guidance must change fresh samples without breaking ownership."""

from dataclasses import replace
from random import Random

import numpy as np
import pytest

from pde_agent_verifier_solver.plugins.reward_guidance import RewardGuidanceConfig, RewardGuidedBackend, RewardGuidedGenerator
from pde_agent_verifier_solver.plugins.coefficient_flow import BackendGeneratorProgram
from pde_agent_verifier_solver.sdk import (
    Artifact, Case, CaseFeedback, CaseRecord, CaseSource, CaseVisibility,
    GenerationKind, GenerationPurpose, GenerationRequest, GeneratorMode,
    RoundFeedback, TaskKind,
    GeneratorCapabilities, GeneratorCheckpoint, ProgramInfo,
)
from pde_agent_verifier_solver.sdk.guidance import GuidanceUnsupported


class _Truth:
    def materialize(self, _query):
        raise AssertionError("selection guidance must not inspect truth")


class _Backend:
    supports_perturbations = True
    public_session_methods = ("source",)
    development_solution_session_methods = ()

    def __init__(self):
        self.live = {}
        self.released = []
        self.requests = []

    def generate(self, _base, guide, request, progress=None):
        assert guide is None
        self.requests.append(request)
        rng = Random(request.seed)
        records = []
        for index in range(request.count):
            anchor = request.anchors[index] if request.anchors else None
            if anchor is not None:
                assert self.live[anchor.case_id] is anchor
            case = Case(
                f"{request.kind.value}-{request.seed}-{index}",
                CaseSource.PERTURBED if anchor else CaseSource.GENERATED,
                request.visibility,
                {"condition": rng.random()},
                request.round_index,
                anchor.case_id if anchor else None,
            )
            assert case.case_id not in self.live
            self.live[case.case_id] = case
            records.append(CaseRecord(case, _Truth()))
        return tuple(records)

    def release_cases(self, case_ids):
        self.released.extend(case_ids)
        for case_id in case_ids:
            self.live.pop(case_id, None)

    def public_session_factory(self, case):
        assert self.live[case.case_id] is case
        return case.public_data


def _feedback():
    candidate = Artifact("solver-r1", TaskKind.OBSERVABLE_MODEL, "S1", object(), "", "", 1, "agent")
    cases = tuple(
        CaseFeedback(Case(f"feedback-{index}", CaseSource.GENERATED, CaseVisibility.HIDDEN, {"condition": index / 20}, 1), float(np.expm1(index / 20)), {})
        for index in range(21)
    )
    return RoundFeedback(1, candidate, cases)


def _wrapped(**options):
    backend = _Backend()
    wrapper = RewardGuidedBackend(
        backend,
        feature_extractor=lambda case: (case.public_data["condition"],),
        feature_id="public-condition-v1",
        base_generator_sha256="a" * 64,
        config=RewardGuidanceConfig(**options),
    )
    guide = wrapper.fit_guidance(None, None, _feedback(), {})
    return backend, wrapper, guide


def _request(kind=GenerationKind.GUIDED, **changes):
    request = GenerationRequest(2, 100, GeneratorMode.ONLINE, kind, CaseVisibility.HIDDEN, 23, GenerationPurpose.SELECTION, settings={"fixed_generator_sha256": "a" * 64})
    return replace(request, **changes)


def test_reward_feedback_changes_distribution_and_releases_every_discard():
    backend, wrapper, guide = _wrapped()
    records = wrapper.generate(None, guide, _request())
    assert len(records) == 100
    assert np.mean([row.case.public_data["condition"] for row in records]) > 0.65
    assert len(backend.released) == 300
    assert set(backend.live) == {row.case.case_id for row in records}
    assert all(set(row.case.public_data) == {"condition"} for row in records)
    assert wrapper.last_generation_report["is_diffusion_or_flow_guidance"] is False
    assert wrapper.last_generation_report["physical_guidance_qualification"] == "required"


def test_frozen_distribution_replays_but_distinct_controller_seed_draws_fresh_roots():
    _, first, guide = _wrapped()
    _, second, replay_guide = _wrapped()
    original = first.generate(None, guide, _request())
    replay = second.generate(None, replay_guide, _request())
    assert guide.sha256 == replay_guide.sha256
    assert [row.case.case_id for row in original] == [row.case.case_id for row in replay]
    new = first.generate(None, guide, _request(seed=24))
    assert not {row.case.case_id for row in original} & {row.case.case_id for row in new}


def test_guided_anchor_retains_original_case_and_public_session_identity():
    backend, wrapper, guide = _wrapped()
    root = wrapper.generate(None, guide, _request(count=1))[0]
    assert backend.live[root.case.case_id] is root.case
    assert wrapper.public_session_factory(root.case) == root.case.public_data
    child = wrapper.generate(None, guide, _request(GenerationKind.PERTURBED, count=1, anchors=(root.case,)))[0]
    assert child.case.parent_case_id == root.case.case_id
    wrapper.release_cases((root.case.case_id, child.case.case_id))
    assert not backend.live


def test_base_requests_remain_unguided():
    backend, wrapper, guide = _wrapped()
    base_request = _request(GenerationKind.BASE, count=3)
    records = wrapper.generate(None, guide, base_request)
    assert backend.requests == [base_request]
    assert len(records) == 3
    assert not backend.released


def test_feature_failure_releases_full_proposal_pool():
    backend, wrapper, guide = _wrapped()
    wrapper.feature_extractor = lambda _case: (float("nan"),)
    with pytest.raises(ValueError, match="finite vector"):
        wrapper.generate(None, guide, _request(count=2))
    assert not backend.live
    assert len(backend.released) == 8


def test_snapshot_binding_and_round_boundary_are_enforced():
    _, wrapper, guide = _wrapped()
    with pytest.raises(ValueError, match="before its target"):
        wrapper.generate(None, guide, _request(round_index=1))
    with pytest.raises(ValueError, match="identity differs"):
        wrapper.generate(None, guide, _request(settings={"fixed_generator_sha256": "b" * 64}))
    with pytest.raises(ValueError, match="config changed"):
        wrapper.generate(None, replace(guide, feature_id="different"), _request())
    with pytest.raises(GuidanceUnsupported, match="frozen reward"):
        wrapper.generate(None, None, _request())


def test_degenerate_feedback_does_not_claim_functional_guidance():
    _, wrapper, _ = _wrapped()
    feedback = _feedback()
    constant = replace(feedback, cases=tuple(replace(row, primary_error=1.0) for row in feedback.cases))
    with pytest.raises(GuidanceUnsupported, match="variation"):
        wrapper.fit_guidance(None, None, constant, {})
    nonfinite = replace(feedback, cases=(replace(feedback.cases[0], primary_error=float("inf")),) + feedback.cases[1:])
    with pytest.raises(ValueError, match="finite and nonnegative"):
        wrapper.fit_guidance(None, None, nonfinite, {})


def test_generator_wrapper_preserves_custom_frozen_checkpoint_and_provenance_guards():
    class BoundProgram(BackendGeneratorProgram):
        info = ProgramInfo("bound-test", "Test", "Test")
        capabilities = GeneratorCapabilities(guidance=False, perturbations=True)

        def load(self, checkpoint, settings, progress=None):
            if checkpoint.payload != "verified-checkpoint":
                raise ValueError("physical checkpoint changed")
            return super().load(checkpoint, settings, progress)

        def sample(self, state, request, progress=None):
            request = replace(request, settings={**request.settings, "bound_provenance": "preserved"})
            return super().sample(state, request, progress)

    backend = _Backend()
    original = BoundProgram(backend)
    wrapped = RewardGuidedGenerator(original, feature_extractor=lambda case: (case.public_data["condition"],), feature_id="public-condition-v1", base_generator_sha256="a" * 64)
    assert original.backend is backend
    assert wrapped.capabilities.guidance
    with pytest.raises(ValueError, match="physical checkpoint changed"):
        wrapped.load(GeneratorCheckpoint(original.info.program_id, "different"), {})
    state = wrapped.load(GeneratorCheckpoint(original.info.program_id, "verified-checkpoint"), {})
    records = wrapped.sample(state, _request(GenerationKind.BASE, count=1))
    assert backend.requests[0].settings["bound_provenance"] == "preserved"
    wrapped.release(state, tuple(row.case.case_id for row in records))
    assert not backend.live
