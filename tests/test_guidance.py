from dataclasses import fields, replace
import hashlib
import unittest

import pde_agent_verifier_solver.sdk.guidance as guidance_module
from pde_agent_verifier_solver.sdk.guidance import (
    CohortMetrics,
    FrozenGuidanceSnapshot,
    GuidanceCapability,
    GuidanceCohortKind,
    GuidanceFailureAudit,
    GuidanceTrainingBudget,
    GuidanceTrainingRequest,
    GuidanceTrainingBackend,
    GuidanceUnsupported,
    GuidedAnswerFirstSample,
    GuidedSampleBatch,
    GuidedSampleRequest,
    GuidedSampleSpec,
    GuidedSamplingPurpose,
    OfficialCommonGuidedCohort,
    SeparatedGuidanceMetrics,
    require_guidance_capability,
    sample_frozen_guidance,
)
from pde_agent_verifier_solver.sdk.models import (
    Case,
    CaseSource,
    CaseVisibility,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def training_request(**changes) -> GuidanceTrainingRequest:
    values = {
        "source_round_index": 0,
        "target_round_index": 1,
        "base_generator_sha256": digest("base-generator"),
        "target_solver_sha256": digest("cold-start-incumbent"),
        "reward_matcher_definition_sha256": digest("reward-matcher"),
        "reward_feedback_uri": "sealed://incumbent-reward-feedback",
        "reward_feedback_sha256": digest("incumbent-reward-feedback"),
        "guidance_config_sha256": digest("guidance-config"),
        "controller_training_seed": 17,
        "training_budget": GuidanceTrainingBudget(
            120.0,
            "one-a100",
            1,
        ),
    }
    values.update(changes)
    return GuidanceTrainingRequest(**values)


def snapshot(
    request: GuidanceTrainingRequest | None = None,
    *,
    artifact_label: str = "guided-model",
    wall_seconds: float = 30.0,
) -> FrozenGuidanceSnapshot:
    return FrozenGuidanceSnapshot(
        training_request=request or training_request(),
        backend_id="guidance-backend-v1",
        guided_artifact_uri=f"sealed://{artifact_label}",
        guided_artifact_sha256=digest(artifact_label),
        campaign_guidance_training_wall_seconds=wall_seconds,
    )


class SequenceClock:
    def __init__(self, *values: float) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


class FakeFrozenOracle:
    guidance_capability = GuidanceCapability.online("guidance-backend-v1")

    def __init__(self, frozen: FrozenGuidanceSnapshot) -> None:
        self.snapshot = frozen
        self.calls = 0

    def generate_answer_first(
        self,
        request: GuidedSampleRequest,
    ) -> tuple[GuidedAnswerFirstSample, ...]:
        self.calls += 1
        result = []
        for index in range(request.spec.sample_count):
            label = f"{request.controller_nonce}-{request.controller_seed}-{index}"
            visibility = request.expected_visibility
            result.append(
                GuidedAnswerFirstSample(
                    sample_id=f"sample-{label}",
                    snapshot_sha256=request.snapshot_sha256,
                    case=Case(
                        case_id=f"case-{label}",
                        source=CaseSource.GENERATED,
                        visibility=visibility,
                        public_data={
                            "answer_first": True,
                            "setting": request.spec.settings.get("focus"),
                        },
                        round_index=(
                            self.snapshot.training_request.target_round_index
                        ),
                    ),
                    reference_answer_sha256=digest(f"answer-{label}"),
                    public_case_sha256=digest(f"case-{label}"),
                    answer_to_case_derivation_sha256=digest(
                        f"derivation-{label}"
                    ),
                    answer_access_locator=(
                        f"oracle://answer/{label}"
                        if visibility is CaseVisibility.DEVELOPMENT
                        else None
                    ),
                )
            )
        return tuple(result)


def draw_request(
    frozen: FrozenGuidanceSnapshot,
    request_id: str,
    controller_seed: int,
    *,
    purpose: GuidedSamplingPurpose = GuidedSamplingPurpose.DEVELOPMENT_ONLINE,
    count: int = 2,
) -> GuidedSampleRequest:
    return GuidedSampleRequest(
        spec=GuidedSampleSpec(count, {"focus": "hard-tail", "temperature": 0.7}),
        request_id=request_id,
        purpose=purpose,
        snapshot_sha256=frozen.snapshot_sha256,
        controller_seed=controller_seed,
        controller_nonce=f"controller-nonce-{request_id}",
        inference_wall_budget_seconds=5.0,
        compute_capacity="shared-gpu-slot",
    )


class GuidanceTests(unittest.TestCase):
    def test_snapshot_binds_training_contract_and_all_required_digests(self) -> None:
        request = training_request()
        frozen = snapshot(request)

        self.assertEqual(request.source_round_index, 0)
        self.assertEqual(request.target_round_index, 1)
        self.assertEqual(
            request.previous_incumbent_sha256,
            request.target_solver_sha256,
        )
        self.assertEqual(
            FrozenGuidanceSnapshot.training_attempts_per_boundary,
            1,
        )
        self.assertEqual(
            GuidanceTrainingRequest.cadence,
            "once_at_round_boundary",
        )

        variants = (
            replace(request, base_generator_sha256=digest("other-base")),
            replace(request, target_solver_sha256=digest("other-solver")),
            replace(
                request,
                reward_matcher_definition_sha256=digest("other-reward"),
            ),
            replace(
                request,
                reward_feedback_sha256=digest("other-reward-feedback"),
            ),
            replace(request, guidance_config_sha256=digest("other-config")),
            replace(request, controller_training_seed=18),
            replace(
                request,
                training_budget=GuidanceTrainingBudget(
                    121.0,
                    "one-a100",
                    1,
                ),
            ),
        )
        identities = {frozen.snapshot_sha256}
        identities.update(snapshot(item).snapshot_sha256 for item in variants)
        identities.add(snapshot(request, artifact_label="other-model").snapshot_sha256)
        self.assertEqual(len(identities), 9)

        self.assertEqual(
            frozen.training_request.base_generator_sha256,
            request.base_generator_sha256,
        )
        self.assertEqual(
            frozen.training_request.target_solver_sha256,
            request.target_solver_sha256,
        )
        self.assertEqual(
            frozen.guided_artifact_sha256,
            digest("guided-model"),
        )
        with self.assertRaisesRegex(ValueError, "campaign wall budget"):
            snapshot(request, wall_seconds=121.0)

    def test_capability_is_optional_and_fails_closed(self) -> None:
        unsupported = GuidanceCapability.unsupported(
            "backend has no audited frozen guidance",
            backend_id="base-only",
        )
        with self.assertRaisesRegex(GuidanceUnsupported, "no audited"):
            require_guidance_capability(unsupported)
        with self.assertRaisesRegex(
            GuidanceUnsupported,
            "GuidanceCapability",
        ):
            require_guidance_capability(True)

        capability = GuidanceCapability.online("guidance-backend-v1")
        self.assertIs(require_guidance_capability(capability), capability)
        self.assertTrue(capability.available)
        self.assertIsNone(capability.protocol_request_limit)
        self.assertIsNone(capability.protocol_sample_limit)

        class ThinBackend:
            guidance_capability = capability

            def train_guidance(self, request):
                return snapshot(request)

            def open_guidance(self, frozen):
                return FakeFrozenOracle(frozen)

        backend: GuidanceTrainingBackend = ThinBackend()
        trained = backend.train_guidance(training_request())
        self.assertIsInstance(
            backend.open_guidance(trained),
            FakeFrozenOracle,
        )

    def test_agent_selects_count_settings_but_controller_assigns_fresh_draws(self) -> None:
        frozen = snapshot()
        oracle = FakeFrozenOracle(frozen)
        spec_fields = {item.name for item in fields(GuidedSampleSpec)}
        self.assertEqual(
            spec_fields,
            {"sample_count", "settings", "settings_sha256"},
        )
        self.assertNotIn("seed", spec_fields)
        self.assertNotIn("nonce", spec_fields)

        batches = []
        for index in range(6):
            batches.append(
                sample_frozen_guidance(
                    oracle,
                    draw_request(
                        frozen,
                        f"development-{index}",
                        controller_seed=1000 + index,
                    ),
                    clock=SequenceClock(
                        float(index),
                        float(index) + 0.2,
                    ),
                )
            )

        sample_ids = [
            sample.sample_id
            for batch in batches
            for sample in batch.samples
        ]
        self.assertEqual(oracle.calls, 6)
        self.assertEqual(len(sample_ids), len(set(sample_ids)))
        self.assertTrue(
            all(
                sample.answer_access_locator is not None
                and sample.case.visibility is CaseVisibility.DEVELOPMENT
                for batch in batches
                for sample in batch.samples
            )
        )
        self.assertAlmostEqual(
            batches[0].campaign_guided_generation_wall_seconds,
            0.2,
        )
        self.assertIsNone(GuidedSampleRequest.protocol_request_limit)
        self.assertIsNone(GuidedSampleRequest.protocol_sample_limit)
        self.assertFalse(hasattr(guidance_module, "GuidedReplayBank"))

    def test_official_evaluation_reuses_one_common_hidden_draw(self) -> None:
        frozen = snapshot()
        batch = sample_frozen_guidance(
            FakeFrozenOracle(frozen),
            draw_request(
                frozen,
                "official-common",
                909,
                purpose=GuidedSamplingPurpose.OFFICIAL_COMMON_HIDDEN,
                count=4,
            ),
            clock=SequenceClock(10.0, 10.5),
        )
        cohort = OfficialCommonGuidedCohort("guided-hidden-round-1", batch)

        first = cohort.samples_for_candidate("candidate-a")
        second = cohort.samples_for_candidate("candidate-b")

        self.assertIs(first, second)
        self.assertTrue(
            all(
                sample.case.visibility is CaseVisibility.HIDDEN
                and sample.answer_access_locator is None
                for sample in first
            )
        )
        leaked = replace(first[0], answer_access_locator="oracle://hidden-leak")
        with self.assertRaisesRegex(ValueError, "cannot expose answer access"):
            GuidedSampleBatch(
                batch.request,
                (leaked, *first[1:]),
                batch.campaign_guided_generation_wall_seconds,
            )
        self.assertNotIn(
            "candidate_id",
            {item.name for item in fields(GuidedSampleRequest)},
        )

    def test_base_and_guided_metrics_stay_separate_without_default_scalar(self) -> None:
        base = CohortMetrics(
            "average-base",
            GuidanceCohortKind.AVERAGE_BASE,
            digest("base-generator"),
            {"relative_error": 0.2},
        )
        guided = CohortMetrics(
            "guided-adversarial",
            GuidanceCohortKind.GUIDED_ADVERSARIAL,
            snapshot().snapshot_sha256,
            {"relative_error": 0.6},
        )
        report = SeparatedGuidanceMetrics(base, guided)

        self.assertNotIn(
            "scalar",
            {item.name for item in fields(SeparatedGuidanceMetrics)},
        )
        self.assertAlmostEqual(
            report.explicitly_weighted_metric(
                "relative_error",
                average_base_weight=3.0,
                guided_adversarial_weight=1.0,
            ),
            0.3,
        )
        with self.assertRaises(TypeError):
            report.average_base.values["relative_error"] = 1.0

    def test_failure_audit_is_exact_and_guidance_times_are_campaign_only(self) -> None:
        exact_reason = "CUDA OOM at sample 7\nworker=gpu-3"
        audit = GuidanceFailureAudit(
            source_round_index=0,
            target_round_index=1,
            stage="online_inference",
            exact_reason=exact_reason,
            artifact_uri="sealed://failure-log",
            artifact_sha256=digest("failure-log"),
            campaign_wall_seconds=4.25,
        )

        self.assertEqual(audit.exact_reason, exact_reason)
        self.assertEqual(audit.artifact_sha256, digest("failure-log"))
        frozen = snapshot()
        batch = sample_frozen_guidance(
            FakeFrozenOracle(frozen),
            draw_request(frozen, "timing", 123),
            clock=SequenceClock(20.0, 20.3),
        )
        timing_fields = {
            item.name for item in fields(FrozenGuidanceSnapshot)
        } | {item.name for item in fields(type(batch))}
        self.assertIn(
            "campaign_guidance_training_wall_seconds",
            timing_fields,
        )
        self.assertIn(
            "campaign_guided_generation_wall_seconds",
            timing_fields,
        )
        self.assertFalse(
            any(name.startswith("candidate_") for name in timing_fields)
        )


if __name__ == "__main__":
    unittest.main()
