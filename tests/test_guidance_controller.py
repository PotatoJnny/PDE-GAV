from dataclasses import fields
import hashlib
import itertools
import unittest

from pde_agent_verifier_solver.engine.guidance_controller import (
    GuidanceLifecycleController,
    GuidanceProtocolConfig,
    GuidanceRewardArtifact,
    GuidanceRewardFeedback,
    GuidanceRewardRow,
)
from pde_agent_verifier_solver.sdk.guidance import (
    CandidateGuidanceSnapshot,
    CandidateGuidedSample,
    CandidateGuidedSampleBatch,
    FrozenGuidanceSnapshot,
    GuidanceCapability,
    GuidanceTrainingBudget,
    GuidedAnswerFirstSample,
    GuidedSampleSpec,
)
from pde_agent_verifier_solver.sdk.models import (
    AnswerQuery,
    AnswerView,
    Case,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    QueryKind,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _Truth:
    def answer(self, query: AnswerQuery) -> AnswerView:
        return AnswerView(
            kind=query.kind,
            fields={name: (0.0,) for name in query.fields},
            points=query.points,
            shape=(1,),
        )



class _SolutionSession:
    def __init__(self, sample_id):
        self.sample_id = sample_id

    def query(self, *, points):
        return {"points": tuple(points)}

class _Oracle:
    guidance_capability = GuidanceCapability.online("fake-guidance-v1")
    development_solution_methods = ("query",)

    def __init__(self, backend, snapshot) -> None:
        self.backend = backend
        self.snapshot = snapshot
        self.records = {}
        self.closed = False
        self.close_calls = 0
        self.opened_sample_ids = []

    def generate_answer_first(self, request):
        self.backend.generation_calls.append(request)
        rows = []
        for index in range(request.spec.sample_count):
            label = (
                f"{request.purpose.value}-{request.controller_seed}-"
                f"{request.controller_nonce}-{index}"
            )
            case = Case(
                case_id=f"guided-{label}",
                source=CaseSource.GENERATED,
                visibility=request.expected_visibility,
                public_data={
                    "answer_first": True,
                    "mode": "guided-physics",
                    "mutable_leaf": bytearray(b"abc"),
                    "distribution_lineage": {
                        "kind": "frozen_reward_guided",
                        "snapshot_sha256": self.snapshot.snapshot_sha256,
                        "base_generator_sha256": digest("base-generator"),
                    },
                },
                round_index=self.snapshot.training_request.target_round_index,
            )
            sample = GuidedAnswerFirstSample(
                sample_id=f"sample-{label}",
                snapshot_sha256=request.snapshot_sha256,
                case=case,
                reference_answer_sha256=digest(f"answer-{label}"),
                public_case_sha256=digest(f"case-{label}"),
                answer_to_case_derivation_sha256=digest("fake-answer-to-case-v1"),
                answer_access_locator=(
                    f"solution-oracle://{label}"
                    if request.expected_visibility is CaseVisibility.DEVELOPMENT
                    else None
                ),
            )
            rows.append(sample)
            self.records[sample.sample_id] = CaseRecord(case, _Truth())
        return tuple(rows)

    def resolve_case_records(self, batch):
        self.backend.resolve_calls += 1
        return tuple(self.records[row.sample_id] for row in batch.samples)

    def open_development_solution_session(self, sample_id):
        if sample_id not in self.records:
            raise KeyError(sample_id)
        self.opened_sample_ids.append(sample_id)
        return _SolutionSession(sample_id)

    def close(self):
        self.close_calls += 1
        self.closed = True
        self.records.clear()


class _Backend:
    guidance_capability = GuidanceCapability.online("fake-guidance-v1")

    def __init__(self) -> None:
        self.feedback = []
        self.training_requests = []
        self.generation_calls = []
        self.resolve_calls = 0

    def seal_reward_feedback(self, feedback):
        self.feedback.append(feedback)
        label = (
            f"reward-{feedback.source_round_index}-"
            f"{feedback.target_solver_sha256}"
        )
        return GuidanceRewardArtifact(
            f"sealed://{label}",
            digest(label),
        )

    def train_guidance(self, request):
        self.training_requests.append(request)
        label = (
            f"guide-{request.source_round_index}-"
            f"{request.target_solver_sha256}"
        )
        return FrozenGuidanceSnapshot(
            training_request=request,
            backend_id="fake-guidance-v1",
            guided_artifact_uri=f"sealed://{label}",
            guided_artifact_sha256=digest(label),
            campaign_guidance_training_wall_seconds=2.0,
        )

    def open_guidance(self, snapshot):
        return _Oracle(self, snapshot)


def config() -> GuidanceProtocolConfig:
    return GuidanceProtocolConfig(
        training_budget=GuidanceTrainingBudget(
            maximum_wall_seconds=20.0,
            compute_capacity="one-test-gpu",
        ),
        guidance_config_sha256=digest("guidance-config"),
        reward_feedback_case_count=2,
        official_round_case_count=3,
        official_terminal_case_count=4,
        inference_wall_budget_seconds=5.0,
        inference_compute_capacity="one-inference-slot",
        official_sample_settings={"temperature": 0.7},
    )


def feedback(source_round: int, solver: str) -> GuidanceRewardFeedback:
    target_round = source_round + 1
    return GuidanceRewardFeedback(
        source_round_index=source_round,
        target_round_index=target_round,
        target_solver_sha256=digest(solver),
        base_generator_sha256=digest("base-generator"),
        reward_matcher_definition_sha256=digest("macro-error-reward"),
        rows=tuple(
            GuidanceRewardRow(
                Case(
                    case_id=f"feedback-{source_round}-{index}",
                    source=CaseSource.GENERATED,
                    visibility=CaseVisibility.HIDDEN,
                    public_data={"question": index},
                    round_index=source_round,
                ),
                primary_error=0.2 + index,
                status="valid",
                metrics={"relative_error": 0.2 + index},
            )
            for index in range(2)
        ),
    )


class GuidanceControllerTests(unittest.TestCase):
    def controller(self, backend):
        seeds = iter(range(100, 1000))
        nonces = (f"nonce-{index}" for index in itertools.count())
        return GuidanceLifecycleController(
            config=config(),
            backend=backend,
            entropy64=lambda: next(seeds),
            nonce_factory=lambda: next(nonces),
        )

    def test_capability_is_optional_but_enabled_lifecycle_fails_closed(self):
        class Unsupported:
            guidance_capability = GuidanceCapability.unsupported(
                "equation backend has no frozen guide"
            )

        with self.assertRaisesRegex(
            RuntimeError,
            "equation backend has no frozen guide",
        ):
            self.controller(Unsupported())

        class Undeclared:
            guidance_capability = True

        with self.assertRaisesRegex(RuntimeError, "GuidanceCapability"):
            self.controller(Undeclared())

    def test_train_once_binds_prior_solver_and_exact_reward_artifact(self):
        backend = _Backend()
        controller = self.controller(backend)
        prior = feedback(0, "cold-start")

        boundary = controller.train_boundary(
            prior,
            controller_training_seed=41,
        )
        request = backend.training_requests[0]

        self.assertEqual(len(backend.feedback), 1)
        self.assertEqual(len(backend.training_requests), 1)
        self.assertEqual(request.source_round_index, 0)
        self.assertEqual(request.target_round_index, 1)
        self.assertEqual(request.target_solver_sha256, digest("cold-start"))
        self.assertEqual(request.reward_feedback_uri, backend.feedback and (
            f"sealed://reward-0-{digest('cold-start')}"
        ))
        self.assertEqual(
            request.reward_feedback_sha256,
            digest(f"reward-0-{digest('cold-start')}"),
        )
        self.assertEqual(
            boundary.snapshot.training_request,
            request,
        )
        with self.assertRaisesRegex(RuntimeError, "already attempted"):
            controller.train_boundary(prior, controller_training_seed=42)
        self.assertEqual(len(backend.training_requests), 1)

    def test_development_oracle_is_fresh_unlimited_and_has_no_truth_resolver(self):
        backend = _Backend()
        boundary = self.controller(backend).train_boundary(
            feedback(0, "cold-start"),
            controller_training_seed=41,
        )
        oracle = boundary.development_oracle("agent-a")

        batches = tuple(
            oracle.sample(GuidedSampleSpec(2, {"focus": "hard-tail"}))
            for _index in range(25)
        )
        ids = tuple(
            sample.sample_id
            for batch in batches
            for sample in batch.samples
        )
        first_id = batches[0].samples[0].sample_id
        solution = oracle.query_completed_solution(
            first_id, "query", points=({"t": 0.1, "x": 0.2},)
        )

        self.assertEqual(oracle.call_count, 25)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIsNone(oracle.protocol_request_limit)
        self.assertIsNone(oracle.protocol_sample_limit)
        self.assertFalse(hasattr(oracle, "resolve_case_records"))
        self.assertFalse(hasattr(oracle, "train_guidance"))
        self.assertEqual(oracle.available_solution_methods, ("query",))
        self.assertIsInstance(oracle.snapshot, CandidateGuidanceSnapshot)
        self.assertEqual(
            {field.name for field in fields(oracle.snapshot)},
            {
                "snapshot_sha256",
                "campaign_guidance_training_wall_seconds",
            },
        )
        self.assertTrue(
            all(
                isinstance(batch, CandidateGuidedSampleBatch)
                and {
                    field.name for field in fields(batch)
                }
                == {
                    "snapshot_sha256",
                    "sample_settings_commitment_sha256",
                    "samples",
                    "campaign_guided_generation_wall_seconds",
                }
                for batch in batches
            )
        )
        self.assertTrue(
            all(
                isinstance(sample, CandidateGuidedSample)
                and {
                    field.name for field in fields(sample)
                }
                == {
                    "sample_id",
                    "case",
                    "answer_to_case_derivation_commitment_sha256",
                }
                and sample.sample_id.startswith("guided-sample-")
                for batch in batches
                for sample in batch.samples
            )
        )
        self.assertNotEqual(oracle._oracle.opened_sample_ids[0], first_id)
        self.assertNotIn(oracle._oracle.opened_sample_ids[0], first_id)
        self.assertEqual(solution["points"], ({"t": 0.1, "x": 0.2},))
        self.assertTrue(all(
            sample.case.source is CaseSource.GENERATED
            and sample.case.visibility is CaseVisibility.DEVELOPMENT
            and sample.case.round_index is None
            and sample.case.parent_case_id is None
            and sample.case.tags == ()
            and sample.case.case_id.startswith("candidate-case-")
            and "distribution_lineage" not in sample.case.public_data
            for batch in batches
            for sample in batch.samples
        ))
        self.assertTrue(all(
            "distribution_lineage" in record.case.public_data
            for record in oracle._oracle.records.values()
        ))
        with self.assertRaises(PermissionError):
            oracle.query_completed_solution(first_id, "_private")
        self.assertTrue(all(
            not hasattr(sample, "reference_answer_sha256")
            and not hasattr(sample, "public_case_sha256")
            and not hasattr(sample, "answer_access_locator")
            for batch in batches
            for sample in batch.samples
        ))
        self.assertTrue(all(
            not hasattr(batch, "request")
            and not hasattr(batch, "request_id")
            for batch in batches
        ))
        public_leaf = batches[0].samples[0].case.public_data["mutable_leaf"]
        raw_leaf = next(iter(oracle._oracle.records.values())).case.public_data[
            "mutable_leaf"
        ]
        public_leaf[0] = ord("z")
        self.assertEqual(raw_leaf, bytearray(b"abc"))

    def test_development_oracle_close_revokes_and_releases_samples(self):
        backend = _Backend()
        boundary = self.controller(backend).train_boundary(
            feedback(0, "cold-start"),
            controller_training_seed=41,
        )
        oracle = boundary.development_oracle("agent-a")
        batch = oracle.sample(GuidedSampleSpec(2))
        underlying = oracle._oracle
        self.assertEqual(len(underlying.records), 2)

        oracle.close()
        oracle.close()

        self.assertTrue(oracle.closed)
        self.assertTrue(underlying.closed)
        self.assertEqual(underlying.close_calls, 1)
        self.assertEqual(underlying.records, {})
        with self.assertRaisesRegex(RuntimeError, "closed"):
            oracle.sample(GuidedSampleSpec(1))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            oracle.query_completed_solution(
                batch.samples[0].sample_id,
                "query",
                points=({"t": 0.0},),
            )

    def test_official_draw_is_one_common_hidden_cohort_not_score_lottery(self):
        backend = _Backend()
        boundary = self.controller(backend).train_boundary(
            feedback(0, "cold-start"),
            controller_training_seed=41,
        )

        first = boundary.official_common_cohort(
            "round-1-guided-hidden",
            sample_count=config().official_round_case_count,
        )
        second = boundary.official_common_cohort(
            "round-1-guided-hidden",
            sample_count=config().official_round_case_count,
        )

        self.assertIs(first, second)
        self.assertEqual(backend.resolve_calls, 1)
        official_calls = [
            row for row in backend.generation_calls
            if row.purpose.value == "official_common_hidden"
        ]
        self.assertEqual(len(official_calls), 1)
        self.assertTrue(all(
            sample.case.visibility is CaseVisibility.HIDDEN
            and sample.answer_access_locator is None
            for sample in first.public.samples_for_candidate("candidate-a")
        ))
        self.assertIs(
            first.public.samples_for_candidate("candidate-a"),
            first.public.samples_for_candidate("candidate-b"),
        )
        with self.assertRaisesRegex(RuntimeError, "cannot be resampled"):
            boundary.official_common_cohort(
                "round-1-guided-hidden",
                sample_count=4,
            )

    def test_terminal_guide_targets_final_solver_as_separate_boundary(self):
        backend = _Backend()
        controller = self.controller(backend)
        round_one = controller.train_boundary(
            feedback(0, "cold-start"),
            controller_training_seed=1,
        )
        terminal = controller.train_boundary(
            feedback(1, "final-solver"),
            controller_training_seed=2,
        )
        terminal_cohort = terminal.official_common_cohort(
            "terminal-final-solver-guided",
            sample_count=config().official_terminal_case_count,
        )

        self.assertEqual(len(backend.training_requests), 2)
        self.assertNotEqual(
            round_one.snapshot.snapshot_sha256,
            terminal.snapshot.snapshot_sha256,
        )
        self.assertEqual(
            terminal.snapshot.training_request.target_solver_sha256,
            digest("final-solver"),
        )
        self.assertEqual(len(terminal_cohort.records), 4)
        timing_names = {
            "campaign_guidance_training_wall_seconds",
            "campaign_guided_generation_wall_seconds",
        }
        self.assertTrue(hasattr(
            terminal.snapshot,
            "campaign_guidance_training_wall_seconds",
        ))
        self.assertTrue(hasattr(
            terminal_cohort.public.batch,
            "campaign_guided_generation_wall_seconds",
        ))
        self.assertFalse(any(name.startswith("candidate_") for name in timing_names))


if __name__ == "__main__":
    unittest.main()
