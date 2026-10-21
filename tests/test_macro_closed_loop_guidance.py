import hashlib
import unittest
from dataclasses import replace

from pde_agent_verifier_solver.engine.macro_closed_loop import (
    MacroAgentSpec,
    MacroClosedLoopRunner,
    MacroLoopConfig,
)
from pde_agent_verifier_solver.engine.selection import EvidenceStage
from pde_agent_verifier_solver.sdk import AnswerView, CaseRecord
from pde_agent_verifier_solver.sdk.guidance import GuidedSampleSpec
from pde_agent_verifier_solver.sdk.macro_loop import (
    CandidateRef,
    CandidateSubmission,
    MacroLoopPlugin,
)

from tests.test_guidance_controller import _Backend, _Oracle, config as guidance_config
from tests.test_macro_closed_loop import (
    _AbsoluteMacroMatcher,
    _Approximation,
    _FixedCaseSource,
    _ImprovingAgent,
    _Objective,
    _ProfileRecordingObjective,
)


class _GuidedZeroTruth:
    def answer(self, query):
        return AnswerView(
            kind=query.kind,
            fields={"macro": (0.0,)},
            axes={"sample": (0.0,)},
            points=query.points,
            shape=(1,),
        )


class _MacroOracle(_Oracle):
    def generate_answer_first(self, request):
        samples = super().generate_answer_first(request)
        for sample in samples:
            self.records[sample.sample_id] = CaseRecord(
                sample.case,
                _GuidedZeroTruth(),
            )
        return samples


class _MacroBackend(_Backend):
    def __init__(self):
        super().__init__()
        self.opened_oracles = []

    def open_guidance(self, snapshot):
        oracle = _MacroOracle(self, snapshot)
        self.opened_oracles.append(oracle)
        return oracle


class _ResidentCaseSource(_FixedCaseSource):
    def __init__(self):
        super().__init__()
        self.resident = {}
        self.released_ids = []

    def sample(self, request):
        records = super().sample(request)
        self.resident.update(
            {record.case.case_id: record for record in records}
        )
        return records

    def release_cases(self, case_ids):
        ids = tuple(case_ids)
        self.released_ids.extend(ids)
        for case_id in ids:
            self.resident.pop(case_id, None)


class _ResidentCheckingBackend(_MacroBackend):
    def __init__(self, source, *, fail_seal=False):
        super().__init__()
        self.source = source
        self.fail_seal = fail_seal
        self.feedback_case_ids = []
        self.seal_saw_resident = []

    def seal_reward_feedback(self, feedback):
        ids = tuple(row.case.case_id for row in feedback.rows)
        self.feedback_case_ids.append(ids)
        self.seal_saw_resident.append(
            all(case_id in self.source.resident for case_id in ids)
        )
        if self.fail_seal:
            raise RuntimeError("intentional guidance seal failure")
        return super().seal_reward_feedback(feedback)


class _GuidanceAwareAgent(_ImprovingAgent):
    def __init__(self):
        super().__init__()
        self.guided_solution = None

    def run(self, context):
        if context.guidance_oracle is None:
            raise AssertionError("round Agent did not receive frozen guide")
        batch = context.guidance_oracle.sample(GuidedSampleSpec(1))
        self.guided_solution = context.guidance_oracle.query_completed_solution(
            batch.samples[0].sample_id,
            "query",
            points=({"t": 0.25, "x": 0.5},),
        )
        return super().run(context)


class MacroClosedLoopGuidanceTests(unittest.TestCase):
    def test_runner_reports_guidance_and_uses_it_as_a_promotion_guardrail(self):
        source = _FixedCaseSource()
        backend = _MacroBackend()
        agent = _GuidanceAwareAgent()
        objective = _ProfileRecordingObjective(_AbsoluteMacroMatcher())
        cold_ref = CandidateRef(
            "cold-start",
            "c" * 64,
            0,
            "frozen-baseline",
        )
        config = MacroLoopConfig(
            name="guided-runner",
            round_count=1,
            agents=(MacroAgentSpec("agent", "Agent"),),
            development_case_count=2,
            selection_case_count=2,
            evidence_stages=(EvidenceStage(2, confidence_z=0.0),),
            terminal_case_count=2,
            agent_case_credits=2,
            seed_base=100,
            guidance_protocol=replace(
                guidance_config(),
                reward_feedback_case_count=2,
                official_round_case_count=2,
                official_terminal_case_count=2,
            ),
        )
        result = MacroClosedLoopRunner(
            legacy_macro_reference=True,
            config=config,
            plugin=MacroLoopPlugin(
                "guided-test",
                source,
                objective,
            ),
            cold_start=CandidateSubmission(cold_ref, _Approximation(0.5)),
            agents={"agent": agent},
            guidance_backend=backend,
        ).run()

        self.assertEqual(len(backend.training_requests), 2)
        reward_identity = hashlib.sha256(
            (
                "macro-guidance-primary-error-v2"
                "\0query_profile=terminal"
                f"\0{_AbsoluteMacroMatcher.contract_id}"
            ).encode("utf-8")
        ).hexdigest()
        self.assertTrue(all(
            request.reward_matcher_definition_sha256 == reward_identity
            for request in backend.training_requests
        ))
        self.assertEqual(
            backend.training_requests[0].target_solver_sha256,
            cold_ref.bundle_sha256,
        )
        self.assertEqual(
            backend.training_requests[1].target_solver_sha256,
            result.selected.bundle_sha256,
        )
        self.assertIsNotNone(agent.guided_solution)
        self.assertTrue(any(
            oracle.closed for oracle in backend.opened_oracles
        ))
        round_guided = result.rounds[0].guidance_evaluation
        terminal_guided = result.terminal.guidance_evaluation
        self.assertIsNotNone(round_guided)
        self.assertIsNotNone(terminal_guided)
        assert round_guided is not None
        assert terminal_guided is not None
        self.assertEqual(len({
            row.guided_batch.cohort_sha256
            for row in round_guided.candidates
        }), 1)
        self.assertTrue(all(
            row.separated_metrics is not None
            for row in round_guided.candidates
        ))
        by_id = {
            row.candidate.candidate_id: row
            for row in round_guided.candidates
        }
        self.assertLess(
            result.rounds[0].finalist_evaluations[0].aggregate.mean,
            result.rounds[0].leaderboard[1].mean,
        )
        self.assertGreater(
            by_id["candidate-r1"].guided_aggregate.mean,
            by_id["cold-start"].guided_aggregate.mean,
        )
        self.assertEqual(result.selected.candidate_id, "cold-start")
        decision = result.rounds[0].promotion_decision
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertFalse(decision.promoted)
        self.assertEqual(
            decision.reason,
            "incumbent_kept_after_robustness_guardrails",
        )
        reports = result.rounds[0].promotion_reports
        self.assertEqual(len(reports), 2)
        self.assertTrue(all(
            {"base", "guided"} <= set(report.distributions)
            for report in reports
        ))
        self.assertEqual(
            terminal_guided.snapshot.training_request.target_solver_sha256,
            result.selected.bundle_sha256,
        )
        guided_profiles = {
            profile
            for case_id, profile in objective.query_profiles
            if case_id.startswith("guided-official_common_hidden-")
        }
        feedback_profiles = {
            profile
            for case_id, profile in objective.query_profiles
            if "-guidance_feedback-" in case_id
        }
        self.assertEqual(guided_profiles, {"terminal"})
        self.assertEqual(feedback_profiles, {"terminal"})

    def test_feedback_is_sealed_before_resident_truth_is_released(self):
        source = _ResidentCaseSource()
        backend = _ResidentCheckingBackend(source)
        cold_ref = CandidateRef(
            "cold-start", "c" * 64, 0, "frozen-baseline"
        )
        config = MacroLoopConfig(
            name="guided-ordering",
            round_count=1,
            agents=(MacroAgentSpec("agent", "Agent"),),
            development_case_count=2,
            selection_case_count=2,
            evidence_stages=(EvidenceStage(2, confidence_z=0.0),),
            terminal_case_count=2,
            agent_case_credits=2,
            seed_base=200,
            guidance_protocol=replace(
                guidance_config(),
                reward_feedback_case_count=2,
                official_round_case_count=2,
                official_terminal_case_count=2,
            ),
        )

        MacroClosedLoopRunner(
            legacy_macro_reference=True,
            config=config,
            plugin=MacroLoopPlugin(
                "guided-order-test",
                source,
                _Objective(_AbsoluteMacroMatcher()),
            ),
            cold_start=CandidateSubmission(cold_ref, _Approximation(0.5)),
            agents={"agent": _GuidanceAwareAgent()},
            guidance_backend=backend,
        ).run()

        self.assertEqual(backend.seal_saw_resident, [True, True])
        released = set(source.released_ids)
        self.assertTrue(all(
            set(ids) <= released for ids in backend.feedback_case_ids
        ))

    def test_feedback_seal_failure_still_releases_resident_truth(self):
        source = _ResidentCaseSource()
        backend = _ResidentCheckingBackend(source, fail_seal=True)
        cold_ref = CandidateRef(
            "cold-start", "c" * 64, 0, "frozen-baseline"
        )
        config = MacroLoopConfig(
            name="guided-ordering-failure",
            round_count=1,
            agents=(MacroAgentSpec("agent", "Agent"),),
            development_case_count=2,
            selection_case_count=2,
            evidence_stages=(EvidenceStage(2, confidence_z=0.0),),
            terminal_case_count=2,
            agent_case_credits=2,
            seed_base=300,
            guidance_protocol=replace(
                guidance_config(),
                reward_feedback_case_count=2,
                official_round_case_count=2,
                official_terminal_case_count=2,
            ),
        )
        runner = MacroClosedLoopRunner(
            legacy_macro_reference=True,
            config=config,
            plugin=MacroLoopPlugin(
                "guided-order-failure-test",
                source,
                _Objective(_AbsoluteMacroMatcher()),
            ),
            cold_start=CandidateSubmission(cold_ref, _Approximation(0.5)),
            agents={"agent": _GuidanceAwareAgent()},
            guidance_backend=backend,
        )

        with self.assertRaisesRegex(
            RuntimeError, "intentional guidance seal failure"
        ):
            runner.run()

        self.assertEqual(backend.seal_saw_resident, [True])
        feedback_ids = set(backend.feedback_case_ids[0])
        self.assertTrue(feedback_ids <= set(source.released_ids))
        self.assertTrue(feedback_ids.isdisjoint(source.resident))


if __name__ == "__main__":
    unittest.main()
