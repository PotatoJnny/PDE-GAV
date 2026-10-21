import unittest
from dataclasses import replace

from pde_agent_verifier_solver.engine.runtime import AgentOutcome
from pde_agent_verifier_solver.panel.controller import build_toy_panel
from pde_agent_verifier_solver.plugins.toy import (
    SineTruth,
    ToyPoissonGenerator,
    ToyResearchAgent,
    ToySineSolver,
)
from pde_agent_verifier_solver.sdk import (
    Artifact,
    CaseRecord,
    CaseVisibility,
    GenerationKind,
    GenerationPurpose,
    TaskKind,
)


def _bare_artifact(index: int) -> Artifact:
    return Artifact(
        artifact_id=f"bare-{index}",
        kind=TaskKind.SOLVER,
        name=f"Bare candidate {index}",
        program=object(),
        summary="AgentOutcome contract fixture.",
        usage="test only",
        round_index=1,
        agent_id="contract-agent",
    )


class RuntimeMultiCandidateTerminalTests(unittest.TestCase):
    def test_initial_candidate_drives_round_one_guidance(self) -> None:
        class RecordingGenerator(ToyPoissonGenerator):
            feedback_sources = ()

            def update_guidance(self, state, feedback, settings, progress=None):
                self.feedback_sources = (*self.feedback_sources, feedback.source_round_index)
                return super().update_guidance(state, feedback, settings, progress)

        initial = Artifact(
            artifact_id="user-s0",
            kind=TaskKind.SOLVER,
            name="User S0",
            program=ToySineSolver(max_resolved_mode=1),
            summary="Explicit user starting solver.",
            usage="solver.solve(case, query)",
            round_index=0,
            agent_id="user",
            metadata={"max_resolved_mode": 1},
        )
        panel = build_toy_panel()
        generator = RecordingGenerator()
        panel.programs = replace(panel.programs, generator=generator)
        launched = panel.create_and_launch({
            "project_name": "S0 round-one guidance",
            "round_count": 1,
            "agents": ["careful-agent"],
            "initial_candidate": initial,
            "allowed_sources": ["base", "guided"],
            "starter_cases": {"guided": 1},
            "max_sampled_cases_per_agent_per_round": 1,
            "round_hidden": {"base": 1, "guided": 1},
            "terminal_hidden": {"base": 1},
        })
        panel.app.wait(launched["run_id"], timeout=5)
        page = panel.run_page(launched["run_id"])

        self.assertEqual(page["status"], "completed", page.get("error"))
        self.assertEqual(page["initial_candidate_id"], "user-s0")
        self.assertIn("guided", page["plan"][0]["allowed_sources"])
        self.assertEqual(generator.feedback_sources[0], 0)

    def test_initial_guidance_cohort_cannot_be_reused_as_official(self) -> None:
        class ReusingGenerator(ToyPoissonGenerator):
            def sample(self, state, request, progress=None):
                records = super().sample(state, request, progress)
                if (
                    request.visibility is CaseVisibility.HIDDEN
                    and request.kind is GenerationKind.BASE
                    and request.purpose in {
                        GenerationPurpose.GUIDANCE_FEEDBACK,
                        GenerationPurpose.PROMOTION,
                    }
                ):
                    return tuple(
                        replace(
                            record,
                            case=replace(record.case, case_id=f"reused-private-{index}"),
                        )
                        for index, record in enumerate(records)
                    )
                return records

        initial = Artifact(
            artifact_id="reuse-s0",
            kind=TaskKind.SOLVER,
            name="Reuse S0",
            program=ToySineSolver(max_resolved_mode=1),
            summary="Initial candidate for freshness boundary test.",
            usage="solver.solve(case, query)",
            round_index=0,
            agent_id="user",
            metadata={"max_resolved_mode": 1},
        )
        panel = build_toy_panel()
        panel.programs = replace(panel.programs, generator=ReusingGenerator())
        launched = panel.create_and_launch({
            "project_name": "S0 hidden isolation",
            "round_count": 1,
            "agents": ["careful-agent"],
            "initial_candidate": initial,
            "allowed_sources": ["base", "guided"],
            "starter_cases": {"base": 1},
            "max_sampled_cases_per_agent_per_round": 1,
            "round_hidden": {"base": 1},
            "terminal_hidden": {"base": 1},
        })
        panel.app.wait(launched["run_id"], timeout=5)
        page = panel.run_page(launched["run_id"])

        self.assertEqual(page["status"], "failed")
        self.assertIn("already used for Initial Guidance", page["error"])

    def test_agent_outcome_flattens_an_uncapped_portfolio_compatibly(self) -> None:
        primary = _bare_artifact(0)
        additional = tuple(_bare_artifact(index) for index in range(1, 65))

        outcome = AgentOutcome(
            primary,
            ("legacy positional observations remain supported",),
            artifacts=additional,
        )

        self.assertIs(outcome.artifact, primary)
        self.assertEqual(outcome.observations, (
            "legacy positional observations remain supported",
        ))
        self.assertEqual(outcome.artifacts, additional)
        self.assertEqual(outcome.nominations, (primary, *additional))

    def test_every_portfolio_nomination_and_incumbent_share_round_exam(self) -> None:
        class PortfolioAgent:
            def run(self, context):
                candidates = tuple(
                    ToyResearchAgent._artifact(
                        context,
                        mode,
                        suffix=f"portfolio-{mode}",
                    )
                    for mode in range(1, 13)
                )
                return AgentOutcome(
                    candidates[0],
                    ("Submitted a twelve-program portfolio.",),
                    artifacts=candidates[1:],
                )

        panel = build_toy_panel()
        panel.programs.agents["portfolio-agent"] = PortfolioAgent()
        launched = panel.create_and_launch({
            "project_name": "Multi-candidate round evaluation",
            "round_count": 2,
            "agents": ["portfolio-agent"],
            "allowed_sources": ["base"],
            "starter_cases": {"base": 1},
            "max_sampled_cases_per_agent_per_round": 1,
            "generator_settings": {
                "guidance_enabled": False,
                "perturbations_enabled": False,
            },
            "round_hidden": {"base": 2},
            "terminal_hidden": {"base": 1},
        })
        panel.app.wait(launched["run_id"], timeout=5)
        session = panel.app.get_run(launched["run_id"])

        self.assertEqual(session.status.value, "completed", session.error)
        first, second = session.result.rounds
        self.assertEqual(len(first.candidate_evaluations), 12)
        self.assertEqual(len(first.promotion_reports), 12)
        self.assertEqual(len(second.candidate_evaluations), 12)
        self.assertEqual(len(second.promotion_reports), 13)
        self.assertIn(
            first.selected.artifact.artifact_id,
            {report.candidate_id for report in second.promotion_reports},
        )

        for round_result in (first, second):
            common_case_ids = {
                tuple(case.case_id for case in evaluation.cases)
                for evaluation in round_result.candidate_evaluations
            }
            self.assertEqual(len(common_case_ids), 1)
            self.assertEqual(
                {report.distributions["base"].case_count
                 for report in round_result.promotion_reports},
                {2},
            )

        agent_events = tuple(
            event
            for event in session.events.snapshot()
            if event.event_type == "agent_finished"
        )
        self.assertEqual(len(agent_events), 2)
        self.assertTrue(all(
            event.payload["nomination_count"] == 12 for event in agent_events
        ))
        self.assertTrue(all(
            len(event.payload["artifact_ids"]) == 12 for event in agent_events
        ))

    def test_terminal_scores_every_round_selection_but_delivers_final_round(self) -> None:
        class FrequencySpecialist:
            def __init__(self, target_frequency):
                self.target_frequency = target_frequency

            def solve(self, case, query, budget=None):
                del budget
                frequency = int(case.public_data["frequency"])
                amplitude = float(case.public_data["amplitude"])
                represented_amplitude = (
                    amplitude if frequency == self.target_frequency else 0.0
                )
                return SineTruth(represented_amplitude, frequency).answer(query)

        class RoundSpecialistAgent:
            def run(self, context):
                target = context.round_plan.round_index
                return AgentOutcome(Artifact(
                    # Agent ids are untrusted labels. Reusing this label across
                    # rounds must not collapse the Controller snapshots.
                    artifact_id="reused-specialist-label",
                    kind=TaskKind.SOLVER,
                    name=f"Frequency {target} specialist",
                    program=FrequencySpecialist(target),
                    summary=f"Solves only frequency {target}.",
                    usage="solver.solve(case, query)",
                    round_index=target,
                    agent_id=context.agent.agent_id,
                    parent_artifact_id=(
                        context.incumbent.artifact_id if context.incumbent else None
                    ),
                    metadata={"max_resolved_mode": target},
                ))

        class PurposeControlledGenerator(ToyPoissonGenerator):
            def sample(self, state, request, progress=None):
                records = super().sample(state, request, progress)
                if (
                    request.visibility is not CaseVisibility.HIDDEN
                    or request.kind is not GenerationKind.BASE
                ):
                    return records
                frequency = (
                    1
                    if request.purpose is GenerationPurpose.TERMINAL
                    else request.round_index
                )
                rewritten = []
                for record in records:
                    amplitude = float(record.case.public_data["amplitude"])
                    rewritten.append(CaseRecord(
                        case=replace(
                            record.case,
                            public_data={
                                **record.case.public_data,
                                "frequency": frequency,
                            },
                        ),
                        truth=SineTruth(amplitude, frequency),
                    ))
                return tuple(rewritten)

        panel = build_toy_panel()
        panel.programs = replace(
            panel.programs,
            generator=PurposeControlledGenerator(),
        )
        panel.programs.agents["specialist-agent"] = RoundSpecialistAgent()
        launched = panel.create_and_launch({
            "project_name": "All round terminal comparison",
            "round_count": 2,
            "agents": ["specialist-agent"],
            "allowed_sources": ["base"],
            "starter_cases": {"base": 1},
            "max_sampled_cases_per_agent_per_round": 1,
            "generator_settings": {
                "guidance_enabled": False,
                "perturbations_enabled": False,
            },
            "round_hidden": {"base": 1},
            "terminal_hidden": {"base": 1},
            "terminal_score_distribution": "base",
        })
        panel.app.wait(launched["run_id"], timeout=5)
        session = panel.app.get_run(launched["run_id"])

        self.assertEqual(session.status.value, "completed", session.error)
        result = session.result
        selected_ids = tuple(
            round_result.selected.artifact.artifact_id
            for round_result in result.rounds
        )
        self.assertEqual(len(set(selected_ids)), 2)
        self.assertTrue(all(value.startswith("candidate-r") for value in selected_ids))
        self.assertEqual(
            tuple(
                round_result.selected.artifact.metadata["submitted_artifact_id"]
                for round_result in result.rounds
            ),
            ("reused-specialist-label", "reused-specialist-label"),
        )
        self.assertEqual(
            tuple(report.candidate_id for report in result.terminal_round_reports),
            selected_ids,
        )
        self.assertAlmostEqual(
            result.terminal_round_reports[0].distributions["base"].mean,
            0.0,
        )
        self.assertAlmostEqual(
            result.terminal_round_reports[1].distributions["base"].mean,
            1.0,
        )
        self.assertEqual(result.best_terminal_round, 1)
        self.assertEqual(
            result.selected.artifact.artifact_id,
            selected_ids[-1],
        )
        self.assertAlmostEqual(result.selected.primary_error, 1.0)
        self.assertAlmostEqual(
            result.terminal_distribution_metrics["base"]["mean"],
            1.0,
        )

    def test_repeated_selection_reuses_one_terminal_report_per_round_slot(self) -> None:
        class KeepIncumbentAgent(ToyResearchAgent):
            def run(self, context):
                if context.incumbent is not None:
                    context.incumbent.metadata["agent_attempted_mutation"] = True
                    return AgentOutcome(
                        observations=("No new Candidate passed development.",)
                    )
                return super().run(context)

        panel = build_toy_panel()
        panel.programs.agents["keep-agent"] = KeepIncumbentAgent()
        launched = panel.create_and_launch({
            "project_name": "Repeated terminal selection",
            "round_count": 2,
            "agents": ["keep-agent"],
            "allowed_sources": ["base"],
            "starter_cases": {"base": 1},
            "max_sampled_cases_per_agent_per_round": 8,
            "generator_settings": {
                "guidance_enabled": False,
                "perturbations_enabled": False,
            },
            "round_hidden": {"base": 1},
            "terminal_hidden": {"base": 1},
        })
        panel.app.wait(launched["run_id"], timeout=5)
        session = panel.app.get_run(launched["run_id"])

        self.assertEqual(session.status.value, "completed", session.error)
        result = session.result
        self.assertEqual(len(result.terminal_round_reports), 2)
        self.assertIs(
            result.terminal_round_reports[0],
            result.terminal_round_reports[1],
        )
        self.assertEqual(result.best_terminal_round, 1)
        self.assertNotIn(
            "agent_attempted_mutation",
            result.rounds[0].selected.artifact.metadata,
        )
        self.assertEqual(
            result.selected.artifact.artifact_id,
            result.rounds[-1].selected.artifact.artifact_id,
        )


if __name__ == "__main__":
    unittest.main()
