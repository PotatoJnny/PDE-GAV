import unittest

from pde_agent_verifier_solver.panel.controller import build_toy_panel
from pde_agent_verifier_solver.panel.server import HTML


class ToyProductFlowTests(unittest.TestCase):
    def test_generated_case_ids_are_derived_from_the_request(self) -> None:
        from pde_agent_verifier_solver.plugins.toy import (
            ToyBaseGeneratorState,
            ToyPoissonGenerator,
        )
        from pde_agent_verifier_solver.sdk import (
            CaseVisibility,
            GenerationKind,
            GenerationPurpose,
            GenerationRequest,
            GeneratorCheckpoint,
            GeneratorMode,
            GeneratorState,
        )

        generator = ToyPoissonGenerator()
        checkpoint = GeneratorCheckpoint(
            generator.info.program_id,
            ToyBaseGeneratorState(5, 1.0, 8),
        )
        state = GeneratorState(checkpoint, checkpoint.payload)
        first = GenerationRequest(
            round_index=2,
            count=2,
            mode=GeneratorMode.ONLINE,
            kind=GenerationKind.BASE,
            visibility=CaseVisibility.DEVELOPMENT,
            seed=197,
            purpose=GenerationPurpose.DEVELOPMENT,
        )
        other = GenerationRequest(
            round_index=2,
            count=1,
            mode=GeneratorMode.ONLINE,
            kind=GenerationKind.BASE,
            visibility=CaseVisibility.DEVELOPMENT,
            seed=211,
            purpose=GenerationPurpose.DEVELOPMENT,
        )

        expected = tuple(
            row.case.case_id for row in generator.sample(state, first)
        )
        generator.sample(state, other)
        repeated = tuple(
            row.case.case_id for row in generator.sample(state, first)
        )

        self.assertEqual(repeated, expected)
        self.assertNotEqual(
            expected[0],
            generator.sample(state, other)[0].case.case_id,
        )

    def test_panel_runs_three_round_campaign_to_a_result(self) -> None:
        from dataclasses import replace
        from pde_agent_verifier_solver.plugins.toy import ToyResearchAgent

        class RecordingAgent(ToyResearchAgent):
            def __init__(self, step_size):
                super().__init__(step_size)
                self.memory_round_counts = []
                self.emitted_logs = []

            def run(self, context):
                self.memory_round_counts.append(len(context.memory.rounds))
                outcome = super().run(context)
                markdown = (
                    f"# {context.agent.name} round {context.round_plan.round_index}\n\n"
                    f"Saw {len(context.memory.rounds)} completed prior rounds."
                )
                self.emitted_logs.append(markdown)
                return replace(outcome, round_log=markdown)

        panel = build_toy_panel()
        recorders = {
            "careful-agent": RecordingAgent(step_size=1),
            "explorer-agent": RecordingAgent(step_size=2),
        }
        panel.programs.agents.update(recorders)
        launched = panel.create_and_launch({
            "project_name": "Toy product test",
            "dataset_kind": "numerical",
            "round_count": 3,
            "sampling_strategy": "agent_adaptive",
            "max_sampled_cases_per_agent_per_round": 12,
            "agent_budget_seconds": 30,
            "basis_size": 8,
            "agents": ["careful-agent", "explorer-agent"],
            "round_hidden": {"original": 1, "base": 3, "guided": 2},
            "terminal_hidden": {"original": 1, "base": 4},
            "promotion": {"distribution": "base", "statistic": "mean"},
        })
        panel.app.wait(launched["run_id"], timeout=5)
        page = panel.run_page(launched["run_id"])

        self.assertEqual(page["status"], "completed", page.get("error"))
        rounds = page["rounds"]
        self.assertEqual(len(rounds), 3)
        self.assertEqual([row["hidden_suite_size"] for row in rounds], [6, 6, 6])
        errors = [row["selected_error"] for row in rounds]
        self.assertGreater(errors[0], errors[1])
        self.assertGreaterEqual(errors[1], errors[2])
        self.assertEqual(page["plan"][0]["sampling_strategy"], "agent_adaptive")
        self.assertLessEqual(sum(rounds[0]["agent_case_counts"]["careful-agent"].values()), 12)
        self.assertIn("guided", rounds[1]["hidden_distributions"][
            rounds[1]["selected_artifact_id"]])
        promotion = rounds[1]["promotion"]
        self.assertTrue(promotion["reports"])
        self.assertEqual(
            {row["candidate_id"] for row in promotion["reports"]},
            {row["candidate_id"] for row in promotion["ranking"]},
        )
        guided = next(
            report["distributions"]["guided"]
            for report in promotion["reports"]
            if "guided" in report["distributions"]
        )
        self.assertEqual(
            set(guided),
            {"case_count", "mean", "cvar", "maximum", "metrics"},
        )
        self.assertIn("relative_l2", guided["metrics"])
        self.assertTrue(promotion["reason"])
        session = panel.app.get_run(launched["run_id"])
        self.assertTrue(session.result.rounds[1].promotion_reports)
        memory = page["memory"]
        self.assertEqual(len(memory["rounds"]), 3)
        self.assertEqual([row["round"] for row in memory["rounds"]], [1, 2, 3])
        self.assertTrue(memory["markdown"].startswith("# Campaign Memory Log"))
        self.assertEqual(
            [len(row["agents"]) for row in memory["rounds"]],
            [2, 2, 2],
        )
        self.assertTrue(all(
            row["controller_markdown"]
            for row in memory["rounds"]
        ))
        self.assertEqual(len(page["candidate_history"]), 6)
        self.assertEqual(len(session.result.memory.rounds), 3)
        self.assertEqual(len(session.result.candidate_history.nodes), 6)
        for agent_id, recorder in recorders.items():
            self.assertEqual(recorder.memory_round_counts, [0, 1, 2])
            for index, round_memory in enumerate(memory["rounds"]):
                agent_log = next(
                    row for row in round_memory["agents"]
                    if row["agent_id"] == agent_id
                )
                self.assertEqual(agent_log["markdown"], recorder.emitted_logs[index])
        self.assertIn("solver.solve", page["result"]["usage"])
        self.assertIn("base", page["result"]["terminal_distributions"])
        self.assertNotIn("guided", page["result"]["terminal_distributions"])
        self.assertEqual(len(page["terminal_round_reports"]), 3)
        self.assertIn(page["best_terminal_round"], {1, 2, 3})
        self.assertEqual(page["master_seed"], 0)
        self.assertFalse(page["allow_grid_truth"])
        self.assertFalse(page["allow_point_truth"])
        self.assertEqual(
            page["development_coordinate_mode"],
            "private_random_points",
        )
        self.assertEqual(page["development_point_count"], 32)
        self.assertIsNone(page["development_fixed_query"])
        self.assertEqual(
            page["terminal_evaluation"]["ranking"],
            {
                "cohort_id": "base_model",
                "distribution": "base",
                "statistic": "mean",
            },
        )
        self.assertTrue(page["seed_ledger"])
        self.assertEqual(page["result"]["seed_ledger"], page["seed_ledger"])
        self.assertEqual(
            page["seed_ledger"],
            dict(session.result.seed_ledger),
        )
        self.assertTrue(any(
            key.startswith("evaluation_plan/development/")
            for key in page["seed_ledger"]
        ))
        self.assertTrue(any(
            key.startswith("evaluation_plan/official_round:")
            for key in page["seed_ledger"]
        ))
        self.assertTrue(any(
            key.startswith("evaluation_plan/terminal_evaluation/")
            for key in page["seed_ledger"]
        ))

        event_types = [event["type"] for event in page["events"]]
        self.assertIn("guidance_ready", event_types)
        self.assertEqual(event_types[-1], "run_finished")

    def test_terminal_hidden_is_fresh_and_does_not_retrain_guidance(self) -> None:
        from dataclasses import replace
        from pde_agent_verifier_solver.plugins.toy import ToyPoissonGenerator
        from pde_agent_verifier_solver.sdk import (
            CaseVisibility,
            GenerationKind,
            GenerationPurpose,
        )

        class RecordingGenerator(ToyPoissonGenerator):
            def __init__(self) -> None:
                super().__init__()
                self.requests = []
                self.guidance_programs = []
                self.hidden_base_fingerprints = {}

            def sample(self, state, request, progress=None):
                self.requests.append(request)
                records = super().sample(state, request, progress)
                if (
                    request.visibility is CaseVisibility.HIDDEN
                    and request.kind is GenerationKind.BASE
                ):
                    self.hidden_base_fingerprints[request.purpose] = tuple(
                        (
                            record.case.public_data["frequency"],
                            record.case.public_data["amplitude"],
                        )
                        for record in records
                    )
                return records

            def update_guidance(self, state, feedback, settings, progress=None):
                self.guidance_programs.append(feedback.selected_program_id)
                return super().update_guidance(state, feedback, settings, progress)

        panel = build_toy_panel()
        generator = RecordingGenerator()
        panel.programs = replace(panel.programs, generator=generator)
        launched = panel.create_and_launch({
            "project_name": "Fresh terminal suite",
            "round_count": 2,
            "agents": ["careful-agent"],
            "round_hidden": {"original": 0, "base": 2, "guided": 1},
            "terminal_hidden": {"original": 0, "base": 2},
        })
        panel.app.wait(launched["run_id"], timeout=5)
        page = panel.run_page(launched["run_id"])

        self.assertEqual(page["status"], "completed", page.get("error"))
        final_base = [
            request
            for request in generator.requests
            if request.visibility is CaseVisibility.HIDDEN
            and request.kind is GenerationKind.BASE
            and request.purpose in {
                GenerationPurpose.PROMOTION,
                GenerationPurpose.TERMINAL,
            }
        ]
        purposes = {request.purpose: request.seed for request in final_base}
        self.assertNotEqual(
            purposes[GenerationPurpose.PROMOTION],
            purposes[GenerationPurpose.TERMINAL],
        )
        self.assertNotEqual(
            generator.hidden_base_fingerprints[GenerationPurpose.PROMOTION],
            generator.hidden_base_fingerprints[GenerationPurpose.TERMINAL],
        )
        terminal_request = next(
            request
            for request in final_base
            if request.purpose is GenerationPurpose.TERMINAL
        )
        self.assertEqual(terminal_request.round_index, 0)
        self.assertEqual(len(generator.guidance_programs), 1)
        event_types = [event["type"] for event in page["events"]]
        self.assertNotIn("terminal_guidance_ready", event_types)

    def test_experimental_dataset_uses_the_same_product_flow(self) -> None:
        panel = build_toy_panel()
        launched = panel.create_and_launch({
            "project_name": "Experimental toy",
            "dataset_kind": "experimental",
            "round_count": 1,
            "sampling_strategy": "agent_adaptive",
            "max_sampled_cases_per_agent_per_round": 10,
            "agent_budget_seconds": 10,
            "problem_settings": {
                "domain": "x in [0, 1]",
                "development_cases": 12,
                "hidden_cases": 4,
            },
            "task_settings": {"primary_metric": "relative_l2"},
            "generator_settings": {
                "basis_size": 6,
                "guidance_enabled": True,
                "perturbations_enabled": True,
            },
            "agents": ["careful-agent"],
        })
        panel.app.wait(launched["run_id"], timeout=5)
        page = panel.run_page(launched["run_id"])
        self.assertEqual(page["status"], "completed", page.get("error"))
        self.assertIsNotNone(page["result"])
        self.assertEqual(page["plan"][0]["sampling_strategy"], "agent_adaptive")


    def test_fixed_recipe_is_the_same_sampler_with_blinded_cases(self) -> None:
        from pde_agent_verifier_solver.plugins.toy import ToyResearchAgent

        class CapturingAgent(ToyResearchAgent):
            def __init__(self) -> None:
                super().__init__()
                self.cases = ()

            def run(self, context):
                self.cases = context.cases
                return super().run(context)

        panel = build_toy_panel()
        agent = CapturingAgent()
        panel.programs.agents["careful-agent"] = agent
        launched = panel.create_and_launch({
            "project_name": "Fixed recipe",
            "sampling_strategy": "fixed_recipe",
            "round_count": 1,
            "max_sampled_cases_per_agent_per_round": 12,
            "agents": ["careful-agent"],
        })
        panel.app.wait(launched["run_id"], timeout=5)
        page = panel.run_page(launched["run_id"])

        self.assertEqual(page["status"], "completed", page.get("error"))
        self.assertEqual(page["plan"][0]["sampling_strategy"], "fixed_recipe")
        counts = page["rounds"][0]["agent_case_counts"]["careful-agent"]
        self.assertEqual(sum(counts.values()), 12)
        self.assertEqual(
            {case.source.value for case in agent.cases},
            {"generated"},
        )
        self.assertEqual(
            {case.visibility.value for case in agent.cases},
            {"development"},
        )
        self.assertTrue(
            all(case.case_id.startswith("candidate-case-") for case in agent.cases)
        )
        self.assertTrue(all(case.parent_case_id is None for case in agent.cases))
        self.assertTrue(all(not case.tags for case in agent.cases))
        for case in agent.cases:
            self.assertNotIn("generator_mode", case.public_data)
            self.assertNotIn("observation_kind", case.public_data)
            self.assertNotIn("reference_kind", case.public_data)
        original_flags = ["observed_u" in case.public_data for case in agent.cases]
        self.assertNotEqual(original_flags, sorted(original_flags, reverse=True))

    def test_first_round_converts_guided_perturbations_without_dropping_cases(self) -> None:
        panel = build_toy_panel()
        counts = {
            "original": 0,
            "base": 1,
            "guided": 1,
            "base_perturbed": 1,
            "guided_perturbed": 1,
        }
        launched = panel.create_and_launch({
            "project_name": "First round conversion",
            "sampling_strategy": "fixed_recipe",
            "round_count": 1,
            "max_sampled_cases_per_agent_per_round": 4,
            "agents": ["careful-agent"],
            "fixed_recipe": counts,
            "round_hidden": counts,
            "terminal_hidden": {
                "original": 0,
                "base": 2,
                "base_perturbed": 2,
            },
        })
        session = panel.app.get_run(launched["run_id"])
        plan = session.campaign.rounds[0]
        self.assertEqual(
            plan.development_sampling.fixed_recipe.as_dict(),
            {
                "original": 0,
                "base": 2,
                "guided": 0,
                "base_perturbed": 2,
                "guided_perturbed": 0,
            },
        )
        self.assertEqual(plan.round_hidden.base, 2)
        self.assertEqual(plan.round_hidden.base_perturbed, 2)
        self.assertIsNone(plan.terminal_hidden)
        terminal = session.campaign.terminal_evaluation
        self.assertIsNotNone(terminal)
        self.assertEqual(terminal.cohorts[0].suite.base, 2)
        self.assertEqual(terminal.cohorts[0].suite.base_perturbed, 2)
        panel.app.wait(launched["run_id"], timeout=5)
        page = panel.run_page(launched["run_id"])
        self.assertEqual(page["status"], "completed", page.get("error"))

    def test_paired_retry_is_shared_by_development_official_and_terminal(self) -> None:
        from dataclasses import replace

        from pde_agent_verifier_solver.plugins.toy import ToyPoissonGenerator
        from pde_agent_verifier_solver.sdk import (
            GenerationKind,
            GenerationPurpose,
            RecoverableSamplingRejection,
        )

        class RejectOncePerPurpose(ToyPoissonGenerator):
            def __init__(self) -> None:
                super().__init__()
                self.rejected = set()
                self.requests = []
                self.accepted_relative_changes = []

            def sample(self, state, request, progress=None):
                self.requests.append(request)
                if (
                    request.kind is GenerationKind.PERTURBED
                    and request.purpose not in self.rejected
                ):
                    self.rejected.add(request.purpose)
                    raise RecoverableSamplingRejection(
                        "test rejection",
                        audit={"gate": "test", "purpose": request.purpose.value},
                    )
                records = super().sample(state, request, progress)
                if request.kind is GenerationKind.PERTURBED:
                    for record, anchor in zip(records, request.anchors):
                        parent_amplitude = float(anchor.public_data["amplitude"])
                        child_amplitude = float(record.case.public_data["amplitude"])
                        self.accepted_relative_changes.append(
                            (
                                request.purpose,
                                abs(child_amplitude - parent_amplitude)
                                / abs(parent_amplitude),
                            )
                        )
                return records

        panel = build_toy_panel()
        generator = RejectOncePerPurpose()
        panel.programs = replace(panel.programs, generator=generator)
        paired_counts = {
            "original": 0,
            "base": 1,
            "guided": 0,
            "base_perturbed": 1,
            "guided_perturbed": 0,
        }
        launched = panel.create_and_launch({
            "project_name": "Shared paired retry",
            "sampling_strategy": "fixed_recipe",
            "round_count": 1,
            "max_sampled_cases_per_agent_per_round": 2,
            "agents": ["careful-agent"],
            "fixed_recipe": paired_counts,
            "round_hidden": paired_counts,
            "terminal_hidden": paired_counts,
            "generator_settings": {
                "perturbation_relative_change_min": 0.07,
                "perturbation_relative_change_max": 0.08,
                "perturbation_epsilon_fraction": 0.002,
                "perturbation_maximum_alpha": 2.5,
            },
        })
        panel.app.wait(launched["run_id"], timeout=5)
        session = panel.app.get_run(launched["run_id"])

        self.assertEqual(session.status.value, "completed", session.error)
        expected = {
            GenerationPurpose.DEVELOPMENT.value,
            GenerationPurpose.PROMOTION.value,
            GenerationPurpose.TERMINAL.value,
        }
        self.assertEqual(
            {row.purpose for row in session.result.sampling_attempts},
            expected,
        )
        for purpose in expected:
            rows = [
                row
                for row in session.result.sampling_attempts
                if row.purpose == purpose
            ]
            self.assertEqual([row.accepted for row in rows], [False, True])
        for purpose in (
            GenerationPurpose.DEVELOPMENT,
            GenerationPurpose.PROMOTION,
            GenerationPurpose.TERMINAL,
        ):
            perturbed = [
                request
                for request in generator.requests
                if request.purpose is purpose
                and request.kind is GenerationKind.PERTURBED
            ]
            self.assertEqual(len(perturbed), 2)
            self.assertTrue(all(request.count == 1 for request in perturbed))
            for request in perturbed:
                self.assertEqual(
                    request.settings["perturbation_relative_change_min"],
                    0.07,
                )
                self.assertEqual(
                    request.settings["perturbation_relative_change_max"],
                    0.08,
                )
                self.assertEqual(
                    request.settings["perturbation_epsilon_fraction"],
                    0.002,
                )
                self.assertEqual(
                    request.settings["perturbation_maximum_alpha"],
                    2.5,
                )
        self.assertEqual(
            {purpose for purpose, _change in generator.accepted_relative_changes},
            {
                GenerationPurpose.DEVELOPMENT,
                GenerationPurpose.PROMOTION,
                GenerationPurpose.TERMINAL,
            },
        )
        self.assertTrue(
            all(
                0.07 <= change <= 0.08
                for _purpose, change in generator.accepted_relative_changes
            )
        )

    def test_product_rejects_solver_dependent_terminal_guidance(self) -> None:
        panel = build_toy_panel()
        with self.assertRaisesRegex(ValueError, "base terminal cohort cannot request guided"):
            panel.create_and_launch({
                "project_name": "Invalid terminal guidance",
                "round_count": 2,
                "agents": ["careful-agent"],
                "round_hidden": {"original": 0, "base": 2, "guided": 1},
                "terminal_hidden": {"original": 0, "base": 2, "guided": 1},
            })

    def test_panel_builds_structured_campaign_terminal_configuration(self) -> None:
        panel = build_toy_panel()
        launched = panel.create_and_launch({
            "project_name": "Structured terminal configuration",
            "round_count": 2,
            "agents": ["careful-agent"],
            "starter_cases": {"base": 1},
            "max_sampled_cases_per_agent_per_round": 2,
            "round_hidden": {"base": 1, "guided": 1},
            "master_seed": 9182,
            "allow_grid_truth": True,
            "allow_point_truth": False,
            "development_coordinate_mode": "user_fixed_query",
            "development_point_count": 11,
            "terminal_evaluation": {
                "cohorts": [
                    {
                        "id": "base_model",
                        "guidance_source": "base",
                        "counts": {"base": 2},
                    },
                    {
                        "id": "round_2_model",
                        "guidance_source": "round_guided",
                        "round": 2,
                        "counts": {"guided": 1},
                    },
                    {
                        "id": "final_model",
                        "guidance_source": "final_guided",
                        "counts": {"guided": 1},
                    },
                ],
                "ranking": {
                    "cohort_id": "base_model",
                    "distribution": "base",
                    "statistic": "maximum",
                },
            },
        })
        panel.app.wait(launched["run_id"], timeout=5)
        session = panel.app.get_run(launched["run_id"])
        page = panel.run_page(launched["run_id"])

        self.assertEqual(page["status"], "completed", page.get("error"))
        campaign = session.campaign
        self.assertEqual(campaign.master_seed, 9182)
        self.assertTrue(campaign.evaluator.allow_grid_truth)
        self.assertFalse(campaign.evaluator.allow_point_truth)
        self.assertEqual(
            campaign.evaluator.development_coordinate_mode.value,
            "user_fixed_query",
        )
        self.assertEqual(campaign.evaluator.development_point_count, 11)
        self.assertTrue(all(plan.terminal_hidden is None for plan in campaign.rounds))
        self.assertEqual(
            [cohort.cohort_id for cohort in campaign.terminal_evaluation.cohorts],
            ["base_model", "round_2_model", "final_model"],
        )
        self.assertEqual(campaign.terminal_evaluation.cohorts[1].round_index, 2)
        self.assertEqual(
            campaign.terminal_evaluation.ranking.statistic,
            "maximum",
        )
        self.assertEqual(page["master_seed"], 9182)
        self.assertTrue(page["allow_grid_truth"])
        self.assertEqual(page["development_coordinate_mode"], "user_fixed_query")
        self.assertEqual(page["development_fixed_query"]["kind"], "grid")
        self.assertEqual(page["terminal_evaluation"]["cohorts"][1]["round"], 2)
        self.assertEqual(len(page["terminal_round_reports"]), 2)
        self.assertEqual(
            set(page["terminal_round_reports"][0]["distributions"]),
            {
                "base_model/base",
                "round_2_model/guided",
                "final_model/guided",
            },
        )

    def test_legacy_terminal_hidden_translates_to_one_campaign_cohort(self) -> None:
        panel = build_toy_panel()
        launched = panel.create_and_launch({
            "project_name": "Legacy terminal input",
            "round_count": 1,
            "agents": ["careful-agent"],
            "round_hidden": {"base": 1},
            "terminal_hidden": {"original": 1, "base": 2},
            "terminal_score_distribution": "original",
        })
        panel.app.wait(launched["run_id"], timeout=5)
        session = panel.app.get_run(launched["run_id"])

        self.assertEqual(session.status.value, "completed", session.error)
        terminal = session.campaign.terminal_evaluation
        self.assertEqual(len(terminal.cohorts), 1)
        self.assertEqual(terminal.cohorts[0].cohort_id, "base_model")
        self.assertEqual(terminal.cohorts[0].suite.original, 1)
        self.assertEqual(terminal.cohorts[0].suite.base, 2)
        self.assertEqual(terminal.ranking.distribution, "original")
        self.assertIsNone(session.campaign.rounds[0].terminal_hidden)

    def test_guidance_disabled_uses_only_unguided_defaults(self) -> None:
        panel = build_toy_panel()
        launched = panel.create_and_launch({
            "project_name": "Unguided defaults",
            "round_count": 2,
            "agents": ["careful-agent"],
            "generator_settings": {
                "guidance_enabled": False,
                "perturbations_enabled": False,
            },
        })
        panel.app.wait(launched["run_id"], timeout=5)
        page = panel.run_page(launched["run_id"])

        self.assertEqual(page["status"], "completed", page.get("error"))
        for plan in page["plan"]:
            self.assertNotIn("guided", plan["allowed_sources"])
            self.assertEqual(plan["round_hidden"]["guided"], 0)

    def test_fixed_case_replay_is_rejected_from_the_product_pipeline(self) -> None:
        panel = build_toy_panel()
        with self.assertRaisesRegex(ValueError, "outside the canonical pipeline"):
            panel.create_and_launch({
                "project_name": "Rejected fixed replay",
                "sampling_strategy": "fixed_recipe",
                "round_count": 2,
                "max_sampled_cases_per_agent_per_round": 2,
                "agents": ["careful-agent"],
                "fixed_recipe": {"original": 2},
                "generator_settings": {
                    "guidance_enabled": False,
                    "perturbations_enabled": False,
                },
                "round_hidden": {"base": 1},
                "terminal_hidden": {"base": 1},
                "cumulative_fixed_cases": True,
            })

    def test_fixed_recipe_does_not_replay_old_cases_by_default(self) -> None:
        from pde_agent_verifier_solver.plugins.toy import ToyResearchAgent

        class CapturingAgent(ToyResearchAgent):
            def __init__(self) -> None:
                super().__init__()
                self.original_frequencies = []

            def run(self, context):
                self.original_frequencies.append(
                    tuple(
                        case.public_data["frequency"]
                        for case in context.cases
                        if "observed_u" in case.public_data
                    )
                )
                return super().run(context)

        panel = build_toy_panel()
        agent = CapturingAgent()
        panel.programs.agents["careful-agent"] = agent
        launched = panel.create_and_launch({
            "project_name": "Non-cumulative fixed Cases",
            "sampling_strategy": "fixed_recipe",
            "round_count": 2,
            "max_sampled_cases_per_agent_per_round": 2,
            "agents": ["careful-agent"],
            "fixed_recipe": {"original": 2},
            "generator_settings": {
                "guidance_enabled": False,
                "perturbations_enabled": False,
            },
            "round_hidden": {"base": 1},
            "terminal_hidden": {"base": 1},
        })
        panel.app.wait(launched["run_id"], timeout=5)
        page = panel.run_page(launched["run_id"])

        self.assertEqual(page["status"], "completed", page.get("error"))
        self.assertEqual(set(agent.original_frequencies[0]), {1, 2})
        self.assertEqual(set(agent.original_frequencies[1]), {3, 4})

    def test_original_hidden_capacity_is_checked_before_launch(self) -> None:
        panel = build_toy_panel()
        with self.assertRaisesRegex(ValueError, "fresh original hidden Cases"):
            panel.create_and_launch({
                "project_name": "Insufficient hidden D0",
                "round_count": 5,
                "agents": ["careful-agent"],
                "problem_settings": {"hidden_cases": 4},
            })


    def test_browser_panel_contains_setup_live_and_result_sections(self) -> None:
        self.assertIn("Setup", HTML)
        self.assertIn("Live Run", HTML)
        self.assertIn("Final Result", HTML)
        self.assertIn("plugin-settings", HTML)
        self.assertIn("Agent chooses Cases", HTML)
        self.assertIn("official hidden evaluation", HTML)
        self.assertIn("CVaR", HTML)
        self.assertIn("Master seed", HTML)
        self.assertIn("Private random points", HTML)
        self.assertIn("User-fixed grid", HTML)
        self.assertIn("Agent-selected query (explicit opt-in)", HTML)
        self.assertIn("Allow grid truth on explicit request", HTML)
        self.assertIn("Terminal comparison across selected rounds", HTML)
        self.assertIn("Reproducibility seed ledger", HTML)
        self.assertNotIn("replay prior fixed Cases", HTML)
        self.assertIn("terminal_evaluation", HTML)
        self.assertNotIn("Terminal: guided", HTML)
        self.assertNotIn('<option value="guided">Guided</option>', HTML)
        self.assertNotIn("Offline pool", HTML)
        setup = build_toy_panel().setup_page()
        problem_fields = {field["name"] for field in setup["forms"]["problem"]}
        generator_fields = {field["name"] for field in setup["forms"]["generator"]}
        self.assertIn("development_cases", problem_fields)
        self.assertIn("basis_size", generator_fields)
        self.assertIn("terminal_evaluation", setup["defaults"])
        self.assertEqual(setup["defaults"]["master_seed"], 0)
        self.assertFalse(setup["defaults"]["allow_grid_truth"])
        self.assertFalse(setup["defaults"]["allow_point_truth"])
        self.assertEqual(
            setup["defaults"]["development_coordinate_mode"],
            "private_random_points",
        )
        self.assertEqual(setup["defaults"]["development_point_count"], 32)
        self.assertNotIn("cumulative_fixed_cases", setup["defaults"])



if __name__ == "__main__":
    unittest.main()
