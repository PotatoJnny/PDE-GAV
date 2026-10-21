import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from pde_agent_verifier_solver.engine.runtime import AgentOutcome, CampaignRunner
from pde_agent_verifier_solver.engine.seeds import SeedLedger
from pde_agent_verifier_solver.engine.unified_runtime import CampaignController
from pde_agent_verifier_solver.panel.controller import build_toy_panel
from pde_agent_verifier_solver.plugins.toy import (
    ToyPoissonGenerator,
    ToyResearchAgent,
)
from pde_agent_verifier_solver.sdk import (
    AnswerView,
    Artifact,
    CaseVisibility,
    EvaluationPhase,
    GenerationKind,
    GenerationPurpose,
    GeneratorCapabilities,
    PerturbCasesRequest,
    SampleCasesRequest,
    SampleSource,
    TaskKind,
)


class UnifiedRuntimeBoundaryTests(unittest.TestCase):
    def test_public_engine_surface_names_only_the_canonical_controller(self) -> None:
        from pde_agent_verifier_solver import engine

        self.assertIn("CampaignController", engine.__all__)
        noncanonical_names = (
            "CampaignRunner",
            "UnifiedCampaignRunner",
            "MacroClosedLoopRunner",
            "LocalStageRunner",
            "RunStage",
            "SlurmStageLauncher",
            "StageCommand",
            "StageResources",
            "StagedRunPlan",
            "read_staged_run",
            "record_stage_event",
        )
        for name in noncanonical_names:
            with self.subTest(name=name):
                self.assertNotIn(name, engine.__all__)
                if name not in {
                    "CampaignRunner",
                    "UnifiedCampaignRunner",
                    "MacroClosedLoopRunner",
                }:
                    self.assertFalse(hasattr(engine, name))
        self.assertIs(engine.UnifiedCampaignRunner, engine.CampaignController)

    def test_compatibility_runner_delegates_to_the_sole_controller(self) -> None:
        runner = object.__new__(CampaignRunner)
        for name in (
            "campaign",
            "dataset",
            "problem",
            "task",
            "generator",
            "agents",
            "events",
            "agent_execution",
        ):
            setattr(runner, name, SimpleNamespace(name=name))

        with patch(
            "pde_agent_verifier_solver.engine.unified_runtime.CampaignController"
        ) as controller:
            controller.return_value.run.return_value = "one-controller-result"
            with self.assertWarns(DeprecationWarning):
                result = runner.run()

        self.assertEqual(result, "one-controller-result")
        controller.assert_called_once_with(
            campaign=runner.campaign,
            dataset=runner.dataset,
            problem=runner.problem,
            task=runner.task,
            generator=runner.generator,
            agents=runner.agents,
            events=runner.events,
            agent_execution=runner.agent_execution,
            public_problem_bundle=None,
            candidate_execution=None,
        )

    def test_hidden_plan_seeds_do_not_depend_on_plugin_case_ids(self) -> None:
        controller = object.__new__(CampaignController)
        controller._seeds = SeedLedger(314159)
        first = {
            "base": (
                SimpleNamespace(case=SimpleNamespace(case_id="random-a", parent_case_id=None)),
                SimpleNamespace(case=SimpleNamespace(case_id="random-b", parent_case_id=None)),
            ),
            "base_perturbed": (
                SimpleNamespace(case=SimpleNamespace(case_id="random-c", parent_case_id=None)),
            ),
        }
        replay = {
            "base": (
                SimpleNamespace(case=SimpleNamespace(case_id="other-x", parent_case_id=None)),
                SimpleNamespace(case=SimpleNamespace(case_id="other-y", parent_case_id=None)),
            ),
            "base_perturbed": (
                SimpleNamespace(case=SimpleNamespace(case_id="other-z", parent_case_id=None)),
            ),
        }
        first_provider = controller._hidden_plan_seed_provider(
            "official_round:1", EvaluationPhase.OFFICIAL, first
        )
        replay_provider = controller._hidden_plan_seed_provider(
            "official_round:1", EvaluationPhase.OFFICIAL, replay
        )

        first_seeds = (
            first_provider(0, "random-a"),
            first_provider(1, "random-b"),
            first_provider(0, "random-c"),
        )
        replay_seeds = (
            replay_provider(0, "other-x"),
            replay_provider(1, "other-y"),
            replay_provider(0, "other-z"),
        )
        self.assertEqual(replay_seeds, first_seeds)
        self.assertEqual(len(set(first_seeds)), len(first_seeds))

    def test_paired_hidden_plans_share_root_seed_but_not_across_cohorts(self):
        controller = object.__new__(CampaignController)
        controller._seeds = SeedLedger(271828)
        controller.evaluation = SimpleNamespace(pair_hidden_plan_seeds=True)
        def row(identity, parent=None):
            return SimpleNamespace(case=SimpleNamespace(case_id=identity, parent_case_id=parent))
        hidden = {"base": (row("a"), row("b")),
                  "base_perturbed": (row("c", "a"), row("d", "b"))}
        first = controller._hidden_plan_seed_provider("official_round:1", EvaluationPhase.OFFICIAL, hidden)
        second = controller._hidden_plan_seed_provider("official_round:2", EvaluationPhase.OFFICIAL, hidden)
        self.assertEqual(first(0, "a"), first(0, "c"))
        self.assertEqual(first(1, "b"), first(1, "d"))
        self.assertNotEqual(first(0, "a"), first(1, "b"))
        self.assertNotEqual(first(0, "a"), second(0, "a"))
        hidden["base_perturbed"] = (row("c", "d"), row("d", "c"))
        with self.assertRaisesRegex(RuntimeError, "cyclic"):
            controller._hidden_plan_seed_provider("cycle", EvaluationPhase.OFFICIAL, hidden)

    def test_offline_only_generator_is_available_only_to_fixed_recipe_ablation(self) -> None:
        class OfflineLifecycleGenerator(ToyPoissonGenerator):
            capabilities = GeneratorCapabilities(
                offline=True,
                online=False,
                guidance=False,
                perturbations=False,
            )

            def __init__(self):
                super().__init__()
                self.requests = []
                self.returned_ids = set()
                self.releases = []

            def sample(self, state, request, progress=None):
                self.requests.append(request)
                records = super().sample(state, request, progress)
                self.returned_ids.update(record.case.case_id for record in records)
                return records

            def release(self, state, case_ids):
                self.releases.append((state, tuple(case_ids)))
                return super().release(state, case_ids)

        panel = build_toy_panel()
        generator = OfflineLifecycleGenerator()
        panel.programs = replace(panel.programs, generator=generator)
        launched = panel.create_and_launch({
            "project_name": "Offline compatibility and release",
            "round_count": 1,
            "agents": ["careful-agent"],
            "sampling_strategy": "fixed_recipe",
            "allowed_sources": ["base"],
            "fixed_recipe": {"original": 0, "base": 1},
            "max_sampled_cases_per_agent_per_round": 1,
            "generator_settings": {
                "guidance_enabled": False,
                "perturbations_enabled": False,
            },
            "round_hidden": {"original": 0, "base": 1, "guided": 0},
            "terminal_hidden": {"original": 0, "base": 1},
        })
        panel.app.wait(launched["run_id"], timeout=5)
        page = panel.run_page(launched["run_id"])

        self.assertEqual(page["status"], "completed", page.get("error"))
        self.assertTrue(generator.requests)
        self.assertEqual({request.mode.value for request in generator.requests}, {"offline"})
        released = [case_id for _state, ids in generator.releases for case_id in ids]
        self.assertEqual(len(generator.releases), 3)
        self.assertEqual(set(released), generator.returned_ids)
        self.assertEqual(len(released), len(set(released)))

    def test_disabled_perturbation_cannot_be_called_by_adaptive_agent(self) -> None:
        class ProbingAgent(ToyResearchAgent):
            blocked = False

            def run(self, context):
                root = context.sampler.sample_cases(
                    SampleCasesRequest(SampleSource.BASE, 1)
                ).cases[0]
                try:
                    context.sampler.perturb_cases(
                        PerturbCasesRequest((root.case_id,))
                    )
                except PermissionError:
                    self.blocked = True
                return super().run(context)

        panel = build_toy_panel()
        agent = ProbingAgent()
        panel.programs.agents["careful-agent"] = agent
        launched = panel.create_and_launch({
            "project_name": "Disabled perturbation boundary",
            "round_count": 1,
            "agents": ["careful-agent"],
            "allowed_sources": ["base"],
            "starter_cases": {"original": 0, "base": 1},
            "max_sampled_cases_per_agent_per_round": 8,
            "generator_settings": {
                "guidance_enabled": False,
                "perturbations_enabled": False,
            },
            "round_hidden": {"original": 0, "base": 1, "guided": 0},
            "terminal_hidden": {"original": 0, "base": 1},
        })
        panel.app.wait(launched["run_id"], timeout=5)
        page = panel.run_page(launched["run_id"])
        self.assertEqual(page["status"], "completed", page.get("error"))
        self.assertTrue(agent.blocked)

    def test_each_adaptive_agent_gets_an_independent_fresh_stream(self) -> None:
        class RecordingGenerator(ToyPoissonGenerator):
            def __init__(self):
                super().__init__()
                self.development_seeds = []

            def sample(self, state, request, progress=None):
                if (
                    request.visibility is CaseVisibility.DEVELOPMENT
                    and request.kind is GenerationKind.BASE
                ):
                    self.development_seeds.append(request.seed)
                return super().sample(state, request, progress)

        class StarterOnlyAgent:
            def __init__(self):
                self.first_public_data = None

            def run(self, context):
                self.first_public_data = dict(context.evaluator.cases[0].public_data)
                return AgentOutcome(
                    ToyResearchAgent._artifact(context, 1, suffix="candidate")
                )

        panel = build_toy_panel()
        generator = RecordingGenerator()
        first = StarterOnlyAgent()
        second = StarterOnlyAgent()
        panel.programs = replace(panel.programs, generator=generator)
        panel.programs.agents["agent-a"] = first
        panel.programs.agents["agent-b"] = second
        launched = panel.create_and_launch({
            "project_name": "Independent adaptive streams",
            "round_count": 1,
            "agents": ["agent-a", "agent-b"],
            "allowed_sources": ["base"],
            "starter_cases": {"original": 0, "base": 1},
            "max_sampled_cases_per_agent_per_round": 1,
            "generator_settings": {
                "guidance_enabled": False,
                "perturbations_enabled": False,
            },
            "round_hidden": {"original": 0, "base": 1, "guided": 0},
            "terminal_hidden": {"original": 0, "base": 1},
        })
        panel.app.wait(launched["run_id"], timeout=5)
        page = panel.run_page(launched["run_id"])
        self.assertEqual(page["status"], "completed", page.get("error"))
        self.assertEqual(len(generator.development_seeds), 2)
        self.assertNotEqual(
            generator.development_seeds[0],
            generator.development_seeds[1],
        )
        self.assertNotEqual(first.first_public_data, second.first_public_data)

    def test_original_development_slice_is_shared_within_round_and_fresh_across_rounds(self) -> None:
        class RecordingOriginalAgent:
            def __init__(self):
                self.case_ids_by_round = {}

            def run(self, context):
                self.case_ids_by_round[context.round_plan.round_index] = tuple(
                    case.case_id for case in context.evaluator.cases
                )
                return AgentOutcome(
                    ToyResearchAgent._artifact(context, 1, suffix="candidate")
                )

        panel = build_toy_panel()
        first = RecordingOriginalAgent()
        second = RecordingOriginalAgent()
        panel.programs.agents["agent-a"] = first
        panel.programs.agents["agent-b"] = second
        launched = panel.create_and_launch({
            "project_name": "Fresh original development slices",
            "round_count": 2,
            "agents": ["agent-a", "agent-b"],
            "allowed_sources": ["original"],
            "starter_cases": {"original": 2},
            "max_sampled_cases_per_agent_per_round": 2,
            "problem_settings": {
                "development_cases": 6,
                "hidden_cases": 4,
            },
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
        self.assertEqual(first.case_ids_by_round, second.case_ids_by_round)
        self.assertTrue(
            set(first.case_ids_by_round[1]).isdisjoint(first.case_ids_by_round[2])
        )

    def test_development_case_id_cannot_be_reused_by_a_later_agent(self) -> None:
        class ReusingDevelopmentGenerator(ToyPoissonGenerator):
            def sample(self, state, request, progress=None):
                records = super().sample(state, request, progress)
                if (
                    request.visibility is CaseVisibility.DEVELOPMENT
                    and request.kind is GenerationKind.BASE
                ):
                    return tuple(
                        replace(
                            record,
                            case=replace(
                                record.case,
                                case_id="reused-development-case",
                            ),
                        )
                        for record in records
                    )
                return records

        panel = build_toy_panel()
        panel.programs = replace(
            panel.programs,
            generator=ReusingDevelopmentGenerator(),
        )
        panel.programs.agents["agent-a"] = ToyResearchAgent()
        panel.programs.agents["agent-b"] = ToyResearchAgent()
        launched = panel.create_and_launch({
            "project_name": "Development ids are globally fresh",
            "round_count": 1,
            "agents": ["agent-a", "agent-b"],
            "allowed_sources": ["base"],
            "starter_cases": {"base": 1},
            "max_sampled_cases_per_agent_per_round": 1,
            "generator_settings": {
                "guidance_enabled": False,
                "perturbations_enabled": False,
            },
            "round_hidden": {"base": 1},
            "terminal_hidden": {"base": 1},
        })
        panel.app.wait(launched["run_id"], timeout=5)
        page = panel.run_page(launched["run_id"])

        self.assertEqual(page["status"], "failed")
        self.assertIn(
            "development Case 'reused-development-case' was already used "
            "for an earlier Development batch",
            page["error"],
        )

    def test_agent_context_cases_tracks_new_adaptive_samples(self) -> None:
        class ContextInspectingAgent:
            observed_counts = ()

            def run(self, context):
                before = len(context.cases)
                context.sampler.sample_cases(
                    SampleCasesRequest(SampleSource.BASE, 1)
                )
                after = len(context.cases)
                self.observed_counts = (before, after)
                return AgentOutcome(
                    ToyResearchAgent._artifact(context, 1, suffix="candidate")
                )

        panel = build_toy_panel()
        agent = ContextInspectingAgent()
        panel.programs.agents["context-agent"] = agent
        launched = panel.create_and_launch({
            "project_name": "Dynamic Agent context",
            "round_count": 1,
            "agents": ["context-agent"],
            "allowed_sources": ["base"],
            "starter_cases": {"base": 1},
            "max_sampled_cases_per_agent_per_round": 2,
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
        self.assertEqual(agent.observed_counts, (1, 2))

    def test_nonfinite_candidate_is_rejected_without_failing_the_round(self) -> None:
        class NonfiniteSolver:
            def solve(self, case, query, budget=None):
                del case, budget
                count = int(query.resolution["x"])
                return AnswerView(
                    kind=query.kind,
                    fields={"u": (float("nan"),) * count},
                )

        class NonfiniteAgent:
            def run(self, context):
                return AgentOutcome(Artifact(
                    artifact_id="nonfinite-candidate",
                    kind=TaskKind.SOLVER,
                    name="Nonfinite candidate",
                    program=NonfiniteSolver(),
                    summary="Deliberately invalid test candidate.",
                    usage="test only",
                    round_index=context.round_plan.round_index,
                    agent_id=context.agent.agent_id,
                    metadata={"max_resolved_mode": 0},
                ))

        panel = build_toy_panel()
        panel.programs.agents["nonfinite-agent"] = NonfiniteAgent()
        launched = panel.create_and_launch({
            "project_name": "Reject nonfinite candidate",
            "round_count": 1,
            "agents": ["careful-agent", "nonfinite-agent"],
            "allowed_sources": ["base"],
            "starter_cases": {"base": 1},
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
        reports = page["rounds"][0]["promotion"]["reports"]
        rejected = next(row for row in reports if not row["technically_eligible"])
        self.assertFalse(rejected["technically_eligible"])
        self.assertEqual(rejected["failures"], ["base:evaluation_failed"])

    def test_agent_may_return_the_incumbent_unchanged(self) -> None:
        class KeepIncumbentAgent(ToyResearchAgent):
            def run(self, context):
                if context.incumbent is not None:
                    return AgentOutcome(
                        context.incumbent,
                        observations=("Kept the incumbent after development checks.",),
                    )
                return super().run(context)

        panel = build_toy_panel()
        panel.programs.agents["keep-agent"] = KeepIncumbentAgent()
        launched = panel.create_and_launch({
            "project_name": "Keep incumbent",
            "round_count": 2,
            "agents": ["keep-agent"],
            "allowed_sources": ["base"],
            "starter_cases": {"base": 1},
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
        self.assertEqual(len(page["rounds"]), 2)
        self.assertEqual(
            page["rounds"][0]["selected_artifact_id"],
            page["rounds"][1]["selected_artifact_id"],
        )

    def test_terminal_rejects_case_reused_from_any_promotion_round(self) -> None:
        class ReusingOfficialGenerator(ToyPoissonGenerator):
            def __init__(self):
                super().__init__()
                self.returned_ids = set()
                self.releases = []

            def sample(self, state, request, progress=None):
                records = super().sample(state, request, progress)
                if (
                    request.visibility is not CaseVisibility.HIDDEN
                    or request.kind is not GenerationKind.BASE
                ):
                    self.returned_ids.update(record.case.case_id for record in records)
                    return records
                source_round = (
                    1
                    if request.purpose is GenerationPurpose.TERMINAL
                    else request.round_index
                )
                rewritten = tuple(
                    replace(
                        record,
                        case=replace(
                            record.case,
                            case_id=f"official-round-{source_round}-base-{index}",
                        ),
                    )
                    for index, record in enumerate(records)
                )
                self.returned_ids.update(record.case.case_id for record in rewritten)
                return rewritten

            def release(self, state, case_ids):
                self.releases.append((state, tuple(case_ids)))
                return super().release(state, case_ids)

        panel = build_toy_panel()
        generator = ReusingOfficialGenerator()
        panel.programs = replace(
            panel.programs,
            generator=generator,
        )
        launched = panel.create_and_launch({
            "project_name": "All-round hidden overlap boundary",
            "round_count": 2,
            "agents": ["careful-agent"],
            "allowed_sources": ["base"],
            "starter_cases": {"original": 0, "base": 1},
            "generator_settings": {
                "guidance_enabled": False,
                "perturbations_enabled": False,
            },
            "round_hidden": {"original": 0, "base": 1, "guided": 0},
            "terminal_hidden": {"original": 0, "base": 1},
        })
        panel.app.wait(launched["run_id"], timeout=5)
        page = panel.run_page(launched["run_id"])
        self.assertEqual(page["status"], "failed")
        self.assertIn("already used for Terminal", page["error"])
        released = [case_id for _state, ids in generator.releases for case_id in ids]
        self.assertEqual(set(released), generator.returned_ids)
        # The deliberately invalid Generator allocated the reused identifier a
        # second time.  Both allocations must be cleaned up even though the
        # freshness ledger rejects the second one.
        reused = "official-round-1-base-0"
        self.assertEqual(released.count(reused), 2)
        self.assertTrue(
            all(released.count(case_id) == 1 for case_id in set(released) - {reused})
        )

    def test_each_generated_cohort_releases_through_its_owning_state(self) -> None:
        class StateOwningGenerator(ToyPoissonGenerator):
            def __init__(self):
                super().__init__()
                self.owners = {}
                self.releases = []

            def sample(self, state, request, progress=None):
                records = super().sample(state, request, progress)
                for record in records:
                    self.owners[record.case.case_id] = id(state)
                return records

            def release(self, state, case_ids):
                state_id = id(state)
                wrong = [
                    case_id
                    for case_id in case_ids
                    if self.owners[case_id] != state_id
                ]
                if wrong:
                    raise AssertionError(
                        f"Cases released through a non-owning Generator state: {wrong}"
                    )
                self.releases.append((state_id, tuple(case_ids)))
                return super().release(state, case_ids)

        panel = build_toy_panel()
        generator = StateOwningGenerator()
        panel.programs = replace(panel.programs, generator=generator)
        launched = panel.create_and_launch({
            "project_name": "State-owned Case lifecycle",
            "round_count": 2,
            "agents": ["careful-agent"],
            "allowed_sources": ["base", "guided"],
            "starter_cases": {"original": 0, "base": 1, "guided": 0},
            "generator_settings": {
                "guidance_enabled": True,
                "perturbations_enabled": False,
            },
            "round_hidden": {"original": 0, "base": 1, "guided": 0},
            "terminal_hidden": {"original": 0, "base": 1},
        })
        panel.app.wait(launched["run_id"], timeout=5)
        page = panel.run_page(launched["run_id"])

        self.assertEqual(page["status"], "completed", page.get("error"))
        released = {
            case_id
            for _state_id, case_ids in generator.releases
            for case_id in case_ids
        }
        self.assertEqual(released, set(generator.owners))
        self.assertGreaterEqual(
            len({state_id for state_id, _case_ids in generator.releases}),
            2,
        )

    def test_agent_exposed_case_cannot_reuse_a_terminal_case_id(self) -> None:
        class ReusingDevelopmentGenerator(ToyPoissonGenerator):
            def sample(self, state, request, progress=None):
                records = super().sample(state, request, progress)
                if request.kind is not GenerationKind.BASE:
                    return records
                if request.purpose not in {
                    GenerationPurpose.DEVELOPMENT,
                    GenerationPurpose.TERMINAL,
                }:
                    return records
                return tuple(
                    replace(
                        record,
                        case=replace(record.case, case_id=f"shared-case-{index}"),
                    )
                    for index, record in enumerate(records)
                )

        panel = build_toy_panel()
        panel.programs = replace(
            panel.programs,
            generator=ReusingDevelopmentGenerator(),
        )
        launched = panel.create_and_launch({
            "project_name": "Development private-cohort isolation",
            "round_count": 1,
            "agents": ["careful-agent"],
            "allowed_sources": ["base"],
            "starter_cases": {"base": 1},
            "max_sampled_cases_per_agent_per_round": 1,
            "generator_settings": {
                "guidance_enabled": False,
                "perturbations_enabled": False,
            },
            "round_hidden": {"base": 1},
            "terminal_hidden": {"base": 1},
        })
        panel.app.wait(launched["run_id"], timeout=5)
        page = panel.run_page(launched["run_id"])

        self.assertEqual(page["status"], "failed")
        self.assertIn("development Case 'shared-case-0'", page["error"])
        self.assertIn("reserved for Terminal", page["error"])

    def test_terminal_generator_cases_do_not_depend_on_round_count(self) -> None:
        def terminal_fingerprint(round_count):
            class RecordingGenerator(ToyPoissonGenerator):
                terminal = ()

                def sample(self, state, request, progress=None):
                    records = super().sample(state, request, progress)
                    if request.purpose is GenerationPurpose.TERMINAL:
                        self.terminal = tuple(
                            (
                                record.case.public_data["frequency"],
                                record.case.public_data["amplitude"],
                            )
                            for record in records
                        )
                    return records

            panel = build_toy_panel()
            generator = RecordingGenerator()
            panel.programs = replace(panel.programs, generator=generator)
            launched = panel.create_and_launch({
                "project_name": f"Common terminal {round_count}",
                "round_count": round_count,
                "agents": ["careful-agent"],
                "allowed_sources": ["base"],
                "starter_cases": {"base": 1},
                "generator_settings": {
                    "guidance_enabled": False,
                    "perturbations_enabled": False,
                },
                "round_hidden": {"base": 1},
                "terminal_hidden": {"base": 2},
            })
            panel.app.wait(launched["run_id"], timeout=5)
            page = panel.run_page(launched["run_id"])
            self.assertEqual(page["status"], "completed", page.get("error"))
            return generator.terminal

        self.assertEqual(terminal_fingerprint(1), terminal_fingerprint(3))


if __name__ == "__main__":
    unittest.main()
