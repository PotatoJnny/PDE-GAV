import unittest
from dataclasses import replace

from pde_agent_verifier_solver.engine.campaign import (
    AgentSpec,
    CampaignCompiler,
    CampaignRecipe,
    CaseCounts,
    DevelopmentSamplingPolicy,
    RoundOverride,
    EvaluatorPolicy,
    GeneratorRecipe,
    RoundSettings,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    Artifact,
    GeneratorCapabilities,
    GeneratorCheckpoint,
    HiddenSuiteSpec,
    OfficialEvaluationSpec,
    SampleSource,
    SamplingStrategy,
    TaskKind,
    TerminalCohortSpec,
    TerminalEvaluationSpec,
    TerminalGuidanceSource,
    TerminalMetricSpec,
)


class CampaignCompilerTests(unittest.TestCase):
    def test_adaptive_mainline_rejects_case_replay(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the canonical pipeline"):
            DevelopmentSamplingPolicy(cumulative_fixed_cases=True)

    def test_starter_only_source_is_preserved_in_the_compiled_plan(self) -> None:
        recipe = self.recipe()
        policy = replace(
            recipe.default_round.development_sampling,
            starter_only_sources=(SampleSource.ORIGINAL,),
        )
        recipe = replace(
            recipe,
            default_round=replace(
                recipe.default_round,
                development_sampling=policy,
            ),
        )

        compiled = CampaignCompiler().compile(
            recipe,
            GeneratorCapabilities(online=True),
        )

        self.assertEqual(
            compiled.rounds[0].development_sampling.starter_only_sources,
            (SampleSource.ORIGINAL,),
        )

    def test_starter_only_source_requires_a_positive_starter_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive starter counts"):
            DevelopmentSamplingPolicy(
                allowed_sources=(SampleSource.ORIGINAL, SampleSource.BASE),
                starter=CaseCounts(base=1),
                starter_only_sources=(SampleSource.ORIGINAL,),
            )

    def test_starter_only_source_must_be_allowed(self) -> None:
        with self.assertRaisesRegex(ValueError, "also appear in allowed_sources"):
            DevelopmentSamplingPolicy(
                allowed_sources=(SampleSource.BASE,),
                starter=CaseCounts(original=1, base=1),
                starter_only_sources=(SampleSource.ORIGINAL,),
            )

    def test_starter_only_source_must_leave_an_adaptive_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one adaptive source"):
            DevelopmentSamplingPolicy(
                allowed_sources=(SampleSource.ORIGINAL,),
                starter=CaseCounts(original=1),
                starter_only_sources=(SampleSource.ORIGINAL,),
            )

    def test_fixed_recipe_cannot_use_starter_only_sources(self) -> None:
        with self.assertRaisesRegex(ValueError, "only to agent_adaptive"):
            DevelopmentSamplingPolicy(
                strategy=SamplingStrategy.FIXED_RECIPE,
                allowed_sources=(SampleSource.ORIGINAL,),
                starter=CaseCounts(original=1),
                fixed_recipe=CaseCounts(original=1),
                starter_only_sources=(SampleSource.ORIGINAL,),
            )

    def recipe(self) -> CampaignRecipe:
        policy = DevelopmentSamplingPolicy(
            strategy=SamplingStrategy.AGENT_ADAPTIVE,
            max_cases_per_agent_per_round=20,
            allowed_sources=(SampleSource.ORIGINAL, SampleSource.BASE),
            starter=CaseCounts(original=2, base=2),
        )
        return CampaignRecipe(
            name="compiler test",
            round_count=2,
            agents=(AgentSpec("agent", "Agent"),),
            default_round=RoundSettings(
                agent_budget_seconds=10,
                development_sampling=policy,
                official_evaluation=OfficialEvaluationSpec(
                    round=HiddenSuiteSpec(base=3),
                    terminal=HiddenSuiteSpec(base=3),
                ),
            ),
            generator=GeneratorRecipe(
                guidance_enabled=False,
                perturbations_enabled=False,
            ),
            evaluator=EvaluatorPolicy(AnswerQuery.grid(fields=("u",), x=33)),
        )

    def test_rejects_the_legacy_case_mix_pipeline(self) -> None:
        recipe = self.recipe()
        legacy = CampaignRecipe(
            **{
                **recipe.__dict__,
                "default_round": RoundSettings(),
            }
        )
        with self.assertRaisesRegex(ValueError, "unique Campaign pipeline"):
            CampaignCompiler().compile(legacy, GeneratorCapabilities())

    def test_compiled_campaign_carries_the_master_seed(self) -> None:
        recipe = CampaignRecipe(**{**self.recipe().__dict__, "master_seed": 7391})
        compiled = CampaignCompiler().compile(
            recipe,
            GeneratorCapabilities(online=True),
        )
        self.assertEqual(compiled.master_seed, 7391)

    def test_adaptive_mainline_requires_an_online_generator(self) -> None:
        with self.assertRaisesRegex(ValueError, "fresh online mainline"):
            CampaignCompiler().compile(self.recipe(), GeneratorCapabilities())

    def test_frozen_generator_requires_an_explicit_checkpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires an existing checkpoint"):
            CampaignCompiler().compile(
                self.recipe(),
                GeneratorCapabilities(online=True, trainable=False),
            )

    def test_frozen_generator_accepts_an_explicit_checkpoint(self) -> None:
        recipe = CampaignRecipe(
            **{
                **self.recipe().__dict__,
                "generator": GeneratorRecipe(
                    guidance_enabled=False,
                    perturbations_enabled=False,
                    checkpoint=GeneratorCheckpoint(
                        generator_id="frozen-generator",
                        payload={"model": "already-loaded"},
                    ),
                ),
            }
        )

        compiled = CampaignCompiler().compile(
            recipe,
            GeneratorCapabilities(online=True, trainable=False),
        )

        self.assertIs(compiled.generator.checkpoint, recipe.generator.checkpoint)

    def test_initial_candidate_allows_round_one_guidance(self) -> None:
        initial = Artifact(
            artifact_id="initial-solver",
            kind=TaskKind.SOLVER,
            name="Initial solver",
            program=object(),
            summary="User-provided starting point",
            usage="solver.solve(case, query)",
            round_index=0,
            agent_id="user",
        )
        policy = DevelopmentSamplingPolicy(
            max_cases_per_agent_per_round=2,
            allowed_sources=(SampleSource.GUIDED,),
            starter=CaseCounts(guided=1),
        )
        terminal = HiddenSuiteSpec(base=1)
        recipe = CampaignRecipe(
            name="guided first round",
            round_count=1,
            agents=(AgentSpec("agent", "Agent"),),
            default_round=RoundSettings(
                development_sampling=policy,
                official_evaluation=OfficialEvaluationSpec(
                    round=HiddenSuiteSpec(base=1, guided=1),
                    terminal=terminal,
                ),
            ),
            generator=GeneratorRecipe(perturbations_enabled=False),
            evaluator=EvaluatorPolicy(AnswerQuery.grid(fields=("u",), x=9)),
            initial_candidate=initial,
        )

        compiled = CampaignCompiler().compile(
            recipe,
            GeneratorCapabilities(online=True, guidance=True),
        )

        self.assertEqual(compiled.initial_candidate, initial)
        self.assertIn(
            SampleSource.GUIDED,
            compiled.rounds[0].development_sampling.allowed_sources,
        )

    def test_terminal_is_one_campaign_level_cross_round_specification(self) -> None:
        terminal = TerminalEvaluationSpec(
            cohorts=(
                TerminalCohortSpec(
                    cohort_id="base_model",
                    suite=HiddenSuiteSpec(base=2, base_perturbed=1),
                ),
                TerminalCohortSpec(
                    cohort_id="round_2_guided",
                    suite=HiddenSuiteSpec(guided=2),
                    guidance_source=TerminalGuidanceSource.ROUND,
                    round_index=2,
                ),
                TerminalCohortSpec(
                    cohort_id="final_guided",
                    suite=HiddenSuiteSpec(guided=2, guided_perturbed=1),
                    guidance_source=TerminalGuidanceSource.FINAL,
                ),
            ),
            ranking=TerminalMetricSpec(
                cohort_id="base_model",
                distribution="base",
            ),
        )
        recipe = CampaignRecipe(
            **{
                **self.recipe().__dict__,
                "generator": GeneratorRecipe(),
                "terminal_evaluation": terminal,
                "default_round": RoundSettings(
                    development_sampling=self.recipe().default_round.development_sampling,
                    official_evaluation=OfficialEvaluationSpec(
                        round=HiddenSuiteSpec(base=2),
                    ),
                ),
            }
        )

        compiled = CampaignCompiler().compile(
            recipe,
            GeneratorCapabilities(
                online=True,
                guidance=True,
                perturbations=True,
            ),
        )

        self.assertEqual(compiled.terminal_evaluation, terminal)
        self.assertEqual(compiled.terminal_score_distribution, "base")



    def test_unified_sampling_compiles_agent_choice_and_fresh_hidden_types(self) -> None:
        policy = DevelopmentSamplingPolicy(
            strategy=SamplingStrategy.AGENT_ADAPTIVE,
            max_cases_per_agent_per_round=8,
            allowed_sources=(
                SampleSource.ORIGINAL,
                SampleSource.BASE,
                SampleSource.GUIDED,
            ),
            starter=CaseCounts(original=2, base=2),
        )
        terminal = HiddenSuiteSpec(original=1, base=3)
        recipe = CampaignRecipe(
            name="unified",
            round_count=2,
            agents=(AgentSpec("agent", "Agent"),),
            default_round=RoundSettings(
                agent_budget_seconds=10,
                development_sampling=policy,
                official_evaluation=OfficialEvaluationSpec(
                    round=HiddenSuiteSpec(original=1, base=2, guided=1),
                    terminal=terminal,
                ),
            ),
            round_overrides={
                1: RoundOverride(
                    official_evaluation=OfficialEvaluationSpec(
                        round=HiddenSuiteSpec(original=1, base=3),
                        terminal=terminal,
                    )
                )
            },
            generator=GeneratorRecipe(perturbations_enabled=False),
            evaluator=EvaluatorPolicy(AnswerQuery.grid(fields=("u",), x=33)),
        )

        compiled = CampaignCompiler().compile(
            recipe,
            GeneratorCapabilities(online=True, guidance=True),
        )

        first, second = compiled.rounds
        self.assertEqual(first.development_count, 8)
        self.assertNotIn(SampleSource.GUIDED, first.development_sampling.allowed_sources)
        self.assertIn(SampleSource.GUIDED, second.development_sampling.allowed_sources)
        self.assertEqual(first.round_hidden.as_dict(), {
            "original": 1,
            "base": 3,
            "guided": 0,
            "base_perturbed": 0,
            "guided_perturbed": 0,
        })
        self.assertEqual(second.round_hidden.guided, 1)
        self.assertEqual(second.terminal_hidden, terminal)

    def test_fixed_recipe_must_fit_the_single_sample_limit(self) -> None:
        policy = DevelopmentSamplingPolicy(
            strategy=SamplingStrategy.FIXED_RECIPE,
            max_cases_per_agent_per_round=3,
            allowed_sources=(SampleSource.ORIGINAL, SampleSource.BASE),
            fixed_recipe=CaseCounts(original=2, base=2),
        )
        recipe = CampaignRecipe(
            name="fixed",
            round_count=1,
            agents=(AgentSpec("agent", "Agent"),),
            default_round=RoundSettings(
                development_sampling=policy,
                official_evaluation=OfficialEvaluationSpec(
                    round=HiddenSuiteSpec(base=1),
                    terminal=HiddenSuiteSpec(base=1),
                ),
            ),
            generator=GeneratorRecipe(
                guidance_enabled=False,
                perturbations_enabled=False,
            ),
            evaluator=EvaluatorPolicy(AnswerQuery.grid(fields=("u",), x=33)),
        )
        with self.assertRaisesRegex(ValueError, "exceed"):
            CampaignCompiler().compile(recipe, GeneratorCapabilities())

    def test_terminal_score_is_independent_of_round_promotion_metric(self) -> None:
        policy = DevelopmentSamplingPolicy(
            strategy=SamplingStrategy.AGENT_ADAPTIVE,
            max_cases_per_agent_per_round=2,
            allowed_sources=(SampleSource.BASE,),
            starter=CaseCounts(base=1),
        )
        recipe = CampaignRecipe(
            name="separate terminal score",
            round_count=1,
            agents=(AgentSpec("agent", "Agent"),),
            default_round=RoundSettings(
                development_sampling=policy,
                official_evaluation=OfficialEvaluationSpec(
                    round=HiddenSuiteSpec(base=1),
                    terminal=HiddenSuiteSpec(original=1),
                ),
            ),
            generator=GeneratorRecipe(
                guidance_enabled=False,
                perturbations_enabled=False,
            ),
            evaluator=EvaluatorPolicy(AnswerQuery.grid(fields=("u",), x=33)),
            promotion={
                "preset": "minimize_metric",
                "settings": {
                    "primary": {"distribution": "base", "statistic": "mean"}
                },
            },
            terminal_score_distribution="original",
        )

        compiled = CampaignCompiler().compile(
            recipe,
            GeneratorCapabilities(online=True),
        )

        self.assertEqual(compiled.terminal_score_distribution, "original")

    def test_terminal_score_rejects_solver_dependent_distribution(self) -> None:
        recipe = self.recipe()
        recipe = CampaignRecipe(
            **{
                **recipe.__dict__,
                "terminal_score_distribution": "guided",
            }
        )
        with self.assertRaisesRegex(ValueError, "terminal score distribution"):
            CampaignCompiler().compile(recipe, GeneratorCapabilities())

    def test_rejects_conflicting_round_terminal_suites(self) -> None:
        policy = DevelopmentSamplingPolicy(
            strategy=SamplingStrategy.AGENT_ADAPTIVE,
            max_cases_per_agent_per_round=2,
            allowed_sources=(SampleSource.BASE,),
            starter=CaseCounts(base=1),
        )
        first_terminal = HiddenSuiteSpec(base=1)
        recipe = CampaignRecipe(
            name="conflicting terminal suites",
            round_count=2,
            agents=(AgentSpec("agent", "Agent"),),
            default_round=RoundSettings(
                development_sampling=policy,
                official_evaluation=OfficialEvaluationSpec(
                    round=HiddenSuiteSpec(base=1),
                    terminal=first_terminal,
                ),
            ),
            round_overrides={
                2: RoundOverride(
                    official_evaluation=OfficialEvaluationSpec(
                        round=HiddenSuiteSpec(base=1),
                        terminal=HiddenSuiteSpec(base=2),
                    )
                )
            },
            generator=GeneratorRecipe(
                guidance_enabled=False,
                perturbations_enabled=False,
            ),
            evaluator=EvaluatorPolicy(AnswerQuery.grid(fields=("u",), x=33)),
        )

        with self.assertRaisesRegex(ValueError, "terminal hidden suite"):
            CampaignCompiler().compile(
                recipe,
                GeneratorCapabilities(online=True),
            )


if __name__ == "__main__":
    unittest.main()
