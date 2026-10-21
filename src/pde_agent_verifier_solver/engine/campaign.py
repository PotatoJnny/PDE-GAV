"""Compile a customer Run recipe into explicit per-round work."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    Artifact,
    DevelopmentCoordinateMode,
    GeneratorCapabilities,
    GeneratorCheckpoint,
)
from pde_agent_verifier_solver.sdk.official_evaluation import (
    HiddenSuiteSpec,
    OfficialEvaluationSpec,
    TerminalEvaluationSpec,
    TerminalGuidanceSource,
    terminal_from_legacy_suite,
)
from pde_agent_verifier_solver.sdk.sampling import (
    SampleSource,
    SamplingStrategy,
)

from .promotion import promotion_metric_references


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    name: str
    prompt: str = "Improve the current scientific program using the available Evaluator."


@dataclass(frozen=True)
class CaseCounts:
    """Exact counts used for starter Cases or a fixed-recipe ablation."""

    original: int = 0
    base: int = 0
    guided: int = 0
    base_perturbed: int = 0
    guided_perturbed: int = 0

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} Case count must be a nonnegative integer")

    @property
    def total(self) -> int:
        return sum(self.as_dict().values())

    def as_dict(self) -> dict[str, int]:
        return {
            "original": self.original,
            "base": self.base,
            "guided": self.guided,
            "base_perturbed": self.base_perturbed,
            "guided_perturbed": self.guided_perturbed,
        }


@dataclass(frozen=True)
class DevelopmentSamplingPolicy:
    """Who chooses development Cases and the one Case limit they share.

    ``starter_only_sources`` lets the user place a finite source in the
    Controller-provided starter set without leaving that source available for
    additional adaptive draws. Starter Cases still spend the ordinary total
    Case allowance; this is an access boundary, not a second budget.
    """

    strategy: SamplingStrategy = SamplingStrategy.AGENT_ADAPTIVE
    # None explicitly removes the cumulative Case quota; time and tool
    # permissions still constrain the Agent round.
    max_cases_per_agent_per_round: int | None = 24
    allowed_sources: tuple[SampleSource, ...] = (
        SampleSource.ORIGINAL,
        SampleSource.BASE,
        SampleSource.GUIDED,
    )
    starter: CaseCounts = field(default_factory=CaseCounts)
    fixed_recipe: CaseCounts | None = None
    hide_case_types: bool = True
    cumulative_fixed_cases: bool = False
    starter_only_sources: tuple[SampleSource, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, SamplingStrategy):
            raise TypeError("sampling strategy must be a SamplingStrategy")
        if self.max_cases_per_agent_per_round is not None and (
            isinstance(self.max_cases_per_agent_per_round, bool)
            or not isinstance(self.max_cases_per_agent_per_round, int)
            or self.max_cases_per_agent_per_round < 1
        ):
            raise ValueError("per-Agent round sample limit must be positive")
        sources = tuple(self.allowed_sources)
        if not sources or any(not isinstance(value, SampleSource) for value in sources):
            raise TypeError("allowed_sources must contain SampleSource values")
        if len(sources) != len(set(sources)):
            raise ValueError("allowed_sources cannot repeat")
        starter_only = tuple(self.starter_only_sources)
        if any(not isinstance(value, SampleSource) for value in starter_only):
            raise TypeError(
                "starter_only_sources must contain SampleSource values"
            )
        if len(starter_only) != len(set(starter_only)):
            raise ValueError("starter_only_sources cannot repeat")
        unavailable = set(starter_only) - set(sources)
        if unavailable:
            raise ValueError(
                "starter_only_sources must also appear in allowed_sources"
            )
        if starter_only and self.strategy is not SamplingStrategy.AGENT_ADAPTIVE:
            raise ValueError(
                "starter_only_sources apply only to agent_adaptive sampling"
            )
        if starter_only and set(starter_only) == set(sources):
            raise ValueError(
                "starter_only_sources must leave at least one adaptive source"
            )
        starter_counts = self.starter.as_dict()
        missing_starter = tuple(
            source.value
            for source in starter_only
            if starter_counts[source.value] == 0
        )
        if missing_starter:
            raise ValueError(
                "starter_only_sources need positive starter counts: "
                f"{list(missing_starter)}"
            )
        if self.cumulative_fixed_cases:
            raise ValueError(
                "Case replay is outside the canonical pipeline; use an explicit "
                "legacy/control-arm runner"
            )
        object.__setattr__(self, "allowed_sources", sources)
        object.__setattr__(self, "starter_only_sources", starter_only)


@dataclass(frozen=True)
class RoundSettings:
    agent_budget_seconds: int = 300
    development_sampling: DevelopmentSamplingPolicy | None = None
    official_evaluation: OfficialEvaluationSpec | None = None


@dataclass(frozen=True)
class RoundOverride:
    agent_budget_seconds: int | None = None
    development_sampling: DevelopmentSamplingPolicy | None = None
    official_evaluation: OfficialEvaluationSpec | None = None


@dataclass(frozen=True)
class GeneratorRecipe:
    basis_size: int = 8
    guidance_enabled: bool = True
    perturbations_enabled: bool = True
    settings: Mapping[str, object] = field(default_factory=dict)
    checkpoint: GeneratorCheckpoint | None = None

    def __post_init__(self) -> None:
        if self.checkpoint is not None and not isinstance(
            self.checkpoint,
            GeneratorCheckpoint,
        ):
            raise TypeError("GeneratorRecipe.checkpoint must be a GeneratorCheckpoint")


@dataclass(frozen=True)
class EvaluatorPolicy:
    default_query: AnswerQuery
    allow_grid_truth: bool = False
    allow_point_truth: bool = False
    allow_development_solution_session: bool = False
    development_coordinate_mode: DevelopmentCoordinateMode = (
        DevelopmentCoordinateMode.PRIVATE_RANDOM_POINTS
    )
    development_point_count: int = 32

    def __post_init__(self) -> None:
        if not isinstance(self.default_query, AnswerQuery):
            raise TypeError("Evaluator default_query must be an AnswerQuery")
        object.__setattr__(
            self,
            "default_query",
            AnswerQuery(
                kind=self.default_query.kind,
                fields=tuple(self.default_query.fields),
                resolution=MappingProxyType(dict(self.default_query.resolution)),
                points=tuple(
                    MappingProxyType(dict(point))
                    for point in self.default_query.points
                ),
            ),
        )
        if not isinstance(self.allow_grid_truth, bool):
            raise TypeError("allow_grid_truth must be boolean")
        if not isinstance(self.allow_point_truth, bool):
            raise TypeError("allow_point_truth must be boolean")
        if not isinstance(self.allow_development_solution_session, bool):
            raise TypeError(
                "allow_development_solution_session must be boolean"
            )
        if not isinstance(self.development_coordinate_mode, DevelopmentCoordinateMode):
            raise TypeError(
                "development_coordinate_mode must be a DevelopmentCoordinateMode"
            )
        if (
            isinstance(self.development_point_count, bool)
            or not isinstance(self.development_point_count, int)
            or self.development_point_count < 1
        ):
            raise ValueError("development_point_count must be a positive integer")


@dataclass(frozen=True)
class CampaignRecipe:
    name: str
    round_count: int
    agents: tuple[AgentSpec, ...]
    default_round: RoundSettings
    generator: GeneratorRecipe
    evaluator: EvaluatorPolicy
    round_overrides: Mapping[int, RoundOverride] = field(default_factory=dict)
    task_settings: Mapping[str, object] = field(default_factory=dict)
    master_seed: int = 0
    initial_candidate: Artifact | None = None
    terminal_evaluation: TerminalEvaluationSpec | None = None

    promotion: Mapping[str, object] = field(
        default_factory=lambda: {
            "preset": "minimize_metric",
            "settings": {"primary": {"distribution": "base", "statistic": "mean"}},
        }
    )
    # Promotion chooses the next-round incumbent. The terminal score is a
    # separate, common final result and therefore has its own distribution.
    terminal_score_distribution: str = "base"


@dataclass(frozen=True)
class RoundPlan:
    round_index: int
    agents: tuple[AgentSpec, ...]
    agent_budget_seconds: int | None
    development_sampling: DevelopmentSamplingPolicy
    round_hidden: HiddenSuiteSpec
    terminal_hidden: HiddenSuiteSpec | None = None

    @property
    def development_count(self) -> int | None:
        policy = self.development_sampling
        if policy.strategy is SamplingStrategy.FIXED_RECIPE:
            if policy.fixed_recipe is None:
                return 0
            return policy.fixed_recipe.total
        return policy.max_cases_per_agent_per_round

    @property
    def original_count(self) -> int:
        """Original count for the explicit fixed-recipe ablation only."""

        policy = self.development_sampling
        if policy.strategy is not SamplingStrategy.FIXED_RECIPE:
            return 0
        return policy.fixed_recipe.original if policy.fixed_recipe is not None else 0

    @property
    def hidden_generated_count(self) -> int:
        """Generated Case count in the one fresh Official cohort."""

        return (
            self.round_hidden.base
            + self.round_hidden.guided
            + self.round_hidden.base_perturbed
            + self.round_hidden.guided_perturbed
        )

    def case_mix_summary(self) -> Mapping[str, int | None]:
        policy = self.development_sampling
        counts = (
            policy.fixed_recipe
            if policy.strategy is SamplingStrategy.FIXED_RECIPE
            else policy.starter
        )
        if counts is None:
            counts = CaseCounts()
        return {
            **counts.as_dict(),
            "sample_limit": policy.max_cases_per_agent_per_round,
        }


@dataclass(frozen=True)
class CompiledCampaign:
    name: str
    rounds: tuple[RoundPlan, ...]
    generator: GeneratorRecipe
    evaluator: EvaluatorPolicy
    task_settings: Mapping[str, object]

    promotion: Mapping[str, object] = field(default_factory=dict)
    terminal_score_distribution: str = "base"
    terminal_evaluation: TerminalEvaluationSpec | None = None
    master_seed: int = 0
    initial_candidate: Artifact | None = None

class CampaignCompiler:
    """Validate product settings and make every round count explicit."""

    def compile(
        self,
        recipe: CampaignRecipe,
        capabilities: GeneratorCapabilities,
    ) -> CompiledCampaign:
        self._validate_recipe(recipe)
        self._validate_generator(recipe, capabilities)

        plans = tuple(
            self._compile_round(recipe, round_index)
            for round_index in range(1, recipe.round_count + 1)
        )
        terminal_evaluation = self._resolve_terminal_evaluation(recipe, plans)
        self._validate_promotion_suites(
            recipe.promotion,
            plans,
        )
        self._validate_terminal_evaluation(recipe, terminal_evaluation)
        return CompiledCampaign(
            name=recipe.name,
            rounds=plans,
            generator=recipe.generator,
            evaluator=recipe.evaluator,
            task_settings=recipe.task_settings,
            promotion=recipe.promotion,
            terminal_score_distribution=(
                terminal_evaluation.ranking.distribution
            ),
            terminal_evaluation=terminal_evaluation,
            master_seed=recipe.master_seed,
            initial_candidate=recipe.initial_candidate,
        )

    @staticmethod
    def _resolve_terminal_evaluation(
        recipe: CampaignRecipe,
        plans: tuple[RoundPlan, ...],
    ) -> TerminalEvaluationSpec:
        """Resolve the former per-round field into the Campaign-level contract.

        Historical recipes remain readable during migration, but the compiled
        state contains exactly one terminal specification.
        """

        suites = {
            plan.terminal_hidden
            for plan in plans
            if plan.terminal_hidden is not None
        }
        if len(suites) > 1:
            raise ValueError(
                "terminal hidden suite must be identical in every round"
            )
        legacy = next(iter(suites), None)
        if recipe.terminal_evaluation is not None:
            if legacy is not None:
                raise ValueError(
                    "declare terminal evaluation at Campaign scope; do not also "
                    "set per-round terminal hidden"
                )
            return recipe.terminal_evaluation
        if legacy is None:
            raise ValueError("Campaign needs a terminal_evaluation specification")
        return terminal_from_legacy_suite(
            legacy,
            score_distribution=recipe.terminal_score_distribution,
        )

    def _compile_round(self, recipe: CampaignRecipe, round_index: int) -> RoundPlan:
        settings = self._settings_for_round(recipe, round_index)
        if settings.development_sampling is None:
            raise ValueError(
                "the unique Campaign pipeline requires development_sampling; "
                "legacy fraction/replay recipes are history-only"
            )
        return self._compile_sampling_round(recipe, round_index, settings)

    def _compile_sampling_round(
        self,
        recipe: CampaignRecipe,
        round_index: int,
        settings: RoundSettings,
    ) -> RoundPlan:
        policy = settings.development_sampling
        official = settings.official_evaluation
        if policy is None or official is None:
            raise ValueError("unified sampling needs development and official settings")
        if (
            round_index == 1
            and recipe.initial_candidate is None
            and SampleSource.GUIDED in policy.allowed_sources
        ):
            policy = replace(
                policy,
                allowed_sources=tuple(
                    source
                    for source in policy.allowed_sources
                    if source is not SampleSource.GUIDED
                ),
            )
        counts = (
            policy.fixed_recipe
            if policy.strategy is SamplingStrategy.FIXED_RECIPE
            else policy.starter
        )
        counts = counts or CaseCounts()
        return RoundPlan(
            round_index=round_index,
            agents=recipe.agents,
            agent_budget_seconds=settings.agent_budget_seconds,
            development_sampling=policy,
            round_hidden=official.round,
            terminal_hidden=official.terminal,
        )

    @staticmethod
    def _settings_for_round(recipe: CampaignRecipe, round_index: int) -> RoundSettings:
        override = recipe.round_overrides.get(round_index)
        if override is None:
            return recipe.default_round
        changes = {
            name: value
            for name, value in vars(override).items()
            if value is not None
        }
        return replace(recipe.default_round, **changes)

    def _validate_recipe(self, recipe: CampaignRecipe) -> None:
        if not recipe.name.strip():
            raise ValueError("campaign name cannot be empty")
        if recipe.round_count <= 0:
            raise ValueError("round_count must be positive")
        if not recipe.agents:
            raise ValueError("at least one Agent is required")
        if len({agent.agent_id for agent in recipe.agents}) != len(recipe.agents):
            raise ValueError("Agent ids must be unique")
        unknown_rounds = set(recipe.round_overrides) - set(
            range(1, recipe.round_count + 1)
        )
        if unknown_rounds:
            raise ValueError(
                f"round overrides are out of range: {sorted(unknown_rounds)}"
            )

        if any(
            self._settings_for_round(recipe, index).development_sampling is None
            for index in range(1, recipe.round_count + 1)
        ):
            raise ValueError(
                "the unique Campaign pipeline requires development_sampling "
                "and official_evaluation in every round"
            )
        if (
            isinstance(recipe.master_seed, bool)
            or not isinstance(recipe.master_seed, int)
            or recipe.master_seed < 0
        ):
            raise ValueError("master_seed must be a nonnegative integer")
        if recipe.initial_candidate is not None:
            if not isinstance(recipe.initial_candidate, Artifact):
                raise TypeError("initial_candidate must be an Artifact")
            if recipe.initial_candidate.round_index != 0:
                raise ValueError("initial_candidate must use round_index=0")
        if not isinstance(recipe.promotion, Mapping):
            raise TypeError("promotion settings must be an object")
        if recipe.terminal_score_distribution not in {
            "original",
            "base",
            "base_perturbed",
            "overall",
        }:
            raise ValueError(
                "terminal score distribution must be original, base, "
                "base_perturbed, or overall"
            )
        for round_index in range(1, recipe.round_count + 1):
            settings = self._settings_for_round(recipe, round_index)
            if settings.agent_budget_seconds <= 0:
                raise ValueError(f"round {round_index}: Agent budget must be positive")
            self._validate_sampling_settings(
                settings,
                round_index,
                has_initial_candidate=recipe.initial_candidate is not None,
            )

        if recipe.generator.basis_size <= 0:
            raise ValueError("Generator basis_size must be positive")

    @staticmethod
    def _validate_sampling_settings(
        settings: RoundSettings,
        round_index: int,
        *,
        has_initial_candidate: bool,
    ) -> None:
        policy = settings.development_sampling
        official = settings.official_evaluation
        if policy is None or official is None:
            raise ValueError(
                f"round {round_index}: unified sampling needs official evaluation settings"
            )
        if policy.strategy is SamplingStrategy.FIXED_RECIPE:
            if policy.fixed_recipe is None:
                raise ValueError(f"round {round_index}: fixed_recipe needs exact Case counts")
            active = policy.fixed_recipe
        else:
            if policy.fixed_recipe is not None:
                raise ValueError(
                    f"round {round_index}: adaptive sampling cannot include a fixed recipe"
                )
            active = policy.starter
        if (policy.max_cases_per_agent_per_round is not None
                and active.total > policy.max_cases_per_agent_per_round):
            raise ValueError(
                f"round {round_index}: planned Cases exceed the per-Agent sample limit"
            )
        for source, count in (
            (SampleSource.ORIGINAL, active.original),
            (SampleSource.BASE, active.base),
            (SampleSource.GUIDED, active.guided),
        ):
            if count and source not in policy.allowed_sources:
                raise ValueError(
                    f"round {round_index}: {source.value} Cases are not an allowed source"
                )
        if active.base_perturbed > active.base:
            raise ValueError(
                f"round {round_index}: base perturbations need distinct base parents"
            )
        if active.guided_perturbed > active.guided:
            raise ValueError(
                f"round {round_index}: guided perturbations need distinct guided parents"
            )
        if round_index == 1 and not has_initial_candidate:
            first_round_guided = (
                active.guided
                + active.guided_perturbed
                + official.round.guided
                + official.round.guided_perturbed
            )
            if first_round_guided:
                raise ValueError("round 1 cannot request guided Cases before feedback exists")
        if official.terminal is not None and (
            official.terminal.guided or official.terminal.guided_perturbed
        ):
            raise ValueError(
                "terminal hidden cannot use guided Cases because the final suite "
                "must be common and independent of the selected program"
            )

    @staticmethod
    def _validate_promotion_suites(
        specification: Mapping[str, object],
        plans: tuple[RoundPlan, ...],
    ) -> None:
        references = promotion_metric_references(specification)
        if references:
            for plan in plans:
                CampaignCompiler._require_hidden_distributions(
                    plan.round_hidden,
                    references,
                    label=f"round {plan.round_index} hidden",
                )

    @staticmethod
    def _validate_terminal_evaluation(
        recipe: CampaignRecipe,
        specification: TerminalEvaluationSpec,
    ) -> None:
        cohorts = {cohort.cohort_id: cohort for cohort in specification.cohorts}
        if specification.ranking.cohort_id != "overall":
            ranking_cohort = cohorts[specification.ranking.cohort_id]
            if ranking_cohort.suite.as_dict().get(
                specification.ranking.distribution,
                0,
            ) <= 0:
                raise ValueError(
                    "terminal ranking requires a nonzero "
                    f"{specification.ranking.distribution} distribution in cohort "
                    f"{ranking_cohort.cohort_id!r}"
                )
        for cohort in specification.cohorts:
            if (
                cohort.guidance_source is TerminalGuidanceSource.ROUND
                and cohort.round_index > recipe.round_count
            ):
                raise ValueError(
                    f"terminal cohort {cohort.cohort_id!r} references round "
                    f"{cohort.round_index}, but the Campaign has only "
                    f"{recipe.round_count} rounds"
                )
            if (
                cohort.guidance_source is not TerminalGuidanceSource.BASE
                and not recipe.generator.guidance_enabled
            ):
                raise ValueError(
                    f"terminal cohort {cohort.cohort_id!r} needs Guidance, "
                    "but Guidance is disabled"
                )
            if (
                cohort.guidance_source is TerminalGuidanceSource.ROUND
                and cohort.round_index == 1
                and recipe.initial_candidate is None
            ):
                raise ValueError(
                    "round-1 guided terminal cohort needs an initial_candidate"
                )

    @staticmethod
    def _require_hidden_distributions(
        suite: HiddenSuiteSpec,
        references,
        *,
        label: str,
    ) -> None:
        counts = suite.as_dict()
        for reference in references:
            count = (
                suite.case_count
                if reference.distribution == "overall"
                else counts.get(reference.distribution, 0)
            )
            if count <= 0:
                raise ValueError(
                    f"{label}: promotion metric {reference.label} requires "
                    f"nonzero {reference.distribution} Cases"
                )

    def _validate_generator(
        self,
        recipe: CampaignRecipe,
        capabilities: GeneratorCapabilities,
    ) -> None:
        if not capabilities.online and not capabilities.offline:
            raise ValueError("the selected Generator cannot sample Cases")
        if not capabilities.trainable and recipe.generator.checkpoint is None:
            raise ValueError(
                "the selected frozen Generator requires an existing checkpoint"
            )
        if recipe.generator.guidance_enabled and not capabilities.guidance:
            raise ValueError("the selected Generator does not support guided generation")
        if recipe.generator.perturbations_enabled and not capabilities.perturbations:
            raise ValueError("the selected Generator does not support perturbations")

        for round_index in range(1, recipe.round_count + 1):
            settings = self._settings_for_round(recipe, round_index)
            policy = settings.development_sampling
            official = settings.official_evaluation
            if policy is None or official is None:
                raise ValueError(
                    f"round {round_index}: canonical sampling settings are required"
                )
            if (
                policy.strategy is SamplingStrategy.AGENT_ADAPTIVE
                and not capabilities.online
            ):
                raise ValueError(
                    "agent_adaptive is the fresh online mainline and requires "
                    "an online Generator; use fixed_recipe explicitly for an "
                    "offline ablation"
                )
            counts = (
                policy.fixed_recipe
                if policy.strategy is SamplingStrategy.FIXED_RECIPE
                else policy.starter
            )
            if counts is None:
                counts = CaseCounts()
            needs_guidance = (
                counts.guided
                or counts.guided_perturbed
                or official.round.guided
                or official.round.guided_perturbed
                or SampleSource.GUIDED in policy.allowed_sources
            )
            needs_perturbation = (
                counts.base_perturbed
                or counts.guided_perturbed
                or official.round.base_perturbed
                or official.round.guided_perturbed
                or (
                    official.terminal is not None
                    and (
                        official.terminal.base_perturbed
                        or official.terminal.guided_perturbed
                    )
                )
            )
            if needs_guidance and not recipe.generator.guidance_enabled:
                raise ValueError("guided Cases are requested but guidance is disabled")
            if needs_perturbation and not recipe.generator.perturbations_enabled:
                raise ValueError("perturbed Cases are requested but perturbation is disabled")

        if recipe.terminal_evaluation is not None:
            terminal_needs_perturbation = any(
                cohort.suite.base_perturbed or cohort.suite.guided_perturbed
                for cohort in recipe.terminal_evaluation.cohorts
            )
            if terminal_needs_perturbation and not recipe.generator.perturbations_enabled:
                raise ValueError(
                    "terminal evaluation requests perturbed Cases but perturbation is disabled"
                )
