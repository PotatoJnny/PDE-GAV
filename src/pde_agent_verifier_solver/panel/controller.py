"""Page models for Setup, Live Run, and Final Result."""

from __future__ import annotations

from typing import Mapping, Sequence
from dataclasses import replace

from pde_agent_verifier_solver.engine.agent_execution import CommandAgentProgram, AGENT_NOMINATION_POLICIES

from pde_agent_verifier_solver.sdk.agent_prompt import AgentPromptSections

from pde_agent_verifier_solver.app.service import (
    ProductApplication,
    ProgramBundle,
)
from pde_agent_verifier_solver.engine.campaign import (
    AgentSpec,
    CampaignRecipe,
    CaseCounts,
    DevelopmentSamplingPolicy,
    RoundOverride,
    EvaluatorPolicy,
    GeneratorRecipe,
    RoundSettings,
)
from pde_agent_verifier_solver.engine.memory import CAMPAIGN_MEMORY_HEADER
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    Artifact,
    DatasetInput,
    DatasetKind,
    DevelopmentCoordinateMode,
    FormField,
    GeneratorCheckpoint,
    PreparedDataset,
    TaskKind,
    HiddenSuiteSpec,
    OfficialEvaluationSpec,
    PromotionCandidateReport,
    PromotionDecision,
    SampleSource,
    SamplingStrategy,
    TerminalCandidateReport,
    TerminalCohortSpec,
    TerminalEvaluationSpec,
    TerminalGuidanceSource,
    TerminalMetricSpec,
    terminal_from_legacy_suite,
)


TASK_CATEGORIES = (
    {
        "id": TaskKind.SOLVER.value,
        "name": "Full-state solver",
        "description": "Return the requested PDE fields on arbitrary supported queries.",
    },
    {
        "id": TaskKind.OBSERVABLE_MODEL.value,
        "name": "Observable model",
        "description": "Return moments or other observables; internals are unrestricted.",
    },
    {
        "id": TaskKind.REDUCED_DYNAMICS.value,
        "name": "Reduced dynamics",
        "description": "Expose a reduced state, its evolution, and an observation map.",
    },
    {
        "id": TaskKind.STATISTICS.value,
        "name": "Statistics model",
        "description": "Return Task-defined reduced-order statistics.",
    },
    {
        "id": TaskKind.CUSTOM.value,
        "name": "Custom scientific program",
        "description": "Use a plugin-defined program interface and metrics.",
    },
)


class PanelController:
    def __init__(
        self, app: ProductApplication, programs: ProgramBundle, *,
        prepared_dataset: PreparedDataset | None = None,
        generator_checkpoint: GeneratorCheckpoint | None = None,
        default_query: AnswerQuery | None = None,
    ) -> None:
        self.app = app
        self.programs = programs
        if prepared_dataset is not None and not isinstance(prepared_dataset, PreparedDataset):
            raise TypeError("Panel prepared_dataset must be a PreparedDataset")
        if generator_checkpoint is not None:
            if not isinstance(generator_checkpoint, GeneratorCheckpoint):
                raise TypeError("Panel generator_checkpoint must be a GeneratorCheckpoint")
            if generator_checkpoint.generator_id != programs.generator.info.program_id:
                raise ValueError("Panel checkpoint belongs to a different Generator")
        if default_query is not None and not isinstance(default_query, AnswerQuery):
            raise TypeError("Panel default_query must be an AnswerQuery")
        self._prepared_dataset = prepared_dataset
        self._generator_checkpoint = generator_checkpoint
        self._default_query = default_query

    def _bound_defaults(self) -> dict[str, object]:
        if self._prepared_dataset is None:
            return {}
        capabilities = self.programs.generator.capabilities
        result = {
            "project_name": self.programs.problem.info.title,
            "guidance_enabled": capabilities.guidance,
            "perturbations_enabled": capabilities.perturbations,
            "allowed_sources": (["original", "base", "guided"] if capabilities.guidance
                                else ["original", "base"]),
        }
        if not self._prepared_dataset.development_cases and not self._prepared_dataset.hidden_cases:
            result.update({
                "allowed_sources": ["base", "guided"] if capabilities.guidance else ["base"],
                "starter_cases": {"base": 2},
                "fixed_recipe": {"base": 12},
                "round_hidden": {"base": 5},
                "terminal_evaluation": {
                    "cohorts": [{"id": "base_model", "guidance_source": "base", "counts": {"base": 4}}],
                    "ranking": {"cohort_id": "base_model", "distribution": "base", "statistic": "mean"},
                },
            })
        return result

    @staticmethod
    def _submission_policies(programs):
        return {key: (program.worker_settings.get("nomination_policy", "all")
                      if isinstance(program, CommandAgentProgram) else "program_defined")
                for key, program in programs.agents.items()}

    def _programs_for_run(self, values, agents):
        if "nomination_policy" not in values:
            return self.programs
        policy = values["nomination_policy"]
        if policy not in AGENT_NOMINATION_POLICIES:
            raise ValueError("nomination_policy must be all, latest or single_primary_v1")
        selected = {agent.agent_id for agent in agents}
        if any(not isinstance(self.programs.agents[key], CommandAgentProgram) for key in selected):
            raise ValueError("Submission selection requires command Agents; local demo Agents return their own result")
        return replace(self.programs, agents={
            key: replace(program, worker_settings={**program.worker_settings, "nomination_policy": policy})
                 if key in selected else program
            for key, program in self.programs.agents.items()})

    def setup_page(self) -> Mapping[str, object]:
        return {
            "problem": _program(self.programs.problem.info),
            "task": _program(self.programs.task.info),
            "generator": _program(self.programs.generator.info),
            "task_categories": TASK_CATEGORIES,
            "candidate_execution": self.app.candidate_execution_settings,
            "agent_submission_policies": self._submission_policies(self.programs),
            "submission_selection_available": all(isinstance(p, CommandAgentProgram) for p in self.programs.agents.values()),
            "forms": {
                "problem": _fields(self.programs.problem.setup_fields()),
                "task": _fields(self.programs.task.setup_fields()),
                "generator": _fields(self.programs.generator.setup_fields()),
            },
            "defaults": {
                "project_name": "Toy Poisson campaign",
                "dataset_kind": DatasetKind.NUMERICAL.value,
                "round_count": 3,
                "sampling_strategy": SamplingStrategy.AGENT_ADAPTIVE.value,
                "max_sampled_cases_per_agent_per_round": 12,
                "agent_budget_seconds": 300,
                "basis_size": 8,
                "agents": list(self.programs.agents),
                "starter_cases": {"original": 2, "base": 2, "guided": 0},
                "fixed_recipe": {"original": 4, "base": 6, "guided": 2},
                "round_hidden": {"original": 1, "base": 3, "guided": 2},
                "terminal_evaluation": {
                    "cohorts": [
                        {
                            "id": "base_model",
                            "guidance_source": "base",
                            "counts": {"original": 1, "base": 4},
                        }
                    ],
                    "ranking": {
                        "cohort_id": "base_model",
                        "distribution": "base",
                        "statistic": "mean",
                    },
                },
                "promotion": {"distribution": "base", "statistic": "mean"},
                "master_seed": 0,
                "allow_grid_truth": False,
                "allow_point_truth": False,
                "development_coordinate_mode": (
                    DevelopmentCoordinateMode.PRIVATE_RANDOM_POINTS.value
                ),
                "development_point_count": 32,
                **self._bound_defaults(),
            },
            "backend": {
                "prepared": self._prepared_dataset is not None,
                "checkpoint_loaded": self._generator_checkpoint is not None,
                "query_fields": list(self._default_query.fields) if self._default_query else None,
                "online": self.programs.generator.capabilities.online,
                "guidance": self.programs.generator.capabilities.guidance,
                "perturbations": self.programs.generator.capabilities.perturbations,
            },
        }

    def _prompt_agents(self, values: Mapping[str, object]) -> tuple[AgentSpec, ...]:
        requested = values.get("agents", tuple(self.programs.agents))
        if not isinstance(requested, Sequence) or isinstance(requested, str):
            raise ValueError("agents must be a list of Agent ids")
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("agents must be nonempty and distinct")
        unknown = set(requested) - set(self.programs.agents)
        if unknown:
            raise ValueError(f"unknown Agents: {sorted(unknown)}")
        sections = values.get("agent_prompt_sections")
        roles = values.get("agent_roles", {})
        if not isinstance(roles, Mapping) or set(roles) - set(requested):
            raise ValueError("agent_roles must map selected Agent ids to Markdown")
        if sections is not None and "agent_prompt" in values:
            raise ValueError("Choose agent_prompt_sections or legacy agent_prompt")
        specs = []
        for agent_id in requested:
            if sections is None:
                prompt = values.get("agent_prompt", "Improve the incumbent and explain the largest remaining failure.")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("agent_prompt must be nonempty text")
                if roles:
                    raise ValueError("agent_roles requires agent_prompt_sections")
            else:
                shared = AgentPromptSections.from_mapping(sections)
                prompt = AgentPromptSections(shared.task, shared.resources,
                                             roles.get(agent_id, shared.role)).render()
            specs.append(AgentSpec(agent_id=agent_id, name=agent_id.replace("-", " ").title(), prompt=prompt))
        return tuple(specs)

    def prompt_preview(self, values: Mapping[str, object]) -> Mapping[str, object]:
        from pde_agent_verifier_solver.engine.agent_prompt import build_agent_context_prompt
        brief = self.programs.task.agent_brief(_mapping(values.get("task_settings")))
        return {"agents": [
            {"agent_id": agent.agent_id, "name": agent.name,
             "prompt": build_agent_context_prompt(agent.prompt, brief)}
            for agent in self._prompt_agents(values)
        ], "note": "Task, resources, role and Task output contract. The execution adapter adds the actual round budgets, tools and submission rules when the Agent starts."}

    def create_and_launch(self, values: Mapping[str, object]) -> Mapping[str, str]:
        agents = self._prompt_agents(values)
        programs = self._programs_for_run(values, agents)
        requested_keys = set(values)
        values = {**self._bound_defaults(), **values}
        project_name = str(values.get("project_name", "Toy Poisson campaign"))
        dataset_kind = DatasetKind(str(values.get("dataset_kind", "numerical")))
        problem_values = _mapping(values.get("problem_settings"))
        task_values = _mapping(values.get("task_settings"))
        generator_values = _mapping(values.get("generator_settings"))
        round_count = int(values.get("round_count", 3))
        guidance_enabled = bool(
            generator_values.get(
                "guidance_enabled",
                values.get("guidance_enabled", True),
            )
        )
        perturbations_enabled = bool(
            generator_values.get(
                "perturbations_enabled",
                values.get("perturbations_enabled", True),
            )
        )
        dataset = DatasetInput(
            dataset_id=f"{project_name.lower().replace(' ', '-')}-dataset",
            kind=dataset_kind,
            location=str(values["dataset_location"]) if values.get("dataset_location") else None,
            problem={"equation": "1D Poisson", **problem_values},
            geometry={
                "x": [0.0, 1.0],
                "description": problem_values.get("domain", "x in [0, 1]"),
            },
            settings={
                **problem_values,
                "development_cases": int(problem_values.get("development_cases", 12)),
                "hidden_cases": int(problem_values.get("hidden_cases", max(4, round_count + 1))),
            },
        )
        if self._prepared_dataset is None:
            project = self.app.create_project(project_name, dataset, programs)
        else:
            if values.get("dataset_location") or problem_values:
                raise ValueError("this Panel uses prepared backend data; rebind to change its data or physical settings")
            project = self.app.create_prepared_project(project_name, self._prepared_dataset, programs)
        strategy = SamplingStrategy(
            str(values.get("sampling_strategy", SamplingStrategy.AGENT_ADAPTIVE.value))
        )
        raw_case_limit = values.get(
            "max_sampled_cases_per_agent_per_round", values.get("case_count", 12)
        )
        max_cases = None if raw_case_limit is None else int(raw_case_limit)
        default_sources = (
            ("original", "base", "guided")
            if guidance_enabled
            else ("original", "base")
        )
        allowed_values = values.get("allowed_sources", default_sources)
        if (self._prepared_dataset is not None and not guidance_enabled
                and "allowed_sources" not in requested_keys):
            allowed_values = tuple(value for value in allowed_values if value != "guided")
        if not isinstance(allowed_values, Sequence) or isinstance(allowed_values, str):
            raise ValueError("allowed_sources must be a list")
        allowed_sources = tuple(SampleSource(str(value)) for value in allowed_values)
        starter = _case_counts(
            values.get("starter_cases"),
            {"original": 2, "base": 2},
        )
        fixed_recipe = _case_counts(
            values.get("fixed_recipe"),
            (
                {"original": 4, "base": 6, "guided": 2}
                if guidance_enabled
                else {"original": 4, "base": 8}
            ),
        )
        initial_candidate = values.get("initial_candidate")
        if initial_candidate is not None and not isinstance(
            initial_candidate,
            Artifact,
        ):
            raise TypeError(
                "initial_candidate must be an Artifact supplied through the SDK"
            )
        if _boolean(
            values.get("cumulative_fixed_cases"),
            default=False,
            name="cumulative_fixed_cases",
        ):
            raise ValueError(
                "Case replay is outside the canonical pipeline; use an explicit "
                "legacy/control-arm runner"
            )
        policy = DevelopmentSamplingPolicy(
            strategy=strategy,
            max_cases_per_agent_per_round=max_cases,
            allowed_sources=allowed_sources,
            starter=starter,
            fixed_recipe=(
                fixed_recipe if strategy is SamplingStrategy.FIXED_RECIPE else None
            ),
            hide_case_types=bool(values.get("hide_case_types", True)),
        )
        first_policy = policy
        if initial_candidate is None and strategy is SamplingStrategy.FIXED_RECIPE and (
            fixed_recipe.guided or fixed_recipe.guided_perturbed
        ):
            first_recipe = CaseCounts(
                original=fixed_recipe.original,
                base=fixed_recipe.base + fixed_recipe.guided,
                base_perturbed=(
                    fixed_recipe.base_perturbed + fixed_recipe.guided_perturbed
                ),
            )
            first_policy = DevelopmentSamplingPolicy(
                strategy=strategy,
                max_cases_per_agent_per_round=max_cases,
                allowed_sources=allowed_sources,
                starter=starter,
                fixed_recipe=first_recipe,
                hide_case_types=policy.hide_case_types,
            )
        round_hidden = _hidden_spec(
            values.get("round_hidden"),
            (
                {"original": 1, "base": 3, "guided": 2}
                if guidance_enabled
                else {"original": 1, "base": 5}
            ),
        )
        terminal_evaluation = _terminal_evaluation_spec(
            values.get("terminal_evaluation"),
            legacy_hidden=values.get("terminal_hidden"),
            legacy_distribution=str(
                values.get("terminal_score_distribution", "base")
            ),
        )
        first_hidden = (
            round_hidden
            if initial_candidate is not None
            else HiddenSuiteSpec(
                original=round_hidden.original,
                base=round_hidden.base + round_hidden.guided,
                guided=0,
                base_perturbed=(
                    round_hidden.base_perturbed + round_hidden.guided_perturbed
                ),
                guided_perturbed=0,
            )
        )
        official = OfficialEvaluationSpec(round=round_hidden)
        first_official = OfficialEvaluationSpec(round=first_hidden)

        recipe = CampaignRecipe(
            name=str(values.get("run_name", f"{project_name} run")),
            round_count=round_count,
            agents=agents,
            default_round=RoundSettings(
                agent_budget_seconds=int(values.get("agent_budget_seconds", 300)),
                development_sampling=policy,
                official_evaluation=official,
            ),
            round_overrides={
                1: RoundOverride(
                    development_sampling=first_policy,
                    official_evaluation=first_official,
                )
            },
            generator=GeneratorRecipe(
                basis_size=int(generator_values.get("basis_size", values.get("basis_size", 8))),
                guidance_enabled=guidance_enabled,
                perturbations_enabled=perturbations_enabled,
                settings=generator_values,
                checkpoint=self._generator_checkpoint,
            ),
            evaluator=EvaluatorPolicy(
                default_query=self._default_query or AnswerQuery.grid(fields=("u",), x=int(values.get("grid_x", 65))),
                allow_grid_truth=_boolean(
                    values.get("allow_grid_truth"),
                    default=False,
                    name="allow_grid_truth",
                ),
                allow_point_truth=_boolean(
                    values.get("allow_point_truth"),
                    default=False,
                    name="allow_point_truth",
                ),
                development_coordinate_mode=DevelopmentCoordinateMode(
                    str(
                        values.get(
                            "development_coordinate_mode",
                            DevelopmentCoordinateMode.PRIVATE_RANDOM_POINTS.value,
                        )
                    )
                ),
                development_point_count=int(
                    values.get("development_point_count", 32)
                ),
            ),
            task_settings=task_values,
            master_seed=int(values.get("master_seed", 0)),
            initial_candidate=initial_candidate,
            terminal_evaluation=terminal_evaluation,
            promotion=_promotion_spec(values.get("promotion")),
        )
        session = self.app.launch_run(project.project_id, recipe)
        return {"project_id": project.project_id, "run_id": session.run_id}

    def run_page(self, run_id: str) -> Mapping[str, object]:
        session = self.app.get_run(run_id)
        project = self.app.get_project(session.project_id)
        events = session.events.snapshot()
        page: dict[str, object] = {
            "run_id": session.run_id,
            "project_id": session.project_id,
            "name": session.campaign.name,
            "status": session.status.value,
            "activity": _agent_recovery_activity(events) if session.status.value == "running" else None,
            "candidate_execution": self.app.candidate_execution_settings,
            "agent_submission_policies": self._submission_policies(project.programs),
            "round_count": len(session.campaign.rounds),
            "master_seed": session.campaign.master_seed,
            "initial_candidate_id": (
                session.campaign.initial_candidate.artifact_id
                if session.campaign.initial_candidate is not None
                else None
            ),
            "allow_grid_truth": session.campaign.evaluator.allow_grid_truth,
            "allow_point_truth": session.campaign.evaluator.allow_point_truth,
            "development_coordinate_mode": (
                session.campaign.evaluator.development_coordinate_mode.value
            ),
            "development_point_count": (
                session.campaign.evaluator.development_point_count
            ),
            "development_fixed_query": (
                _query_view(session.campaign.evaluator.default_query)
                if session.campaign.evaluator.development_coordinate_mode
                is DevelopmentCoordinateMode.USER_FIXED_QUERY
                else None
            ),
            "terminal_evaluation": _terminal_evaluation_view(
                session.campaign.terminal_evaluation
            ),
            "events": [
                {
                    "sequence": event.sequence,
                    "type": event.event_type,
                    "message": event.message,
                    "round": event.round_index,
                    "payload": dict(event.payload),
                }
                for event in events
            ],
            "plan": [
                {
                    "round": plan.round_index,
                    "budget_seconds": plan.agent_budget_seconds,
                    "case_mix": dict(plan.case_mix_summary()),
                    "hidden_generated": plan.hidden_generated_count,
                    "sampling_strategy": (
                        plan.development_sampling.strategy.value
                    ),
                    "sample_limit": (
                        plan.development_sampling.max_cases_per_agent_per_round
                    ),
                    "allowed_sources": [
                        source.value
                        for source in plan.development_sampling.allowed_sources
                    ],
                    "round_hidden": plan.round_hidden.as_dict(),
                }
                for plan in session.campaign.rounds
            ],
        }
        if project.reconstruction_report is not None:
            page["reconstruction"] = project.reconstruction_report.as_dict()
        # Memory is useful while a Campaign is still running and is most
        # important when a round aborts. Every completed or aborted round emits
        # one self-contained update, so the panel can reconstruct the current
        # handoff without waiting for a final CampaignResult.
        live_memory, live_candidate_history = _memory_from_events(events)
        if live_memory is not None:
            page["memory"] = live_memory
            page["candidate_history"] = live_candidate_history
        if session.error:
            page["error"] = session.error
        if session.result is not None:
            terminal_round_reports = getattr(
                session.result,
                "terminal_round_reports",
                (),
            )
            if terminal_round_reports:
                page["terminal_round_reports"] = [
                    {
                        "round": round_result.round_index,
                        **_candidate_report_view(report),
                    }
                    for round_result, report in zip(
                        session.result.rounds,
                        terminal_round_reports,
                    )
                ]
            best_terminal_round = getattr(
                session.result,
                "best_terminal_round",
                None,
            )
            if best_terminal_round is not None:
                page["best_terminal_round"] = best_terminal_round
            seed_ledger = dict(
                getattr(session.result, "seed_ledger", {})
            )
            page["seed_ledger"] = seed_ledger
            page["rounds"] = [
                {
                    "round": result.round_index,
                    "hidden_suite_size": result.hidden_suite_size,
                    "case_mix": dict(result.case_mix),
                    "selected_artifact_id": result.selected.artifact.artifact_id,
                    "selected_error": result.selected.primary_error,
                    "agent_case_counts": {
                        agent_id: dict(counts)
                        for agent_id, counts in result.agent_case_counts.items()
                    },
                    "hidden_distributions": dict(result.hidden_distribution_metrics),
                    "promotion": _promotion_view(
                        result.promotion_reports,
                        result.promotion_decision,
                    ),
                    "candidates": [
                        {
                            "artifact_id": evaluation.artifact.artifact_id,
                            "name": evaluation.artifact.name,
                            "agent_id": evaluation.artifact.agent_id,
                            "primary_error": evaluation.primary_error,
                            "metrics": dict(evaluation.metrics),
                        }
                        for evaluation in result.candidate_evaluations
                    ],
                }
                for result in session.result.rounds
            ]
            # Scientific memory is a chronological record of what the Agents
            # tried and what the Controller actually selected. Candidate
            # ancestry is useful too, but it is program history rather than
            # scientific memory, so expose it as a separate panel section.
            page["memory"] = {
                "markdown": session.result.memory.markdown,
                "rounds": [
                    {
                        "round": round_log.round_index,
                        "status": round_log.status,
                        "agents": [
                            {
                                "agent_id": agent_log.agent_id,
                                "agent_name": agent_log.agent_name,
                                "status": agent_log.status,
                                "markdown": agent_log.markdown,
                            }
                            for agent_log in round_log.agent_logs
                        ],
                        "controller_markdown": round_log.controller_markdown,
                        "merged_markdown": round_log.merged_markdown,
                    }
                    for round_log in session.result.memory.rounds
                ],
            }
            page["candidate_history"] = [
                {
                    "id": node.node_id,
                    "parent": node.parent_node_id,
                    "round": node.round_index,
                    "agent_id": node.agent_id,
                    "artifact_id": node.artifact_id,
                    "title": node.title,
                    "summary": node.summary,
                    "official_distributions": {
                        name: {
                            "case_count": summary.case_count,
                            "mean": summary.mean,
                            "cvar": summary.cvar,
                            "maximum": summary.maximum,
                            "metrics": dict(summary.metrics),
                        }
                        for name, summary in node.official_distributions.items()
                    },
                    "failures": node.failures,
                    "status": node.status,
                    "selected": node.selected,
                }
                for node in session.result.candidate_history.nodes
            ]
            card = self.programs.task.result_card(session.result.selected)
            page["result"] = {
                "artifact_id": session.result.selected.artifact.artifact_id,
                "name": session.result.selected.artifact.name,
                "summary": session.result.selected.artifact.summary,
                "usage": session.result.selected.artifact.usage,
                "primary_error": session.result.selected.primary_error,
                "metrics": dict(session.result.selected.metrics),
                "terminal_score_distribution": (
                    session.campaign.terminal_score_distribution
                ),
                "terminal_distributions": dict(session.result.terminal_distribution_metrics),
                "seed_ledger": seed_ledger,
                "card": {
                    "title": card.title,
                    "summary": card.summary,
                    "values": dict(card.values),
                },
            }
        return page


def _agent_recovery_activity(events: Sequence[object]) -> str | None:
    waiting = set()
    for event in events:
        kind = getattr(event, "event_type", None)
        payload = getattr(event, "payload", {})
        agent_id = payload.get("agent_id") if isinstance(payload, Mapping) else None
        if not isinstance(agent_id, str):
            continue
        if kind == "agent_recovering":
            waiting.add(agent_id)
        elif kind in {"agent_started", "agent_resumed", "agent_finished", "agent_failed"}:
            waiting.discard(agent_id)
    if waiting:
        return "Waiting for model service; saved Agent work will resume"
    return None


def _memory_from_events(
    events: Sequence[object],
) -> tuple[Mapping[str, object] | None, list[Mapping[str, object]]]:
    """Rebuild the latest Campaign memory view from durable round events."""

    rounds: list[Mapping[str, object]] = []
    candidate_history: list[Mapping[str, object]] = []
    for event in events:
        if getattr(event, "event_type", None) != "campaign_memory_updated":
            continue
        payload = getattr(event, "payload", {})
        if not isinstance(payload, Mapping):
            continue
        round_memory = payload.get("round_memory")
        if not isinstance(round_memory, Mapping):
            continue
        merged = round_memory.get("merged_markdown")
        if not isinstance(merged, str) or not merged.strip():
            continue
        rounds.append(dict(round_memory))
        history = payload.get("candidate_history")
        if isinstance(history, (list, tuple)) and all(
            isinstance(node, Mapping) for node in history
        ):
            candidate_history = [dict(node) for node in history]

    if not rounds:
        return None, []
    markdown = "\n\n".join(
        (
            CAMPAIGN_MEMORY_HEADER,
            *(str(round_log["merged_markdown"]) for round_log in rounds),
        )
    )
    return {"markdown": markdown, "rounds": rounds}, candidate_history


def build_toy_panel() -> PanelController:
    from pde_agent_verifier_solver.plugins.toy import (
        ToyPoissonGenerator,
        ToyPoissonProblem,
        ToyResearchAgent,
        ToySolverTask,
    )

    programs = ProgramBundle(
        problem=ToyPoissonProblem(),
        task=ToySolverTask(),
        generator=ToyPoissonGenerator(),
        agents={
            "careful-agent": ToyResearchAgent(step_size=1),
            "explorer-agent": ToyResearchAgent(step_size=2),
        },
    )
    # The bundled Toy Agents are trusted Python objects. Keep their cooperative
    # execution an explicit local-development choice rather than the formal
    # ProductApplication default.
    return PanelController(ProductApplication.local_development(), programs)


def _program(info) -> Mapping[str, str]:
    return {
        "id": info.program_id,
        "title": info.title,
        "description": info.description,
    }


def _fields(fields: Sequence[FormField]) -> list[Mapping[str, object]]:
    return [
        {
            "name": item.name,
            "label": item.label,
            "type": item.field_type,
            "default": item.default,
            "help": item.help_text,
            "choices": item.choices,
        }
        for item in fields
    ]


def _mapping(value: object) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("settings must be an object")
    return value


def _case_counts(value: object, defaults: Mapping[str, int]) -> CaseCounts:
    # A user-supplied recipe is complete: omitted categories mean zero. The
    # defaults are used only when the whole field is absent. Merging a partial
    # recipe with defaults made an explicit {"original": 2} silently add
    # generated Cases and could exceed the user's round limit.
    data = dict(defaults) if value is None else dict(_mapping(value))
    return CaseCounts(
        original=int(data.get("original", 0)),
        base=int(data.get("base", 0)),
        guided=int(data.get("guided", 0)),
        base_perturbed=int(data.get("base_perturbed", 0)),
        guided_perturbed=int(data.get("guided_perturbed", 0)),
    )


def _hidden_spec(value: object, defaults: Mapping[str, int]) -> HiddenSuiteSpec:
    # Hidden-suite counts follow the same complete-value rule as development
    # recipes, so the panel never evaluates an unrequested Case type.
    data = dict(defaults) if value is None else dict(_mapping(value))
    return HiddenSuiteSpec(
        original=int(data.get("original", 0)),
        base=int(data.get("base", 0)),
        guided=int(data.get("guided", 0)),
        base_perturbed=int(data.get("base_perturbed", 0)),
        guided_perturbed=int(data.get("guided_perturbed", 0)),
    )


def _terminal_evaluation_spec(
    value: object,
    *,
    legacy_hidden: object,
    legacy_distribution: str,
) -> TerminalEvaluationSpec:
    """Parse the canonical terminal contract or translate the former UI field."""

    if value is None:
        legacy_suite = _hidden_spec(
            legacy_hidden,
            {"original": 1, "base": 4},
        )
        return terminal_from_legacy_suite(
            legacy_suite,
            score_distribution=legacy_distribution,
        )
    if legacy_hidden is not None:
        raise ValueError(
            "set terminal_evaluation or terminal_hidden, not both"
        )

    data = _mapping(value)
    cohort_values = data.get("cohorts")
    if (
        not isinstance(cohort_values, Sequence)
        or isinstance(cohort_values, (str, bytes))
        or not cohort_values
    ):
        raise ValueError("terminal_evaluation.cohorts must be a nonempty list")

    cohorts: list[TerminalCohortSpec] = []
    for index, cohort_value in enumerate(cohort_values):
        cohort = _mapping(cohort_value)
        if "counts" not in cohort:
            raise ValueError(f"terminal cohort {index + 1} needs counts")
        round_value = cohort.get("round")
        if isinstance(round_value, bool):
            raise ValueError("terminal cohort round must be a positive integer")
        round_index = None if round_value is None else int(round_value)
        cohorts.append(
            TerminalCohortSpec(
                cohort_id=str(cohort.get("id", "")),
                guidance_source=TerminalGuidanceSource(
                    str(cohort.get("guidance_source", "base"))
                ),
                round_index=round_index,
                suite=_hidden_spec(cohort["counts"], {}),
            )
        )

    ranking_value = data.get("ranking")
    if not isinstance(ranking_value, Mapping):
        raise ValueError("terminal_evaluation.ranking must be an object")
    ranking = _mapping(ranking_value)
    return TerminalEvaluationSpec(
        cohorts=tuple(cohorts),
        ranking=TerminalMetricSpec(
            cohort_id=str(ranking.get("cohort_id", "")),
            distribution=str(ranking.get("distribution", "")),
            statistic=str(ranking.get("statistic", "mean")),
        ),
    )


def _terminal_evaluation_view(
    specification: TerminalEvaluationSpec | None,
) -> Mapping[str, object] | None:
    if specification is None:
        return None
    cohorts = []
    for cohort in specification.cohorts:
        item: dict[str, object] = {
            "id": cohort.cohort_id,
            "guidance_source": cohort.guidance_source.value,
            "counts": cohort.suite.as_dict(),
        }
        if cohort.round_index is not None:
            item["round"] = cohort.round_index
        cohorts.append(item)
    return {
        "cohorts": cohorts,
        "ranking": {
            "cohort_id": specification.ranking.cohort_id,
            "distribution": specification.ranking.distribution,
            "statistic": specification.ranking.statistic,
        },
    }


def _boolean(value: object, *, default: bool, name: str) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _query_view(query: AnswerQuery) -> Mapping[str, object]:
    return {
        "kind": query.kind.value,
        "fields": list(query.fields),
        "resolution": dict(query.resolution),
        "points": [dict(point) for point in query.points],
    }


def _candidate_report_view(
    report: PromotionCandidateReport | TerminalCandidateReport,
) -> Mapping[str, object]:
    return {
        "candidate_id": report.candidate_id,
        "technically_eligible": report.technically_eligible,
        "distributions": {
            name: {
                "case_count": metric.case_count,
                "mean": metric.mean,
                "cvar": metric.cvar,
                "maximum": metric.maximum,
                "metrics": dict(metric.metrics),
            }
            for name, metric in report.distributions.items()
        },
        "guardrails": dict(report.guardrails),
        "failures": list(report.failures),
    }


def _promotion_view(
    reports: Sequence[PromotionCandidateReport],
    decision: PromotionDecision | None,
) -> Mapping[str, object] | None:
    """Project the canonical Promotion reports into the simple Panel API."""

    if decision is None:
        return None
    candidate_ids = {
        report.candidate_digest: report.candidate_id
        for report in reports
    }
    return {
        "selected_candidate_id": candidate_ids.get(
            decision.selected_digest,
            "Selected candidate",
        ),
        "promoted": decision.promoted,
        "reason": decision.reason,
        "reports": [_candidate_report_view(report) for report in reports],
        "ranking": [
            {
                "candidate_id": candidate_ids.get(
                    entry.candidate_digest,
                    "Unknown candidate",
                ),
                "rank": entry.rank,
                "eligible": entry.eligible,
                "reason": entry.reason,
            }
            for entry in decision.ranking
        ],
    }


def _promotion_spec(value: object) -> Mapping[str, object]:
    data = dict(_mapping(value))
    if "preset" in data or "program" in data:
        return data
    distribution = str(data.get("distribution", "base"))
    statistic = str(data.get("statistic", "mean"))
    settings: dict[str, object] = {
        "primary": {"distribution": distribution, "statistic": statistic},
        "require_improvement": bool(data.get("require_improvement", True)),
        "minimum_improvement": float(data.get("minimum_improvement", 0.0)),
    }
    if "tie_breakers" in data:
        settings["tie_breakers"] = data["tie_breakers"]
    if "guardrails" in data:
        settings["guardrails"] = data["guardrails"]
    return {"preset": "minimize_metric", "settings": settings}
