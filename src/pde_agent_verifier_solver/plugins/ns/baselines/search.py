"""External fixed-data NS controls, separate from the canonical Controller.

Only development Cases enter Agent contexts. Selection uses one frozen hidden
validation plan. No terminal/test bank is accepted by this API.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from pde_agent_verifier_solver.sdk.evaluation_devices import bind_evaluation_device_pool

from pde_agent_verifier_solver.engine.agent_execution import (
    AgentExecutionError, AgentInfrastructureError,
)
from pde_agent_verifier_solver.engine.campaign import (
    AgentSpec, CaseCounts, DevelopmentSamplingPolicy, EvaluatorPolicy, RoundPlan,
)
from pde_agent_verifier_solver.engine.evaluator import AgentEvaluator, HiddenVerifier
from pde_agent_verifier_solver.engine.memory import (
    AgentRoundLog, CampaignMemorySnapshot, CampaignRoundLog,
)
from pde_agent_verifier_solver.engine.python_candidate import (
    PythonCandidateCodec, PythonCandidateProgram,
)
from pde_agent_verifier_solver.engine.runtime import AgentContext, AgentOutcome
from pde_agent_verifier_solver.sdk import (
    Artifact, CaseRecord, CaseVisibility, HiddenSuiteSpec,
    SampleSource, SamplingStrategy, TaskKind,
)
from ..tasks import NSSolverTask
from .workflows import LITERATURE_METHODS, LiteratureOptions

METHODS = ("refinement", "competitive_refinement", "bon_repair", "era_futs_ns_adaptation", "aide_ns_adaptation",
           "codepde_repeated_sample_ns_adaptation", *LITERATURE_METHODS)


@dataclass(frozen=True)
class NSBaselineConfig:
    method: str
    output_directory: str | Path
    seed: int = 0
    iterations: int = 4
    independent_candidates: int = 4
    repair_iterations: int = 1
    agent_budget_seconds: int | None = 300
    prompt: str = "Develop an accurate and efficient Navier--Stokes solver."
    era_c_puct: float = 1.0
    aide_num_drafts: int = 5
    aide_debug_prob: float = 0.5
    aide_max_debug_depth: int = 3
    codepde_num_repeated_samples: int = 2
    codepde_num_debugging_trials_per_sample: int = 5
    literature_options: Mapping[str, object] = field(default_factory=dict)
    competition_rounds: int = 5
    competition_roles: tuple[AgentSpec, ...] = (
        AgentSpec("ns-broad-explorer", "NS broad explorer",
                  "Explore alternative numerical solver designs and test their accuracy."),
        AgentSpec("ns-critical-refiner", "NS critical refiner",
                  "Diagnose the incumbent's errors and improve its numerical accuracy."),
    )

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"unknown NS baseline method: {self.method}")
        if self.method in LITERATURE_METHODS:
            options = LiteratureOptions(**dict(self.literature_options))
            if options.protocol == "completion_v2" and self.method == "autopde_ns_adaptation":
                if options.autopde_attempts != 1 or options.autopde_review_seconds is not None:
                    raise ValueError("completion AutoPDE requires one attempt and no review deadline")
        elif self.literature_options:
            raise ValueError("literature_options only apply to literature workflows")
        for name in ("seed", "iterations", "independent_candidates",
                     "repair_iterations", "agent_budget_seconds", "codepde_num_repeated_samples",
                     "codepde_num_debugging_trials_per_sample", "competition_rounds"):
            value = getattr(self, name)
            if name == "agent_budget_seconds" and self.method in LITERATURE_METHODS and LiteratureOptions(**dict(self.literature_options)).protocol == "completion_v2":
                if value is not None:
                    raise ValueError("completion_v2 requires null Agent time limit")
                continue
            minimum = 0 if name in {"seed", "repair_iterations"} else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if (isinstance(self.era_c_puct, bool)
                or not math.isfinite(self.era_c_puct) or self.era_c_puct < 0):
            raise ValueError("era_c_puct must be finite and nonnegative")
        if self.method == "aide_ns_adaptation":
            from .aide import AIDESearchOptions
            AIDESearchOptions(self.aide_num_drafts, self.aide_debug_prob, self.aide_max_debug_depth)
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("baseline prompt cannot be empty")
        roles = tuple(self.competition_roles)
        if (not roles or any(not isinstance(role, AgentSpec) for role in roles)
                or any(not isinstance(value, str) or not value.strip()
                       for role in roles for value in (role.agent_id, role.name, role.prompt))
                or len({role.agent_id for role in roles}) != len(roles)):
            raise ValueError("competition_roles requires nonempty, uniquely identified AgentSpecs")
        object.__setattr__(self, "competition_roles", roles)

    @property
    def turn_count(self):
        if self.method in LITERATURE_METHODS:
            options = LiteratureOptions(**dict(self.literature_options))
            if self.method == "codepde_refinement_ns_adaptation":
                return (options.codepde_samples + len(options.codepde_reference_counts)) * options.codepde_trials
            if self.method == "autopde_ns_adaptation":
                return (2 if options.protocol == "completion_v2" else
                        options.autopde_attempts * (2 * options.json_attempts + 2))
            if self.method == "pde_sharp_ns_adaptation":
                return ((1 if options.protocol == "completion_v2" else 5) * options.json_attempts + options.sharp_samples
                        + options.sharp_cycles * options.sharp_judges * (
                            options.json_attempts + options.sharp_hybrid_rounds))
            return (5 if options.protocol == "completion_v2" else 3) * options.json_attempts + options.autonumerics_top_k * options.autonumerics_fresh_starts * (
                1 + options.autonumerics_coarse_repairs + options.autonumerics_fine_repairs)
        if self.method == "competitive_refinement":
            return self.competition_rounds * len(self.competition_roles)
        if self.method == "codepde_repeated_sample_ns_adaptation":
            return self.codepde_num_repeated_samples * self.codepde_num_debugging_trials_per_sample
        if self.method == "bon_repair":
            return self.independent_candidates + self.repair_iterations
        return self.iterations


@dataclass(frozen=True)
class NSBaselineResult:
    selected: Artifact
    selection_error: float
    output_directory: Path
    report: Mapping[str, object]


def _json_value(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _json_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (Enum, Path)):
        return value.value if isinstance(value, Enum) else str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path, value):
    from pde_agent_verifier_solver.engine.checkpoint_store import atomic_json
    atomic_json(Path(path), _json_value(value))


def _write_cost_report(root):
    try:
        from pde_agent_verifier_solver.reporting.costs import collect_run_costs
        _write_json(root / "costs.json", collect_run_costs(root))
        (root / "costs_status.json").unlink(missing_ok=True)
    except Exception as exc:
        _write_json(root / "costs_status.json", {
            "status": "failed", "error": f"{type(exc).__name__}: {exc}",
            "scientific_result_unchanged": True,
        })


def _memory(history):
    rounds = []
    for index, (notes, error) in enumerate(history, 1):
        summary = f"External baseline validation error: {error!r}."
        rounds.append(CampaignRoundLog(
            round_index=index,
            agent_logs=(AgentRoundLog(index, "ns-baseline", "NS baseline", notes or "No notes."),),
            controller_markdown=summary,
            merged_markdown=f"## Baseline turn {index}\n\n{notes}\n\n{summary}",
        ))
    return CampaignMemorySnapshot(tuple(rounds))


def _validate_banks(development, selection):
    for label, rows, visibility in (
        ("development", development, CaseVisibility.DEVELOPMENT),
        ("selection", selection, CaseVisibility.HIDDEN),
    ):
        if not rows or any(not isinstance(row, CaseRecord) for row in rows):
            raise ValueError(f"{label} requires a nonempty CaseRecord bank")
        ids = [row.case.case_id for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{label} bank contains duplicate case ids")
        if any(row.case.visibility is not visibility for row in rows):
            raise ValueError(f"{label} bank requires {visibility.value} visibility")
    if {row.case.case_id for row in development} & {row.case.case_id for row in selection}:
        raise ValueError("development and selection banks must be disjoint")


def run_ns_baseline(
    config: NSBaselineConfig, *,
    development_records: Sequence[CaseRecord],
    selection_records: Sequence[CaseRecord],
    evaluator_policy: EvaluatorPolicy,
    agent_program,
    agent_execution,
    initial_candidate: Artifact | None = None,
    task_settings: Mapping[str, object] | None = None,
    public_problem_bundle=None,
    candidate_execution=None,
    generator=None,
    generator_state=None,
    selection_evaluation=None,
    continuation=None,
    pause_control=None,
    development_environment=None,
    task_program=None,
) -> NSBaselineResult:
    """Run a fixed-data NS control or a declared literature search adaptation.

    Each turn nominates exactly one final bundle; malformed/empty turns spend
    their turn budget and remain in the failure trace. Independent branches all
    start from S0 (or none) and empty history. Repair starts from the validation
    winner. Ties retain the earlier candidate. Output is a frozen Python bundle
    for external common testing, never a test-selected candidate.
    """
    if config.method == "codepde_repeated_sample_ns_adaptation" and initial_candidate is not None:
        raise ValueError("CodePDE repeated_sample requires cold start; S0 is not permitted")
    development, selection = tuple(development_records), tuple(selection_records)
    _validate_banks(development, selection)
    if initial_candidate is not None and (
        initial_candidate.kind is not TaskKind.SOLVER
        or not isinstance(initial_candidate.program, PythonCandidateProgram)
    ):
        raise TypeError("NS baseline S0 must be a frozen Python solver bundle")
    root = Path(config.output_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "baseline_plan.json").exists() and continuation is None:
        raise FileExistsError("baseline output already contains a plan; use a new directory")
    if continuation is not None and (root / "baseline_result.json").is_file():
        previous = json.loads((root / "baseline_result.json").read_text())
        if previous.get("status") == "failed":
            raise RuntimeError("terminal failed baseline cannot resume")
    if pause_control is not None and config.method not in ("competitive_refinement", *LITERATURE_METHODS):
        raise ValueError("segmented baseline runs require competition or a literature workflow")
    if development_environment is not None and config.method not in LITERATURE_METHODS:
        raise ValueError("paired online environments currently require a literature workflow")
    from pde_agent_verifier_solver.engine.campaign_continuation import CampaignPaused
    task, settings = task_program or NSSolverTask(), dict(task_settings or {})
    codec = PythonCandidateCodec(task_kind=TaskKind.SOLVER)
    def make_verifier():
        return HiddenVerifier(
            selection_evaluation if selection_evaluation is not None else task,
            evaluator_policy.default_query, settings, seed=config.seed,
            generator=generator, generator_state=generator_state,
            candidate_execution=candidate_execution,
        )
    borrow_agent_gpus = bool(getattr(agent_execution, "evaluation_device_map", None))
    verifier = None if borrow_agent_gpus else make_verifier()
    _write_json(root / "baseline_plan.json", {
        "schema": "ns_fixed_data_baseline_plan_v1", "config": config,
        "canonical_campaign": False,
        "selection_protocol": "fixed_hidden_validation_mean_primary_error",
        "terminal_selection": False,
        "development_case_ids": [row.case.case_id for row in development],
        "selection_case_ids": [row.case.case_id for row in selection],
        "evaluator_policy": evaluator_policy, "task_settings": settings,
        "planned_agent_turns": config.turn_count,
        "planned_turns_are_upper_bound": config.method in ("codepde_repeated_sample_ns_adaptation", *LITERATURE_METHODS),
        "planned_agent_wall_seconds": (LiteratureOptions(**dict(config.literature_options)).total_wall_seconds
                                       if config.method in LITERATURE_METHODS else
                                       config.turn_count * config.agent_budget_seconds),
        "development_environment": (development_environment.description if development_environment else "fixed_bank"),
        "time_budget_mode": "completion" if config.agent_budget_seconds is None else "hard_wall",
    })
    state = continuation if continuation is not None else {}
    candidates = state.setdefault("candidates", {})
    errors = state.setdefault("errors", {})
    histories = state.setdefault("histories", {})
    turns = state.setdefault("turns", [])
    selections = state.setdefault("selections", [])
    hidden_progress = state.setdefault("hidden_progress", {})
    if development_environment is not None:
        development_environment.bind(state.setdefault("development_environment", {}))
    turn_index = 0

    def checkpoint():
        if pause_control is not None:
            pause_control.save(state)

    def before_hidden_batch():
        if pause_control is not None and pause_control.should_pause(
            "baseline_selection", getattr(candidate_execution, "case_hard_wall_seconds", 0)
            + pause_control.hidden_start_margin_seconds,
        ):
            checkpoint()
            raise CampaignPaused(stage="baseline_selection")

    def register(artifact):
        if artifact.kind is not TaskKind.SOLVER or not isinstance(
            artifact.program, PythonCandidateProgram
        ):
            raise TypeError("baseline nominations must be frozen Python solver bundles")
        key = artifact.program.tree_sha256
        candidates.setdefault(key, artifact)
        return key

    def score(key):
        # Duplicate proposals still spend Agent turns; fixed validation is cached.
        if key in errors:
            return errors[key]
        if not borrow_agent_gpus:
            return score_in_scope(key, verifier, incremental=pause_control is not None)
        programs = (tuple(agent_program.values()) if isinstance(agent_program, Mapping)
                    else (agent_program,))
        pool = None
        try:
            # Lease and cleanup errors stay outside candidate-error handling.
            with agent_execution.reserve_evaluation_devices(programs) as pool:
                if pool is None:
                    raise AgentInfrastructureError("configured selection GPU pool is unavailable")
                device_event("evaluation_devices_borrowed", key, pool)
                with bind_evaluation_device_pool(pool):
                    # Construction captures the pool. Incremental state keeps
                    # plans/truth identical across leases and allocation resumes.
                    return score_in_scope(key, make_verifier(), incremental=True)
        finally:
            if pool is not None:
                device_event("evaluation_devices_returned" if pool.cleanup_confirmed
                             else "evaluation_devices_quarantined", key, pool)

    def device_event(event, key, pool):
        with (root / "evaluation_device_events.jsonl").open("a") as handle:
            handle.write(json.dumps({"event_type": event, "phase": "selection",
                                     "artifact_id": candidates[key].artifact_id,
                                     "devices": pool.devices,
                                     "cleanup_confirmed": pool.cleanup_confirmed}) + "\n")

    def score_in_scope(key, scoped_verifier, *, incremental):
        artifact, failure = candidates[key], None
        try:
            evaluation = (scoped_verifier.evaluate(artifact, selection) if not incremental else
                          scoped_verifier.evaluate_incremental(
                              artifact, selection, progress=hidden_progress,
                              before_batch=before_hidden_batch, checkpoint=checkpoint,
                              batch_size=(pause_control.hidden_batch_size if pause_control is not None
                                          else getattr(candidate_execution, "max_workers", 28)),
                          ))
            error = float(evaluation.primary_error)
            case_errors = [float(row.primary_error) for row in evaluation.cases]
            if tuple(row.case_id for row in evaluation.cases) != tuple(
                row.case.case_id for row in selection
            ):
                error, failure = math.inf, "incomplete_selection_results"
            elif not math.isfinite(error) or any(not math.isfinite(x) for x in case_errors):
                error, failure = math.inf, "nonfinite_selection_error"
        except CampaignPaused:
            raise
        except Exception as exc:
            from pde_agent_verifier_solver.engine.common_evaluation import _is_infrastructure_failure
            if _is_infrastructure_failure(exc):
                from pde_agent_verifier_solver.engine.candidate_execution import CandidateExecutionInfrastructureError
                if isinstance(exc, CandidateExecutionInfrastructureError):
                    raise
                raise CandidateExecutionInfrastructureError(
                    "Controller fixed-bank evaluation failed"
                ) from exc
            error, case_errors = math.inf, []
            failure = f"{type(exc).__name__}: {exc}"
        errors[key] = error
        selections.append({"tree_sha256": key, "artifact_id": artifact.artifact_id,
                           "primary_error": error, "case_errors": case_errors,
                           "failure": failure})
        _write_json(root / "selection_results.json", selections)
        return error

    def propose(parent, history, phase):
        nonlocal turn_index
        turn_index += 1
        turn_seed = int.from_bytes(sha256(
            f"ns-baseline-agent-v1:{config.seed}:{turn_index}".encode()
        ).digest()[:8], "big")
        agent = AgentSpec("ns-baseline", "NS baseline", config.prompt)
        sampling = DevelopmentSamplingPolicy(
            strategy=SamplingStrategy.FIXED_RECIPE,
            max_cases_per_agent_per_round=len(development),
            allowed_sources=(SampleSource.ORIGINAL,),
            fixed_recipe=CaseCounts(original=len(development)),
        )
        evaluator = AgentEvaluator(
            development, task, evaluator_policy, settings,
            seed=turn_seed, generator=generator, generator_state=generator_state,
            candidate_execution=candidate_execution,
        )
        prompt = "\n\n".join((
            config.prompt, task.agent_brief(settings),
            "This is an external fixed-data baseline. Use only the supplied "
            "development Cases and Evaluator. No online sampling is available. "
            "Nominate exactly one final Python solver bundle. The hidden "
            "validation bank selects candidates; terminal tests are not accessible.",
            f"Search phase: {phase}. Development feedback permissions are unchanged.",
        ))
        context = AgentContext(
            round_plan=RoundPlan(turn_index, (agent,), config.agent_budget_seconds,
                                 sampling, HiddenSuiteSpec(base=len(selection))),
            agent=agent, prompt=prompt, evaluator=evaluator,
            incumbent=None if parent is None else candidates[parent],
            memory=_memory(history), seed=turn_seed,
            public_problem_bundle=public_problem_bundle,
            candidate_solve_budget_seconds=getattr(
                candidate_execution, "case_hard_wall_seconds", None,
            ),
        )
        row = {"turn": turn_index, "phase": phase, "seed": turn_seed,
               "parent_tree_sha256": parent, "history_turns": len(history)}
        try:
            outcome = agent_execution.execute(agent_program, context, config.agent_budget_seconds)
            if not isinstance(outcome, AgentOutcome) or len(outcome.nominations) != 1:
                raise ValueError("a baseline turn must nominate exactly one final bundle")
            key = register(outcome.nominations[0])
            error = score(key)
            notes = outcome.round_log or "\n".join(outcome.observations)
            histories.setdefault(key, (*history, (notes, error if math.isfinite(error) else None)))
            row.update(status=outcome.execution_status, tree_sha256=key,
                       selection_error=error, observations=outcome.observations)
            return key
        except AgentInfrastructureError:
            raise
        except (AgentExecutionError, ValueError, TypeError) as exc:
            row.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
            return None
        finally:
            evaluator.cancel_pending_evaluations()
            turns.append(row)
            _write_json(root / "baseline_turns.json", turns)

    def better(current, proposed):
        if proposed is not None and (current is None or score(proposed) < score(current)):
            return proposed
        return current

    selected = None if initial_candidate is None else register(initial_candidate)
    initial_key = selected
    if continuation is not None and state.get("initial_key") != initial_key:
        raise ValueError("baseline resume initial Candidate hash differs")
    state["initial_key"] = initial_key
    if selected is not None:
        histories.setdefault(selected, ())
    try:
        if selected is not None:
            score(selected)
        if config.method in LITERATURE_METHODS:
            from .literature import run_literature_search
            selected, turn_index = run_literature_search(
                config, development=development, selection=selection, task=task,
                settings=settings, evaluator_policy=evaluator_policy,
                agent_program=agent_program, agent_execution=agent_execution,
                public_problem_bundle=public_problem_bundle, candidate_execution=candidate_execution,
                generator=generator, generator_state=generator_state, initial_key=selected,
                candidates=candidates, register=register, score=score, errors=errors,
                turns=turns, selections=selections, state=state.setdefault("literature", {}),
                checkpoint=checkpoint, pause_control=pause_control, environment=development_environment,
            )
        elif config.method == "competitive_refinement":
            from .competition import run_fixed_bank_competition
            selected, turn_index = run_fixed_bank_competition(
                config, development=development, selection=selection, task=task,
                settings=settings, evaluator_policy=evaluator_policy,
                agent_program=agent_program, agent_execution=agent_execution,
                public_problem_bundle=public_problem_bundle,
                candidate_execution=candidate_execution, generator=generator,
                generator_state=generator_state, selected=selected,
                candidates=candidates, register=register, score=score,
                turns=turns, selections=selections,
                continuation=state.setdefault("competition", {}),
                pause_control=pause_control, checkpoint=checkpoint,
            )
        elif config.method == "refinement":
            history = ()
            for _ in range(config.iterations):
                candidate = propose(selected, history, "refinement")
                selected = better(selected, candidate)
                if candidate is not None:
                    history = histories[candidate]
        elif config.method == "bon_repair":
            for _ in range(config.independent_candidates):
                selected = better(selected, propose(initial_key, (), "independent"))
            history = () if selected is None else histories[selected]
            for _ in range(config.repair_iterations):
                candidate = propose(selected, history, "repair")
                selected = better(selected, candidate)
                if candidate is not None:
                    history = histories[candidate]
        elif config.method == "codepde_repeated_sample_ns_adaptation":
            from .codepde import run_codepde_ns
            selected = run_codepde_ns(
                config, initial_key=initial_key, propose=propose, score=score,
                history_for=lambda key: histories.get(key, ()),
                failure_for=lambda key: next(
                    (row["failure"] for row in selections if row["tree_sha256"] == key), None,
                ),
            )
        elif config.method == "aide_ns_adaptation":
            from .aide import AIDESearchOptions, run_official_aide_search
            selected = run_official_aide_search(
                config, initial_key=initial_key, propose=propose, score=score,
                history_for=lambda key: histories.get(key, ()),
                options=AIDESearchOptions(
                    config.aide_num_drafts, config.aide_debug_prob, config.aide_max_debug_depth,
                ),
                diagnostics_for=lambda key: next(
                    (row["failure"] for row in selections if row["tree_sha256"] == key), None,
                ),
            )
        else:
            from .era import run_official_futs
            selected = run_official_futs(
                config, initial_key=initial_key, propose=propose, score=score,
                history_for=lambda key: histories.get(key, ()),
            )
        if selected is None or not math.isfinite(score(selected)):
            raise RuntimeError("no candidate passed fixed validation; no solver selected")
        source = candidates[selected]
        exported = codec.bind_existing(
            source.program.bundle_locator, storage_root=root / "result_bundle",
            artifact_id=source.artifact_id, name=source.name, summary=source.summary,
            usage=source.usage, round_index=source.round_index, agent_id=source.agent_id,
        )
        report = {
            "schema": "ns_fixed_data_baseline_result_v1", "status": "completed",
            "method": config.method, "selected_artifact_id": exported.artifact_id,
            "selected_bundle": str(exported.program.bundle_locator),
            "selected_tree_sha256": exported.program.tree_sha256,
            "selection_error": errors[selected], "selection_metric": "mean_primary_error",
            "selection_protocol": "fixed_hidden_validation", "terminal_score": None,
            "completed_agent_turns": len(turns), "unique_scored_candidates": len(errors),
            "failed_turns": sum(row.get("status") == "failed" or "failure" in row for row in turns),
        }
        _write_json(root / "baseline_result.json", report)
        return NSBaselineResult(exported, errors[selected], root, report)
    except CampaignPaused as exc:
        checkpoint()
        _write_json(root / "baseline_status.json", {
            "status": "paused", "phase": exc.stage,
            "completed_agent_turns": len(turns),
        })
        raise
    except BaseException as exc:
        _write_json(root / "baseline_result.json", {
            "schema": "ns_fixed_data_baseline_result_v1", "status": "failed",
            "method": config.method, "error": f"{type(exc).__name__}: {exc}",
            "completed_agent_turns": len(turns),
        })
        raise
    finally:
        for filename, service in (("execution_audits.json", agent_execution),
                                  ("candidate_execution_audits.json", candidate_execution)):
            audits = getattr(service, "audits", None)
            _write_json(root / filename, [] if audits is None else audits)

        _write_cost_report(root)
