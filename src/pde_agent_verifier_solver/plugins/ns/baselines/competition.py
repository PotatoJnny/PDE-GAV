"""Fixed-bank competition using the existing Agent batch and memory interfaces."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import math
from pathlib import Path
from typing import Mapping

from pde_agent_verifier_solver.engine.agent_execution import (
    AgentExecutionRequest, execute_agent_batch,
)
from pde_agent_verifier_solver.engine.campaign import (
    CaseCounts, DevelopmentSamplingPolicy, RoundPlan,
)
from pde_agent_verifier_solver.engine.evaluator import AgentEvaluator
from pde_agent_verifier_solver.engine.memory import (
    AgentRoundLog, CampaignMemoryLog, CandidateHistory,
)
from pde_agent_verifier_solver.engine.runtime import AgentContext, AgentOutcome
from pde_agent_verifier_solver.sdk import (
    HiddenSuiteSpec, PromotionMetricSummary, SampleSource, SamplingStrategy,
)


def run_fixed_bank_competition(
    config, *, development, selection, task, settings, evaluator_policy,
    agent_program, agent_execution, public_problem_bundle, candidate_execution,
    generator, generator_state, selected, candidates, register, score, turns, selections,
    continuation=None, pause_control=None, checkpoint=lambda: None,
):
    """Run simultaneous roles, then select on the same reused hidden bank.

    ``score`` belongs to the parent fixed-data composition and caches by frozen
    program hash. It owns one HiddenVerifier for the entire search, preserving
    identical validation Cases, query coordinates and truth for every round.
    """
    from .search import _write_json

    root = Path(config.output_directory).resolve()
    roles = config.competition_roles
    programs = agent_program if isinstance(agent_program, Mapping) else {
        role.agent_id: agent_program for role in roles
    }
    if set(programs) != {role.agent_id for role in roles}:
        raise ValueError("competition agent programs must match all configured role ids")
    from pde_agent_verifier_solver.engine.campaign_continuation import CampaignPaused
    progress = continuation if continuation is not None else {}
    if progress.get("phase") == "development_running":
        raise RuntimeError("cannot resume an interrupted Agent development budget")
    fresh = not progress
    memory = progress.setdefault("memory", CampaignMemoryLog())
    lineage = progress.setdefault("lineage", CandidateHistory())
    rounds = progress.setdefault("rounds", [])
    selected = progress.setdefault("selected", selected)
    if fresh and selected is not None:
        lineage.add_candidate(candidates[selected])

    def record_selection(artifact, key):
        error = score(key)
        if not lineage.needs_official_result(artifact.artifact_id):
            return error
        result = next(row for row in selections if row["tree_sha256"] == key)
        finite = math.isfinite(error)
        case_errors = sorted(result["case_errors"], reverse=True)
        tail_count = max(1, math.ceil(.2 * len(case_errors)))
        summaries = {"original": PromotionMetricSummary(
            len(selection), error,
            sum(case_errors[:tail_count]) / tail_count, max(case_errors),
        )} if finite else {}
        lineage.record_official_result(
            artifact.artifact_id, summaries,
            () if finite else (result["failure"] or "nonfinite_selection_error",),
            technically_eligible=finite,
        )
        return error

    if selected is not None:
        record_selection(candidates[selected], selected)

    sampling = DevelopmentSamplingPolicy(
        strategy=SamplingStrategy.FIXED_RECIPE,
        max_cases_per_agent_per_round=len(development),
        allowed_sources=(SampleSource.ORIGINAL,),
        fixed_recipe=CaseCounts(original=len(development)),
    )
    for round_index in range(progress.get("next_round", 1), config.competition_rounds + 1):
        incumbent = None if selected is None else candidates[selected]
        parent_key = selected
        plan = RoundPlan(round_index, roles, config.agent_budget_seconds,
                         sampling, HiddenSuiteSpec(original=len(selection)))
        active = progress.get("active")
        if active is None:
            if pause_control is not None and pause_control.should_pause(
                "baseline_development", config.agent_budget_seconds
                + pause_control.agent_start_margin_seconds,
            ):
                checkpoint()
                raise CampaignPaused(stage="baseline_development")
            requests, evaluators = [], []
            for role in roles:
                seed = int.from_bytes(sha256(
                    f"ns-fixed-competition-v1:{config.seed}:{round_index}:{role.agent_id}".encode()
                ).digest()[:8], "big")
                evaluator = AgentEvaluator(
                    development, task, evaluator_policy, settings, seed=seed,
                    generator=generator, generator_state=generator_state,
                    candidate_execution=candidate_execution,
                )
                evaluators.append(evaluator)
                context = AgentContext(
                    round_plan=plan, agent=role,
                    prompt="\n\n".join((
                        role.prompt, config.prompt,
                        "Task output requirements\n------------------------\n" + task.agent_brief(settings),
                        "This is a fixed-data solver development competition. Both Agents "
                        "receive the same fixed development bank and the same incumbent. "
                        "Use the supplied Cases and Evaluator; no online sampling is available. "
                        "You may submit multiple frozen Python solver bundles. After both Agents "
                        "finish, the incumbent and all valid nominations are compared on one "
                        "separate fixed hidden validation bank. Minimize mean primary error. "
                        "The winning program and both Agents' notes enter the next round. "
                        "Agent-authored notes are fallible research records; Controller "
                        "outcome blocks contain the Official results. "
                        "Terminal tests are not accessible.",
                    )), evaluator=evaluator, incumbent=incumbent,
                    memory=memory.snapshot(), candidate_history=lineage.snapshot(),
                    seed=seed, public_problem_bundle=public_problem_bundle,
                    candidate_solve_budget_seconds=getattr(
                        candidate_execution, "case_hard_wall_seconds", None,
                    ),
                )
                requests.append(AgentExecutionRequest(
                    programs[role.agent_id], context, config.agent_budget_seconds,
                ))
            _write_json(root / "baseline_status.json", {
                "status": "running", "phase": "development", "round": round_index,
                "completed_rounds": len(rounds), "completed_agent_turns": len(turns),
                "incumbent_tree_sha256": parent_key,
            })
            progress["phase"] = "development_running"
            checkpoint()
            try:
                results = execute_agent_batch(agent_execution, requests)
            finally:
                for evaluator in evaluators:
                    evaluator.cancel_pending_evaluations()
            # All Agent contexts are frozen before any outcome is scored or shared.
            logs, nominations = [], []
            for result in results:
                role, context = result.request.context.agent, result.request.context
                row = {"turn": len(turns) + 1, "round": round_index,
                       "agent_id": role.agent_id, "phase": "competitive_refinement",
                       "seed": context.seed, "parent_tree_sha256": parent_key,
                       "history_rounds": len(context.memory.rounds), "nominations": []}
                outcome = result.outcome
                if result.error is not None or not isinstance(outcome, AgentOutcome):
                    failure = (f"{type(result.error).__name__}: {result.error}" if result.error
                               else "TypeError: Agent returned no AgentOutcome")
                    row.update(status="failed", failure=failure)
                    logs.append(AgentRoundLog(round_index, role.agent_id, role.name,
                                              failure, status="failed"))
                else:
                    row.update(status=outcome.execution_status,
                               observations=outcome.observations, round_log=outcome.round_log)
                    logs.append(AgentRoundLog(
                        round_index, role.agent_id, role.name,
                        outcome.round_log or "\n".join(outcome.observations) or "No notes.",
                        status=outcome.execution_status,
                    ))
                    local_ids = {}
                    for index, proposed in enumerate(outcome.nominations, 1):
                        nomination = {"submitted_artifact_id": proposed.artifact_id}
                        try:
                            submitted_parent = proposed.parent_artifact_id
                            parent_id = local_ids.get(submitted_parent, submitted_parent)
                            if parent_id is None or not lineage.contains(parent_id):
                                parent_id = None if incumbent is None else incumbent.artifact_id
                            artifact = replace(
                                proposed,
                                artifact_id=f"baseline-r{round_index}-{role.agent_id}-n{index}",
                                round_index=round_index, agent_id=role.agent_id,
                                parent_artifact_id=parent_id,
                            )
                            key = register(artifact)
                            lineage.add_candidate(artifact)
                            local_ids[proposed.artifact_id] = artifact.artifact_id
                            nomination.update(artifact_id=artifact.artifact_id,
                                              tree_sha256=key, parent_artifact_id=parent_id,
                                              submitted_parent_artifact_id=submitted_parent)
                            nominations.append((artifact, key, nomination))
                        except (TypeError, ValueError) as exc:
                            nomination.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
                        row["nominations"].append(nomination)
                    if not outcome.nominations:
                        row["failure"] = "no_nominations"
                turns.append(row)
            _write_json(root / "baseline_turns.json", turns)
            _write_json(root / "candidate_history.json", lineage.snapshot())
            _write_json(root / "baseline_status.json", {
                "status": "running", "phase": "selection", "round": round_index,
                "completed_rounds": len(rounds), "completed_agent_turns": len(turns),
            })
            progress["active"] = {"logs": logs, "nominations": nominations}
            progress["phase"] = "development_done"
            checkpoint()
        else:
            logs, nominations = active["logs"], active["nominations"]
        for artifact, key, nomination in nominations:
            error = record_selection(artifact, key)
            nomination.update(selection_error=error,
                              status="eligible" if math.isfinite(error) else "ineligible")
            if math.isfinite(error) and (selected is None or error < score(selected)):
                selected = key
        if selected is not None:
            lineage.mark_selected(candidates[selected])
        selection_rows = ([{"tree_sha256": parent_key, "role": "incumbent",
                            "selection_error": score(parent_key)}] if parent_key else [])
        selection_rows.extend({"tree_sha256": key, "artifact_id": artifact.artifact_id,
                               "selection_error": score(key)}
                              for artifact, key, _ in nominations)
        summary = (
            "Fixed hidden validation; mean primary error, ties retain the incumbent "
            "or earlier nomination. Validation is reused across rounds and is not a test.\n\n"
            + "\n".join(f"- {row.get('artifact_id', 'Incumbent')} ({row['tree_sha256']}): "
                         f"{row['selection_error']!r}" for row in selection_rows)
            + f"\n\nSelected program hash: {selected}."
        )
        memory.append_round(round_index, logs, summary)
        rounds.append({"round": round_index, "incumbent_tree_sha256": parent_key,
                       "selected_tree_sha256": selected,
                       "selected_bundle": (None if selected is None else
                                           str(candidates[selected].program.bundle_locator)),
                       "selection": selection_rows,
                       "agent_ids": [role.agent_id for role in roles]})
        _write_json(root / "baseline_turns.json", turns)
        _write_json(root / "baseline_rounds.json", rounds)
        _write_json(root / "candidate_history.json", lineage.snapshot())
        _write_json(root / "campaign_memory.json", memory.snapshot())
        (root / "CAMPAIGN_MEMORY.md").write_text(memory.snapshot().markdown + "\n")
        progress.update(selected=selected, next_round=round_index + 1,
                        active=None, phase="round_done")
        checkpoint()
    _write_json(root / "baseline_status.json", {
        "status": "rounds_completed", "completed_rounds": len(rounds),
        "completed_agent_turns": len(turns), "selected_tree_sha256": selected,
    })
    return selected, len(turns)
