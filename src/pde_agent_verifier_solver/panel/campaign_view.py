"""Product-facing view models for live and completed PDE campaigns.

This module deliberately returns scientific progress and result metrics only.
Cluster paths, commands, scheduler ids, and raw log text never enter the panel API.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
from typing import Any, Mapping

from pde_agent_verifier_solver.engine.staged import read_staged_run


_PHASES = (
    ("data", "Dataset & representation"),
    ("generator", "Train Generator"),
    ("campaign", "Agent–Generator rounds"),
    ("result", "Final evaluation"),
)

_STAGE_ACTIVITY = {
    "macro_agent_round": "Agent is improving and validating the macroscopic model",
    "project_train": "Mapping training cases into the continuous basis",
    "project_validation": "Mapping validation cases into the continuous basis",
    "project_hidden_reference": "Preparing held-out reference cases",
    "fit_pca": "Learning the reduced coefficient basis",
    "select_pca": "Choosing the coefficient basis from physical validation",
    "train_generator": "Training the Base Generator",
    "audit_generator": "Comparing Generator checkpoints on validation cases",
    "select_generator": "Choosing the validated Base Generator",
    "prepare_hidden": "Building the fixed hidden reference suite",
    "prepare_initial_hidden": "Building the fixed hidden reference suite",
    "adversarial_rounds": "Agent and Generator are improving the solver",
    "prepare_campaign": "Preparing public Agent workspaces and Evaluator settings",
    "agent_a": "Agent A is improving a candidate with the Generator and Evaluator",
    "agent_b": "Agent B is improving a candidate with the Generator and Evaluator",
    "final_evaluation": "Comparing the final candidates on fresh hidden cases",
    "publish_result": "Packaging the selected scientific program",
}

_EVENT_ACTIVITY = {
    "solver_workspace_prepared": "Agent is reviewing the new challenge cases",
    "hard_physical_verifier": "Checking generated cases against the physics contract",
    "solver_agent": "Agent produced a solver candidate",
    "generated_current_solver_eval": "Evaluating the candidate on new cases",
    "generated_cumulative_solver_eval": "Evaluating the candidate on replay cases",
    "hidden_continuous_holdout_solver_eval": "Checking generalization on hidden cases",
    "reward_tables_written": "Preparing solver feedback for the Generator",
    "diffusion_policy_update": "Updating guided generation from solver weaknesses",
    "round_summary_written": "Round evaluation is complete",
    "compact_memory_updated": "Saving what the Agent learned this round",
    "final_summary_written": "Final solver result is ready",
}


def build_dashboard_snapshot(runs_root: Path | str) -> dict[str, object]:
    """Build the complete payload consumed by the local and hosted panels."""

    root = Path(runs_root).resolve()
    runs: list[dict[str, object]] = []
    for run_root in _run_directories(root):
        try:
            runs.append(build_campaign_view(run_root))
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue

    status_order = {"running": 0, "submitted": 1, "failed": 2, "completed": 3}
    runs.sort(key=lambda row: str(row.get("updated_at") or ""), reverse=True)
    runs.sort(key=lambda row: status_order.get(str(row.get("status")), 4))
    counts = {
        status: sum(row.get("status") == status for row in runs)
        for status in ("running", "completed", "failed", "submitted")
    }
    return {
        "schema": "pde_panel_snapshot_v1",
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
        "runs": runs,
    }


def build_hosted_dashboard_snapshot(runs_root: Path | str) -> dict[str, object]:
    """Build the small metric-only snapshot allowed to leave the cluster.

    The hosted panel receives campaign names, progress, and numerical summaries.
    Solver source and original case identifiers stay on the cluster.
    """

    snapshot = json.loads(
        json.dumps(
            build_dashboard_snapshot(runs_root),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    for run in snapshot["runs"]:
        result = run.get("result")
        if not isinstance(result, dict):
            continue
        artifact = result.get("artifact")
        if isinstance(artifact, dict):
            source_available = bool(artifact.get("source"))
            result["artifact"] = {
                "available": False,
                "available_on_cluster": source_available,
            }
        for key in ("hidden_distribution", "generated_distribution"):
            rows = result.get(key)
            if not isinstance(rows, list):
                continue
            for position, row in enumerate(rows, start=1):
                if isinstance(row, dict):
                    row["case"] = f"Case {position}"
    return snapshot


def build_campaign_view(run_root: Path | str) -> dict[str, object]:
    """Turn one staged run folder into a human-facing campaign view."""

    root = Path(run_root).resolve()
    run_file = _read_json(root / "run.json")
    staged = read_staged_run(root, log_lines=0)
    stages = [dict(row) for row in staged.get("stages", [])]
    status = str(staged.get("status", "submitted"))
    settings = _campaign_settings(run_file, root)
    rounds = _round_views(root)
    rounds_completed = sum(row.get("status") == "completed" for row in rounds)
    standalone_result = staged.get("result")
    if status == "completed" and not rounds and isinstance(
        standalone_result, Mapping
    ):
        rounds_completed = (
            _integer(standalone_result.get("rounds_completed")) or 1
        )
    rounds_requested = int(settings.get("rounds") or max(rounds_completed, 1))
    stage_views = [
        _stage_view(root, stage, rounds_completed, rounds_requested)
        for stage in stages
    ]
    phases = _phase_views(stage_views)
    completed_units = sum(float(row["progress"]) for row in stage_views)
    progress = completed_units / len(stage_views) if stage_views else 0.0
    if status == "completed":
        progress = 1.0

    active_stage = next(
        (row for row in stage_views if row["status"] in {"running", "failed"}),
        next((row for row in stage_views if row["status"] == "submitted"), None),
    )
    current_phase = (
        str(active_stage["phase_name"])
        if active_stage
        else ("Complete" if status == "completed" else "Waiting to start")
    )
    current_activity = (
        str(active_stage["activity"])
        if active_stage
        else ("Final result is ready" if status == "completed" else "Waiting for compute")
    )
    round_activity = _current_round_activity(root, rounds_completed, rounds_requested)
    if active_stage and active_stage["phase"] == "campaign" and round_activity:
        current_activity = round_activity

    result = _result_view(root, staged.get("result"), rounds, settings)
    summary = _campaign_summary(rounds, result)
    created_at = str(run_file.get("created_at") or "")
    updated_at = _updated_at(root, stages, created_at)
    finished_at = max(
        (
            str(row.get("finished_at") or "")
            for row in stages
            if row.get("finished_at")
        ),
        default="",
    )
    elapsed_seconds = _elapsed_seconds(created_at, finished_at or None)

    problem_id = str(run_file.get("problem") or staged.get("problem") or "PDE")
    problem = _problem_label(problem_id, staged.get("result"))
    result_payload = staged.get("result") or {}
    task_kind = str(
        result_payload.get("kind")
        or run_file.get("task")
        or _default_task_kind(problem_id)
    )
    view: dict[str, object] = {
        "id": root.name,
        "name": str(run_file.get("name") or root.name),
        "problem": problem,
        "problem_id": str(run_file.get("problem") or ""),
        "task": _task_label(task_kind),
        "status": status,
        "created_at": created_at or None,
        "updated_at": updated_at or None,
        "finished_at": finished_at or None,
        "elapsed_seconds": elapsed_seconds,
        "progress": round(progress, 4),
        "progress_percent": round(progress * 100),
        "current_phase": current_phase,
        "current_activity": current_activity,
        "rounds_completed": rounds_completed,
        "rounds_requested": rounds_requested,
        "cases_per_round": settings.get("cases_per_round"),
        "phases": phases,
        "live": _live_view(root, rounds, settings, current_activity),
        "summary": summary,
        "result": result,
    }
    if status == "failed":
        view["issue"] = _friendly_issue(root, active_stage)
    return view


def _campaign_settings(run: Mapping[str, object], root: Path) -> dict[str, object]:
    final = _read_json(
        root / "campaign" / "agent_adversarial_loop" / "final_summary.json",
        default={},
    )
    pca_selection = _read_json(
        root / "artifacts" / "pca" / "pca_selection.json", default={}
    )
    flow_selection = _read_json(
        root / "artifacts" / "base_generator" / "flow_selection.json", default={}
    )
    rounds = (
        _integer(final.get("rounds_requested"))
        or _integer(_argument(run, "--campaign-rounds-total"))
        or _integer(_argument(run, "--rounds"))
    )
    cases_per_round = _integer(final.get("cases_per_round")) or _integer(
        _argument(run, "--cases-per-round")
    )
    fixed_hidden_cases = _integer(final.get("hidden_holdout_cases")) or _integer(
        _argument(run, "--hidden-holdout-cases")
    )
    return {
        "rounds": rounds or 1,
        "cases_per_round": cases_per_round,
        "hidden_cases": fixed_hidden_cases,
        "hidden_initial_cases": fixed_hidden_cases,
        "hidden_cases_per_round": _integer(
            final.get("hidden_generated_cases_per_round")
        )
        or _integer(_argument(run, "--hidden-generated-cases-per-round")),
        "hidden_accumulate": bool(final.get("hidden_accumulate"))
        or _argument(run, "--hidden-accumulate") is not None,
        "train_cases": _integer(_argument_for_stage(run, "project_train", "--max-cases")),
        "validation_cases": _integer(
            _argument_for_stage(run, "project_validation", "--max-cases")
        ),
        "basis_rank": _integer(pca_selection.get("selected_rank"))
        or _integer(_argument_for_stage(run, "train_generator", "--rank")),
        "selected_generator_step": _integer(flow_selection.get("selected_step")),
        "generator_selection_rule": flow_selection.get("rule"),
        "generator_baseline_trials": _integer(
            flow_selection.get("baseline_trial_count")
        ),
        "generator_candidates": [
            {
                "step": _integer(row.get("step")),
                "eligible": bool(row.get("eligible")),
            }
            for row in flow_selection.get("candidates", [])
            if isinstance(row, Mapping)
        ],
        "generator_steps": _integer(
            _argument_for_stage(run, "train_generator", "--steps")
        ),
        "original_cases": _integer(final.get("original_cases_per_round"))
        or _integer(_argument(run, "--original-cases-per-round")),
        "unperturbed_cases": _integer(_argument(run, "--unperturbed-cases")),
        "perturbed_cases": _integer(_argument(run, "--perturbation-cases")),
        "replay_cases": _integer(_argument(run, "--replay-perturbation-cases")),
        "guided": bool(final.get("policy_update_enabled"))
        or _argument(run, "--policy-update-command") is not None
        or _argument(run, "--update-policy-command") is not None,
        "generation_batch_size": _integer(
            _argument(run, "--generation-batch-size")
        ),
        "guidance_pool_size": _integer(
            _argument(run, "--guidance-pool-size")
        ),
    }


def _stage_view(
    root: Path,
    stage: Mapping[str, object],
    rounds_completed: int,
    rounds_requested: int,
) -> dict[str, object]:
    stage_id = str(stage.get("id") or stage.get("stage_id") or "")
    status = str(stage.get("status") or "submitted")
    phase = _stage_phase(stage_id)
    progress = 1.0 if status == "completed" else 0.0
    detail = None

    if status == "running":
        if stage_id.startswith("project_"):
            split = stage_id.removeprefix("project_")
            completed, total = _projection_progress(root, split)
            if total:
                progress = min(completed / total, 1.0)
                detail = f"{completed:,} of {total:,} cases"
        elif stage_id == "train_generator":
            step, total, validation_loss = _generator_progress(root)
            if total:
                progress = min(step / total, 1.0)
                detail = f"Step {step:,} of {total:,}"
                if validation_loss is not None:
                    detail += f" · validation loss {validation_loss:.4g}"
        elif phase == "campaign" and rounds_requested:
            stage_round = _stage_round_index(stage_id)
            if stage_round is None:
                subprogress = _round_subprogress(root, rounds_completed)
                progress = min(
                    (rounds_completed + subprogress) / rounds_requested, 1.0
                )
                shown_round = min(rounds_completed + 1, rounds_requested)
            else:
                progress = _round_subprogress(root, stage_round)
                shown_round = min(stage_round + 1, rounds_requested)
            detail = f"Round {shown_round} of {rounds_requested}"

    return {
        "id": stage_id,
        "phase": phase,
        "phase_name": dict(_PHASES).get(phase, "Campaign"),
        "status": status,
        "progress": round(progress, 4),
        "activity": _stage_activity(stage_id, stage),
        "detail": detail,
    }


def _phase_views(stages: list[dict[str, object]]) -> list[dict[str, object]]:
    views = []
    for phase_id, title in _PHASES:
        members = [row for row in stages if row["phase"] == phase_id]
        if not members:
            continue
        statuses = [str(row["status"]) for row in members]
        if "failed" in statuses:
            status = "failed"
        elif all(value == "completed" for value in statuses):
            status = "completed"
        elif "running" in statuses:
            status = "running"
        else:
            status = "submitted"
        views.append(
            {
                "id": phase_id,
                "name": title,
                "status": status,
                "progress": round(
                    sum(float(row["progress"]) for row in members) / len(members), 4
                ),
                "detail": next(
                    (row["detail"] for row in members if row.get("detail")), None
                ),
            }
        )
    return views


def _round_views(root: Path) -> list[dict[str, object]]:
    rounds_root = root / "campaign" / "agent_adversarial_loop" / "rounds"
    rows: list[dict[str, object]] = []
    if not rounds_root.is_dir():
        return rows
    for directory in sorted(rounds_root.glob("round_*")):
        if not directory.is_dir():
            continue
        summary = _read_json(directory / "round_summary.json", default={})
        index = _integer(summary.get("round"))
        if index is None:
            try:
                index = int(directory.name.rsplit("_", 1)[-1])
            except ValueError:
                continue
        current = _metric_summary(
            summary.get("generated_summary")
            or _read_json(directory / "generated_solver_metrics.json", default={})
        )
        previous = _metric_summary(
            summary.get("previous_solver_generated_summary")
            or _read_json(directory / "previous_solver_generated_metrics.json", default={})
        )
        hidden_fixed = _metric_summary(
            summary.get("hidden_holdout_summary")
            or _read_json(directory / "hidden_holdout_solver_metrics.json", default={})
        )
        hidden_current = _metric_summary(
            summary.get("hidden_generated_current_summary")
            or _read_json(
                directory / "hidden_generated_current_solver_metrics.json", default={}
            )
        )
        hidden_cumulative = _metric_summary(
            summary.get("hidden_generated_cumulative_summary")
            or _read_json(
                directory / "hidden_generated_cumulative_solver_metrics.json",
                default={},
            )
        )
        hidden = _metric_summary(
            summary.get("hidden_all_summary")
            or _read_json(directory / "hidden_all_solver_metrics.json", default={})
        ) or hidden_fixed
        cumulative = _metric_summary(
            summary.get("cumulative_generated_summary")
            or _read_json(directory / "cumulative_generated_solver_metrics.json", default={})
        )
        verification = summary.get("hard_verification_summary") or {}
        rows.append(
            {
                "round": index + 1,
                "status": "completed" if summary else "running",
                "generated_cases": _integer(summary.get("generated_case_count"))
                or _integer(current.get("n")),
                "replay_cases": _integer(summary.get("historical_replay_case_count"))
                or 0,
                "feedback_cases": _integer(summary.get("solver_feedback_case_count")),
                "case_mix": summary.get("case_mix")
                if isinstance(summary.get("case_mix"), Mapping)
                else None,
                "current": current or None,
                "previous": previous or None,
                "hidden": hidden or None,
                "hidden_fixed": hidden_fixed or None,
                "hidden_current": hidden_current or None,
                "hidden_cumulative": hidden_cumulative or None,
                "cumulative": cumulative or None,
                "improvement": _number(summary.get("solver_improvement_on_generated")),
                "cumulative_improvement": _number(
                    summary.get("solver_improvement_on_cumulative_generated")
                ),
                "physics": {
                    "accepted": _integer(verification.get("accepted")) or 0,
                    "total": _integer(verification.get("n")) or 0,
                },
                "guided_update": bool((summary.get("policy_update") or {}).get("enabled")),
            }
        )
    return rows


def _live_view(
    root: Path,
    rounds: list[dict[str, object]],
    settings: Mapping[str, object],
    activity: str,
) -> dict[str, object]:
    generator = _generator_view(root, settings)
    current_round = next(
        (int(row["round"]) for row in rounds if row["status"] == "running"),
        min(len(rounds) + 1, int(settings.get("rounds") or 1)),
    )
    latest = next(
        (row for row in reversed(rounds) if row.get("current")),
        None,
    )
    events = []
    for row in rounds[-3:]:
        if row.get("status") != "completed":
            continue
        current = (row.get("current") or {}).get("mean_relative_l2")
        hidden = (row.get("hidden") or {}).get("mean_relative_l2")
        text = f"Round {row['round']} evaluated {row.get('generated_cases') or 0} new cases"
        if current is not None:
            text += f" · solver error {current:.4f}"
        events.append({"round": row["round"], "title": text, "metric": hidden})
    return {
        "current_round": current_round,
        "rounds_requested": settings.get("rounds"),
        "activity": activity,
        "latest_metrics": {
            "solver_error": (latest.get("current") or {}).get("mean_relative_l2")
            if latest
            else None,
            "previous_error": (latest.get("previous") or {}).get("mean_relative_l2")
            if latest
            else None,
            "hidden_error": (latest.get("hidden") or {}).get("mean_relative_l2")
            if latest
            else None,
        },
        "generator": generator,
        "case_mix": {
            "original": settings.get("original_cases"),
            "generated": settings.get("unperturbed_cases"),
            "base_or_guided": settings.get("unperturbed_cases"),
            "perturbed": settings.get("perturbed_cases"),
            "replay": settings.get("replay_cases"),
            "guided": settings.get("guided"),
        },
        "hidden_suite": {
            "fixed_reference": settings.get("hidden_initial_cases"),
            "new_per_round": settings.get("hidden_cases_per_round"),
            "accumulate": settings.get("hidden_accumulate"),
        },
        "recent_events": events,
    }


def _result_view(
    root: Path,
    raw_result: object,
    rounds: list[dict[str, object]],
    settings: Mapping[str, object],
) -> dict[str, object] | None:
    if not isinstance(raw_result, Mapping):
        return None
    completed = [row for row in rounds if row.get("status") == "completed"]
    if not completed:
        return _standalone_result_view(root, raw_result, settings)
    final = completed[-1]
    initial_previous = (
        (completed[0].get("previous") or {}).get("mean_relative_l2")
        if completed
        else None
    )
    final_current = (final.get("current") or {}).get("mean_relative_l2")
    overall_improvement = None
    if initial_previous and final_current is not None:
        overall_improvement = (initial_previous - final_current) / initial_previous

    final_round_root = (
        root / "campaign" / "agent_adversarial_loop" / "rounds"
        / f"round_{int(final['round']) - 1:03d}"
    )
    hidden_metrics = _read_json(
        final_round_root / "hidden_all_solver_metrics.json", default={}
    ) or _read_json(
        final_round_root / "hidden_holdout_solver_metrics.json", default={}
    )
    generated_metrics = _read_json(
        final_round_root / "cumulative_generated_solver_metrics.json",
        default={},
    )
    final_summary = _read_json(
        root / "campaign" / "agent_adversarial_loop" / "final_summary.json",
        default={},
    )
    solver_source = None
    solver_path = root / "result" / "solver" / "submission.py"
    if solver_path.is_file() and solver_path.stat().st_size <= 500_000:
        solver_source = solver_path.read_text(encoding="utf-8", errors="replace")
    physics_accepted = sum(int((row.get("physics") or {}).get("accepted") or 0) for row in completed)
    physics_total = sum(int((row.get("physics") or {}).get("total") or 0) for row in completed)

    return {
        "kind": str(raw_result.get("kind") or "solver"),
        "name": str(raw_result.get("name") or "Final result"),
        "primary_error": _number(raw_result.get("primary_error")),
        "overall_improvement": _number(overall_improvement),
        "rounds": completed,
        "hidden_distribution": _errors(hidden_metrics),
        "generated_distribution": _errors(generated_metrics),
        "physics": {"accepted": physics_accepted, "total": physics_total},
        "wall_time_seconds": _number(final_summary.get("wall_time_seconds")),
        "generator": _generator_view(root, settings),
        "memory": _memory_view(root),
        "artifact": {
            "available": solver_source is not None,
            "filename": "submission.py",
            "source": solver_source,
            "usage": "Import solve_case from submission.py to evaluate a PDE case.",
        },
    }


def _standalone_result_view(
    root: Path,
    raw_result: Mapping[str, object],
    settings: Mapping[str, object],
) -> dict[str, object]:
    """Build a Result for campaigns that do not use sequential round folders.

    VPFP runs two Agents in parallel and stores their common comparison in one
    formal summary, so forcing it into the NS round-file layout would make the
    panel misleading.  This view keeps the same Result page while retaining
    the experiment's actual shape.
    """

    formal = _read_json(
        root / "artifacts" / "formal_evaluation" / "formal_summary.json",
        default={},
    )
    model_path = root / "result" / "model" / "submission.py"
    if not model_path.is_file():
        model_path = root / "result" / "solver" / "submission.py"
    source = None
    if model_path.is_file() and model_path.stat().st_size <= 500_000:
        source = model_path.read_text(encoding="utf-8", errors="replace")

    memory = raw_result.get("memory")
    memory_note = memory.get("note") if isinstance(memory, Mapping) else None
    winner = str(raw_result.get("winner") or "")
    insights = []
    if winner:
        insights.append(f"{winner} supplied the selected final program.")
    if memory_note:
        insights.append(str(memory_note))

    broad_cases = _integer(formal.get("broad_root_case_count")) or 0
    frontier_cases = _integer(formal.get("frontier_root_case_count")) or 0
    bundle_files = []
    if model_path.is_file():
        bundle_files = sorted(
            path.name
            for path in model_path.parent.iterdir()
            if path.is_file() and not path.is_symlink()
        )

    terminal_acceptance = raw_result.get("terminal_acceptance")
    if not isinstance(terminal_acceptance, bool):
        raw_terminal = raw_result.get("terminal")
        terminal_acceptance = (
            raw_terminal.get("supports_selected_over_cold_start")
            if isinstance(raw_terminal, Mapping)
            and isinstance(
                raw_terminal.get("supports_selected_over_cold_start"),
                bool,
            )
            else None
        )
    raw_overall_improvement = _number(raw_result.get("overall_improvement"))
    result: dict[str, object] = {
        "kind": str(raw_result.get("kind") or "custom"),
        "name": str(raw_result.get("name") or "Final result"),
        "primary_error": _number(raw_result.get("primary_error")),
        "overall_improvement": (
            None if terminal_acceptance is False else raw_overall_improvement
        ),
        "active_objective_improvement": _number(
            raw_result.get("active_objective_improvement")
        ),
        "terminal_acceptance": terminal_acceptance,
        "rounds": [],
        "hidden_distribution": [],
        "generated_distribution": [],
        "physics": {"accepted": 0, "total": 0},
        "wall_time_seconds": _number(
            raw_result.get("wall_time_seconds")
            if raw_result.get("wall_time_seconds") is not None
            else formal.get("wall_time_seconds")
        ),
        "evaluated_cases": (
            _integer(raw_result.get("evaluated_cases"))
            if raw_result.get("evaluated_cases") is not None
            else broad_cases + frontier_cases
        ),
        "generator": _generator_view(root, settings),
        "memory": {
            "rounds_summarized": 1 if insights else 0,
            "insights": insights,
        },
        "comparison": raw_result.get("evaluation", {}),
        "artifact": {
            "available": source is not None,
            "filename": (
                model_path.name if model_path.is_file() else "submission.py"
            ),
            "bundle_files": bundle_files,
            "source": source,
            "usage": str(
                raw_result.get("usage")
                or "See the Task usage instructions."
            ),
        },
    }
    paired_objective = _paired_objective_view(raw_result)
    if paired_objective is not None:
        result["paired_objective"] = paired_objective
    terminal = _standalone_terminal_view(raw_result)
    if terminal is not None:
        result["terminal"] = terminal
    promotion_history = _promotion_history_view(raw_result)
    if promotion_history is not None:
        result["promotion_history"] = promotion_history
    return result


def _promotion_history_view(
    raw_result: Mapping[str, object],
) -> dict[str, object] | None:
    """Project complete promotion evidence into a compact customer-facing view."""

    raw_rounds = raw_result.get("rounds")
    if not isinstance(raw_rounds, list):
        return None
    policy = raw_result.get("controller_policy")
    program = (
        policy.get("promotion_program")
        if isinstance(policy, Mapping)
        and isinstance(policy.get("promotion_program"), str)
        else None
    )
    distribution_order = (
        "original",
        "base",
        "guided",
        "base_perturbed",
        "guided_perturbed",
        "replay",
    )
    rows: list[dict[str, object]] = []
    for position, raw_round in enumerate(raw_rounds, start=1):
        if not isinstance(raw_round, Mapping):
            continue
        raw_reports = raw_round.get("promotion_reports")
        decision = raw_round.get("promotion_decision")
        if not isinstance(raw_reports, list) or not isinstance(decision, Mapping):
            continue
        reports: list[dict[str, object]] = []
        labels_by_digest: dict[str, str] = {}
        for candidate_position, raw_report in enumerate(raw_reports, start=1):
            if not isinstance(raw_report, Mapping):
                continue
            candidate_id = raw_report.get("candidate_id")
            label = (
                str(candidate_id)
                if isinstance(candidate_id, str) and candidate_id
                else f"Candidate {candidate_position}"
            )
            digest = raw_report.get("candidate_sha256")
            if isinstance(digest, str) and digest:
                labels_by_digest[digest] = label
            raw_distributions = raw_report.get("distributions")
            distributions: list[dict[str, object]] = []
            if isinstance(raw_distributions, Mapping):
                for name in distribution_order:
                    metric = raw_distributions.get(name)
                    if not isinstance(metric, Mapping):
                        continue
                    distributions.append(
                        {
                            "name": name,
                            "case_count": _integer(metric.get("case_count")),
                            "mean": _number(metric.get("mean")),
                            "cvar": _number(metric.get("cvar")),
                            "max": _number(metric.get("max")),
                        }
                    )
            failures = raw_report.get("failures")
            reports.append(
                {
                    "candidate": label,
                    "technically_eligible": bool(
                        raw_report.get("technically_eligible")
                    ),
                    "distributions": distributions,
                    "failures": [
                        item for item in failures if isinstance(item, str)
                    ]
                    if isinstance(failures, list)
                    else [],
                }
            )
        selected_digest = decision.get("selected_sha256")
        selected = (
            labels_by_digest.get(selected_digest, "Selected candidate")
            if isinstance(selected_digest, str)
            else "Selected candidate"
        )
        ranking = []
        raw_ranking = decision.get("ranking")
        if isinstance(raw_ranking, list):
            for raw_entry in raw_ranking:
                if not isinstance(raw_entry, Mapping):
                    continue
                digest = raw_entry.get("candidate_sha256")
                ranking.append(
                    {
                        "candidate": labels_by_digest.get(digest, "Candidate")
                        if isinstance(digest, str)
                        else "Candidate",
                        "rank": _integer(raw_entry.get("rank")),
                        "eligible": bool(raw_entry.get("eligible")),
                        "reason": str(raw_entry.get("reason") or ""),
                    }
                )
        round_index = _integer(raw_round.get("round_index"))
        rows.append(
            {
                "round": (round_index + 1) if round_index is not None else position,
                "selected": selected,
                "promoted": bool(decision.get("promoted")),
                "reason": str(decision.get("reason") or ""),
                "reports": reports,
                "ranking": ranking,
            }
        )
    if not rows:
        return None
    return {"program": program, "rounds": rows}


def _paired_objective_view(
    raw_result: Mapping[str, object],
) -> dict[str, object] | None:
    macro_objective = raw_result.get("macro_objective")
    if not isinstance(macro_objective, Mapping):
        return None
    paired = macro_objective.get("paired_objective")
    if not isinstance(paired, Mapping):
        return None
    mode = paired.get("mode")
    if not isinstance(mode, str) or not mode:
        return None
    return {
        "mode": mode,
        "base_weight": _number(paired.get("base_weight")),
        "perturbed_weight": _number(paired.get("perturbed_weight")),
        "guardrail_relative_margin": _number(
            paired.get("guardrail_relative_margin")
        ),
        "guardrail_confidence_z": _number(
            paired.get("guardrail_confidence_z")
        ),
    }


def _standalone_terminal_view(
    raw_result: Mapping[str, object],
) -> dict[str, object] | None:
    terminal = raw_result.get("terminal")
    if not isinstance(terminal, Mapping):
        return None
    active_objective = terminal.get("active_objective")
    if not isinstance(active_objective, str) or not active_objective:
        active_objective = None
    distributions = _terminal_distribution_aggregates_view(
        terminal.get("distribution_aggregates")
    )
    guardrail = _perturbed_guardrail_view(
        terminal.get("perturbed_guardrail")
    )
    supports_selected = terminal.get("supports_selected_over_cold_start")
    if not isinstance(supports_selected, bool):
        supports_selected = None
    if (
        active_objective is None
        and distributions is None
        and guardrail is None
        and supports_selected is None
    ):
        return None
    return {
        "active_objective": active_objective,
        "distribution_aggregates": distributions,
        "perturbed_guardrail": guardrail,
        "supports_selected_over_cold_start": supports_selected,
    }


def _terminal_distribution_aggregates_view(
    value: object,
) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[str, object] = {}
    for candidate_name in ("cold_start", "selected"):
        candidate = value.get(candidate_name)
        if not isinstance(candidate, Mapping):
            continue
        distributions: dict[str, object] = {}
        for distribution_name in ("generated", "perturbed", "composite"):
            aggregate = _robust_aggregate_view(candidate.get(distribution_name))
            if aggregate is not None:
                distributions[distribution_name] = aggregate
        if distributions:
            result[candidate_name] = distributions
    return result or None


def _robust_aggregate_view(value: object) -> dict[str, float | None] | None:
    if not isinstance(value, Mapping):
        return None
    result = {
        "mean": _number(value.get("mean")),
        "cvar": _number(value.get("cvar")),
        "max": _number(value.get("max")),
    }
    return result if any(item is not None for item in result.values()) else None


def _perturbed_guardrail_view(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    passed = value.get("passed")
    failure_values = value.get("failures")
    failures = (
        [item for item in failure_values if isinstance(item, str)]
        if isinstance(failure_values, list)
        else []
    )
    return {
        "root_count": _integer(value.get("root_count")),
        "candidate_mean": _number(value.get("candidate_mean")),
        "incumbent_mean": _number(value.get("incumbent_mean")),
        "mean_excess": _number(value.get("mean_excess")),
        "standard_error": _number(value.get("standard_error")),
        "confidence_lower": _number(value.get("confidence_lower")),
        "confidence_z": _number(value.get("confidence_z")),
        "practical_relative_margin": _number(
            value.get("practical_relative_margin")
        ),
        "passed": passed if isinstance(passed, bool) else None,
        "failures": failures,
    }



def _campaign_summary(
    rounds: list[dict[str, object]],
    result: dict[str, object] | None,
) -> dict[str, object]:
    latest = next((row for row in reversed(rounds) if row.get("current")), None)
    hidden = (latest.get("hidden") or {}) if latest else {}
    current = (latest.get("current") or {}) if latest else {}
    return {
        "primary_error": result.get("primary_error") if result else None,
        "hidden_error": hidden.get("mean_relative_l2")
        or (result.get("primary_error") if result else None),
        "generated_error": current.get("mean_relative_l2"),
        "improvement": result.get("overall_improvement") if result else (
            latest.get("improvement") if latest else None
        ),
        "evaluated_cases": (
            sum(int(row.get("generated_cases") or 0) for row in rounds)
            or (int(result.get("evaluated_cases") or 0) if result else 0)
        ),
    }


def _generator_view(root: Path, settings: Mapping[str, object]) -> dict[str, object]:
    pca = _read_json(root / "artifacts" / "pca" / "pca_audit.json", default={})
    ranks = []
    for row in pca.get("ranks", []) if isinstance(pca.get("ranks"), list) else []:
        if not isinstance(row, Mapping):
            continue
        ranks.append(
            {
                "rank": _integer(row.get("rank")),
                "explained_variance": _number(row.get("explained_variance_fraction")),
            }
        )
    train_events = _read_jsonl(root / "artifacts" / "base_generator" / "metrics.jsonl")
    by_step: dict[int, dict[str, object]] = {}
    for event in train_events:
        step = _integer(event.get("step"))
        loss = _number(event.get("loss"))
        if step is None or loss is None:
            continue
        row = by_step.setdefault(step, {"step": step})
        if event.get("event") == "validation":
            row["validation_loss"] = loss
        elif event.get("event") == "train":
            row["train_loss"] = loss
    step, total, validation_loss = _generator_progress(root)
    return {
        "basis_rank": settings.get("basis_rank"),
        "selected_checkpoint_step": settings.get("selected_generator_step"),
        "basis": ranks,
        "train_cases": settings.get("train_cases"),
        "validation_cases": settings.get("validation_cases"),
        "training_step": step,
        "training_steps": total,
        "validation_loss": validation_loss,
        "generation_batch_size": settings.get("generation_batch_size"),
        "guidance_pool_size": settings.get("guidance_pool_size"),
        "loss_history": [by_step[key] for key in sorted(by_step)],
    }


def _memory_view(root: Path) -> dict[str, object]:
    payload = _read_json(
        root / "campaign" / "agent_adversarial_loop" / "agent_memory_compact.json",
        default={},
    )
    insights: list[str] = []
    champion = payload.get("current_champion") or {}
    if isinstance(champion, Mapping) and champion.get("round") is not None:
        error = _number(champion.get("generated_mean_relative_l2"))
        text = f"Best retained solver came from round {int(champion['round']) + 1}"
        if error is not None:
            text += f" with generated-case error {error:.4f}"
        insights.append(text + ".")
    for key in ("reliable_insights", "failure_modes", "overfit_warnings", "open_hypotheses"):
        values = payload.get(key)
        if isinstance(values, list):
            for value in values[:2]:
                if isinstance(value, str) and value.strip():
                    insights.append(value.strip())
                elif isinstance(value, Mapping) and value.get("text"):
                    insights.append(str(value["text"]).strip())
    return {
        "rounds_summarized": _integer(payload.get("rounds_summarized")) or 0,
        "insights": insights[:5],
    }


def _friendly_issue(
    root: Path, active_stage: Mapping[str, object] | None
) -> dict[str, object]:
    stage_id = str((active_stage or {}).get("id") or "")
    error_text = ""
    candidates = sorted((root / "logs").glob(f"{stage_id}_*.err"))
    if candidates:
        error_text = candidates[-1].read_text(encoding="utf-8", errors="replace")[-5000:]
    if "No module named" in error_text:
        description = "A required Python package was unavailable while preparing the dataset."
    elif "policy update command failed" in error_text:
        description = "The Generator guidance update failed after the solver evaluation."
    elif stage_id:
        description = (
            f"The campaign stopped during {_stage_activity(stage_id, {}).lower()}."
        )
    else:
        description = "The campaign stopped before a final result was produced."
    return {
        "title": "Campaign needs attention",
        "description": description,
        "phase": (active_stage or {}).get("phase_name"),
    }


def _current_round_activity(
    root: Path, rounds_completed: int, rounds_requested: int
) -> str | None:
    events = _read_jsonl(
        root / "campaign" / "agent_adversarial_loop" / "agent_events.jsonl"
    )
    for event in reversed(events):
        name = str(event.get("name") or "")
        if name in _EVENT_ACTIVITY:
            if name == "final_summary_written":
                return "Final solver result is ready"
            return _EVENT_ACTIVITY[name]
    rounds_root = root / "campaign" / "agent_adversarial_loop" / "rounds"
    if rounds_root.is_dir() and rounds_completed < rounds_requested:
        return f"Generating challenge cases for round {rounds_completed + 1}"
    return None


def _round_subprogress(root: Path, round_index: int) -> float:
    events = _read_jsonl(
        root / "campaign" / "agent_adversarial_loop" / "agent_events.jsonl"
    )
    scores = {
        "solver_workspace_prepared": 0.15,
        "hard_physical_verifier": 0.25,
        "solver_agent": 0.45,
        "generated_current_solver_eval": 0.60,
        "generated_cumulative_solver_eval": 0.70,
        "hidden_continuous_holdout_solver_eval": 0.80,
        "reward_tables_written": 0.86,
        "diffusion_policy_update": 0.94,
        "round_summary_written": 1.0,
    }
    best = 0.05
    for event in events:
        attrs = event.get("attributes") or {}
        if not isinstance(attrs, Mapping) or _integer(attrs.get("round")) != round_index:
            continue
        best = max(best, scores.get(str(event.get("name") or ""), 0.05))
    return best


def _projection_progress(root: Path, split: str) -> tuple[int, int]:
    path = root / "artifacts" / "coefficients" / split / "projection_progress.jsonl"
    rows = _read_jsonl(path)
    if not rows:
        return 0, 0
    latest = rows[-1]
    return (
        _integer(latest.get("completed_cases")) or 0,
        _integer(latest.get("total_cases")) or 0,
    )


def _generator_progress(root: Path) -> tuple[int, int, float | None]:
    report = _read_json(
        root / "artifacts" / "base_generator" / "train_report.json", default={}
    )
    config = _read_json(
        root / "artifacts" / "base_generator" / "train_config.json", default={}
    )
    events = _read_jsonl(root / "artifacts" / "base_generator" / "metrics.jsonl")
    latest_step = max(
        (_integer(row.get("step")) or 0 for row in events),
        default=_integer(report.get("final_step")) or 0,
    )
    validation_loss = _number(report.get("final_validation_loss"))
    for row in reversed(events):
        if row.get("event") == "validation":
            validation_loss = _number(row.get("loss"))
            break
    return latest_step, _integer(config.get("steps")) or 0, validation_loss


def _metric_summary(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    source = value.get("summary") if isinstance(value.get("summary"), Mapping) else value
    keys = (
        "mean_relative_l2",
        "median_relative_l2",
        "q10_relative_l2",
        "q90_relative_l2",
        "min_relative_l2",
        "max_relative_l2",
        "mean_velocity_rmse",
        "mean_truth_velocity_rms",
    )
    result = {key: _number(source.get(key)) for key in keys}
    result["n"] = _integer(source.get("n"))
    result["ok"] = _integer(source.get("ok"))
    return {key: value for key, value in result.items() if value is not None}


def _errors(metrics: Mapping[str, object]) -> list[dict[str, object]]:
    rows = metrics.get("results") if isinstance(metrics, Mapping) else None
    if not isinstance(rows, list):
        return []
    values = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        error = _number(row.get("relative_l2"))
        if error is not None:
            values.append(
                {
                    "case": str(row.get("case_id") or f"Case {index + 1}"),
                    "error": error,
                }
            )
    return values


def _argument(run: Mapping[str, object], name: str) -> str | None:
    for stage in run.get("stages", []) if isinstance(run.get("stages"), list) else []:
        if not isinstance(stage, Mapping):
            continue
        value = _argument_in_commands(stage.get("commands"), name)
        if value is not None:
            return value
    return None


def _argument_for_stage(
    run: Mapping[str, object], stage_id: str, name: str
) -> str | None:
    for stage in run.get("stages", []) if isinstance(run.get("stages"), list) else []:
        if isinstance(stage, Mapping) and stage.get("id") == stage_id:
            return _argument_in_commands(stage.get("commands"), name)
    return None


def _argument_in_commands(commands: object, name: str) -> str | None:
    if not isinstance(commands, list):
        return None
    for command in commands:
        if isinstance(command, Mapping):
            argv = command.get("argv")
        else:
            argv = command
        if not isinstance(argv, list):
            continue
        values = [str(value) for value in argv]
        found = _argument_in_argv(values, name)
        if found is not None:
            return found
        nested = _argument_in_argv(values, "--generate-command")
        if nested:
            try:
                found = _argument_in_argv(shlex.split(nested), name)
            except ValueError:
                found = None
            if found is not None:
                return found
    return None


def _argument_in_argv(argv: list[str], name: str) -> str | None:
    try:
        index = argv.index(name)
    except ValueError:
        return None
    return argv[index + 1] if index + 1 < len(argv) else ""


def _stage_phase(stage_id: str) -> str:
    lowered = stage_id.lower()
    if lowered == "prepare_campaign":
        return "data"
    if lowered == "final_evaluation":
        return "result"
    if "publish" in lowered or lowered in {"result", "final"}:
        return "result"
    if any(token in lowered for token in ("round", "agent", "campaign", "adversarial")):
        return "campaign"
    if "generator" in lowered or "hidden" in lowered:
        return "generator"
    return "data"


def _stage_round_index(stage_id: str) -> int | None:
    prefix = "adversarial_round_"
    if not stage_id.startswith(prefix):
        return None
    try:
        return int(stage_id.removeprefix(prefix))
    except ValueError:
        return None


def _stage_activity(stage_id: str, stage: Mapping[str, object]) -> str:
    round_index = _stage_round_index(stage_id)
    if round_index is not None:
        return (
            "Agent and Generator are improving the solver "
            f"in round {round_index + 1}"
        )
    return _STAGE_ACTIVITY.get(stage_id, _friendly_stage_title(stage))


def _friendly_stage_title(stage: Mapping[str, object]) -> str:
    title = str(stage.get("title") or "Working on the campaign")
    return title.replace("continuous coefficients", "continuous representation")


def _problem_label(problem_id: str, raw_result: object) -> str:
    if isinstance(raw_result, Mapping) and raw_result.get("problem"):
        return str(raw_result["problem"]).replace("--", "–")
    return problem_id.replace("_", " ").upper() if len(problem_id) <= 5 else problem_id.title()


def _default_task_kind(problem_id: str) -> str:
    """Use the registered PDE default for older run files without task metadata."""

    try:
        from pde_agent_verifier_solver.plugins.registry import get_plugin

        return get_plugin(problem_id).default_task
    except ValueError:
        return "solver"


def _task_label(kind: str) -> str:
    return {
        "solver": "Solver",
        "reduced_dynamics": "Reduced-order model",
        "observable_model": "Observable model",
        "statistics": "Statistics model",
    }.get(kind, kind.replace("_", " ").title())


def _updated_at(
    root: Path, stages: list[Mapping[str, object]], created_at: str
) -> str:
    candidates = [
        str(row.get("updated_at") or "")
        for row in stages
        if row.get("updated_at")
    ]
    events = _read_jsonl(
        root / "campaign" / "agent_adversarial_loop" / "agent_events.jsonl"
    )
    candidates.extend(str(row.get("ts") or "") for row in events if row.get("ts"))
    return max(candidates, default=created_at)


def _elapsed_seconds(start: str, end: str | None) -> float | None:
    start_dt = _datetime(start)
    end_dt = _datetime(end) if end else datetime.now(timezone.utc)
    if start_dt is None or end_dt is None:
        return None
    return max((end_dt - start_dt).total_seconds(), 0.0)


def _datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _run_directories(root: Path) -> list[Path]:
    if (root / "run.json").is_file():
        return [root]
    if not root.is_dir():
        return []
    return sorted(
        child
        for child in root.iterdir()
        if child.is_dir() and (child / "run.json").is_file()
    )
