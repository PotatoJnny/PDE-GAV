#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_verifier.codex_exec.client import CodexCliClient
from agent_verifier.neural_generator.agent_loop_trace import (
    AgentLoopRecorder,
    ingest_agent_memory_inbox,
    record_eval_feedback,
    write_round_memory_cards,
    write_workspace_memory_files,
)
from agent_verifier.neural_generator.triplane_agent_solver import (
    WorkspaceEvaluatorBroker,
    evaluate_solver_file,
    reward_spec_from_autoencoder_checkpoint,
    summarize_metric_rows,
    write_private_manifest,
    write_workspace_evaluator_tool,
)
from agent_verifier.neural_generator.triplane_dataset import divergence, iter_case_dirs, load_case
from agent_verifier.neural_generator.triplane_verifier import (
    CONTINUOUS_MMS_TRUTH_CONTRACT,
    require_continuous_mms_case_contract,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a 10-round agent-written-solver adversarial loop on triplane PDE cases.")
    parser.add_argument("--generated-cases-root", default=None, help="Existing generated case pool. Used in round slices unless --generate-command is set.")
    parser.add_argument("--generate-command", default=None, help="Optional shell command that writes generated cases for each round.")
    parser.add_argument("--split-manifest", default=None, help="Optional D0 split manifest; adversarial_guard is used for difficulty comparison.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--round-index-offset", type=int, default=0, help="Absolute round index for the first round in this invocation.")
    parser.add_argument(
        "--campaign-rounds-total",
        type=int,
        default=0,
        help=(
            "Total rounds in the campaign when this invocation runs only a "
            "slice. This lets one-round Slurm jobs produce one cumulative "
            "campaign summary. Zero keeps the legacy single-job behavior."
        ),
    )
    parser.add_argument("--existing-case-offset", type=int, default=0, help="Skip this many sorted cases before slicing an existing case pool.")
    parser.add_argument("--cases-per-round", type=int, default=8192)
    parser.add_argument(
        "--generated-cases-per-round",
        type=int,
        default=0,
        help="Generated plus perturbed cases within the total round budget.",
    )
    parser.add_argument("--original-split-manifest", default="")
    parser.add_argument("--original-split", default="validation")
    parser.add_argument("--original-cases-per-round", type=int, default=0)
    parser.add_argument("--original-selection-seed", type=int, default=20260814)
    parser.add_argument("--guard-cases", type=int, default=5000)
    parser.add_argument(
        "--hidden-holdout-cases-root",
        default="",
        help=(
            "Private continuous-MMS holdout evaluated only after each agent has "
            "submitted; it is never copied into the solver workspace."
        ),
    )
    parser.add_argument("--hidden-holdout-cases", type=int, default=0)
    parser.add_argument(
        "--hidden-generate-command",
        default="",
        help="Optional command that creates a fresh private hidden cohort each round.",
    )
    parser.add_argument("--hidden-generated-cases-per-round", type=int, default=0)
    parser.add_argument(
        "--hidden-accumulate",
        action="store_true",
        help="Evaluate every solver on all generated hidden cohorts so far.",
    )
    parser.add_argument("--include-historical-replay", action="store_true", help="Expose earlier generated rounds to the solver evaluator for replay feedback.")
    parser.add_argument("--cumulative-reward-table", action="store_true", help="Write reward_table.jsonl from the current solver on all replay+current generated cases.")
    parser.add_argument("--solver-backend", default="codex_mock", help="codex, codex_mock, or python_agent:/path/to/submission.py")
    parser.add_argument(
        "--agent-prompt-file",
        default="",
        help=(
            "Optional researcher-written scientific objective. It is placed "
            "before the standard solver interface and workspace instructions."
        ),
    )
    parser.add_argument("--initial-previous-submission", default="", help="Optional previous-round solver copied into previous_solver/submission.py.")
    parser.add_argument("--codex-timeout-seconds", type=int, default=1800)
    parser.add_argument("--codex-max-attempts", type=int, default=2)
    parser.add_argument(
        "--codex-persistent-search-seconds",
        type=int,
        default=0,
        help=(
            "When positive, keep invoking Codex after clean submissions until "
            "this cumulative Codex wall-time budget is consumed. Every valid "
            "candidate is evaluated on the agent-visible development cases and "
            "the best candidate is retained."
        ),
    )
    parser.add_argument(
        "--codex-persistent-search-max-attempts",
        type=int,
        default=0,
        help=(
            "Safety cap for persistent-search invocations; zero means no cap. "
            "The run is marked unsuccessful if the cap is reached before the "
            "requested Codex wall-time budget."
        ),
    )
    parser.add_argument("--codex-resume-finish", action="store_true", help="Resume an existing agent workspace and require prompt finalization instead of further broad experiments.")
    parser.add_argument("--codex-command-template", default=None)
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--codex-reasoning-effort", default="")
    parser.add_argument("--codex-allow-network", action="store_true")
    parser.add_argument("--solver-workspace-root", default=None)
    parser.add_argument(
        "--solver-cache-guidance",
        action="store_true",
        help="Tell the Agent how to reuse fixed geometry and operator setup.",
    )
    parser.add_argument("--eval-nx", type=int, default=0)
    parser.add_argument("--eval-ny", type=int, default=0)
    parser.add_argument("--eval-nt", type=int, default=0)
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=1,
        help="Number of formal evaluator cases to run in parallel.",
    )
    parser.add_argument("--failure-reward", type=float, default=1.0e6)
    parser.add_argument(
        "--reward-metric",
        choices=("relative_l2", "train_scaled_velocity_rmse"),
        default="relative_l2",
    )
    parser.add_argument(
        "--policy-reward-objective",
        choices=("solver_error", "solver_learning_progress"),
        default="solver_error",
        help=(
            "Generator reward. solver_learning_progress uses the paired decrease "
            "in evaluator loss from the inherited solver to the submitted solver."
        ),
    )
    parser.add_argument(
        "--formal-evaluation-scope",
        choices=("auto", "current", "cumulative"),
        default="auto",
        help=(
            "Cases used for formal old/new solver evaluation. Agent-visible replay "
            "is controlled independently by --include-historical-replay."
        ),
    )
    parser.add_argument(
        "--reward-normalizer-checkpoint",
        default="",
        help="AE checkpoint containing the frozen train-split field normalizer required by train-scaled reward.",
    )
    parser.add_argument("--policy-update-command", default=None, help="Optional shell command run after each round with reward-table env vars.")
    parser.add_argument("--skip-hard-verifier", action="store_true", help="Legacy bypass for non-formal, externally audited cases.")
    parser.add_argument(
        "--require-continuous-mms-cases",
        action="store_true",
        help="Require generated and replay cases to expose the continuous-MMS hidden-solution contract; numerical hard audit is not required.",
    )
    parser.add_argument("--hard-divergence-rel-max", type=float, default=5.0e-2)
    parser.add_argument("--hard-boundary-velocity-rel-max", type=float, default=1.0e-6)
    parser.add_argument("--hard-obstacle-velocity-rel-max", type=float, default=1.0e-6)
    parser.add_argument("--hard-forcing-obstacle-rel-max", type=float, default=1.0e-7)
    parser.add_argument("--keep-existing-workspaces", action="store_true")
    args = parser.parse_args(argv)
    researcher_prompt = _read_researcher_prompt(str(args.agent_prompt_file))

    if int(args.rounds) <= 0:
        raise SystemExit("--rounds must be positive")
    if int(args.campaign_rounds_total) < 0:
        raise SystemExit("--campaign-rounds-total cannot be negative")
    if (
        int(args.campaign_rounds_total) > 0
        and int(args.round_index_offset) + int(args.rounds)
        > int(args.campaign_rounds_total)
    ):
        raise SystemExit("round slice extends past --campaign-rounds-total")
    if int(args.cases_per_round) <= 0:
        raise SystemExit("--cases-per-round must be positive")
    if int(args.original_cases_per_round) < 0:
        raise SystemExit("--original-cases-per-round cannot be negative")
    if int(args.generated_cases_per_round) < 0:
        raise SystemExit("--generated-cases-per-round cannot be negative")
    if int(args.generated_cases_per_round) == 0:
        args.generated_cases_per_round = (
            int(args.cases_per_round) - int(args.original_cases_per_round)
        )
    if (
        int(args.original_cases_per_round)
        + int(args.generated_cases_per_round)
        != int(args.cases_per_round)
    ):
        raise SystemExit(
            "original and generated case counts must equal --cases-per-round"
        )
    if int(args.original_cases_per_round) > 0 and not str(args.original_split_manifest):
        raise SystemExit("original cases require --original-split-manifest")
    if int(args.hidden_generated_cases_per_round) < 0:
        raise SystemExit("--hidden-generated-cases-per-round cannot be negative")
    if int(args.hidden_generated_cases_per_round) > 0 and not str(args.hidden_generate_command):
        raise SystemExit("per-round hidden cases require --hidden-generate-command")
    if int(args.existing_case_offset) < 0:
        raise SystemExit("--existing-case-offset must be non-negative")
    if int(args.codex_timeout_seconds) <= 0:
        raise SystemExit("--codex-timeout-seconds must be positive")
    if int(args.codex_max_attempts) <= 0:
        raise SystemExit("--codex-max-attempts must be positive")
    if int(args.eval_workers) <= 0:
        raise SystemExit("--eval-workers must be positive")
    if int(args.codex_persistent_search_seconds) < 0:
        raise SystemExit("--codex-persistent-search-seconds cannot be negative")
    if int(args.codex_persistent_search_max_attempts) < 0:
        raise SystemExit("--codex-persistent-search-max-attempts cannot be negative")
    if bool(args.codex_resume_finish) and int(args.codex_persistent_search_seconds) > 0:
        raise SystemExit("--codex-resume-finish is incompatible with persistent search")
    if not args.generated_cases_root and not args.generate_command:
        raise SystemExit("provide --generated-cases-root or --generate-command")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reward_spec = _load_reward_spec(
        metric=str(args.reward_metric),
        normalizer_checkpoint=str(args.reward_normalizer_checkpoint),
    )
    (out / "reward_spec.json").write_text(json.dumps(reward_spec, indent=2, sort_keys=True) + "\n")
    started = time.time()
    recorder = AgentLoopRecorder(out, trace_id=f"triplane_agent_loop_{int(started)}")
    loop_span = recorder.event(
        kind="agent.loop",
        name="triplane_agent_adversarial_loop",
        attributes={
            "rounds": int(args.rounds),
            "round_index_offset": int(args.round_index_offset),
            "existing_case_offset": int(args.existing_case_offset),
            "cases_per_round": int(args.cases_per_round),
            "original_cases_per_round": int(args.original_cases_per_round),
            "generated_cases_per_round": int(args.generated_cases_per_round),
            "hidden_generated_cases_per_round": int(args.hidden_generated_cases_per_round),
            "solver_backend": str(args.solver_backend),
            "policy_update_enabled": bool(args.policy_update_command),
            "include_historical_replay": bool(args.include_historical_replay),
            "cumulative_reward_table": bool(args.cumulative_reward_table),
            "require_continuous_mms_cases": bool(args.require_continuous_mms_cases),
            "reward_metric": str(reward_spec["metric"]),
            "eval_workers": int(args.eval_workers),
        },
    )
    all_existing_cases = iter_case_dirs(args.generated_cases_root) if args.generated_cases_root else []
    required_existing_cases = int(args.existing_case_offset) + int(args.rounds) * int(args.generated_cases_per_round)
    if args.generated_cases_root and len(all_existing_cases) < required_existing_cases:
        raise SystemExit(
            f"generated case pool has {len(all_existing_cases)} cases, but {required_existing_cases} are required after offset"
        )
    original_pool = _load_split_dirs(
        str(args.original_split_manifest), str(args.original_split)
    )
    campaign_rounds = (
        int(args.campaign_rounds_total)
        if int(args.campaign_rounds_total) > 0
        else int(args.round_index_offset) + int(args.rounds)
    )
    required_original = campaign_rounds * int(args.original_cases_per_round)
    if required_original > len(original_pool):
        raise SystemExit(
            f"original split has {len(original_pool)} cases; {required_original} are required"
        )
    guard_dirs = _load_guard_dirs(args.split_manifest, int(args.guard_cases)) if args.split_manifest else []
    hidden_holdout_dirs = (
        iter_case_dirs(args.hidden_holdout_cases_root)
        if str(args.hidden_holdout_cases_root)
        else []
    )
    if int(args.hidden_holdout_cases) < 0:
        raise SystemExit("--hidden-holdout-cases cannot be negative")
    if int(args.hidden_holdout_cases) > 0:
        hidden_holdout_dirs = hidden_holdout_dirs[: int(args.hidden_holdout_cases)]
    if str(args.hidden_holdout_cases_root) and not hidden_holdout_dirs:
        raise SystemExit("hidden holdout root contains no cases")
    if args.require_continuous_mms_cases:
        _require_continuous_case_dirs(hidden_holdout_dirs)
    guard_public_root = None
    if guard_dirs:
        guard_public_root = out / "shared_guard_public_cases"
        _prepare_public_case_cache(
            guard_dirs,
            guard_public_root,
            metadata_path=out / "shared_guard_public_cases.json",
        )

    previous_submission: Path | None = Path(args.initial_previous_submission) if args.initial_previous_submission else None
    if previous_submission is not None and not previous_submission.exists():
        raise SystemExit(f"--initial-previous-submission does not exist: {previous_submission}")
    if args.policy_reward_objective == "solver_learning_progress" and previous_submission is None:
        raise SystemExit("solver_learning_progress requires --initial-previous-submission for the first round")
    round_summaries = _load_completed_round_summaries(
        out=out,
        before_round=int(args.round_index_offset),
    )
    for local_round_index in range(int(args.rounds)):
        round_previous_submission = previous_submission
        round_index = int(args.round_index_offset) + int(local_round_index)
        round_dir = out / "rounds" / f"round_{round_index:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        round_span = recorder.event(
            kind="agent.round",
            name=f"round_{round_index:03d}",
            parent_span_id=loop_span,
            attributes={
                "round": int(round_index),
                "local_round": int(local_round_index),
                "cases_per_round": int(args.cases_per_round),
            },
        )
        raw_generated_case_dirs = _round_generated_cases(
            round_index=round_index,
            local_round_index=local_round_index,
            round_dir=round_dir,
            args=args,
            all_existing_cases=all_existing_cases,
        )
        generated_budget = int(args.generated_cases_per_round)
        if len(raw_generated_case_dirs) < generated_budget:
            raise SystemExit(
                f"round {round_index} has {len(raw_generated_case_dirs)} generated "
                f"cases; need {generated_budget}"
            )
        _promote_legacy_continuous_mms_cli(args, round_dir=round_dir)
        hard_case_dirs, hard_summary = _hard_filter_cases(raw_generated_case_dirs, round_dir=round_dir, args=args)
        if len(hard_case_dirs) < generated_budget:
            raise SystemExit(
                f"round {round_index} has {len(hard_case_dirs)} hard-passed "
                f"generated cases; need {generated_budget}"
            )
        generated_only_case_dirs = hard_case_dirs[:generated_budget]
        original_case_dirs = _select_original_cases(
            original_pool,
            round_index=round_index,
            count=int(args.original_cases_per_round),
            seed=int(args.original_selection_seed),
        )
        generated_case_dirs = _dedupe_case_dirs(
            [*original_case_dirs, *generated_only_case_dirs]
        )
        if len(generated_case_dirs) != int(args.cases_per_round):
            raise SystemExit("round case mix does not equal the configured total")
        round_case_mix = {
            "original": len(original_case_dirs),
            "generated_and_perturbed": len(generated_only_case_dirs),
            "total": len(generated_case_dirs),
        }
        (round_dir / "round_case_mix.json").write_text(
            json.dumps(round_case_mix, indent=2, sort_keys=True) + "\n"
        )
        hidden_current_dirs = _round_hidden_cases(
            round_index=round_index,
            round_dir=round_dir,
            args=args,
        )
        hidden_historical_dirs = (
            _load_historical_hidden_case_dirs(out=out, current_round_index=round_index)
            if bool(args.hidden_accumulate)
            else []
        )
        hidden_cumulative_dirs = _dedupe_case_dirs(
            [*hidden_historical_dirs, *hidden_current_dirs]
        )
        historical_case_dirs = _load_historical_generated_case_dirs(out=out, current_round_index=round_index) if args.include_historical_replay else []
        solver_case_dirs = _dedupe_case_dirs([*historical_case_dirs, *generated_case_dirs])
        workspace, case_aliases = _prepare_solver_workspace(
            round_index=round_index,
            round_dir=round_dir,
            private_case_dirs=solver_case_dirs,
            previous_submission=previous_submission,
            solver_workspace_root=Path(args.solver_workspace_root) if args.solver_workspace_root else None,
            keep_existing=bool(args.keep_existing_workspaces),
            eval_density={"nx": int(args.eval_nx), "ny": int(args.eval_ny), "nt": int(args.eval_nt)},
            compact_memory=recorder.compact_memory(),
            researcher_prompt=researcher_prompt,
            solver_cache_guidance=bool(args.solver_cache_guidance),
        )
        alias_to_original = {alias: original for original, alias in case_aliases.items()}
        private_manifest = round_dir / "private_eval" / "generated_private_manifest.json"
        _write_private_manifest_with_aliases(
            private_manifest,
            public_cases_root=workspace / "cases",
            private_case_dirs=generated_case_dirs,
            case_aliases=case_aliases,
            truth_contract=(
                _truth_contract_for_cases(generated_case_dirs)
                if args.require_continuous_mms_cases else None
            ),
            reward_spec=reward_spec,
        )
        solver_feedback_manifest = private_manifest
        if solver_case_dirs != generated_case_dirs:
            solver_feedback_manifest = round_dir / "private_eval" / "solver_feedback_private_manifest.json"
            _write_private_manifest_with_aliases(
                solver_feedback_manifest,
                public_cases_root=workspace / "cases",
                private_case_dirs=solver_case_dirs,
                case_aliases=case_aliases,
                truth_contract=(
                    _truth_contract_for_cases(solver_case_dirs)
                    if args.require_continuous_mms_cases else None
                ),
                reward_spec=reward_spec,
            )
        formal_cumulative = _formal_evaluation_is_cumulative(args, solver_feedback_manifest != private_manifest)
        previous_generated_eval = None
        previous_cumulative_eval = None
        if round_previous_submission is not None and round_previous_submission.exists():
            if formal_cumulative:
                previous_cumulative_eval = _evaluate_previous_solver_on_cumulative_cases(
                    submission=round_previous_submission,
                    public_cases_root=workspace / "cases",
                    cumulative_manifest=solver_feedback_manifest,
                    current_manifest=private_manifest,
                    args=args,
                    label="previous_solver_on_cumulative_generated_replay",
                )
                _write_eval_artifacts(round_dir, "previous_solver_cumulative_generated_metrics", previous_cumulative_eval)
                previous_generated_eval = (
                    _subset_evaluation(
                        previous_cumulative_eval,
                        target_manifest=private_manifest,
                        label="previous_solver_on_generated",
                    )
                    if solver_feedback_manifest != private_manifest
                    else dict(previous_cumulative_eval, label="previous_solver_on_generated")
                )
            else:
                previous_generated_eval = _evaluate_or_missing(
                    submission=round_previous_submission,
                    public_cases_root=workspace / "cases",
                    private_manifest=private_manifest,
                    args=args,
                    label="previous_solver_on_generated",
                )
            _write_eval_artifacts(round_dir, "previous_solver_generated_metrics", previous_generated_eval)
        submission = _run_solver_agent(
            backend=str(args.solver_backend),
            round_index=round_index,
            workspace=workspace,
            prompt_path=workspace / "prompt.md",
            private_manifest=solver_feedback_manifest,
            round_dir=round_dir,
            args=args,
        )
        if submission is None or not submission.exists():
            raise SystemExit(f"round {round_index} solver agent did not produce submission.py")
        standalone_preflight = _standalone_submission_preflight(
            submission=submission,
            workspace=workspace,
            private_manifest=solver_feedback_manifest,
            round_dir=round_dir,
            nx=int(getattr(args, "eval_nx", 0)),
            ny=int(getattr(args, "eval_ny", 0)),
            nt=int(getattr(args, "eval_nt", 0)),
            failure_reward=float(getattr(args, "failure_reward", 1.0e6)),
        )
        if not bool(standalone_preflight["passed"]):
            raise SystemExit(
                f"round {round_index} submission failed standalone preflight: "
                f"{standalone_preflight.get('failure')}"
            )
        artifact_submission = round_dir / "submission.py"
        shutil.copy2(submission, artifact_submission)
        previous_submission = artifact_submission

        cumulative_eval = None
        if formal_cumulative:
            cumulative_eval = _evaluate_or_missing(
                submission=artifact_submission,
                public_cases_root=workspace / "cases",
                private_manifest=solver_feedback_manifest,
                args=args,
                label="cumulative_generated_replay",
            )
            _write_eval_artifacts(round_dir, "cumulative_generated_solver_metrics", cumulative_eval)
            generated_eval = (
                _subset_evaluation(cumulative_eval, target_manifest=private_manifest, label="generated")
                if solver_feedback_manifest != private_manifest
                else dict(cumulative_eval, label="generated")
            )
        else:
            generated_eval = _evaluate_or_missing(
                submission=artifact_submission,
                public_cases_root=workspace / "cases",
                private_manifest=private_manifest,
                args=args,
                label="generated",
            )
        _write_eval_artifacts(round_dir, "generated_solver_metrics", generated_eval)
        guard_eval = None
        if guard_dirs:
            assert guard_public_root is not None
            guard_manifest = round_dir / "private_eval" / "guard_private_manifest.json"
            write_private_manifest(
                guard_manifest,
                public_cases_root=guard_public_root,
                private_case_dirs=guard_dirs,
                reward_spec=reward_spec,
            )
            guard_eval = _evaluate_or_missing(
                submission=artifact_submission,
                public_cases_root=guard_public_root,
                private_manifest=guard_manifest,
                args=args,
                label="guard",
            )
            _write_eval_artifacts(round_dir, "guard_solver_metrics", guard_eval)

        hidden_holdout_eval = None
        hidden_generated_current_eval = None
        hidden_generated_cumulative_eval = None
        hidden_all_eval = None
        hidden_all_dirs = _dedupe_case_dirs(
            [*hidden_holdout_dirs, *hidden_cumulative_dirs]
        )
        if hidden_all_dirs:
            hidden_public_root = out / "shared_hidden_public_cases"
            _prepare_public_case_cache(
                hidden_all_dirs,
                hidden_public_root,
                metadata_path=out / "shared_hidden_public_cases.json",
            )
            hidden_all_manifest = round_dir / "private_eval" / "hidden_all_private_manifest.json"
            write_private_manifest(
                hidden_all_manifest,
                public_cases_root=hidden_public_root,
                private_case_dirs=hidden_all_dirs,
                reward_spec=reward_spec,
            )
            hidden_all_eval = _evaluate_or_missing(
                submission=artifact_submission,
                public_cases_root=hidden_public_root,
                private_manifest=hidden_all_manifest,
                args=args,
                label="hidden_all_cumulative",
            )
            _write_eval_artifacts(round_dir, "hidden_all_solver_metrics", hidden_all_eval)
            if hidden_holdout_dirs:
                hidden_holdout_manifest = round_dir / "private_eval" / "hidden_holdout_private_manifest.json"
                write_private_manifest(
                    hidden_holdout_manifest,
                    public_cases_root=hidden_public_root,
                    private_case_dirs=hidden_holdout_dirs,
                    reward_spec=reward_spec,
                )
                hidden_holdout_eval = _subset_evaluation(
                    hidden_all_eval,
                    target_manifest=hidden_holdout_manifest,
                    label="hidden_fixed_reference",
                )
            if hidden_cumulative_dirs:
                hidden_cumulative_manifest = round_dir / "private_eval" / "hidden_generated_cumulative_private_manifest.json"
                write_private_manifest(
                    hidden_cumulative_manifest,
                    public_cases_root=hidden_public_root,
                    private_case_dirs=hidden_cumulative_dirs,
                    reward_spec=reward_spec,
                )
                hidden_generated_cumulative_eval = _subset_evaluation(
                    hidden_all_eval,
                    target_manifest=hidden_cumulative_manifest,
                    label="hidden_generated_cumulative",
                )
            if hidden_current_dirs:
                hidden_current_manifest = round_dir / "private_eval" / "hidden_generated_current_private_manifest.json"
                write_private_manifest(
                    hidden_current_manifest,
                    public_cases_root=hidden_public_root,
                    private_case_dirs=hidden_current_dirs,
                    reward_spec=reward_spec,
                )
                hidden_generated_current_eval = _subset_evaluation(
                    hidden_all_eval,
                    target_manifest=hidden_current_manifest,
                    label="hidden_generated_current",
                )
        if hidden_holdout_eval is not None:
            _write_eval_artifacts(
                round_dir, "hidden_holdout_solver_metrics", hidden_holdout_eval
            )
        if hidden_generated_cumulative_eval is not None:
            _write_eval_artifacts(round_dir, "hidden_generated_cumulative_solver_metrics", hidden_generated_cumulative_eval)
        if hidden_generated_current_eval is not None:
            _write_eval_artifacts(round_dir, "hidden_generated_current_solver_metrics", hidden_generated_current_eval)

        current_reward_rows = _write_policy_reward_table(
            round_dir / "current_reward_table.jsonl",
            current_eval=generated_eval,
            previous_eval=previous_generated_eval,
            objective=str(args.policy_reward_objective),
            case_id_aliases=alias_to_original,
        )
        reward_eval = cumulative_eval if args.cumulative_reward_table and cumulative_eval is not None else generated_eval
        previous_reward_eval = (
            previous_cumulative_eval
            if args.cumulative_reward_table and previous_cumulative_eval is not None
            else previous_generated_eval
        )
        reward_table = _write_policy_reward_table(
            round_dir / "reward_table.jsonl",
            current_eval=reward_eval,
            previous_eval=previous_reward_eval,
            objective=str(args.policy_reward_objective),
            case_id_aliases=alias_to_original,
        )
        if cumulative_eval is not None:
            _write_policy_reward_table(
                round_dir / "cumulative_reward_table.jsonl",
                current_eval=cumulative_eval,
                previous_eval=previous_cumulative_eval,
                objective=str(args.policy_reward_objective),
                case_id_aliases=alias_to_original,
            )
        policy_reward_summary = _summarize_policy_reward_rows(reward_table)
        policy_reward_summary.update(
            {
                "objective": str(args.policy_reward_objective),
                "formal_evaluation_scope": "cumulative" if formal_cumulative else "current",
                "current_round_rows": len(current_reward_rows),
                "policy_rows": len(reward_table),
            }
        )
        (round_dir / "policy_reward_summary.json").write_text(
            json.dumps(policy_reward_summary, indent=2, sort_keys=True) + "\n"
        )
        policy_update = _run_policy_update_command(
            command=args.policy_update_command,
            round_index=round_index,
            round_dir=round_dir,
            reward_table=round_dir / "reward_table.jsonl",
            generated_metrics=round_dir / "generated_solver_metrics.json",
            submission=artifact_submission,
        )
        summary = _round_summary(
            round_index=round_index,
            generated_case_count=len(generated_case_dirs),
            round_case_mix=round_case_mix,
            workspace=workspace,
            submission=artifact_submission,
            previous_submission=round_previous_submission,
            generated_eval=generated_eval,
            previous_generated_eval=previous_generated_eval,
            cumulative_eval=cumulative_eval,
            previous_cumulative_eval=previous_cumulative_eval,
            guard_eval=guard_eval,
            hidden_holdout_eval=hidden_holdout_eval,
            hidden_generated_current_eval=hidden_generated_current_eval,
            hidden_generated_cumulative_eval=hidden_generated_cumulative_eval,
            hidden_all_eval=hidden_all_eval,
            hard_summary=hard_summary,
            historical_replay_case_count=len(historical_case_dirs),
            solver_feedback_case_count=len(solver_case_dirs),
            cumulative_reward_table=bool(args.cumulative_reward_table),
            policy_update=policy_update,
        )
        summary["policy_reward_objective"] = str(args.policy_reward_objective)
        summary["formal_evaluation_scope"] = "cumulative" if formal_cumulative else "current"
        summary["policy_reward_summary"] = policy_reward_summary
        if round_summaries and previous_generated_eval is not None:
            previous_policy_mean = (previous_generated_eval.get("summary") or {}).get("mean_relative_l2")
            previous_round_mean = (round_summaries[-1].get("generated_summary") or {}).get("mean_relative_l2")
            summary["generator_difficulty_change_under_previous_solver"] = _relative_change(previous_policy_mean, previous_round_mean)
        (round_dir / "round_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (round_dir / "round_summary.md").write_text(_round_markdown(summary))
        round_summaries.append(summary)
        _append_jsonl(out / "round_trace.jsonl", [summary])
        _record_round_observability(
            recorder=recorder,
            round_index=round_index,
            round_dir=round_dir,
            workspace=workspace,
            submission=artifact_submission,
            generated_eval=generated_eval,
            cumulative_eval=cumulative_eval,
            guard_eval=guard_eval,
            hidden_holdout_eval=hidden_holdout_eval,
            hard_summary=hard_summary,
            policy_update=policy_update,
            summary=summary,
            parent_span_id=round_span,
        )

    final = _final_summary(args=args, out=out, round_summaries=round_summaries, started=started)
    (out / "final_summary.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    (out / "final_summary.md").write_text(_final_markdown(final))
    final_span = recorder.event(
        kind="artifact.write",
        name="final_summary_written",
        parent_span_id=loop_span,
        output_artifacts=["final_summary"],
    )
    recorder.artifact(artifact_id="final_summary", kind="summary", path=out / "final_summary.json", created_by_span=final_span)
    print(json.dumps(final, indent=2, sort_keys=True))


def _record_round_observability(
    *,
    recorder: AgentLoopRecorder,
    round_index: int,
    round_dir: Path,
    workspace: Path,
    submission: Path | None,
    generated_eval: dict[str, Any],
    cumulative_eval: dict[str, Any] | None,
    guard_eval: dict[str, Any] | None,
    hidden_holdout_eval: dict[str, Any] | None,
    hard_summary: dict[str, Any],
    policy_update: dict[str, Any],
    summary: dict[str, Any],
    parent_span_id: str,
) -> None:
    prefix = f"round{round_index:03d}"
    workspace_span = recorder.event(
        kind="artifact.write",
        name="solver_workspace_prepared",
        parent_span_id=parent_span_id,
        attributes={"round": int(round_index), "workspace": str(workspace)},
        output_artifacts=[f"{prefix}_prompt", f"{prefix}_case_manifest", f"{prefix}_workspace_memory"],
    )
    for artifact_id, kind, path in (
        (f"{prefix}_prompt", "prompt", workspace / "prompt.md"),
        (f"{prefix}_case_manifest", "case_manifest", workspace / "case_manifest.json"),
        (f"{prefix}_workspace_memory", "compact_memory", workspace / "agent_memory_compact.json"),
    ):
        if path.exists():
            recorder.artifact(artifact_id=artifact_id, kind=kind, path=path, created_by_span=workspace_span, metadata={"round": int(round_index)})

    hard_span = recorder.event(
        kind="eval.run",
        name="hard_physical_verifier",
        parent_span_id=parent_span_id,
        attributes={"round": int(round_index), "summary": hard_summary},
        output_artifacts=[f"{prefix}_hard_verification"],
    )
    hard_path = round_dir / "hard_verification.json"
    if hard_path.exists():
        recorder.artifact(
            artifact_id=f"{prefix}_hard_verification",
            kind="hard_verification",
            path=hard_path,
            created_by_span=hard_span,
            metadata={"round": int(round_index)},
        )

    agent_span = recorder.event(
        kind="agent.run",
        name="solver_agent",
        status="ok" if submission is not None and submission.exists() else "missing_submission",
        parent_span_id=parent_span_id,
        attributes={"round": int(round_index), "workspace": str(workspace)},
        input_artifacts=[f"{prefix}_prompt", f"{prefix}_workspace_memory"],
        output_artifacts=[f"{prefix}_submission"] if submission is not None and submission.exists() else [],
    )
    ingest_agent_memory_inbox(recorder, workspace / "agent_memory_inbox.jsonl", round_index=round_index, evidence_span=agent_span)
    if submission is not None and submission.exists():
        recorder.artifact(
            artifact_id=f"{prefix}_submission",
            kind="solver_submission",
            path=submission,
            created_by_span=agent_span,
            metadata={"round": int(round_index)},
        )

    evaluator_audit = round_dir / "private_eval" / "evaluator_broker_audit.jsonl"
    evaluator_requests = round_dir / "private_eval" / "evaluator_requests"
    if evaluator_audit.exists():
        evaluator_history_span = recorder.event(
            kind="eval.history",
            name="agent_evaluator_history",
            parent_span_id=parent_span_id,
            attributes={"round": int(round_index)},
            input_artifacts=[f"{prefix}_submission"],
            output_artifacts=[
                f"{prefix}_evaluator_audit",
                f"{prefix}_evaluator_requests",
            ],
        )
        recorder.artifact(
            artifact_id=f"{prefix}_evaluator_audit",
            kind="evaluator_request_index",
            path=evaluator_audit,
            created_by_span=evaluator_history_span,
            metadata={"round": int(round_index)},
        )
        if evaluator_requests.exists():
            recorder.artifact(
                artifact_id=f"{prefix}_evaluator_requests",
                kind="evaluator_results",
                path=evaluator_requests,
                created_by_span=evaluator_history_span,
                metadata={"round": int(round_index)},
            )

    _record_eval_observability(
        recorder=recorder,
        round_index=round_index,
        prefix=prefix,
        split="generated_current",
        name="generated_current_solver_eval",
        payload=generated_eval,
        path=round_dir / "generated_solver_metrics.json",
        parent_span_id=parent_span_id,
        submission=submission,
    )
    if cumulative_eval is not None:
        _record_eval_observability(
            recorder=recorder,
            round_index=round_index,
            prefix=prefix,
            split="generated_cumulative",
            name="generated_cumulative_solver_eval",
            payload=cumulative_eval,
            path=round_dir / "cumulative_generated_solver_metrics.json",
            parent_span_id=parent_span_id,
            submission=submission,
        )
    if guard_eval is not None:
        _record_eval_observability(
            recorder=recorder,
            round_index=round_index,
            prefix=prefix,
            split="d0_guard",
            name="d0_guard_solver_eval",
            payload=guard_eval,
            path=round_dir / "guard_solver_metrics.json",
            parent_span_id=parent_span_id,
            submission=submission,
        )
    if hidden_holdout_eval is not None:
        _record_eval_observability(
            recorder=recorder,
            round_index=round_index,
            prefix=prefix,
            split="hidden_continuous_holdout",
            name="hidden_continuous_holdout_solver_eval",
            payload=hidden_holdout_eval,
            path=round_dir / "hidden_holdout_solver_metrics.json",
            parent_span_id=parent_span_id,
            submission=submission,
        )

    reward_outputs: list[str] = []
    for stem in ("current_reward_table", "reward_table", "cumulative_reward_table"):
        if (round_dir / f"{stem}.jsonl").exists():
            reward_outputs.append(f"{prefix}_{stem}")
    reward_span = recorder.event(
        kind="artifact.write",
        name="reward_tables_written",
        parent_span_id=parent_span_id,
        attributes={"round": int(round_index)},
        output_artifacts=reward_outputs,
    )
    for stem in ("current_reward_table", "reward_table", "cumulative_reward_table"):
        path = round_dir / f"{stem}.jsonl"
        if path.exists():
            recorder.artifact(
                artifact_id=f"{prefix}_{stem}",
                kind="reward_table",
                path=path,
                created_by_span=reward_span,
                metadata={"round": int(round_index), "table": stem},
            )

    policy_span = recorder.event(
        kind="policy.update",
        name="diffusion_policy_update",
        status="ok" if bool(policy_update.get("enabled")) else "disabled",
        parent_span_id=parent_span_id,
        attributes={"round": int(round_index), **policy_update},
        input_artifacts=[f"{prefix}_reward_table"],
        output_artifacts=[f"{prefix}_policy_update"] if (round_dir / "policy_update.json").exists() else [],
    )
    policy_path = round_dir / "policy_update.json"
    if policy_path.exists():
        recorder.artifact(
            artifact_id=f"{prefix}_policy_update",
            kind="policy_update",
            path=policy_path,
            created_by_span=policy_span,
            metadata={"round": int(round_index)},
        )

    summary_span = recorder.event(
        kind="artifact.write",
        name="round_summary_written",
        parent_span_id=parent_span_id,
        attributes={"round": int(round_index)},
        output_artifacts=[f"{prefix}_summary"],
    )
    recorder.artifact(
        artifact_id=f"{prefix}_summary",
        kind="summary",
        path=round_dir / "round_summary.json",
        created_by_span=summary_span,
        metadata={"round": int(round_index)},
    )
    write_round_memory_cards(recorder, round_index=round_index, summary=summary, evidence_span=summary_span)
    compact_path = recorder.write_compact_memory()
    memory_span = recorder.event(
        kind="memory.write",
        name="compact_memory_updated",
        parent_span_id=parent_span_id,
        attributes={"round": int(round_index)},
        output_artifacts=["agent_memory_compact"],
    )
    recorder.artifact(artifact_id="agent_memory_compact", kind="compact_memory", path=compact_path, created_by_span=memory_span)


def _record_eval_observability(
    *,
    recorder: AgentLoopRecorder,
    round_index: int,
    prefix: str,
    split: str,
    name: str,
    payload: dict[str, Any],
    path: Path,
    parent_span_id: str,
    submission: Path | None,
) -> None:
    artifact_id = f"{prefix}_{split}_solver_metrics"
    submission_artifact = f"{prefix}_submission" if submission is not None and submission.exists() else None
    span = recorder.event(
        kind="eval.run",
        name=name,
        parent_span_id=parent_span_id,
        attributes={"round": int(round_index), "split": split, "summary": payload.get("summary", {}) if isinstance(payload, dict) else {}},
        input_artifacts=[submission_artifact] if submission_artifact else [],
        output_artifacts=[artifact_id],
    )
    if path.exists():
        recorder.artifact(
            artifact_id=artifact_id,
            kind="eval_metrics",
            path=path,
            created_by_span=span,
            metadata={"round": int(round_index), "split": split},
        )
    record_eval_feedback(
        recorder,
        round_index=round_index,
        eval_payload=payload,
        split=split,
        target_span_id=span,
        target_artifact_id=submission_artifact,
    )


def _round_generated_cases(*, round_index: int, local_round_index: int, round_dir: Path, args, all_existing_cases: list[Path]) -> list[Path]:
    if not args.generate_command:
        start = int(args.existing_case_offset) + int(local_round_index) * int(args.generated_cases_per_round)
        stop = start + int(args.generated_cases_per_round)
        return all_existing_cases[start:stop]
    generated_root = round_dir / "generated_cases"
    generated_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "AVF_EXPLICIT_CASE_MIX": "1",
            "AVF_LOOP_ROOT": str(Path(args.out).resolve()),
            "AVF_CASE_ID_PREFIX": "ns_public",
            "AVF_ROUND_INDEX": str(round_index),
            "AVF_ROUND_DIR": str(round_dir),
            "AVF_ROUND_GENERATED_CASES": str(generated_root),
            "AVF_CASES_PER_ROUND": str(args.generated_cases_per_round),
        }
    )
    logs = round_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "generate.stdout.log").open("w") as stdout, (logs / "generate.stderr.log").open("w") as stderr:
        proc = subprocess.run(str(args.generate_command), shell=True, cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr, text=True)
    (round_dir / "generate_invocation.json").write_text(
        json.dumps({"command": str(args.generate_command), "returncode": proc.returncode, "generated_root": str(generated_root)}, indent=2, sort_keys=True) + "\n"
    )
    if proc.returncode != 0:
        raise SystemExit(f"round {round_index} generate command failed with return code {proc.returncode}")
    cases = iter_case_dirs(generated_root)
    if not cases:
        nested = sorted(round_dir.rglob("generated_cases"))
        for root in nested:
            cases = iter_case_dirs(root)
            if cases:
                break
    return cases


def _round_hidden_cases(
    *, round_index: int, round_dir: Path, args
) -> list[Path]:
    count = int(args.hidden_generated_cases_per_round)
    if count == 0:
        return []
    hidden_round_dir = round_dir / "hidden_generation"
    hidden_root = hidden_round_dir / "generated_cases"
    hidden_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "AVF_EXPLICIT_CASE_MIX": "1",
            "AVF_LOOP_ROOT": str(Path(args.out).resolve()),
            "AVF_CASE_ID_PREFIX": "ns_hidden",
            "AVF_ROUND_INDEX": str(round_index),
            "AVF_ROUND_DIR": str(hidden_round_dir),
            "AVF_ROUND_GENERATED_CASES": str(hidden_root),
            "AVF_CASES_PER_ROUND": str(count),
        }
    )
    logs = round_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "generate_hidden.stdout.log").open("w") as stdout, (
        logs / "generate_hidden.stderr.log"
    ).open("w") as stderr:
        process = subprocess.run(
            str(args.hidden_generate_command),
            shell=True,
            cwd=str(ROOT),
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
    (round_dir / "hidden_generate_invocation.json").write_text(
        json.dumps(
            {
                "command": str(args.hidden_generate_command),
                "returncode": process.returncode,
                "generated_root": str(hidden_root),
                "cases": count,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if process.returncode != 0:
        raise SystemExit(
            f"round {round_index} hidden generation failed with return code "
            f"{process.returncode}"
        )
    cases = iter_case_dirs(hidden_root)
    if len(cases) != count:
        raise SystemExit(
            f"round {round_index} generated {len(cases)} hidden cases; expected {count}"
        )
    _require_continuous_case_dirs(cases)
    return cases


def _load_historical_hidden_case_dirs(
    *, out: Path, current_round_index: int
) -> list[Path]:
    cases: list[Path] = []
    for round_index in range(max(0, int(current_round_index))):
        root = (
            out
            / "rounds"
            / f"round_{round_index:03d}"
            / "hidden_generation"
            / "generated_cases"
        )
        if not root.is_dir():
            raise SystemExit(f"missing prior generated hidden cohort: {root}")
        cases.extend(iter_case_dirs(root))
    return cases


def _load_split_dirs(manifest_value: str, split: str) -> list[Path]:
    if not str(manifest_value):
        return []
    manifest_path = Path(manifest_value).resolve()
    payload = json.loads(manifest_path.read_text())
    rows = payload.get("splits", {}).get(str(split), [])
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"dataset split {split!r} is missing or empty")
    paths = []
    for row in rows:
        case_id = str(row.get("case_id", ""))
        raw = Path(str(row.get("case_dir", "")))
        candidates = (
            raw,
            ROOT / raw,
            manifest_path.parent / raw,
            manifest_path.parent / "cases" / case_id,
        )
        resolved = next((path.resolve() for path in candidates if path.is_dir()), None)
        if resolved is None:
            raise SystemExit(f"cannot resolve original dataset case {case_id!r}")
        paths.append(resolved)
    return paths


def _select_original_cases(
    pool: list[Path], *, round_index: int, count: int, seed: int
) -> list[Path]:
    if count == 0:
        return []
    order = np.random.default_rng(int(seed)).permutation(len(pool))
    start = int(round_index) * int(count)
    stop = start + int(count)
    if stop > len(order):
        raise SystemExit("original dataset pool is too small for non-repeating rounds")
    return [pool[int(index)] for index in order[start:stop]]


def _truth_contract_for_cases(case_dirs: list[Path]) -> str | None:
    if case_dirs and all(
        (case_dir / "hidden_continuous_case.pt").is_file()
        for case_dir in case_dirs
    ):
        return CONTINUOUS_MMS_TRUTH_CONTRACT
    return None


def _read_researcher_prompt(value: str) -> str:
    """Read the optional prompt addition once before any round starts."""

    if not value.strip():
        return ""
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"--agent-prompt-file does not exist: {path}")
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise SystemExit("--agent-prompt-file cannot be empty")
    return prompt


def _prepare_solver_workspace(
    *,
    round_index: int,
    round_dir: Path,
    private_case_dirs: list[Path],
    previous_submission: Path | None,
    solver_workspace_root: Path | None,
    keep_existing: bool,
    eval_density: dict[str, int],
    compact_memory: dict[str, Any],
    researcher_prompt: str,
    solver_cache_guidance: bool = False,
) -> tuple[Path, dict[str, str]]:
    workspace = (solver_workspace_root / f"round_{round_index:03d}") if solver_workspace_root else round_dir / "solver_workspace"
    if workspace.exists() and not keep_existing:
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    cases_root = workspace / "cases"
    case_aliases = {path.name: f"case_{idx:06d}" for idx, path in enumerate(private_case_dirs)}
    _copy_public_cases(private_case_dirs, cases_root, case_aliases=case_aliases)
    previous_root = workspace / "previous_solver"
    previous_root.mkdir(parents=True, exist_ok=True)
    if previous_submission is not None and previous_submission.exists():
        shutil.copy2(previous_submission, previous_root / "submission.py")
    tools_root = workspace / "tools"
    tools_root.mkdir(parents=True, exist_ok=True)
    write_workspace_evaluator_tool(
        tools_root / "evaluate_solver.py",
        broker_only=True,
    )
    write_workspace_memory_files(workspace, compact_memory)
    (workspace / "README.md").write_text(_workspace_readme())
    standard_prompt = _solver_prompt(
        round_index=round_index,
        eval_density=eval_density,
        has_previous=bool(previous_submission),
        solver_cache_guidance=solver_cache_guidance,
    )
    prompt = standard_prompt
    if researcher_prompt:
        prompt = (
            "# Researcher-provided scientific objective\n\n"
            + researcher_prompt.rstrip()
            + "\n\n# Solver interface and workspace\n\n"
            + standard_prompt
        )
    (workspace / "prompt.md").write_text(prompt)
    (workspace / "case_manifest.json").write_text(
        json.dumps(
            {
                "cases": [
                    {"case_id": case_aliases[path.name], "case_dir": f"cases/{case_aliases[path.name]}"}
                    for path in private_case_dirs
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (round_dir / "solver_workspace_path.txt").write_text(str(workspace) + "\n")
    (round_dir / "private_eval" / "case_aliases.json").parent.mkdir(parents=True, exist_ok=True)
    (round_dir / "private_eval" / "case_aliases.json").write_text(json.dumps({alias: original for original, alias in case_aliases.items()}, indent=2, sort_keys=True) + "\n")
    return workspace, case_aliases


def _load_historical_generated_case_dirs(*, out: Path, current_round_index: int) -> list[Path]:
    rounds_root = out / "rounds"
    if not rounds_root.exists():
        return []
    case_dirs: list[Path] = []
    for round_dir in sorted(rounds_root.glob("round_*")):
        try:
            round_index = int(round_dir.name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if round_index >= int(current_round_index):
            continue
        manifest = round_dir / "private_eval" / "generated_private_manifest.json"
        if not manifest.exists():
            raise SystemExit(f"historical replay requested but missing manifest: {manifest}")
        payload = json.loads(manifest.read_text())
        for row in payload.get("cases", []):
            private_case_dir = Path(row["private_case_dir"])
            if not private_case_dir.exists():
                raise SystemExit(f"historical replay case no longer exists: {private_case_dir}")
            case_dirs.append(private_case_dir)
    return case_dirs


def _dedupe_case_dirs(case_dirs: list[Path]) -> list[Path]:
    seen: dict[str, Path] = {}
    out: list[Path] = []
    for path in case_dirs:
        case_id = path.name
        previous = seen.get(case_id)
        if previous is not None:
            if previous.resolve() != path.resolve():
                raise SystemExit(f"duplicate case_id in replay set: {case_id} ({previous} vs {path})")
            continue
        seen[case_id] = path
        out.append(path)
    return out


def _promote_legacy_continuous_mms_cli(args, *, round_dir: Path) -> None:
    if not bool(args.skip_hard_verifier) or bool(args.require_continuous_mms_cases):
        return
    receipt_path = round_dir / "precomputed_hard_audit.json"
    if not receipt_path.exists():
        return
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    if (
        receipt.get("schema") != "precomputed_continuous_hard_audit_receipt_v1"
        or receipt.get("truth_contract") != CONTINUOUS_MMS_TRUTH_CONTRACT
        or receipt.get("all_hard_passed") is not True
    ):
        return
    args.skip_hard_verifier = False
    args.require_continuous_mms_cases = True
    payload = {
        "schema": "continuous_mms_legacy_cli_promotion_v1",
        "truth_contract": CONTINUOUS_MMS_TRUTH_CONTRACT,
        "legacy_skip_hard_verifier_promoted": True,
        "source_receipt": str(receipt_path),
    }
    (round_dir / "continuous_mms_cli_promotion.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def _hard_filter_cases(case_dirs: list[Path], *, round_dir: Path, args) -> tuple[list[Path], dict[str, Any]]:
    receipt = None
    if args.require_continuous_mms_cases:
        rows, receipt = _preverified_continuous_mms_rows(case_dirs, round_dir=round_dir)
        mode = "continuous_mms_contract_only"
        enabled = True
    elif args.skip_hard_verifier:
        rows = [{"case_id": path.name, "accepted": True, "hard_passed": True, "skipped": True, "failures": [], "metrics": {}} for path in case_dirs]
        mode = "legacy_external_bypass"
        enabled = False
    else:
        rows = [_generic_hard_verify_case(path, args=args) for path in case_dirs]
        mode = "legacy_grid_finite_difference"
        enabled = True
    accepted = [path for path, row in zip(case_dirs, rows) if bool(row.get("accepted"))]
    summary = {
        "enabled": enabled,
        "mode": mode,
        "truth_contract": CONTINUOUS_MMS_TRUTH_CONTRACT if args.require_continuous_mms_cases else None,
        "upstream_receipt": receipt,
        "n": len(rows),
        "accepted": len(accepted),
        "hard_passed": int(sum(bool(row.get("hard_passed")) for row in rows)),
        "contract_passed": int(sum(bool(row.get("contract_passed")) for row in rows)),
        "failure_counts": _failure_counts(rows),
    }
    payload = {"summary": summary, "results": rows}
    (round_dir / "hard_verification.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (round_dir / "hard_verification.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return accepted, summary


def _preverified_continuous_mms_rows(case_dirs: list[Path], *, round_dir: Path) -> tuple[list[dict[str, Any]], str | None]:
    receipt_path = round_dir / "precomputed_continuous_contract.json"
    receipt = None
    if receipt_path.exists():
        try:
            payload = json.loads(receipt_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid continuous MMS contract receipt: {receipt_path}") from exc
        if (
            payload.get("schema") != "precomputed_continuous_contract_receipt_v1"
            or payload.get("truth_contract") != CONTINUOUS_MMS_TRUTH_CONTRACT
            or payload.get("contract_passed") is not True
            or int(payload.get("linked_cases", -1)) != len(case_dirs)
        ):
            raise SystemExit(f"invalid continuous MMS contract receipt: {receipt_path}")
        receipt = str(receipt_path)
    rows = []
    for case_dir in case_dirs:
        try:
            require_continuous_mms_case_contract(case_dir)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        metadata_path = case_dir / "generator_metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        verifier = metadata.get("verifier") or {}
        rows.append(
            {
                "case_id": case_dir.name,
                "case_dir": str(case_dir),
                "accepted": True,
                "hard_passed": None,
                "contract_passed": True,
                "distribution_passed": verifier.get("distribution_passed"),
                "verification_mode": "continuous_mms_contract_only",
                "failures": [],
                "metrics": verifier.get("metrics") or {},
            }
        )
    return rows, receipt


def _require_continuous_case_dirs(case_dirs: list[Path]) -> None:
    for case_dir in case_dirs:
        try:
            require_continuous_mms_case_contract(case_dir)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_reward_spec(*, metric: str, normalizer_checkpoint: str) -> dict[str, Any]:
    if metric == "relative_l2":
        return {"metric": metric}
    checkpoint = Path(normalizer_checkpoint)
    if not normalizer_checkpoint or not checkpoint.exists():
        raise SystemExit(
            "train_scaled_velocity_rmse requires an existing --reward-normalizer-checkpoint"
        )
    try:
        return reward_spec_from_autoencoder_checkpoint(checkpoint)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def _generic_hard_verify_case(case_dir: Path, *, args) -> dict[str, Any]:
    failures: list[str] = []
    metrics: dict[str, float] = {}
    try:
        case = load_case(case_dir)
        public = case.public_arrays
        hidden = case.hidden_arrays
        required_public = {"x", "y", "t", "initial_velocity", "forcing", "fluid_mask", "obstacle_mask", "outer_boundary_mask"}
        required_hidden = {"u_true", "pressure"}
        missing_public = sorted(required_public - set(public))
        missing_hidden = sorted(required_hidden - set(hidden))
        if missing_public:
            failures.append(f"missing_public_arrays:{','.join(missing_public)}")
        if missing_hidden:
            failures.append(f"missing_hidden_arrays:{','.join(missing_hidden)}")
        if failures:
            return _hard_row(case_dir, failures=failures, metrics=metrics)
        velocity = np.asarray(hidden["u_true"], dtype=np.float64)
        pressure = np.asarray(hidden["pressure"], dtype=np.float64)
        forcing = np.asarray(public["forcing"], dtype=np.float64)
        initial = np.asarray(public["initial_velocity"], dtype=np.float64)
        fluid = np.asarray(public["fluid_mask"], dtype=bool)
        obstacle = np.asarray(public["obstacle_mask"], dtype=bool)
        outer = np.asarray(public["outer_boundary_mask"], dtype=bool)
        x = np.asarray(public["x"], dtype=np.float64)
        y = np.asarray(public["y"], dtype=np.float64)
        t = np.asarray(public["t"], dtype=np.float64)
        if velocity.ndim != 4 or velocity.shape[1] != 2:
            failures.append("u_true_shape_invalid")
        if velocity.ndim == 4 and pressure.shape != (velocity.shape[0], velocity.shape[2], velocity.shape[3]):
            failures.append("pressure_shape_invalid")
        if velocity.ndim == 4 and forcing.shape != velocity.shape:
            failures.append("forcing_shape_invalid")
        if velocity.ndim == 4 and initial.shape != velocity[0].shape:
            failures.append("initial_velocity_shape_invalid")
        if velocity.ndim == 4 and t.shape != (velocity.shape[0],):
            failures.append("time_shape_invalid")
        if velocity.ndim == 4 and fluid.shape != velocity.shape[-2:]:
            failures.append("fluid_mask_shape_invalid")
        if obstacle.shape != fluid.shape or outer.shape != fluid.shape:
            failures.append("geometry_mask_shape_invalid")
        finite_arrays = [velocity, pressure, forcing, initial, x, y]
        if not all(np.all(np.isfinite(arr)) for arr in finite_arrays):
            failures.append("non_finite_arrays")
        if failures:
            return _hard_row(case_dir, failures=failures, metrics=metrics)
        fluid_eval = fluid if np.any(fluid) else np.ones_like(fluid, dtype=bool)
        velocity_scale = _masked_rms(velocity, fluid_eval) + 1.0e-12
        forcing_scale = _masked_rms(forcing, fluid_eval) + 1.0e-12
        spacing = max(min(_median_spacing(x), _median_spacing(y)), 1.0e-12)
        interior = fluid_eval & ~outer & ~obstacle
        if not np.any(interior):
            interior = fluid_eval
        div = divergence(velocity, x, y)
        metrics = {
            "initial_velocity_rel": float(np.linalg.norm(initial - velocity[0]) / (np.linalg.norm(velocity[0]) + 1.0e-12)),
            "divergence_rel": _masked_rms_scalar(div, interior) / (velocity_scale / spacing + 1.0e-12),
            "outer_boundary_velocity_rel": _masked_rms(velocity, outer) / velocity_scale if np.any(outer) else 0.0,
            "obstacle_velocity_rel": _masked_rms(velocity, obstacle) / velocity_scale if np.any(obstacle) else 0.0,
            "forcing_inside_obstacle_rel": _masked_rms(forcing, obstacle) / forcing_scale if np.any(obstacle) else 0.0,
        }
        _check_max(failures, metrics, "initial_velocity_rel", 1.0e-6)
        _check_max(failures, metrics, "divergence_rel", float(args.hard_divergence_rel_max))
        _check_max(failures, metrics, "outer_boundary_velocity_rel", float(args.hard_boundary_velocity_rel_max))
        _check_max(failures, metrics, "obstacle_velocity_rel", float(args.hard_obstacle_velocity_rel_max))
        _check_max(failures, metrics, "forcing_inside_obstacle_rel", float(args.hard_forcing_obstacle_rel_max))
    except Exception as exc:  # noqa: BLE001
        failures.append(f"exception:{type(exc).__name__}:{exc}")
    return _hard_row(case_dir, failures=failures, metrics=metrics)


def _hard_row(case_dir: Path, *, failures: list[str], metrics: dict[str, float]) -> dict[str, Any]:
    hard_passed = not failures
    return {
        "case_id": case_dir.name,
        "case_dir": str(case_dir),
        "accepted": hard_passed,
        "hard_passed": hard_passed,
        "distribution_passed": None,
        "metrics": metrics,
        "failures": failures,
    }


def _check_max(failures: list[str], metrics: dict[str, float], name: str, maximum: float) -> None:
    value = float(metrics.get(name, np.inf))
    if not np.isfinite(value) or value > float(maximum):
        failures.append(f"{name}>{float(maximum):.6g}")


def _failure_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for failure in row.get("failures", []):
            counts[str(failure)] = counts.get(str(failure), 0) + 1
    return counts


def _masked_rms(vector: np.ndarray, mask: np.ndarray) -> float:
    arr = np.asarray(vector, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr[:, :, m]))))


def _masked_rms_scalar(field: np.ndarray, mask: np.ndarray) -> float:
    arr = np.asarray(field, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr[:, m]))))


def _median_spacing(axis: np.ndarray) -> float:
    values = np.asarray(axis, dtype=np.float64).reshape(-1)
    if values.size < 2:
        return 1.0
    return float(np.median(np.abs(np.diff(values))))


def _copy_public_cases(case_dirs: list[Path], dst_root: Path, *, case_aliases: dict[str, str] | None = None) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)
    for src in case_dirs:
        public_name = case_aliases.get(src.name, src.name) if case_aliases else src.name
        dst = dst_root / public_name
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("public_question.json", "public_arrays.npz"):
            src_path = src / name
            if not src_path.exists():
                continue
            dst_path = dst / name
            if case_aliases and name == "public_question.json":
                payload = json.loads(src_path.read_text())
                payload["case_id"] = public_name
                payload["case_source"] = "public_example"
                dst_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            else:
                _link_or_copy_public_file(src_path, dst_path)


def _link_or_copy_public_file(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _prepare_public_case_cache(case_dirs: list[Path], dst_root: Path, *, metadata_path: Path) -> None:
    fingerprint = _public_case_cache_fingerprint(case_dirs)
    expected = {
        "abi": "public_case_cache_v1",
        "case_count": len(case_dirs),
        "fingerprint": fingerprint,
    }
    if metadata_path.exists():
        try:
            existing = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            existing = None
        if existing == expected and _public_case_cache_complete(case_dirs, dst_root):
            return
    if dst_root.exists():
        shutil.rmtree(dst_root)
    _copy_public_cases(case_dirs, dst_root)
    if not _public_case_cache_complete(case_dirs, dst_root):
        raise RuntimeError(f"public case cache is incomplete: {dst_root}")
    metadata_path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")


def _public_case_cache_fingerprint(case_dirs: list[Path]) -> str:
    rows = []
    for case_dir in case_dirs:
        files = []
        for name in ("public_question.json", "public_arrays.npz"):
            path = case_dir / name
            if path.exists():
                stat = path.stat()
                files.append({"name": name, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
        rows.append({"case_dir": str(case_dir), "files": files})
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _public_case_cache_complete(case_dirs: list[Path], dst_root: Path) -> bool:
    for case_dir in case_dirs:
        dst = dst_root / case_dir.name
        for name in ("public_question.json", "public_arrays.npz"):
            if (case_dir / name).exists() and not (dst / name).exists():
                return False
    return True


def _write_private_manifest_with_aliases(
    path: str | Path,
    *,
    public_cases_root: str | Path,
    private_case_dirs: list[str | Path],
    case_aliases: dict[str, str],
    truth_contract: str | None = None,
    reward_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    public_root = Path(public_cases_root)
    cases = []
    for private_case_dir in private_case_dirs:
        private_path = Path(private_case_dir)
        alias = case_aliases[private_path.name]
        cases.append(
            {
                "case_id": alias,
                "public_case_dir": str(public_root / alias),
                "private_case_dir": str(private_path),
            }
        )
    payload = {
        "public_cases_root": str(public_root),
        "cases": cases,
        "note": "Private evaluator manifest with public anonymous case aliases. Do not place this file in a solver agent workspace.",
    }
    if truth_contract is not None:
        payload["truth_contract"] = str(truth_contract)
    if reward_spec is not None:
        payload["reward_spec"] = dict(reward_spec)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _run_solver_agent(
    *,
    backend: str,
    round_index: int,
    workspace: Path,
    prompt_path: Path,
    private_manifest: Path,
    round_dir: Path,
    args,
) -> Path | None:
    submission = workspace / "submission.py"
    if backend == "codex_mock":
        _write_mock_solver(submission)
        invocation = {
            "round_index": int(round_index),
            "backend": backend,
            "actual_codex_invoked": False,
            "submission_exists": True,
            "generated_submission_path": str(submission),
        }
        (round_dir / "codex_invocation.json").write_text(json.dumps(invocation, indent=2, sort_keys=True) + "\n")
        return submission
    if backend.startswith("python_agent:"):
        src = Path(backend.split(":", 1)[1])
        shutil.copy2(src, submission)
        (round_dir / "python_agent_invocation.json").write_text(json.dumps({"source": str(src), "submission": str(submission)}, indent=2, sort_keys=True) + "\n")
        return submission
    if backend != "codex":
        raise ValueError(f"unsupported solver backend: {backend}")

    configured_timeout = int(args.codex_timeout_seconds)
    persistent_budget = max(
        0, int(getattr(args, "codex_persistent_search_seconds", 0))
    )
    persistent_search = persistent_budget > 0
    persistent_attempt_cap = max(
        0, int(getattr(args, "codex_persistent_search_max_attempts", 0))
    )
    normal_attempt_cap = max(1, int(args.codex_max_attempts))
    max_attempts = (
        persistent_attempt_cap
        if persistent_search and persistent_attempt_cap > 0
        else (1_000_000 if persistent_search else normal_attempt_cap)
    )
    client = CodexCliClient(
        command_template=args.codex_command_template,
        timeout_seconds=configured_timeout,
        model=args.codex_model,
        reasoning_effort=getattr(args, "codex_reasoning_effort", ""),
        allow_network=bool(args.codex_allow_network),
        read_isolated=True,
        mock=False,
    )
    attempt_records: list[dict[str, Any]] = []
    codex_wall_time_seconds = 0.0
    search_root = round_dir / "codex_persistent_search"
    champion_path = search_root / "champion" / "submission.py"
    champion_score: float | None = None
    champion_attempt: int | None = None
    if persistent_search:
        search_root.mkdir(parents=True, exist_ok=True)
        (workspace / "search_champion").mkdir(parents=True, exist_ok=True)

    old_manifest = os.environ.pop("AVF_TRIPLANE_EVAL_MANIFEST", None)
    broker = WorkspaceEvaluatorBroker(
        workspace=workspace,
        private_manifest=private_manifest,
        audit_path=round_dir / "private_eval" / "evaluator_broker_audit.jsonl",
    )
    broker_environment: dict[str, str] = {}
    old_broker_environment: dict[str, str | None] = {}
    try:
        broker.__enter__()
        broker_environment = broker.environment
        old_broker_environment = {
            key: os.environ.get(key)
            for key in broker_environment
        }
        os.environ.update(broker_environment)
        for attempt in range(max_attempts):
            if persistent_search and codex_wall_time_seconds >= persistent_budget:
                break
            if persistent_search:
                remaining = max(1.0, persistent_budget - codex_wall_time_seconds)
                client.timeout_seconds = max(
                    1, min(configured_timeout, int(remaining + 0.999999))
                )

            if attempt == 0 and bool(getattr(args, "codex_resume_finish", False)):
                attempt_prompt_path = workspace / "prompt_resume_finish.md"
                attempt_prompt = (
                    "A previous solver-agent attempt timed out after extensive experiments in this existing workspace. "
                    "Do not start any new broad search or parameter sweep. Read the existing evaluation files, candidate solvers, saved "
                    "analysis, and the end of the timed-out attempt log. If that work reached a useful general fit or conclusion without "
                    "materializing it, implement and directly compare that candidate now. Do not merely copy previous_solver/submission.py "
                    "unless the existing evidence and one direct comparison show that it remains best. Write the selected method to "
                    "submission.py as one self-contained file, run a syntax check and focused interface validation, record one concise "
                    "durable memory note if useful, and exit promptly.\n"
                )
                attempt_prompt_path.write_text(attempt_prompt)
                logs_dir = round_dir / "logs" / "resume_finish"
            elif attempt == 0:
                attempt_prompt_path = prompt_path
                attempt_prompt = prompt_path.read_text()
                logs_dir = round_dir / "logs" / "search_000" if persistent_search else round_dir / "logs"
            elif persistent_search:
                attempt_prompt_path = workspace / f"prompt_search_{attempt:03d}.md"
                attempt_prompt = _persistent_solver_search_prompt(
                    attempt=attempt,
                    requested_seconds=persistent_budget,
                    consumed_seconds=codex_wall_time_seconds,
                    champion_score=champion_score,
                )
                attempt_prompt_path.write_text(attempt_prompt)
                logs_dir = round_dir / "logs" / f"search_{attempt:03d}"
            else:
                attempt_prompt_path = workspace / f"prompt_retry_{attempt:03d}.md"
                attempt_prompt = (
                    "The previous solver-agent attempt did not complete cleanly. Continue from the work already in this workspace. "
                    "Inspect the existing candidate solvers, saved analysis, evaluation files, and the end of the previous attempt log. "
                    "A timed-out attempt may have reached a useful conclusion or fit without materializing and evaluating it; if so, "
                    "implement that general candidate and run a focused validation. Do not restart broad experiments. Do not merely copy "
                    "previous_solver/submission.py unless the existing evidence and one direct comparison show that no completed or nearly "
                    "completed general candidate improves it. If submission_preflight.json reports a failure, fix it by putting every "
                    "required solver definition directly in submission.py; sibling candidate files are not exported. Finish promptly with "
                    "a valid, self-contained submission.py and a clean exit.\n"
                )
                attempt_prompt_path.write_text(attempt_prompt)
                logs_dir = round_dir / "logs" / f"retry_{attempt:03d}"

            invocation_started = time.perf_counter()
            record = client.invoke(
                round_index=round_index,
                workspace=workspace,
                prompt_text=attempt_prompt,
                prompt_path=attempt_prompt_path,
                logs_dir=logs_dir,
                expected_submission=submission,
            )
            measured_wall_time = time.perf_counter() - invocation_started
            record_payload = record.as_dict()
            reported_wall_time = record_payload.get("wall_time_seconds")
            if not isinstance(reported_wall_time, (int, float)):
                reported_wall_time = measured_wall_time
            reported_wall_time = max(0.0, float(reported_wall_time))
            if persistent_search:
                codex_wall_time_seconds += reported_wall_time
                record_payload["persistent_search_cumulative_codex_wall_time_seconds"] = codex_wall_time_seconds
                record_payload["persistent_search_requested_codex_wall_time_seconds"] = persistent_budget

            basic_clean_attempt = (
                record.returncode == 0
                and not bool(record.timed_out)
                and bool(record.submission_exists)
                and submission.exists()
            )
            standalone_preflight = None
            may_preflight = basic_clean_attempt or (
                persistent_search and bool(record.submission_exists) and submission.exists()
            )
            if may_preflight:
                standalone_preflight = _standalone_submission_preflight(
                    submission=submission,
                    workspace=workspace,
                    private_manifest=private_manifest,
                    round_dir=round_dir,
                    nx=int(getattr(args, "eval_nx", 0)),
                    ny=int(getattr(args, "eval_ny", 0)),
                    nt=int(getattr(args, "eval_nt", 0)),
                    failure_reward=float(getattr(args, "failure_reward", 1.0e6)),
                )
                record_payload["standalone_preflight"] = standalone_preflight

            candidate_valid = bool((standalone_preflight or {}).get("passed"))
            if persistent_search and candidate_valid:
                candidate = _evaluate_persistent_solver_candidate(
                    attempt=attempt,
                    submission=submission,
                    workspace=workspace,
                    private_manifest=private_manifest,
                    round_dir=round_dir,
                    args=args,
                )
                candidate_score = candidate.get("score")
                accepted = bool(
                    candidate.get("valid")
                    and isinstance(candidate_score, (int, float))
                    and (
                        champion_score is None
                        or float(candidate_score) < float(champion_score) - 1.0e-12
                    )
                )
                if accepted:
                    champion_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(Path(str(candidate["candidate_submission"])), champion_path)
                    shutil.copy2(champion_path, workspace / "search_champion" / "submission.py")
                    champion_score = float(candidate_score)
                    champion_attempt = int(attempt)
                elif champion_path.exists():
                    shutil.copy2(champion_path, submission)
                record_payload["persistent_search_candidate"] = candidate
                record_payload["persistent_search_accepted_as_champion"] = accepted
                _append_jsonl(
                    workspace / "search_history.jsonl",
                    [
                        {
                            "attempt": int(attempt),
                            "candidate_valid": bool(candidate.get("valid")),
                            "candidate_score_metric": candidate.get("score_metric"),
                            "candidate_score": candidate_score,
                            "accepted_as_champion": accepted,
                            "champion_attempt": champion_attempt,
                            "champion_score": champion_score,
                            "cumulative_codex_wall_time_seconds": codex_wall_time_seconds,
                            "note": "Scores use only the agent-visible development evaluator; lower is better.",
                        }
                    ],
                )
            elif persistent_search and champion_path.exists():
                shutil.copy2(champion_path, submission)

            clean_attempt = basic_clean_attempt and candidate_valid
            attempt_records.append(record_payload)
            if clean_attempt and not persistent_search:
                break
    finally:
        for key in broker_environment:
            old_value = old_broker_environment.get(key)
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
        broker.close()
        if old_manifest is None:
            os.environ.pop("AVF_TRIPLANE_EVAL_MANIFEST", None)
        else:
            os.environ["AVF_TRIPLANE_EVAL_MANIFEST"] = old_manifest

    if not attempt_records:
        raise RuntimeError("solver agent made no Codex invocation")
    persistent_budget_completed = (
        persistent_search and codex_wall_time_seconds >= float(persistent_budget)
    )
    if persistent_search and champion_path.exists():
        shutil.copy2(champion_path, submission)
    invocation = dict(attempt_records[-1])
    invocation["attempt_count"] = len(attempt_records)
    invocation["max_attempts"] = (
        persistent_attempt_cap if persistent_search else normal_attempt_cap
    )
    invocation["attempts"] = attempt_records
    invocation["evaluator_transport"] = "unix_socket_scalar_broker_v1"
    invocation["private_manifest_exposed_to_agent"] = False
    invocation["filesystem_read_isolation"] = "bubblewrap_workspace_only_v1"
    invocation["parent_project_and_scratch_visible_to_agent"] = False
    if persistent_search:
        invocation["persistent_search"] = {
            "enabled": True,
            "requested_codex_wall_time_seconds": persistent_budget,
            "consumed_codex_wall_time_seconds": codex_wall_time_seconds,
            "budget_completed": persistent_budget_completed,
            "attempt_cap": persistent_attempt_cap,
            "attempt_count": len(attempt_records),
            "champion_attempt": champion_attempt,
            "champion_score_metric": "mean_difficulty_reward",
            "champion_score": champion_score,
            "champion_submission": str(champion_path) if champion_path.exists() else None,
            "stop_reason": (
                "budget_consumed"
                if persistent_budget_completed
                else "attempt_cap_reached_before_budget"
            ),
        }
        clean_invocation = bool(
            persistent_budget_completed
            and champion_path.exists()
            and submission.exists()
            and champion_score is not None
        )
    else:
        clean_invocation = (
            invocation.get("returncode") == 0
            and not bool(invocation.get("timed_out"))
            and bool(invocation.get("submission_exists"))
            and submission.exists()
            and bool((invocation.get("standalone_preflight") or {}).get("passed"))
        )
    invocation["clean_invocation"] = clean_invocation
    (round_dir / "codex_invocation.json").write_text(json.dumps(invocation, indent=2, sort_keys=True) + "\n")
    return submission if clean_invocation else None


def _evaluate_persistent_solver_candidate(
    *,
    attempt: int,
    submission: Path,
    workspace: Path,
    private_manifest: Path,
    round_dir: Path,
    args,
) -> dict[str, Any]:
    attempt_dir = round_dir / "codex_persistent_search" / f"attempt_{attempt:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    candidate_submission = attempt_dir / "submission.py"
    shutil.copy2(submission, candidate_submission)
    preflight = round_dir / "submission_preflight.json"
    if preflight.exists():
        shutil.copy2(preflight, attempt_dir / "submission_preflight.json")
    try:
        evaluation = evaluate_solver_file(
            solver_path=candidate_submission,
            public_cases_root=workspace / "cases",
            private_manifest_path=private_manifest,
            nx=int(getattr(args, "eval_nx", 0)),
            ny=int(getattr(args, "eval_ny", 0)),
            nt=int(getattr(args, "eval_nt", 0)),
            failure_reward=float(getattr(args, "failure_reward", 1.0e6)),
            max_workers=int(getattr(args, "eval_workers", 1)),
        )
        (attempt_dir / "development_evaluation.json").write_text(
            json.dumps(evaluation, indent=2, sort_keys=True) + "\n"
        )
        summary = evaluation.get("summary") or {}
        score = summary.get("mean_difficulty_reward")
        valid = bool(
            isinstance(score, (int, float))
            and np.isfinite(float(score))
            and int(summary.get("n", 0)) > 0
            and int(summary.get("ok", -1)) == int(summary.get("n", 0))
        )
        failure = None
    except Exception as exc:  # noqa: BLE001
        summary = None
        score = None
        valid = False
        failure = repr(exc)
    result = {
        "attempt": int(attempt),
        "candidate_submission": str(candidate_submission),
        "candidate_sha256": _file_sha256(candidate_submission),
        "score_metric": "mean_difficulty_reward",
        "score": float(score) if isinstance(score, (int, float)) else None,
        "valid": valid,
        "summary": summary,
        "failure": failure,
    }
    (attempt_dir / "candidate_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def _standalone_submission_preflight(
    *,
    submission: Path,
    workspace: Path,
    private_manifest: Path,
    round_dir: Path,
    nx: int,
    ny: int,
    nt: int,
    failure_reward: float,
) -> dict[str, Any]:
    standalone_root = round_dir / "submission_preflight"
    if standalone_root.exists():
        shutil.rmtree(standalone_root)
    standalone_root.mkdir(parents=True, exist_ok=True)
    standalone_submission = standalone_root / "submission.py"
    shutil.copy2(submission, standalone_submission)

    source_manifest = json.loads(private_manifest.read_text())
    cases = list(source_manifest.get("cases") or [])
    if not cases:
        raise ValueError(f"submission preflight manifest has no cases: {private_manifest}")
    preflight_manifest_payload = dict(source_manifest)
    preflight_manifest_payload["cases"] = [cases[0]]
    preflight_manifest = standalone_root / "private_manifest.json"
    preflight_manifest.write_text(
        json.dumps(preflight_manifest_payload, indent=2, sort_keys=True) + "\n"
    )

    try:
        evaluation = evaluate_solver_file(
            solver_path=standalone_submission,
            public_cases_root=workspace / "cases",
            private_manifest_path=preflight_manifest,
            nx=int(nx),
            ny=int(ny),
            nt=int(nt),
            failure_reward=float(failure_reward),
        )
        first_result = evaluation["results"][0]
        passed = first_result.get("status") == "ok"
        failure = None if passed else {
            "status": first_result.get("status"),
            "exception": first_result.get("exception"),
        }
        summary = evaluation.get("summary")
    except Exception as exc:  # noqa: BLE001
        passed = False
        failure = {"status": "preflight_exception", "exception": repr(exc)}
        summary = None
    report = {
        "schema": "triplane_standalone_submission_preflight_v1",
        "passed": bool(passed),
        "source_submission": str(submission),
        "standalone_submission": str(standalone_submission),
        "private_manifest": str(preflight_manifest),
        "case_id": str(cases[0]["case_id"]),
        "query_density": {"nx": int(nx), "ny": int(ny), "nt": int(nt)},
        "summary": summary,
        "failure": failure,
    }
    (round_dir / "submission_preflight.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def _formal_evaluation_is_cumulative(args, has_historical_replay: bool) -> bool:
    scope = str(args.formal_evaluation_scope)
    if scope == "current":
        if bool(args.cumulative_reward_table):
            raise ValueError("--cumulative-reward-table requires --formal-evaluation-scope cumulative or auto")
        return False
    if scope == "cumulative":
        return True
    if scope != "auto":
        raise ValueError(f"unsupported formal evaluation scope: {scope}")
    return bool(args.cumulative_reward_table) or bool(has_historical_replay)


def _evaluate_or_missing(*, submission: Path | None, public_cases_root: Path, private_manifest: Path, args, label: str) -> dict[str, Any]:
    if submission is None or not submission.exists():
        manifest = json.loads(private_manifest.read_text())
        reward_metric = str((manifest.get("reward_spec") or {}).get("metric") or "relative_l2")
        rows = [
            {
                "case_id": row["case_id"],
                "status": "missing_submission",
                "relative_l2": float(args.failure_reward),
                "difficulty_reward": float(args.failure_reward),
                "reward_metric": reward_metric,
                "velocity_rmse": float(args.failure_reward),
                "truth_velocity_rms": 0.0,
                "train_scaled_velocity_rmse": float(args.failure_reward),
                "wall_time_seconds": 0.0,
            }
            for row in manifest["cases"]
        ]
        return {"solver": None, "cases_root": str(public_cases_root), "private_manifest": str(private_manifest), "label": label, "reward_spec": manifest.get("reward_spec"), "summary": summarize_metric_rows(rows), "results": rows}
    payload = evaluate_solver_file(
        solver_path=submission,
        public_cases_root=public_cases_root,
        private_manifest_path=private_manifest,
        nx=int(getattr(args, "eval_nx", 0)),
        ny=int(getattr(args, "eval_ny", 0)),
        nt=int(getattr(args, "eval_nt", 0)),
        failure_reward=float(getattr(args, "failure_reward", 1.0e6)),
        checkpoint_path=private_manifest.parent.parent / "eval_checkpoints" / f"{label}.json",
        checkpoint_interval=64,
        max_workers=int(getattr(args, "eval_workers", 1)),
    )
    payload["label"] = label
    return payload


def _evaluate_previous_solver_on_cumulative_cases(
    *,
    submission: Path,
    public_cases_root: Path,
    cumulative_manifest: Path,
    current_manifest: Path,
    args,
    label: str,
) -> dict[str, Any]:
    """Evaluate the previous solver without repeating completed historical cases.

    Round ``r - 1`` already evaluated its submitted solver on every case from
    rounds ``0..r - 1``. In round ``r`` that exact file becomes the previous
    solver, so only the newly generated cases need another evaluation. Prior
    rows are reused only after checking the solver, settings, IDs, and private
    case paths. If any check fails, the normal full evaluation is used.
    """

    reusable = _load_reusable_previous_round_evaluation(
        submission=submission,
        cumulative_manifest=cumulative_manifest,
        current_manifest=current_manifest,
        args=args,
    )
    if reusable is None:
        return _evaluate_or_missing(
            submission=submission,
            public_cases_root=public_cases_root,
            private_manifest=cumulative_manifest,
            args=args,
            label=label,
        )

    previous_payload, previous_metrics_path, expected_case_ids, current_case_ids = reusable
    current_case_id_set = set(current_case_ids)
    previous_rows = {
        str(row["case_id"]): row for row in previous_payload.get("results", [])
    }
    new_payload = evaluate_solver_file(
        solver_path=submission,
        public_cases_root=public_cases_root,
        private_manifest_path=cumulative_manifest,
        nx=int(getattr(args, "eval_nx", 0)),
        ny=int(getattr(args, "eval_ny", 0)),
        nt=int(getattr(args, "eval_nt", 0)),
        failure_reward=float(getattr(args, "failure_reward", 1.0e6)),
        checkpoint_path=(
            cumulative_manifest.parent.parent
            / "eval_checkpoints"
            / f"{label}_current_round.json"
        ),
        checkpoint_interval=64,
        selected_case_ids=current_case_ids,
        max_workers=int(getattr(args, "eval_workers", 1)),
    )
    new_rows = {str(row["case_id"]): row for row in new_payload.get("results", [])}
    if set(new_rows) != current_case_id_set:
        raise RuntimeError(
            "current-round evaluation returned an unexpected case set: "
            f"expected {len(current_case_id_set)}, received {len(new_rows)}"
        )

    rows_by_case_id = {**previous_rows, **new_rows}
    missing = [case_id for case_id in expected_case_ids if case_id not in rows_by_case_id]
    if missing:
        raise RuntimeError(f"reused cumulative evaluation is missing {len(missing)} cases")
    rows = [rows_by_case_id[case_id] for case_id in expected_case_ids]
    payload = dict(new_payload)
    payload.update(
        {
            "private_manifest": str(cumulative_manifest),
            "label": label,
            "summary": summarize_metric_rows(rows),
            "results": rows,
            "evaluation_reuse": {
                "source": str(previous_metrics_path),
                "reused_historical_cases": len(previous_rows),
                "evaluated_current_cases": len(new_rows),
            },
        }
    )
    return payload


def _load_reusable_previous_round_evaluation(
    *,
    submission: Path,
    cumulative_manifest: Path,
    current_manifest: Path,
    args,
) -> tuple[dict[str, Any], Path, list[str], list[str]] | None:
    """Return a compatible prior cumulative result, or ``None`` to recompute."""

    round_dir = cumulative_manifest.parent.parent
    try:
        round_index = int(round_dir.name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None
    if round_index <= 0:
        return None

    previous_metrics_path = (
        round_dir.parent
        / f"round_{round_index - 1:03d}"
        / "cumulative_generated_solver_metrics.json"
    )
    if not previous_metrics_path.is_file():
        return None
    try:
        previous_payload = json.loads(previous_metrics_path.read_text())
        cumulative_payload = json.loads(cumulative_manifest.read_text())
        current_payload = json.loads(current_manifest.read_text())
        previous_manifest_path = Path(str(previous_payload["private_manifest"]))
        previous_manifest = json.loads(previous_manifest_path.read_text())
        previous_solver = Path(str(previous_payload["solver"]))
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None

    if not previous_solver.is_file() or _file_sha256(previous_solver) != _file_sha256(submission):
        return None
    desired_density = {
        "nx": int(getattr(args, "eval_nx", 0)),
        "ny": int(getattr(args, "eval_ny", 0)),
        "nt": int(getattr(args, "eval_nt", 0)),
    }
    if previous_payload.get("query_density") != desired_density:
        return None
    if previous_payload.get("solver_input_mode", "query") != "query":
        return None
    if previous_payload.get("reward_spec") != cumulative_payload.get("reward_spec"):
        return None
    if previous_manifest.get("truth_contract") != cumulative_payload.get("truth_contract"):
        return None

    expected_case_ids = [str(row["case_id"]) for row in cumulative_payload.get("cases", [])]
    current_case_ids = [str(row["case_id"]) for row in current_payload.get("cases", [])]
    if (
        not expected_case_ids
        or not current_case_ids
        or len(expected_case_ids) != len(set(expected_case_ids))
        or len(current_case_ids) != len(set(current_case_ids))
    ):
        return None
    current_case_id_set = set(current_case_ids)
    expected_case_id_set = set(expected_case_ids)
    if not current_case_id_set.issubset(expected_case_id_set):
        return None
    historical_case_ids = expected_case_id_set - current_case_id_set

    previous_rows = previous_payload.get("results")
    if not isinstance(previous_rows, list):
        return None
    previous_row_ids = [str(row.get("case_id")) for row in previous_rows]
    if (
        set(previous_row_ids) != historical_case_ids
        or len(previous_row_ids) != len(historical_case_ids)
    ):
        return None
    if any(str(row.get("status")) != "ok" for row in previous_rows):
        return None

    previous_cases = {
        str(row["case_id"]): str(Path(row["private_case_dir"]).resolve())
        for row in previous_manifest.get("cases", [])
    }
    cumulative_cases = {
        str(row["case_id"]): str(Path(row["private_case_dir"]).resolve())
        for row in cumulative_payload.get("cases", [])
    }
    if any(
        previous_cases.get(case_id) != cumulative_cases.get(case_id)
        for case_id in historical_case_ids
    ):
        return None

    return previous_payload, previous_metrics_path, expected_case_ids, current_case_ids


def _subset_evaluation(source: dict[str, Any], *, target_manifest: Path, label: str) -> dict[str, Any]:
    manifest = json.loads(target_manifest.read_text())
    source_rows = {str(row["case_id"]): row for row in source.get("results", [])}
    case_ids = [str(row["case_id"]) for row in manifest.get("cases", [])]
    missing = [case_id for case_id in case_ids if case_id not in source_rows]
    if missing:
        raise RuntimeError(f"cumulative evaluation is missing {len(missing)} requested current-round cases")
    rows = [source_rows[case_id] for case_id in case_ids]
    payload = dict(source)
    payload.update(
        {
            "private_manifest": str(target_manifest),
            "label": label,
            "summary": summarize_metric_rows(rows),
            "results": rows,
            "derived_from_cumulative_evaluation": True,
        }
    )
    return payload


def _write_eval_artifacts(round_dir: Path, stem: str, payload: dict[str, Any]) -> None:
    (round_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (round_dir / f"{stem}.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in payload.get("results", [])))


def _write_reward_table(path: Path, generated_eval: dict[str, Any], *, case_id_aliases: dict[str, str] | None = None) -> list[dict[str, Any]]:
    rows = []
    for row in generated_eval.get("results", []):
        public_case_id = str(row.get("case_id"))
        private_case_id = case_id_aliases.get(public_case_id, public_case_id) if case_id_aliases else public_case_id
        rows.append(
            {
                "case_id": private_case_id,
                "public_case_id": public_case_id if case_id_aliases else None,
                "reward": row.get("difficulty_reward"),
                "difficulty": row.get("difficulty_reward"),
                "reward_metric": row.get("reward_metric", "relative_l2"),
                "relative_l2": row.get("relative_l2"),
                "velocity_rmse": row.get("velocity_rmse"),
                "truth_velocity_rms": row.get("truth_velocity_rms"),
                "train_scaled_velocity_rmse": row.get("train_scaled_velocity_rmse"),
                "status": row.get("status"),
                "wall_time_seconds": row.get("wall_time_seconds"),
                "query_density": row.get("query_density"),
            }
        )
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return rows


def _write_policy_reward_table(
    path: Path,
    *,
    current_eval: dict[str, Any],
    previous_eval: dict[str, Any] | None,
    objective: str,
    case_id_aliases: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    if objective == "solver_error":
        rows = _write_reward_table(path, current_eval, case_id_aliases=case_id_aliases)
        for row in rows:
            row.update(
                {
                    "reward_objective": "solver_error",
                    "current_solver_loss": row.get("difficulty"),
                    "previous_solver_loss": None,
                    "solver_learning_progress": None,
                }
            )
        _write_jsonl_rows(path, rows)
        return rows
    if objective != "solver_learning_progress":
        raise ValueError(f"unsupported policy reward objective: {objective}")
    if previous_eval is None:
        raise ValueError("solver_learning_progress requires a previous-solver evaluation")

    current_rows = list(current_eval.get("results", []))
    previous_rows = list(previous_eval.get("results", []))
    previous_by_id = {str(row.get("case_id")): row for row in previous_rows}
    if len(previous_by_id) != len(previous_rows):
        raise ValueError("previous-solver evaluation contains duplicate case IDs")
    current_ids = [str(row.get("case_id")) for row in current_rows]
    if len(set(current_ids)) != len(current_ids):
        raise ValueError("current-solver evaluation contains duplicate case IDs")
    missing = [case_id for case_id in current_ids if case_id not in previous_by_id]
    extra = sorted(set(previous_by_id) - set(current_ids))
    if missing or extra:
        raise ValueError(
            "solver learning-progress evaluations do not cover identical cases: "
            f"missing_previous={len(missing)}, extra_previous={len(extra)}"
        )

    rows: list[dict[str, Any]] = []
    for current in current_rows:
        public_case_id = str(current.get("case_id"))
        previous = previous_by_id[public_case_id]
        current_metric = str(current.get("reward_metric", "relative_l2"))
        previous_metric = str(previous.get("reward_metric", "relative_l2"))
        if current_metric != previous_metric:
            raise ValueError(
                f"reward metric mismatch for {public_case_id}: {previous_metric} != {current_metric}"
            )
        current_status = str(current.get("status", "unknown"))
        previous_status = str(previous.get("status", "unknown"))
        paired_ok = current_status == "ok" and previous_status == "ok"
        current_loss = _finite_float_or_none(current.get("difficulty_reward"))
        previous_loss = _finite_float_or_none(previous.get("difficulty_reward"))
        paired_ok = paired_ok and current_loss is not None and previous_loss is not None
        learning_progress = float(previous_loss - current_loss) if paired_ok else None
        private_case_id = (
            case_id_aliases.get(public_case_id, public_case_id)
            if case_id_aliases
            else public_case_id
        )
        rows.append(
            {
                "case_id": private_case_id,
                "public_case_id": public_case_id if case_id_aliases else None,
                "reward": learning_progress,
                "reward_objective": "solver_learning_progress",
                "difficulty": current_loss,
                "reward_metric": current_metric,
                "previous_solver_loss": previous_loss,
                "current_solver_loss": current_loss,
                "solver_learning_progress": learning_progress,
                "previous_relative_l2": previous.get("relative_l2"),
                "relative_l2": current.get("relative_l2"),
                "velocity_rmse": current.get("velocity_rmse"),
                "truth_velocity_rms": current.get("truth_velocity_rms"),
                "train_scaled_velocity_rmse": current.get("train_scaled_velocity_rmse"),
                "previous_status": previous_status,
                "current_status": current_status,
                "status": "ok" if paired_ok else "invalid_solver_learning_progress_pair",
                "previous_wall_time_seconds": previous.get("wall_time_seconds"),
                "current_wall_time_seconds": current.get("wall_time_seconds"),
                "query_density": current.get("query_density"),
            }
        )
    _write_jsonl_rows(path, rows)
    return rows


def _summarize_policy_reward_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if row.get("status") == "ok" and _finite_float_or_none(row.get("reward")) is not None
    ]
    rewards = np.asarray([float(row["reward"]) for row in valid], dtype=np.float64)
    summary: dict[str, Any] = {
        "n": len(rows),
        "ok": len(valid),
        "status_counts": _value_counts(str(row.get("status", "unknown")) for row in rows),
    }
    if not valid:
        return summary
    summary.update(
        {
            "mean_reward": float(np.mean(rewards)),
            "median_reward": float(np.median(rewards)),
            "q10_reward": float(np.quantile(rewards, 0.10)),
            "q90_reward": float(np.quantile(rewards, 0.90)),
            "positive_fraction": float(np.mean(rewards > 0.0)),
            "negative_fraction": float(np.mean(rewards < 0.0)),
            "zero_fraction": float(np.mean(rewards == 0.0)),
        }
    )
    previous_losses = [
        float(row["previous_solver_loss"])
        for row in valid
        if _finite_float_or_none(row.get("previous_solver_loss")) is not None
    ]
    current_losses = [
        float(row["current_solver_loss"])
        for row in valid
        if _finite_float_or_none(row.get("current_solver_loss")) is not None
    ]
    if len(previous_losses) == len(valid) and len(current_losses) == len(valid):
        summary["mean_previous_solver_loss"] = float(np.mean(previous_losses))
        summary["mean_current_solver_loss"] = float(np.mean(current_losses))
        summary["relative_mean_solver_improvement"] = float(
            np.mean(rewards) / max(abs(float(np.mean(previous_losses))), 1.0e-12)
        )
    return summary


def _finite_float_or_none(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    resolved = float(value)
    return resolved if np.isfinite(resolved) else None


def _value_counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _run_policy_update_command(
    *,
    command: str | None,
    round_index: int,
    round_dir: Path,
    reward_table: Path,
    generated_metrics: Path,
    submission: Path | None,
) -> dict[str, Any]:
    if not command:
        return {"enabled": False, "skipped_reason": "no_policy_update_command"}
    env = os.environ.copy()
    env.update(
        {
            "AVF_ROUND_INDEX": str(round_index),
            "AVF_ROUND_DIR": str(round_dir),
            "AVF_REWARD_TABLE": str(reward_table),
            "AVF_GENERATED_METRICS": str(generated_metrics),
            "AVF_SOLVER_SUBMISSION": str(submission or ""),
        }
    )
    logs = round_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "policy_update.stdout.log").open("w") as stdout, (logs / "policy_update.stderr.log").open("w") as stderr:
        proc = subprocess.run(str(command), shell=True, cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr, text=True)
    payload = {"enabled": True, "command": str(command), "returncode": int(proc.returncode)}
    (round_dir / "policy_update.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if proc.returncode != 0:
        raise SystemExit(f"round {round_index} policy update command failed with return code {proc.returncode}")
    return payload


def _round_summary(
    *,
    round_index: int,
    generated_case_count: int,
    round_case_mix: dict[str, int],
    workspace: Path,
    submission: Path | None,
    previous_submission: Path | None,
    generated_eval: dict[str, Any],
    previous_generated_eval: dict[str, Any] | None,
    cumulative_eval: dict[str, Any] | None,
    previous_cumulative_eval: dict[str, Any] | None,
    guard_eval: dict[str, Any] | None,
    hidden_holdout_eval: dict[str, Any] | None,
    hidden_generated_current_eval: dict[str, Any] | None,
    hidden_generated_cumulative_eval: dict[str, Any] | None,
    hidden_all_eval: dict[str, Any] | None,
    hard_summary: dict[str, Any],
    historical_replay_case_count: int,
    solver_feedback_case_count: int,
    cumulative_reward_table: bool,
    policy_update: dict[str, Any],
) -> dict[str, Any]:
    gen_summary = generated_eval.get("summary", {})
    previous_gen_summary = previous_generated_eval.get("summary", {}) if previous_generated_eval else {}
    cumulative_summary = cumulative_eval.get("summary", {}) if cumulative_eval else {}
    previous_cumulative_summary = previous_cumulative_eval.get("summary", {}) if previous_cumulative_eval else {}
    guard_summary = guard_eval.get("summary", {}) if guard_eval else {}
    hidden_holdout_summary = (
        hidden_holdout_eval.get("summary", {}) if hidden_holdout_eval else {}
    )
    hidden_generated_current_summary = (
        hidden_generated_current_eval.get("summary", {})
        if hidden_generated_current_eval else {}
    )
    hidden_generated_cumulative_summary = (
        hidden_generated_cumulative_eval.get("summary", {})
        if hidden_generated_cumulative_eval else {}
    )
    hidden_all_summary = hidden_all_eval.get("summary", {}) if hidden_all_eval else {}
    gen_mean = gen_summary.get("mean_relative_l2")
    guard_mean = guard_summary.get("mean_relative_l2")
    not_easier = None
    if isinstance(gen_mean, (int, float)) and isinstance(guard_mean, (int, float)):
        not_easier = float(gen_mean) >= 0.9 * float(guard_mean)
    return {
        "round": int(round_index),
        "generated_case_count": int(generated_case_count),
        "case_mix": dict(round_case_mix),
        "historical_replay_case_count": int(historical_replay_case_count),
        "solver_feedback_case_count": int(solver_feedback_case_count),
        "cumulative_reward_table": bool(cumulative_reward_table),
        "solver_workspace": str(workspace),
        "submission": str(submission) if submission else None,
        "previous_submission": str(previous_submission) if previous_submission else None,
        "hard_verification_summary": hard_summary,
        "generated_summary": gen_summary,
        "previous_solver_generated_summary": previous_gen_summary or None,
        "solver_improvement_on_generated": _relative_reduction(
            previous_gen_summary.get("mean_relative_l2"),
            gen_summary.get("mean_relative_l2"),
        ),
        "cumulative_generated_summary": cumulative_summary or None,
        "previous_solver_cumulative_generated_summary": previous_cumulative_summary or None,
        "solver_improvement_on_cumulative_generated": _relative_reduction(
            previous_cumulative_summary.get("mean_relative_l2"),
            cumulative_summary.get("mean_relative_l2"),
        ),
        "guard_summary": guard_summary or None,
        "hidden_holdout_summary": hidden_holdout_summary or None,
        "hidden_generated_current_summary": hidden_generated_current_summary or None,
        "hidden_generated_cumulative_summary": hidden_generated_cumulative_summary or None,
        "hidden_all_summary": hidden_all_summary or None,
        "generated_not_substantially_easier_than_guard": not_easier,
        "policy_update": policy_update,
    }


def _relative_reduction(before: Any, after: Any) -> float | None:
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)) or float(before) == 0.0:
        return None
    return (float(before) - float(after)) / abs(float(before))


def _relative_change(current: Any, reference: Any) -> float | None:
    if not isinstance(current, (int, float)) or not isinstance(reference, (int, float)) or float(reference) == 0.0:
        return None
    return (float(current) - float(reference)) / abs(float(reference))


def _final_summary(*, args, out: Path, round_summaries: list[dict[str, Any]], started: float) -> dict[str, Any]:
    rounds_requested = (
        int(args.campaign_rounds_total)
        if int(args.campaign_rounds_total) > 0
        else int(args.rounds)
    )
    return {
        "out": str(out),
        "rounds_requested": rounds_requested,
        "round_index_offset": int(args.round_index_offset),
        "rounds_completed": len(round_summaries),
        "cases_per_round": int(args.cases_per_round),
        "original_cases_per_round": int(args.original_cases_per_round),
        "generated_cases_per_round": int(args.generated_cases_per_round),
        "hidden_generated_cases_per_round": int(args.hidden_generated_cases_per_round),
        "hidden_accumulate": bool(args.hidden_accumulate),
        "solver_backend": str(args.solver_backend),
        "initial_previous_submission": str(args.initial_previous_submission or ""),
        "eval_density": {"nx": int(args.eval_nx), "ny": int(args.eval_ny), "nt": int(args.eval_nt)},
        "policy_update_enabled": bool(args.policy_update_command),
        "include_historical_replay": bool(args.include_historical_replay),
        "cumulative_reward_table": bool(args.cumulative_reward_table),
        "hidden_holdout_cases_root": str(args.hidden_holdout_cases_root or ""),
        "hidden_holdout_cases": int(args.hidden_holdout_cases),
        "hidden_holdout_mean_relative_l2_by_round": [
            (row.get("hidden_holdout_summary") or {}).get("mean_relative_l2")
            for row in round_summaries
        ],
        "hidden_generated_cumulative_mean_relative_l2_by_round": [
            (row.get("hidden_generated_cumulative_summary") or {}).get("mean_relative_l2")
            for row in round_summaries
        ],
        "hidden_all_mean_relative_l2_by_round": [
            (row.get("hidden_all_summary") or {}).get("mean_relative_l2")
            for row in round_summaries
        ],
        "generated_not_substantially_easier_rounds": sum(row.get("generated_not_substantially_easier_than_guard") is True for row in round_summaries),
        "wall_time_seconds": float(time.time() - started),
        "rounds": round_summaries,
    }


def _load_completed_round_summaries(
    *, out: Path, before_round: int
) -> list[dict[str, Any]]:
    """Load the completed prefix used by a later one-round Slurm stage.

    A split campaign is still one scientific run. Requiring every earlier
    round summary prevents a continuation job from silently dropping replay,
    metrics, or memory history from the final campaign result.
    """

    summaries: list[dict[str, Any]] = []
    for round_index in range(max(0, int(before_round))):
        path = out / "rounds" / f"round_{round_index:03d}" / "round_summary.json"
        if not path.is_file():
            raise SystemExit(
                f"cannot continue at round {before_round}: missing {path}"
            )
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or int(payload.get("round", -1)) != round_index:
            raise SystemExit(f"invalid prior round summary: {path}")
        summaries.append(payload)
    return summaries


def _load_guard_dirs(split_manifest: str | None, limit: int) -> list[Path]:
    if not split_manifest:
        return []
    payload = json.loads(Path(split_manifest).read_text())
    rows = payload.get("splits", {}).get("adversarial_guard", [])
    return [Path(row["case_dir"]) for row in rows[: max(0, int(limit))]]


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_mock_solver(path: Path) -> None:
    path.write_text(
        "from __future__ import annotations\n\n"
        "from pathlib import Path\n"
        "import numpy as np\n\n"
        "def _interp_axis(arr, old_axis, new_axis, axis):\n"
        "    old = np.asarray(old_axis, dtype=float).reshape(-1)\n"
        "    new = np.asarray(new_axis, dtype=float).reshape(-1)\n"
        "    if old.size == new.size and np.allclose(old, new):\n"
        "        return np.asarray(arr, dtype=float)\n"
        "    moved = np.moveaxis(np.asarray(arr, dtype=float), axis, 0)\n"
        "    flat = moved.reshape(moved.shape[0], -1)\n"
        "    out = np.empty((new.size, flat.shape[1]), dtype=float)\n"
        "    for j in range(flat.shape[1]):\n"
        "        out[:, j] = np.interp(new, old, flat[:, j])\n"
        "    return np.moveaxis(out.reshape((new.size, *moved.shape[1:])), 0, axis)\n\n"
        "def solve_case(case_dir: str, query_grid=None):\n"
        "    with np.load(Path(case_dir) / 'public_arrays.npz') as data:\n"
        "        initial = np.asarray(data['initial_velocity'], dtype=float)\n"
        "        old_x = np.asarray(data['x'], dtype=float)\n"
        "        old_y = np.asarray(data['y'], dtype=float)\n"
        "        old_t = np.asarray(data['t'], dtype=float)\n"
        "        t = old_t if query_grid is None else np.asarray(query_grid.get('t', old_t), dtype=float)\n"
        "        x = old_x if query_grid is None else np.asarray(query_grid.get('x', old_x), dtype=float)\n"
        "        y = old_y if query_grid is None else np.asarray(query_grid.get('y', old_y), dtype=float)\n"
        "    frame = _interp_axis(initial, old_x, x, axis=1)\n"
        "    frame = _interp_axis(frame, old_y, y, axis=2)\n"
        "    pred = np.repeat(frame[None, ...], int(t.size), axis=0)\n"
        "    return {'velocity': pred, 'metadata': {'strategy': 'repeat_initial_mock'}}\n"
    )


def _workspace_readme() -> str:
    return """# Solver Workspace\n\nPublic PDE cases are in `cases/`. Write the final solver to `submission.py`.\nUse `tools/evaluate_solver.py` to evaluate a candidate solver and get scalar error feedback.\n"""


def _persistent_solver_search_prompt(
    *,
    attempt: int,
    requested_seconds: int,
    consumed_seconds: float,
    champion_score: float | None,
) -> str:
    remaining = max(0.0, float(requested_seconds) - float(consumed_seconds))
    champion = "not established yet" if champion_score is None else f"{champion_score:.12g}"
    return f"""Continue the solver search in this existing workspace; this is persistent-search attempt {attempt}.

The outer controller has been asked to spend a cumulative {requested_seconds} seconds of actual Codex work on this round. It has consumed approximately {consumed_seconds:.1f} seconds and has approximately {remaining:.1f} seconds left. The current retained champion development score is {champion} (lower is better).

Do not merely inspect the current submission and report that it is complete. Use this attempt to test at least one concrete new general hypothesis, numerical method, structural regularity, parameterization, or focused improvement. Read `search_history.jsonl` when present, treat `submission.py` and `search_champion/submission.py` as the retained champion, inspect prior experiment artifacts in the workspace, and use `tools/evaluate_solver.py` for scalar public-development feedback. You may keep a worse experimental candidate in a separate workspace file, but leave the best self-contained candidate you found in `submission.py` before exiting.

The original constraints still apply: no example-ID branches, no per-example lookup tables, no hidden files, and no dependencies on sibling files in the final export. The method must apply unchanged to a new anonymous case. Keep experimenting rather than spending time on narration.
"""

def _solver_prompt(
    *,
    round_index: int,
    eval_density: dict[str, int],
    has_previous: bool,
    solver_cache_guidance: bool = False,
) -> str:
    eval_parts = ["python", "tools/evaluate_solver.py", "--solver", "submission.py", "--cases", "cases"]
    for key, value in eval_density.items():
        if int(value) > 0:
            eval_parts.extend([f"--{key}", str(value)])
    eval_parts.extend(["--out", "eval.json"])
    eval_cmd = " ".join(eval_parts)
    previous = "A reference solver is available at `previous_solver/submission.py`. You may improve it, combine it with new ideas, or replace it entirely." if has_previous else "No reference solver is provided for this round."
    cache_guidance = """

This family has fixed geometry. Avoid rebuilding geometry-only masks, sparse operators, and factorizations for every case. You may define this optional hook:

def prepare_solver(public_cases_root: str) -> None:
    ...

The evaluator calls it once after loading submission.py in each evaluator process. Store reusable setup in module-level variables, then read it from solve_case. The final solver still has to be one self-contained submission.py.
""" if solver_cache_guidance else ""
    return f"""You are writing a high-accuracy solver for a fixed-geometry incompressible Navier-Stokes problem family.

Your goal is not to solve only the visible examples. Your goal is a general solver that should work well on any case with this same public problem interface and geometry class.

Available files:
  cases/
    Anonymous public example cases. Each case contains public arrays, masks, grid, time values, initial velocity, forcing, viscosity, and public metadata.
  tools/evaluate_solver.py
    Evaluation interface for scalar error feedback on the visible examples.
  previous_solver/submission.py
    {previous}
  agent_memory_compact.json
    Distilled cross-round memory from previous solver attempts: useful strategies, negative results, failure modes, and open hypotheses. Treat it as guidance, not as per-case data.
  agent_memory_inbox.jsonl / agent_memory_inbox_schema.json
    Optional place to leave short, general observations for future rounds. Do not use it for per-example parameters or lookup tables.

You may inspect the public files, run experiments, tune numerical methods, train lightweight models, and use available CPU/GPU resources on this server.

Use this evaluator while developing:
  {eval_cmd}

The evaluator returns per-example scalar errors. It does not reveal reference solutions. The visible examples are training/development examples only.

If you discover a durable general insight, you may append one concise JSONL record to `agent_memory_inbox.jsonl` following `agent_memory_inbox_schema.json`.

Hard constraints for the final solver:
  - Do not write per-example parameters, lookup tables, or branches tied to example numbers, filenames, ordering, or manifest position.
  - Do not infer hidden answers from evaluator calls by memorizing the visible examples.
  - Use only general information available in each public case: arrays, masks, grid, time, initial condition, forcing, viscosity, and geometry metadata.
  - The same algorithm should apply to a new anonymous case from this geometry family without code changes.
  - The final export contains only submission.py. Put all required solver logic in that file; do not import sibling workspace files.

Write the final solver to:
  submission.py

`submission.py` must define:

def solve_case(case_dir: str, query_grid: dict | None = None) -> dict:
    ...

Return a dict containing:
  velocity: numpy array with shape [T, 2, X, Y]
  pressure: optional numpy array with shape [T, X, Y]
  metadata: dict
{cache_guidance}

Do not use files outside this workspace. Produce the best general solver you can.
"""


def _round_markdown(summary: dict[str, Any]) -> str:
    gen = summary.get("generated_summary") or {}
    previous_gen = summary.get("previous_solver_generated_summary") or {}
    guard = summary.get("guard_summary") or {}
    hidden_holdout = summary.get("hidden_holdout_summary") or {}
    return "\n".join(
        [
            f"# Round {summary['round']} Summary",
            "",
            f"- generated cases: `{summary['generated_case_count']}`",
            f"- historical replay cases: `{summary.get('historical_replay_case_count')}`",
            f"- solver feedback cases: `{summary.get('solver_feedback_case_count')}`",
            f"- hard verifier accepted: `{(summary.get('hard_verification_summary') or {}).get('accepted')}`",
            f"- generated mean relative L2: `{gen.get('mean_relative_l2')}`",
            f"- previous solver on generated mean relative L2: `{previous_gen.get('mean_relative_l2')}`",
            f"- solver relative improvement on generated: `{summary.get('solver_improvement_on_generated')}`",
            f"- generator relative difficulty change under previous solver: `{summary.get('generator_difficulty_change_under_previous_solver')}`",
            f"- generated median relative L2: `{gen.get('median_relative_l2')}`",
            f"- guard mean relative L2: `{guard.get('mean_relative_l2')}`",
            f"- hidden holdout mean relative L2: `{hidden_holdout.get('mean_relative_l2')}`",
            f"- not substantially easier than guard: `{summary.get('generated_not_substantially_easier_than_guard')}`",
            f"- policy update: `{summary.get('policy_update')}`",
        ]
    ) + "\n"


def _final_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Triplane Agent Adversarial Loop Summary",
            "",
            f"- rounds completed: `{summary['rounds_completed']}` / `{summary['rounds_requested']}`",
            f"- cases per round: `{summary['cases_per_round']}`",
            f"- solver backend: `{summary['solver_backend']}`",
            f"- policy update enabled: `{summary['policy_update_enabled']}`",
            f"- generated-not-easier rounds: `{summary['generated_not_substantially_easier_rounds']}`",
        ]
    ) + "\n"


if __name__ == "__main__":
    main(sys.argv[1:])
