"""Compile the existing structured NS experiment into one product Run."""

from __future__ import annotations

import os

from dataclasses import dataclass
import json
from pathlib import Path
import shlex
from typing import Any, Mapping

from pde_agent_verifier_solver.engine.staged import (
    RunStage,
    StageCommand,
    StageResources,
    StagedRunPlan,
)


PRODUCT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_IMPLEMENTATION_ROOT = PRODUCT_ROOT / "legacy" / "ns"


@dataclass(frozen=True)
class NSCampaignConfig:
    source_path: Path
    name: str
    dataset_manifest: Path
    dataset_format: str
    run_root: Path
    implementation_root: Path
    backend: str
    values: Mapping[str, Any]

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.values.get(name, {})
        if not isinstance(value, Mapping):
            raise ValueError(f"config section {name!r} must be an object")
        return value


def load_ns_campaign_config(
    path: Path | str,
    *,
    dataset_override: Path | str | None = None,
    run_root_override: Path | str | None = None,
    backend_override: str | None = None,
) -> NSCampaignConfig:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("campaign config must contain one JSON object")
    problem = str(payload.get("problem", "")).lower()
    if problem not in {"ns", "navier_stokes", "navier-stokes"}:
        raise ValueError("this config is not a Navier--Stokes campaign")
    name = str(payload.get("name", "")).strip()
    if not name:
        raise ValueError("campaign config needs a nonempty name")
    dataset = _section(payload, "dataset")
    dataset_value = dataset_override or dataset.get("manifest")
    if not dataset_value:
        raise ValueError("provide the initial NS dataset manifest")
    execution = _section(payload, "execution")
    run_root_value = run_root_override or execution.get("run_root")
    if not run_root_value:
        run_root_value = PRODUCT_ROOT / "runs" / _slug(name)
    implementation_value = payload.get("implementation_root")
    implementation_root = (
        _resolve_path(implementation_value, source.parent)
        if implementation_value
        else DEFAULT_IMPLEMENTATION_ROOT
    )
    backend = str(backend_override or execution.get("backend", "slurm"))
    if backend not in {"local", "slurm"}:
        raise ValueError("execution.backend must be 'local' or 'slurm'")
    config = NSCampaignConfig(
        source_path=source,
        name=name,
        dataset_manifest=_resolve_path(dataset_value, source.parent),
        dataset_format=str(dataset.get("format", "ns_d0_v1")),
        run_root=_resolve_path(run_root_value, source.parent),
        implementation_root=implementation_root.resolve(),
        backend=backend,
        values=payload,
    )
    _validate_settings(config)
    return config


def preflight_ns_campaign(config: NSCampaignConfig) -> Mapping[str, object]:
    if config.dataset_format != "ns_d0_v1":
        raise ValueError(
            "the first NS integration accepts dataset format 'ns_d0_v1'; "
            "other numerical and observation adapters will be added as ProblemPrograms"
        )
    manifest_path = config.dataset_manifest
    if not manifest_path.is_file():
        raise FileNotFoundError(f"NS dataset manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest.get("splits", {})
    if not isinstance(splits, Mapping):
        raise ValueError("NS dataset manifest has no splits object")
    projection = config.section("projection")
    hidden = config.section("hidden")
    campaign = config.section("campaign")
    case_mix = _child_section(campaign, "case_mix")
    hidden_cases = _integer(
        hidden, "initial_cases", _integer(hidden, "cases", 128)
    )
    hidden_projection_cases = _integer(projection, "hidden_reference_cases", 0)
    required_counts = {
        "train": _integer(projection, "train_cases", 0),
        "validation": _integer(projection, "validation_cases", 0),
        "d0_distribution_test": max(hidden_projection_cases, hidden_cases),
    }
    original_cases = _integer(case_mix, "original", 0)
    if original_cases:
        original_split = str(case_mix.get("original_split", "validation"))
        required_counts[original_split] = max(
            required_counts.get(original_split, 0),
            original_cases * _integer(campaign, "rounds", 5),
        )
    guard_cases = _integer(campaign, "guard_cases", 0)
    if guard_cases:
        required_counts["adversarial_guard"] = max(
            required_counts.get("adversarial_guard", 0), guard_cases
        )
    split_counts = {}
    for split, requested in required_counts.items():
        rows = splits.get(split)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"NS dataset split {split!r} is missing or empty")
        if requested > 0 and len(rows) < requested:
            raise ValueError(
                f"NS dataset split {split!r} has {len(rows)} cases, but {requested} are requested"
            )
        split_counts[split] = len(rows)
        _require_case_files(
            rows[0],
            manifest_path=manifest_path,
            implementation_root=config.implementation_root,
        )
    missing_implementation = [
        str(path)
        for path in _required_implementation_files(config)
        if not path.is_file()
    ]
    if missing_implementation:
        raise FileNotFoundError(
            "the bundled NS implementation is incomplete: " + ", ".join(missing_implementation)
        )
    prompt_file = _agent_prompt(config)
    if prompt_file is not None and not prompt_file.is_file():
        raise FileNotFoundError(f"NS Agent prompt does not exist: {prompt_file}")
    return {
        "dataset": str(manifest_path),
        "dataset_format": config.dataset_format,
        "split_counts": split_counts,
        "implementation_root": str(config.implementation_root),
        "run_root": str(config.run_root),
        "backend": config.backend,
    }


class NSCampaignCompiler:
    """Compile the historical NS staged lifecycle for explicit legacy runs."""

    def compile(
        self,
        config: NSCampaignConfig,
        *,
        legacy_staged_intent: bool = False,
    ) -> StagedRunPlan:
        if legacy_staged_intent is not True:
            raise PermissionError(
                "NSCampaignCompiler is a history-only staged compiler; pass "
                "legacy_staged_intent=True for an explicit legacy/control-arm run"
            )
        paths = _RunPaths(config.run_root)
        projection = config.section("projection")
        pca = config.section("pca")
        flow = config.section("flow")
        generator = config.section("generator")
        hidden = config.section("hidden")
        campaign = config.section("campaign")
        perturb = config.section("perturbations")
        guidance = config.section("guidance")
        agent = config.section("agent")
        evaluator = config.section("evaluator")

        rounds = _integer(campaign, "rounds", 5)
        cases_per_round = _integer(campaign, "cases_per_round", 1024)
        case_mix = _child_section(campaign, "case_mix")
        pca_tangent = _integer(perturb, "pca_tangent", 48)
        spectral = _integer(perturb, "spectral_complement", 96)
        pca_normal = _integer(perturb, "pca_normal", 48)
        perturbation_cases = pca_tangent + spectral + pca_normal
        original_cases = _integer(case_mix, "original", 0)
        generated_cases = _integer(
            case_mix,
            "generated",
            cases_per_round - original_cases - perturbation_cases,
        )
        configured_perturbed = _integer(
            case_mix, "perturbed", perturbation_cases
        )
        hidden_cases = _integer(
            hidden, "initial_cases", _integer(hidden, "cases", 128)
        )
        hidden_cases_per_round = _integer(hidden, "cases_per_round", 0)
        flow_steps = _integer(flow, "steps", 30_000)
        checkpoint_every = _integer(flow, "checkpoint_every", 5000)
        flow_candidate_steps = _checkpoint_steps(flow_steps, checkpoint_every)
        flow_checkpoint = paths.selected_flow_checkpoint
        pca_checkpoint = paths.pca / "structured_coefficient_pca.pt"

        project_common = (
            "--split-manifest", str(config.dataset_manifest),
            "--batch-cases", str(_integer(projection, "batch_cases", 32)),
            "--shard-cases", str(_integer(projection, "shard_cases", 256)),
            "--loader-workers", str(_integer(projection, "loader_workers", 8)),
            "--spatial-mode-x", str(_integer(projection, "spatial_mode_x", 16)),
            "--spatial-mode-y", str(_integer(projection, "spatial_mode_y", 16)),
            "--time-order", str(_integer(projection, "time_order", 12)),
            "--geometry-mask-transition", str(_number(projection, "geometry_mask_transition", 0.05)),
            "--ridge-relative", str(_number(projection, "ridge_relative", 1.0e-6)),
            "--device", str(projection.get("device", "cuda")),
        )
        project_script = _script(config, "project_d0_to_structured_coefficients.py")
        project_train = StageCommand(
            ("python", str(project_script), *project_common, "--split", "train", "--out", str(paths.train_coefficients), "--max-cases", str(_integer(projection, "train_cases", 0))),
            "Project training fields",
        )
        project_validation = StageCommand(
            ("python", str(project_script), *project_common, "--split", "validation", "--out", str(paths.validation_coefficients), "--max-cases", str(_integer(projection, "validation_cases", 0))),
            "Project validation fields",
        )
        project_hidden_reference = StageCommand(
            (
                "python", str(project_script), *project_common,
                "--split", "d0_distribution_test",
                "--out", str(paths.hidden_reference_coefficients),
                "--max-cases", str(_integer(projection, "hidden_reference_cases", 0)),
            ),
            "Project held-out reference fields",
        )

        pca_command = StageCommand(
            (
                "python", str(_script(config, "audit_ns_structured_coefficient_pca.py")),
                "--train-manifest", str(paths.train_manifest),
                "--validation-manifest", str(paths.validation_manifest),
                "--out", str(paths.pca),
                "--ranks", _ranks(pca.get("ranks", [64, 128, 256])),
                "--train-cases", str(_integer(pca, "train_cases", 0)),
                "--validation-cases", str(_integer(pca, "validation_cases", 2048)),
                "--batch-cases", str(_integer(pca, "batch_cases", 16)),
                "--loader-workers", str(_integer(pca, "loader_workers", 8)),
                "--oversample", str(_integer(pca, "oversample", 16)),
                "--power-iterations", str(_integer(pca, "power_iterations", 4)),
                "--seed", str(_integer(pca, "seed", 20260809)),
                "--device", str(pca.get("device", "cuda")),
            ),
            "Fit PCA",
        )
        pca_selection = _child_section(pca, "selection")
        select_pca_command = StageCommand(
            (
                "python", "-m",
                "pde_agent_verifier_solver.plugins.ns.selection", "pca",
                "--audit", str(paths.pca / "pca_audit.json"),
                "--out", str(paths.pca_selection),
                "--metric", str(
                    pca_selection.get("metric", "manifold_velocity_relative_l2")
                ),
                "--statistic", str(pca_selection.get("statistic", "median")),
                "--relative-tolerance", str(
                    _number(pca_selection, "relative_tolerance", 0.05)
                ),
            ),
            "Select PCA rank from physical validation",
        )
        flow_command = StageCommand(
            (
                "python", str(_script(config, "train_ns_structured_coefficient_flow.py")),
                "--train-manifest", str(paths.train_manifest),
                "--validation-manifest", str(paths.validation_manifest),
                "--pca-checkpoint", str(pca_checkpoint),
                "--out", str(paths.flow),
                "--rank-selection", str(paths.pca_selection),
                "--steps", str(flow_steps),
                "--batch-size", str(_integer(flow, "batch_size", 512)),
                "--validation-cases", str(_integer(flow, "validation_cases", 1024)),
                "--width", str(_integer(flow, "width", 768)),
                "--depth", str(_integer(flow, "depth", 8)),
                "--time-width", str(_integer(flow, "time_width", 128)),
                "--lr", str(_number(flow, "lr", 2.0e-4)),
                "--minimum-lr-fraction", str(_number(flow, "minimum_lr_fraction", 0.1)),
                "--warmup-steps", str(_integer(flow, "warmup_steps", 1000)),
                "--weight-decay", str(_number(flow, "weight_decay", 1.0e-4)),
                "--ema-decay", str(_number(flow, "ema_decay", 0.999)),
                "--max-grad-norm", str(_number(flow, "max_grad_norm", 1.0)),
                "--log-every", str(_integer(flow, "log_every", 100)),
                "--validate-every", str(_integer(flow, "validate_every", 1000)),
                "--checkpoint-every", str(_integer(flow, "checkpoint_every", 5000)),
                "--seed", str(_integer(flow, "seed", 20260809)),
                "--amp", str(flow.get("amp", "bfloat16")),
                "--device", str(flow.get("device", "cuda")),
            ),
            "Train Base Rectified Flow",
        )

        # Every saved checkpoint is tested with the same validation sampling
        # cohort.  The selector below consumes these comparable reports.
        flow_selection = _child_section(flow, "selection")
        selection_samples = _integer(flow_selection, "sample_cases", 512)
        selection_materialized = _integer(
            flow_selection, "materialized_cases", 64
        )
        selection_reference = _integer(flow_selection, "reference_cases", 128)
        baseline_repeats = _integer(flow_selection, "baseline_repeats", 32)
        flow_candidate_commands: list[StageCommand] = []
        for step in flow_candidate_steps:
            candidate_root = paths.flow_candidates / f"step_{step:07d}"
            candidate_checkpoint = (
                paths.flow / "checkpoints" / f"flow_step_{step:07d}.pt"
            )
            flow_candidate_commands.extend(
                (
                    StageCommand(
                        (
                            "python", str(_script(config, "sample_ns_structured_coefficient_flow.py")),
                            "--flow-checkpoint", str(candidate_checkpoint),
                            "--pca-checkpoint", str(pca_checkpoint),
                            "--train-manifest", str(paths.train_manifest),
                            "--validation-manifest", str(paths.validation_manifest),
                            "--out", str(candidate_root / "sample"),
                            "--samples", str(selection_samples),
                            "--sample-batch", str(_integer(generator, "sample_batch", 256)),
                            "--integration-steps", str(_integer(generator, "integration_steps", 256)),
                            "--train-audit-cases", str(_integer(generator, "train_audit_cases", 80_000)),
                            "--validation-audit-cases", str(_integer(generator, "validation_audit_cases", 2048)),
                            "--metric-rank", str(_integer(generator, "metric_rank", 64)),
                            "--baseline-repeats", str(baseline_repeats),
                            "--paired-base-audit-cases", "0",
                            "--seed", str(_integer(flow_selection, "seed", 20260812)),
                            "--state", str(generator.get("flow_state", "raw")),
                            "--device", str(generator.get("device", "cuda")),
                        ),
                        f"Audit Flow checkpoint {step}",
                    ),
                    StageCommand(
                        (
                            "python", str(_script(config, "materialize_ns_structured_flow_samples.py")),
                            "--sample-audit", str(candidate_root / "sample" / "sample_audit.json"),
                            "--reference-manifest", str(paths.validation_manifest),
                            "--out", str(candidate_root / "materialized"),
                            "--generated-cases", str(selection_materialized),
                            "--reference-cases", str(selection_reference),
                            "--case-id-prefix", f"ns_flow_validation_{step:07d}",
                            "--selection-seed", str(_integer(flow_selection, "selection_seed", 20260813)),
                            "--baseline-repeats", str(baseline_repeats),
                            "--source-batch-points", str(_integer(generator, "source_batch_points", 8192)),
                            "--device", str(generator.get("device", "cuda")),
                        ),
                        f"Check physical samples from checkpoint {step}",
                    ),
                )
            )
        select_flow_argv: list[str] = [
            "python", "-m", "pde_agent_verifier_solver.plugins.ns.selection",
            "flow",
            "--flow-root", str(paths.flow),
            "--candidate-root", str(paths.flow_candidates),
            "--steps", ":".join(str(step) for step in flow_candidate_steps),
            "--out", str(paths.flow_selection),
            "--selected-checkpoint", str(paths.selected_flow_checkpoint),
            "--minimum-contract-pass-fraction", str(
                _number(flow_selection, "minimum_contract_pass_fraction", 1.0)
            ),
            "--selection-rule", str(
                flow_selection.get("rule", "empirical_d0_gate")
            ),
            "--baseline-lower-quantile", str(
                _number(flow_selection, "baseline_lower_quantile", 0.05)
            ),
            "--baseline-upper-quantile", str(
                _number(flow_selection, "baseline_upper_quantile", 0.95)
            ),
        ]
        weights = flow_selection.get("weights", {})
        if not isinstance(weights, Mapping):
            raise ValueError("flow.selection.weights must be an object")
        for name, value in weights.items():
            select_flow_argv.extend(("--weight", f"{name}={float(value)}"))
        select_flow_command = StageCommand(
            tuple(select_flow_argv),
            "Select the validated Base Generator checkpoint",
        )

        sample_hidden = StageCommand(
            (
                "python", str(_script(config, "sample_ns_structured_coefficient_flow.py")),
                "--flow-checkpoint", str(flow_checkpoint),
                "--pca-checkpoint", str(pca_checkpoint),
                "--train-manifest", str(paths.train_manifest),
                "--validation-manifest", str(paths.validation_manifest),
                "--out", str(paths.hidden_sample),
                "--samples", str(_integer(hidden, "sample_cases", 2048)),
                "--sample-batch", str(_integer(generator, "sample_batch", 256)),
                "--integration-steps", str(_integer(generator, "integration_steps", 256)),
                "--train-audit-cases", str(_integer(generator, "train_audit_cases", 80_000)),
                "--validation-audit-cases", str(_integer(generator, "validation_audit_cases", 2048)),
                "--metric-rank", str(_integer(generator, "metric_rank", 64)),
                "--paired-base-audit-cases", "0",
                "--seed", str(_integer(hidden, "seed", 20260810)),
                "--state", str(generator.get("flow_state", "raw")),
                "--device", str(generator.get("device", "cuda")),
            ),
            "Sample hidden cohort",
        )
        materialize_hidden = StageCommand(
            (
                "python", str(_script(config, "materialize_ns_structured_flow_samples.py")),
                "--sample-audit", str(paths.hidden_sample / "sample_audit.json"),
                "--reference-manifest", str(paths.hidden_reference_manifest),
                "--out", str(paths.hidden_materialization),
                "--generated-cases", str(hidden_cases),
                "--reference-cases", str(hidden_cases),
                "--case-id-prefix", "ns_hidden_generated",
                "--selection-seed", str(_integer(hidden, "selection_seed", 20260811)),
                "--source-batch-points", str(_integer(generator, "source_batch_points", 8192)),
                "--device", str(generator.get("device", "cuda")),
            ),
            "Materialize hidden truth",
        )
        if configured_perturbed != perturbation_cases:
            raise ValueError(
                "campaign.case_mix.perturbed must equal the perturbation family total"
            )

        generate_command = self._round_generator_command(
            config=config,
            paths=paths,
            flow_checkpoint=flow_checkpoint,
            pca_checkpoint=pca_checkpoint,
            unperturbed_cases=generated_cases,
            pca_tangent=pca_tangent,
            spectral=spectral,
            pca_normal=pca_normal,
        )
        hidden_generate_command = self._round_hidden_generator_command(
            config=config,
            paths=paths,
            flow_checkpoint=flow_checkpoint,
            pca_checkpoint=pca_checkpoint,
        )
        update_command = self._guidance_command(
            config=config,
            flow_checkpoint=flow_checkpoint,
            pca_checkpoint=pca_checkpoint,
        )
        agent_budget = _integer(agent, "budget_seconds", 3600)
        agent_mode = str(agent.get("execution_mode", "single"))
        shared_loop_arguments = [
            "python", str(_script(config, "run_triplane_agent_adversarial_loop.py")),
            "--generate-command", shlex.join(generate_command),
            "--out", str(paths.loop),
            "--cases-per-round", str(cases_per_round),
            "--generated-cases-per-round", str(generated_cases + perturbation_cases),
            "--original-split-manifest", str(config.dataset_manifest),
            "--original-split", str(case_mix.get("original_split", "validation")),
            "--original-cases-per-round", str(original_cases),
            "--original-selection-seed", str(_integer(case_mix, "original_seed", 20260814)),
            "--guard-cases", str(_integer(campaign, "guard_cases", 0)),
            "--hidden-holdout-cases-root", str(paths.hidden_reference_cases),
            "--hidden-holdout-cases", str(hidden_cases),
            "--hidden-generate-command", shlex.join(hidden_generate_command),
            "--hidden-generated-cases-per-round", str(hidden_cases_per_round),
            "--solver-backend", str(agent.get("backend", "codex")),
            "--codex-timeout-seconds", str(
                _integer(agent, "invocation_timeout_seconds", agent_budget)
            ),
            "--codex-max-attempts", str(_integer(agent, "max_attempts", 1)),
            "--policy-reward-objective", str(guidance.get("objective", "solver_error")),
            "--formal-evaluation-scope", str(evaluator.get("formal_scope", "cumulative")),
            "--eval-workers", str(_integer(evaluator, "workers", 1)),
            "--policy-update-command", shlex.join(update_command),
            "--require-continuous-mms-cases",
            "--include-historical-replay",
            "--cumulative-reward-table",
        ]
        if _boolean(hidden, "accumulate", True):
            shared_loop_arguments.append("--hidden-accumulate")
        if agent_mode == "persistent":
            shared_loop_arguments.extend(
                (
                    "--codex-persistent-search-seconds", str(agent_budget),
                    "--codex-persistent-search-max-attempts", str(
                        _integer(agent, "persistent_max_attempts", 0)
                    ),
                )
            )
        prompt_file = _agent_prompt(config)
        if prompt_file is not None:
            shared_loop_arguments.extend(("--agent-prompt-file", str(prompt_file)))
        if _integer(campaign, "guard_cases", 0) > 0:
            shared_loop_arguments.extend(("--split-manifest", str(config.dataset_manifest)))
        model = str(agent.get("model", "gpt-5.6-sol"))
        reasoning = str(agent.get("reasoning_effort", "max"))
        if model:
            shared_loop_arguments.extend(("--codex-model", model))
        if reasoning:
            shared_loop_arguments.extend(("--codex-reasoning-effort", reasoning))
        if _boolean(agent, "allow_network", True):
            shared_loop_arguments.append("--codex-allow-network")
        if _boolean(agent, "cache_fixed_setup", False):
            shared_loop_arguments.append("--solver-cache-guidance")
        for axis in ("nx", "ny", "nt"):
            value = _integer(evaluator, axis, 0)
            if value:
                shared_loop_arguments.extend((f"--eval-{axis}", str(value)))

        # One scientific round per scheduler stage keeps a long formal run
        # inside the cluster's per-job wall-time limit. Earlier round outputs
        # are the continuation state, so an interrupted campaign resumes by
        # resubmitting only the first incomplete stage.
        adversarial_stages: list[RunStage] = []
        for round_index in range(rounds):
            stage_id = f"adversarial_round_{round_index:03d}"
            previous_stage_id = f"adversarial_round_{round_index - 1:03d}"
            previous_solver = (
                _initial_solver(config)
                if round_index == 0
                else paths.loop
                / "rounds"
                / f"round_{round_index - 1:03d}"
                / "submission.py"
            )
            round_arguments = list(shared_loop_arguments)
            round_arguments.extend(
                (
                    "--rounds", "1",
                    "--round-index-offset", str(round_index),
                    "--campaign-rounds-total", str(rounds),
                    "--initial-previous-submission", str(previous_solver),
                )
            )
            round_root = paths.loop / "rounds" / f"round_{round_index:03d}"
            expected_outputs = (
                round_root / "round_summary.json",
                round_root / "submission.py",
            )
            if round_index == rounds - 1:
                expected_outputs += (paths.loop / "final_summary.json",)
            adversarial_stages.append(
                RunStage(
                    stage_id,
                    f"Run Generator--Agent--Verifier round {round_index + 1} of {rounds}",
                    (
                        StageCommand(
                            tuple(round_arguments),
                            f"Run adversarial round {round_index + 1} of {rounds}",
                        ),
                    ),
                    depends_on=(
                        ("prepare_initial_hidden",)
                        if round_index == 0
                        else (previous_stage_id,)
                    ),
                    resources=_resources(config, "adversarial_rounds", gpu=True),
                    expected_outputs=expected_outputs,
                )
            )
        final_adversarial_stage_id = adversarial_stages[-1].stage_id

        base_report = StageCommand(
            (
                "python", str(_script(config, "summarize_ns_structured_guided_5round.py")),
                "--loop-root", str(paths.loop), "--out", str(paths.report),
                "--expected-rounds", str(rounds),
                "--expected-cases-per-round", str(cases_per_round),
                "--expected-original-per-round", str(original_cases),
            ),
            "Summarize solver evolution",
        )
        perturbation_report = StageCommand(
            (
                "python", str(_script(config, "summarize_ns_structured_guided_perturbed_5round.py")),
                "--loop-root", str(paths.loop), "--out", str(paths.report),
                "--expected-rounds", str(rounds),
                "--expected-cases-per-round", str(cases_per_round),
                "--expected-original-per-round", str(original_cases),
                "--expected-unperturbed-per-round", str(generated_cases),
                "--expected-perturbations-per-round", str(perturbation_cases),
                "--expected-replay-perturbations", str(_integer(perturb, "replay", 96)),
                "--expected-pca-tangent", str(pca_tangent),
                "--expected-spectral-complement", str(spectral),
                "--expected-pca-normal", str(pca_normal),
            ),
            "Summarize perturbation results",
        )
        publish = StageCommand(
            (
                "python", "-m", "pde_agent_verifier_solver.plugins.ns.publish",
                "--run-root", str(config.run_root),
                "--loop-root", str(paths.loop),
                "--report-root", str(paths.report),
                "--rounds", str(rounds),
            ),
            "Publish final solver",
        )

        stages = (
            RunStage(
                "project_train", "Map training data to continuous coefficients",
                (project_train,), resources=_resources(config, "project_train", gpu=True),
                expected_outputs=(paths.train_manifest,),
            ),
            RunStage(
                "project_validation", "Map validation data to continuous coefficients",
                (project_validation,), resources=_resources(config, "project_validation", gpu=True),
                expected_outputs=(paths.validation_manifest,),
            ),
            RunStage(
                "project_hidden_reference", "Map held-out data to reference coefficients",
                (project_hidden_reference,),
                resources=_resources(config, "project_hidden_reference", gpu=True),
                expected_outputs=(paths.hidden_reference_manifest,),
            ),
            RunStage(
                "fit_pca", "Fit the coefficient basis",
                (pca_command,), depends_on=("project_train", "project_validation"),
                resources=_resources(config, "fit_pca", gpu=True),
                expected_outputs=(pca_checkpoint,),
            ),
            RunStage(
                "select_pca", "Choose the physical PCA rank",
                (select_pca_command,), depends_on=("fit_pca",),
                resources=_resources(config, "select_pca", gpu=False),
                expected_outputs=(paths.pca_selection,),
            ),
            RunStage(
                "train_generator", "Train the Base Generator",
                (flow_command,), depends_on=("select_pca",),
                resources=_resources(config, "train_generator", gpu=True),
                expected_outputs=(
                    paths.flow / "checkpoints" / f"flow_step_{flow_steps:07d}.pt",
                    paths.flow / "train_report.json",
                ),
            ),
            RunStage(
                "audit_generator", "Compare Generator checkpoints on validation data",
                tuple(flow_candidate_commands), depends_on=("train_generator",),
                resources=_resources(config, "audit_generator", gpu=True),
                expected_outputs=tuple(
                    paths.flow_candidates
                    / f"step_{step:07d}"
                    / "materialized"
                    / "answer_first_report.json"
                    for step in flow_candidate_steps
                ),
            ),
            RunStage(
                "select_generator", "Choose the validated Base Generator",
                (select_flow_command,), depends_on=("audit_generator",),
                resources=_resources(config, "select_generator", gpu=False),
                expected_outputs=(paths.flow_selection, flow_checkpoint),
            ),
            RunStage(
                "prepare_initial_hidden", "Prepare the fixed reference hidden suite",
                (sample_hidden, materialize_hidden),
                depends_on=("select_generator", "project_hidden_reference"),
                resources=_resources(config, "prepare_initial_hidden", gpu=True),
                expected_outputs=(paths.hidden_reference_cases,),
            ),
        ) + tuple(adversarial_stages) + (
            RunStage(
                "publish_result", "Build the final solver result",
                (base_report, perturbation_report, publish),
                depends_on=(final_adversarial_stage_id,),
                resources=_resources(config, "publish_result", gpu=True),
                expected_outputs=(paths.result_json, paths.final_solver),
            ),
        )
        execution = config.section("execution")
        modules = execution.get("modules", ["scipy-stack", "hdf5", "pytest"])
        if not isinstance(modules, list) or not all(isinstance(item, str) for item in modules):
            raise ValueError("execution.modules must be a list of module names")
        return StagedRunPlan(
            name=config.name,
            problem="ns",
            task=str(config.section("task").get("kind", "solver")),
            run_root=config.run_root,
            workdir=config.implementation_root,
            product_root=PRODUCT_ROOT,
            implementation_root=config.implementation_root,
            stages=stages,
            modules=tuple(modules),
            account=str(execution.get("account", os.environ.get("PDE_GAV_SLURM_ACCOUNT", ""))),
            qos=str(execution.get("qos", "normal")),
            result_path=paths.result_json,
        )

    @staticmethod
    def _round_generator_command(
        *,
        config: NSCampaignConfig,
        paths: "_RunPaths",
        flow_checkpoint: Path,
        pca_checkpoint: Path,
        unperturbed_cases: int,
        pca_tangent: int,
        spectral: int,
        pca_normal: int,
    ) -> tuple[str, ...]:
        generator = config.section("generator")
        perturb = config.section("perturbations")
        case_mix = _child_section(config.section("campaign"), "case_mix")
        return (
            "python", str(_script(config, "generate_ns_structured_guided_perturbed_round.py")),
            "--flow-checkpoint", str(flow_checkpoint),
            "--pca-checkpoint", str(pca_checkpoint),
            "--train-manifest", str(paths.train_manifest),
            "--validation-manifest", str(paths.validation_manifest),
            "--reference-manifest", str(paths.validation_manifest),
            "--integration-steps", str(_integer(generator, "integration_steps", 256)),
            "--sample-batch", str(_integer(generator, "sample_batch", 256)),
            "--train-audit-cases", str(_integer(generator, "train_audit_cases", 80_000)),
            "--validation-audit-cases", str(_integer(generator, "validation_audit_cases", 2048)),
            "--metric-rank", str(_integer(generator, "metric_rank", 64)),
            "--paired-base-audit-cases", str(_integer(generator, "paired_base_audit_cases", 128)),
            "--guidance-strength", str(_number(config.section("guidance"), "strength", 0.10)),
            "--guidance-max-flow-ratio", str(_number(config.section("guidance"), "max_flow_ratio", 0.25)),
            "--guided-fraction", str(_number(case_mix, "guided_fraction", 0.5)),
            "--source-batch-points", str(_integer(generator, "source_batch_points", 8192)),
            "--seed", str(_integer(generator, "seed", 20260810)),
            "--selection-seed", str(_integer(perturb, "selection_seed", 2026081601)),
            "--perturbation-seed", str(_integer(perturb, "seed", 2026081602)),
            "--unperturbed-cases", str(unperturbed_cases),
            "--perturbation-cases", str(pca_tangent + spectral + pca_normal),
            "--replay-perturbation-cases", str(_integer(perturb, "replay", 96)),
            "--pca-tangent-cases", str(pca_tangent),
            "--spectral-complement-cases", str(spectral),
            "--pca-normal-cases", str(pca_normal),
            "--target-question-norm", str(_number(perturb, "target_question_norm", 0.01)),
            "--reference-velocity-scale", str(_number(perturb, "reference_velocity_scale", 0.005531590859331513)),
            "--reference-forcing-scale", str(_number(perturb, "reference_forcing_scale", 0.39948812205030987)),
            "--spectral-max-mode", str(_integer(perturb, "spectral_max_mode", 20)),
            "--cuda-cache-clear-every", str(
                _integer(generator, "cuda_cache_clear_every", 1)
            ),
            "--device", str(generator.get("device", "cuda")),
        )

    @staticmethod
    def _round_hidden_generator_command(
        *,
        config: NSCampaignConfig,
        paths: "_RunPaths",
        flow_checkpoint: Path,
        pca_checkpoint: Path,
    ) -> tuple[str, ...]:
        generator = config.section("generator")
        hidden = config.section("hidden")
        guidance = config.section("guidance")
        return (
            "python", str(_script(config, "generate_ns_structured_guided_round.py")),
            "--flow-checkpoint", str(flow_checkpoint),
            "--pca-checkpoint", str(pca_checkpoint),
            "--train-manifest", str(paths.train_manifest),
            "--validation-manifest", str(paths.validation_manifest),
            "--reference-manifest", str(paths.validation_manifest),
            "--integration-steps", str(_integer(generator, "integration_steps", 256)),
            "--sample-batch", str(_integer(generator, "sample_batch", 256)),
            "--train-audit-cases", str(_integer(generator, "train_audit_cases", 80_000)),
            "--validation-audit-cases", str(_integer(generator, "validation_audit_cases", 2048)),
            "--metric-rank", str(_integer(generator, "metric_rank", 64)),
            "--paired-base-audit-cases", str(_integer(generator, "paired_base_audit_cases", 128)),
            "--guidance-strength", str(_number(guidance, "strength", 0.10)),
            "--guidance-max-flow-ratio", str(_number(guidance, "max_flow_ratio", 0.25)),
            "--guided-fraction", str(_number(hidden, "guided_fraction", 0.5)),
            "--source-batch-points", str(_integer(generator, "source_batch_points", 8192)),
            "--seed", str(_integer(hidden, "round_seed", 20260815)),
            "--device", str(generator.get("device", "cuda")),
        )

    @staticmethod
    def _guidance_command(
        *, config: NSCampaignConfig, flow_checkpoint: Path, pca_checkpoint: Path
    ) -> tuple[str, ...]:
        guidance = config.section("guidance")
        return (
            "python", str(_script(config, "update_ns_structured_guidance_from_mixed_round.py")),
            "--flow-checkpoint", str(flow_checkpoint),
            "--pca-checkpoint", str(pca_checkpoint),
            "--steps", str(_integer(guidance, "steps", 5000)),
            "--batch-size", str(_integer(guidance, "batch_size", 256)),
            "--width", str(_integer(guidance, "width", 512)),
            "--depth", str(_integer(guidance, "depth", 6)),
            "--time-width", str(_integer(guidance, "time_width", 128)),
            "--minimum-cases", str(_integer(guidance, "minimum_cases", 32)),
            "--lr", str(_number(guidance, "lr", 2.0e-4)),
            "--seed", str(_integer(guidance, "seed", 20260810)),
            "--device", str(guidance.get("device", "cuda")),
        )


@dataclass(frozen=True)
class _RunPaths:
    root: Path

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def train_coefficients(self) -> Path:
        return self.artifacts / "coefficients" / "train"

    @property
    def validation_coefficients(self) -> Path:
        return self.artifacts / "coefficients" / "validation"

    @property
    def hidden_reference_coefficients(self) -> Path:
        return self.artifacts / "coefficients" / "hidden_reference"

    @property
    def train_manifest(self) -> Path:
        return self.train_coefficients / "coefficient_manifest.json"

    @property
    def validation_manifest(self) -> Path:
        return self.validation_coefficients / "coefficient_manifest.json"

    @property
    def hidden_reference_manifest(self) -> Path:
        return self.hidden_reference_coefficients / "coefficient_manifest.json"

    @property
    def pca(self) -> Path:
        return self.artifacts / "pca"

    @property
    def pca_selection(self) -> Path:
        return self.pca / "pca_selection.json"

    @property
    def flow(self) -> Path:
        return self.artifacts / "base_generator"

    @property
    def flow_candidates(self) -> Path:
        return self.artifacts / "generator_validation"

    @property
    def flow_selection(self) -> Path:
        return self.flow / "flow_selection.json"

    @property
    def selected_flow_checkpoint(self) -> Path:
        return self.flow / "selected_flow.pt"

    @property
    def hidden_sample(self) -> Path:
        return self.artifacts / "hidden_suite" / "sample"

    @property
    def hidden_materialization(self) -> Path:
        return self.artifacts / "hidden_suite" / "materialized"

    @property
    def hidden_reference_cases(self) -> Path:
        return self.hidden_materialization / "reference_pca_cases"

    @property
    def loop(self) -> Path:
        return self.root / "campaign" / "agent_adversarial_loop"

    @property
    def report(self) -> Path:
        return self.root / "result" / "report"

    @property
    def final_solver(self) -> Path:
        return self.root / "result" / "solver" / "submission.py"

    @property
    def result_json(self) -> Path:
        return self.root / "result" / "result.json"


def _validate_settings(config: NSCampaignConfig) -> None:
    projection = config.section("projection")
    pca = config.section("pca")
    flow = config.section("flow")
    hidden = config.section("hidden")
    campaign = config.section("campaign")
    perturb = config.section("perturbations")
    guidance = config.section("guidance")
    agent = config.section("agent")
    generator = config.section("generator")
    evaluator = config.section("evaluator")
    flow_selection = _child_section(flow, "selection")
    case_mix = _child_section(campaign, "case_mix")
    positive = {
        "projection.batch_cases": _integer(projection, "batch_cases", 32),
        "pca maximum rank": max(_rank_values(pca.get("ranks", [64, 128, 256]))),
        "flow.steps": _integer(flow, "steps", 30_000),
        "flow.selection.sample_cases": _integer(flow_selection, "sample_cases", 512),
        "flow.selection.materialized_cases": _integer(
            flow_selection, "materialized_cases", 64
        ),
        "flow.selection.reference_cases": _integer(
            flow_selection, "reference_cases", 128
        ),
        "flow.selection.baseline_repeats": _integer(
            flow_selection, "baseline_repeats", 32
        ),
        "hidden.initial_cases": _integer(
            hidden, "initial_cases", _integer(hidden, "cases", 128)
        ),
        "hidden.sample_cases": _integer(hidden, "sample_cases", 2048),
        "campaign.rounds": _integer(campaign, "rounds", 5),
        "campaign.cases_per_round": _integer(campaign, "cases_per_round", 1024),
        "guidance.steps": _integer(guidance, "steps", 5000),
        "guidance.minimum_cases": _integer(guidance, "minimum_cases", 32),
        "agent.budget_seconds": _integer(agent, "budget_seconds", 3600),
        "evaluator.workers": _integer(evaluator, "workers", 1),
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"NS settings must be positive: {invalid}")
    if _integer(generator, "cuda_cache_clear_every", 1) < 0:
        raise ValueError("generator.cuda_cache_clear_every cannot be negative")
    if _integer(hidden, "cases_per_round", 0) < 0:
        raise ValueError("hidden.cases_per_round cannot be negative")
    if str(agent.get("execution_mode", "single")) not in {"single", "persistent"}:
        raise ValueError("agent.execution_mode must be 'single' or 'persistent'")
    for label, fraction in (
        ("campaign.case_mix.guided_fraction", _number(case_mix, "guided_fraction", 0.5)),
        ("hidden.guided_fraction", _number(hidden, "guided_fraction", 0.5)),
    ):
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"{label} must lie in [0, 1]")
    hidden_cases = _integer(
        hidden, "initial_cases", _integer(hidden, "cases", 128)
    )
    if _integer(hidden, "sample_cases", 2048) < hidden_cases:
        raise ValueError("hidden.sample_cases must be at least hidden.initial_cases")
    hidden_projection_cases = _integer(projection, "hidden_reference_cases", 0)
    if 0 < hidden_projection_cases < hidden_cases:
        raise ValueError(
            "projection.hidden_reference_cases must cover every hidden case"
        )
    families = [
        _integer(perturb, "pca_tangent", 48),
        _integer(perturb, "spectral_complement", 96),
        _integer(perturb, "pca_normal", 48),
    ]
    if min(families) <= 0:
        raise ValueError("all three NS perturbation family counts must be positive")
    total = sum(families)
    cases = _integer(campaign, "cases_per_round", 1024)
    original = _integer(case_mix, "original", 0)
    generated = _integer(case_mix, "generated", cases - original - total)
    perturbed = _integer(case_mix, "perturbed", total)
    if min(original, generated, perturbed) < 0:
        raise ValueError("campaign.case_mix counts cannot be negative")
    if generated <= 0:
        raise ValueError("campaign.case_mix.generated must be positive")
    if original > 0 and not str(
        case_mix.get("original_split", "validation")
    ).strip():
        raise ValueError("campaign.case_mix.original_split cannot be empty")
    if original + generated + perturbed != cases:
        raise ValueError("campaign.case_mix counts must sum to cases_per_round")
    if perturbed != total:
        raise ValueError(
            "campaign.case_mix.perturbed must equal perturbation family counts"
        )
    replay = _integer(perturb, "replay", 96)
    if total >= cases:
        raise ValueError("perturbations must leave at least one unperturbed case per round")
    if total > generated:
        raise ValueError(
            "generated cases must provide enough first-round perturbation anchors"
        )
    if not 0 <= replay <= total:
        raise ValueError("perturbation replay count must lie within the perturbation total")
    if any(count * replay % total for count in families):
        raise ValueError(
            "perturbation replay must split exactly across all three families"
        )
    steps = _integer(flow, "steps", 30_000)
    checkpoint_every = _integer(flow, "checkpoint_every", 5000)
    if checkpoint_every > steps:
        raise ValueError("flow.checkpoint_every cannot exceed flow.steps")
    selection_samples = _integer(flow_selection, "sample_cases", 512)
    if _integer(flow_selection, "materialized_cases", 64) > selection_samples:
        raise ValueError(
            "flow selection cannot materialize more generated cases than it samples"
        )
    projected_validation_cases = _integer(projection, "validation_cases", 0)
    selection_reference_cases = _integer(
        flow_selection, "reference_cases", 128
    )
    if (
        projected_validation_cases > 0
        and selection_reference_cases > projected_validation_cases
    ):
        raise ValueError(
            "flow.selection.reference_cases cannot exceed projected validation cases"
        )
    selection_rule = str(
        flow_selection.get("rule", "empirical_d0_gate")
    )
    if selection_rule not in {"weighted_rank", "empirical_d0_gate"}:
        raise ValueError(
            "flow.selection.rule must be weighted_rank or empirical_d0_gate"
        )
    lower_quantile = _number(
        flow_selection, "baseline_lower_quantile", 0.05
    )
    upper_quantile = _number(
        flow_selection, "baseline_upper_quantile", 0.95
    )
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError(
            "flow.selection baseline quantiles must satisfy "
            "0 <= lower < upper <= 1"
        )
    if selection_rule == "empirical_d0_gate":
        materialized_cases = _integer(
            flow_selection, "materialized_cases", 64
        )
        if selection_reference_cases < 2 * materialized_cases:
            raise ValueError(
                "empirical physical baseline needs reference_cases at least "
                "twice materialized_cases"
            )
        validation_audit_cases = _integer(
            generator, "validation_audit_cases", 2048
        )
        if validation_audit_cases < 2 * selection_samples:
            raise ValueError(
                "empirical latent baseline needs validation_audit_cases at "
                "least twice sample_cases"
            )
        if _integer(flow_selection, "baseline_repeats", 32) < 2:
            raise ValueError(
                "flow.selection.baseline_repeats must be at least two"
            )
    if _number(
        _child_section(pca, "selection"), "relative_tolerance", 0.05
    ) < 0.0:
        raise ValueError("pca.selection.relative_tolerance cannot be negative")


def _required_implementation_files(config: NSCampaignConfig) -> tuple[Path, ...]:
    names = (
        "project_d0_to_structured_coefficients.py",
        "audit_ns_structured_coefficient_pca.py",
        "train_ns_structured_coefficient_flow.py",
        "sample_ns_structured_coefficient_flow.py",
        "materialize_ns_structured_flow_samples.py",
        "generate_ns_structured_guided_perturbed_round.py",
        "generate_ns_structured_guided_round.py",
        "run_ns_structured_solver_robustness_round.py",
        "train_ns_structured_reward_guidance.py",
        "update_ns_structured_guidance_from_mixed_round.py",
        "run_triplane_agent_adversarial_loop.py",
        "summarize_ns_structured_guided_5round.py",
        "summarize_ns_structured_guided_perturbed_5round.py",
    )
    return tuple(_script(config, name) for name in names) + (_initial_solver(config),)


def _initial_solver(config: NSCampaignConfig) -> Path:
    task = config.section("task")
    value = task.get("initial_solver")
    return (
        _resolve_path(value, config.source_path.parent)
        if value
        else config.implementation_root
        / "baselines"
        / "triplane"
        / "spectral_stokes_projection_submission.py"
    )


def _agent_prompt(config: NSCampaignConfig) -> Path | None:
    """Return an optional researcher-written addition to the NS Agent prompt."""

    value = config.section("agent").get("prompt_file")
    if value is None or not str(value).strip():
        return None
    return _resolve_path(str(value), config.source_path.parent)


def _script(config: NSCampaignConfig, name: str) -> Path:
    return config.implementation_root / "scripts" / name


def _resources(config: NSCampaignConfig, stage: str, *, gpu: bool) -> StageResources:
    defaults = {
        "project_train": (12, "96G", "12:00:00"),
        "project_validation": (12, "96G", "12:00:00"),
        "project_hidden_reference": (12, "96G", "12:00:00"),
        "fit_pca": (12, "96G", "04:00:00"),
        "select_pca": (2, "8G", "00:10:00"),
        "train_generator": (8, "64G", "08:00:00"),
        "audit_generator": (8, "64G", "12:00:00"),
        "select_generator": (2, "8G", "00:10:00"),
        "prepare_initial_hidden": (8, "64G", "04:00:00"),
        "adversarial_rounds": (16, "128G", "1-00:00:00"),
        "publish_result": (4, "16G", "01:00:00"),
    }
    cpus, memory, time_limit = defaults[stage]
    execution = config.section("execution")
    resources = execution.get("resources", {})
    if not isinstance(resources, Mapping):
        raise ValueError("execution.resources must be an object")
    common = resources.get("default", {})
    override = resources.get(stage, {})
    if not isinstance(common, Mapping) or not isinstance(override, Mapping):
        raise ValueError("each execution resource entry must be an object")
    merged = {**common, **override}
    partition = str(
        merged.get(
            "partition",
            execution.get("gpu_partition" if gpu else "cpu_partition", "gpubase_l40s_b3"),
        )
    )
    return StageResources(
        cpus=int(merged.get("cpus", cpus)),
        memory=str(merged.get("memory", memory)),
        time_limit=str(merged.get("time", time_limit)),
        partition=partition,
        gres=str(merged.get("gres", execution.get("gpu_gres", "gpu:l40s:1") if gpu else "")),
    )


def _require_case_files(
    row: Mapping[str, Any], *, manifest_path: Path, implementation_root: Path
) -> None:
    case_id = str(row.get("case_id", ""))
    raw = Path(str(row.get("case_dir", "")))
    candidates = (
        raw,
        implementation_root / raw,
        manifest_path.parent / "cases" / case_id,
    )
    case_dir = next((path.resolve() for path in candidates if path.is_dir()), None)
    if case_dir is None:
        raise FileNotFoundError(f"cannot resolve NS dataset case {case_id!r}")
    required = ("public_question.json", "hidden_truth_cache.npz", "case_spec.json")
    missing = [name for name in required if not (case_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"NS dataset case {case_id!r} is missing {missing}")


def _section(values: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = values.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {name!r} must be an object")
    return value


def _child_section(values: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = values.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _integer(values: Mapping[str, Any], name: str, default: int) -> int:
    value = values.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _number(values: Mapping[str, Any], name: str, default: float) -> float:
    value = values.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _boolean(values: Mapping[str, Any], name: str, default: bool) -> bool:
    value = values.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _rank_values(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        raw = value.replace(":", ",").split(",")
    elif isinstance(value, list):
        raw = value
    else:
        raise ValueError("pca.ranks must be a list or comma-separated string")
    ranks = tuple(sorted({int(item) for item in raw if str(item).strip()}))
    if not ranks or min(ranks) <= 0:
        raise ValueError("pca.ranks must contain positive integers")
    return ranks


def _ranks(value: Any) -> str:
    return ":".join(str(item) for item in _rank_values(value))


def _checkpoint_steps(steps: int, checkpoint_every: int) -> tuple[int, ...]:
    values = list(range(int(checkpoint_every), int(steps) + 1, int(checkpoint_every)))
    if not values or values[-1] != int(steps):
        values.append(int(steps))
    return tuple(values)


def _resolve_path(value: Path | str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _slug(value: str) -> str:
    slug = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return "-".join(part for part in slug.split("-") if part) or "ns-run"
