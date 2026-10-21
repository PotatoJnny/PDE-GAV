#!/usr/bin/env python3
"""Compose and run one formal NS Campaign through the canonical Controller.

This launcher contains no scientific lifecycle.  It translates one explicit
JSON recipe into SDK programs, then calls ``CampaignController`` exactly once.
Both the structured numerical and completed-field PIV profiles use the same
Controller, AgentExecutionService, Candidate codec, and result writer.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Lock
import time
import traceback
from types import MappingProxyType
from typing import Mapping, Sequence

from pde_agent_verifier_solver.engine import (
    AgentSpec,
    CampaignCompiler,
    CampaignController,
    CampaignRecipe,
    CaseCounts,
    CommandAgentProgram,
    DevelopmentSamplingPolicy,
    EvaluatorPolicy,
    EventStream,
    GeneratorRecipe,
    PythonCandidateCodec,
    PythonTaskWorkerSpec,
    RoundSettings,
    RoundOverride,
    SubprocessAgentExecutionService,
    SubprocessCandidateExecutionService,
)
from pde_agent_verifier_solver.engine.guidance_controller import (
    GuidanceProtocolConfig,
)
from pde_agent_verifier_solver.engine.memory import CAMPAIGN_MEMORY_HEADER
from pde_agent_verifier_solver.reporting.costs import MeasuredGenerator, collect_run_costs
from pde_agent_verifier_solver.plugins.ns import (
    NSProblem,
    NSSolverTask,
    StructuredNSGuidanceConfig,
    assemble_completed_piv,
    build_canonical_structured_ns_generator,
    ns_public_problem_bundle,
)
from pde_agent_verifier_solver.plugins.ns.development_capability import (
    NSDevelopmentCapabilityFacade,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    Artifact,
    CandidateBundleLimits,
    DatasetInput,
    DatasetKind,
    DevelopmentCoordinateMode,
    GuidanceTrainingBudget,
    HiddenSuiteSpec,
    OfficialEvaluationSpec,
    SampleSource,
    SamplingStrategy,
    TaskKind,
    TerminalCohortSpec,
    TerminalEvaluationSpec,
    TerminalGuidanceSource,
    TerminalMetricSpec,
    inspect_candidate_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "canonical_ns_campaign_v1"
PIV_MAX_CANDIDATE_BUNDLE_BYTES = 64 * 1024 * 1024
PROFILES = frozenset({"structured_numerical", "experimental_piv", "experimental_mms"})


class _Composition:
    def __init__(
        self,
        *,
        campaign,
        dataset,
        problem,
        task,
        generator,
        public_problem_bundle,
        agents,
        execution,
        candidate_execution,
        codec,
        provenance,
    ) -> None:
        self.campaign = campaign
        self.dataset = dataset
        self.problem = problem
        self.task = task
        self.generator = generator
        self.public_problem_bundle = public_problem_bundle
        self.agents = agents
        self.execution = execution
        self.candidate_execution = candidate_execution
        self.codec = codec
        self.provenance = provenance


def _execute(argv: Sequence[str] | None = None, *, segment_budget=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an NS Campaign from an explicit canonical configuration."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--segment-seconds", type=float)
    parser.add_argument(
        "--codex-binary",
        type=Path,
        default=Path(os.environ.get('CODEX_BINARY', 'codex')),
    )
    parser.add_argument(
        "--auth-file",
        type=Path,
        default=Path(os.environ.get('CODEX_AUTH_FILE', os.path.expanduser('~/.codex/auth.json'))),
    )
    parser.add_argument(
        "--worker-gpus",
        default=os.environ.get("PDE_AGENT_WORKER_GPU_DEVICES", ""),
        help="One comma-separated Slurm GPU token per configured Agent.",
    )
    parser.add_argument(
        "--controller-gpu",
        default=os.environ.get("PDE_CONTROLLER_GPU_DEVICE", ""),
        help="The distinct GPU token visible to the Controller.",
    )
    parser.add_argument(
        "--candidate-gpus", default=None,
        help="Bind configured Official/Terminal GPU role to allocation tokens without changing scientific config.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Load all data/checkpoints and compile the Campaign, but do not run it.",
    )
    args = parser.parse_args(argv)

    config, config_path = _load_config(args.config)
    _validate_config(config)
    candidate_gpu_tokens = None if args.candidate_gpus is None else tuple(args.candidate_gpus.split(","))
    worker_gpus, controller_gpu = _gpu_allocation(
        args.worker_gpus,
        args.controller_gpu,
        agent_count=len(_mapping_sequence(config, "agents")),
    )
    from pde_agent_verifier_solver.engine.controller_gpu_visibility import configure_controller_evaluation_devices
    evaluation_device_map = (configure_controller_evaluation_devices(controller_gpu, worker_gpus)
                             if os.environ.get("CUDA_VISIBLE_DEVICES") else {})
    codex_binary = _required_executable(args.codex_binary, "Codex binary")
    auth_file = _required_regular_file(args.auth_file, "Codex auth file")
    if auth_file.is_symlink():
        raise ValueError("Codex auth file cannot be a symlink")

    run_root = args.run_root.expanduser().resolve()
    if args.preflight_only and config["profile"] == "experimental_mms":
        frozen = run_root / "controller_private" / "experimental_mms_preparation.json"
        if not frozen.is_file() and (config.get("reconstruction") is not None
                                    or not _section(config, "generator").get("artifacts")):
            raise ValueError("experimental MMS preflight requires a saved continuous dataset and Generator checkpoint; "
                             "prepare them through a Project or a normal Run first")
    checkpoint_path = run_root / "checkpoints" / "checkpoint.json"
    if args.resume and not checkpoint_path.is_file():
        raise FileNotFoundError("resume requires a committed Campaign checkpoint")
    owned_empty = (segment_budget is not None and run_root.is_dir()
                   and {item.name for item in run_root.iterdir()} == {".checkpoint-writer.lock"})
    if run_root.exists() and not args.resume and not owned_empty:
        raise FileExistsError(f"run root already exists: {run_root}")
    run_root.mkdir(parents=True, mode=0o700, exist_ok=args.resume or owned_empty)
    os.chmod(run_root, 0o700)
    writer = _RunWriter(run_root)
    from uuid import uuid4
    segment_relative = Path("segments") / uuid4().hex if args.resume else Path()
    audit_root = run_root / segment_relative
    def write_metadata(name, value):
        writer.write_json(str(segment_relative / name), value)
    previous_audits = {
        filename: (json.loads((run_root / filename).read_text())
                   if args.resume and (run_root / filename).is_file() else [])
        for filename in ("execution_audits.json", "candidate_execution_audits.json")
    }
    writer.write_json(
        "status.json",
        {
            "schema": "canonical_ns_campaign_status_v1",
            "status": "preflighting",
            "profile": config["profile"],
            "started_at_unix": time.time(),
        },
    )
    write_metadata(
        "resolved_config.json",
        {
            **config,
            "source_config": str(config_path),
            "run_root": str(run_root),
            "worker_gpu_tokens": worker_gpus,
            "controller_gpu_token": controller_gpu,
            "candidate_gpu_tokens_override": candidate_gpu_tokens,
        },
    )
    write_metadata("source_state.json", _source_state(config_path))

    from pde_agent_verifier_solver.engine.campaign_continuation import CampaignPaused
    execution = None
    candidate_execution = None
    composition = None
    journals = {}
    def flush_audits():
        if composition is not None:
            flush_operations = getattr(composition.generator, "flush_operations", None)
            if callable(flush_operations):
                flush_operations()
        for filename, service in (("execution_audits.json", execution),
                                  ("candidate_execution_audits.json", candidate_execution)):
            if service is not None:
                if filename not in journals:
                    from pde_agent_verifier_solver.engine.audit_journal import AuditJournal
                    journals[filename] = AuditJournal(run_root / (filename + "l"),
                        prior_rows=previous_audits[filename], encode=_jsonable)
                journals[filename].append(getattr(service, "audits", ()))

    try:
        composition = _compose(
            config=config,
            config_path=config_path,
            run_root=run_root,
            codex_binary=codex_binary,
            auth_file=auth_file,
            worker_gpus=worker_gpus,
            controller_gpu=controller_gpu,
            evaluation_device_map=evaluation_device_map,
            candidate_gpu_tokens=candidate_gpu_tokens,
        )
        execution = composition.execution
        candidate_execution = composition.candidate_execution
        write_metadata("input_provenance.json", composition.provenance)
        write_metadata(
            "compiled_campaign.json",
            _jsonable(composition.campaign),
        )
        if args.preflight_only:
            writer.write_json(
                "status.json",
                {
                    "schema": "canonical_ns_campaign_status_v1",
                    "status": "preflight_completed",
                    "profile": config["profile"],
                    "finished_at_unix": time.time(),
                },
            )
            print(f"preflight completed: {run_root}", flush=True)
            return 0

        hidden_evaluation_audit = _bind_hidden_evaluation_audit(
            config, config_path, composition, audit_root,
        )
        if hidden_evaluation_audit is not None:
            composition.generator = hidden_evaluation_audit
        composition.generator = MeasuredGenerator(
            composition.generator, run_root / "generator_operations.jsonl", resume=args.resume,
        )
        events = EventStream()
        events.subscribe(writer.record_event)
        writer.write_json(
            "status.json",
            {
                "schema": "canonical_ns_campaign_status_v1",
                "status": "running",
                "profile": config["profile"],
                "started_at_unix": time.time(),
            },
        )
        controller = CampaignController(
            campaign=composition.campaign,
            dataset=composition.dataset,
            problem=composition.problem,
            task=composition.task,
            generator=composition.generator,
            agents=composition.agents,
            events=events,
            agent_execution=composition.execution,
            public_problem_bundle=composition.public_problem_bundle,
            candidate_execution=composition.candidate_execution,
        )
        continuation = None
        pause_control = None
        store = None
        if segment_budget is not None:
            from pde_agent_verifier_solver.engine.checkpoint_store import CheckpointStore, source_identity
            from pde_agent_verifier_solver.engine.campaign_continuation import CampaignPauseControl
            if not callable(getattr(composition.generator, "export_runtime_state", None)):
                raise ValueError("this NS Generator does not support segmented continuation")
            identity = {
                "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                "source_sha256": source_identity(ROOT),
                "generator_inputs": composition.provenance.get("generator_artifacts", {}),
            }
            store = CheckpointStore(run_root / "checkpoints", identity=identity,
                                    generator=composition.generator)
            if args.resume:
                continuation = store.load()
                writer.restore_events(continuation.controller_state["events"], segment_relative)
            def save_continuation(value):
                flush_audits()
                store.save(value)
            pause_control = CampaignPauseControl(save_continuation, segment_budget.should_pause,
                                                  hidden_batch_size=composition.candidate_execution.max_workers)
            composition.candidate_execution.allocation_cancellation_event = segment_budget.cancel
        result = controller.run(continuation=continuation, pause_control=pause_control)
        if hidden_evaluation_audit is not None:
            hidden_evaluation_audit.assert_complete()
        writer.write_json("result.json", _jsonable(result))
        writer.write_text("CAMPAIGN_MEMORY.md", result.memory.markdown + "\n")
        writer.write_json(
            "candidate_history.json",
            _jsonable(result.candidate_history),
        )
        _publish_selected_candidate(result.selected.artifact, run_root / "selected_candidate")
        writer.write_json(
            "execution_audits.json",
            [_jsonable(audit) for audit in composition.execution.audits],
        )
        writer.write_json(
            "candidate_execution_audits.json",
            [
                _jsonable(audit)
                for audit in composition.candidate_execution.audits
            ],
        )
        writer.write_json(
            "status.json",
            {
                "schema": "canonical_ns_campaign_status_v1",
                "status": "completed",
                "profile": config["profile"],
                "campaign_name": result.campaign_name,
                "selected_artifact_id": result.selected.artifact.artifact_id,
                "best_terminal_round": result.best_terminal_round,
                "rounds_completed": len(result.rounds),
                "finished_at_unix": time.time(),
            },
        )
        _record_cost_report(writer, run_root)
        print(f"campaign completed: {run_root}", flush=True)
        return 0
    except CampaignPaused as pause:
        writer.write_json("status.json", {
            "schema": "canonical_ns_campaign_status_v1", "status": "paused",
            "profile": config["profile"], "phase": pause.stage,
            "checkpoint_path": str(checkpoint_path), "updated_at_unix": time.time(),
        })
        print(f"campaign paused: {run_root} ({pause.stage})", flush=True)
        return 75
    except BaseException as error:
        try:
            writer.write_text("failure_traceback.txt", traceback.format_exc())
            if execution is not None:
                writer.write_json(
                    "execution_audits.json",
                    [_jsonable(audit) for audit in execution.audits],
                )
            if candidate_execution is not None:
                writer.write_json(
                    "candidate_execution_audits.json",
                    [
                        _jsonable(audit)
                        for audit in candidate_execution.audits
                    ],
                )
            _record_cost_report(writer, run_root)
            writer.recover_memory_from_events()
            writer.write_json(
                "status.json",
                {
                    "schema": "canonical_ns_campaign_status_v1",
                    "status": "failed",
                    "profile": config.get("profile"),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "finished_at_unix": time.time(),
                },
            )
        except BaseException as persistence_error:
            error.add_note(
                "failure persistence also failed: "
                f"{type(persistence_error).__name__}: {persistence_error}"
            )
        raise
    finally:
        primary = sys.exc_info()[1]
        errors = []
        for service in (execution, candidate_execution):
            if service is not None:
                try:
                    close = getattr(service, "close", None)
                    if callable(close):
                        close()
                except BaseException as error:
                    errors.append(error)
        try:
            flush_audits()
            for filename, journal in journals.items():
                writer.write_json(filename, journal.rows)
        except BaseException as error:
            errors.append(error)
        _record_cost_report(writer, run_root)
        if errors:
            from pde_agent_verifier_solver.engine.candidate_execution import CandidateExecutionInfrastructureError
            failure = CandidateExecutionInfrastructureError("Campaign service cleanup or audit persistence failed")
            for error in errors:
                failure.add_note(f"{type(error).__name__}: {error}")
            writer.write_text("cleanup_failure_traceback.txt", "\n".join(
                "".join(traceback.format_exception(type(error), error, error.__traceback__))
                for error in errors))
            writer.write_json("status.json", {
                "schema": "canonical_ns_campaign_status_v1", "status": "failed",
                "error_type": type(failure).__name__, "error": str(failure),
                "finished_at_unix": time.time(),
            })
            if primary is not None and not isinstance(primary, CampaignPaused):
                primary.add_note(str(failure))
            else:
                raise failure from errors[0]


def _main(argv: Sequence[str] | None = None, *, segment_budget=None) -> int:
    # Setup and checkpoint validation can fail before services exist. Preserve
    # that diagnosis without overwriting an unrelated, pre-existing Run.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    preliminary, _ = parser.parse_known_args(argv)
    root = preliminary.run_root.expanduser().resolve() if preliminary.run_root else None
    owned = root is not None and (preliminary.resume or not root.exists() or
            (root.is_dir() and {item.name for item in root.iterdir()} <= {".checkpoint-writer.lock"}))
    try:
        return _execute(argv, segment_budget=segment_budget)
    except Exception as error:
        if owned:
            try:
                root.mkdir(parents=True, exist_ok=True, mode=0o700)
                writer = _RunWriter(root)
                writer.write_text("failure_traceback.txt", traceback.format_exc())
                writer.write_json("status.json", {
                    "schema": "canonical_ns_campaign_status_v1", "status": "failed",
                    "error_type": type(error).__name__, "error": str(error),
                    "finished_at_unix": time.time(),
                })
            except Exception as persistence_error:
                error.add_note(f"Setup failure persistence also failed: {persistence_error}")
        raise


def main(argv: Sequence[str] | None = None, *, segment_budget=None) -> int:
    from pde_agent_verifier_solver.engine.checkpoint_store import RunSegmentBudget, run_writer_lock
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--segment-seconds", type=float)
    parser.add_argument("--resume", action="store_true")
    preliminary, _ = parser.parse_known_args(argv)
    owned = segment_budget is None and preliminary.segment_seconds is not None
    if owned:
        segment_budget = RunSegmentBudget(preliminary.segment_seconds)
    if preliminary.resume and segment_budget is None:
        raise ValueError("--resume requires an allocation budget")
    if segment_budget is None or preliminary.run_root is None:
        return _main(argv, segment_budget=segment_budget)
    try:
        with run_writer_lock(preliminary.run_root), segment_budget.signals():
            return _main(argv, segment_budget=segment_budget)
    finally:
        if owned:
            segment_budget.close()


def _record_cost_report(writer, run_root: Path) -> None:
    """Accounting failures remain visible without discarding scientific results."""
    try:
        report = collect_run_costs(run_root)
    except Exception as error:
        report = {"schema": "run_resource_report_v1", "status": "failed",
                  "usage_coverage": "incomplete", "error_type": type(error).__name__,
                  "error": str(error)}
    writer.write_json("costs.json", report)


def _hidden_evaluation_manifest_key(config: Mapping[str, object]) -> str | None:
    configured = tuple(
        key for key in ("shared_evaluation_suite_manifest", "official_audit_manifest")
        if config.get(key) is not None
    )
    if len(configured) > 1:
        raise ValueError(
            "shared_evaluation_suite_manifest and official_audit_manifest are mutually exclusive"
        )
    if not configured:
        return None
    key = configured[0]
    _text(config, key)
    if config["profile"] != "structured_numerical" and not (
        config["profile"] == "experimental_mms"
        and key == "shared_evaluation_suite_manifest"
    ):
        raise ValueError(
            "NS hidden evaluation banks require the structured numerical profile "
            "or experimental_mms with a shared continuous suite"
        )
    return key


def _bind_hidden_evaluation_audit(config, config_path, composition, run_root):
    """Bind only the explicitly chosen hidden evaluator; preserve Agent tools."""

    key = _hidden_evaluation_manifest_key(config)
    if key is None:
        return None
    if key == "shared_evaluation_suite_manifest":
        from pde_agent_verifier_solver.plugins.ns.mixed_evaluation import SharedNSMixedEvaluationGenerator
        wrapper = SharedNSMixedEvaluationGenerator
        audit_name = "shared_evaluation_case_audit.jsonl"
    else:
        from pde_agent_verifier_solver.plugins.ns.official_audit import OfficialAuditGenerator
        wrapper = OfficialAuditGenerator
        audit_name = "official_case_audit.jsonl"
    return wrapper(
        composition.generator,
        _config_path(config_path, _text(config, key)),
        run_root / "controller_private" / audit_name,
        composition.campaign.master_seed,
        {**composition.campaign.task_settings,
         "default_query": composition.campaign.evaluator.default_query},
    )


def _compose(
    *,
    config: Mapping[str, object],
    config_path: Path,
    run_root: Path,
    codex_binary: Path,
    auth_file: Path,
    worker_gpus: tuple[str, ...],
    controller_gpu: str,
    evaluation_device_map: Mapping[str, str] | None = None,
    evaluation_workers: int | None = None,
    candidate_gpu_tokens: tuple[str, ...] | None = None,
) -> _Composition:
    profile = str(config["profile"])
    execution_config = _section(config, "execution")
    if candidate_gpu_tokens is not None:
        if "gpu_execution" not in execution_config:
            raise ValueError("--candidate-gpus requires configured GPU execution")
        execution_config = {**execution_config, "gpu_execution": {
            **execution_config["gpu_execution"], "candidate_devices": list(candidate_gpu_tokens)}}
    from pde_agent_verifier_solver.engine.gpu_execution_config import (
        candidate_gpu_kwargs, candidate_execution_capabilities, agent_gpu_worker_settings,
        validate_gpu_run_budget,
    )
    validate_gpu_run_budget(config)
    gpu_options = candidate_gpu_kwargs(
        execution_config, reserved_devices=(controller_gpu, *worker_gpus),
    )
    candidate_device = _candidate_device(execution_config)
    # Dedicated GPU configuration takes precedence over Controller sharing.
    if not gpu_options and candidate_device == "cpu_or_cuda":
        gpu_options = {"gpu_devices": (controller_gpu,)}
    candidate_gpus = gpu_options.get("gpu_devices", ())
    from pde_agent_verifier_solver.engine.gpu_execution_config import development_solver_placement
    placement = development_solver_placement(execution_config)
    development_gpus = ({_text(row, "agent_id"): (gpu,) for row, gpu in
                        zip(_mapping_sequence(config, "agents"), worker_gpus)}
                       if placement == "agent_gpu" else {})
    candidate_device = candidate_execution_capabilities(execution_config)["device"]
    round_count = _integer(config, "round_count", minimum=1)
    hard_wall_seconds = _integer(execution_config, "agent_hard_wall_seconds", minimum=1)
    model = _text(execution_config, "model")
    reasoning_effort = _text(execution_config, "reasoning_effort")
    guidance_enabled, perturbations_enabled = _generator_switches(config)
    hidden_capable_guidance, hidden_capable_perturbations = _effective_generator_switches(config)
    if profile == "experimental_mms":
        from pde_agent_verifier_solver.plugins.ns.tasks.experimental_mms import NSExperimentalMMSSolverTask
        task = NSExperimentalMMSSolverTask()
    else:
        task = NSSolverTask()
    candidate_config = _section(config, "candidate")
    codec = PythonCandidateCodec(
        task_kind=TaskKind.SOLVER,
        limits=_candidate_bundle_limits(profile, candidate_config),
    )
    initial_bundle = (
        _config_path(config_path, _text(candidate_config, "initial_bundle"))
        if candidate_config.get("initial_bundle") is not None
        else None
    )
    initial = None
    if profile == "structured_numerical":
        dataset, problem, generator, generator_recipe, public_bundle, provenance = (
            _compose_numerical(config, config_path, run_root)
        )
        if initial_bundle is not None:
            initial = codec.bind_existing(
                initial_bundle,
                storage_root=run_root / "controller_candidates",
                artifact_id="user-ns-s0",
                name=_text(candidate_config, "initial_name"),
                summary=_text(candidate_config, "initial_summary"),
                usage=(
                    "Call prepare_solution(case, solve_budget), then evaluate the "
                    "returned continuous handle at a points query."
                ),
            )
    elif profile == "experimental_mms":
        dataset, problem, generator, generator_recipe, public_bundle, provenance = (
            _compose_experimental_mms(config, config_path, run_root)
        )
        if initial_bundle is not None:
            initial = codec.bind_existing(
                initial_bundle, storage_root=run_root / "controller_candidates",
                artifact_id="user-ns-experimental-mms-s0",
                name=_text(candidate_config, "initial_name"),
                summary=_text(candidate_config, "initial_summary"),
                usage="Call prepare_solution(case, solve_budget), then evaluate the continuous handle.",
            )
    elif profile == "experimental_piv":
        assembly, provenance = _compose_piv(config, config_path)
        dataset = assembly.prepared_dataset
        problem = assembly.problem
        generator = assembly.generator
        generator_recipe = assembly.generator_recipe(
            guidance_enabled=hidden_capable_guidance,
            perturbations_enabled=hidden_capable_perturbations,
        )
        public_bundle = None
        if initial_bundle is not None:
            initial = assembly.bind_initial_candidate(
                bundle_path=initial_bundle,
                storage_root=run_root / "controller_candidates",
                candidate_codec=codec,
            )
    else:  # guarded by static validation
        raise ValueError(f"unknown NS profile {profile!r}")

    agent_specs = tuple(
        AgentSpec(
            agent_id=_text(row, "agent_id"),
            name=_text(row, "name"),
            prompt=_text(row, "prompt"),
        )
        for row in _mapping_sequence(config, "agents")
    )
    policy = _development_policy(_section(config, "development"))
    official = OfficialEvaluationSpec(
        round=_hidden_suite(_section(config, "official"))
    )
    evaluator = _section(config, "evaluator")
    evaluator_policy = EvaluatorPolicy(
        default_query=AnswerQuery.at_points(
            fields=("velocity_x", "velocity_y"),
            points=({"t": 0.0, "x": 0.0, "y": 0.0},),
        ),
        allow_grid_truth=_boolean(evaluator, "allow_grid_truth"),
        allow_point_truth=_boolean(evaluator, "allow_point_truth"),
        allow_development_solution_session=_boolean(
            evaluator,
            "allow_development_solution_session",
        ),
        development_coordinate_mode=DevelopmentCoordinateMode(
            _text(evaluator, "development_coordinate_mode")
        ),
        development_point_count=_integer(
            evaluator,
            "development_point_count",
            minimum=1,
        ),
    )
    terminal = _terminal_evaluation(_section(config, "terminal"))
    task_settings = {
        "primary_metric": "velocity_relative_l2",
        "include_pressure": False,
        "development_point_count": evaluator_policy.development_point_count,
        "official_point_count": _integer(evaluator, "official_point_count", minimum=1),
        "terminal_point_count": _integer(evaluator, "terminal_point_count", minimum=1),
    }
    campaign = CampaignCompiler().compile(
        CampaignRecipe(
            name=_text(config, "campaign_id"),
            round_count=round_count,
            agents=agent_specs,
            default_round=RoundSettings(
                agent_budget_seconds=hard_wall_seconds,
                development_sampling=policy,
                official_evaluation=official,
            ),
            generator=generator_recipe,
            evaluator=evaluator_policy,
            round_overrides=_round_overrides(config),
            task_settings=task_settings,
            master_seed=_integer(config, "master_seed", minimum=0),
            initial_candidate=initial,
            terminal_evaluation=terminal,
            promotion=dict(_section(config, "promotion")),
            terminal_score_distribution=terminal.ranking.distribution,
        ),
        generator.capabilities,
    )

    if evaluation_workers is not None:
        from ns_campaign_evaluation_only import EvaluationOnlyAgents, InactiveAgent
        if isinstance(evaluation_workers, bool) or not isinstance(evaluation_workers, int) or evaluation_workers < 1:
            raise ValueError("evaluation_workers must be a positive integer")
        controller_rpc_timeout_seconds = _integer(execution_config, "controller_rpc_timeout_seconds", minimum=1)
        programs = {agent.agent_id: InactiveAgent() for agent in agent_specs}
        execution = EvaluationOnlyAgents()
    else:
        worker = _section(config, "worker")
        worker_script = _config_path(config_path, _text(worker, "script"))
        worker_config = _config_path(config_path, _text(worker, "codex_config"))
        _required_regular_file(worker_script, "canonical Agent worker")
        _required_regular_file(worker_config, "canonical Codex config")
        controller_rpc_timeout_seconds = _integer(
            _section(config, "execution"),
            "controller_rpc_timeout_seconds",
            minimum=1,
        )
        runtime_root = run_root / "agent_runtime"
        runtime_root.mkdir(mode=0o700, exist_ok=True)
        runtime_root.chmod(0o700)
        source_path = str(ROOT / "src")
        programs = {}
        for agent, gpu in zip(agent_specs, worker_gpus, strict=True):
            programs[agent.agent_id] = CommandAgentProgram(
                (
                    sys.executable,
                    str(worker_script),
                    "--codex-binary",
                    str(codex_binary),
                    "--auth-file",
                    str(auth_file),
                    "--runtime-root",
                    str(runtime_root),
                    "--config",
                    str(worker_config),
                ),
                codec.decode_nomination,
                incumbent_encoder=codec.encode_incumbent,
                environment={
                    "PYTHONPATH": source_path,
                    "OMP_NUM_THREADS": "8",
                    "MKL_NUM_THREADS": "8",
                },
                worker_settings={
                    "profile": profile,
                    "nomination_policy": execution_config.get("nomination_policy", "all"),
                    "transient_resume_max_attempts": execution_config.get("transient_resume_max_attempts", 1),
                    "transient_resume_until_deadline": execution_config.get("transient_resume_until_deadline", True),
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "candidate_abi": "prepare_solution_then_evaluate_points_v1",
                    "development_capabilities": {
                        "guidance_enabled": guidance_enabled,
                        "perturbations_enabled": perturbations_enabled,
                    },
                    "candidate_execution": candidate_execution_capabilities(execution_config),
                    **agent_gpu_worker_settings(execution_config),
                    "controller_rpc_timeout_seconds": controller_rpc_timeout_seconds,
                },
                gpu_devices=(gpu,),
            )
        execution = SubprocessAgentExecutionService(
            run_root / "agent_turns",
            hard_wall_seconds=hard_wall_seconds,
            controller_gpu_devices=(controller_gpu,),
            evaluation_device_map=evaluation_device_map,
            poll_interval_seconds=0.25,
            capability_facade=NSDevelopmentCapabilityFacade(
                guidance_enabled=guidance_enabled,
                perturbations_enabled=perturbations_enabled,
            ),
        )
    candidate_execution = SubprocessCandidateExecutionService(
        run_root / "candidate_execution",
        task=PythonTaskWorkerSpec.from_task_type(type(task)),
        evaluation_settings={
            **task_settings,
            "default_query": evaluator_policy.default_query,
        },
        development_gpu_devices=development_gpus,
        trace_calls=execution_config.get("trace_calls", False),
        solver_batch_size=execution_config.get("solver_batch_size", 1),
        gpu_slots_per_device=execution_config.get("gpu_slots_per_device", 1),
        oracle_concurrency_per_device=execution_config.get("oracle_concurrency_per_device", 1),
        development_oracle_max_workers=execution_config.get("development_oracle_max_workers"),
        max_workers=(evaluation_workers if evaluation_workers is not None else _integer(
            _section(config, "execution"), "candidate_case_workers", minimum=1,
        )),
        oracle_max_workers=execution_config.get("oracle_max_workers", min(128,
            evaluation_workers if evaluation_workers is not None else execution_config["candidate_case_workers"])),
        case_hard_wall_seconds=_integer(
            _section(config, "execution"),
            "candidate_case_hard_wall_seconds",
            minimum=1,
        ),
        development_batch_hard_wall_seconds=_integer(
            _section(config, "execution"),
            "development_candidate_batch_hard_wall_seconds",
            minimum=1,
        ),
        python_executable=sys.executable,
        **gpu_options,
    )
    return _Composition(
        campaign=campaign,
        dataset=dataset,
        problem=problem,
        task=task,
        generator=generator,
        public_problem_bundle=public_bundle,
        agents=MappingProxyType(programs),
        execution=execution,
        candidate_execution=candidate_execution,
        codec=codec,
        provenance={
            **provenance,
            "controller_evaluation_device_map": dict(evaluation_device_map or {}),
            "profile": profile,
            "initial_candidate": (
                {
                    "bundle": str(initial.program.bundle_locator),
                    "tree_sha256": initial.program.tree_sha256,
                }
                if initial is not None
                else None
            ),
            "candidate_bundle_policy": {
                "max_total_bytes": codec.limits.max_total_bytes,
                "max_file_bytes": codec.limits.max_file_bytes,
            },
            "generator_capabilities": {
                "guidance_enabled": hidden_capable_guidance,
                "perturbations_enabled": hidden_capable_perturbations,
            },
            "agent_protocol": {
                "development_capabilities": {
                    "guidance_enabled": guidance_enabled,
                    "perturbations_enabled": perturbations_enabled,
                },
                "rounds": round_count,
                "parallel_agents": len(agent_specs),
                "time_budget_mode": "hard_wall",
                "hard_wall_seconds_per_agent_per_round": hard_wall_seconds,
                "controller_rpc_timeout_seconds": controller_rpc_timeout_seconds,
                "candidate_case_workers": candidate_execution.max_workers,
                "parallel_evaluation": dict(candidate_execution.execution_settings_snapshot()),
                "candidate_device": candidate_device,
                "candidate_gpu_devices": list(candidate_gpus),
                "candidate_gpu_workers_per_device": (
                    candidate_execution.gpu_device_pool.slots_per_device if candidate_gpus else 0
                ),
                "candidate_case_hard_wall_seconds": (
                    candidate_execution.case_hard_wall_seconds
                ),
                "development_candidate_batch_hard_wall_seconds": (
                    candidate_execution.development_batch_hard_wall_seconds
                ),
                "model": model,
                "reasoning_effort": reasoning_effort,
            },
        },
    )


def _structured_guidance_configuration(guidance):
    backend_settings = _section(guidance, "backend")
    guidance_config = StructuredNSGuidanceConfig(
        training_steps=_integer(backend_settings, "training_steps", minimum=1),
        batch_size=_integer(backend_settings, "batch_size", minimum=1),
        width=_integer(backend_settings, "width", minimum=1),
        depth=_integer(backend_settings, "depth", minimum=1),
        time_width=_integer(backend_settings, "time_width", minimum=2),
        learning_rate=_number(backend_settings, "learning_rate", minimum=0.0, strict=True),
        weight_decay=_number(backend_settings, "weight_decay", minimum=0.0),
        maximum_gradient_norm=_number(
            backend_settings,
            "maximum_gradient_norm",
            minimum=0.0,
            strict=True,
        ),
        minimum_cases=_integer(backend_settings, "minimum_cases", minimum=1),
        guidance_strength=_number(backend_settings, "guidance_strength", minimum=0.0),
        guidance_max_flow_ratio=_number(
            backend_settings,
            "guidance_max_flow_ratio",
            minimum=0.0,
            strict=True,
        ),
    )
    protocol = _section(guidance, "protocol")
    declared_guidance_hash = guidance_config.sha256
    training_budget = _section(protocol, "training_budget")
    guidance_protocol = GuidanceProtocolConfig(
        training_budget=GuidanceTrainingBudget(
            maximum_wall_seconds=_number(
                training_budget,
                "maximum_wall_seconds",
                minimum=0.0,
                strict=True,
            ),
            compute_capacity=_text(training_budget, "compute_capacity"),
            maximum_concurrency=_integer(
                training_budget,
                "maximum_concurrency",
                minimum=1,
            ),
        ),
        guidance_config_sha256=declared_guidance_hash,
        # The adapter replaces this legacy identity field with the actual clean
        # Base count in each common Official cohort before training.
        reward_feedback_case_count=_integer(
            protocol,
            "reward_feedback_case_count",
            minimum=1,
        ),
        official_round_case_count=_integer(
            protocol,
            "official_round_case_count",
            minimum=1,
        ),
        official_terminal_case_count=_integer(
            protocol,
            "official_terminal_case_count",
            minimum=1,
        ),
        inference_wall_budget_seconds=_number(
            protocol,
            "inference_wall_budget_seconds",
            minimum=0.0,
            strict=True,
        ),
        inference_compute_capacity=_text(protocol, "inference_compute_capacity"),
        official_sample_settings=dict(_section(protocol, "official_sample_settings")),
    )
    return guidance_config, guidance_protocol


def _compose_numerical(config, config_path: Path, run_root: Path):
    dataset_config = _section(config, "dataset")
    dataset_input = DatasetInput(
        dataset_id=_text(dataset_config, "dataset_id"),
        kind=DatasetKind.NUMERICAL,
        location=str(_config_path(config_path, _text(dataset_config, "manifest"))),
        settings=dict(_section(dataset_config, "settings")),
    )
    problem = NSProblem()
    prepared = problem.prepare_dataset(dataset_input)
    generator_config = _section(config, "generator")
    artifacts = _section(generator_config, "artifacts")
    expected_hashes = generator_config.get("artifact_sha256", {})
    artifact_paths = {
        name: _config_path(config_path, str(value))
        for name, value in artifacts.items()
    }
    actual_hashes = {}
    for name, path in artifact_paths.items():
        _required_regular_file(path, f"numerical Generator artifact {name}")
        actual_hashes[name] = expected_hashes.get(name)  # Optional archived metadata.
    runtime = _section(generator_config, "runtime")
    guidance = _section(generator_config, "guidance")
    guidance_config, guidance_protocol = _structured_guidance_configuration(guidance)
    development_guidance, development_perturbations = _generator_switches(config)
    bundle = build_canonical_structured_ns_generator(
        generator_id=_text(generator_config, "generator_id"),
        generator_sha256=_text(generator_config, "generator_sha256"),
        flow_checkpoint=artifact_paths["flow_checkpoint"],
        pca_checkpoint=artifact_paths["pca_checkpoint"],
        time_interval=tuple(_number_sequence(runtime, "time_interval", length=2)),
        viscosity=_number(runtime, "viscosity", minimum=0.0),
        perturbation_amplitude=_number(
            runtime,
            "perturbation_amplitude",
            minimum=0.0,
            strict=True,
        ),
        device=_text(runtime, "device"),
        integration_steps=_integer(runtime, "integration_steps", minimum=1),
        flow_state=_text(runtime, "flow_state"),
        integration_method=_text(runtime, "integration_method"),
        cuda_graph_sampling=(
            _boolean(runtime, "cuda_graph_sampling")
            if "cuda_graph_sampling" in runtime else False
        ),
        perturbation_family=_text(runtime, "perturbation_family"),
        spectral_max_mode=_integer(runtime, "spectral_max_mode", minimum=1),
        max_points_per_request=_integer(
            runtime,
            "max_points_per_request",
            minimum=1,
        ),
        guidance_artifact_root=run_root / "controller_private" / "ns_guidance",
        guidance_protocol=guidance_protocol,
        guidance_config=guidance_config,
        development_guidance_enabled=development_guidance,
        development_perturbations_enabled=development_perturbations,
    )
    guidance_enabled, perturbations_enabled = _effective_generator_switches(config)
    recipe = GeneratorRecipe(
        basis_size=int(bundle.components.flow_sampler.pca.rank),
        guidance_enabled=guidance_enabled,
        perturbations_enabled=perturbations_enabled,
        settings={
            "device": _text(runtime, "device"),
            "integration_steps": _integer(runtime, "integration_steps", minimum=1),
            "flow_state": _text(runtime, "flow_state"),
            "cuda_graph_sampling": (
                _boolean(runtime, "cuda_graph_sampling")
                if "cuda_graph_sampling" in runtime else False
            ),
        },
        checkpoint=bundle.checkpoint,
    )
    return (
        prepared,
        problem,
        bundle.generator,
        recipe,
        ns_public_problem_bundle(guidance_supported=True),
        {
            "dataset_manifest": dataset_input.location,
            "dataset_summary": _jsonable(prepared.training_summary),
            "generator_artifacts": {
                name: {"path": str(artifact_paths[name]), "sha256": digest}
                for name, digest in actual_hashes.items()
            },
            "guidance_config_sha256": guidance_config.sha256,
        },
    )


def _compose_experimental_mms(config, config_path: Path, run_root: Path):
    """NS-only composition; preparation uses the existing Project lifecycle."""
    from pde_agent_verifier_solver.app import ProductApplication, ProgramBundle
    from pde_agent_verifier_solver.sdk import ReconstructionRecipe
    from pde_agent_verifier_solver.plugins.ns.product import build_ns_experimental_mms_plugin
    from pde_agent_verifier_solver.plugins.ns.experimental_mms_project import ContinuousPIVReconstruction
    from pde_agent_verifier_solver.plugins.ns.experimental_mms import assemble_experimental_mms

    dataset_config = _section(config, "dataset")
    generator_config = _section(config, "generator")
    manifest = _config_path(config_path, _text(dataset_config, "manifest"))
    private = run_root / "controller_private"
    saved_dataset = private / "experimental_mms_dataset.json"
    artifact_config = _section(generator_config, "artifacts")
    artifact_paths = {name: _config_path(config_path, _text(artifact_config, name))
                      for name in ("flow_checkpoint", "pca_checkpoint")}
    for path in artifact_paths.values():
        _required_regular_file(path, "numerical Generator checkpoint")
    report_path = private / "reconstruction_report.json"
    preparation_path = private / "experimental_mms_preparation.json"
    preparation_config = {
        "dataset": dict(dataset_config), "manifest": str(manifest),
        "reconstruction": config.get("reconstruction"),
        "generator": dict(generator_config), "master_seed": config["master_seed"],
    }
    frozen = None
    if preparation_path.exists():
        frozen = json.loads(preparation_path.read_text())
        if frozen.get("config") != preparation_config:
            raise ValueError("experimental MMS preparation config differs from this Run's frozen inputs")
        for path in (saved_dataset, report_path):
            if path.name in frozen.get("sha256", {}) and not path.is_file():
                raise FileNotFoundError(path)
    elif any(path.exists() for path in (saved_dataset, report_path)):
        raise ValueError("incomplete experimental MMS preparation; use a fresh Run directory")
    dataset_input = DatasetInput(
        dataset_id=_text(dataset_config, "dataset_id"), kind=DatasetKind.EXPERIMENTAL,
        location=str(saved_dataset if frozen else manifest),
        settings={} if frozen else dict(_section(dataset_config, "settings")),
    )
    plugin = build_ns_experimental_mms_plugin()
    programs = ProgramBundle(plugin.problem, plugin.task(), plugin.generator, {}, plugin.public_problem_bundle)
    reconstruction_config = config.get("reconstruction")
    reconstruction = None
    if reconstruction_config is not None and frozen is None:
        if not isinstance(reconstruction_config, Mapping):
            raise ValueError("experimental MMS reconstruction must be an object")
        reconstruction = ReconstructionRecipe(
            program=ContinuousPIVReconstruction(),
            data_settings=dict(reconstruction_config.get("data_settings", {})),
            method_settings=dict(reconstruction_config.get("method_settings", {})),
            seed=_integer(config, "master_seed", minimum=0),
        )
    project = ProductApplication.local_development().create_project(
        "Experimental NS preparation", dataset_input, programs, reconstruction=reconstruction,
    )
    terminal_original = sum(
        _hidden_suite(_section(row, "suite")).original
        for row in _mapping_sequence(_section(config, "terminal"), "cohorts")
    )
    final_count = int(project.dataset.training_summary.get("terminal_final_test_cases", 0))
    if terminal_original != final_count:
        raise ValueError("experimental MMS Terminal must reserve the entire final_test split")
    official_original = sum(
        (override.official_evaluation.round if override.official_evaluation else _hidden_suite(_section(config, "official"))).original
        for index in range(1, _integer(config, "round_count", minimum=1)+1)
        for override in (_round_overrides(config).get(index, RoundOverride()),)
    )
    if official_original > int(project.dataset.training_summary.get("official_hidden_cases", 0)):
        raise ValueError("experimental MMS Official cohorts exceed the verifier split")
    overrides = _round_overrides(config)
    default_policy = _development_policy(_section(config, "development"))
    required_originals = 0
    for index in range(1, _integer(config, "round_count", minimum=1) + 1):
        policy = overrides.get(index, RoundOverride()).development_sampling or default_policy
        counts = policy.fixed_recipe if policy.strategy is SamplingStrategy.FIXED_RECIPE else policy.starter
        required_originals += (counts or CaseCounts()).original
    if required_originals > len(project.dataset.development_cases):
        raise ValueError("experimental MMS Development originals exceed the fresh development split")
    settings = dict(_section(generator_config, "runtime"))
    actual_hashes = {name: generator_config.get("artifact_sha256", {}).get(name)
                     for name in artifact_paths}
    guidance, perturbations = _effective_generator_switches(config)
    development_guidance, development_perturbations = _generator_switches(config)
    guidance_config = guidance_protocol = guidance_root = None
    if guidance:
        guidance_config, guidance_protocol = _structured_guidance_configuration(_section(generator_config, "guidance"))
        guidance_root = run_root / "guidance"
    assembly = assemble_experimental_mms(
        project.dataset, generator_settings=settings,
        flow_checkpoint=artifact_paths["flow_checkpoint"], pca_checkpoint=artifact_paths["pca_checkpoint"],
        guidance_artifact_root=guidance_root, guidance_protocol=guidance_protocol, guidance_config=guidance_config,
        development_guidance_enabled=development_guidance,
        development_perturbations_enabled=development_perturbations,
    )
    if frozen is None:
        private.mkdir(parents=True, exist_ok=True)
        from pde_agent_verifier_solver.plugins.ns.experimental_mms_project import save_prepared_mms_dataset
        save_prepared_mms_dataset(project.dataset, saved_dataset)
        if project.reconstruction_report is not None:
            report_path.write_text(json.dumps(project.reconstruction_report.as_dict(), indent=2) + "\n")
        frozen = {"config": preparation_config, "generator_sha256": actual_hashes, "sha256": {
            path.name: _file_sha256(path) for path in (saved_dataset, report_path) if path.is_file()
        }}
        preparation_path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    reconstruction_report = json.loads(report_path.read_text()) if report_path.is_file() else None
    return (
        project.dataset, plugin.problem, assembly.generator,
        assembly.generator_recipe(settings={}, guidance_enabled=guidance,
                                  perturbations_enabled=perturbations),
        plugin.public_problem_bundle,
        {"dataset_manifest": str(manifest), "prepared_dataset_manifest": str(saved_dataset),
         "dataset_summary": _jsonable(project.dataset.training_summary),
         "generator_checkpoint": {"path": str(artifact_paths["flow_checkpoint"]),
                                  "sha256": actual_hashes["flow_checkpoint"]},
         "generator_artifacts": {name: {"path": str(artifact_paths[name]), "sha256": digest}
                                 for name, digest in actual_hashes.items()},
         "reconstruction_report": reconstruction_report},
    )


def _compose_piv(config, config_path: Path):
    dataset = _section(config, "dataset")
    generator = _section(config, "generator")
    manifest = _config_path(config_path, _text(dataset, "manifest"))
    checkpoint = _config_path(config_path, _text(generator, "checkpoint"))
    assembly = assemble_completed_piv(
        manifest_path=manifest,
        generator_checkpoint_path=checkpoint,
        dataset_id=_text(dataset, "dataset_id"),
        dataset_settings=dict(_section(dataset, "settings")),
        generator_settings=dict(_section(generator, "settings")),
    )
    terminal_original = sum(
        _hidden_suite(_section(row, "suite")).original
        for row in _mapping_sequence(_section(config, "terminal"), "cohorts")
    )
    round_count = _integer(config, "round_count", minimum=1)
    overrides = _round_overrides(config)
    default_policy = _development_policy(_section(config, "development"))
    default_official = _hidden_suite(_section(config, "official"))
    official_original = 0
    development_original = 0
    for index in range(1, round_count + 1):
        override = overrides.get(index, RoundOverride())
        policy = override.development_sampling or default_policy
        official = override.official_evaluation.round if override.official_evaluation else default_official
        counts = policy.fixed_recipe if policy.strategy is SamplingStrategy.FIXED_RECIPE else policy.starter
        official_original += official.original
        development_original += (counts or CaseCounts()).original
    if terminal_original > assembly.terminal_final_test_cases:
        raise ValueError(
            "PIV Terminal original count exceeds the final_test split"
        )
    if official_original > assembly.official_hidden_cases:
        raise ValueError(
            "PIV Official cohorts exceed the available verifier split"
        )
    if development_original > len(assembly.prepared_dataset.development_cases):
        raise ValueError("PIV Development originals exceed the fresh development split")
    return (
        assembly,
        {
            "dataset_manifest": str(manifest),
            "dataset_summary": _jsonable(assembly.prepared_dataset.training_summary),
            "generator_checkpoint": {
                "path": str(checkpoint),
                "sha256": _file_sha256(checkpoint),
                "report": _jsonable(assembly.checkpoint.report),
            },
            "terminal_final_test_cases": assembly.terminal_final_test_cases,
            "official_hidden_cases": assembly.official_hidden_cases,
            "project_reconstruction": _jsonable(
                assembly.project_reconstruction
            ),
        },
    )


def _generator_switches(config: Mapping[str, object]) -> tuple[bool, bool]:
    generator = _section(config, "generator")
    # These are Agent development permissions. Hidden evaluation may require
    # wider scientific support, computed by _effective_generator_switches.
    # Existing recipes predate the flags and retain their full G/P behavior.
    return tuple(
        _boolean(generator, key) if key in generator else True
        for key in ("guidance_enabled", "perturbations_enabled")
    )


def _effective_generator_switches(config: Mapping[str, object]) -> tuple[bool, bool]:
    """Scientific support needed by development OR hidden evaluation.

    The explicit generator switches remain development tool permissions. This
    does not choose or replace the Guided model used to generate hidden Cases.
    """

    guidance, perturbations = _generator_switches(config)
    suites = [_hidden_suite(_section(config, "official"))]
    suites.extend(
        override.official_evaluation.round
        for override in _round_overrides(config).values()
        if override.official_evaluation is not None
    )
    terminal = _terminal_evaluation(_section(config, "terminal"))
    suites.extend(cohort.suite for cohort in terminal.cohorts)
    guidance = guidance or any(
        cohort.guidance_source is not TerminalGuidanceSource.BASE
        for cohort in terminal.cohorts
    )
    return (
        guidance or any(suite.guided or suite.guided_perturbed for suite in suites),
        perturbations or any(suite.base_perturbed or suite.guided_perturbed for suite in suites),
    )


def _development_policy(settings: Mapping[str, object]) -> DevelopmentSamplingPolicy:
    if settings.get("case_replay") is not False or settings.get("cumulative_fixed_cases", False):
        raise ValueError("canonical NS configuration must explicitly disable Case replay")
    return DevelopmentSamplingPolicy(
        strategy=SamplingStrategy(_text(settings, "strategy")),
        max_cases_per_agent_per_round=(
            None if "max_cases_per_agent_per_round" in settings
            and settings["max_cases_per_agent_per_round"] is None
            else _integer(settings, "max_cases_per_agent_per_round", minimum=1)
        ),
        allowed_sources=tuple(
            SampleSource(value) for value in _string_sequence(settings, "allowed_sources")
        ),
        starter=_case_counts(_section(settings, "starter")),
        starter_only_sources=tuple(
            SampleSource(value)
            for value in _string_sequence(
                {"starter_only_sources": settings.get("starter_only_sources", [])},
                "starter_only_sources", allow_empty=True,
            )
        ),
        fixed_recipe=(
            _case_counts(_section(settings, "fixed_recipe"))
            if settings.get("fixed_recipe") is not None else None
        ),
        hide_case_types=_boolean(settings, "hide_case_types") if "hide_case_types" in settings else True,
        cumulative_fixed_cases=False,
    )


def _round_overrides(config: Mapping[str, object]) -> dict[int, RoundOverride]:
    raw = config.get("round_overrides", {})
    if not isinstance(raw, Mapping):
        raise TypeError("round_overrides must be an object keyed by round number")
    result = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key:
            raise ValueError("round_overrides keys must be positive integer strings")
        index = int(key)
        if not 1 <= index <= _integer(config, "round_count", minimum=1):
            raise ValueError("round_overrides key is outside the Campaign")
        if not isinstance(value, Mapping) or set(value) - {
            "development", "official", "agent_active_budget_seconds", "agent_hard_wall_seconds",
        }:
            raise ValueError("unsupported round override fields")
        result[index] = RoundOverride(
            development_sampling=(
                _development_policy(_section(value, "development")) if "development" in value else None
            ),
            official_evaluation=(
                OfficialEvaluationSpec(round=_hidden_suite(_section(value, "official")))
                if "official" in value else None
            ),
            agent_budget_seconds=(
                _integer(value, "agent_hard_wall_seconds", minimum=1)
                if "agent_hard_wall_seconds" in value else None
            ),
        )
    return result


def _candidate_device(execution: Mapping[str, object]) -> str:
    # Older frozen recipes retain their CPU evaluation semantics.
    device = execution.get("candidate_device", "cpu")
    if device not in ("cpu", "cpu_or_cuda"):
        raise ValueError("execution.candidate_device must be 'cpu' or 'cpu_or_cuda'")
    return device


def _validate_config(
    config: Mapping[str, object], *, inspect_initial_candidate: bool = True,
) -> None:
    """Validate configurable recipes without changing the canonical lifecycle."""
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"config schema must be {CONFIG_SCHEMA}")
    if config.get("profile") not in PROFILES:
        raise ValueError(f"profile must be one of {sorted(PROFILES)}")
    _hidden_evaluation_manifest_key(config)
    rounds = _integer(config, "round_count", minimum=1)
    _text(config, "campaign_id")
    _integer(config, "master_seed", minimum=0)
    execution = _section(config, "execution")
    _candidate_device(execution)
    from pde_agent_verifier_solver.engine.gpu_execution_config import gpu_execution_settings, validate_gpu_run_budget
    validate_gpu_run_budget(config)
    gpu_enabled = gpu_execution_settings(execution) is not None
    from pde_agent_verifier_solver.engine.gpu_execution_config import development_solver_placement
    development_solver_placement(execution)
    if execution.get("nomination_policy", "all") not in ("all", "latest", "single_primary_v1"):
        raise ValueError("execution.nomination_policy must be all, latest or single_primary_v1")
    if type(execution.get("transient_resume_until_deadline", True)) is not bool:
        raise ValueError("execution.transient_resume_until_deadline must be a boolean")
    if "transient_resume_max_attempts" in execution:
        attempts = _integer(execution, "transient_resume_max_attempts", minimum=1)
        if attempts > 3:
            raise ValueError("execution.transient_resume_max_attempts must be at most 3")
    agent_count = _integer(execution, "agent_count", minimum=1)
    hard = _integer(execution, "agent_hard_wall_seconds", minimum=1)
    rpc = _integer(execution, "controller_rpc_timeout_seconds", minimum=1)
    _integer(execution, "candidate_case_workers", minimum=1)
    for key in ("solver_batch_size", "gpu_slots_per_device", "oracle_max_workers", "oracle_concurrency_per_device"):
        value = execution.get(key, 1)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 128:
            raise ValueError(f"execution.{key} must be an integer in [1, 128]")
    if execution.get("oracle_max_workers", 1) > min(128, execution["candidate_case_workers"] * execution.get("solver_batch_size", 1)):
        raise ValueError("Oracle concurrency exceeds the maximum in-flight Case count")

    development_oracle = execution.get("development_oracle_max_workers", execution.get("oracle_max_workers", 1))
    if (isinstance(development_oracle, bool) or not isinstance(development_oracle, int)
            or not 1 <= development_oracle <= execution.get("oracle_max_workers", 1)):
        raise ValueError("execution.development_oracle_max_workers must fit oracle_max_workers")

    case_wall = _integer(execution, "candidate_case_hard_wall_seconds", minimum=1)
    batch_wall = _integer(execution, "development_candidate_batch_hard_wall_seconds", minimum=1)
    if case_wall > batch_wall or batch_wall >= rpc:
        raise ValueError("Candidate Case wall must fit the batch wall, which must be below RPC timeout")
    _text(execution, "model")
    _text(execution, "reasoning_effort")
    agents = _mapping_sequence(config, "agents")
    if len(agents) != agent_count:
        raise ValueError("execution.agent_count must match the configured Agent roles")
    identifiers = tuple(_text(row, "agent_id") for row in agents)
    if len(set(identifiers)) != agent_count:
        raise ValueError("Agent ids must be distinct")
    for row in agents:
        _text(row, "name")
        _text(row, "prompt")
    policy = _development_policy(_section(config, "development"))
    official = _hidden_suite(_section(config, "official"))
    overrides = _round_overrides(config)
    guidance, perturbations = _generator_switches(config)
    effective_guidance, _ = _effective_generator_switches(config)
    candidate = _section(config, "candidate")
    limits = _candidate_bundle_limits(str(config["profile"]), candidate)
    has_initial = candidate.get("initial_bundle") is not None
    if has_initial:
        initial_path = _config_path(
            Path(_text(config, "_config_path")), _text(candidate, "initial_bundle"),
        )
        if inspect_initial_candidate:
            inspect_candidate_bundle(initial_path, limits=limits)
    for index in range(1, rounds + 1):
        override = overrides.get(index, RoundOverride())
        current = override.development_sampling or policy
        hidden = override.official_evaluation.round if override.official_evaluation else official
        if override.agent_budget_seconds is not None and override.agent_budget_seconds > hard:
            raise ValueError("round total-time limit exceeds the Agent hard wall")
        settings = RoundSettings(
            agent_budget_seconds=override.agent_budget_seconds or hard,
            development_sampling=current,
            official_evaluation=OfficialEvaluationSpec(round=hidden),
        )
        CampaignCompiler._validate_sampling_settings(
            settings, index, has_initial_candidate=has_initial,
        )
        counts = current.fixed_recipe if current.strategy is SamplingStrategy.FIXED_RECIPE else current.starter
        counts = counts or CaseCounts()
        if not guidance and (
            counts.guided or counts.guided_perturbed
            or SampleSource.GUIDED in current.allowed_sources
        ):
            raise ValueError("guided development Cases are requested but guidance is disabled")
        if not perturbations and (
            counts.base_perturbed or counts.guided_perturbed
        ):
            raise ValueError("perturbed development Cases are requested but perturbation is disabled")
        if config["profile"] == "experimental_piv":
            if (current.strategy is SamplingStrategy.AGENT_ADAPTIVE
                and SampleSource.ORIGINAL in current.allowed_sources
                and SampleSource.ORIGINAL not in current.starter_only_sources):
                raise ValueError("PIV original Cases must be restricted to the starter set")
        elif config["profile"] == "structured_numerical":
            if current.starter_only_sources:
                raise ValueError("structured numerical NS has no starter-only Case source")
            if effective_guidance:
                minimum_feedback = _integer(
                    _section(_section(_section(config, "generator"), "guidance"), "backend"),
                    "minimum_cases", minimum=1,
                )
                if hidden.base < minimum_feedback:
                    raise ValueError(
                        "structured NS Official clean Base feedback must meet Guidance minimum_cases"
                    )


def _candidate_bundle_limits(
    profile: str,
    settings: Mapping[str, object],
) -> CandidateBundleLimits:
    if profile != "experimental_piv":
        return CandidateBundleLimits()
    maximum = _integer(settings, "max_bundle_bytes", minimum=1)
    if maximum != PIV_MAX_CANDIDATE_BUNDLE_BYTES:
        raise ValueError(
            "formal PIV Candidate bundle limit must be 67108864 bytes"
        )
    return CandidateBundleLimits(
        max_total_bytes=maximum,
        max_file_bytes=maximum,
    )


def _terminal_evaluation(settings: Mapping[str, object]) -> TerminalEvaluationSpec:
    cohorts = []
    for row in _mapping_sequence(settings, "cohorts"):
        source = TerminalGuidanceSource(_text(row, "guidance_source"))
        round_index = row.get("round_index")
        if round_index is not None:
            if isinstance(round_index, bool) or not isinstance(round_index, int):
                raise TypeError("terminal round_index must be an integer")
        cohorts.append(
            TerminalCohortSpec(
                cohort_id=_text(row, "cohort_id"),
                suite=_hidden_suite(_section(row, "suite")),
                guidance_source=source,
                round_index=round_index,
            )
        )
    ranking = _section(settings, "ranking")
    return TerminalEvaluationSpec(
        cohorts=tuple(cohorts),
        ranking=TerminalMetricSpec(
            cohort_id=_text(ranking, "cohort_id"),
            distribution=_text(ranking, "distribution"),
            statistic=_text(ranking, "statistic"),
        ),
    )


def _hidden_suite(settings: Mapping[str, object]) -> HiddenSuiteSpec:
    counts = _case_counts(settings)
    return HiddenSuiteSpec(**counts.as_dict())


def _case_counts(settings: Mapping[str, object]) -> CaseCounts:
    expected = (
        "original",
        "base",
        "guided",
        "base_perturbed",
        "guided_perturbed",
    )
    return CaseCounts(**{name: _integer(settings, name, minimum=0) for name in expected})


def _gpu_allocation(
    workers: str, controller: str, *, agent_count: int = 2,
) -> tuple[tuple[str, ...], str]:
    values = tuple(value.strip() for value in workers.split(",") if value.strip())
    controller = controller.strip()
    if len(values) != agent_count or not controller:
        raise ValueError(f"NS execution needs {agent_count} worker GPUs and one controller GPU")
    if len(set((*values, controller))) != agent_count + 1:
        raise ValueError("NS GPU tokens must be distinct")
    return values, controller


class _RunWriter:
    def __init__(self, root: Path, *, resume: bool = False) -> None:
        self.root = root
        self._lock = Lock()
        self._memory_rounds: list[str] = []
        if resume and (root / "events.jsonl").is_file():
            for line in (root / "events.jsonl").read_text().splitlines():
                if line.strip():
                    memory = json.loads(line).get("payload", {}).get("round_memory", {})
                    if memory.get("merged_markdown"):
                        self._memory_rounds.append(memory["merged_markdown"])

    def restore_events(self, history, segment_relative):
        history = EventStream(history).snapshot()
        content = "".join(json.dumps(_jsonable(event), sort_keys=True, allow_nan=False) + "\n"
                          for event in history)
        path = self.root / "events.jsonl"
        prior = path.read_text() if path.exists() else ""
        if prior != content:
            self.write_text(str(segment_relative / "events_before_resume.jsonl"), prior)
        self.write_text("events.jsonl", content)
        self._memory_rounds = []
        for event in history:
            memory = event.payload.get("round_memory", {})
            if memory.get("merged_markdown"):
                self._memory_rounds.append(memory["merged_markdown"])
        self.write_text("CAMPAIGN_MEMORY.md", self._render_memory())

    def write_json(self, relative: str, value: object) -> None:
        self.write_text(
            relative,
            json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        )

    def write_text(self, relative: str, value: str) -> None:
        destination = self.root / relative
        temporary = self.root / f".{destination.name}.tmp-{os.getpid()}"
        with self._lock:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(value, encoding="utf-8")
            os.chmod(temporary, 0o600)
            temporary.replace(destination)

    def record_event(self, event) -> None:
        payload = _jsonable(event)
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
        with self._lock:
            with (self.root / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
            event_payload = getattr(event, "payload", {})
            round_memory = (
                event_payload.get("round_memory")
                if isinstance(event_payload, Mapping)
                else None
            )
            if isinstance(round_memory, Mapping):
                merged = round_memory.get("merged_markdown")
                if isinstance(merged, str) and merged.strip():
                    self._memory_rounds.append(merged)
                    self._write_text_locked(
                        "CAMPAIGN_MEMORY.md",
                        self._render_memory(),
                    )
            status = {
                "schema": "canonical_ns_campaign_status_v1",
                "status": "running",
                "last_event_sequence": getattr(event, "sequence", None),
                "last_event_type": getattr(event, "event_type", None),
                "last_event_message": getattr(event, "message", None),
                "round_index": getattr(event, "round_index", None),
                "updated_at_unix": time.time(),
            }
            temporary = self.root / f".status.json.tmp-{os.getpid()}"
            temporary.write_text(
                json.dumps(status, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.chmod(temporary, 0o600)
            temporary.replace(self.root / "status.json")

    def recover_memory_from_events(self) -> None:
        if self._memory_rounds:
            self.write_text("CAMPAIGN_MEMORY.md", self._render_memory())

    def _render_memory(self) -> str:
        return "\n\n".join((CAMPAIGN_MEMORY_HEADER, *self._memory_rounds)) + "\n"

    def _write_text_locked(self, relative: str, value: str) -> None:
        destination = self.root / relative
        temporary = self.root / f".{destination.name}.tmp-{os.getpid()}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(value, encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(destination)


def _publish_selected_candidate(artifact: Artifact, destination: Path) -> None:
    source = getattr(artifact.program, "bundle_locator", None)
    if source is None:
        raise TypeError("selected formal NS Candidate is not a Python bundle")
    expected = getattr(artifact.program, "tree_sha256", None)
    bundle = inspect_candidate_bundle(source, recorded_identity=expected)
    if destination.exists():
        inspect_candidate_bundle(destination, recorded_identity=bundle.tree_sha256)
        return
    from uuid import uuid4
    temporary = destination.with_name(f".{destination.name}.publishing-{uuid4().hex}")
    try:
        shutil.copytree(bundle.root, temporary, symlinks=False)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _load_config(path: Path) -> tuple[dict[str, object], Path]:
    source = path.expanduser().resolve()
    raw = os.path.expandvars(source.read_text(encoding="utf-8"))
    if "${" in raw:
        raise ValueError("campaign config contains unresolved environment variables")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("campaign config must be a JSON object")
    from pde_agent_verifier_solver.sdk.agent_prompt import resolve_agent_prompt_files
    value = resolve_agent_prompt_files(value, relative_to=source.parent)
    value["_config_path"] = str(source)
    return value, source


def _git_prefix(repository_root: Path) -> tuple[str, ...]:
    private_git_dir = repository_root / ".git-real"
    return (
        (
            "git",
            f"--git-dir={private_git_dir}",
            f"--work-tree={repository_root}",
        )
        if private_git_dir.is_dir()
        else ("git", "-C", str(repository_root))
    )


def _source_state(config_path: Path) -> Mapping[str, object]:
    git_prefix = _git_prefix(ROOT)

    def git(*arguments: str) -> str | None:
        completed = subprocess.run(
            (*git_prefix, *arguments),
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {
        "repository_root": str(ROOT),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "status_porcelain": git("status", "--porcelain", "--untracked-files=all"),
        "config": str(config_path),
        "python": sys.executable,
    }


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Artifact):
        return {
            "artifact_id": value.artifact_id,
            "kind": value.kind.value,
            "name": value.name,
            "summary": value.summary,
            "usage": value.usage,
            "round_index": value.round_index,
            "agent_id": value.agent_id,
            "parent_artifact_id": value.parent_artifact_id,
            "metadata": _jsonable(value.metadata),
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
            if field.name != "program"
        }
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


def _section(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    result = value.get(name)
    if not isinstance(result, Mapping):
        raise TypeError(f"{name} must be an object")
    return result


def _mapping_sequence(value: Mapping[str, object], name: str) -> tuple[Mapping[str, object], ...]:
    result = value.get(name)
    if isinstance(result, (str, bytes)) or not isinstance(result, Sequence):
        raise TypeError(f"{name} must be an array")
    rows = tuple(result)
    if any(not isinstance(row, Mapping) for row in rows):
        raise TypeError(f"{name} must contain objects")
    return rows


def _string_sequence(
    value: Mapping[str, object],
    name: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    result = value.get(name)
    if isinstance(result, (str, bytes)) or not isinstance(result, Sequence):
        raise TypeError(f"{name} must be an array of strings")
    rows = tuple(result)
    if (not rows and not allow_empty) or any(
        not isinstance(row, str) or not row.strip() for row in rows
    ):
        raise ValueError(f"{name} must contain nonempty strings")
    return rows


def _number_sequence(
    value: Mapping[str, object],
    name: str,
    *,
    length: int,
) -> tuple[float, ...]:
    result = value.get(name)
    if isinstance(result, (str, bytes)) or not isinstance(result, Sequence):
        raise TypeError(f"{name} must be a numeric array")
    rows = tuple(result)
    if len(rows) != length:
        raise ValueError(f"{name} must contain {length} values")
    return tuple(_finite_number(item, name) for item in rows)


def _text(value: Mapping[str, object], name: str) -> str:
    result = value.get(name)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"{name} must be nonempty text")
    return result.strip()


def _integer(value: Mapping[str, object], name: str, *, minimum: int) -> int:
    result = value.get(name)
    if isinstance(result, bool) or not isinstance(result, int) or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _boolean(value: Mapping[str, object], name: str) -> bool:
    result = value.get(name)
    if not isinstance(result, bool):
        raise TypeError(f"{name} must be boolean")
    return result


def _number(
    value: Mapping[str, object],
    name: str,
    *,
    minimum: float,
    strict: bool = False,
) -> float:
    result = _finite_number(value.get(name), name)
    if result < minimum or (strict and result == minimum):
        relation = ">" if strict else ">="
        raise ValueError(f"{name} must be {relation} {minimum}")
    return result


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _config_path(config_path: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _required_regular_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is absent: {resolved}")
    return resolved


def _required_executable(path: Path, label: str) -> Path:
    resolved = _required_regular_file(path, label)
    if not os.access(resolved, os.X_OK):
        raise PermissionError(f"{label} is not executable: {resolved}")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
