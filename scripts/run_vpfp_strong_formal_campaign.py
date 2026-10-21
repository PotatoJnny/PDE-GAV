#!/usr/bin/env python3
"""User-approved strong VPFP five-round composition of the canonical Controller.

The scientific JSON is immutable. Only the measured Candidate worker count is
resolved from the mandatory numerical-equivalence preflight.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Event, Thread
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_canonical_vpfp_campaign import FIELDS, Writer, file_sha, jsonable, persist_guidance
from pde_agent_verifier_solver.engine import (
    AgentSpec, CampaignRecipe, CaseCounts, CommandAgentProgram,
    DevelopmentSamplingPolicy, EvaluatorPolicy, EventStream, GeneratorRecipe,
    PythonCandidateCodec, PythonTaskWorkerSpec, RoundSettings,
    SubprocessAgentExecutionService, SubprocessCandidateExecutionService,
)
from pde_agent_verifier_solver.engine.campaign import CampaignCompiler
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, Artifact, DevelopmentCoordinateMode, GeneratorCapabilities, GeneratorCheckpoint,
    HiddenSuiteSpec, OfficialEvaluationSpec,
    SampleSource, SamplingStrategy, TaskKind, TerminalCohortSpec,
    TerminalEvaluationSpec, TerminalGuidanceSource, TerminalMetricSpec,
    inspect_candidate_bundle,
)

SCHEMA = "canonical_vpfp_strong_formal_v1"
OFFICIAL = {"base": 342, "guided": 341, "base_perturbed": 341, "guided_perturbed": 0}
TERMINAL = [
    {"cohort_id": "common_base", "guidance_source": "base",
     "suite": {"base": 256, "base_perturbed": 128}},
    {"cohort_id": "common_guided", "guidance_source": "round_guided", "round_index": 1,
     "suite": {"guided": 256, "guided_perturbed": 128}},
]
POLICY = "first_80_clean_base_feedback_order"
S0_SHA = "d97d86d7a76f1683595cedb82bd957379ed8d5bc3ff779b82f1bbbad93615f16"


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config.get("schema") != SCHEMA:
        raise ValueError("wrong formal VPFP schema; historical recipes are not formal defaults")
    from pde_agent_verifier_solver.engine.gpu_execution_config import (
        gpu_execution_settings, validate_gpu_run_budget,
    )
    gpu_enabled = gpu_execution_settings(config["execution"]) is not None
    validate_gpu_run_budget(config)
    expected = {
        "master_seed": 101, "round_count": 5,
        "official": OFFICIAL,
        "terminal": {"cohorts": TERMINAL, "ranking": {
            "cohort_id": "overall", "distribution": "overall", "statistic": "mean"}},
        "promotion": {"preset": "minimize_metric", "settings": {
            "primary": {"distribution": "overall", "statistic": "mean"}}},
        "development": {"max_cases_per_agent_per_round": 32, "starter_base": 4},
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError("user-approved formal setting changed: " + key)
    execution = config["execution"]
    for key, expected_value in {
        "agent_hard_wall_seconds": 14400, "model": "gpt-5.6-sol", "reasoning_effort": "max",
        "controller_rpc_timeout_seconds": 7200, "candidate_case_hard_wall_seconds": 900,
        "development_candidate_batch_hard_wall_seconds": 6900,
        "candidate_case_workers": 16, "candidate_worker_options": [4, 8, 16],
        "candidate_worker_selection": "minimum_elapsed_among_all_numerically_verified_preflight_options",
    }.items():
        if execution.get(key) != expected_value:
            raise ValueError("user-approved execution setting changed: " + key)
    if "agent_active_budget_seconds" in execution:
        raise ValueError("formal VPFP uses total wall time, never legacy active-time")
    agents = config["agents"]
    if len(agents) != 2 or len({row["agent_id"] for row in agents}) != 2:
        raise ValueError("formal VPFP requires two distinct competing Agents")
    if any(not row.get("prompt", "").strip() for row in agents):
        raise ValueError("each Agent requires an explicit VPFP prompt")
    task_expected = {
        "require_prepared_solution": True,
        "development_point_count": 128, "official_point_count": 256, "terminal_point_count": 256,
        **({"candidate_budget_mode": "ns_remaining_case_v1"} if gpu_enabled
           else {"candidate_compute_budget": 180}),
    }
    if not gpu_enabled and config["task"].get("candidate_budget_mode") is not None:
        raise ValueError("the shared Case budget mode requires explicit shared_gpu_v1")
    for key, expected_value in task_expected.items():
        if config["task"].get(key) != expected_value:
            raise ValueError("approved VPFP scoring setting changed: " + key)
    expected_guide = dict(steps=2000, batch_size=4, learning_rate=2.e-4, beta_std=.5,
                          fit_count=64, validation_count=16, target_ratio=.1,
                          max_flow_ratio=.25, wall_seconds=1800)
    if config["backend"]["guidance_config"] != expected_guide:
        raise ValueError("preserve the approved 64+16/2000-step VPFP Guidance recipe")
    if config["backend"].get("feedback_selection") != POLICY or config["backend"]["device"] != "cuda":
        raise ValueError("formal feedback selection or Controller device changed")
    if config["candidate"].get("initial_bundle_sha256") != S0_SHA:
        raise ValueError("common S0 bundle identity changed")
    return config


def recipe(config, initial):
    cohorts = tuple(TerminalCohortSpec(
        cohort_id=row["cohort_id"], suite=HiddenSuiteSpec(**row["suite"]),
        guidance_source=TerminalGuidanceSource(row["guidance_source"]),
        round_index=row.get("round_index"),
    ) for row in config["terminal"]["cohorts"])
    return CampaignRecipe(
        name=config["campaign_id"], round_count=config["round_count"],
        agents=tuple(AgentSpec(**row) for row in config["agents"]),
        default_round=RoundSettings(
            agent_budget_seconds=config["execution"]["agent_hard_wall_seconds"],
            development_sampling=DevelopmentSamplingPolicy(
                strategy=SamplingStrategy.AGENT_ADAPTIVE,
                max_cases_per_agent_per_round=config["development"]["max_cases_per_agent_per_round"],
                allowed_sources=(SampleSource.BASE, SampleSource.GUIDED),
                starter=CaseCounts(base=config["development"]["starter_base"]),
                hide_case_types=True, cumulative_fixed_cases=False),
            official_evaluation=OfficialEvaluationSpec(round=HiddenSuiteSpec(**config["official"]))),
        generator=GeneratorRecipe(guidance_enabled=True, perturbations_enabled=True, settings={}),
        evaluator=EvaluatorPolicy(
            default_query=AnswerQuery.at_points(fields=FIELDS, points=({"t": .5, "x": 0., "y": 0.},)),
            development_coordinate_mode=DevelopmentCoordinateMode.PRIVATE_RANDOM_POINTS,
            development_point_count=config["task"]["development_point_count"],
            allow_grid_truth=False, allow_point_truth=False, allow_development_solution_session=False),
        task_settings=dict(config["task"]), master_seed=config["master_seed"], initial_candidate=initial,
        promotion=config["promotion"],
        terminal_evaluation=TerminalEvaluationSpec(
            cohorts=cohorts, ranking=TerminalMetricSpec(**config["terminal"]["ranking"])),
        terminal_score_distribution="overall",
    )


def gpu_mapping(config, worker_gpus, controller_gpu):
    tokens = tuple(worker_gpus)
    if (len(tokens) != 2 or len(set(tokens)) != 2 or not controller_gpu
            or controller_gpu in tokens
            or any(not value or "," in value for value in (*tokens, controller_gpu))):
        raise ValueError("two distinct Agent GPU tokens and a separate Controller GPU are required")
    return dict(zip((row["agent_id"] for row in config["agents"]), tokens, strict=True))


def validate_preflight(path, config_path, revision):
    report = json.loads(Path(path).read_text())
    for key in ("preflight_passed", "execution_passed", "source_final_verified",
                "config_final_verified", "solver_final_verified", "base_final_verified"):
        if report.get(key) is not True:
            raise ValueError("formal launch requires passed preflight: " + key)
    selected = report.get("selected_worker_count")
    if type(selected) is not int or selected not in (4, 8, 16):
        raise ValueError("preflight must resolve one approved Candidate worker count")
    return selected, report


def build_agent_programs(config, config_path, root, codec, *, codex_binary, auth_file, mapping):
    from pde_agent_verifier_solver.engine.gpu_execution_config import (
        candidate_execution_capabilities, agent_gpu_worker_settings,
    )
    runtime = root / "agent_runtime"
    runtime.mkdir(mode=0o700)
    command = (
        sys.executable, str((config_path.parent / config["worker"]["script"]).resolve()),
        "--codex-binary", str(codex_binary), "--auth-file", str(auth_file),
        "--runtime-root", str(runtime), "--config",
        str((config_path.parent / config["worker"]["codex_config"]).resolve()),
    )
    execution = config["execution"]
    return {row["agent_id"]: CommandAgentProgram(
        command, codec.decode_nomination, incumbent_encoder=codec.encode_incumbent,
        environment={"PYTHONPATH": str(ROOT / "src"),
                     "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", os.defpath),
                     "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
        worker_settings={
            "profile": "vpfp_strong_formal_5round_full",
            "model": execution["model"], "reasoning_effort": execution["reasoning_effort"],
            "candidate_abi": "prepare_solution_then_evaluate_points_v1",
            "candidate_execution": candidate_execution_capabilities(execution),
            **agent_gpu_worker_settings(execution),
            "controller_rpc_timeout_seconds": execution["controller_rpc_timeout_seconds"],
        },
        gpu_devices=(mapping[row["agent_id"]],),
    ) for row in config["agents"]}


def compose(config, config_path, root, *, events, codex_binary, auth_file,
            worker_gpus, controller_gpu, candidate_workers, evaluation_device_map=None):
    from pde_agent_verifier_solver.engine.gpu_execution_config import candidate_gpu_kwargs, validate_gpu_run_budget
    validate_gpu_run_budget(config)
    gpu_options = candidate_gpu_kwargs(
        config["execution"], reserved_devices=(controller_gpu, *worker_gpus),
    )
    from pde_agent_verifier_solver.plugins.vpfp.iteration_backend import build_vpfp_strong_iteration_binding
    mapping = gpu_mapping(config, worker_gpus, controller_gpu)
    binding = build_vpfp_strong_iteration_binding(**config["backend"], output_root=root / "private_backend")
    codec = PythonCandidateCodec(task_kind=TaskKind.OBSERVABLE_MODEL)
    initial_path = (config_path.parent / config["candidate"]["initial_bundle"]).resolve()
    if inspect_candidate_bundle(initial_path).tree_sha256 != S0_SHA:
        raise ValueError("initial S0 source differs from the approved frozen bundle")
    initial = codec.bind_existing(initial_path, storage_root=root / "controller_candidates",
        artifact_id="vpfp-strong-s0", name=config["candidate"]["initial_name"],
        summary=config["candidate"]["initial_summary"], usage="prepare_solution then evaluate points")
    plan = recipe(config, initial)
    programs = build_agent_programs(config, config_path, root, codec, codex_binary=codex_binary,
                                    auth_file=auth_file, mapping=mapping)
    agents = SubprocessAgentExecutionService(
        root / "agent_turns", hard_wall_seconds=config["execution"]["agent_hard_wall_seconds"],
        enforce_active_budget=False, controller_gpu_devices=(controller_gpu,),
        evaluation_device_map=evaluation_device_map, poll_interval_seconds=.25)
    task = binding.plugin.task("observable_model")
    candidates = SubprocessCandidateExecutionService(
        root / "candidate_execution", task=PythonTaskWorkerSpec.from_task_type(type(task)),
        evaluation_settings={**plan.task_settings, "default_query": plan.evaluator.default_query},
        max_workers=candidate_workers,
        # Replicated evaluation sessions are independent across GPUs. Ordinary
        # development sessions still hold their existing Generator/runtime lock.
        oracle_max_workers=min(128, candidate_workers) if evaluation_device_map else 1,
        case_hard_wall_seconds=config["execution"]["candidate_case_hard_wall_seconds"],
        development_batch_hard_wall_seconds=config["execution"]["development_candidate_batch_hard_wall_seconds"],
        python_executable=sys.executable, **gpu_options)
    controller = binding.controller(plan, programs, agent_execution=agents,
        candidate_execution=candidates, events=events, task_id="observable_model")
    return controller, agents, candidates, binding


def source_identity():
    dirty = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"], text=True)
    if dirty.strip():
        raise ValueError("formal execution requires a clean committed source checkout")
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vpfp_strong_formal_5round_s101.json")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--codex-binary", type=Path)
    parser.add_argument("--auth-file", type=Path)
    parser.add_argument("--worker-gpus", default=os.environ.get("PDE_AGENT_WORKER_GPU_DEVICES", ""))
    parser.add_argument("--controller-gpu", default=os.environ.get("PDE_CONTROLLER_GPU_DEVICE", ""))
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    plan = recipe(config, None)
    bundle = inspect_candidate_bundle((config_path.parent / config["candidate"]["initial_bundle"]).resolve())
    if bundle.tree_sha256 != S0_SHA:
        raise ValueError("approved common S0 bundle has changed")
    if args.validate_only:
        # Compile using a typed, non-executable S0/checkpoint descriptor.
        # Actual bundle and physical checkpoint binding still happen in compose.
        descriptor = Artifact("static-s0", TaskKind.OBSERVABLE_MODEL, "Static common S0",
                              None, "Bundle identity checked", "Not executable", 0, "configuration")
        static_plan = replace(plan, initial_candidate=descriptor,
            generator=replace(plan.generator, checkpoint=GeneratorCheckpoint("static-vpfp", {})))
        CampaignCompiler().compile(static_plan, GeneratorCapabilities(
            trainable=False, offline=False, online=True, guidance=True, perturbations=True))
        print(json.dumps({"status": "static_configuration_validated", "round_count": plan.round_count,
            "agent_count": len(plan.agents), "official_count": 1024, "terminal_count": 768,
            "model_called": False, "assets_loaded": False, "s0_sha256": bundle.tree_sha256}), flush=True)
        return 0
    if not os.environ.get("SLURM_JOB_ID", "").isdigit() or not os.environ.get("SLURM_TMPDIR"):
        parser.error("formal Agent execution requires a Slurm compute allocation")
    if any(value is None for value in (args.run_root, args.codex_binary, args.auth_file, args.preflight_report)):
        parser.error("execution requires run root, Codex executable, auth source and passed preflight report")
    mapping = gpu_mapping(config, tuple(args.worker_gpus.split(",")), args.controller_gpu)
    from pde_agent_verifier_solver.engine.controller_gpu_visibility import configure_controller_evaluation_devices
    evaluation_device_map = (configure_controller_evaluation_devices(args.controller_gpu, tuple(args.worker_gpus.split(",")))
                             if os.environ.get("CUDA_VISIBLE_DEVICES") else {})
    if args.auth_file.is_symlink() or not args.auth_file.is_file():
        raise ValueError("auth source must be a regular nonsymlink file")
    if not args.codex_binary.is_file() or not os.access(args.codex_binary, os.X_OK):
        raise ValueError("Agent executable is unavailable")
    revision = source_identity()
    config_sha = file_sha(config_path)
    candidate_workers, preflight = validate_preflight(args.preflight_report, config_path, revision)
    root = args.run_root.resolve()
    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    writer = Writer(root)
    binding = agents = candidates = controller = None
    stop = Event()
    heartbeat = None
    terminal_manifest = None
    try:
        writer.write("resolved_config.json", {**config, "config_sha256": config_sha,
            "source": str(ROOT), "commit": revision, "worker_gpu_mapping": mapping,
            "controller_gpu": args.controller_gpu, "controller_evaluation_device_map": evaluation_device_map,
            "resolved_candidate_case_workers": candidate_workers,
            "preflight_report_sha256": file_sha(args.preflight_report)})
        writer.write("preflight_receipt.json", preflight)
        events = EventStream()
        events.subscribe(writer.event)
        controller, agents, candidates, binding = compose(
            config, config_path, root, events=events, codex_binary=args.codex_binary.resolve(),
            auth_file=args.auth_file.resolve(), worker_gpus=tuple(args.worker_gpus.split(",")),
            controller_gpu=args.controller_gpu, candidate_workers=candidate_workers,
            evaluation_device_map=evaluation_device_map)
        writer.write("input_provenance.json", dict(binding.report))
        writer.write("compiled_campaign.json", controller.campaign)

        def observe(event):
            nonlocal terminal_manifest
            if event.event_type in ("guidance_ready", "round_finished"):
                persist_guidance(binding, writer)
                writer.write("execution_audits.json", agents.audits)
                writer.write("candidate_execution_audits.json", candidates.audits)
            if (event.event_type == "terminal_cohort_sealed"
                    and event.payload.get("cohort_id") == "common_guided"):
                from pde_agent_verifier_solver.plugins.vpfp.formal_evaluation_bank import export_terminal_bank
                terminal_manifest = export_terminal_bank(
                    controller, binding, root / "private_backend" / "common_terminal",
                    initial_solver_sha256=S0_SHA)
                writer.write("terminal_bank_receipt.json", terminal_manifest)
            if event.event_type == "agent_started" and terminal_manifest is None:
                raise RuntimeError("common S0 terminal bank must be frozen before Agent development")

        events.subscribe(observe)

        def progress():
            emitted = 0
            while not stop.wait(30):
                audits = candidates.audits
                if len(audits) > emitted:
                    with (root / "candidate_execution_audits.jsonl").open("a") as handle:
                        for row in audits[emitted:]:
                            handle.write(json.dumps(jsonable(row), sort_keys=True, allow_nan=False) + "\n")
                    emitted = len(audits)
                value = {"updated_at_unix": time.time(), "candidate_evaluations": len(audits),
                    "candidate_status_counts": dict(Counter(row.status for row in audits)),
                    "agent_turns_finished": len(agents.audits),
                    "guidance_rounds_trained": len(binding.plugin.generator.backend.training_audits),
                    "terminal_bank_frozen": terminal_manifest is not None}
                writer.write("progress.json", value)
                print(json.dumps({"heartbeat": value}), flush=True)

        heartbeat = Thread(target=progress, daemon=True)
        heartbeat.start()
        from run_plasma_canonical_smoke import _forbid_online_vpfp_generation
        guard = {}
        with _forbid_online_vpfp_generation(guard):
            result = controller.run()
        if len(result.rounds) != 5 or terminal_manifest is None:
            raise RuntimeError("formal run did not complete all approved rounds and terminal bank")
        if len(binding.plugin.generator.backend.training_audits) != 5:
            raise RuntimeError("formal run must train exactly G1 through G5")
        writer.write("online_coarse_solver_guard.json", guard)
        writer.write("result.json", result)
        writer.write("candidate_history.json", result.candidate_history)
        (root / "CAMPAIGN_MEMORY.md").write_text(result.memory.markdown + "\n")
        selected = inspect_candidate_bundle(result.selected.artifact.program.bundle_locator)
        if selected.tree_sha256 != result.selected.artifact.program.tree_sha256:
            raise RuntimeError("selected program changed after Controller freeze")
        shutil.copytree(selected.root, root / "selected_candidate", symlinks=False)
        if inspect_candidate_bundle(root / "selected_candidate").tree_sha256 != selected.tree_sha256:
            raise RuntimeError("delivered program differs from selected snapshot")
        writer.write("status.json", {"status": "completed", "rounds_completed": len(result.rounds),
            "selected_artifact_id": result.selected.artifact.artifact_id,
            "selected_bundle_sha256": selected.tree_sha256, "best_terminal_round": result.best_terminal_round,
            "agent_execution_status_counts": dict(Counter(row.status for row in agents.audits)),
            "nomination_count": sum(row.nomination_count for row in agents.audits),
            "finished_at_unix": time.time(), "delivers": "last_round_selected_not_terminal_best"})
        return 0
    except BaseException as error:
        (root / "failure_traceback.txt").write_text(traceback.format_exc())
        writer.write("status.json", {"status": "failed", "error_type": type(error).__name__,
            "error": str(error), "finished_at_unix": time.time()})
        raise
    finally:
        stop.set()
        if heartbeat is not None:
            heartbeat.join(timeout=5)
        writer.write("execution_audits.json", () if agents is None else agents.audits)
        writer.write("candidate_execution_audits.json", () if candidates is None else candidates.audits)
        persist_guidance(binding, writer)


if __name__ == "__main__":
    raise SystemExit(main())
