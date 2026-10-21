#!/usr/bin/env python3
"""Historical strong VPFP diagnostic composition, not the submission defaults.

The archived one-Agent recipe uses legacy active-time accounting and pure-BASE
promotion. Real execution requires explicit --allow-historical-profile; the
latest submission profile must be frozen separately, without borrowing these
numbers. This still uses the shared CampaignController, not another lifecycle.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields, is_dataclass
from enum import Enum
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import RLock
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pde_agent_verifier_solver.engine import (
    AgentSpec, CampaignRecipe, CaseCounts, CommandAgentProgram,
    DevelopmentSamplingPolicy, EvaluatorPolicy, EventStream, GeneratorRecipe,
    PythonCandidateCodec, PythonTaskWorkerSpec, RoundSettings,
    SubprocessAgentExecutionService, SubprocessCandidateExecutionService,
)
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, DevelopmentCoordinateMode, HiddenSuiteSpec, OfficialEvaluationSpec,
    SampleSource, SamplingStrategy, TaskKind, TerminalCohortSpec,
    TerminalEvaluationSpec, TerminalGuidanceSource, TerminalMetricSpec,
    inspect_candidate_bundle,
)

FIELDS = ("rho", "momentum_x", "momentum_y", "kinetic_energy")
SCHEMA = "canonical_vpfp_strong_iteration_v1"


def file_sha(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if is_dataclass(value):
        return {field.name: jsonable(getattr(value, field.name))
                for field in fields(value) if field.name not in ("program", "guide", "sampler")}
    from collections.abc import Mapping
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


class Writer:
    def __init__(self, root):
        self.root, self.lock = Path(root), RLock()

    def write(self, name, value):
        with self.lock:
            path = self.root / name
            temporary = path.with_name(path.name + ".partial")
            temporary.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
            temporary.chmod(0o600)
            temporary.replace(path)

    def event(self, event):
        with self.lock:
            with (self.root / "events.jsonl").open("a") as handle:
                handle.write(json.dumps(jsonable(event), sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
            self.write("status.json", {"status": "running", "last_event": event.event_type,
                "round_index": event.round_index, "message": event.message, "updated_at_unix": time.time()})
            if event.event_type in ("guidance_ready", "round_started", "round_finished", "campaign_completed"):
                print(json.dumps({"event": event.event_type, "round": event.round_index}), flush=True)


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config.get("schema") != SCHEMA:
        raise ValueError("wrong canonical strong VPFP config schema")
    guide = config["backend"]["guidance_config"]
    if config["official"] != {"base": guide["fit_count"] + guide["validation_count"]}:
        raise ValueError("Official must be clean BASE with exactly the current fit+dev feedback count")
    if (guide["fit_count"], guide["validation_count"], guide["steps"]) != (64, 16, 2000):
        raise ValueError("historical VPFP uses the validated 64+16, 2000-update training recipe")
    if config["round_count"] < 2 or len(config["agents"]) != 1:
        raise ValueError("this resource profile requires at least two rounds and one Agent")
    if config["task"].get("require_prepared_solution") is not True:
        raise ValueError("formal VPFP requires query-blind preparation")
    if config["backend"]["device"] != "cuda":
        raise ValueError("formal IC40 generation/training uses the Controller GPU")
    return config


def recipe(config, initial):
    terminal = config["terminal"]
    cohorts = [TerminalCohortSpec(cohort_id="base_model",
        suite=HiddenSuiteSpec(base=terminal["base"], base_perturbed=terminal["base_perturbed"]),
        guidance_source=TerminalGuidanceSource.BASE)]
    for index in range(1, config["round_count"] + 1):
        cohorts.append(TerminalCohortSpec(cohort_id=f"guide_round_{index}",
            suite=HiddenSuiteSpec(guided=terminal["guided_per_round"]),
            guidance_source=TerminalGuidanceSource.ROUND, round_index=index))
    return CampaignRecipe(
        name=config["campaign_id"], round_count=config["round_count"],
        agents=tuple(AgentSpec(**row) for row in config["agents"]),
        default_round=RoundSettings(
            agent_budget_seconds=config["execution"]["agent_active_budget_seconds"],
            development_sampling=DevelopmentSamplingPolicy(strategy=SamplingStrategy.AGENT_ADAPTIVE,
                max_cases_per_agent_per_round=config["development"]["max_cases_per_agent_per_round"],
                allowed_sources=(SampleSource.BASE, SampleSource.GUIDED),
                starter=CaseCounts(base=config["development"]["starter_base"]),
                hide_case_types=True, cumulative_fixed_cases=False),
            official_evaluation=OfficialEvaluationSpec(round=HiddenSuiteSpec(**config["official"]))),
        generator=GeneratorRecipe(guidance_enabled=True, perturbations_enabled=True, settings={}),
        evaluator=EvaluatorPolicy(default_query=AnswerQuery.at_points(fields=FIELDS,
                points=({"t": .5, "x": 0., "y": 0.},)),
            development_coordinate_mode=DevelopmentCoordinateMode.PRIVATE_RANDOM_POINTS,
            development_point_count=config["task"]["development_point_count"],
            allow_grid_truth=False, allow_point_truth=False, allow_development_solution_session=False),
        task_settings=dict(config["task"]), master_seed=config["master_seed"], initial_candidate=initial,
        promotion={"preset": "minimize_metric", "settings": {
            "primary": {"distribution": "base", "statistic": "mean"}}},
        terminal_evaluation=TerminalEvaluationSpec(cohorts=tuple(cohorts),
            ranking=TerminalMetricSpec(cohort_id="base_model", distribution="base", statistic="mean")),
        terminal_score_distribution="base",
    )


def compose(config, config_path, root, *, events, codex_binary, auth_file, worker_gpu, controller_gpu):
    from pde_agent_verifier_solver.plugins.vpfp.iteration_backend import build_vpfp_strong_iteration_binding
    binding = build_vpfp_strong_iteration_binding(**config["backend"], output_root=root / "private_backend")
    codec = PythonCandidateCodec(task_kind=TaskKind.OBSERVABLE_MODEL)
    initial_path = (config_path.parent / config["candidate"]["initial_bundle"]).resolve()
    initial = codec.bind_existing(initial_path, storage_root=root / "controller_candidates",
        artifact_id="vpfp-strong-s0", name=config["candidate"]["initial_name"],
        summary=config["candidate"]["initial_summary"], usage="prepare_solution then evaluate points")
    plan = recipe(config, initial)
    execution_config = config["execution"]
    runtime = root / "agent_runtime"
    runtime.mkdir(mode=0o700)
    command = (sys.executable, str((config_path.parent / config["worker"]["script"]).resolve()),
        "--codex-binary", str(codex_binary), "--auth-file", str(auth_file),
        "--runtime-root", str(runtime), "--config",
        str((config_path.parent / config["worker"]["codex_config"]).resolve()))
    programs = {row["agent_id"]: CommandAgentProgram(command, codec.decode_nomination,
        incumbent_encoder=codec.encode_incumbent,
        environment={"PYTHONPATH": str(ROOT / "src"),
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", os.defpath),
            "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
        worker_settings={"profile": "vpfp_strong_online_round_guidance",
            "model": execution_config["model"], "reasoning_effort": execution_config["reasoning_effort"],
            "transient_resume_until_deadline": execution_config.get("transient_resume_until_deadline", True),
            "candidate_abi": "prepare_solution_then_evaluate_points_v1",
            "candidate_execution": {"device": "cpu", "cuda_available": False},
            "controller_rpc_timeout_seconds": execution_config["controller_rpc_timeout_seconds"]},
        gpu_devices=(worker_gpu,)) for row in config["agents"]}
    agents = SubprocessAgentExecutionService(root / "agent_turns",
        hard_wall_seconds=execution_config["agent_hard_wall_seconds"],
        enforce_active_budget=True,  # Explicit legacy diagnostic, not the submission default.
        controller_gpu_devices=(controller_gpu,), poll_interval_seconds=.25)
    task = binding.plugin.task("observable_model")
    candidates = SubprocessCandidateExecutionService(root / "candidate_execution",
        task=PythonTaskWorkerSpec.from_task_type(type(task)),
        evaluation_settings={**plan.task_settings, "default_query": plan.evaluator.default_query},
        max_workers=execution_config["candidate_case_workers"],
        case_hard_wall_seconds=execution_config["candidate_case_hard_wall_seconds"],
        development_batch_hard_wall_seconds=execution_config["development_candidate_batch_hard_wall_seconds"],
        python_executable=sys.executable)
    controller = binding.controller(plan, programs, agent_execution=agents,
        candidate_execution=candidates, events=events, task_id="observable_model")
    return controller, agents, candidates, binding


def persist_guidance(binding, writer):
    if binding is None:
        return
    backend = binding.plugin.generator.backend
    writer.write("guidance_audits.json", {"training": backend.training_audits,
        "sampling": backend.sampling_audits, "retained_draw_count": len(backend._draws),
        "role": "per_round_training_recipe_not_pretrained_guidance_asset"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vpfp_strong_canonical_2round.json")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--codex-binary", type=Path)
    parser.add_argument("--auth-file", type=Path)
    parser.add_argument("--worker-gpu", default=os.environ.get("PDE_AGENT_WORKER_GPU_DEVICES", ""))
    parser.add_argument("--controller-gpu", default=os.environ.get("PDE_CONTROLLER_GPU_DEVICE", ""))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-historical-profile", action="store_true",
        help="Explicitly run the archived diagnostic; not the current submission protocol")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    initial_path = (config_path.parent / config["candidate"]["initial_bundle"]).resolve()
    bundle = inspect_candidate_bundle(initial_path)
    recipe(config, None)
    if args.validate_only:
        print(json.dumps({"status": "static_configuration_validated", "initial_bundle_sha256": bundle.tree_sha256,
            "assets_loaded": False, "model_called": False}), flush=True)
        return 0
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_TMPDIR"):
        parser.error("real assets and Agent execution require a compute allocation")
    if not args.allow_historical_profile:
        parser.error("this is a historical diagnostic, not the default submission profile; "
                     "explicit --allow-historical-profile is required")
    if args.run_root is None or args.codex_binary is None or args.auth_file is None:
        parser.error("execution requires --run-root, --codex-binary, --auth-file")
    if (not args.worker_gpu or not args.controller_gpu or args.worker_gpu == args.controller_gpu
            or "," in args.worker_gpu or "," in args.controller_gpu):
        raise ValueError("use distinct single worker and Controller GPU tokens")
    if args.auth_file.is_symlink() or not args.auth_file.is_file():
        raise ValueError("auth source must be a regular nonsymlink file")
    if not args.codex_binary.is_file() or not os.access(args.codex_binary, os.X_OK):
        raise ValueError("Agent executable unavailable")
    status = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"], text=True).strip()
    if status:
        raise ValueError("formal campaign requires a clean committed source checkout")
    root = args.run_root.resolve()
    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    writer = Writer(root)
    binding = agents = candidates = None
    try:
        writer.write("resolved_config.json", {**config, "config_sha256": file_sha(config_path),
            "source": str(ROOT), "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "worker_gpu": args.worker_gpu, "controller_gpu": args.controller_gpu})
        events = EventStream()
        events.subscribe(writer.event)
        controller, agents, candidates, binding = compose(config, config_path, root, events=events,
            codex_binary=args.codex_binary.resolve(), auth_file=args.auth_file.resolve(),
            worker_gpu=args.worker_gpu, controller_gpu=args.controller_gpu)
        writer.write("input_provenance.json", dict(binding.report))
        writer.write("compiled_campaign.json", controller.campaign)
        events.subscribe(lambda event: persist_guidance(binding, writer)
            if event.event_type in ("guidance_ready", "round_finished") else None)
        result = controller.run()
        writer.write("result.json", result)
        writer.write("candidate_history.json", result.candidate_history)
        (root / "CAMPAIGN_MEMORY.md").write_text(result.memory.markdown + "\n")
        selected = inspect_candidate_bundle(result.selected.artifact.program.bundle_locator,
                                            recorded_identity=result.selected.artifact.program.tree_sha256)
        shutil.copytree(selected.root, root / "selected_candidate", symlinks=False)
        writer.write("status.json", {"status": "completed", "rounds_completed": len(result.rounds),
            "selected_artifact_id": result.selected.artifact.artifact_id,
            "best_terminal_round": result.best_terminal_round,
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
        writer.write("execution_audits.json", () if agents is None else agents.audits)
        writer.write("candidate_execution_audits.json", () if candidates is None else candidates.audits)
        persist_guidance(binding, writer)


if __name__ == "__main__":
    raise SystemExit(main())
