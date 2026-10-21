#!/usr/bin/env python3
"""Compose one frozen Base+Guidance+Perturbation Artemis canonical experiment.

No scientific lifecycle is implemented here: the admitted backend binding
creates the sole CampaignController, and the existing Codex worker and Python
Candidate codec execute every Agent turn. --validate-only is a CPU-only static
configuration check; --preflight-only loads real assets but starts no Agent.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from threading import Lock
import time
import traceback
from typing import Mapping, Sequence
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from pde_agent_verifier_solver.engine import (
    AgentSpec, CampaignRecipe, CaseCounts, CommandAgentProgram,
    DevelopmentSamplingPolicy, EvaluatorPolicy, EventStream, GeneratorRecipe,
    PythonCandidateCodec, RoundSettings, SubprocessAgentExecutionService,
    PythonTaskWorkerSpec, SubprocessCandidateExecutionService,
)
from pde_agent_verifier_solver.engine.memory import CAMPAIGN_MEMORY_HEADER
from pde_agent_verifier_solver.sdk import (
    AnswerQuery, Artifact, DevelopmentCoordinateMode, HiddenSuiteSpec,
    OfficialEvaluationSpec, SampleSource, SamplingStrategy, TaskKind,
    TerminalCohortSpec, TerminalEvaluationSpec, TerminalGuidanceSource,
    TerminalMetricSpec, inspect_candidate_bundle,
)

CONFIG_SCHEMA = "canonical_artemis_campaign_v1"
# Artemis-only cold deployment pin; the canonical Controller is unmodified.
BACKEND_CONFIG_SHA256 = "0433ef1edefce3c49dc0f085ec0f886c467e350eddf9d5c187e6ca8eda1b3651"
MODEL, REASONING_EFFORT = "gpt-5.6-sol", "max"
ROUND_COUNT, AGENT_COUNT = 5, 2
ACTIVE_SECONDS, HARD_WALL_SECONDS, RPC_TIMEOUT_SECONDS = 7200, 14400, 7200
CANDIDATE_WORKERS, CANDIDATE_WALL_SECONDS, DEVELOPMENT_BATCH_SECONDS = 28, 900, 6900
RECOVERY_SOURCE_HEAD = "da3cb03449dfd725cc72a145e591c0776f3f44fa"
RECOVERY_RESOLVED_CONFIG_SHA256 = "61d4a99f99fa67caef04e99fe85dd93a39c4c243c631b7ab4ad977e49d9242cd"
UPSTREAM_PROTOCOL_REVISION = "0e8aeade88bff80da79a8ba008700638f32511b2"
_GPU_TOKEN = re.compile(r"^(?:[0-9]+|GPU-[0-9A-Fa-f-]+|MIG-[A-Za-z0-9._:/-]+)$")
_COUNTS = ("original", "base", "guided", "base_perturbed", "guided_perturbed")


def _counts(base: int, perturbed: int = 0) -> dict[str, int]:
    return dict(zip(_COUNTS, (0, base, 0, perturbed, 0), strict=True))


_FROZEN_SECTIONS = {
    "schema": CONFIG_SCHEMA,
    "profile": "artemis_admitted_reward_guided",
    "round_count": ROUND_COUNT,
    "execution": {
        "agent_count": AGENT_COUNT, "agent_active_budget_seconds": ACTIVE_SECONDS,
        "agent_hard_wall_seconds": HARD_WALL_SECONDS,
        "controller_rpc_timeout_seconds": RPC_TIMEOUT_SECONDS,
        "candidate_case_workers": CANDIDATE_WORKERS,
        "candidate_case_hard_wall_seconds": CANDIDATE_WALL_SECONDS,
        "development_candidate_batch_hard_wall_seconds": DEVELOPMENT_BATCH_SECONDS,
        "model": MODEL, "reasoning_effort": REASONING_EFFORT,
    },
    "generator": {
        "guidance_enabled": True, "perturbations_enabled": True,
        "settings": {"guidance_maximum_wall_seconds": 3600.0},
        "feedback_source": "canonical_selected_solver_official",
        "feedback_case_count": 128,
    },
    "development": {
        "strategy": "agent_adaptive", "case_replay": False,
        "max_cases_per_agent_per_round": 256, "allowed_sources": ["base", "guided"],
        "starter": _counts(16),
    },
    "official": _counts(128),
    "terminal": {
        "cohort_id": "base_model", "guidance_source": "base",
        "suite": _counts(256, 256),
        "ranking": {"distribution": "base", "statistic": "mean"},
    },
    "promotion": {
        "preset": "minimize_metric",
        "settings": {"primary": {"distribution": "base", "statistic": "mean"}},
    },
    "evaluator": {
        "development_coordinate_mode": "private_random_points",
        "development_point_count": 16, "official_point_count": 32,
        "terminal_point_count": 32, "allow_grid_truth": False,
        "allow_point_truth": False, "allow_development_solution_session": False,
    },
    "task": {"require_prepared_solution": True, "causal_horizon": True,
             "candidate_compute_budget": 600},
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value.strip()


def _validate_frozen_protocol(config: Mapping[str, object]) -> None:
    from pde_agent_verifier_solver.engine.gpu_execution_config import (
        gpu_execution_settings, legacy_execution_settings, validate_gpu_run_budget,
    )
    expected_keys = set(_FROZEN_SECTIONS) | {"campaign_id", "master_seed", "backend", "candidate", "worker", "agents"}
    if set(config) != expected_keys:
        raise ValueError("formal Artemis configuration has missing or unknown fields")
    gpu_enabled = gpu_execution_settings(config["execution"]) is not None
    validate_gpu_run_budget(config)
    config = dict(config, execution=legacy_execution_settings(config["execution"]))
    frozen = _FROZEN_SECTIONS
    if gpu_enabled:
        # A new explicit execution profile, not a mutation of the historical
        # recipe. All non-budget scientific fields remain frozen below.
        execution = dict(_FROZEN_SECTIONS["execution"])
        execution.pop("agent_active_budget_seconds")
        task = dict(_FROZEN_SECTIONS["task"])
        task.pop("candidate_compute_budget")
        task["candidate_budget_mode"] = "ns_remaining_case_v1"
        frozen = {**_FROZEN_SECTIONS, "execution": execution, "task": task}
    for key, expected in frozen.items():
        if _canonical_json(config[key]) != _canonical_json(expected):
            raise ValueError(f"formal Artemis frozen {key} configuration changed")
    _text(config["campaign_id"], "campaign_id")
    seed = config["master_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("master_seed must be a nonnegative integer")
    backend = config["backend"]
    if not isinstance(backend, dict) or set(backend) - {"sha256"} != {"config"}:
        raise ValueError("backend requires config")
    _text(backend["config"], "backend.config")
    for section, keys in (
        ("candidate", {"initial_bundle", "initial_name", "initial_summary"}),
        ("worker", {"script", "codex_config"}),
    ):
        row = config[section]
        if not isinstance(row, dict) or set(row) != keys:
            raise ValueError(f"invalid {section} settings")
        for key in keys:
            _text(row[key], f"{section}.{key}")
    agents = config["agents"]
    if not isinstance(agents, list) or len(agents) != AGENT_COUNT:
        raise ValueError("formal Artemis requires exactly two Agents")
    for row in agents:
        if not isinstance(row, dict) or set(row) != {"agent_id", "name", "prompt"}:
            raise ValueError("invalid Agent settings")
        for key, value in row.items():
            _text(value, f"Agent.{key}")
    if len({row["agent_id"] for row in agents}) != AGENT_COUNT:
        raise ValueError("Agent IDs must be distinct")


def _path(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else config_path.parent / path).resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> tuple[dict[str, object], Path]:
    source = path.expanduser().resolve()
    config = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration must be a JSON object")
    _validate_frozen_protocol(config)
    return config, source


def _static_inputs(config, config_path: Path) -> dict[str, object]:
    backend_path = _path(config_path, config["backend"]["config"])
    backend = json.loads(backend_path.read_text(encoding="utf-8"))
    if backend.get("physics") != "artemis" or not isinstance(backend.get("factory_settings"), dict):
        raise ValueError("Artemis backend config has an invalid schema")
    settings = backend["factory_settings"]
    if (
        settings.get("device") != "cuda"
        or settings.get("guidance_adapter_factory") is not None
        or not isinstance(settings.get("reward_guidance_settings"), dict)
        or not settings["reward_guidance_settings"]
    ):
        raise ValueError("formal Artemis requires the admitted CUDA backend with frozen reward guidance")
    for key, filename in (("script", "run_canonical_codex_agent.py"), ("codex_config", "canonical_codex.config.toml")):
        path = _path(config_path, config["worker"][key])
        if path != ROOT / "scripts" / filename or not path.is_file():
            raise ValueError(f"formal Artemis must use the canonical worker {filename}")
    candidate_path = _path(config_path, config["candidate"]["initial_bundle"])
    bundle = inspect_candidate_bundle(candidate_path)
    return {
        "backend_config": str(backend_path), "backend_config_sha256": _file_sha256(backend_path),
        "factory_settings": settings, "initial_bundle": str(candidate_path),
        "initial_bundle_tree_sha256": bundle.tree_sha256,
    }


def _gpu_allocation(workers: str, controller: str) -> tuple[tuple[str, str], str]:
    values = tuple(value.strip() for value in workers.split(","))
    controller = controller.strip()
    if len(values) != AGENT_COUNT or any(not _GPU_TOKEN.fullmatch(value) for value in (*values, controller)):
        raise ValueError("formal Artemis needs two worker GPUs and one controller GPU")
    if len(set((*values, controller))) != AGENT_COUNT + 1:
        raise ValueError("formal Artemis GPU tokens must be distinct")
    return (values[0], values[1]), controller


def _execution_inputs(inputs, path):
    """Change only the admitted Backend's execution implementation, not science."""
    if path is None:
        return inputs
    path = Path(path).resolve(strict=True)
    payload = json.loads(path.read_text())
    if payload.get("physics") != "artemis":
        raise ValueError("execution backend must be Artemis")
    settings = dict(payload["factory_settings"])
    original = dict(inputs["factory_settings"])
    implementation_keys = {"physical_checkout", "expected_physical_revision", "oracle_settings"}
    if ({key: value for key, value in settings.items() if key not in implementation_keys}
            != {key: value for key, value in original.items() if key not in implementation_keys}):
        raise ValueError("execution override changes scientific assets or Guidance settings")
    return {**inputs, "factory_settings": settings,
            "execution_backend_config": str(path),
            "execution_backend_config_sha256": _file_sha256(path),
            "original_backend_config_sha256": inputs["backend_config_sha256"]}


def _recipe(config, initial: Artifact) -> CampaignRecipe:
    from pde_agent_verifier_solver.engine.gpu_execution_config import gpu_execution_settings
    gpu_enabled = gpu_execution_settings(config["execution"]) is not None
    terminal = config["terminal"]
    return CampaignRecipe(
        name=config["campaign_id"], round_count=ROUND_COUNT,
        agents=tuple(AgentSpec(**row) for row in config["agents"]),
        default_round=RoundSettings(
            agent_budget_seconds=(HARD_WALL_SECONDS if gpu_enabled else ACTIVE_SECONDS),
            development_sampling=DevelopmentSamplingPolicy(
                strategy=SamplingStrategy.AGENT_ADAPTIVE,
                max_cases_per_agent_per_round=256,
                allowed_sources=(SampleSource.BASE, SampleSource.GUIDED), starter=CaseCounts(base=16),
                hide_case_types=True, cumulative_fixed_cases=False,
            ),
            official_evaluation=OfficialEvaluationSpec(round=HiddenSuiteSpec(**config["official"])),
        ),
        generator=GeneratorRecipe(
            guidance_enabled=True, perturbations_enabled=True,
            settings=dict(config["generator"]["settings"]),
        ),
        evaluator=EvaluatorPolicy(
            default_query=AnswerQuery.at_points(fields=("M0", "M1", "M2", "M3", "E"), points=({"t": 0.5, "x": 0.0},)),
            development_coordinate_mode=DevelopmentCoordinateMode.PRIVATE_RANDOM_POINTS,
            development_point_count=16, allow_grid_truth=False, allow_point_truth=False,
            allow_development_solution_session=False,
        ),
        task_settings={**config["task"], **{name: config["evaluator"][name] for name in ("development_point_count", "official_point_count", "terminal_point_count")}},
        master_seed=config["master_seed"], initial_candidate=initial,
        promotion=config["promotion"],
        terminal_evaluation=TerminalEvaluationSpec(
            cohorts=(TerminalCohortSpec(
                cohort_id=terminal["cohort_id"], suite=HiddenSuiteSpec(**terminal["suite"]),
                guidance_source=TerminalGuidanceSource.BASE,
            ),),
            ranking=TerminalMetricSpec(cohort_id=terminal["cohort_id"], **terminal["ranking"]),
        ),
        terminal_score_distribution="base",
    )


def _compose(*, config, config_path, inputs, run_root, codex_binary, auth_file, worker_gpus, controller_gpu, events,
             candidate_workers=CANDIDATE_WORKERS, reuse_agent_gpus_for_evaluation=False):
    from pde_agent_verifier_solver.engine.gpu_execution_config import (
        candidate_gpu_kwargs, candidate_execution_capabilities, agent_gpu_worker_settings,
        validate_gpu_run_budget,
    )
    validate_gpu_run_budget(config)
    gpu_options = candidate_gpu_kwargs(
        config["execution"], reserved_devices=(controller_gpu, *worker_gpus),
    )
    # The PDE-specific factory is the only scientific import in this launcher.
    from pde_agent_verifier_solver.plugins.vp_1d1v.product import load_artemis_backend

    binding = load_artemis_backend(**inputs["factory_settings"])
    if binding.report.get("guidance_enabled") is not True or binding.report.get("perturbations_enabled") is not True:
        raise ValueError("loaded Artemis backend does not match the Base+Guidance+Perturbation experiment")
    codec = PythonCandidateCodec(task_kind=TaskKind.OBSERVABLE_MODEL)
    initial = codec.bind_existing(
        inputs["initial_bundle"], storage_root=run_root / "controller_candidates",
        artifact_id="user-artemis-s0", name=config["candidate"]["initial_name"],
        summary=config["candidate"]["initial_summary"],
        usage="Call prepare_solution(case, solve_budget), then handle.evaluate(query).",
    )
    runtime_root = run_root / "agent_runtime"
    runtime_root.mkdir(mode=0o700)
    programs = {
        row["agent_id"]: CommandAgentProgram(
            (
                sys.executable, str(_path(config_path, config["worker"]["script"])),
                "--codex-binary", str(codex_binary), "--auth-file", str(auth_file),
                "--runtime-root", str(runtime_root), "--config",
                str(_path(config_path, config["worker"]["codex_config"])),
            ),
            codec.decode_nomination, incumbent_encoder=codec.encode_incumbent,
            environment={
                "PYTHONPATH": str(ROOT / "src"),
                "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", os.defpath),
                "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
            },
            worker_settings={
                "profile": config["profile"], "model": MODEL, "reasoning_effort": REASONING_EFFORT,
                "transient_resume_until_deadline": config["execution"].get("transient_resume_until_deadline", True),
                "candidate_abi": "prepare_solution_then_evaluate_points_v1",
                "candidate_execution": candidate_execution_capabilities(config["execution"]),
                **agent_gpu_worker_settings(config["execution"]),
                "controller_rpc_timeout_seconds": RPC_TIMEOUT_SECONDS,
            },
            gpu_devices=(gpu,),
        )
        for row, gpu in zip(config["agents"], worker_gpus, strict=True)
    }
    execution = SubprocessAgentExecutionService(
        run_root / "agent_turns", hard_wall_seconds=HARD_WALL_SECONDS,
        enforce_active_budget=False,
        controller_gpu_devices=(controller_gpu,), poll_interval_seconds=0.25,
    )
    recipe = _recipe(config, initial)
    task = binding.plugin.task("observable_model")
    candidate_execution = SubprocessCandidateExecutionService(
        run_root / "candidate_execution",
        task=PythonTaskWorkerSpec.from_task_type(type(task)),
        evaluation_settings={**recipe.task_settings, "default_query": recipe.evaluator.default_query},
        max_workers=candidate_workers,
        case_hard_wall_seconds=CANDIDATE_WALL_SECONDS,
        development_batch_hard_wall_seconds=DEVELOPMENT_BATCH_SECONDS,
        python_executable=sys.executable,
        **gpu_options,
    )
    if reuse_agent_gpus_for_evaluation:
        from pde_agent_verifier_solver.plugins.vp_1d1v.evaluation_gpu_reuse import ArtemisHiddenEvaluationScope
        candidate_execution.hidden_evaluation_scope = ArtemisHiddenEvaluationScope(
            binding, candidate_execution, backend_settings=inputs["factory_settings"],
            devices=(controller_gpu, *worker_gpus), events=events,
        )
    controller = binding.controller(
        recipe, programs, agent_execution=execution, candidate_execution=candidate_execution,
        events=events, task_id="observable_model",
    )
    return controller, execution, {
        **inputs, "backend": dict(binding.report),
        "initial_candidate_sha256": initial.program.tree_sha256,
        "hidden_evaluation_gpu_policy": {
            "reuse_idle_agent_gpus": reuse_agent_gpus_for_evaluation,
            "devices": (controller_gpu, *worker_gpus) if reuse_agent_gpus_for_evaluation else (controller_gpu,),
            "development_oracle_devices": (controller_gpu,),
            "new_slurm_allocation": False,
        },
        "candidate_execution_policy": {
            "max_workers": candidate_workers,
            "original_max_workers": CANDIDATE_WORKERS,
            "solve_budget_seconds": config["task"].get("candidate_compute_budget"),
            **({"candidate_budget_mode": config["task"]["candidate_budget_mode"]}
               if "candidate_budget_mode" in config["task"] else {}),
            "solve_budget_clock": "wall_minus_controller_metered_queue_and_lock_wait",
            "oracle_compute_charged": True,
            "case_hard_wall_seconds": CANDIDATE_WALL_SECONDS,
            "legacy_clock_compatibility": "scoped_entry_module_perf_counter_aliases_only",
        },
        "agent_protocol": {
            "agent_count": AGENT_COUNT,
            "time_budget_mode": "hard_wall",
            "enforce_active_budget": execution.enforce_active_budget,
            "requested_budget_seconds": recipe.default_round.agent_budget_seconds,
            "effective_hard_wall_seconds": execution._wall_budget(recipe.default_round.agent_budget_seconds),
            "service_hard_wall_limit_seconds": execution.hard_wall_seconds,
            "active_budget_seconds": None,
            "includes_controller_tool_wait": True,
            "model": MODEL, "reasoning_effort": REASONING_EFFORT,
        },
        # Keep the original serialized fields solely as the frozen source-run
        # identity. They do not describe the merged execution service's clock.
        "source_execution_config": dict(config["execution"]),
        "protocol_revision": UPSTREAM_PROTOCOL_REVISION,
        "guidance_protocol": {
            "feedback_source": config["generator"]["feedback_source"],
            "feedback_case_count": config["generator"]["feedback_case_count"],
            "initial_feedback": "private S0 clean-Base cohort, distinct from Official and Terminal",
            "initial_feedback_case_count": config["official"]["base"],
            "legacy_feedback_case_count": 384,
            "feedback_count_reason": "The canonical Controller derives S0 and selected-solver feedback from the Official Base count; its independent legacy 384-case cohort is not ported.",
            "training_maximum_wall_seconds": config["generator"]["settings"]["guidance_maximum_wall_seconds"],
            "training_compute_capacity": "one-controller-l40s",
            "training_maximum_concurrency": 1,
            "shared_round_frozen_guidance": True,
            "guided_requests_optional": True,
            "activation_gate_noop_is_valid": True,
            "model_hyperparameters": "unchanged from the previously exercised OU Hermite reward-guidance configuration",
            "model_quality_requalification_required": False,
            "audit_artifact": "guidance_audits.json",
        },
        "assurance": {"agent_process_isolated": True, "candidate_execution_isolated": True,
                      "formal_private_scoring_claimed": True, "hostile_filesystem_sandbox": False},
        "terminal_semantics": "Common held-out cohort for all S1..S5; deliver S5 regardless of diagnostic best_terminal_round.",
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
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value) if field.name != "program"}
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


class _RunWriter:
    def __init__(self, root: Path):
        self.root, self._lock, self._memory_rounds = root, Lock(), {}

    def _write_locked(self, relative: str, value: str):
        destination = self.root / relative
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.partial")
        with temporary.open("x", encoding="utf-8") as handle:
            os.chmod(temporary, 0o600)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)

    def write_text(self, relative: str, value: str):
        with self._lock:
            self._write_locked(relative, value)

    def write_json(self, relative: str, value: object):
        self.write_text(relative, json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n")

    def _memory(self):
        return "\n\n".join((CAMPAIGN_MEMORY_HEADER, *(self._memory_rounds[index] for index in sorted(self._memory_rounds)))) + "\n"

    def record_event(self, event):
        with self._lock:
            with (self.root / "events.jsonl").open("a", encoding="utf-8") as handle:
                os.chmod(handle.name, 0o600)
                handle.write(json.dumps(_jsonable(event), sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            node = event.payload.get("round_memory", {})
            if isinstance(node, Mapping) and isinstance(node.get("merged_markdown"), str) and event.round_index is not None:
                self._memory_rounds[event.round_index] = node["merged_markdown"]
                self._write_locked("CAMPAIGN_MEMORY.md", self._memory())
            self._write_locked("status.json", json.dumps({
                "schema": "canonical_artemis_campaign_status_v1", "status": "running",
                "last_event_sequence": event.sequence, "last_event_type": event.event_type,
                "last_event_message": event.message, "round_index": event.round_index,
                "updated_at_unix": time.time(),
            }, indent=2, sort_keys=True) + "\n")

    def recover_memory_from_events(self):
        if self._memory_rounds:
            self.write_text("CAMPAIGN_MEMORY.md", self._memory())


def _source_state():
    def git(*args):
        result = subprocess.run(("git", "-C", str(ROOT), *args), text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"repository_root": str(ROOT), "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "status_porcelain": git("status", "--porcelain", "--untracked-files=all"),
            "python": sys.executable}


def _persist_guidance_audits(controller, writer):
    """Persist Backend-only cost/identity metadata, never feedback answer arrays."""
    generator = getattr(controller, "generator", None)
    backend = getattr(generator, "backend", None)
    adapter = getattr(backend, "guidance_adapter", None)
    writer.write_json("guidance_audits.json", {
        "schema": "canonical_artemis_guidance_audits_v1",
        "training": tuple(getattr(adapter, "training_audits", ())),
        "sampling": tuple(getattr(adapter, "sampling_audits", ())),
    })


def _publish_selected_candidate(artifact: Artifact, destination: Path):
    source = getattr(artifact.program, "bundle_locator", None)
    if source is None:
        raise TypeError("selected Artemis Candidate is not a Python bundle")
    bundle = inspect_candidate_bundle(source, recorded_identity=artifact.program.tree_sha256)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    shutil.copytree(bundle.root, destination, symlinks=False)


def _round_one_recovery(controller, frozen, writer):
    """Adapt frozen inputs to the existing Controller's audited replay hook."""
    from scripts.artemis_round_recovery import stage_frozen_round_one, verify_official_replay
    from pde_agent_verifier_solver.engine.evaluator import HiddenVerifier
    from pde_agent_verifier_solver.engine.round_recovery import RoundDevelopmentRecovery
    from pde_agent_verifier_solver.sdk import EvaluationPhase

    frozen = stage_frozen_round_one(frozen, writer.root / "recovered_r1_candidates")
    writer.write_json("recovery_input_manifest.json", {
        **dict(frozen.manifest), "source_manifest_sha256": frozen.source_manifest_sha256,
        "later_round_workspace_imported": False,
        "hostile_filesystem_isolation_claimed": False,
    })

    def verify(plan, hidden, artifacts):
        if plan.round_index != 1 or set(hidden) != {"base"}:
            raise ValueError("Artemis recovery must re-evaluate the original R1 Base suite")
        if {artifact.artifact_id for artifact in artifacts} != {
            "candidate-s0", *(artifact.artifact_id for artifact in frozen.artifacts)
        }:
            raise ValueError("Artemis recovery must compare S0 and all four original nominees")
        seed_provider = controller._hidden_plan_seed_provider(
            "official_round:1", EvaluationPhase.OFFICIAL, hidden)
        verifier = HiddenVerifier(
            controller.evaluation, controller.campaign.evaluator.default_query,
            controller.campaign.task_settings, phase=EvaluationPhase.OFFICIAL,
            plan_seed_provider=seed_provider,
        )
        records = tuple(hidden["base"])
        plans = tuple(verifier._plan_for(record, index) for index, record in enumerate(records))
        report = verify_official_replay(frozen, tuple(record.case for record in records), plans)
        writer.write_json("recovery_cohort_verification.json", report)
        return report

    return RoundDevelopmentRecovery(
        round_index=1, artifacts=frozen.artifacts, agent_logs=frozen.agent_logs,
        agent_case_counts=frozen.agent_case_counts, source_run_id="artemis-job-5309385",
        source_manifest_sha256=frozen.source_manifest_sha256, verify_official=verify,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/artemis_canonical_5round.json")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--codex-binary", type=Path)
    parser.add_argument("--auth-file", type=Path)
    parser.add_argument("--worker-gpus", default=os.environ.get("PDE_AGENT_WORKER_GPU_DEVICES", ""))
    parser.add_argument("--controller-gpu", default=os.environ.get("PDE_CONTROLLER_GPU_DEVICE", ""))
    parser.add_argument("--recover-round-one-from", type=Path,
                        help="Cold replay of frozen R1 submissions; never imports old Official/R2 memory")
    parser.add_argument("--candidate-workers", type=int,
                        help="Recovery-only execution concurrency override; scientific budgets remain frozen")
    parser.add_argument("--backend-execution-config", type=Path,
                        help="Explicit Backend execution override; assets and Guidance must remain identical")
    parser.add_argument("--reuse-agent-gpus-for-evaluation", action="store_true",
                        help="Use all three allocated GPUs during hidden evaluation after Agent cleanup")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    config, config_path = _load_config(args.config)
    inputs = _execution_inputs(_static_inputs(config, config_path), args.backend_execution_config)
    if args.candidate_workers is not None and (
        args.recover_round_one_from is None or not 1 <= args.candidate_workers <= CANDIDATE_WORKERS
    ):
        parser.error("--candidate-workers requires R1 recovery and must be between 1 and 28")
    candidate_workers = args.candidate_workers or CANDIDATE_WORKERS
    frozen = None
    if args.recover_round_one_from is not None:
        from scripts.artemis_round_recovery import load_frozen_round_one
        frozen = load_frozen_round_one(
            args.recover_round_one_from, expected_source_head=RECOVERY_SOURCE_HEAD,
            expected_config_sha256=RECOVERY_RESOLVED_CONFIG_SHA256,
            expected_backend_config_sha256=BACKEND_CONFIG_SHA256,
        )
        if frozen.manifest["initial_bundle_tree_sha256"] != inputs["initial_bundle_tree_sha256"]:
            raise ValueError("recovery S0 differs from the original frozen initial candidate")
    if args.validate_only:
        print(json.dumps({"status": "static_configuration_validated", "campaign_id": config["campaign_id"],
                          "backend_config_sha256": inputs["backend_config_sha256"],
                          "assets_loaded": False, "model_called": False}, sort_keys=True), flush=True)
        return 0
    if args.run_root is None or args.codex_binary is None or args.auth_file is None:
        parser.error("execution requires --run-root, --codex-binary and --auth-file")
    worker_gpus, controller_gpu = _gpu_allocation(args.worker_gpus, args.controller_gpu)
    if not args.codex_binary.is_file() or not os.access(args.codex_binary, os.X_OK):
        raise ValueError("Codex binary must be an executable regular file")
    if args.auth_file.is_symlink() or not args.auth_file.is_file():
        raise ValueError("Codex auth source must be a regular non-symlink file")
    run_root = args.run_root.expanduser()
    if run_root.exists() or run_root.is_symlink():
        raise FileExistsError(f"refusing to reuse run root: {run_root}")
    run_root = run_root.resolve()
    run_root.mkdir(parents=True, mode=0o700)
    writer = _RunWriter(run_root)
    controller = execution = candidate_execution = None
    try:
        writer.write_json("status.json", {"status": "preflighting", "started_at_unix": time.time()})
        writer.write_json("resolved_config.json", {**config, "source_config": str(config_path),
                          "config_sha256": _file_sha256(config_path), "run_root": str(run_root),
                          "worker_gpu_tokens": worker_gpus, "controller_gpu_token": controller_gpu,
                          "effective_candidate_workers": candidate_workers,
                          "reuse_agent_gpus_for_evaluation": args.reuse_agent_gpus_for_evaluation,
                          "execution_backend_config": inputs.get("execution_backend_config"),
                          "execution_backend_config_sha256": inputs.get("execution_backend_config_sha256"),
                          "round_one_recovery": frozen is not None})
        state = _source_state()
        writer.write_json("source_state.json", state)
        if state["commit"] is None or state["status_porcelain"] != "":
            raise ValueError("formal Artemis execution requires a clean, committed checkout")
        events = EventStream()
        events.subscribe(writer.record_event)
        controller, execution, provenance = _compose(
            config=config, config_path=config_path, inputs=inputs, run_root=run_root,
            codex_binary=args.codex_binary.resolve(), auth_file=args.auth_file.resolve(),
            worker_gpus=worker_gpus, controller_gpu=controller_gpu, events=events,
            candidate_workers=candidate_workers,
            reuse_agent_gpus_for_evaluation=args.reuse_agent_gpus_for_evaluation,
        )
        candidate_execution = controller.candidate_execution
        events.subscribe(lambda event: _persist_guidance_audits(controller, writer)
                         if event.event_type in ("guidance_ready", "terminal_guidance_ready", "round_finished") else None)
        _persist_guidance_audits(controller, writer)
        writer.write_json("input_provenance.json", provenance)
        writer.write_json("compiled_campaign.json", controller.campaign)
        if args.preflight_only:
            writer.write_json("status.json", {"status": "preflight_completed", "assets_loaded": True,
                              "model_called": False, "finished_at_unix": time.time()})
            return 0
        result = (controller.run() if frozen is None else
                  controller.run(recovery=_round_one_recovery(controller, frozen, writer)))
        writer.write_json("result.json", result)
        writer.write_text("CAMPAIGN_MEMORY.md", result.memory.markdown + "\n")
        writer.write_json("candidate_history.json", result.candidate_history)
        writer.write_json("execution_audits.json", execution.audits)
        writer.write_json("candidate_execution_audits.json", candidate_execution.audits)
        _persist_guidance_audits(controller, writer)
        _publish_selected_candidate(result.selected.artifact, run_root / "selected_candidate")
        writer.write_json("status.json", {
            "status": "completed", "campaign_id": result.campaign_name,
            "selected_artifact_id": result.selected.artifact.artifact_id,
            "best_terminal_round": result.best_terminal_round, "rounds_completed": len(result.rounds),
            "agent_execution_status_counts": dict(Counter(audit.status for audit in execution.audits)),
            "nomination_count": sum(audit.nomination_count for audit in execution.audits),
            "candidate_execution_isolated": True, "finished_at_unix": time.time(),
        })
        print(f"Artemis campaign completed: {run_root}", flush=True)
        return 0
    except BaseException as error:
        try:
            writer.write_text("failure_traceback.txt", traceback.format_exc())
            writer.write_json("execution_audits.json", () if execution is None else execution.audits)
            writer.write_json("candidate_execution_audits.json", () if candidate_execution is None else candidate_execution.audits)
            _persist_guidance_audits(controller, writer)
            writer.recover_memory_from_events()
            writer.write_json("status.json", {"status": "failed", "error_type": type(error).__name__,
                              "error": str(error), "finished_at_unix": time.time()})
        except BaseException as persistence_error:
            error.add_note(f"failure persistence also failed: {type(persistence_error).__name__}: {persistence_error}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
