"""Legacy formal filesystem composition retained for reproducibility.

This module still calls the historical MacroClosedLoopRunner and is not the
canonical product pipeline. New runs enter CampaignController; reusable
subprocess and bundle behavior here is migration source material for an
AgentExecutionService implementation.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, replace
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import signal
import stat
from statistics import fmean
import subprocess
import tempfile
import time
from types import MappingProxyType
from typing import Mapping, Sequence

from pde_agent_verifier_solver.sdk import (
    AgentBundleSubmission,
    CandidateRef,
    CandidateSubmission,
    CohortKind,
    ExecutionAssurance,
    MacroLoopPlugin,
    PromotionProgram,
    RecoverableSamplingRejection,
    candidate_ref_from_attestation,
)

from pde_agent_verifier_solver.sdk.candidate_bundle import (
    CandidateBundleError,
    inspect_candidate_bundle,
)

from .guidance_controller import (
    DevelopmentGuidanceOracle,
    GuidanceProtocolConfig,
    MacroGuidanceBackend,
)
from .macro_closed_loop import (
    CompetitiveSearchPolicy,
    MacroAgentDevelopmentResult,
    MacroAgentSpec,
    MacroClosedLoopRunner,
    MacroLoopConfig,
    MacroLoopResult,
    RoundFeedback,
    RoundHandoff,
    SamplingAttemptAudit,
    SearchCompletionAudit,
    SearchLeaderboardEntry,
)
from .promotion import BaseImprovementWithRobustnessGuardrails
from .reference_agent_runtime import (
    ReferenceFilesystemBundleAttestor,
    ReferenceGuidanceOracleProxy,
    ReferenceGuidanceOracleRPC,
    ReferenceFilesystemMacroAgentProgram,
    _reference_candidate_submission,
    _stage_reference_candidate_bundle,
    SubprocessReferenceAgentLauncher,
)
from .selection import EvidenceStage, PairedObjectivePolicy


REFERENCE_ITERATION_RESULT_SCHEMA = "pde-reference-agent-iteration-result-v5"
REFERENCE_ITERATION_PROTOCOL = "unified_macro_reference_v4"
_COLD_AGENT_ID = "controller-cold-start"
REFERENCE_PROCESS_WORKER_REQUEST_SCHEMA = "reference-process-agent-worker-request-v1"
REFERENCE_PROCESS_WORKER_REPORT_SCHEMA = "reference-process-agent-worker-report-v1"
REFERENCE_PRESELECTION_RECOVERY_SCHEMA = (
    "reference-process-preselection-recovery-v1"
)
REFERENCE_PRESELECTION_RECOVERY_PHASE = (
    "round_0001_workers_complete_selection_uncommitted"
)
_FRESH_AGENT_ID_PREFIX = "fresh-macro-agent"
_MAX_WORKER_REPORT_BYTES = 32 * 1024 * 1024
_MAX_RECOVERY_MANIFEST_BYTES = 2 * 1024 * 1024
_EXPLICIT_AGENT_ENVIRONMENT = (
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "SLURM_JOB_ID",
    "SLURM_TMPDIR",
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_GPUS",
    "SLURM_GPUS",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "USER",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CPATH",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)


class ReferenceIterationError(RuntimeError):
    """A reference Agent iteration cannot produce an auditable result."""


@dataclass(frozen=True)
class ReferenceMidRoundRecovery:
    """Immutable authorization for one fail-closed pre-selection recovery."""

    manifest_path: Path
    expected_manifest_sha256: str

    def __post_init__(self) -> None:
        source = Path(self.manifest_path).expanduser().resolve()
        if source == Path(source.anchor):
            raise ValueError("recovery manifest path must be specific")
        digest = self.expected_manifest_sha256
        if not _is_sha256(digest):
            raise ValueError("recovery manifest digest must be lowercase SHA-256")
        object.__setattr__(self, "manifest_path", source)


@dataclass(frozen=True)
class _ValidatedRecoveredDevelopmentRound:
    results: tuple[MacroAgentDevelopmentResult, ...]
    audit: Mapping[str, object]


@dataclass(frozen=True)
class ReferenceProcessWorkerConfig:
    """Non-scientific execution layout for process-isolated Agent workers."""

    command: tuple[str, ...]
    gpu_devices: tuple[str, ...]
    controller_gpu_device: str
    node_local_root: Path
    cpus_per_worker: int
    worker_timeout_seconds: float

    def __post_init__(self) -> None:
        command = tuple(self.command)
        devices = tuple(self.gpu_devices)
        if not command or any(
            not isinstance(value, str) or not value for value in command
        ):
            raise ValueError("process worker command must contain nonempty argv")
        if not devices or any(
            not isinstance(value, str)
            or not value
            or "," in value
            for value in devices
        ):
            raise ValueError("worker GPU tokens must be nonempty singleton values")
        if len(set(devices)) != len(devices):
            raise ValueError("worker GPU tokens must be unique")
        controller = self.controller_gpu_device
        if (
            not isinstance(controller, str)
            or not controller
            or "," in controller
            or controller in devices
        ):
            raise ValueError(
                "controller GPU token must be one distinct singleton value"
            )
        root = Path(self.node_local_root).expanduser().resolve()
        if root == Path(root.anchor):
            raise ValueError("worker node-local root must be a specific directory")
        if (
            isinstance(self.cpus_per_worker, bool)
            or not isinstance(self.cpus_per_worker, int)
            or self.cpus_per_worker < 1
        ):
            raise ValueError("cpus_per_worker must be a positive integer")
        timeout = float(self.worker_timeout_seconds)
        if not isfinite(timeout) or timeout <= 0.0:
            raise ValueError("worker timeout must be finite and positive")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "gpu_devices", devices)
        object.__setattr__(self, "node_local_root", root)
        object.__setattr__(self, "worker_timeout_seconds", timeout)


@dataclass(frozen=True)
class ReferenceIterationConfig:
    name: str
    run_root: Path
    cold_start_candidate: Path
    agent_launcher: Path
    codex_binary: Path
    auth_file: Path
    prompt: str
    model: str
    reasoning_effort: str
    agent_timeout_seconds: float
    agent_hard_timeout_seconds: float
    round_count: int
    development_case_count: int
    selection_case_count: int
    evidence_stages: tuple[EvidenceStage, ...]
    terminal_case_count: int
    development_case_credits: int
    seed_base: int
    seed_stride: int
    agent_count: int = 1
    search_policy: CompetitiveSearchPolicy = CompetitiveSearchPolicy()
    agent_roles: tuple[str, ...] = ()
    development_pool_multiplier: int = 1
    development_screen_representative_limit: int = 3
    development_release_root_limit: int = 0
    development_screen_root_stages: tuple[int, ...] = ()
    development_screen_root_credit_limit: int = 0
    development_screen_timeout_seconds: float = 300.0
    development_screen_confidence_z: float = 1.96
    development_screen_continuation_max_mean_ratio: float | None = 1.05
    cvar_tail_fraction: float = 0.1
    allow_development_truth: bool = False
    terminal_confidence_z: float = 0.0
    terminal_min_win_rate: float = 0.0
    terminal_require_cvar_no_regression: bool = False
    candidate_compute_budget: float | None = None
    candidate_query_seconds_per_point_limit: float | None = None
    paired_perturbations: bool = True
    paired_objective_policy: PairedObjectivePolicy = PairedObjectivePolicy()
    paired_sampling_max_attempts: int = 16
    cohort_store_root: Path | None = None
    guidance_protocol: GuidanceProtocolConfig | None = None
    promotion_program: PromotionProgram = BaseImprovementWithRobustnessGuardrails()

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.prompt.strip():
            raise ValueError("iteration name and Agent prompt cannot be empty")
        if not self.model.strip() or not self.reasoning_effort.strip():
            raise ValueError("Agent model and reasoning effort cannot be empty")
        cold_start_candidate = Path(
            self.cold_start_candidate
        ).expanduser().resolve()
        try:
            inspect_candidate_bundle(cold_start_candidate)
        except (CandidateBundleError, FileNotFoundError, OSError) as exc:
            raise ValueError(
                f"cold_start_candidate is not a valid candidate bundle: "
                f"{cold_start_candidate}"
            ) from exc
        object.__setattr__(
            self, "cold_start_candidate", cold_start_candidate
        )
        for label in ("agent_launcher", "codex_binary", "auth_file"):
            path = Path(getattr(self, label)).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"{label} is not a regular file: {path}")
            object.__setattr__(self, label, path)
        for label in ("agent_launcher", "codex_binary"):
            if not os.access(getattr(self, label), os.X_OK):
                raise PermissionError(f"{label} is not executable")
        run_root = Path(self.run_root).expanduser().resolve()
        if run_root == Path(run_root.anchor):
            raise ValueError("run_root must be a specific directory")
        object.__setattr__(self, "run_root", run_root)
        if self.cohort_store_root is not None:
            cohort_store_root = Path(self.cohort_store_root).expanduser().resolve()
            if cohort_store_root == Path(cohort_store_root.anchor):
                raise ValueError("cohort_store_root must be a specific directory")
            object.__setattr__(
                self, "cohort_store_root", cohort_store_root
            )
        if not isfinite(float(self.agent_timeout_seconds)) or (
            self.agent_timeout_seconds <= 0.0
        ):
            raise ValueError("agent_timeout_seconds must be finite and positive")
        if not isfinite(float(self.agent_hard_timeout_seconds)) or (
            self.agent_hard_timeout_seconds < self.agent_timeout_seconds
        ):
            raise ValueError(
                "agent_hard_timeout_seconds must be finite and at least the "
                "active-time timeout"
            )
        if (
            self.development_screen_root_stages
            and self.agent_hard_timeout_seconds <= self.agent_timeout_seconds
        ):
            raise ValueError("development screens need hard-wall service allowance")
        if (
            isinstance(self.development_screen_timeout_seconds, bool)
            or not isfinite(float(self.development_screen_timeout_seconds))
            or float(self.development_screen_timeout_seconds) <= 0.0
        ):
            raise ValueError(
                "development_screen_timeout_seconds must be finite and positive"
            )
        if self.development_screen_root_stages and (
            self.agent_hard_timeout_seconds - self.agent_timeout_seconds
            < self.development_screen_timeout_seconds
        ):
            raise ValueError(
                "hard-wall service allowance must cover one screen timeout"
            )
        if (
            isinstance(self.paired_sampling_max_attempts, bool)
            or self.paired_sampling_max_attempts < 1
        ):
            raise ValueError("paired_sampling_max_attempts must be positive")
        if (
            isinstance(self.agent_count, bool)
            or not isinstance(self.agent_count, int)
            or self.agent_count < 1
        ):
            raise ValueError("agent_count must be a positive integer")
        if not isinstance(self.search_policy, CompetitiveSearchPolicy):
            raise TypeError("search_policy must be a CompetitiveSearchPolicy")
        if self.guidance_protocol is not None and not isinstance(
            self.guidance_protocol, GuidanceProtocolConfig
        ):
            raise TypeError(
                "guidance_protocol must be GuidanceProtocolConfig or None"
            )
        promotion_id = getattr(self.promotion_program, "program_id", None)
        if not isinstance(promotion_id, str) or not promotion_id.strip():
            raise TypeError("promotion_program must provide a nonempty program_id")
        if not callable(getattr(self.promotion_program, "select", None)):
            raise TypeError("promotion_program must provide select(...)")
        if isinstance(self.agent_roles, (str, bytes)):
            raise TypeError("agent_roles must be a sequence of strings")
        try:
            roles = tuple(self.agent_roles)
        except TypeError as exc:
            raise TypeError(
                "agent_roles must be a sequence of strings"
            ) from exc
        if roles and len(roles) != self.agent_count:
            raise ValueError(
                "agent_roles must be empty or contain one role per Agent"
            )
        if any(
            not isinstance(role, str) or not role.strip()
            for role in roles
        ):
            raise ValueError("agent_roles must contain nonempty strings")
        object.__setattr__(
            self,
            "agent_roles",
            tuple(role.strip() for role in roles),
        )
        if (
            isinstance(self.development_screen_representative_limit, bool)
            or not isinstance(self.development_screen_representative_limit, int)
            or not 0 <= self.development_screen_representative_limit <= 3
        ):
            raise ValueError(
                "development_screen_representative_limit must lie in [0, 3]"
            )

        if not 0.0 < float(self.cvar_tail_fraction) <= 1.0:
            raise ValueError("cvar_tail_fraction must be in (0, 1]")


def _reference_agent_specs(
    config: ReferenceIterationConfig,
) -> tuple[MacroAgentSpec, ...]:
    return tuple(
        MacroAgentSpec(
            agent_id=f"{_FRESH_AGENT_ID_PREFIX}-{index + 1:02d}",
            name=f"Fresh macroscopic ROM Agent {index + 1}",
            prompt=(
                config.prompt
                if not config.agent_roles
                else (
                    f"{config.prompt}\n\n"
                    "Controller-assigned independent search role: "
                    f"{config.agent_roles[index]}"
                )
            ),
        )
        for index in range(config.agent_count)
    )


def _reference_loop_config(
    config: ReferenceIterationConfig,
    agent_specs: tuple[MacroAgentSpec, ...],
) -> MacroLoopConfig:
    return MacroLoopConfig(
        name=config.name,
        round_count=config.round_count,
        agents=agent_specs,
        development_case_count=config.development_case_count,
        selection_case_count=config.selection_case_count,
        evidence_stages=config.evidence_stages,
        terminal_case_count=config.terminal_case_count,
        agent_case_credits=config.development_case_credits,
        seed_base=config.seed_base,
        seed_stride=config.seed_stride,
        development_pool_multiplier=config.development_pool_multiplier,
        development_screen_representative_limit=(
            config.development_screen_representative_limit
        ),
        development_release_root_limit=(
            config.development_release_root_limit
        ),
        development_screen_root_stages=(
            config.development_screen_root_stages
        ),
        development_screen_root_credit_limit=(
            config.development_screen_root_credit_limit
        ),
        development_screen_confidence_z=(
            config.development_screen_confidence_z
        ),
        development_screen_continuation_max_mean_ratio=(
            config.development_screen_continuation_max_mean_ratio
        ),
        cvar_tail_fraction=config.cvar_tail_fraction,
        terminal_confidence_z=config.terminal_confidence_z,
        terminal_min_win_rate=config.terminal_min_win_rate,
        terminal_require_cvar_no_regression=(
            config.terminal_require_cvar_no_regression
        ),
        allow_development_truth=config.allow_development_truth,
        candidate_compute_budget=config.candidate_compute_budget,
        candidate_query_seconds_per_point_limit=(
            config.candidate_query_seconds_per_point_limit
        ),
        execution_assurance=ExecutionAssurance.REFERENCE_IN_PROCESS,
        paired_perturbations=config.paired_perturbations,
        paired_objective_policy=config.paired_objective_policy,
        paired_sampling_max_attempts=config.paired_sampling_max_attempts,
        search_policy=config.search_policy,
        guidance_protocol=config.guidance_protocol,
        promotion_program=config.promotion_program,
    )


def _reference_agent_program(
    *,
    config: ReferenceIterationConfig,
    run_root: Path,
    run_id: str,
    attestor: ReferenceFilesystemBundleAttestor,
) -> ReferenceFilesystemMacroAgentProgram:
    launcher = SubprocessReferenceAgentLauncher(
        command=(
            str(config.agent_launcher),
            "--model",
            config.model,
            "--reasoning-effort",
            config.reasoning_effort,
            "--codex-binary",
            str(config.codex_binary),
            "--auth-file",
            str(config.auth_file),
        ),
        environment=_agent_environment(os.environ),
        timeout_seconds=config.agent_timeout_seconds,
        hard_timeout_seconds=config.agent_hard_timeout_seconds,
    )
    return ReferenceFilesystemMacroAgentProgram(
        run_root=run_root,
        run_id=run_id,
        launcher=launcher,
        screen_timeout_seconds=config.development_screen_timeout_seconds,
        bundle_attestor=attestor,
    )


@dataclass
class _ReferenceRunningWorker:
    agent: MacroAgentSpec
    gpu_device: str
    slot: int
    round_index: int
    parent_sha256: str
    request_sha256: str
    report_path: Path
    process: subprocess.Popen
    stdout_stream: object
    stderr_stream: object
    started_monotonic_ns: int
    guidance_rpc: ReferenceGuidanceOracleRPC | None = None


class ReferenceProcessRoundDeveloper:
    """Run same-round Agent development concurrently on singleton GPU workers."""

    def __init__(
        self,
        *,
        execution: ReferenceProcessWorkerConfig,
        iteration_config: ReferenceIterationConfig,
        loop_config: MacroLoopConfig,
        controller_root: Path,
        central_attestor: ReferenceFilesystemBundleAttestor,
        campaign_cold_start: CandidateSubmission,
        plugin_identity: Mapping[str, object],
        recovered_round: _ValidatedRecoveredDevelopmentRound | None = None,
    ) -> None:
        self.execution = execution
        self.iteration_config = iteration_config
        self.loop_config = loop_config
        self.controller_root = Path(controller_root).resolve()
        self.central_attestor = central_attestor
        self.campaign_cold_start = campaign_cold_start
        self.plugin_identity = dict(plugin_identity)
        self._recovered_round = recovered_round
        self._recovered_round_consumed = False
        if len(execution.gpu_devices) != len(loop_config.agents):
            raise ValueError("one distinct worker GPU is required per Agent")
        self._round_audits: list[Mapping[str, object]] = []

    def run_agents(
        self,
        *,
        round_index: int,
        agents: tuple[MacroAgentSpec, ...],
        incumbent: CandidateSubmission,
        prior_handoff: RoundHandoff | None,
        prior_leaderboard: tuple[SearchLeaderboardEntry, ...],
        shared_candidates: tuple[CandidateRef, ...],
        guidance_oracles: Mapping[
            str, DevelopmentGuidanceOracle
        ] | None = None,
    ) -> tuple[MacroAgentDevelopmentResult, ...]:
        if tuple(agents) != tuple(self.loop_config.agents):
            raise RuntimeError("worker executor Agent order changed")
        oracle_map = {} if guidance_oracles is None else dict(
            guidance_oracles
        )
        if oracle_map and set(oracle_map) != {
            agent.agent_id for agent in agents
        }:
            raise ValueError(
                "worker guidance mapping must contain exactly one Oracle "
                "per Agent"
            )
        if any(
            not isinstance(oracle, DevelopmentGuidanceOracle)
            for oracle in oracle_map.values()
        ):
            raise TypeError(
                "worker guidance mapping contains an invalid Oracle"
            )
        if self._recovered_round is not None:
            if not self._recovered_round_consumed:
                if round_index != 1:
                    raise ReferenceIterationError(
                        "recovered development must be consumed at round one"
                    )
                if incumbent.ref != self.campaign_cold_start.ref:
                    raise ReferenceIterationError(
                        "recovered development received a changed cold incumbent"
                    )
                if (
                    prior_handoff is not None
                    or prior_leaderboard
                    or shared_candidates
                ):
                    raise ReferenceIterationError(
                        "recovered first round received downstream controller state"
                    )
                if oracle_map:
                    raise ReferenceIterationError(
                        "guided development cannot be recovered without a bound snapshot"
                    )
                self._recovered_round_consumed = True
                self._round_audits.append(
                    MappingProxyType(
                        {
                            "round_index": 1,
                            "same_round_barrier": True,
                            "overlap_verified": True,
                            "status": "recovered_source_workers",
                            "current_workers_spawned": False,
                        }
                    )
                )
                return self._recovered_round.results
            if round_index == 1:
                raise ReferenceIterationError(
                    "recovered development cannot be consumed twice"
                )
        round_root = (
            self.controller_root
            / "agent_workers"
            / f"round-{round_index:04d}"
        )
        round_root.mkdir(parents=True, exist_ok=False, mode=0o700)
        running: dict[str, _ReferenceRunningWorker] = {}
        results: dict[str, MacroAgentDevelopmentResult] = {}
        audits: dict[str, dict[str, object]] = {}
        try:
            for slot, (agent, gpu_device) in enumerate(
                zip(agents, self.execution.gpu_devices, strict=True)
            ):
                try:
                    worker = self._spawn_worker(
                        round_index=round_index,
                        round_root=round_root,
                        agent=agent,
                        slot=slot,
                        gpu_device=gpu_device,
                        incumbent=incumbent.ref,
                        prior_handoff=prior_handoff,
                        prior_leaderboard=prior_leaderboard,
                        shared_candidates=shared_candidates,
                        guidance_oracle=oracle_map.get(agent.agent_id),
                    )
                except (KeyboardInterrupt, GeneratorExit):
                    raise
                except BaseException as exc:
                    results[agent.agent_id] = _worker_failure_result(
                        agent.agent_id,
                        f"worker_spawn_failed:{type(exc).__name__}",
                    )
                    audits[agent.agent_id] = {
                        "agent_id": agent.agent_id,
                        "slot": slot,
                        "gpu_device": gpu_device,
                        "status": "spawn_failed",
                        "failure_type": type(exc).__name__,
                    }
                else:
                    running[agent.agent_id] = worker
                    audits[agent.agent_id] = {
                        "agent_id": agent.agent_id,
                        "slot": slot,
                        "gpu_device": gpu_device,
                        "pid": worker.process.pid,
                        "status": "running",
                        "started_monotonic_ns": worker.started_monotonic_ns,
                        "guidance_oracle_hosted": (
                            worker.guidance_rpc is not None
                        ),
                    }

            deadline = time.monotonic() + self.execution.worker_timeout_seconds
            for agent in agents:
                worker = running.get(agent.agent_id)
                if worker is None:
                    continue
                remaining = max(0.0, deadline - time.monotonic())
                timed_out = False
                try:
                    returncode = worker.process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _terminate_worker_process(worker.process)
                    returncode = worker.process.returncode
                finally:
                    worker.stdout_stream.close()
                    worker.stderr_stream.close()
                    if worker.guidance_rpc is not None:
                        worker.guidance_rpc.close()
                        worker.guidance_rpc = None
                audit = audits[agent.agent_id]
                audit["returncode"] = returncode
                audit["finished_monotonic_ns"] = time.monotonic_ns()
                if timed_out:
                    audit["status"] = "timeout"
                    results[agent.agent_id] = _worker_failure_result(
                        agent.agent_id, "worker_failed:TimeoutExpired"
                    )
                    continue
                if returncode != 0:
                    audit["status"] = "nonzero_exit"
                    results[agent.agent_id] = _worker_failure_result(
                        agent.agent_id, "worker_failed:NonzeroExit"
                    )
                    continue
                try:
                    result, report_audit = self._load_worker_result(worker)
                except (KeyboardInterrupt, GeneratorExit):
                    raise
                except BaseException as exc:
                    audit["status"] = "invalid_report"
                    audit["failure_type"] = type(exc).__name__
                    results[agent.agent_id] = _worker_failure_result(
                        agent.agent_id,
                        f"worker_report_failed:{type(exc).__name__}",
                    )
                    continue
                audit.update(report_audit)
                audit["status"] = "complete"
                results[agent.agent_id] = result
        except BaseException:
            for worker in running.values():
                _terminate_worker_process(worker.process)
                try:
                    worker.stdout_stream.close()
                    worker.stderr_stream.close()
                except BaseException:
                    pass
                if worker.guidance_rpc is not None:
                    worker.guidance_rpc.close()
                    worker.guidance_rpc = None
            raise

        complete_audits = [
            audits[agent.agent_id]
            for agent in agents
            if audits[agent.agent_id].get("status") == "complete"
        ]
        overlap_verified = (
            len(complete_audits) == len(agents)
            and _verify_worker_topology(complete_audits)
        )
        self._round_audits.append(
            MappingProxyType(
                {
                    "round_index": round_index,
                    "same_round_barrier": True,
                    "overlap_verified": overlap_verified,
                    "workers": [audits[agent.agent_id] for agent in agents],
                }
            )
        )
        return tuple(results[agent.agent_id] for agent in agents)

    def _spawn_worker(
        self,
        *,
        round_index: int,
        round_root: Path,
        agent: MacroAgentSpec,
        slot: int,
        gpu_device: str,
        incumbent: CandidateRef,
        prior_handoff: RoundHandoff | None,
        prior_leaderboard: tuple[SearchLeaderboardEntry, ...],
        shared_candidates: tuple[CandidateRef, ...],
        guidance_oracle: DevelopmentGuidanceOracle | None,
    ) -> _ReferenceRunningWorker:
        worker_token = hashlib.sha256(
            agent.agent_id.encode("utf-8")
        ).hexdigest()[:12]
        worker_root = round_root / f"agent-{slot:02d}-{worker_token}"
        worker_root.mkdir(mode=0o700)
        with ExitStack() as startup:
            guidance_rpc = None
            guidance_capability = None
            if guidance_oracle is not None:
                guidance_rpc = startup.enter_context(
                    ReferenceGuidanceOracleRPC(
                        guidance_oracle,
                        workspace=worker_root / "guidance_transport",
                    )
                )
                guidance_capability = guidance_rpc.worker_capability()
            request = {
                "schema": REFERENCE_PROCESS_WORKER_REQUEST_SCHEMA,
                "campaign": {
                    "name": self.iteration_config.name,
                    "seed_base": self.iteration_config.seed_base,
                    "seed_stride": self.iteration_config.seed_stride,
                    "agent_count": self.iteration_config.agent_count,
                },
                "round_index": round_index,
                "agent": {
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "prompt": agent.prompt,
                },
                "campaign_cold_start": _private_candidate_payload(
                    self.campaign_cold_start.ref
                ),
                "incumbent": _private_candidate_payload(incumbent),
                "prior_handoff": _private_handoff_payload(prior_handoff),
                "prior_leaderboard": [
                    _private_leaderboard_payload(row)
                    for row in prior_leaderboard
                ],
                "shared_candidates": [
                    _private_candidate_payload(row)
                    for row in shared_candidates
                ],
                "guidance_oracle": guidance_capability,
            }
            request_bytes = _canonical_json_bytes(request)
            request_sha256 = hashlib.sha256(request_bytes).hexdigest()
            request_path = worker_root / "request.json"
            report_path = worker_root / "report.json"
            _atomic_write(request_path, request_bytes, mode=0o400)
            stdout_stream = startup.enter_context(
                (worker_root / "worker.stdout.log").open("xb")
            )
            stderr_stream = startup.enter_context(
                (worker_root / "worker.stderr.log").open("xb")
            )
            environment = dict(os.environ)
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu_device,
                    "PDE_ASSIGNED_AGENT_GPU_SLOT": str(slot),
                    "PDE_ASSIGNED_AGENT_GPU_TOKEN": gpu_device,
                    "OMP_NUM_THREADS": str(self.execution.cpus_per_worker),
                    "MKL_NUM_THREADS": str(self.execution.cpus_per_worker),
                    "PYTHONPYCACHEPREFIX": str(
                        self.execution.node_local_root
                        / "pycache"
                        / f"agent-{slot:02d}"
                    ),
                }
            )
            if self.iteration_config.cohort_store_root is None:
                environment.pop("PLASMA_COHORT_STORAGE_ROOT", None)
            else:
                environment["PLASMA_COHORT_STORAGE_ROOT"] = str(
                    self.execution.node_local_root
                    / "cohorts"
                    / f"round-{round_index:04d}"
                    / f"agent-{slot:02d}"
                )
            command = (
                *self.execution.command,
                "--agent-worker-request",
                str(request_path),
                "--agent-worker-result",
                str(report_path),
            )
            started_ns = time.monotonic_ns()
            process = subprocess.Popen(
                command,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_stream,
                stderr=stderr_stream,
                start_new_session=True,
            )
            startup.callback(_terminate_worker_process, process)
            worker = _ReferenceRunningWorker(
                agent=agent,
                gpu_device=gpu_device,
                slot=slot,
                round_index=round_index,
                parent_sha256=incumbent.bundle_sha256,
                request_sha256=request_sha256,
                report_path=report_path,
                process=process,
                stdout_stream=stdout_stream,
                stderr_stream=stderr_stream,
                started_monotonic_ns=started_ns,
                guidance_rpc=guidance_rpc,
            )
            startup.pop_all()
            return worker

    def _load_worker_result(
        self,
        worker: _ReferenceRunningWorker,
    ) -> tuple[MacroAgentDevelopmentResult, Mapping[str, object]]:
        report = _load_bounded_json(
            worker.report_path,
            maximum_bytes=_MAX_WORKER_REPORT_BYTES,
        )
        if report.get("schema") != REFERENCE_PROCESS_WORKER_REPORT_SCHEMA:
            raise ReferenceIterationError("worker report schema changed")
        if report.get("request_sha256") != worker.request_sha256:
            raise ReferenceIterationError("worker report request digest changed")
        if report.get("status") != "complete":
            raise ReferenceIterationError("worker did not report completion")
        if report.get("round_index") != worker.round_index:
            raise ReferenceIterationError("worker report round changed")
        if report.get("agent_id") != worker.agent.agent_id:
            raise ReferenceIterationError("worker report Agent identity changed")
        if dict(report.get("plugin_identity", {})) != self.plugin_identity:
            raise ReferenceIterationError("worker plugin identity changed")
        gpu = _validated_gpu_report(
            report.get("gpu"),
            expected_pid=worker.process.pid,
            expected_token=worker.gpu_device,
            expected_slot=worker.slot,
        )
        development = report.get("development")
        if not isinstance(development, Mapping):
            raise ReferenceIterationError("worker development report is absent")
        result = _development_result_from_report(
            agent=worker.agent,
            raw=development,
            central_attestor=self.central_attestor,
            expected_round_index=worker.round_index,
            expected_parent_sha256=worker.parent_sha256,
        )
        audit = {
            "worker_started_monotonic_ns": gpu["started_monotonic_ns"],
            "worker_finished_monotonic_ns": gpu["finished_monotonic_ns"],
            "cuda_device_count": gpu["device_count"],
            "cuda_current_device": gpu["current_device"],
            "cuda_device_name": gpu["device_name"],
            "cuda_device_uuid": gpu.get("device_uuid"),
            "visible_devices": gpu["visible_devices"],
            "plugin_identity_verified": True,
            "request_sha256": worker.request_sha256,
        }
        return result, audit

    def audit_payload(self) -> Mapping[str, object]:
        return {
            "schema": "reference-process-agent-worker-execution-v1",
            "topology": (
                "recovered_round_one_then_singleton_gpu_workers"
                if self._recovered_round is not None
                else "one_process_and_singleton_gpu_per_agent_plus_central_gpu"
            ),
            "controller": {
                "pid": os.getpid(),
                "gpu_device": self.execution.controller_gpu_device,
            },
            "worker_count": len(self.execution.gpu_devices),
            "cpus_per_worker": self.execution.cpus_per_worker,
            "recovery": (
                dict(self._recovered_round.audit)
                if self._recovered_round is not None
                else None
            ),
            "rounds": [dict(row) for row in self._round_audits],
        }


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_sha1(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_exact_keys(
    raw: object,
    expected: set[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise ReferenceIterationError(f"{label} fields changed")
    return raw


def _read_exact_json(
    path: Path,
    *,
    maximum_bytes: int,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, object], bytes]:
    if not _is_sha256(expected_sha256):
        raise ReferenceIterationError(f"{label} digest is invalid")
    source = Path(path).expanduser()
    try:
        metadata = source.lstat()
    except OSError as exc:
        raise ReferenceIterationError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size <= 0
        or metadata.st_size > maximum_bytes
    ):
        raise ReferenceIterationError(f"{label} is not a bounded regular file")
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise ReferenceIterationError(f"{label} could not be read") from exc
    if len(payload) != metadata.st_size:
        raise ReferenceIterationError(f"{label} changed while being read")
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReferenceIterationError(f"{label} is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ReferenceIterationError(f"{label} root must be an object")
    return decoded, payload


def _load_bounded_json(path: Path, *, maximum_bytes: int) -> dict[str, object]:
    source = Path(path).resolve()
    metadata = source.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum_bytes:
        raise ReferenceIterationError("worker IPC file is not a bounded regular file")
    payload = source.read_bytes()
    if len(payload) != metadata.st_size:
        raise ReferenceIterationError("worker IPC file changed while being read")
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReferenceIterationError("worker IPC is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ReferenceIterationError("worker IPC root must be an object")
    return decoded


def load_reference_process_worker_request(
    path: Path,
) -> tuple[Mapping[str, object], str]:
    """Load one controller-created immutable worker request and its digest."""

    payload = _load_bounded_json(path, maximum_bytes=_MAX_WORKER_REPORT_BYTES)
    if payload.get("schema") != REFERENCE_PROCESS_WORKER_REQUEST_SCHEMA:
        raise ReferenceIterationError("worker request schema changed")
    digest = hashlib.sha256(Path(path).resolve().read_bytes()).hexdigest()
    return MappingProxyType(payload), digest


def write_reference_process_worker_report(
    path: Path,
    report: Mapping[str, object],
) -> None:
    if report.get("schema") != REFERENCE_PROCESS_WORKER_REPORT_SCHEMA:
        raise ValueError("worker report schema changed")
    payload = _canonical_json_bytes(report)
    if len(payload) > _MAX_WORKER_REPORT_BYTES:
        raise ValueError("worker report exceeds its transport bound")
    _atomic_write(Path(path).resolve(), payload, mode=0o400)


def _private_candidate_payload(candidate: CandidateRef) -> Mapping[str, object]:
    if candidate.bundle_uri is None:
        raise ReferenceIterationError("process handoff candidate has no sealed URI")
    return {
        "candidate_id": candidate.candidate_id,
        "bundle_sha256": candidate.bundle_sha256,
        "round_index": candidate.round_index,
        "agent_id": candidate.agent_id,
        "parent_sha256": candidate.parent_sha256,
        "bundle_uri": candidate.bundle_uri,
    }


def _private_candidate_from_payload(raw: object) -> CandidateRef:
    if not isinstance(raw, Mapping):
        raise ReferenceIterationError("worker candidate reference is absent")
    expected = {
        "candidate_id",
        "bundle_sha256",
        "round_index",
        "agent_id",
        "parent_sha256",
        "bundle_uri",
    }
    if set(raw) != expected:
        raise ReferenceIterationError("worker candidate reference fields changed")
    values = dict(raw)
    if not isinstance(values["bundle_uri"], str):
        raise ReferenceIterationError("worker candidate URI is invalid")
    try:
        return CandidateRef(**values)
    except (TypeError, ValueError) as exc:
        raise ReferenceIterationError("worker candidate reference is invalid") from exc


def _private_feedback_payload(feedback: RoundFeedback) -> Mapping[str, object]:
    return {
        "previous_incumbent_sha256": feedback.previous_incumbent_sha256,
        "selected_sha256": feedback.selected_sha256,
        "promoted": feedback.promoted,
        "decision_reason": feedback.decision_reason,
        "evidence_case_count": feedback.evidence_case_count,
        "paired_mean_improvement": feedback.paired_mean_improvement,
        "paired_confidence_lower": feedback.paired_confidence_lower,
        "paired_win_rate": feedback.paired_win_rate,
        "confidence_group_size": feedback.confidence_group_size,
        "perturbed_guardrail_passed": feedback.perturbed_guardrail_passed,
        "perturbed_guardrail_confidence_lower": (
            feedback.perturbed_guardrail_confidence_lower
        ),
    }


def _private_feedback_from_payload(raw: object) -> RoundFeedback:
    if not isinstance(raw, Mapping):
        raise ReferenceIterationError("worker round feedback is absent")
    try:
        return RoundFeedback(**dict(raw))
    except (TypeError, ValueError) as exc:
        raise ReferenceIterationError("worker round feedback is invalid") from exc


def _private_handoff_payload(
    handoff: RoundHandoff | None,
) -> Mapping[str, object] | None:
    if handoff is None:
        return None
    return {
        "round_index": handoff.round_index,
        "selected": _private_candidate_payload(handoff.selected),
        "promoted": handoff.promoted,
        "challenger": (
            _private_candidate_payload(handoff.challenger)
            if handoff.challenger is not None
            else None
        ),
        "decision_reason": handoff.decision_reason,
        "feedback": _private_feedback_payload(handoff.feedback),
        "previous_handoff_sha256": handoff.previous_handoff_sha256,
        "handoff_sha256": handoff.handoff_sha256,
    }


def _private_handoff_from_payload(raw: object) -> RoundHandoff | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ReferenceIterationError("worker handoff is invalid")
    try:
        return RoundHandoff(
            round_index=raw["round_index"],
            selected=_private_candidate_from_payload(raw["selected"]),
            promoted=raw["promoted"],
            challenger=(
                _private_candidate_from_payload(raw["challenger"])
                if raw.get("challenger") is not None
                else None
            ),
            decision_reason=raw["decision_reason"],
            feedback=_private_feedback_from_payload(raw["feedback"]),
            previous_handoff_sha256=raw["previous_handoff_sha256"],
            handoff_sha256=raw["handoff_sha256"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReferenceIterationError("worker handoff fields are invalid") from exc


def _private_leaderboard_payload(
    row: SearchLeaderboardEntry,
) -> Mapping[str, object]:
    return {
        "rank": row.rank,
        "candidate": _private_candidate_payload(row.candidate),
        "agent_id": row.agent_id,
        "eligible": row.eligible,
        "mean": row.mean,
        "cvar": row.cvar,
        "maximum": row.maximum,
        "generated_mean": row.generated_mean,
        "perturbed_mean": row.perturbed_mean,
        "guardrail_passed": row.guardrail_passed,
        "roles": list(row.roles),
    }


def _private_leaderboard_from_payload(raw: object) -> SearchLeaderboardEntry:
    if not isinstance(raw, Mapping):
        raise ReferenceIterationError("worker leaderboard row is invalid")
    try:
        return SearchLeaderboardEntry(
            rank=raw["rank"],
            candidate=_private_candidate_from_payload(raw["candidate"]),
            agent_id=raw["agent_id"],
            eligible=raw["eligible"],
            mean=raw["mean"],
            cvar=raw["cvar"],
            maximum=raw["maximum"],
            generated_mean=raw.get("generated_mean"),
            perturbed_mean=raw.get("perturbed_mean"),
            guardrail_passed=raw.get("guardrail_passed"),
            roles=tuple(_string_sequence(raw.get("roles", ()), "leaderboard roles")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReferenceIterationError("worker leaderboard fields are invalid") from exc


def _search_completion_payload(
    audit: SearchCompletionAudit | None,
) -> Mapping[str, object] | None:
    if audit is None:
        return None
    return {
        "agent_id": audit.agent_id,
        "distinct_screened_candidates": audit.distinct_screened_candidates,
        "full_stage_candidates": audit.full_stage_candidates,
        "released_roots": audit.released_roots,
        "final_candidate_sha256": audit.final_candidate_sha256,
        "final_completed_root_count": audit.final_completed_root_count,
        "sufficient": audit.sufficient,
        "failures": list(audit.failures),
    }


def _search_completion_from_payload(raw: object) -> SearchCompletionAudit | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ReferenceIterationError("worker search audit is invalid")
    try:
        return SearchCompletionAudit(
            agent_id=raw["agent_id"],
            distinct_screened_candidates=raw["distinct_screened_candidates"],
            full_stage_candidates=raw["full_stage_candidates"],
            released_roots=raw["released_roots"],
            final_candidate_sha256=raw.get("final_candidate_sha256"),
            final_completed_root_count=raw["final_completed_root_count"],
            sufficient=raw["sufficient"],
            failures=tuple(_string_sequence(raw.get("failures", ()), "search failures")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReferenceIterationError("worker search audit fields are invalid") from exc


def _sampling_attempt_payload(audit: SamplingAttemptAudit) -> Mapping[str, object]:
    return {
        "round_index": audit.round_index,
        "cohort": audit.cohort.value,
        "attempt_index": audit.attempt_index,
        "maximum_attempts": audit.maximum_attempts,
        "base_seed": audit.base_seed,
        "perturbed_seed": audit.perturbed_seed,
        "root_case_ids": list(audit.root_case_ids),
        "accepted": audit.accepted,
        "rejection_audit": (
            dict(audit.rejection_audit)
            if audit.rejection_audit is not None
            else None
        ),
    }


def _sampling_attempt_from_payload(raw: object) -> SamplingAttemptAudit:
    if not isinstance(raw, Mapping):
        raise ReferenceIterationError("worker sampling audit is invalid")
    rejection = raw.get("rejection_audit")
    if rejection is not None and not isinstance(rejection, Mapping):
        raise ReferenceIterationError("worker rejection audit is invalid")
    try:
        return SamplingAttemptAudit(
            round_index=raw["round_index"],
            cohort=CohortKind(raw["cohort"]),
            attempt_index=raw["attempt_index"],
            maximum_attempts=raw["maximum_attempts"],
            base_seed=raw["base_seed"],
            perturbed_seed=raw["perturbed_seed"],
            root_case_ids=tuple(
                _string_sequence(raw["root_case_ids"], "sampling root ids")
            ),
            accepted=raw["accepted"],
            rejection_audit=(
                MappingProxyType(dict(rejection)) if rejection is not None else None
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReferenceIterationError("worker sampling audit fields are invalid") from exc


def _string_sequence(raw: object, label: str) -> tuple[str, ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ReferenceIterationError(f"{label} must be a sequence")
    values = tuple(raw)
    if any(not isinstance(value, str) or not value for value in values):
        raise ReferenceIterationError(f"{label} contains an invalid string")
    return values


def _worker_failure_result(agent_id: str, failure: str) -> MacroAgentDevelopmentResult:
    return MacroAgentDevelopmentResult(
        agent_id=agent_id,
        portfolio=(),
        development_case_ids=(),
        sampled_case_ids=(),
        sampling_attempts=(),
        failure=failure,
    )


def _terminate_worker_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5.0)


def _validated_gpu_report(
    raw: object,
    *,
    expected_pid: int,
    expected_token: str,
    expected_slot: int,
) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise ReferenceIterationError("worker GPU report is absent")
    gpu = dict(raw)
    integers = (
        "pid",
        "device_count",
        "current_device",
        "started_monotonic_ns",
        "assigned_slot",
        "finished_monotonic_ns",
    )
    if any(
        isinstance(gpu.get(name), bool) or not isinstance(gpu.get(name), int)
        for name in integers
    ):
        raise ReferenceIterationError("worker GPU integer audit is invalid")
    if (
        gpu["pid"] != expected_pid
        or gpu.get("available") is not True
        or gpu["device_count"] != 1
        or gpu["current_device"] != 0
        or gpu.get("visible_devices") != expected_token
        or gpu["assigned_slot"] != expected_slot
        or gpu.get("assigned_token") != expected_token
        or gpu["started_monotonic_ns"] >= gpu["finished_monotonic_ns"]
        or not isinstance(gpu.get("device_name"), str)
        or not gpu["device_name"]
    ):
        raise ReferenceIterationError("worker singleton GPU binding is invalid")
    device_uuid = gpu.get("device_uuid")
    if device_uuid is not None and (
        not isinstance(device_uuid, str) or not device_uuid
    ):
        raise ReferenceIterationError("worker GPU UUID is invalid")
    return gpu


def _verify_worker_topology(audits: Sequence[Mapping[str, object]]) -> bool:
    if not audits:
        return False
    pids = [row.get("pid") for row in audits]
    devices = [row.get("gpu_device") for row in audits]
    if len(set(pids)) != len(pids) or len(set(devices)) != len(devices):
        raise ReferenceIterationError("completed workers did not use unique processes/GPUs")
    if len(audits) == 1:
        return True
    started = [int(row["worker_started_monotonic_ns"]) for row in audits]
    finished = [int(row["worker_finished_monotonic_ns"]) for row in audits]
    if max(started) >= min(finished):
        raise ReferenceIterationError("same-round Agent workers did not overlap")
    uuids = [row.get("cuda_device_uuid") for row in audits]
    known_uuids = [value for value in uuids if value is not None]
    if known_uuids and len(known_uuids) == len(uuids) and (
        len(set(known_uuids)) != len(known_uuids)
    ):
        raise ReferenceIterationError("completed workers resolved to the same GPU UUID")
    return True


def _development_result_payload(
    result: MacroAgentDevelopmentResult,
) -> Mapping[str, object]:
    return {
        "portfolio": [
            {
                "candidate": _private_candidate_payload(candidate.ref),
                "roles": list(roles),
            }
            for candidate, roles in result.portfolio
        ],
        "development_case_ids": list(result.development_case_ids),
        "sampled_case_ids": list(result.sampled_case_ids),
        "sampling_attempts": [
            _sampling_attempt_payload(row) for row in result.sampling_attempts
        ],
        "search_completion": _search_completion_payload(result.search_completion),
        "failure": result.failure,
    }


def _development_result_from_report(
    *,
    agent: MacroAgentSpec,
    raw: Mapping[str, object],
    central_attestor: ReferenceFilesystemBundleAttestor,
    expected_round_index: int,
    expected_parent_sha256: str,
) -> MacroAgentDevelopmentResult:
    portfolio_raw = raw.get("portfolio")
    if isinstance(portfolio_raw, (str, bytes)) or not isinstance(
        portfolio_raw, Sequence
    ):
        raise ReferenceIterationError("worker portfolio is invalid")
    portfolio = []
    for item in portfolio_raw:
        if not isinstance(item, Mapping):
            raise ReferenceIterationError("worker portfolio row is invalid")
        claimed = _private_candidate_from_payload(item.get("candidate"))
        if (
            claimed.agent_id != agent.agent_id
            or claimed.round_index != expected_round_index
            or claimed.parent_sha256 != expected_parent_sha256
        ):
            raise ReferenceIterationError("worker candidate lineage changed")
        attestation = central_attestor.attest(
            AgentBundleSubmission(
                candidate_id=claimed.candidate_id,
                bundle_uri=claimed.bundle_uri or "",
            ),
            round_index=expected_round_index,
            agent_id=agent.agent_id,
            parent_sha256=expected_parent_sha256,
        )
        candidate = candidate_ref_from_attestation(attestation)
        if candidate.bundle_sha256 != claimed.bundle_sha256:
            raise ReferenceIterationError("worker candidate digest changed")
        roles = _string_sequence(item.get("roles", ()), "portfolio roles")
        portfolio.append((_candidate_submission_from_ref(candidate), roles))
    failure = raw.get("failure")
    if failure is not None and (not isinstance(failure, str) or not failure):
        raise ReferenceIterationError("worker development failure is invalid")
    return MacroAgentDevelopmentResult(
        agent_id=agent.agent_id,
        portfolio=tuple(portfolio),
        development_case_ids=_string_sequence(
            raw.get("development_case_ids"), "development case ids"
        ),
        sampled_case_ids=_string_sequence(
            raw.get("sampled_case_ids"), "sampled case ids"
        ),
        sampling_attempts=tuple(
            _sampling_attempt_from_payload(row)
            for row in _sequence(raw.get("sampling_attempts"), "sampling attempts")
        ),
        search_completion=_search_completion_from_payload(
            raw.get("search_completion")
        ),
        failure=failure,
    )


def _sequence(raw: object, label: str) -> tuple[object, ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ReferenceIterationError(f"{label} must be a sequence")
    return tuple(raw)


def _recovery_relative_path(
    source_root: Path,
    raw: object,
    label: str,
) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise ReferenceIterationError(f"{label} must be a relative path")
    candidate = (source_root / raw).resolve()
    try:
        candidate.relative_to(source_root)
    except ValueError as exc:
        raise ReferenceIterationError(f"{label} escapes the source run") from exc
    return candidate


def _recovery_plugin_identity(
    raw: object,
    label: str,
) -> dict[str, object]:
    identity = _require_exact_keys(
        raw,
        {"plugin_id", "generator_id", "generator_sha256"},
        label,
    )
    if (
        not isinstance(identity["plugin_id"], str)
        or not identity["plugin_id"]
        or not isinstance(identity["generator_id"], str)
        or not identity["generator_id"]
        or not _is_sha256(identity["generator_sha256"])
    ):
        raise ReferenceIterationError(f"{label} is invalid")
    return dict(identity)


def _candidate_commitment(candidate: CandidateRef) -> tuple[object, ...]:
    return (
        candidate.candidate_id,
        candidate.bundle_sha256,
        candidate.round_index,
        candidate.agent_id,
        candidate.parent_sha256,
    )


def _load_recovered_development_round(
    *,
    recovery: ReferenceMidRoundRecovery,
    config: ReferenceIterationConfig,
    agent_specs: tuple[MacroAgentSpec, ...],
    cold_start: CandidateSubmission,
    central_attestor: ReferenceFilesystemBundleAttestor,
    controller_root: Path,
    plugin_identity: Mapping[str, object],
    provenance: Mapping[str, object],
) -> _ValidatedRecoveredDevelopmentRound:
    manifest, manifest_bytes = _read_exact_json(
        recovery.manifest_path,
        maximum_bytes=_MAX_RECOVERY_MANIFEST_BYTES,
        expected_sha256=recovery.expected_manifest_sha256,
        label="recovery manifest",
    )
    _require_exact_keys(
        manifest,
        {
            "schema",
            "phase",
            "source",
            "target",
            "workers",
            "source_common_evidence_imported",
        },
        "recovery manifest",
    )
    if manifest["schema"] != REFERENCE_PRESELECTION_RECOVERY_SCHEMA:
        raise ReferenceIterationError("recovery manifest schema changed")
    if manifest["phase"] != REFERENCE_PRESELECTION_RECOVERY_PHASE:
        raise ReferenceIterationError("recovery boundary is not supported")
    if manifest["source_common_evidence_imported"] is not False:
        raise ReferenceIterationError(
            "source common-cohort evidence cannot be imported"
        )

    source = _require_exact_keys(
        manifest["source"],
        {
            "run_root",
            "campaign_seed_sha256",
            "config_sha256",
            "unified_commit",
            "campaign_id",
            "seed_base",
            "seed_stride",
            "plugin_identity",
            "cold_start_sha256",
        },
        "recovery source",
    )
    target = _require_exact_keys(
        manifest["target"],
        {
            "unified_commit",
            "plugin_identity",
            "official_scoring_protocol",
            "fresh_official_evaluation",
        },
        "recovery target",
    )
    source_root_raw = source["run_root"]
    if not isinstance(source_root_raw, str) or not Path(source_root_raw).is_absolute():
        raise ReferenceIterationError("recovery source run root must be absolute")
    source_root = Path(source_root_raw).expanduser().resolve()
    if source_root == Path(source_root.anchor) or not source_root.is_dir():
        raise ReferenceIterationError("recovery source run root is invalid")
    target_root = config.run_root.resolve()
    if (
        source_root == target_root
        or source_root in target_root.parents
        or target_root in source_root.parents
    ):
        raise ReferenceIterationError(
            "recovery source and target roots must be distinct siblings"
        )
    for forbidden in (
        source_root / "result",
        source_root / "_controller" / "sampling_ledger.json",
        source_root / "_controller" / "recovery_receipt.json",
        source_root / "_controller" / "agent_workers" / "round-0002",
    ):
        if forbidden.exists():
            raise ReferenceIterationError(
                "recovery source contains downstream controller state"
            )

    for name in (
        "campaign_seed_sha256",
        "config_sha256",
        "cold_start_sha256",
    ):
        if not _is_sha256(source[name]):
            raise ReferenceIterationError(f"recovery source {name} is invalid")
    for name in ("seed_base", "seed_stride"):
        if isinstance(source[name], bool) or not isinstance(source[name], int):
            raise ReferenceIterationError(f"recovery source {name} is invalid")
    if not _is_git_sha1(source["unified_commit"]):
        raise ReferenceIterationError("recovery source unified_commit is invalid")
    if (
        source["campaign_id"] != config.name
        or source["seed_base"] != config.seed_base
        or source["seed_stride"] != config.seed_stride
        or source["cold_start_sha256"] != cold_start.ref.bundle_sha256
        or source["config_sha256"] != provenance.get("config_sha256")
    ):
        raise ReferenceIterationError("recovery source campaign identity changed")

    source_identity = _recovery_plugin_identity(
        source["plugin_identity"], "recovery source plugin identity"
    )
    target_identity = _recovery_plugin_identity(
        target["plugin_identity"], "recovery target plugin identity"
    )
    if source_identity["plugin_id"] != target_identity["plugin_id"] or (
        source_identity["generator_id"] != target_identity["generator_id"]
    ):
        raise ReferenceIterationError("recovery changes the plugin contract")
    if target_identity != dict(plugin_identity):
        raise ReferenceIterationError("recovery target plugin identity changed")
    if (
        not _is_git_sha1(target["unified_commit"])
        or target["unified_commit"] != provenance.get("unified_commit")
        or target["official_scoring_protocol"] != "official-common-stream-v1"
        or target["fresh_official_evaluation"] is not True
    ):
        raise ReferenceIterationError("recovery target protocol identity changed")

    seed_record, _seed_bytes = _read_exact_json(
        source_root / "campaign_seed.json",
        maximum_bytes=_MAX_RECOVERY_MANIFEST_BYTES,
        expected_sha256=str(source["campaign_seed_sha256"]),
        label="source campaign seed",
    )
    if (
        seed_record.get("schema") != "unified_plasma_campaign_seed_v1"
        or seed_record.get("campaign_id") != config.name
        or seed_record.get("seed_base") != config.seed_base
        or seed_record.get("config_sha256") != source["config_sha256"]
        or seed_record.get("unified_commit") != source["unified_commit"]
        or seed_record.get("seed_derivation") != "macro_loop_seed_schedule_v1"
        or seed_record.get("controller_policy")
        != provenance.get("controller_policy")
    ):
        raise ReferenceIterationError("source campaign seed record changed")

    worker_rows = _sequence(manifest["workers"], "recovery workers")
    if len(worker_rows) != len(agent_specs):
        raise ReferenceIterationError(
            "recovery must contain every configured Agent exactly once"
        )
    round_root = source_root / "_controller" / "agent_workers" / "round-0001"
    if not round_root.is_dir():
        raise ReferenceIterationError("source round-one worker root is absent")
    validated: list[dict[str, object]] = []
    source_gpu_audits: list[dict[str, object]] = []
    expected_worker_roots: set[Path] = set()
    candidate_ids: set[str] = set()
    candidate_digests: set[str] = set()

    for expected_slot, (spec, raw_worker) in enumerate(
        zip(agent_specs, worker_rows, strict=True)
    ):
        worker = _require_exact_keys(
            raw_worker,
            {
                "slot",
                "agent_id",
                "request_path",
                "request_sha256",
                "report_path",
                "report_sha256",
                "candidate_id",
                "candidate_sha256",
                "bundle_path",
            },
            "recovery worker",
        )
        if worker["slot"] != expected_slot or worker["agent_id"] != spec.agent_id:
            raise ReferenceIterationError("recovery worker order changed")
        if (
            not _is_sha256(worker["request_sha256"])
            or not _is_sha256(worker["report_sha256"])
            or not _is_sha256(worker["candidate_sha256"])
            or not isinstance(worker["candidate_id"], str)
            or not worker["candidate_id"]
        ):
            raise ReferenceIterationError("recovery worker commitment is invalid")

        request_path = _recovery_relative_path(
            source_root, worker["request_path"], "recovery request path"
        )
        report_path = _recovery_relative_path(
            source_root, worker["report_path"], "recovery report path"
        )
        bundle_path = _recovery_relative_path(
            source_root, worker["bundle_path"], "recovery bundle path"
        )
        if (
            request_path.name != "request.json"
            or report_path.name != "report.json"
            or request_path.parent != report_path.parent
            or request_path.parent.parent != round_root
        ):
            raise ReferenceIterationError("recovery worker IPC layout changed")
        expected_worker_roots.add(request_path.parent)
        try:
            bundle_relative = bundle_path.relative_to(source_root)
        except ValueError as exc:
            raise ReferenceIterationError("recovery bundle escaped source") from exc
        if tuple(bundle_relative.parts[:2]) != (
            "_controller",
            "sealed_python_bundles",
        ):
            raise ReferenceIterationError(
                "recovery must use a controller-sealed source bundle"
            )

        request, request_bytes = _read_exact_json(
            request_path,
            maximum_bytes=_MAX_WORKER_REPORT_BYTES,
            expected_sha256=str(worker["request_sha256"]),
            label="source worker request",
        )
        _require_exact_keys(
            request,
            {
                "schema",
                "campaign",
                "round_index",
                "agent",
                "campaign_cold_start",
                "incumbent",
                "prior_handoff",
                "prior_leaderboard",
                "shared_candidates",
                "guidance_oracle",
            },
            "source worker request",
        )
        if request["schema"] != REFERENCE_PROCESS_WORKER_REQUEST_SCHEMA:
            raise ReferenceIterationError("source worker request schema changed")
        if request["campaign"] != {
            "name": config.name,
            "seed_base": config.seed_base,
            "seed_stride": config.seed_stride,
            "agent_count": config.agent_count,
        }:
            raise ReferenceIterationError("source worker campaign changed")
        if request["round_index"] != 1 or request["agent"] != {
            "agent_id": spec.agent_id,
            "name": spec.name,
            "prompt": spec.prompt,
        }:
            raise ReferenceIterationError("source worker round or Agent changed")
        request_cold = _private_candidate_from_payload(
            request["campaign_cold_start"]
        )
        request_incumbent = _private_candidate_from_payload(request["incumbent"])
        if (
            _candidate_commitment(request_cold)
            != _candidate_commitment(cold_start.ref)
            or _candidate_commitment(request_incumbent)
            != _candidate_commitment(cold_start.ref)
            or request["prior_handoff"] is not None
            or _sequence(request["prior_leaderboard"], "source leaderboard")
            or _sequence(request["shared_candidates"], "source shared candidates")
            or request["guidance_oracle"] is not None
        ):
            raise ReferenceIterationError(
                "source worker is not a clean first-round development"
            )

        report, report_bytes = _read_exact_json(
            report_path,
            maximum_bytes=_MAX_WORKER_REPORT_BYTES,
            expected_sha256=str(worker["report_sha256"]),
            label="source worker report",
        )
        _require_exact_keys(
            report,
            {
                "schema",
                "request_sha256",
                "status",
                "agent_id",
                "round_index",
                "plugin_identity",
                "gpu",
                "development",
            },
            "source worker report",
        )
        if (
            report["schema"] != REFERENCE_PROCESS_WORKER_REPORT_SCHEMA
            or report["request_sha256"] != worker["request_sha256"]
            or report["status"] != "complete"
            or report["agent_id"] != spec.agent_id
            or report["round_index"] != 1
            or report["plugin_identity"] != source_identity
        ):
            raise ReferenceIterationError("source worker report binding changed")
        gpu_raw = report["gpu"]
        if not isinstance(gpu_raw, Mapping):
            raise ReferenceIterationError("source worker GPU report is absent")
        gpu = _validated_gpu_report(
            gpu_raw,
            expected_pid=gpu_raw.get("pid"),
            expected_token=gpu_raw.get("assigned_token"),
            expected_slot=expected_slot,
        )
        source_gpu_audits.append(
            {
                "pid": gpu["pid"],
                "gpu_device": gpu["assigned_token"],
                "worker_started_monotonic_ns": gpu["started_monotonic_ns"],
                "worker_finished_monotonic_ns": gpu["finished_monotonic_ns"],
                "cuda_device_uuid": gpu.get("device_uuid"),
            }
        )

        development = _require_exact_keys(
            report["development"],
            {
                "portfolio",
                "development_case_ids",
                "sampled_case_ids",
                "sampling_attempts",
                "search_completion",
                "failure",
            },
            "source worker development",
        )
        portfolio = _sequence(development["portfolio"], "source portfolio")
        if len(portfolio) != 1 or development["failure"] is not None:
            raise ReferenceIterationError(
                "source worker did not complete one official nomination"
            )
        nomination = _require_exact_keys(
            portfolio[0], {"candidate", "roles"}, "source nomination"
        )
        claimed = _private_candidate_from_payload(nomination["candidate"])
        roles = _string_sequence(nomination["roles"], "source nomination roles")
        if (
            not roles
            or claimed.agent_id != spec.agent_id
            or claimed.round_index != 1
            or claimed.parent_sha256 != cold_start.ref.bundle_sha256
            or claimed.candidate_id != worker["candidate_id"]
            or claimed.bundle_sha256 != worker["candidate_sha256"]
        ):
            raise ReferenceIterationError("source nomination lineage changed")
        historical_completion = _search_completion_from_payload(
            development["search_completion"]
        )
        if (
            historical_completion is None
            or historical_completion.agent_id != spec.agent_id
            or historical_completion.final_candidate_sha256
            != claimed.bundle_sha256
            or historical_completion.sufficient is not True
        ):
            raise ReferenceIterationError("source search completion changed")
        if claimed.candidate_id in candidate_ids or (
            claimed.bundle_sha256 in candidate_digests
        ):
            raise ReferenceIterationError("recovery nominations are not distinct")
        candidate_ids.add(claimed.candidate_id)
        candidate_digests.add(claimed.bundle_sha256)
        development_ids = _sequence(
            development["development_case_ids"], "historical development cases"
        )
        sampled_ids = _sequence(
            development["sampled_case_ids"], "historical sampled cases"
        )
        sampling_attempts = _sequence(
            development["sampling_attempts"], "historical sampling attempts"
        )
        validated.append(
            {
                "slot": expected_slot,
                "agent": spec,
                "request_path": request_path,
                "request_sha256": worker["request_sha256"],
                "request_bytes": request_bytes,
                "report_path": report_path,
                "report_sha256": worker["report_sha256"],
                "report_bytes": report_bytes,
                "bundle_path": bundle_path,
                "candidate_id": claimed.candidate_id,
                "candidate_sha256": claimed.bundle_sha256,
                "roles": roles,
                "gpu": gpu,
                "historical_development_case_count": len(development_ids),
                "historical_sampled_case_count": len(sampled_ids),
                "historical_sampling_attempt_count": len(sampling_attempts),
                "historical_search_completion": _search_completion_payload(
                    historical_completion
                ),
            }
        )

    source_worker_entries = {
        child.resolve() for child in round_root.iterdir()
    }
    if source_worker_entries != expected_worker_roots:
        raise ReferenceIterationError("source round-one worker set changed")
    _verify_worker_topology(source_gpu_audits)

    results: list[MacroAgentDevelopmentResult] = []
    receipt_workers: list[dict[str, object]] = []
    evidence_root = controller_root / "recovery_evidence"
    evidence_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    _atomic_write(evidence_root / "manifest.json", manifest_bytes, mode=0o400)
    for row in validated:
        spec = row["agent"]
        assert isinstance(spec, MacroAgentSpec)
        attestation = central_attestor.attest(
            AgentBundleSubmission(
                candidate_id=str(row["candidate_id"]),
                bundle_uri=str(row["bundle_path"]),
            ),
            round_index=1,
            agent_id=spec.agent_id,
            parent_sha256=cold_start.ref.bundle_sha256,
        )
        candidate = candidate_ref_from_attestation(attestation)
        if candidate.bundle_sha256 != row["candidate_sha256"]:
            raise ReferenceIterationError("recovered candidate digest changed")
        results.append(
            MacroAgentDevelopmentResult(
                agent_id=spec.agent_id,
                portfolio=((_candidate_submission_from_ref(candidate), row["roles"]),),
                development_case_ids=(),
                sampled_case_ids=(),
                sampling_attempts=(),
                search_completion=None,
                failure=None,
            )
        )
        prefix = f"agent-{int(row['slot']):02d}"
        _atomic_write(
            evidence_root / f"{prefix}-request.json",
            row["request_bytes"],
            mode=0o400,
        )
        _atomic_write(
            evidence_root / f"{prefix}-report.json",
            row["report_bytes"],
            mode=0o400,
        )
        gpu = row["gpu"]
        receipt_workers.append(
            {
                "slot": row["slot"],
                "agent_id": spec.agent_id,
                "request_sha256": row["request_sha256"],
                "report_sha256": row["report_sha256"],
                "candidate_id": candidate.candidate_id,
                "candidate_sha256": candidate.bundle_sha256,
                "source_gpu": {
                    "pid": gpu["pid"],
                    "assigned_token": gpu["assigned_token"],
                    "device_uuid": gpu.get("device_uuid"),
                    "started_monotonic_ns": gpu["started_monotonic_ns"],
                    "finished_monotonic_ns": gpu["finished_monotonic_ns"],
                },
                "historical_development_case_count": row[
                    "historical_development_case_count"
                ],
                "historical_sampled_case_count": row[
                    "historical_sampled_case_count"
                ],
                "historical_sampling_attempt_count": row[
                    "historical_sampling_attempt_count"
                ],
                "historical_search_completion": row[
                    "historical_search_completion"
                ],
            }
        )

    audit = {
        "schema": "reference-process-preselection-recovery-receipt-v1",
        "phase": REFERENCE_PRESELECTION_RECOVERY_PHASE,
        "manifest_sha256": recovery.expected_manifest_sha256,
        "source_run_root": str(source_root),
        "source_campaign_seed_sha256": source["campaign_seed_sha256"],
        "source_unified_commit": source["unified_commit"],
        "target_unified_commit": target["unified_commit"],
        "source_plugin_identity": source_identity,
        "target_plugin_identity": target_identity,
        "source_worker_topology_verified": True,
        "source_common_evidence_imported": False,
        "source_sampling_audits_imported": False,
        "source_search_completion_imported": False,
        "target_official_evaluation": "fresh_streamed_common_cohort",
        "current_round_one_workers_spawned": False,
        "workers": receipt_workers,
    }
    _atomic_write(
        controller_root / "recovery_receipt.json",
        (
            json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8"),
        mode=0o400,
    )
    return _ValidatedRecoveredDevelopmentRound(
        results=tuple(results),
        audit=MappingProxyType(audit),
    )


def run_reference_process_agent_worker(
    *,
    config: ReferenceIterationConfig,
    plugin: MacroLoopPlugin,
    request: Mapping[str, object],
    request_sha256: str,
    worker_root: Path,
    gpu_audit: Mapping[str, object],
    legacy_macro_reference: bool = False,
) -> Mapping[str, object]:
    """Execute exactly one private development phase inside a legacy worker."""

    if legacy_macro_reference is not True:
        raise PermissionError(
            "legacy Macro worker execution is disabled by default; pass "
            "legacy_macro_reference=True for an explicit history/control-arm run"
        )

    if request.get("schema") != REFERENCE_PROCESS_WORKER_REQUEST_SCHEMA:
        raise ReferenceIterationError("worker request schema changed")
    if len(request_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in request_sha256
    ):
        raise ReferenceIterationError("worker request digest is invalid")
    campaign = request.get("campaign")
    if not isinstance(campaign, Mapping) or dict(campaign) != {
        "name": config.name,
        "seed_base": config.seed_base,
        "seed_stride": config.seed_stride,
        "agent_count": config.agent_count,
    }:
        raise ReferenceIterationError("worker campaign identity changed")
    round_index = request.get("round_index")
    if (
        isinstance(round_index, bool)
        or not isinstance(round_index, int)
        or not 1 <= round_index <= config.round_count
    ):
        raise ReferenceIterationError("worker round index is invalid")
    agent_raw = request.get("agent")
    if not isinstance(agent_raw, Mapping):
        raise ReferenceIterationError("worker Agent specification is absent")
    expected_specs = {
        spec.agent_id: spec for spec in _reference_agent_specs(config)
    }
    agent_id = agent_raw.get("agent_id")
    if not isinstance(agent_id, str) or agent_id not in expected_specs:
        raise ReferenceIterationError("worker Agent identity is invalid")
    agent = expected_specs[agent_id]
    if dict(agent_raw) != {
        "agent_id": agent.agent_id,
        "name": agent.name,
        "prompt": agent.prompt,
    }:
        raise ReferenceIterationError("worker Agent specification changed")

    campaign_cold_ref = _private_candidate_from_payload(
        request.get("campaign_cold_start")
    )
    incumbent_ref = _private_candidate_from_payload(request.get("incumbent"))
    prior_handoff = _private_handoff_from_payload(request.get("prior_handoff"))
    if round_index == 1 and prior_handoff is not None:
        raise ReferenceIterationError("first-round worker received a prior handoff")
    if round_index > 1 and (
        prior_handoff is None or prior_handoff.round_index != round_index - 1
    ):
        raise ReferenceIterationError("worker prior handoff round changed")
    prior_leaderboard = tuple(
        _private_leaderboard_from_payload(row)
        for row in _sequence(
            request.get("prior_leaderboard"), "prior leaderboard"
        )
    )
    shared_candidates = tuple(
        _private_candidate_from_payload(row)
        for row in _sequence(
            request.get("shared_candidates"), "shared candidates"
        )
    )

    root = Path(worker_root).resolve()
    if root == Path(root.anchor):
        raise ReferenceIterationError("worker root must be a specific directory")
    cohort_root = None
    if config.cohort_store_root is not None:
        cohort_environment = os.environ.get("PLASMA_COHORT_STORAGE_ROOT")
        if not cohort_environment:
            raise ReferenceIterationError("worker cohort storage root is absent")
        cohort_root = Path(cohort_environment).expanduser().resolve()
        if cohort_root == Path(cohort_root.anchor):
            raise ReferenceIterationError("worker cohort root must be specific")
    worker_config = replace(
        config,
        run_root=root,
        cohort_store_root=cohort_root,
        agent_count=1,
        agent_roles=(),
        guidance_protocol=None,
    )
    attestor = ReferenceFilesystemBundleAttestor(
        root / "sealed_python_bundles"
    )
    cold_start = _candidate_submission_from_ref(campaign_cold_ref)
    incumbent = _candidate_submission_from_ref(incumbent_ref)
    loop_config = _reference_loop_config(worker_config, (agent,))
    program = _reference_agent_program(
        config=worker_config,
        run_root=root / "agent_runtime",
        run_id=(
            f"{config.name}:{config.run_root.name}:"
            f"round-{round_index}:{agent.agent_id}"
        ),
        attestor=attestor,
    )
    runner = MacroClosedLoopRunner(
        legacy_macro_reference=True,
        config=loop_config,
        plugin=plugin,
        cold_start=cold_start,
        agents={agent.agent_id: program},
        cohort_store_root=cohort_root,
    )
    guidance_oracle = ReferenceGuidanceOracleProxy.from_worker_capability(
        request.get("guidance_oracle"),
        worker_root=root,
        timeout_seconds=config.agent_hard_timeout_seconds,
    )
    result = runner.run_agent_development(
        round_index=round_index,
        agent=agent,
        incumbent=incumbent,
        prior_handoff=prior_handoff,
        prior_leaderboard=prior_leaderboard,
        shared_candidates=shared_candidates,
        guidance_oracle=guidance_oracle,
    )
    gpu = dict(gpu_audit)
    gpu["finished_monotonic_ns"] = time.monotonic_ns()
    return {
        "schema": REFERENCE_PROCESS_WORKER_REPORT_SCHEMA,
        "request_sha256": request_sha256,
        "status": "complete",
        "agent_id": agent.agent_id,
        "round_index": round_index,
        "plugin_identity": {
            "plugin_id": plugin.plugin_id,
            "generator_id": plugin.cases.generator_id,
            "generator_sha256": plugin.cases.artifact_sha256,
        },
        "gpu": gpu,
        "development": _development_result_payload(result),
    }


def _candidate_submission_from_ref(ref: CandidateRef) -> CandidateSubmission:
    if ref.bundle_uri is None:
        raise ReferenceIterationError("sealed candidate URI is absent")
    return _reference_candidate_submission(ref)


def run_reference_agent_iteration(
    *,
    config: ReferenceIterationConfig,
    plugin: MacroLoopPlugin,
    provenance: Mapping[str, object],
    process_workers: ReferenceProcessWorkerConfig | None = None,
    guidance_backend: MacroGuidanceBackend | None = None,
    recovery: ReferenceMidRoundRecovery | None = None,
    legacy_macro_reference: bool = False,
) -> Mapping[str, object]:
    """Replay one legacy formal Macro iteration in a new target root.

    This history/control-arm API is not the canonical product entry; new runs
    use CampaignController.
    """

    if legacy_macro_reference is not True:
        raise PermissionError(
            "legacy Macro iteration is disabled by default; pass "
            "legacy_macro_reference=True for an explicit history/control-arm run"
        )

    if recovery is not None:
        if process_workers is None:
            raise ReferenceIterationError(
                "mid-round recovery requires process-isolated Agent workers"
            )
        if config.guidance_protocol is not None or guidance_backend is not None:
            raise ReferenceIterationError(
                "guided development recovery requires a bound guidance snapshot"
            )

    started = time.monotonic()
    config.run_root.mkdir(parents=True, exist_ok=True)
    controller_root = config.run_root / "_controller"
    result_root = config.run_root / "result"
    if controller_root.exists() or result_root.exists():
        raise FileExistsError("reference iteration output already exists")

    attestor = ReferenceFilesystemBundleAttestor(
        controller_root / "sealed_python_bundles"
    )
    cold_attestation = attestor.attest(
        AgentBundleSubmission(
            candidate_id=f"{plugin.plugin_id}-fresh-cold-start",
            bundle_uri=str(config.cold_start_candidate),
        ),
        round_index=0,
        agent_id=_COLD_AGENT_ID,
        parent_sha256=None,
    )
    cold_ref = candidate_ref_from_attestation(cold_attestation)
    cold_start = _candidate_submission_from_ref(cold_ref)

    agent_specs = _reference_agent_specs(config)
    loop_config = _reference_loop_config(config, agent_specs)
    plugin_identity = None
    recovered_round = None
    if process_workers is not None:
        plugin_identity = {
            "plugin_id": plugin.plugin_id,
            "generator_id": plugin.cases.generator_id,
            "generator_sha256": plugin.cases.artifact_sha256,
        }
        if recovery is not None:
            recovered_round = _load_recovered_development_round(
                recovery=recovery,
                config=config,
                agent_specs=agent_specs,
                cold_start=cold_start,
                central_attestor=attestor,
                controller_root=controller_root,
                plugin_identity=plugin_identity,
                provenance=provenance,
            )
    if process_workers is None:
        agent_program = _reference_agent_program(
            config=config,
            run_root=controller_root / "agent_runtime",
            run_id=f"{config.name}:{config.run_root.name}",
            attestor=attestor,
        )
        agents = {spec.agent_id: agent_program for spec in agent_specs}
        development_executor = None
    else:
        if len(process_workers.gpu_devices) != len(agent_specs):
            raise ValueError("one distinct worker GPU is required per Agent")
        agents = {}
        assert plugin_identity is not None
        development_executor = ReferenceProcessRoundDeveloper(
            execution=process_workers,
            iteration_config=config,
            loop_config=loop_config,
            controller_root=controller_root,
            central_attestor=attestor,
            campaign_cold_start=cold_start,
            plugin_identity=plugin_identity,
            recovered_round=recovered_round,
        )
    runner = MacroClosedLoopRunner(
        legacy_macro_reference=True,
        config=loop_config,
        plugin=plugin,
        cold_start=cold_start,
        agents=agents,
        cohort_store_root=config.cohort_store_root,
        development_executor=development_executor,
        guidance_backend=guidance_backend,
    )
    try:
        result = runner.run()
    except RecoverableSamplingRejection as exc:
        _write_sampling_ledger(
            controller_root,
            config,
            status="exhausted",
            attempts=runner.sampling_attempts,
            failure_audit=exc.audit,
        )
        raise
    _write_sampling_ledger(
        controller_root,
        config,
        status="complete",
        attempts=result.sampling_attempts,
    )
    wall_time_seconds = max(0.0, time.monotonic() - started)
    effective_provenance = dict(provenance)
    if development_executor is not None:
        if recovered_round is not None:
            effective_provenance["preselection_recovery"] = dict(
                recovered_round.audit
            )
        effective_provenance["agent_worker_execution"] = (
            development_executor.audit_payload()
        )
    return _publish_result(
        config=config,
        recovery_audit=(recovered_round.audit if recovered_round else None),
        result=result,
        provenance=effective_provenance,
        wall_time_seconds=wall_time_seconds,
    )


def _write_sampling_ledger(
    controller_root: Path,
    config: ReferenceIterationConfig,
    *,
    status: str,
    attempts=(),
    failure_audit: Mapping[str, object] | None = None,
) -> None:
    if status not in {"complete", "exhausted"}:
        raise ValueError("sampling ledger status is invalid")
    rows = [
        {
            "round_index": row.round_index,
            "cohort": row.cohort.value,
            "attempt_index": row.attempt_index,
            "maximum_attempts": row.maximum_attempts,
            "base_seed": row.base_seed,
            "perturbed_seed": row.perturbed_seed,
            "root_case_ids": list(row.root_case_ids),
            "accepted": row.accepted,
            "rejection_audit": (
                dict(row.rejection_audit)
                if row.rejection_audit is not None
                else None
            ),
        }
        for row in attempts
    ]
    exhausted_attempts = (
        list(failure_audit.get("attempts", ()))
        if isinstance(failure_audit, Mapping)
        else []
    )
    payload = {
        "schema": "controller_private_paired_sampling_ledger_v1",
        "status": status,
        "paired_perturbations": config.paired_perturbations,
        "paired_sampling_max_attempts": config.paired_sampling_max_attempts,
        "attempt_count": len(rows) if rows else len(exhausted_attempts),
        "rejection_count": (
            sum(not row["accepted"] for row in rows)
            if rows
            else len(exhausted_attempts)
        ),
        "attempts": rows,
        "failure_audit": (
            dict(failure_audit)
            if failure_audit is not None
            else None
        ),
        "agent_visible": False,
    }
    _atomic_write(
        controller_root / "sampling_ledger.json",
        (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode("utf-8"),
        mode=0o600,
    )


def _publish_result(
    *,
    config: ReferenceIterationConfig,
    result: MacroLoopResult,
    provenance: Mapping[str, object],
    wall_time_seconds: float,
    recovery_audit: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    if result.execution_assurance is not ExecutionAssurance.REFERENCE_IN_PROCESS:
        raise ReferenceIterationError("reference run changed execution assurance")
    if result.formal_claims_allowed:
        raise ReferenceIterationError("reference run cannot publish formal claims")

    if result.selected.bundle_uri is None:
        raise ReferenceIterationError(
            "selected candidate has no sealed bundle URI"
        )
    result_root = config.run_root / "result"
    result_root.mkdir(parents=True, exist_ok=False)
    model_root = result_root / "model"
    with tempfile.TemporaryDirectory(
        prefix=".model-stage-",
        dir=result_root,
    ) as temporary:
        staged_root = Path(temporary) / "model"
        staged_entrypoint = _stage_reference_candidate_bundle(
            result.selected.bundle_uri,
            staged_root,
            expected_sha256=result.selected.bundle_sha256,
            legacy_filename="submission.py",
        )
        bundle = inspect_candidate_bundle(staged_entrypoint)
        entrypoint_relative = staged_entrypoint.relative_to(staged_root)
        artifact_files = [member.path for member in bundle.members]
        artifact_bytes = bundle.total_bytes
        artifact_tree_sha256 = bundle.tree_sha256
        staged_root.chmod(0o700)
        os.replace(staged_root, model_root)
        model_root.chmod(0o500)
    model_path = model_root / entrypoint_relative
    (
        selected_error,
        cold_start_error,
        active_objective_improvement,
    ) = _terminal_claim_metrics(result.terminal)
    terminal_acceptance = result.terminal.supports_selected_over_cold_start
    overall_improvement = (
        active_objective_improvement if terminal_acceptance else None
    )
    recovered = recovery_audit is not None
    payload = {
        "schema": REFERENCE_ITERATION_RESULT_SCHEMA,
        "protocol": REFERENCE_ITERATION_PROTOCOL,
        "name": config.name,
        "problem": result.plugin_id,
        "kind": result.public_result_kind,
        "task": result.public_result_task,
        "matched_quantity_metrics": list(result.published_metric_names),
        "model": str(model_path),
        "primary_metric": result.macro_contract_id,
        "primary_error": selected_error,
        "cold_start_error": cold_start_error,
        "overall_improvement": overall_improvement,
        "active_objective_improvement": active_objective_improvement,
        "terminal_acceptance": terminal_acceptance,
        "rounds_completed": len(result.rounds),
        "wall_time_seconds": float(wall_time_seconds),
        "execution_assurance": result.execution_assurance.value,
        "formal_claims_allowed": False,
        "reference_only": True,
        "fresh_agent_start": not recovered,
        "fresh_generator_draws": not recovered,
        "fresh_official_generator_draws": True,
        "recovered_round_one_development": recovered,
        "recovery": (
            dict(recovery_audit)
            if recovery_audit is not None
            else None
        ),
        "paired_perturbations": result.paired_perturbations,
        "sampling": {
            "attempt_count": len(result.sampling_attempts),
            "rejection_count": sum(
                not row.accepted for row in result.sampling_attempts
            ),
        },
        "limitations": [
            "Agent and candidate execution share the controller user and node.",
            "Candidate time budgets are cooperative in this reference runtime.",
            "This experiment cannot support a formal isolation claim.",
            (
                "Campaign seed and case artifacts are not isolated against "
                "same-user path traversal in this reference runtime."
            ),
            *(
                ["Round-one Agent development was recovered from an exact, "
                 "source-transition-bound pre-selection checkpoint."]
                if recovered else []
            ),
        ],
        "generator": {
            "generator_id": result.generator_id,
            "generator_sha256": result.generator_sha256,
        },
        "macro_objective": {
            "contract_id": result.macro_contract_id,
            "published_metric_names": list(result.published_metric_names),
            "paired_objective": {
                "mode": result.paired_objective_policy.mode.value,
                "base_weight": result.paired_objective_policy.base_weight,
                "perturbed_weight": result.paired_objective_policy.perturbed_weight,
                "guardrail_relative_margin": (
                    result.paired_objective_policy.guardrail_relative_margin
                ),
                "guardrail_confidence_z": result.paired_objective_policy.guardrail_confidence_z,
            },
        },
        "artifact": {
            "path": str(model_root),
            "entrypoint": entrypoint_relative.as_posix(),
            "files": artifact_files,
            "bytes": artifact_bytes,
            "tree_sha256": artifact_tree_sha256,
            "sha256": result.selected.bundle_sha256,
            "archive_sha256": result.selected.bundle_sha256,
            "single_file": len(artifact_files) == 1,
        },
        "candidates": {
            "cold_start": _candidate_payload(result.cold_start),
            "selected": _candidate_payload(result.selected),
        },
        "controller_policy": {
            "round_count": config.round_count,
            "agent_count": config.agent_count,
            "agent_roles": list(config.agent_roles),
            "search_protocol": {
                "min_distinct_screened_candidates": (
                    config.search_policy.min_distinct_screened_candidates
                ),
                "min_full_stage_candidates": (
                    config.search_policy.min_full_stage_candidates
                ),
                "min_released_roots": config.search_policy.min_released_roots,
                "require_final_full_stage": (
                    config.search_policy.require_final_full_stage
                ),
                "leaderboard_limit": config.search_policy.leaderboard_limit,
                "cross_share_after_round": config.search_policy.cross_share_after_round,
            },
            "guidance": (
                None
                if config.guidance_protocol is None
                else {
                    "cadence": "once_at_round_boundary",
                    "guidance_config_sha256": (
                        config.guidance_protocol.guidance_config_sha256
                    ),
                    "reward_feedback_case_count": (
                        config.guidance_protocol.reward_feedback_case_count
                    ),
                    "official_round_case_count": (
                        config.guidance_protocol.official_round_case_count
                    ),
                    "official_terminal_case_count": (
                        config.guidance_protocol.official_terminal_case_count
                    ),
                    "training_budget": (
                        config.guidance_protocol.training_budget.identity_payload()
                    ),
                    "inference_wall_budget_seconds_per_request": (
                        config.guidance_protocol.inference_wall_budget_seconds
                    ),
                    "inference_compute_capacity": (
                        config.guidance_protocol.inference_compute_capacity
                    ),
                    "online_development_sample_quota": None,
                }
            ),
            "promotion_program": getattr(
                result,
                "promotion_program_id",
                config.promotion_program.program_id,
            ),
            "statistical_unit": "independent_root",
            "case_count_unit": "member_case",
            "agent_active_timeout_seconds": config.agent_timeout_seconds,
            "agent_hard_wall_timeout_seconds": (
                config.agent_hard_timeout_seconds
            ),
            "agent_timeout_basis": (
                "active_wall_excluding_credit_consuming_controller_service"
            ),
            "development_screen_timeout_seconds": (
                config.development_screen_timeout_seconds
            ),
            "root_member_count": 2 if config.paired_perturbations else 1,
            "development_credit_scope": (
                "per_request_capacity_no_lifetime_quota"
                if config.development_screen_root_stages
                else "per_agent_per_round_case_credits"
            ),
            "development_mode": (
                "fresh_completed_release_and_screen"
                if config.development_screen_root_stages
                else "fixed_development_evaluator"
            ),
            "development_case_credits_active": not bool(
                config.development_screen_root_stages
            ),
            "development_case_count": config.development_case_count,
            "development_pool_multiplier": config.development_pool_multiplier,
            "development_screen_representative_limit": (
                config.development_screen_representative_limit
            ),
            "development_release_root_limit": (
                config.development_release_root_limit
            ),
            "development_screen_root_stages": list(
                config.development_screen_root_stages
            ),
            "development_screen_root_credit_limit": (
                config.development_screen_root_credit_limit
            ),
            "development_screen_confidence_z": (
                config.development_screen_confidence_z
            ),
            "development_screen_continuation_max_mean_ratio": (
                config.development_screen_continuation_max_mean_ratio
            ),
            "allow_development_truth": config.allow_development_truth,
            "selection_case_count": config.selection_case_count,
            "evidence_stages": [
                {
                    "case_count": stage.case_count,
                    "confidence_z": stage.confidence_z,
                    "min_mean_improvement": stage.min_mean_improvement,
                    "min_win_rate": stage.min_win_rate,
                    "require_cvar_no_regression": (
                        stage.require_cvar_no_regression
                    ),
                }
                for stage in config.evidence_stages
            ],
            "terminal_case_count": config.terminal_case_count,
            "development_case_credits": config.development_case_credits,
            "cvar_tail_fraction": config.cvar_tail_fraction,
            "terminal_confidence_z": config.terminal_confidence_z,
            "terminal_min_win_rate": config.terminal_min_win_rate,
            "terminal_require_cvar_no_regression": (
                config.terminal_require_cvar_no_regression
            ),
            "candidate_compute_budget": config.candidate_compute_budget,
            "candidate_query_seconds_per_point_limit": (
                config.candidate_query_seconds_per_point_limit
            ),
            "paired_perturbations": config.paired_perturbations,
            "paired_objective": {
                "mode": config.paired_objective_policy.mode.value,
                "base_weight": config.paired_objective_policy.base_weight,
                "perturbed_weight": config.paired_objective_policy.perturbed_weight,
                "guardrail_relative_margin": (
                    config.paired_objective_policy.guardrail_relative_margin
                ),
                "guardrail_confidence_z": config.paired_objective_policy.guardrail_confidence_z,
            },
            "paired_sampling_max_attempts": (
                config.paired_sampling_max_attempts
            ),
            "seed_base": config.seed_base,
            "seed_stride": config.seed_stride,
        },
        "rounds": [_round_payload(row) for row in result.rounds],
        "terminal": _terminal_payload(result),
        "provenance": dict(provenance),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    output = config.run_root / "result" / "result.json"
    _atomic_write(
        output,
        (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
        mode=0o644,
    )
    return payload


def _round_payload(row) -> Mapping[str, object]:
    finalists = []
    for finalist in row.finalist_evaluations:
        finalists.append(
            {
                "candidate": _candidate_payload(finalist.candidate),
                "eligible": finalist.score.eligible,
                "errors": list(finalist.score.errors),
                "aggregate": _aggregate_payload(finalist.aggregate),
                "distribution_aggregates": _paired_aggregates_payload(
                    finalist.objective_aggregates
                ),
                "perturbed_guardrail": _guardrail_payload(
                    finalist.perturbed_guardrail
                ),
            }
        )
    leaderboard = [
        {
            "rank": entry.rank,
            "candidate": _candidate_payload(entry.candidate),
            "agent_id": entry.agent_id,
            "eligible": entry.eligible,
            "loss": {
                "mean": entry.mean,
                "cvar": entry.cvar,
                "max": entry.maximum,
                "generated_mean": entry.generated_mean,
                "perturbed_mean": entry.perturbed_mean,
            },
            "guardrail_passed": entry.guardrail_passed,
            "roles": list(entry.roles),
        }
        for entry in row.leaderboard
    ]
    search_completion = {
        agent_id: {
            "agent_id": audit.agent_id,
            "distinct_screened_candidates": (
                audit.distinct_screened_candidates
            ),
            "full_stage_candidates": audit.full_stage_candidates,
            "released_roots": audit.released_roots,
            "final_candidate_sha256": audit.final_candidate_sha256,
            "final_completed_root_count": audit.final_completed_root_count,
            "sufficient": audit.sufficient,
            "failures": list(audit.failures),
        }
        for agent_id, audit in sorted(row.search_completion.items())
    }
    return {
        "round_index": row.round_index,
        "development_case_count": len(row.development_case_ids),
        "submitted_candidates": [
            _candidate_payload(candidate) for candidate in row.candidate_refs
        ],
        "candidate_portfolio_roles": {
            candidate_id: list(roles)
            for candidate_id, roles in row.candidate_portfolio_roles.items()
        },
        "agent_failures": dict(row.agent_failures),
        "leaderboard": leaderboard,
        "search_completion": search_completion,
        "finalists": finalists,
        "challenger": _candidate_payload(row.challenger),
        "selected": _candidate_payload(row.selected),
        "promotion": _promotion_payload(row.promotion),
        "promotion_distribution_aggregates": {
            "challenger": _paired_aggregates_payload(
                row.promotion_challenger_objective_aggregates
            ),
            "incumbent": _paired_aggregates_payload(
                row.promotion_incumbent_objective_aggregates
            ),
        },
        "promotion_perturbed_guardrail": _guardrail_payload(
            row.promotion_perturbed_guardrail
        ),
        "guidance_evaluation": _guidance_evaluation_payload(
            getattr(row, "guidance_evaluation", None)
        ),
        "promotion_reports": _promotion_reports_payload(
            getattr(row, "promotion_reports", ())
        ),
        "promotion_decision": _promotion_decision_payload(
            getattr(row, "promotion_decision", None)
        ),
        "handoff": {
            "sha256": row.handoff.handoff_sha256,
            "previous_sha256": row.handoff.previous_handoff_sha256,
            "promoted": row.handoff.promoted,
            "decision_reason": row.handoff.decision_reason,
            "evidence_case_count": row.handoff.feedback.evidence_case_count,
            "confidence_group_size": row.handoff.feedback.confidence_group_size,
            "paired_mean_improvement": (
                row.handoff.feedback.paired_mean_improvement
            ),
            "paired_confidence_lower": (
                row.handoff.feedback.paired_confidence_lower
            ),
            "paired_win_rate": row.handoff.feedback.paired_win_rate,
            "perturbed_guardrail_passed": (
                row.handoff.feedback.perturbed_guardrail_passed
            ),
            "perturbed_guardrail_confidence_lower": (
                row.handoff.feedback.perturbed_guardrail_confidence_lower
            ),
        },
    }


def _promotion_reports_payload(reports) -> list[Mapping[str, object]]:
    return [
        {
            "candidate_id": row.candidate_id,
            "candidate_sha256": row.candidate_digest,
            "technically_eligible": row.technically_eligible,
            "distributions": {
                name: {
                    "case_count": metric.case_count,
                    "mean": metric.mean,
                    "cvar": metric.cvar,
                    "max": metric.maximum,
                }
                for name, metric in sorted(row.distributions.items())
            },
            "guardrails": dict(row.guardrails),
            "failures": list(row.failures),
        }
        for row in reports
    ]


def _promotion_decision_payload(decision) -> Mapping[str, object] | None:
    if decision is None:
        return None
    return {
        "selected_sha256": decision.selected_digest,
        "promoted": decision.promoted,
        "reason": decision.reason,
        "ranking": [
            {
                "candidate_sha256": row.candidate_digest,
                "rank": row.rank,
                "eligible": row.eligible,
                "reason": row.reason,
            }
            for row in decision.ranking
        ],
    }


def _promotion_payload(promotion) -> Mapping[str, object] | None:
    if promotion is None:
        return None
    return {
        "incumbent_candidate_id": promotion.incumbent_artifact_id,
        "incumbent_sha256": promotion.incumbent_digest,
        "selected_candidate_id": promotion.selected_artifact_id,
        "selected_sha256": promotion.selected_digest,
        "promoted": promotion.promoted,
        "reason": promotion.reason,
        "comparisons": [
            {
                "case_count": row.stage.case_count,
                "confidence_group_size": row.confidence_group_size,
                "mean_improvement": row.mean_improvement,
                "confidence_lower": row.confidence_lower,
                "confidence_upper": row.confidence_upper,
                "win_rate": row.win_rate,
                "passed": row.passed,
                "failures": _public_comparison_failures(row.failures),
                "candidate_aggregate": _aggregate_payload(
                    row.candidate_aggregate
                ),
                "incumbent_aggregate": _aggregate_payload(
                    row.incumbent_aggregate
                ),
            }
            for row in promotion.comparisons
        ],
    }


def _terminal_claim_metrics(terminal) -> tuple[
    float | None,
    float | None,
    float | None,
]:
    if not (
        terminal.cold_start_score.eligible
        and terminal.selected_score.eligible
    ):
        return None, None, None
    cold_mean = fmean(terminal.cold_start_score.errors)
    selected_mean = fmean(terminal.selected_score.errors)
    return (
        selected_mean,
        cold_mean,
        _relative_improvement(cold_mean, selected_mean),
    )


def _guidance_evaluation_payload(value) -> Mapping[str, object] | None:
    """Publish aggregate guide evidence without hidden cases or answer locators."""

    if value is None:
        return None
    request = value.snapshot.training_request

    def cohort_metrics_payload(metrics) -> Mapping[str, object]:
        return {
            "cohort_id": metrics.cohort_id,
            "kind": metrics.kind.value,
            "generator_or_guidance_sha256": (
                metrics.generator_or_guidance_sha256
            ),
            "values": {
                name: float(metric)
                for name, metric in sorted(metrics.values.items())
            },
        }

    candidates = []
    for row in value.candidates:
        separated = row.separated_metrics
        candidates.append(
            {
                "candidate": _candidate_payload(row.candidate),
                "eligible": row.guided_score.eligible,
                "guided_aggregate": _aggregate_payload(
                    row.guided_aggregate
                ),
                "separated_metrics": (
                    None
                    if separated is None
                    else {
                        "average_base": cohort_metrics_payload(
                            separated.average_base
                        ),
                        "guided_adversarial": cohort_metrics_payload(
                            separated.guided_adversarial
                        ),
                    }
                ),
            }
        )
    return {
        "snapshot": {
            "snapshot_sha256": value.snapshot.snapshot_sha256,
            "backend_id": value.snapshot.backend_id,
            "source_round_index": request.source_round_index,
            "target_round_index": request.target_round_index,
            "target_solver_sha256": request.target_solver_sha256,
            "base_generator_sha256": request.base_generator_sha256,
            "reward_matcher_definition_sha256": (
                request.reward_matcher_definition_sha256
            ),
            "guidance_config_sha256": request.guidance_config_sha256,
            "training_budget": request.training_budget.identity_payload(),
        },
        "cohort_id": value.cohort_id,
        "cohort_sha256": value.cohort_sha256,
        "campaign_guided_generation_wall_seconds": float(
            value.campaign_guided_generation_wall_seconds
        ),
        "candidates": candidates,
    }


def _terminal_payload(result: MacroLoopResult) -> Mapping[str, object]:
    terminal = result.terminal
    evaluation_valid = (
        terminal.cold_start_score.eligible
        and terminal.selected_score.eligible
    )
    return {
        "cohort_id": terminal.cohort_id,
        "cohort_sha256": terminal.cohort_sha256,
        "case_count": len(terminal.selected.scores),
        "member_case_count": len(terminal.selected.scores),
        "root_count": len(set(terminal.selected.root_ids)),
        "objective_sample_count": len(terminal.selected_score.errors),
        "active_objective": (
            "generated"
            if result.paired_objective_policy.mode.value
            == "primary_with_perturbed_guardrail"
            else (
                "weighted_composite"
                if result.paired_perturbations
                else "macro_error"
            )
        ),
        "evaluation_status": "valid" if evaluation_valid else "invalid",
        "cold_start": _terminal_candidate_payload(
            result.cold_start,
            terminal.cold_start,
            terminal.cold_start_score,
        ),
        "selected": _terminal_candidate_payload(
            result.selected,
            terminal.selected,
            terminal.selected_score,
        ),
        "distribution_aggregates": {
            "cold_start": _paired_aggregates_payload(
                terminal.cold_start_objective_aggregates
            ),
            "selected": _paired_aggregates_payload(
                terminal.selected_objective_aggregates
            ),
        },
        "perturbed_guardrail": _guardrail_payload(
            terminal.perturbed_guardrail
        ),
        "guidance_evaluation": _guidance_evaluation_payload(
            getattr(terminal, "guidance_evaluation", None)
        ),
        "supports_selected_over_cold_start": (
            terminal.supports_selected_over_cold_start
        ),
        "comparison": (
            {
                "mean_improvement": terminal.comparison.mean_improvement,
                "confidence_lower": terminal.comparison.confidence_lower,
                "confidence_upper": terminal.comparison.confidence_upper,
                "win_rate": terminal.comparison.win_rate,
                "passed": terminal.comparison.passed,
                "failures": _public_comparison_failures(
                    terminal.comparison.failures
                ),
            }
            if terminal.comparison is not None
            else None
        ),
    }


def _terminal_candidate_payload(candidate, batch, score) -> Mapping[str, object]:
    statuses = [row.status for row in batch.scores]
    failure_rows = [list(row.failures) for row in batch.scores]
    failure_counts: dict[str, int] = {}
    for failures in failure_rows:
        for code in failures:
            failure_counts[code] = failure_counts.get(code, 0) + 1

    root_members: dict[str, list[str]] = {}
    for root_id, status in zip(batch.root_ids, statuses):
        root_members.setdefault(root_id, []).append(status)
    root_statuses = []
    for members in root_members.values():
        if "invalid" in members:
            root_statuses.append("invalid")
        elif "rejected" in members:
            root_statuses.append("rejected")
        else:
            root_statuses.append("valid")

    return {
        "candidate": _candidate_payload(candidate),
        "eligible": score.eligible,
        "errors": list(score.errors),
        "mean_error": fmean(score.errors),
        "statuses": statuses,
        "failures": failure_rows,
        "failure_counts": {
            code: failure_counts[code] for code in sorted(failure_counts)
        },
        "valid_member_count": statuses.count("valid"),
        "rejected_member_count": statuses.count("rejected"),
        "invalid_member_count": statuses.count("invalid"),
        "root_count": len(root_statuses),
        "valid_root_count": root_statuses.count("valid"),
        "rejected_root_count": root_statuses.count("rejected"),
        "invalid_root_count": root_statuses.count("invalid"),
    }


def _candidate_payload(candidate: CandidateRef | None) -> Mapping[str, object] | None:
    if candidate is None:
        return None
    return {
        "candidate_id": candidate.candidate_id,
        "bundle_sha256": candidate.bundle_sha256,
        "round_index": candidate.round_index,
        "agent_id": candidate.agent_id,
        "parent_sha256": candidate.parent_sha256,
    }


def _public_comparison_failures(values) -> list[str]:
    public = []
    for value in values:
        if value.startswith("candidate_gate:"):
            value = "candidate_gate_failure"
        elif value.startswith("incumbent_gate:"):
            value = "incumbent_gate_failure"
        if value not in public:
            public.append(value)
    return public


def _paired_aggregates_payload(value) -> Mapping[str, object] | None:
    if value is None:
        return None
    return {
        "generated": _aggregate_payload(value.generated),
        "perturbed": _aggregate_payload(value.perturbed),
        "composite": _aggregate_payload(value.composite),
    }


def _guardrail_payload(value) -> Mapping[str, object] | None:
    if value is None:
        return None
    return {
        "root_count": value.root_count,
        "candidate_mean": float(value.candidate_mean),
        "incumbent_mean": float(value.incumbent_mean),
        "mean_excess": float(value.mean_excess),
        "standard_error": float(value.standard_error),
        "confidence_lower": float(value.confidence_lower),
        "confidence_z": float(value.confidence_z),
        "practical_relative_margin": float(
            value.practical_relative_margin
        ),
        "passed": value.passed,
        "failures": list(value.failures),
    }


def _aggregate_payload(value) -> Mapping[str, float] | None:
    if value is None:
        return None
    return {
        "mean": float(value.mean),
        "cvar": float(value.cvar),
        "max": float(value.max),
    }


def _relative_improvement(cold_error: float, selected_error: float) -> float | None:
    if cold_error > 0.0:
        return (cold_error - selected_error) / cold_error
    return 0.0 if selected_error == cold_error else None


def _atomic_write(path: Path, payload: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _agent_environment(environment: Mapping[str, str]) -> Mapping[str, str]:
    return {
        name: environment[name]
        for name in _EXPLICIT_AGENT_ENVIRONMENT
        if name in environment
    }


__all__ = [
    "REFERENCE_ITERATION_PROTOCOL",
    "REFERENCE_ITERATION_RESULT_SCHEMA",
    "REFERENCE_PROCESS_WORKER_REPORT_SCHEMA",
    "REFERENCE_PROCESS_WORKER_REQUEST_SCHEMA",
    "REFERENCE_PRESELECTION_RECOVERY_PHASE",
    "REFERENCE_PRESELECTION_RECOVERY_SCHEMA",
    "ReferenceIterationConfig",
    "ReferenceIterationError",
    "ReferenceMidRoundRecovery",
    "ReferenceProcessRoundDeveloper",
    "ReferenceProcessWorkerConfig",
    "load_reference_process_worker_request",
    "run_reference_agent_iteration",
    "run_reference_process_agent_worker",
    "write_reference_process_worker_report",
]
