"""Bounded execution of split Candidate evaluations.

The Controller owns evaluation plans, truth, scoring, and aggregation.  This
module may receive only the candidate-facing ``Case`` and its public
``EvaluationPlan``.  The subprocess adapter executes one fresh process per
Candidate/Case pair, returns a finite prediction through the safe JSON/NPY
wire codec, and lets the Controller score that prediction afterwards.

This is a process-failure and resource-containment boundary, not a hostile
filesystem sandbox.  Candidate processes run as the current OS user and the
assurance record deliberately says so.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib import import_module
import json
import logging
import math
import os
from pathlib import Path
from queue import Empty, Full, Queue
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
from threading import BoundedSemaphore, Condition, Event, Lock, Thread, enumerate as enumerate_threads
import time
import traceback
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Artifact,
    CandidateEvaluationOutput,
    Case,
    CaseSource,
    CaseVisibility,
    EvaluationPhase,
    EvaluationPlan,
    EvaluationProgram,
    QueryKind,
)
from pde_agent_verifier_solver.sdk.public_sessions import (
    bind_public_session_factory,
)
from pde_agent_verifier_solver.sdk.solve_budget import (
    ORACLE_WAIT_TIMING_SCHEMA,
    _capture_oracle_wait,
    _current_candidate_solve_budget_meter,
    _record_oracle_response_wait,
    bind_candidate_solve_budget,
)

from pde_agent_verifier_solver.sdk._candidate_timing import (
    bind_queue_wait_observer,
)

from .agent_wire import AgentWireCodec
from .python_candidate import PythonCandidateProgram
from .candidate_diagnostics import (
    candidate_diagnostic, candidate_exception_diagnostic, normalize_candidate_diagnostic,
)
from .fair_execution import FairExecutor, FairAdmission, bind_oracle_policy
from ..sdk.oracle_profiling import capture_oracle_profile
from .gpu_execution import (
    GpuDevicePool, initialize_cuda_device, synchronize_cuda_device,
    validate_cuda_metadata, validated_gpu_devices, validated_memory_fraction,
)


CANDIDATE_WORKER_REQUEST_SCHEMA = "canonical-candidate-worker-request-v1"
CANDIDATE_WORKER_RESPONSE_SCHEMA = "canonical-candidate-worker-response-v1"
CANDIDATE_ORACLE_REQUEST_SCHEMA = "canonical-candidate-oracle-request-v1"
CANDIDATE_ORACLE_RESPONSE_SCHEMA = "canonical-candidate-oracle-response-v2"
CANDIDATE_EXECUTION_AUDIT_SCHEMA = "canonical-candidate-execution-audit-v1"
_ORACLE_PATH = "/v1/candidate-public-oracle"
_TOKEN_HEADER = "X-PDE-Candidate-Token"
_MAX_RPC_BYTES = 64 * 1024 * 1024
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
# Routine scientific point arrays should stay in the local HTTP message. Keep
# large arrays bounded in sidecars; the generic Agent codec retains its default.
_ORACLE_INLINE_ARRAY_BYTES = 1024 * 1024


PublicSessionInvoker = Callable[
    [str, str, tuple[object, ...], Mapping[str, object]],
    object,
]
PublicSessionCloser = Callable[[str], None]


class CandidateExecutionError(RuntimeError):
    """A Candidate execution adapter could not produce a valid output."""


class _OracleRequestFailure(CandidateExecutionError):
    def __init__(self, queue_wait_seconds: float, lock_wait_seconds: float) -> None:
        super().__init__("Candidate public-oracle call failed")
        self.queue_wait_seconds = queue_wait_seconds
        self.lock_wait_seconds = lock_wait_seconds


class CandidateExecutionTimeout(CandidateExecutionError):
    """A Candidate/Case subprocess exceeded its user-selected hard wall."""


class CandidateExecutionBatchTimeout(CandidateExecutionTimeout):
    """A Development Candidate batch exceeded its Controller RPC-safe wall."""


class CandidateExecutionCancelled(CandidateExecutionError):
    """The owning Agent turn revoked a pending Candidate evaluation."""


class _CandidatePeerCancelled(CandidateExecutionError):
    """A Case stopped because another Case already failed the same batch."""


class CandidateExecutionInfrastructureError(CandidateExecutionError):
    """The worker boundary or its cleanup could not be established."""


class CandidateExecutionService(Protocol):
    """Execute Candidate code without receiving Controller-owned truth."""

    candidate_invocation_process: str

    def execute_cases(
        self,
        artifact: Artifact,
        cases: Sequence[Case],
        plans: Sequence[EvaluationPlan],
        evaluation: EvaluationProgram,
        *,
        public_session_methods: Sequence[str] = (),
        invoke_public_session: PublicSessionInvoker | None = None,
        close_public_session: PublicSessionCloser | None = None,
        cancellation_event: Event | None = None,
        development_batch_deadline: float | None = None,
    ) -> tuple[CandidateEvaluationOutput, ...]: ...


@dataclass(frozen=True)
class PythonTaskWorkerSpec:
    """Trusted import recipe for reconstructing a Task in a clean worker."""

    module: str
    qualname: str
    init_kwargs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.module, str) or not self.module.strip():
            raise ValueError("Task worker module must be nonempty")
        if not isinstance(self.qualname, str) or not self.qualname.strip():
            raise ValueError("Task worker qualname must be nonempty")
        parts = self.qualname.split(".")
        if any(not part.isidentifier() for part in parts) or "<locals>" in parts:
            raise ValueError("Task worker qualname must be importable")
        if not isinstance(self.init_kwargs, Mapping) or any(
            not isinstance(name, str) for name in self.init_kwargs
        ):
            raise TypeError("Task worker init_kwargs must be a string-keyed mapping")
        object.__setattr__(
            self,
            "init_kwargs",
            MappingProxyType(dict(self.init_kwargs)),
        )

    @classmethod
    def from_task_type(
        cls,
        task_type: type,
        *,
        init_kwargs: Mapping[str, object] | None = None,
    ) -> "PythonTaskWorkerSpec":
        if not isinstance(task_type, type):
            raise TypeError("task_type must be a class")
        return cls(
            task_type.__module__,
            task_type.__qualname__,
            {} if init_kwargs is None else init_kwargs,
        )


@dataclass(frozen=True)
class CandidateExecutionAudit:
    schema: str
    execution_id: str
    artifact_id: str
    case_id: str
    phase: str
    case_index: int
    status: str
    pid: int | None
    return_code: int | None
    wall_seconds: float
    timeout_seconds: float
    process_group_cleaned: bool
    workspace: str
    stdout_path: str
    stderr_path: str
    response_path: str
    error_code: str | None = None
    candidate_process_isolated: bool = True
    hostile_filesystem_sandbox: bool = False
    controller_truth_shared: bool = False
    oracle_queue_wait_seconds: float = 0.0
    charged_wall_seconds: float = 0.0
    oracle_call_count: int = 0
    oracle_service_wall_seconds: float = 0.0
    oracle_codec_wall_seconds: float = 0.0
    budget_timing: str = "worker_wall_excluding_controller_oracle_queue"
    solver_batch_size: int = 1
    solver_batch_id: str | None = None
    oracle_case_queue_wait_seconds: float | None = None
    worker_wall_seconds: float = 0.0
    controller_postprocess_wall_seconds: float = 0.0
    controller_setup_wall_seconds: float = 0.0
    cleanup_wall_seconds: float = 0.0
    oracle_last_method: str | None = None
    oracle_inflight_method: str | None = None
    oracle_inflight_wall_seconds: float = 0.0
    execution_started: bool = False
    batch_failure_code: str | None = None
    gpu_device: str | None = None
    gpu_queue_wait_seconds: float = 0.0
    gpu_metadata: dict[str, object] = field(default_factory=dict)
    requester_agent_id: str | None = None
    solver_queue_wait_seconds: float = 0.0


class InProcessCandidateExecutionService:
    """Compatibility adapter for trusted local split-evaluation programs."""

    candidate_invocation_process = "controller"

    def execute_cases(
        self,
        artifact: Artifact,
        cases: Sequence[Case],
        plans: Sequence[EvaluationPlan],
        evaluation: EvaluationProgram,
        *,
        public_session_methods: Sequence[str] = (),
        invoke_public_session: PublicSessionInvoker | None = None,
        close_public_session: PublicSessionCloser | None = None,
        cancellation_event: Event | None = None,
        development_batch_deadline: float | None = None,
    ) -> tuple[CandidateEvaluationOutput, ...]:
        prepared_cases, prepared_plans = _validated_cases_and_plans(cases, plans)
        _validate_cancellation_event(cancellation_event)
        deadline = _validated_development_batch_deadline(
            development_batch_deadline
        )
        methods = _validated_methods(public_session_methods)
        if methods and invoke_public_session is None:
            raise ValueError("public Candidate methods need an invocation port")
        if methods and close_public_session is None:
            raise ValueError("public Candidate methods need a lifecycle closer")
        execute = getattr(evaluation, "execute_candidate", None)
        if not callable(execute):
            raise CandidateExecutionError(
                "EvaluationProgram does not support split Candidate execution"
            )
        outputs = []
        for case, plan in zip(prepared_cases, prepared_plans):
            _raise_if_candidate_execution_cancelled(
                Event(),
                cancellation_event,
                deadline if plan.phase is EvaluationPhase.DEVELOPMENT else None,
            )
            factory = (
                None
                if not methods
                else lambda bound_case, invoke=invoke_public_session, allowed=methods: (
                    _LocalPublicSession(bound_case, allowed, invoke)
                )
            )
            primary_error: BaseException | None = None
            output: object | None = None
            try:
                with bind_candidate_solve_budget(new_case=True), bind_public_session_factory(
                    factory,
                    case,
                    allowed_methods=methods,
                ):
                    output = execute(artifact, case, plan)
            except BaseException as error:
                primary_error = error
            finally:
                if close_public_session is not None:
                    try:
                        close_public_session(case.case_id)
                    except BaseException as cleanup_error:
                        if primary_error is None:
                            primary_error = cleanup_error
                        else:
                            _attach_execution_error(primary_error, cleanup_error)
            if primary_error is not None:
                raise primary_error
            _raise_if_candidate_execution_cancelled(
                Event(),
                cancellation_event,
                deadline if plan.phase is EvaluationPhase.DEVELOPMENT else None,
            )
            outputs.append(_require_candidate_output(output))
        return tuple(outputs)


class SubprocessCandidateExecutionService:
    """Run a Candidate/Case or optional solver batch in a bounded subprocess.

    The service deliberately parallelizes Cases, not sampling, plans, truth,
    scoring, or aggregation.  Consequently completion order cannot alter the
    scientific cohort or the returned result order.
    """

    candidate_invocation_process = "bounded_same_user_subprocess"

    def __init__(
        self,
        root: str | Path,
        *,
        task: PythonTaskWorkerSpec,
        evaluation_settings: Mapping[str, object],
        max_workers: int,
        case_hard_wall_seconds: float,
        development_batch_hard_wall_seconds: float | None = None,
        oracle_drain_timeout_seconds: float = 1.0,
        oracle_host_count: int = 1,
        oracle_max_workers: int = 1,
        oracle_inline_array_bytes: int = _ORACLE_INLINE_ARRAY_BYTES,
        oracle_trusted_results: bool = True,
        development_oracle_max_workers: int | None = None,
        python_executable: str | Path | None = None,
        environment: Mapping[str, str] | None = None,
        gpu_devices: Sequence[str] = (),
        gpu_device_pool: GpuDevicePool | None = None,
        gpu_memory_fraction: float | None = None,
        gpu_slots_per_device: int = 1,
        solver_batch_size: int = 1,
        oracle_concurrency_per_device: int = 1,
        development_gpu_devices: Mapping[str, Sequence[str]] | None = None,
        trace_calls: bool = False,
    ) -> None:
        if not isinstance(task, PythonTaskWorkerSpec):
            raise TypeError("task must be a PythonTaskWorkerSpec")
        for name, value in (("gpu_slots_per_device", gpu_slots_per_device),
                            ("solver_batch_size", solver_batch_size),
                            ("oracle_concurrency_per_device", oracle_concurrency_per_device)):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 128:
                raise ValueError(f"{name} must be an integer in [1, 128]")
        self.solver_batch_size = solver_batch_size
        self.oracle_concurrency_per_device = oracle_concurrency_per_device
        devices = validated_gpu_devices(gpu_devices)
        if not isinstance(trace_calls, bool):
            raise TypeError("trace_calls must be a boolean")
        routes = dict(development_gpu_devices or {})
        if any(not isinstance(key, str) or not key for key in routes):
            raise ValueError("development GPU routes require nonempty Agent ids")
        routes = {key: validated_gpu_devices(value) for key, value in routes.items()}
        if routes and (not devices or any(not value for value in routes.values())):
            raise ValueError("Agent routes need GPU devices and a separate Official GPU")
        all_devices = validated_gpu_devices((*devices, *(d for value in routes.values() for d in value)))
        if routes and max_workers < len(routes):
            raise ValueError("candidate_case_workers must allow at least one worker per Agent")
        self.development_gpu_devices = MappingProxyType(routes)
        memory_fraction = validated_memory_fraction(gpu_memory_fraction)
        if gpu_device_pool is not None:
            if not isinstance(gpu_device_pool, GpuDevicePool):
                raise TypeError("gpu_device_pool must be a GpuDevicePool")
            if not all_devices or not set(all_devices).issubset(gpu_device_pool.devices):
                raise ValueError("GPU service must explicitly select devices from its shared pool")
        if not devices and memory_fraction is not None:
            raise ValueError("GPU memory fraction requires explicit GPU devices")
        if not isinstance(evaluation_settings, Mapping):
            raise TypeError("evaluation_settings must be a mapping")
        if isinstance(max_workers, bool) or not isinstance(max_workers, int):
            raise TypeError("max_workers must be an integer")
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.oracle_host_count = oracle_host_count
        if isinstance(oracle_max_workers, bool) or not isinstance(oracle_max_workers, int):
            raise TypeError("oracle_max_workers must be an integer")
        if not 1 <= oracle_max_workers <= min(max_workers * solver_batch_size, 128):
            raise ValueError("oracle_max_workers must be between 1 and min(max_workers * solver_batch_size, 128)")
        if development_oracle_max_workers is None:
            development_oracle_max_workers = oracle_max_workers
        if (isinstance(development_oracle_max_workers, bool)
                or not isinstance(development_oracle_max_workers, int)
                or not 1 <= development_oracle_max_workers <= oracle_max_workers):
            raise ValueError("development_oracle_max_workers must be between 1 and oracle_max_workers")
        self.development_oracle_max_workers = development_oracle_max_workers
        timeout = float(case_hard_wall_seconds)
        if (
            isinstance(case_hard_wall_seconds, bool)
            or not math.isfinite(timeout)
            or timeout <= 0.0
        ):
            raise ValueError("case_hard_wall_seconds must be positive and finite")
        development_timeout = (
            None
            if development_batch_hard_wall_seconds is None
            else float(development_batch_hard_wall_seconds)
        )
        if development_timeout is not None and (
            isinstance(development_batch_hard_wall_seconds, bool)
            or not math.isfinite(development_timeout)
            or development_timeout <= 0.0
        ):
            raise ValueError(
                "development_batch_hard_wall_seconds must be positive and finite"
            )
        oracle_drain_timeout = float(oracle_drain_timeout_seconds)
        if (
            isinstance(oracle_drain_timeout_seconds, bool)
            or not math.isfinite(oracle_drain_timeout)
            or oracle_drain_timeout < 0.0
        ):
            raise ValueError(
                "oracle_drain_timeout_seconds must be non-negative and finite"
            )
        _validate_oracle_transport(oracle_inline_array_bytes, oracle_trusted_results)
        executable = Path(python_executable or sys.executable).expanduser().absolute()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError("Candidate worker Python must be executable")
        supplied_environment = dict(environment or {})
        if any(
            not isinstance(name, str)
            or not isinstance(value, str)
            or "\x00" in name
            or "\x00" in value
            for name, value in supplied_environment.items()
        ):
            raise TypeError("Candidate worker environment must contain strings")
        reserved = {
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "PYTHONNOUSERSITE",
            "PYTHONDONTWRITEBYTECODE",
        }
        if reserved & set(supplied_environment):
            raise ValueError(
                "Candidate worker environment cannot override managed values"
            )

        allocated = os.environ.get("SLURM_JOB_GPUS") or os.environ.get("CUDA_VISIBLE_DEVICES")
        if all_devices and allocated is not None and not set(all_devices).issubset(allocated.split(",")):
            raise ValueError("Candidate GPUs must already be CUDA-visible or explicitly Slurm-allocated")

        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError("Candidate execution root must be a real directory")
        os.chmod(self.root, 0o700)
        self.task = task
        self.evaluation_settings = dict(evaluation_settings)
        self.max_workers = max_workers
        self.oracle_max_workers = oracle_max_workers
        self.case_hard_wall_seconds = timeout
        # Set by a resumable allocation owner. This is infrastructure cancellation,
        # separate from the scientific per-Case budget and never used by Agents.
        self.allocation_cancellation_event: Event | None = None
        self.development_batch_hard_wall_seconds = development_timeout
        self.oracle_drain_timeout_seconds = oracle_drain_timeout
        self.oracle_inline_array_bytes = oracle_inline_array_bytes
        self.oracle_trusted_results = oracle_trusted_results
        self.python_executable = executable
        self.environment = MappingProxyType(supplied_environment)
        self.gpu_devices = devices
        self.gpu_device_pool = gpu_device_pool or (GpuDevicePool(all_devices, slots_per_device=gpu_slots_per_device) if all_devices else None)
        self.gpu_memory_fraction = memory_fraction
        self._audit_lock = Lock()
        self._audits: list[CandidateExecutionAudit] = []
        self._worker_slots = BoundedSemaphore(max_workers)
        # One service-owned executor makes ``max_workers`` a genuine global
        # helper-thread/process bound even when two Agents evaluate
        # concurrently. Per-call executors would each create ``max_workers``
        # threads while only the subprocess semaphore was shared.
        self._oracle_admission = FairAdmission(oracle_max_workers)
        self._development_oracle_admission = FairAdmission(development_oracle_max_workers)
        actual_workers = (max_workers if self.gpu_device_pool is None else
                          min(max_workers, len(devices) * self.gpu_device_pool.slots_per_device))
        self._executor = FairExecutor(
            max_workers=actual_workers,
            thread_name_prefix="candidate-case",
        )
        self._development_executors = {
            key: FairExecutor(max_workers=min(max_workers // len(routes),
                len(value) * self.gpu_device_pool.slots_per_device),
                thread_name_prefix=f"candidate-{key}") for key, value in routes.items()}
        from .execution_trace import ExecutionTrace
        self._trace = ExecutionTrace(self.root / "execution_trace.jsonl") if trace_calls else None
        self._process_lock = Lock()
        self._processes: dict[int, tuple[str, subprocess.Popen[bytes]]] = {}
        self._health_lock = Lock()
        self._unhealthy_reason: str | None = None

    def for_development(self, agent_id):
        if not self.development_gpu_devices:
            return self
        if agent_id not in self.development_gpu_devices:
            raise ValueError(f"No development GPU assigned to Agent {agent_id}")
        from .candidate_routing import AgentCandidateExecution
        return AgentCandidateExecution(self, agent_id)

    def trace(self, event, **fields):
        if self._trace is not None:
            self._trace.record(event, **fields)

    def close(self):
        for executor in (self._executor, *self._development_executors.values()):
            executor.shutdown(wait=True, cancel_futures=True)
        if self._trace is not None:
            self._trace.close()

    def execution_settings_snapshot(self):
        return {"development_solver_placement": ("agent_gpu" if self.development_gpu_devices else "shared"),
                "development_gpu_devices": {k: list(v) for k, v in self.development_gpu_devices.items()},
                "development_workers_per_agent": {k: v.max_workers for k, v in self._development_executors.items()},
                "trace_calls": self._trace is not None,
                "solver_batch_size": self.solver_batch_size,
                "candidate_case_workers": self.max_workers,
                "gpu_devices": list(self.gpu_devices),
                "gpu_slots_per_device": (self.gpu_device_pool.slots_per_device if self.gpu_device_pool else 0),
                "oracle_max_workers": self.oracle_max_workers,
                "development_oracle_max_workers": self.development_oracle_max_workers,
                "oracle_concurrency_per_device": self.oracle_concurrency_per_device,
                "scheduling": "round_robin_evaluation_requests",
                "case_charged_wall_seconds": self.case_hard_wall_seconds,
                "batch_budget": "case_limit_times_actual_batch_size",
                "batch_wait_credit": "intersection_of_all_member_waits"}

    @property
    def audits(self) -> tuple[CandidateExecutionAudit, ...]:
        with self._audit_lock:
            return tuple(self._audits)

    @property
    def oracle_host_count(self) -> int:
        """Hidden-evaluation hosts; configure only at an idle phase boundary.

        Each call snapshots this setting. Development always uses one host.
        Multiple hosts require an independently concurrency-safe invocation
        port (for example, isolated GPU lanes); they do not remove its locks.
        """

        return self._oracle_host_count

    @oracle_host_count.setter
    def oracle_host_count(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("oracle_host_count must be an integer")
        if value < 1:
            raise ValueError("oracle_host_count must be positive")
        self._oracle_host_count = value

    def execute_cases(
        self,
        artifact: Artifact,
        cases: Sequence[Case],
        plans: Sequence[EvaluationPlan],
        evaluation: EvaluationProgram,
        *,
        public_session_methods: Sequence[str] = (),
        invoke_public_session: PublicSessionInvoker | None = None,
        close_public_session: PublicSessionCloser | None = None,
        cancellation_event: Event | None = None,
        development_batch_deadline: float | None = None,
        _development_agent_id: str | None = None,
    ) -> tuple[CandidateEvaluationOutput, ...]:
        self._raise_if_healthy()
        prepared_cases, prepared_plans = _validated_cases_and_plans(cases, plans)
        executor = self._executor
        route_devices = self.gpu_devices
        if self.development_gpu_devices:
            development = all(p.phase is EvaluationPhase.DEVELOPMENT for p in prepared_plans)
            if _development_agent_id is None:
                if any(p.phase is EvaluationPhase.DEVELOPMENT for p in prepared_plans):
                    raise ValueError("Development evaluation requires a Controller-bound Agent route")
            else:
                if not development or _development_agent_id not in self.development_gpu_devices:
                    raise ValueError("Agent route is valid only for its Development evaluations")
                executor = self._development_executors[_development_agent_id]
                route_devices = self.development_gpu_devices[_development_agent_id]
        elif _development_agent_id is not None:
            raise ValueError("Development GPU routing is not configured")
        _validate_cancellation_event(cancellation_event)
        caller_batch_deadline = _validated_development_batch_deadline(
            development_batch_deadline
        )
        if _caller_cancelled(cancellation_event):
            raise CandidateExecutionCancelled(
                "Candidate evaluation was cancelled by its Agent turn"
            )
        if not isinstance(artifact.program, PythonCandidateProgram):
            raise CandidateExecutionError(
                "subprocess Candidate execution requires a frozen Python bundle"
            )
        if getattr(evaluation, "supports_split_candidate_execution", False) is not True:
            raise CandidateExecutionError(
                "EvaluationProgram does not support split Candidate execution"
            )
        methods = _validated_methods(public_session_methods)
        if methods and invoke_public_session is None:
            raise ValueError("public Candidate methods need an invocation port")
        if methods and close_public_session is None:
            raise ValueError("public Candidate methods need a lifecycle closer")
        expected_program_id = getattr(evaluation, "program_id", None)
        if not isinstance(expected_program_id, str) or not expected_program_id:
            raise CandidateExecutionError("EvaluationProgram has no stable program_id")

        development_plans = tuple(
            plan.phase is EvaluationPhase.DEVELOPMENT for plan in prepared_plans
        )
        if any(development_plans) and not all(development_plans):
            raise ValueError(
                "Candidate execution cannot mix Development and hidden phases"
            )
        allocation_event = None if all(development_plans) else self.allocation_cancellation_event
        if allocation_event is not None:
            _validate_cancellation_event(allocation_event)
            if cancellation_event is not None:
                raise ValueError("hidden allocation cancellation cannot replace a caller cancellation event")
            cancellation_event = allocation_event
        use_batch = (self.solver_batch_size > 1
                     and getattr(evaluation, "supports_batch_candidate_execution", False)
                     and artifact.program.supports_batch_solution)
        # A common host owns the exact shared-wait ledger for each solver batch.
        oracle_host_count = 1 if all(development_plans) or use_batch else self.oracle_host_count
        batch_deadline = None
        if all(development_plans):
            if self.development_batch_hard_wall_seconds is not None:
                batch_deadline = (
                    time.monotonic()
                    + self.development_batch_hard_wall_seconds
                )
            if caller_batch_deadline is not None:
                batch_deadline = (
                    caller_batch_deadline
                    if batch_deadline is None
                    else min(batch_deadline, caller_batch_deadline)
                )

        execution_id = uuid4().hex
        retained_call_root = self.root / f"evaluation-{execution_id}"
        call_root = _candidate_call_workspace(retained_call_root)
        cancel = Event()
        first_error: BaseException | None = None
        oracle_closed_cleanly = False
        results: list[CandidateEvaluationOutput | None] = [None] * len(prepared_cases)
        from contextlib import nullcontext
        development_oracle = all(plan.phase == EvaluationPhase.DEVELOPMENT for plan in prepared_plans)
        oracle_workers = self.development_oracle_max_workers if development_oracle else self.oracle_max_workers
        def shared_invoke(case_id, method, args, kwargs):
            admission = self._development_oracle_admission.enter(execution_id) if development_oracle else nullcontext()
            with admission, self._oracle_admission.enter(execution_id), bind_oracle_policy(
                    execution_id, self.oracle_concurrency_per_device):
                return invoke_public_session(case_id, method, args, kwargs)

        try:
            host_factory = (
                _CandidateOracleHost
                if oracle_host_count == 1
                else _CandidateOracleHostGroup
            )
            oracle = host_factory(
                call_root / "oracle-arrays",
                case_ids=tuple(case.case_id for case in prepared_cases),
                methods=methods,
                invoke=(shared_invoke if invoke_public_session is not None else None),
                close=close_public_session,
                on_drain_timeout=self._mark_unhealthy,
                oracle_inline_array_bytes=self.oracle_inline_array_bytes,
                oracle_trusted_results=self.oracle_trusted_results,
                **({} if oracle_host_count == 1 else {"host_count": oracle_host_count}),
                max_workers=min(oracle_workers, len(prepared_cases)),
                **({"trace_call": lambda **fields: self.trace(fields.pop("event", "oracle_call"),
                    execution_id=execution_id, requester_agent_id=_development_agent_id, **fields)}
                   if self._trace is not None else {}),
            )
        except BaseException as error:
            self._mark_unhealthy(
                "Candidate oracle initialization could not be confirmed"
            )
            raise CandidateExecutionInfrastructureError(
                "Candidate public oracle could not be initialized"
            ) from error
        try:
            oracle.start()
            futures: dict[Future[CandidateEvaluationOutput], int] = {}
            next_index = 0
            chunks = {}
            submission_times = {}

            def save_result(index, value):
                indices = chunks[index]
                if not use_batch:
                    results[index] = value
                else:
                    if len(value) != len(indices):
                        raise CandidateExecutionInfrastructureError("worker batch output count differs")
                    for member_index, output in zip(indices, value):
                        results[member_index] = output

            def record_cancelled_chunk(index, error):
                for member_index in chunks[index]:
                    self._record_unstarted_audit(
                        execution_id=execution_id, index=member_index, artifact=artifact,
                        case=prepared_cases[member_index], plan=prepared_plans[member_index],
                        call_root=call_root, error=error,
                        requester_agent_id=_development_agent_id, submitted_at=submission_times.get(index))

            def fill_submission_window() -> None:
                nonlocal next_index
                while (
                    first_error is None
                    and len(futures) < self.max_workers
                    and next_index < len(prepared_cases)
                ):
                    index = next_index
                    indices = tuple(range(index, min(len(prepared_cases), index +
                                                    (self.solver_batch_size if use_batch else 1))))
                    chunks[index] = indices
                    members = tuple((i, prepared_cases[i], prepared_plans[i]) for i in indices) if use_batch else ()
                    submitted_at = time.monotonic()
                    submission_times[index] = submitted_at
                    self.trace("solver_submitted", execution_id=execution_id,
                        requester_agent_id=_development_agent_id, case_index=index,
                        case_ids=[prepared_cases[i].case_id for i in indices], gpu_devices=list(route_devices))
                    future = executor.submit_for(
                        execution_id, self._execute_one,
                        execution_id,
                        index,
                        artifact,
                        prepared_cases[index],
                        prepared_plans[index],
                        expected_program_id,
                        (oracle._host_for_case(prepared_cases[index].case_id)
                         if isinstance(oracle, _CandidateOracleHostGroup) else oracle),
                        call_root,
                        cancel,
                        cancellation_event,
                        batch_deadline,
                        members,
                        route_devices=route_devices,
                        requester_agent_id=_development_agent_id,
                        submitted_at=submitted_at,
                    )
                    futures[future] = index
                    next_index += len(indices)

            try:
                fill_submission_window()
                while futures:
                    _raise_if_candidate_execution_cancelled(
                        cancel,
                        cancellation_event,
                        batch_deadline,
                    )
                    completed, _pending = wait(
                        tuple(futures),
                        timeout=0.1,
                        return_when=FIRST_COMPLETED,
                    )
                    if not completed:
                        continue
                    for future in completed:
                        index = futures.pop(future)
                        try:
                            save_result(index, future.result())
                        except BaseException as error:
                            if first_error is None:
                                first_error = error
                                cancel.set()
                                for cleanup_error in self._terminate_active_processes(
                                    execution_id
                                ):
                                    _attach_execution_error(error, cleanup_error)
                            for pending in futures:
                                pending.cancel()
                    fill_submission_window()
            except BaseException as error:
                if first_error is None:
                    first_error = error
            finally:
                if first_error is not None:
                    cancel.set()
                    for pending in futures:
                        pending.cancel()
                    for cleanup_error in self._terminate_active_processes(
                        execution_id
                    ):
                        _attach_execution_error(first_error, cleanup_error)
                    # Do not close the per-batch oracle or delete its workspace
                    # while a shared-executor task can still enter it. Queued
                    # tasks cancel immediately; running tasks poll ``cancel``
                    # and settle after their owned process group is terminated.
                    while futures:
                        completed = {
                            future for future in futures if future.done()
                        }
                        if not completed:
                            time.sleep(0.05)
                            continue
                        for future in completed:
                            index = futures.pop(future)
                            if future.cancelled():
                                try:
                                    record_cancelled_chunk(index, first_error)
                                except BaseException as audit_error:
                                    self._mark_unhealthy(
                                        "Candidate execution audit persistence "
                                        "could not be confirmed"
                                    )
                                    _attach_execution_error(
                                        first_error,
                                        audit_error,
                                    )
                                continue
                            try:
                                # Preserve a successful peer even if another Case
                                # or the allocation cancellation ended the batch.
                                save_result(index, future.result())
                            except BaseException as peer_error:
                                if peer_error is not first_error:
                                    _attach_execution_error(
                                        first_error,
                                        peer_error,
                                    )
                    while next_index < len(prepared_cases):
                        index = next_index
                        next_index += 1
                        try:
                            self._record_unstarted_audit(
                                execution_id=execution_id,
                                index=index,
                                artifact=artifact,
                                case=prepared_cases[index],
                                plan=prepared_plans[index],
                                call_root=call_root,
                                error=first_error,
                                requester_agent_id=_development_agent_id,
                            )
                        except BaseException as audit_error:
                            self._mark_unhealthy(
                                "Candidate execution audit persistence could "
                                "not be confirmed"
                            )
                            _attach_execution_error(first_error, audit_error)
        except BaseException as error:
            if first_error is None:
                first_error = error
        finally:
            try:
                oracle.close(
                    # Failed candidates still need the configured, bounded
                    # shutdown grace. A zero-grace join can quarantine an idle
                    # service merely because its threads have not been scheduled.
                    drain_timeout_seconds=self.oracle_drain_timeout_seconds
                )
                oracle_closed_cleanly = True
            except BaseException as cleanup_error:
                if first_error is None:
                    first_error = cleanup_error
                else:
                    _attach_execution_error(first_error, cleanup_error)

        if first_error is not None:
            # Completed cases have already been removed; retain the shared
            # transport inputs needed to interpret failed workers' requests.
            failure_archived = False
            try:
                oracle_arrays = call_root / "oracle-arrays"
                if oracle_arrays.exists():
                    _retain_candidate_workspace(
                        oracle_arrays, retained_call_root / "oracle-arrays"
                    )
                if call_root != retained_call_root:
                    _write_private_json(
                        retained_call_root / "workspace_locations.json",
                        {"runtime_workspace": str(call_root),
                         "retained_workspace": str(retained_call_root)},
                    )
                failure_archived = True
            except BaseException as retention_error:
                self._mark_unhealthy("Candidate failure evidence could not be retained")
                _attach_execution_error(first_error, retention_error)
            with self._health_lock:
                cleanup_confirmed = self._unhealthy_reason is None
            with self._audit_lock:
                audited_indices = {audit.case_index for audit in self._audits
                                   if audit.execution_id == execution_id}
            # Ordinary rejected solvers are common during development. Once
            # diagnostics are durable, avoid retaining a second complete copy
            # on node-local scratch. Keep it if any process, Oracle, audit or
            # archival cleanup could not be confirmed.
            if (call_root != retained_call_root and failure_archived
                    and oracle_closed_cleanly and cleanup_confirmed
                    and audited_indices == set(range(len(prepared_cases)))):
                try:
                    shutil.rmtree(call_root)
                except BaseException as cleanup_error:
                    self._mark_unhealthy("Candidate failure scratch cleanup could not be confirmed")
                    _attach_execution_error(first_error, cleanup_error)
            # Scratch cleanup can itself quarantine the service. Re-read after
            # that operation before classifying the exception as resumable.
            with self._health_lock:
                cleanup_confirmed = self._unhealthy_reason is None
            first_error._candidate_partial_outputs = {
                index: output for index, output in enumerate(results) if output is not None
            }
            if (allocation_event is not None and allocation_event.is_set()
                    and isinstance(first_error, CandidateExecutionCancelled)):
                if not (oracle_closed_cleanly and cleanup_confirmed and failure_archived):
                    raise CandidateExecutionInfrastructureError(
                        "allocation pause could not confirm Candidate/Oracle cleanup"
                    ) from first_error
                first_error._allocation_pause_requested = True
            raise first_error
        if any(output is None for output in results):
            raise CandidateExecutionInfrastructureError(
                "Candidate execution returned an incomplete ordered result"
            )
        try:
            shutil.rmtree(call_root)
        except BaseException as error:
            self._mark_unhealthy(
                "completed Candidate batch workspace cleanup could not be confirmed"
            )
            raise CandidateExecutionInfrastructureError(
                "completed Candidate batch workspace could not be removed"
            ) from error
        return tuple(output for output in results if output is not None)

    def _execute_one(
        self,
        execution_id: str,
        index: int,
        artifact: Artifact,
        case: Case,
        plan: EvaluationPlan,
        expected_program_id: str,
        oracle: _CandidateOracleHost,
        call_root: Path,
        cancel: Event,
        cancellation_event: Event | None,
        batch_deadline: float | None,
        batch_members: tuple = (),
        *, route_devices: tuple[str, ...] | None = None,
        requester_agent_id: str | None = None,
        submitted_at: float | None = None,
    ) -> CandidateEvaluationOutput:
        dequeued_at = time.monotonic()
        solver_queue_wait_seconds = max(0., dequeued_at-submitted_at) if submitted_at is not None else 0.
        route_devices = self.gpu_devices if route_devices is None else route_devices
        underlying_oracle = oracle
        timeout_seconds = self.case_hard_wall_seconds * (len(batch_members) or 1)
        if batch_members:
            from .candidate_batch import BatchOracleView
            oracle = BatchOracleView(oracle, tuple(row[1].case_id for row in batch_members))
        task_root = call_root / f"case-{index:06d}-{uuid4().hex}"
        task_root.mkdir(mode=0o700)
        codec = AgentWireCodec(task_root / "arrays", durable_sidecars=False)
        retained_task_root = self.root / f"evaluation-{execution_id}" / task_root.name
        request_path = task_root / "request.json"
        response_path = task_root / "response.json"
        stdout_path = task_root / "stdout.log"
        stderr_path = task_root / "stderr.log"
        command = (
            str(self.python_executable),
            "-m",
            "pde_agent_verifier_solver.engine.candidate_worker",
            "--request",
            str(request_path),
            "--response",
            str(response_path),
        )
        started = time.monotonic()
        process: subprocess.Popen[bytes] | None = None
        return_code: int | None = None
        cleaned = True
        slot_acquired = False
        error_code: str | None = None
        status = "failed"
        output: CandidateEvaluationOutput | None = None
        primary_error: BaseException | None = None
        worker_started: float | None = None
        charged_wall_seconds = 0.0
        worker_finished: float | None = None
        oracle_queue_wait_seconds = 0.0
        oracle_metrics: dict[str, object] = {}
        termination_wall_seconds = 0.0
        gpu_device: str | None = None
        gpu_queue_wait_seconds = 0.0
        gpu_metadata: dict[str, object] = {}
        try:
            _raise_if_candidate_execution_cancelled(
                cancel,
                cancellation_event,
                batch_deadline,
            )
            self._raise_if_healthy()
            request_payload = _worker_request_payload(
                execution_id=execution_id,
                case_index=index,
                task=self.task,
                evaluation_settings=self.evaluation_settings,
                expected_program_id=expected_program_id,
                artifact=artifact,
                case=case,
                plan=plan,
                oracle_manifest=oracle.manifest_for_case(case.case_id),
                codec=codec,
            )
            if batch_members:
                request_payload = dict(request_payload)
                request_payload["batch_members"] = [
                    {"index": i, "case": _case_to_payload(member, codec),
                     "plan": _plan_to_payload(member_plan, codec),
                     "public_oracle": dict(underlying_oracle.manifest_for_case(member.case_id))}
                    for i, member, member_plan in batch_members]
            while not slot_acquired:
                _raise_if_candidate_execution_cancelled(
                    cancel,
                    cancellation_event,
                    batch_deadline,
                )
                self._raise_if_healthy()
                slot_acquired = self._worker_slots.acquire(timeout=0.1)
            _raise_if_candidate_execution_cancelled(
                cancel,
                cancellation_event,
                batch_deadline,
            )
            self._raise_if_healthy()
            request_payload = dict(request_payload)
            if self.gpu_device_pool is not None:
                gpu_wait_started = time.monotonic()
                try:
                    while gpu_device is None:
                        _raise_if_candidate_execution_cancelled(cancel, cancellation_event, batch_deadline)
                        self._raise_if_healthy()
                        try:
                            gpu_device = self.gpu_device_pool.acquire(timeout=0.1, devices=route_devices)
                        except RuntimeError as error:
                            raise CandidateExecutionInfrastructureError(str(error)) from error
                finally:
                    gpu_queue_wait_seconds = time.monotonic() - gpu_wait_started
                _raise_if_candidate_execution_cancelled(cancel, cancellation_event, batch_deadline)
                request_payload["device_policy"] = {
                    "schema": "candidate-device-policy-v1", "device": "cuda",
                    "assigned_gpu": gpu_device, "memory_fraction": self.gpu_memory_fraction,
                }
            request_payload["execution_budget"] = {
                "case_charged_wall_seconds": timeout_seconds,
            }
            _write_private_json(request_path, request_payload, durable=False)
            environment = dict(self._worker_environment(task_root))
            if gpu_device is not None:
                environment["CUDA_VISIBLE_DEVICES"] = gpu_device
            with stdout_path.open("xb") as stdout_handle, stderr_path.open(
                "xb"
            ) as stderr_handle:
                # Share the exact launch clock via the trusted environment.
                # Encoding/writing the request and waiting for a worker slot
                # are Controller setup; interpreter imports consume the limit.
                worker_started = time.monotonic()
                environment["PDE_CANDIDATE_STARTED_AT_MONOTONIC"] = str(worker_started)
                process = subprocess.Popen(
                    command,
                    cwd=task_root,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    start_new_session=True,
                )
            self.trace("solver_started", execution_id=execution_id,
                requester_agent_id=requester_agent_id, case_index=index, pid=process.pid,
                gpu_device=gpu_device, solver_queue_wait_seconds=solver_queue_wait_seconds,
                gpu_queue_wait_seconds=gpu_queue_wait_seconds,
                case_ids=[m[1].case_id for m in batch_members] if batch_members else [case.case_id])
            with self._process_lock:
                self._processes[process.pid] = (execution_id, process)
            while True:
                return_code = process.poll()
                if return_code is not None:
                    worker_finished = time.monotonic()
                    oracle_queue_wait_seconds = oracle.state.queue_wait_seconds(case.case_id)
                    oracle_metrics = oracle.state.execution_metrics(case.case_id)
                    break
                _raise_if_candidate_execution_cancelled(
                    cancel,
                    cancellation_event,
                    batch_deadline,
                )
                self._raise_if_healthy()
                charged_wall_seconds = max(0.0, time.monotonic() - worker_started
                    - oracle.state.queue_wait_seconds(case.case_id))
                remaining = timeout_seconds - charged_wall_seconds
                if remaining <= 0.0:
                    error_code = "candidate_case_timeout"
                    worker_finished = time.monotonic()
                    oracle_queue_wait_seconds = oracle.state.queue_wait_seconds(case.case_id)
                    oracle_metrics = oracle.state.execution_metrics(case.case_id)
                    termination_started = time.monotonic()
                    try:
                        _terminate_process_group(process)
                    finally:
                        termination_wall_seconds += time.monotonic() - termination_started
                    return_code = process.poll()
                    raise CandidateExecutionTimeout(
                        "Candidate Case execution exceeded its hard wall"
                    )
                time.sleep(min(0.1, remaining))
            _raise_if_candidate_execution_cancelled(
                cancel,
                cancellation_event,
                batch_deadline,
            )
            self._raise_if_healthy()
            if return_code != 0:
                diagnostic = {}
                if gpu_device is not None:
                    try:
                        failure_response = _read_bounded_json(response_path, _MAX_RESPONSE_BYTES)
                        gpu_metadata = validate_cuda_metadata(failure_response.get("device_execution"), gpu_device)
                    except (ValueError, CandidateExecutionInfrastructureError):
                        pass  # Failed readiness still fails; never manufacture a pass.
                error_code = _failed_worker_code(
                    response_path,
                    execution_id=execution_id,
                    case_index=index,
                    case_id=case.case_id,
                    diagnostic=diagnostic,
                )
                failure_type = {
                    "candidate_execution_infrastructure_failure": CandidateExecutionInfrastructureError,
                    "candidate_case_timeout": CandidateExecutionTimeout,
                    "candidate_development_batch_timeout": CandidateExecutionBatchTimeout,
                    "candidate_execution_cancelled": CandidateExecutionCancelled,
                }.get(error_code, CandidateExecutionError)
                failure = failure_type(
                    "Candidate Case worker failed; Controller-private diagnostics "
                    f"were retained in {retained_task_root}"
                )
                if diagnostic:
                    failure.candidate_diagnostic = diagnostic
                failure._candidate_case_index = index
                raise failure
            response = _read_bounded_json(response_path, _MAX_RESPONSE_BYTES)
            if gpu_device is not None:
                try:
                    gpu_metadata = validate_cuda_metadata(response.get("device_execution"), gpu_device)
                except ValueError as error:
                    raise CandidateExecutionInfrastructureError(str(error)) from error
            output = _worker_output_from_payload(
                response,
                expected_execution_id=execution_id,
                expected_case_index=index,
                expected_case_id=case.case_id,
                codec=codec,
            )
            if batch_members:
                payloads = response.get("batch_outputs")
                if not isinstance(payloads, list) or len(payloads) != len(batch_members):
                    raise CandidateExecutionInfrastructureError("worker batch outputs are missing")
                output = tuple(_worker_output_from_payload(
                    {**response, "case_index": row.get("case_index"), "case_id": row.get("case_id"),
                     "output": row.get("output")}, expected_execution_id=execution_id,
                    expected_case_index=i, expected_case_id=member.case_id, codec=codec)
                    for row, (i, member, _plan) in zip(payloads, batch_members))
            status = "completed"
        except BaseException as error:
            if error_code is None:
                error_code = _candidate_error_code(error)
            primary_error = error

        if worker_finished is None:
            worker_finished = time.monotonic()
            oracle_queue_wait_seconds = oracle.state.queue_wait_seconds(case.case_id)
            oracle_metrics = oracle.state.execution_metrics(case.case_id)
        worker_wall_seconds = (0.0 if worker_started is None or process is None else
                               max(0.0, worker_finished - worker_started))
        charged_wall_seconds = max(0.0, worker_wall_seconds - oracle_queue_wait_seconds)
        cleanup_started = time.monotonic()
        # Controller response decoding and cleanup are not Candidate work.
        controller_postprocess_wall_seconds = max(
            0.0, cleanup_started - worker_finished - termination_wall_seconds)

        cleanup_errors: list[BaseException] = []
        if process is not None:
            try:
                if process.poll() is None:
                    _terminate_process_group(process)
                return_code = process.poll()
            except BaseException as error:
                cleanup_errors.append(error)
            try:
                cleaned = _cleanup_process_group(process.pid)
            except BaseException as error:
                cleaned = False
                cleanup_errors.append(error)
            finally:
                with self._process_lock:
                    self._processes.pop(process.pid, None)
            if not cleaned:
                self._mark_unhealthy(
                    "Candidate process group cleanup could not be confirmed"
                )
                cleanup_errors.append(
                    CandidateExecutionInfrastructureError(
                        "Candidate process group cleanup could not be confirmed"
                    )
                )
        try:
            oracle.finish_case(
                case.case_id,
                drain_timeout_seconds=self.oracle_drain_timeout_seconds,
            )
        except BaseException as error:
            error_code = error_code or "candidate_public_session_cleanup_failed"
            cleanup_errors.append(error)
        if slot_acquired:
            self._worker_slots.release()
        if gpu_device is not None:
            # Never offer a GPU to the next solver until its old process group
            # is gone. Uncertain cleanup quarantines the shared slot.
            try:
                assert self.gpu_device_pool is not None
                self.gpu_device_pool.release(gpu_device, healthy=cleaned and not cleanup_errors)
            except BaseException as error:
                cleanup_errors.append(error)
                self._mark_unhealthy("Candidate GPU lease cleanup could not be confirmed")

        for cleanup_error in cleanup_errors:
            status = "failed"
            error_code = error_code or "candidate_process_cleanup_unconfirmed"
            if primary_error is None:
                primary_error = cleanup_error
            else:
                _attach_execution_error(primary_error, cleanup_error)

        if primary_error is None:
            try:
                oracle_failures = oracle.state.failure_diagnostics(case.case_id)
                if oracle_failures:
                    # A solver may handle a backend error and return a valid
                    # fallback. Preserve that backend cause even on success,
                    # without adding writes to error-free numerical calls.
                    _write_private_json(self.root / "oracle_failures" / f"{execution_id}-{index:06d}.json", {
                        "execution_id": execution_id, "artifact_id": artifact.artifact_id,
                        "case_id": case.case_id, "phase": plan.phase.value,
                        "oracle_failures": oracle_failures,
                    })
                shutil.rmtree(task_root)
            except BaseException as error:
                status = "failed"
                error_code = "candidate_workspace_cleanup_failed"
                self._mark_unhealthy(
                    "completed Candidate workspace cleanup could not be confirmed"
                )
                primary_error = CandidateExecutionInfrastructureError(
                    "completed Candidate workspace could not be removed"
                )
                primary_error.__cause__ = error

        if primary_error is not None:
            try:
                _retain_candidate_workspace(task_root, retained_task_root)
                task_root = retained_task_root
                stdout_path = task_root / "stdout.log"
                stderr_path = task_root / "stderr.log"
                response_path = task_root / "response.json"
                _write_private_json(task_root / "execution_failure.json", {
                    "execution_id": execution_id,
                    "artifact_id": artifact.artifact_id,
                    "case_id": case.case_id,
                    "phase": plan.phase.value,
                    "error_type": type(primary_error).__name__,
                    "message": _bounded_diagnostic(str(primary_error)),
                    "traceback": _bounded_diagnostic("".join(traceback.format_exception(
                        type(primary_error), primary_error, primary_error.__traceback__))),
                    "cleanup_errors": [
                        {"error_type": type(error).__name__, "message": _bounded_diagnostic(str(error))}
                        for error in getattr(primary_error, "_candidate_cleanup_errors", ())
                    ],
                    "oracle_failures": oracle.state.failure_diagnostics(case.case_id),
                    "oracle_progress_at_worker_stop": oracle_metrics,
                    "service_unhealthy_reason": self._unhealthy_reason,
                })
            except BaseException as retention_error:
                self._mark_unhealthy("Candidate failure evidence could not be retained")
                _attach_execution_error(primary_error, retention_error)

        audit = CandidateExecutionAudit(
            schema=CANDIDATE_EXECUTION_AUDIT_SCHEMA,
            execution_id=execution_id,
            artifact_id=artifact.artifact_id,
            case_id=case.case_id,
            phase=plan.phase.value,
            case_index=index,
            status=status,
            pid=None if process is None else process.pid,
            return_code=return_code,
            wall_seconds=max(0.0, time.monotonic() - started),
            timeout_seconds=timeout_seconds,
            oracle_queue_wait_seconds=oracle_queue_wait_seconds,
            charged_wall_seconds=charged_wall_seconds,
            **oracle_metrics,
            execution_started=process is not None,
            worker_wall_seconds=worker_wall_seconds,
            controller_postprocess_wall_seconds=controller_postprocess_wall_seconds,
            controller_setup_wall_seconds=max(0.0, (worker_started or worker_finished) - started),
            cleanup_wall_seconds=termination_wall_seconds + max(0.0, time.monotonic() - cleanup_started),
            process_group_cleaned=cleaned,
            workspace=str(task_root),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            response_path=str(response_path),
            error_code=error_code,
            gpu_device=gpu_device,
            gpu_queue_wait_seconds=gpu_queue_wait_seconds,
            gpu_metadata=gpu_metadata,
            requester_agent_id=requester_agent_id,
            solver_queue_wait_seconds=solver_queue_wait_seconds,
        )
        self.trace("solver_finished", execution_id=execution_id, requester_agent_id=requester_agent_id,
            case_index=index, pid=None if process is None else process.pid, gpu_device=gpu_device,
            status=status, worker_wall_seconds=worker_wall_seconds,
            solver_queue_wait_seconds=solver_queue_wait_seconds,
            oracle_queue_wait_seconds=oracle_queue_wait_seconds,
            gpu_metadata=gpu_metadata, error_code=error_code)
        try:
            if batch_members:
                for i, member, member_plan in batch_members:
                    self._record_audit(replace(
                        audit, case_index=i, case_id=member.case_id, phase=member_plan.phase.value,
                        solver_batch_size=len(batch_members), solver_batch_id=task_root.name,
                        oracle_case_queue_wait_seconds=underlying_oracle.state.queue_wait_seconds(member.case_id),
                        budget_timing="shared_batch_wall_excluding_all_members_waiting",
                        **underlying_oracle.state.execution_metrics(member.case_id)))
            else:
                self._record_audit(audit)
        except BaseException as audit_error:
            self._mark_unhealthy(
                "Candidate execution audit persistence could not be confirmed"
            )
            if primary_error is None:
                primary_error = CandidateExecutionInfrastructureError(
                    "Candidate execution audit could not be persisted"
                )
                primary_error.__cause__ = audit_error
            else:
                _attach_execution_error(primary_error, audit_error)

        if primary_error is not None:
            raise primary_error
        if output is None:  # pragma: no cover - guarded by status transition
            raise CandidateExecutionInfrastructureError(
                "Candidate execution completed without an output"
            )
        return output

    def _worker_environment(
        self, task_root: Path, *, gpu_device: str | None = None,
    ) -> Mapping[str, str]:
        inherited_names = (
            "PATH",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
            "LIBRARY_PATH",
            "CPATH",
            "VIRTUAL_ENV",
            "CUDA_DEVICE_ORDER",
        )
        environment = {
            name: os.environ[name]
            for name in inherited_names
            if name in os.environ
        }
        environment.update(self.environment)
        # Use the Controller's current SDK in subprocesses, even when this
        # interpreter has an editable installation pointing at an old checkout.
        environment["PYTHONPATH"] = os.pathsep.join(value for value in
            (str(Path(__file__).resolve().parents[2]), environment.get("PYTHONPATH", "")) if value)
        executable_directory = str(self.python_executable.parent)
        environment["PATH"] = os.pathsep.join(
            value
            for value in (executable_directory, environment.get("PATH", ""))
            if value
        )
        virtual_environment = self.python_executable.parent.parent
        if (
            self.python_executable.parent.name == "bin"
            and (virtual_environment / "pyvenv.cfg").is_file()
        ):
            environment["VIRTUAL_ENV"] = str(virtual_environment)
        else:
            environment.pop("VIRTUAL_ENV", None)
        environment.update(
            {
                "HOME": str(task_root),
                "TMPDIR": str(task_root),
                "TMP": str(task_root),
                "TEMP": str(task_root),
                "CUDA_VISIBLE_DEVICES": gpu_device or "",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "BLIS_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        return environment

    def _terminate_active_processes(
        self,
        execution_id: str,
    ) -> tuple[BaseException, ...]:
        with self._process_lock:
            processes = tuple(
                process
                for owner, process in self._processes.values()
                if owner == execution_id
            )
        errors: list[BaseException] = []
        for process in processes:
            try:
                _terminate_process_group(process)
            except BaseException as error:
                errors.append(error)
        return tuple(errors)

    def _mark_unhealthy(self, reason: str) -> None:
        with self._health_lock:
            if self._unhealthy_reason is not None:
                return
            self._unhealthy_reason = _bounded_diagnostic(reason, 1_024)
            retained_reason = self._unhealthy_reason
        # Persist only on failure, off the successful numerical hot path.
        try:
            _write_private_json(self.root / f"service-failure-{uuid4().hex}.json", {
                "schema": "candidate-execution-service-failure-v1",
                "reason": retained_reason,
                "observed_at_unix": time.time(),
                "thread_stack": _bounded_diagnostic("".join(traceback.format_stack())),
                "oracle_thread_stacks": _oracle_thread_stacks(),
            })
        except BaseException:
            logging.getLogger(__name__).exception("Could not persist candidate service failure")

    def _raise_if_healthy(self) -> None:
        with self._health_lock:
            reason = self._unhealthy_reason
        if reason is not None:
            raise CandidateExecutionInfrastructureError(
                "Candidate execution service is unavailable after prior "
                f"resource cleanup could not be confirmed: {reason}"
            )

    def _record_audit(self, audit: CandidateExecutionAudit) -> None:
        payload = {
            field: getattr(audit, field)
            for field in audit.__dataclass_fields__
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self._audit_lock:
            self._audits.append(audit)
            path = self.root / "candidate_execution_audits.jsonl"
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                0o600,
            )
            with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def _record_unstarted_audit(
        self,
        *,
        execution_id: str,
        index: int,
        artifact: Artifact,
        case: Case,
        plan: EvaluationPlan,
        call_root: Path,
        error: BaseException,
        requester_agent_id: str | None = None,
        submitted_at: float | None = None,
    ) -> None:
        """Record a future canceled before worker launch, including its owner."""

        workspace = call_root / f"case-{index:06d}-not-started"
        queue_seconds = 0. if submitted_at is None else max(0., time.monotonic()-submitted_at)
        self.trace("solver_not_started", execution_id=execution_id, requester_agent_id=requester_agent_id,
            case_index=index, case_ids=[case.case_id], solver_queue_wait_seconds=queue_seconds,
            error_code=_candidate_error_code(error))
        self._record_audit(
            CandidateExecutionAudit(
                schema=CANDIDATE_EXECUTION_AUDIT_SCHEMA,
                execution_id=execution_id,
                artifact_id=artifact.artifact_id,
                case_id=case.case_id,
                phase=plan.phase.value,
                case_index=index,
                status="failed",
                pid=None,
                return_code=None,
                wall_seconds=0.0,
                timeout_seconds=self.case_hard_wall_seconds,
                process_group_cleaned=True,
                workspace=str(workspace),
                stdout_path=str(workspace / "stdout.log"),
                stderr_path=str(workspace / "stderr.log"),
                response_path=str(workspace / "response.json"),
                error_code=("candidate_execution_cancelled" if isinstance(error, CandidateExecutionCancelled)
                            else "candidate_not_started"),
                execution_started=False,
                batch_failure_code=_candidate_error_code(error),
                requester_agent_id=requester_agent_id,
                solver_queue_wait_seconds=queue_seconds,
            )
        )


class _LocalPublicSession:
    def __init__(
        self,
        case: Case,
        methods: Sequence[str],
        invoke: PublicSessionInvoker | None,
    ) -> None:
        self._case_id = case.case_id
        self._methods = frozenset(methods)
        self._invoke = invoke

    def __getattr__(self, name: str):
        if name not in self._methods:
            raise AttributeError(name)
        invoke = self._invoke
        if invoke is None:  # pragma: no cover - guarded by caller
            raise RuntimeError("public Candidate session is unavailable")

        def call(*args: object, **kwargs: object) -> object:
            started = time.perf_counter()
            with _capture_oracle_wait() as timing:
                try:
                    return invoke(self._case_id, name, tuple(args), dict(kwargs))
                finally:
                    _record_oracle_response_wait(
                        uuid4().hex,
                        {
                            "schema": ORACLE_WAIT_TIMING_SCHEMA,
                            "queue_wait_seconds": 0.0,
                            "lock_wait_seconds": timing.lock_wait_seconds,
                        },
                        rpc_wall_seconds=time.perf_counter() - started,
                    )

        return call

    def close(self) -> None:
        return None


@dataclass
class _QueuedOracleCall:
    request_id: str
    case_id: str
    method: str
    args: tuple[object, ...]
    kwargs: Mapping[str, object]
    queue_wait_before: float
    status: str = "queued"
    result: Mapping[str, object] | None = None
    error: BaseException | None = None
    # The upstream total includes both admitted-queue and plugin-lock wait.
    queue_wait_seconds: float = 0.0
    lock_wait_seconds: float = 0.0
    submitted_at_unix: float = field(default_factory=time.time)
    submitted_at_monotonic: float = field(default_factory=time.monotonic)


class _CandidateOracleState:
    """Authenticate before queueing; keep scientific work off the HTTP handler."""

    def __init__(self, codec, methods, invoke, authorizations, *, max_workers=1,
                 result_codec=None, trace_call=None) -> None:
        self._trace_call = trace_call
        self.codec = codec
        self.result_codec = codec if result_codec is None else result_codec
        self.methods = frozenset(methods)
        self.invoke = invoke
        self._lifecycle = Condition(Lock())
        self._accepting = True
        self._case_by_token = dict(authorizations)
        self._token_by_case = {case_id: token for token, case_id in authorizations.items()}
        if len(self._case_by_token) != len(self._token_by_case):
            raise ValueError("Candidate oracle Case authorizations must be unique")
        self._active_by_case = {case_id: 0 for case_id in self._token_by_case}
        self._wait_total = {case_id: 0.0 for case_id in self._token_by_case}
        self._wait_started: dict[str, float] = {}
        self._wait_depth: dict[str, int] = {}
        self._requests: dict[str, _QueuedOracleCall] = {}
        self._wait_groups = {}
        self._group_by_case = {}
        self._request_by_case: dict[str, str] = {}
        self._work: Queue[object] = Queue(maxsize=128)
        self._calls = {key: 0 for key in self._token_by_case}
        self._service_seconds = {key: 0.0 for key in self._token_by_case}
        self._codec_seconds = {key: 0.0 for key in self._token_by_case}
        self._service_started: dict[str, tuple[float, float]] = {}
        self._last_method: dict[str, str] = {}
        self._failures: dict[str, list[dict[str, object]]] = {}
        self._max_workers = max_workers
        self._workers: list[Thread] = []
        # A queued evaluation can own a host before any Candidate gets a global
        # execution slot. Allocate scientific threads only as real RPCs arrive.

    def register_wait_group(self, case_ids):
        with self._lifecycle:
            key = tuple(case_ids)
            if any(case not in self._token_by_case or case in self._group_by_case for case in key):
                raise ValueError("solver batch must own distinct authorized Cases")
            self._wait_groups[key] = {"total": 0., "started": None}
            for case in key:
                self._group_by_case[case] = key

    def _group_wait_locked(self, key):
        row = self._wait_groups[key]
        return row["total"] + (0. if row["started"] is None else time.monotonic() - row["started"])

    def group_wait_seconds(self, case_ids):
        with self._lifecycle:
            return self._group_wait_locked(tuple(case_ids))

    def _update_wait_group_locked(self, case_id):
        key = self._group_by_case.get(case_id)
        if key is None:
            return
        row = self._wait_groups[key]
        waiting = all(self._wait_depth.get(case, 0) > 0 for case in key)
        if waiting and row["started"] is None:
            row["started"] = time.monotonic()
        elif not waiting and row["started"] is not None:
            row["total"] += time.monotonic() - row["started"]
            row["started"] = None

    def _authorize_locked(self, token: str) -> str:
        if not self._accepting:
            raise PermissionError("Candidate oracle has expired")
        case_id = self._case_by_token.get(token)
        if case_id is None:
            raise PermissionError("Candidate oracle token is not authorized")
        return case_id

    def _decode_request(self, token, params):
        with self._lifecycle:
            authorized_case_id = self._authorize_locked(token)
        if set(params) != {"case_id", "method", "args", "kwargs"}:
            raise ValueError("Candidate oracle request fields are invalid")
        case_id, method = params.get("case_id"), params.get("method")
        if case_id != authorized_case_id:
            raise PermissionError("Candidate oracle token is not authorized for this Case")
        if not isinstance(method, str) or method not in self.methods:
            raise PermissionError("Candidate oracle method is not authorized")
        codec_started = time.monotonic()
        args = self.codec.decode(params.get("args"))
        kwargs = self.codec.decode(params.get("kwargs"))
        codec_seconds = time.monotonic() - codec_started
        if not isinstance(args, tuple) or not isinstance(kwargs, Mapping):
            raise ValueError("Candidate oracle arguments are invalid")
        if any(not isinstance(name, str) for name in kwargs):
            raise ValueError("Candidate oracle keyword names are invalid")
        return case_id, method, args, dict(kwargs), codec_seconds

    def _enqueue_locked(self, request_id, decoded):
        case_id, method, args, kwargs, codec_seconds = decoded
        if request_id in self._requests or case_id in self._request_by_case:
            raise ValueError("Candidate Case already has an outstanding oracle request")
        if len(self._requests) >= 128:
            raise ValueError("Candidate oracle request queue is full")
        demand = 1 + sum(call.status in {"queued", "running"}
                         for call in self._requests.values())
        if len(self._workers) < min(self._max_workers, demand):
            worker = Thread(target=self._serve_work,
                name=f"candidate-public-oracle-science-{len(self._workers)}", daemon=True)
            worker.start()
            self._workers.append(worker)
        call = _QueuedOracleCall(request_id, case_id, method, args, dict(kwargs), self._wait_total[case_id])
        self._requests[request_id] = call
        self._request_by_case[case_id] = request_id
        self._codec_seconds[case_id] += codec_seconds
        self._change_wait_locked(case_id, True)
        try:
            self._work.put_nowait(call)
        except Full:
            self._change_wait_locked(case_id, False)
            del self._requests[request_id]
            del self._request_by_case[case_id]
            raise ValueError("Candidate oracle request queue is full") from None


    def submit(self, token: str, request_id: str, params: Mapping[str, object]) -> None:
        decoded = self._decode_request(token, params)
        with self._lifecycle:
            self._authorize_locked(token)
            self._enqueue_locked(request_id, decoded)

    def submit_many(self, entries):
        # Decode all members before admission: a bad capability never triggers a
        # partial batch. Scientific scheduling and wait accounting stay per Case.
        decoded = [(entry, self._decode_request(entry['token'], entry['params']))
                   for entry in entries]
        with self._lifecycle:
            ids = [entry['request_id'] for entry, _ in decoded]
            cases = [data[0] for _, data in decoded]
            if (len(set(ids)) != len(ids) or len(set(cases)) != len(cases)
                    or any(key in self._requests for key in ids)
                    or any(key in self._request_by_case for key in cases)
                    or len(self._requests) + len(entries) > 128):
                raise ValueError('Oracle batch contains duplicate or outstanding requests')
            for entry, _ in decoded:
                self._authorize_locked(entry['token'])
            for entry, data in decoded:
                self._enqueue_locked(entry['request_id'], data)

    def poll_many(self, entries):
        # Do not consume any completed member until the entire batch is ready.
        with self._lifecycle:
            for entry in entries:
                case_id = self._authorize_locked(entry['token'])
                call = self._requests.get(entry['request_id'])
                if call is None or call.case_id != case_id:
                    raise PermissionError('Oracle batch request is not authorized')
            if any(self._requests[e['request_id']].status in {'queued', 'running'} for e in entries):
                return None
        rows = []
        for entry in entries:
            try:
                result = self.poll(entry['token'], entry['request_id'])
                rows.append({'request_id': entry['request_id'], 'result': result})
            except _OracleRequestFailure as error:
                rows.append({'request_id': entry['request_id'], 'result': {
                    'error': 'oracle_call_failed',
                    'queue_wait_seconds': error.queue_wait_seconds,
                    'lock_wait_seconds': error.lock_wait_seconds}})
        return rows

    def poll(self, token: str, request_id: str) -> Mapping[str, object] | None:
        with self._lifecycle:
            case_id = self._authorize_locked(token)
            call = self._requests.get(request_id)
            if call is None or call.case_id != case_id:
                raise PermissionError("Candidate oracle request is not authorized")
            if call.status in {"queued", "running"}:
                return None
            del self._requests[request_id]
            self._request_by_case.pop(case_id, None)
            if call.error is not None:
                raise _OracleRequestFailure(
                    call.queue_wait_seconds, call.lock_wait_seconds
                ) from call.error
            if call.result is None:
                raise CandidateExecutionCancelled("Candidate oracle request was cancelled")
            return {
                **call.result,
                "queue_wait_seconds": call.queue_wait_seconds,
                "lock_wait_seconds": call.lock_wait_seconds,
                **({"group_wait_seconds": self._group_wait_locked(self._group_by_case[case_id])}
                   if case_id in self._group_by_case else {}),
            }

    def _change_wait_locked(self, case_id: str, waiting: bool) -> None:
        depth = self._wait_depth.get(case_id, 0)
        if waiting:
            if depth == 0:
                self._wait_started[case_id] = time.monotonic()
            self._wait_depth[case_id] = depth + 1
        elif depth:
            self._wait_depth[case_id] = depth - 1
            if depth == 1:
                self._wait_total[case_id] += max(0.0, time.monotonic() - self._wait_started.pop(case_id))

        self._update_wait_group_locked(case_id)

    def _observe_wait(self, case_id: str, waiting: bool) -> None:
        with self._lifecycle:
            self._change_wait_locked(case_id, waiting)

    def queue_wait_seconds(self, case_id: str) -> float:
        with self._lifecycle:
            started = self._wait_started.get(case_id)
            return self._wait_total.get(case_id, 0.0) + (
                0.0 if started is None else max(0.0, time.monotonic() - started)
            )

    def execution_metrics(self, case_id: str) -> dict[str, object]:
        with self._lifecycle:
            service = self._service_seconds.get(case_id, 0.0)
            running = self._service_started.get(case_id)
            if running is not None:
                now = time.monotonic()
                waiting = self._wait_total[case_id]
                if case_id in self._wait_started:
                    waiting += now - self._wait_started[case_id]
                service += max(0.0, now - running[0] - (waiting - running[1]))
            request_id = self._request_by_case.get(case_id)
            call = self._requests.get(request_id)
            return {"oracle_call_count": self._calls.get(case_id, 0),
                    "oracle_service_wall_seconds": service,
                    "oracle_codec_wall_seconds": self._codec_seconds.get(case_id, 0.0),
                    "oracle_last_method": self._last_method.get(case_id),
                    "oracle_inflight_method": call.method if call is not None and running is not None else None,
                    "oracle_inflight_wall_seconds": (0.0 if running is None else
                                                       max(0.0, time.monotonic() - running[0]))}

    def failure_diagnostics(self, case_id: str) -> tuple[Mapping[str, object], ...]:
        with self._lifecycle:
            return tuple(dict(row) for row in self._failures.get(case_id, ()))

    def _check_case_active(self, case_id: str) -> None:
        with self._lifecycle:
            if not self._accepting or case_id not in self._token_by_case:
                raise CandidateExecutionCancelled("Candidate oracle Case has been revoked")

    def _serve_work(self) -> None:
        while True:
            item = self._work.get()
            try:
                if item is None:
                    return
                call = item
                with self._lifecycle:
                    if call.status != "queued":
                        continue
                    self._change_wait_locked(call.case_id, False)
                    if not self._accepting or call.case_id not in self._token_by_case:
                        call.status = "cancelled"
                        continue
                    call.status = "running"
                    self._active_by_case[call.case_id] += 1
                    self._calls[call.case_id] += 1
                    self._last_method[call.case_id] = call.method
                    self._service_started[call.case_id] = (
                        time.monotonic(), self._wait_total[call.case_id])
                wait_timing = None
                profile = None
                call_started = time.monotonic()
                service_before = self._service_seconds[call.case_id]
                codec_before = self._codec_seconds[call.case_id]
                try:
                    if self.invoke is None:
                        raise PermissionError("Candidate oracle is unavailable")
                    with _capture_oracle_wait() as wait_timing, bind_queue_wait_observer(
                        lambda waiting: self._observe_wait(call.case_id, waiting),
                        check_cancelled=lambda: self._check_case_active(call.case_id),
                    ):
                        with capture_oracle_profile(self._trace_call is not None) as profile:
                            value = self.invoke(call.case_id, call.method, call.args, call.kwargs)
                    self._finish_service_timing(call.case_id)
                    codec_started = time.monotonic()
                    try:
                        call.result = {"value": self.result_codec.encode(value)}
                    finally:
                        with self._lifecycle:
                            self._codec_seconds[call.case_id] += time.monotonic() - codec_started
                except BaseException as error:
                    call.error = error
                    diagnostic = {
                        "request_id": call.request_id,
                        "method": call.method,
                        "error_type": type(error).__name__,
                        "message": _bounded_diagnostic(str(error)),
                        "traceback": _bounded_diagnostic("".join(traceback.format_exception(
                            type(error), error, error.__traceback__))),
                    }
                    with self._lifecycle:
                        failures = self._failures.setdefault(call.case_id, [])
                        # First failures are useful; bounded storage prevents a
                        # candidate repeatedly catching errors growing it forever.
                        if len(failures) < 8:
                            failures.append(diagnostic)
                finally:
                    self._finish_service_timing(call.case_id)
                    with self._lifecycle:
                        call.queue_wait_seconds = max(0.0, self._wait_total[call.case_id] - call.queue_wait_before)
                        # Classification is a decomposition of that total, not
                        # another refundable interval. The live parent ledger
                        # remains the authoritative execution deadline source.
                        call.lock_wait_seconds = min(
                            call.queue_wait_seconds,
                            0.0 if wait_timing is None else wait_timing.lock_wait_seconds,
                        )
                        service_wall = self._service_seconds[call.case_id]-service_before
                        codec_wall = self._codec_seconds[call.case_id]-codec_before
                        call.status = "completed"
                        self._active_by_case[call.case_id] -= 1
                        self._lifecycle.notify_all()
                    if self._trace_call is not None:
                        shape = getattr(call.args[0], "shape", ()) if call.args else ()
                        self._trace_call(request_id=call.request_id, case_id=call.case_id,
                            method=call.method, point_count=int(shape[0]) if shape else None,
                            status="completed" if call.error is None else "failed",
                            submitted_at_unix=call.submitted_at_unix,
                            started_at_monotonic=call_started,
                            total_wall_seconds=time.monotonic()-call.submitted_at_monotonic,
                            host_queue_seconds=call_started-call.submitted_at_monotonic,
                            queue_wait_seconds=call.queue_wait_seconds,
                            service_wall_seconds=service_wall,
                            codec_wall_seconds=codec_wall,
                            numerical_profile=profile,
                            error_type=None if call.error is None else type(call.error).__name__)
            finally:
                self._work.task_done()

    def _finish_service_timing(self, case_id: str) -> None:
        with self._lifecycle:
            running = self._service_started.pop(case_id, None)
            if running is not None:
                self._service_seconds[case_id] += max(
                    0.0, time.monotonic() - running[0]
                    - (self._wait_total[case_id] - running[1]))

    def _cancel_queued_locked(self, case_id: str) -> None:
        request_id = self._request_by_case.get(case_id)
        call = self._requests.get(request_id)
        if call is not None and call.status == "queued":
            self._change_wait_locked(case_id, False)
            call.status = "cancelled"
            call.queue_wait_seconds = max(0., self._wait_total[case_id]-call.queue_wait_before)
            if self._trace_call is not None:
                wall = max(0., time.monotonic()-call.submitted_at_monotonic)
                shape = getattr(call.args[0], "shape", ()) if call.args else ()
                self._trace_call(request_id=call.request_id, case_id=case_id,
                    method=call.method, point_count=int(shape[0]) if shape else None,
                    status="cancelled_before_service", submitted_at_unix=call.submitted_at_unix,
                    started_at_monotonic=None, total_wall_seconds=wall, host_queue_seconds=wall,
                    queue_wait_seconds=call.queue_wait_seconds, service_wall_seconds=0.,
                    codec_wall_seconds=0., error_type="CandidateExecutionCancelled")

    def revoke_case(self, case_id: str) -> None:
        with self._lifecycle:
            token = self._token_by_case.pop(case_id, None)
            if token is not None:
                self._case_by_token.pop(token, None)
            self._cancel_queued_locked(case_id)
            self._lifecycle.notify_all()

    def revoke_all(self) -> None:
        with self._lifecycle:
            self._accepting = False
            self._case_by_token.clear()
            self._token_by_case.clear()
            for case_id in tuple(self._request_by_case):
                self._cancel_queued_locked(case_id)
            self._lifecycle.notify_all()

    def stop_worker(self, timeout: float) -> bool:
        # Revocation has made queued work inert; discard it before signalling.
        while True:
            try:
                self._work.get_nowait()
            except Empty:
                break
            self._work.task_done()
        for _ in self._workers:
            self._work.put_nowait(None)
        deadline = time.monotonic() + max(timeout, 0.0)
        for worker in self._workers:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))
        return not any(worker.is_alive() for worker in self._workers)

    def wait_case_drained(self, case_id: str, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._lifecycle:
            while self._active_by_case.get(case_id, 0):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._lifecycle.wait(remaining)
            return True

    def wait_all_drained(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._lifecycle:
            while any(self._active_by_case.values()):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._lifecycle.wait(remaining)
            return True

class _CandidateOracleHTTPServer(HTTPServer):
    """Bounded quick HTTP handling, separate from serial scientific work.

    Admission authenticates and records queue wait before returning. Polls stay
    responsive while a trusted numerical call runs or waits on the plugin lock;
    the number of threads does not grow with the number of Candidate requests.
    """

    allow_reuse_address = True
    request_queue_size = 128
    _accepted_queue_size = 128
    _accepted_socket_timeout_seconds = 30.0

    def __init__(self, address, handler, *, state):
        super().__init__(address, handler)
        self.state = state
        self._accepted_requests: Queue[object] = Queue(
            maxsize=self._accepted_queue_size
        )
        self._request_worker = Thread(
            target=self._serve_accepted_requests,
            name="candidate-public-oracle-handler",
            daemon=True,
        )
        try:
            self._request_worker.start()
        except BaseException:
            super().server_close()
            raise

    def process_request(self, request, client_address) -> None:
        try:
            request.settimeout(self._accepted_socket_timeout_seconds)
            self._accepted_requests.put_nowait((request, client_address))
        except Full:
            self.shutdown_request(request)

    def _serve_accepted_requests(self) -> None:
        while True:
            item = self._accepted_requests.get()
            try:
                if item is None:
                    return
                request, client_address = item
                try:
                    self.finish_request(request, client_address)
                except Exception:
                    self.handle_error(request, client_address)
                finally:
                    self.shutdown_request(request)
            finally:
                self._accepted_requests.task_done()

    def close_accepted_requests(self, *, join_timeout_seconds: float) -> bool:
        """Close queued sockets and stop the worker unless its handler is stuck."""

        while True:
            try:
                item = self._accepted_requests.get_nowait()
            except Empty:
                break
            try:
                if item is not None:
                    request, _client_address = item
                    self.shutdown_request(request)
            finally:
                self._accepted_requests.task_done()
        try:
            self._accepted_requests.put_nowait(None)
        except Full:  # pragma: no cover - queue was drained immediately above
            return False
        self._request_worker.join(timeout=max(join_timeout_seconds, 0.0))
        return not self._request_worker.is_alive()


class _CandidateOracleHost:
    def __init__(
        self,
        sidecar_root: Path,
        *,
        case_ids: Sequence[str],
        methods: Sequence[str],
        invoke: PublicSessionInvoker | None,
        close: PublicSessionCloser | None,
        on_drain_timeout: Callable[[str], None] | None = None,
        max_workers: int = 1,
        oracle_inline_array_bytes: int = _ORACLE_INLINE_ARRAY_BYTES,
        oracle_trusted_results: bool = True,
        trace_call=None,
    ) -> None:
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or not 1 <= max_workers <= 128:
            raise ValueError("Oracle max_workers must be an integer in [1, 128]")
        prepared_case_ids = tuple(case_ids)
        if any(
            not isinstance(case_id, str) or not case_id
            for case_id in prepared_case_ids
        ):
            raise ValueError("Candidate oracle Case ids must be nonempty strings")
        if len(prepared_case_ids) != len(set(prepared_case_ids)):
            raise ValueError("Candidate oracle Case ids must be unique")
        _validate_oracle_transport(oracle_inline_array_bytes, oracle_trusted_results)
        self.codec = AgentWireCodec(
            sidecar_root, inline_array_bytes=oracle_inline_array_bytes,
            durable_sidecars=False,
        )
        self.result_codec = (
            AgentWireCodec(sidecar_root, inline_array_bytes=oracle_inline_array_bytes,
                           durable_sidecars=False, validate_array_finiteness=False)
            if oracle_trusted_results else self.codec
        )
        self._oracle_trusted_results = oracle_trusted_results
        self._tokens_by_case = {
            case_id: secrets.token_urlsafe(32) for case_id in prepared_case_ids
        }
        self.state = _CandidateOracleState(
            self.codec,
            methods,
            invoke,
            {
                token: case_id
                for case_id, token in self._tokens_by_case.items()
            },
            max_workers=max_workers,
            result_codec=self.result_codec,
            trace_call=trace_call,
        )
        self._methods = tuple(methods)
        self._sidecar_root = sidecar_root
        self._close_session = close
        self._on_drain_timeout = on_drain_timeout
        try:
            self._server = _CandidateOracleHTTPServer(
                ("127.0.0.1", 0), _candidate_oracle_handler(), state=self.state,
            )
        except BaseException:
            self.state.revoke_all()
            self.state.stop_worker(0.1)
            raise
        host, port = self._server.server_address
        self.endpoint = f"http://{host}:{port}{_ORACLE_PATH}"
        self._thread = Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.05},
            name="candidate-public-oracle",
            daemon=True,
        )
        self._started = False
        self._closed = False
        self._lock = Lock()
        self._finished_cases: set[str] = set()
        self._case_finish_locks = {case_id: Lock() for case_id in prepared_case_ids}

    def manifest_for_case(self, case_id: str) -> Mapping[str, object]:
        with self._lock:
            if self._closed or case_id in self._finished_cases:
                raise CandidateExecutionInfrastructureError(
                    "Candidate oracle Case authorization has expired"
                )
            try:
                token = self._tokens_by_case[case_id]
            except KeyError as error:
                raise CandidateExecutionInfrastructureError(
                    "Candidate oracle Case is not authorized"
                ) from error
            return MappingProxyType(
                {
                    "schema": "canonical-candidate-public-oracle-v1",
                    "available": bool(self._methods),
                    "endpoint": self.endpoint,
                    "token": token,
                    "case_id": case_id,
                    "methods": list(self._methods),
                    "sidecar_root": str(self._sidecar_root),
                    "max_rpc_bytes": _MAX_RPC_BYTES,
                    "inline_array_bytes": self.codec.inline_array_bytes,
                    "oracle_inline_array_bytes": self.codec.inline_array_bytes,
                    "oracle_trusted_results": self._oracle_trusted_results,
                    "bulk_queries": True,
                    "wait_timing_schema": ORACLE_WAIT_TIMING_SCHEMA,
                }
            )

    def finish_case(
        self,
        case_id: str,
        *,
        drain_timeout_seconds: float = 30.0,
    ) -> None:
        timeout = _validated_oracle_drain_timeout(drain_timeout_seconds)
        deadline = time.monotonic() + timeout
        case_lock = self._case_finish_locks.get(case_id)
        if case_lock is None:
            raise CandidateExecutionInfrastructureError("Candidate oracle cannot finish an unknown Case")
        if not case_lock.acquire(timeout=timeout):
            message = "Candidate public session close is already in progress"
            self._retain_unhealthy(message)
            raise CandidateExecutionInfrastructureError(message)
        try:
            with self._lock:
                if case_id in self._finished_cases:
                    return
            self.state.revoke_case(case_id)
            if not self.state.wait_case_drained(case_id, max(0.0, deadline - time.monotonic())):
                message = "Candidate oracle Case calls did not drain"
                self._retain_unhealthy(message)
                with self._lock:
                    self._finished_cases.add(case_id)
                self._defer_case_close(case_id)
                raise CandidateExecutionInfrastructureError(message)
            try:
                if self._close_session is not None:
                    self._close_session(case_id)
            except BaseException:
                self._retain_unhealthy("Candidate public session cleanup could not be confirmed")
                raise
            finally:
                with self._lock:
                    self._finished_cases.add(case_id)
        finally:
            case_lock.release()

    def start(self) -> None:
        with self._lock:
            if self._started or self._closed:
                raise RuntimeError("Candidate oracle host is single-use")
            try:
                self._thread.start()
            except BaseException as error:
                self.state.revoke_all()
                try:
                    self._server.server_close()
                finally:
                    self._server.close_accepted_requests(
                        join_timeout_seconds=0.1,
                    )
                    self.state.stop_worker(0.1)
                    self._closed = True
                    self._retain_unhealthy(
                        "Candidate oracle listener could not start"
                    )
                raise CandidateExecutionInfrastructureError(
                    "Candidate oracle listener could not start"
                ) from error
            self._started = True

    def close(
        self, *, drain_timeout_seconds: float = 30.0,
        _deadline_monotonic: float | None = None,
    ) -> None:
        timeout = _validated_oracle_drain_timeout(drain_timeout_seconds)
        # Multi-host calls pass one absolute deadline; never reset their grace.
        deadline = (time.monotonic() + max(timeout, 0.1)
                    if _deadline_monotonic is None else _deadline_monotonic)
        with self._lock:
            if self._closed:
                return
            # Revoke admission atomically, then release the host lock before
            # waiting for per-Case cleanup. finish_case uses the opposite scope.
            self._closed = True
        primary_error: BaseException | None = None

        def retain(error: BaseException) -> None:
            nonlocal primary_error
            if primary_error is None:
                primary_error = error
            else:
                _attach_execution_error(primary_error, error)

        def remaining() -> float:
            return max(0.0, deadline - time.monotonic())

        self.state.revoke_all()
        if self._started:
            try:
                self._server.shutdown()
            except BaseException as error:
                retain(error)
        try:
            self._server.server_close()
        except BaseException as error:
            retain(error)
        if self._started:
            self._thread.join(timeout=remaining())
            if self._thread.is_alive():
                retain(CandidateExecutionInfrastructureError("Candidate oracle listener did not stop"))
        request_worker_stopped = self._server.close_accepted_requests(join_timeout_seconds=remaining())
        science_worker_stopped = self.state.stop_worker(remaining())
        if not self.state.wait_all_drained(remaining()):
            message = "Candidate oracle calls did not drain"
            self._retain_unhealthy(message)
            retain(CandidateExecutionInfrastructureError(message))
        elif not request_worker_stopped or not science_worker_stopped:
            message = "Candidate oracle request worker did not stop"
            self._retain_unhealthy(message)
            retain(CandidateExecutionInfrastructureError(message))
        for case_id in self._tokens_by_case:
            try:
                # Same per-Case idempotence boundary as the worker cleanup path;
                # concurrent finish_case/close cannot close one session twice.
                self.finish_case(case_id, drain_timeout_seconds=remaining())
            except BaseException as error:
                retain(error)
        if primary_error is not None:
            self._retain_unhealthy("Candidate oracle cleanup could not be confirmed")
            raise primary_error

    def _retain_unhealthy(self, reason: str) -> None:
        callback = self._on_drain_timeout
        if callback is not None:
            callback(reason)

    def _defer_case_close(self, case_id: str) -> None:
        """Close once an already-running trusted oracle call eventually returns."""

        close = self._close_session
        if close is None:
            return

        def close_after_drain() -> None:
            while not self.state.wait_case_drained(case_id, 60.0):
                pass
            try:
                close(case_id)
            except BaseException:
                # The initiating execution has already failed closed and its
                # synchronous error path cannot safely await this daemon.
                return

        Thread(
            target=close_after_drain,
            name=f"candidate-oracle-close-{case_id[:12]}",
            daemon=True,
        ).start()


class _CandidateOracleHostGroup:
    """Route immutable Case assignments to independent single-worker hosts.

    A host retains its existing queue timing, token authorization and drain
    lifecycle. This group changes transport concurrency only: it neither
    manufactures Cases nor changes the scientific invocation port's locking.
    """

    def __init__(
        self,
        sidecar_root: Path,
        *,
        case_ids: Sequence[str],
        methods: Sequence[str],
        invoke: PublicSessionInvoker | None,
        close: PublicSessionCloser | None,
        host_count: int,
        on_drain_timeout: Callable[[str], None] | None = None,
        max_workers: int = 1,
        oracle_inline_array_bytes: int = _ORACLE_INLINE_ARRAY_BYTES,
        oracle_trusted_results: bool = True,
        trace_call=None,
    ) -> None:
        _validate_oracle_transport(oracle_inline_array_bytes, oracle_trusted_results)
        if isinstance(host_count, bool) or not isinstance(host_count, int):
            raise TypeError("oracle host_count must be an integer")
        if host_count < 1:
            raise ValueError("oracle host_count must be positive")
        prepared = tuple(case_ids)
        if not prepared or any(not isinstance(case_id, str) or not case_id for case_id in prepared):
            raise ValueError("Candidate oracle Case ids must be nonempty strings")
        if len(prepared) != len(set(prepared)):
            raise ValueError("Candidate oracle Case ids must be unique")
        self._hosts: list[_CandidateOracleHost] = []
        self._host_by_case: dict[str, _CandidateOracleHost] = {}
        try:
            # Preserve index % requested_count affinity without creating empty
            # hosts when the cohort is smaller than the available lane count.
            for lane in range(min(host_count, len(prepared))):
                lane_cases = prepared[lane::host_count]
                host = _CandidateOracleHost(
                    sidecar_root / f"lane-{lane:03d}",
                    case_ids=lane_cases,
                    methods=methods,
                    invoke=invoke,
                    close=close,
                    on_drain_timeout=on_drain_timeout,
                    max_workers=min(max_workers, len(lane_cases)),
                    oracle_inline_array_bytes=oracle_inline_array_bytes,
                    oracle_trusted_results=oracle_trusted_results,
                    trace_call=trace_call,
                )
                self._hosts.append(host)
                self._host_by_case.update((case_id, host) for case_id in lane_cases)
        except BaseException as error:
            try:
                self.close(drain_timeout_seconds=0.0)
            except BaseException as cleanup_error:
                _attach_execution_error(error, cleanup_error)
            raise

    def _host_for_case(self, case_id: str) -> _CandidateOracleHost:
        try:
            return self._host_by_case[case_id]
        except KeyError as error:
            raise CandidateExecutionInfrastructureError(
                "Candidate oracle Case is not authorized"
            ) from error

    def manifest_for_case(self, case_id: str) -> Mapping[str, object]:
        return self._host_for_case(case_id).manifest_for_case(case_id)

    def finish_case(self, case_id: str, *, drain_timeout_seconds: float = 30.0) -> None:
        self._host_for_case(case_id).finish_case(
            case_id, drain_timeout_seconds=drain_timeout_seconds
        )

    def start(self) -> None:
        try:
            for host in self._hosts:
                host.start()
        except BaseException as error:
            try:
                self.close(drain_timeout_seconds=0.0)
            except BaseException as cleanup_error:
                _attach_execution_error(error, cleanup_error)
            raise

    def close(self, *, drain_timeout_seconds: float = 30.0) -> None:
        timeout = _validated_oracle_drain_timeout(drain_timeout_seconds)
        primary_error: BaseException | None = None
        deadline = time.monotonic() + max(timeout, 0.1)
        for host in self._hosts:
            try:
                host.close(drain_timeout_seconds=max(0.0, deadline - time.monotonic()),
                           _deadline_monotonic=deadline)
            except BaseException as error:
                if primary_error is None:
                    primary_error = error
                else:
                    _attach_execution_error(primary_error, error)
        if primary_error is not None:
            raise primary_error


class CandidatePublicSessionClient:
    """Worker-side proxy for one public-only Controller capability."""

    def __init__(self, manifest: Mapping[str, object]) -> None:
        if manifest.get("schema") != "canonical-candidate-public-oracle-v1":
            raise CandidateExecutionInfrastructureError(
                "Candidate public-oracle manifest is invalid"
            )
        if manifest.get("wait_timing_schema") != ORACLE_WAIT_TIMING_SCHEMA:
            raise CandidateExecutionInfrastructureError(
                "Candidate public-oracle wait timing contract is invalid"
            )
        endpoint = manifest.get("endpoint")
        parsed = urlparse(endpoint if isinstance(endpoint, str) else "")
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.path != _ORACLE_PATH
            or parsed.query
            or parsed.fragment
        ):
            raise CandidateExecutionInfrastructureError(
                "Candidate public-oracle endpoint is not loopback"
            )
        token = manifest.get("token")
        case_id = manifest.get("case_id")
        methods = manifest.get("methods")
        sidecar_root = manifest.get("sidecar_root")
        if not isinstance(token, str) or not token:
            raise CandidateExecutionInfrastructureError(
                "Candidate public-oracle token is invalid"
            )
        if not isinstance(case_id, str) or not case_id:
            raise CandidateExecutionInfrastructureError(
                "Candidate public-oracle Case authorization is invalid"
            )
        if not isinstance(methods, list):
            raise CandidateExecutionInfrastructureError(
                "Candidate public-oracle methods are invalid"
            )
        self.endpoint = endpoint
        self.token = token
        self.case_id = case_id
        self.methods = _validated_methods(methods)
        self.bulk_queries = manifest.get("bulk_queries") is True
        self.available = manifest.get("available") is True
        if self.available != bool(self.methods):
            raise CandidateExecutionInfrastructureError(
                "Candidate public-oracle availability is inconsistent"
            )
        if not isinstance(sidecar_root, str) or not sidecar_root:
            raise CandidateExecutionInfrastructureError(
                "Candidate public-oracle sidecar root is invalid"
            )
        # Missing field means an older 64 KiB host, so new clients can still
        # communicate with existing manifests. Reject unbounded inline offers.
        inline_array_bytes = manifest.get(
            "oracle_inline_array_bytes", manifest.get("inline_array_bytes", 64 * 1024))
        trusted_results = manifest.get("oracle_trusted_results", "inline_array_bytes" in manifest)
        try:
            _validate_oracle_transport(inline_array_bytes, trusted_results)
        except (TypeError, ValueError) as error:
            raise CandidateExecutionInfrastructureError(
                "Candidate public-oracle inline array limit or transport policy is invalid"
            ) from error
        self.codec = AgentWireCodec(
            sidecar_root, inline_array_bytes=inline_array_bytes,
            durable_sidecars=False,
        )
        self.result_codec = (
            AgentWireCodec(sidecar_root, inline_array_bytes=inline_array_bytes,
                           durable_sidecars=False, validate_array_finiteness=False)
            if trusted_results else self.codec
        )
        self._opener = build_opener(ProxyHandler({}))
        self.queue_wait_seconds = 0.0
        # Candidate helper threads do not inherit ContextVars. Keep the trusted
        # execution-owned meter, so successful serialized RPCs credit the same
        # ledger measured by the Task on the main worker thread.
        self._solve_meter = _current_candidate_solve_budget_meter()

    def session_factory(self, case: Case) -> object:
        if not self.available:
            raise RuntimeError("this equation does not expose a public case session")
        if case.case_id != self.case_id:
            raise PermissionError(
                "Candidate public-oracle token is scoped to another Case"
            )
        return _RemotePublicSession(self, case.case_id)

    @staticmethod
    def invoke_many(clients, method, points_by_case, *, max_workers=32):
        """One transport exchange per compatible host, independent Case work."""
        clients = tuple(clients)
        if not clients:
            return {}
        groups = {}
        for client in clients:
            if method not in client.methods:
                raise AttributeError(method)
            groups.setdefault(client.endpoint, []).append(client)
        # Honour the SDK's outgoing concurrency limit and older host manifests.
        if (len(groups) != 1 or len(clients) > max_workers
                or not all(c.bulk_queries for c in clients)):
            with ThreadPoolExecutor(max_workers=min(max_workers, len(clients))) as pool:
                futures = {c.case_id: pool.submit(c.invoke, c.case_id, method,
                    (points_by_case[c.case_id],), {}) for c in clients}
                return {key: future.result() for key, future in futures.items()}
        first = clients[0]
        entries = [{'token': c.token, 'request_id': uuid4().hex, 'params': {
            'case_id': c.case_id, 'method': method,
            'args': c.codec.encode((points_by_case[c.case_id],)),
            'kwargs': c.codec.encode({})}} for c in clients]
        batch_id = uuid4().hex
        action = 'submit_many'
        delay = 0.
        started = time.perf_counter()
        while True:
            body = json.dumps({'schema': CANDIDATE_ORACLE_REQUEST_SCHEMA,
                'request_id': batch_id, 'action': action,
                'params': {'requests': entries}}, separators=(',', ':')).encode()
            if len(body) > _MAX_RPC_BYTES:
                # No bulk work has been submitted yet. Large independent queries
                # retain the existing sidecar/per-Case transport path.
                with ThreadPoolExecutor(max_workers=min(max_workers, len(clients))) as pool:
                    futures = {c.case_id: pool.submit(c.invoke, c.case_id, method,
                        (points_by_case[c.case_id],), {}) for c in clients}
                    return {key: f.result() for key, f in futures.items()}
            request = Request(first.endpoint, data=body, method='POST',
                headers={'Content-Type': 'application/json', _TOKEN_HEADER: first.token})
            try:
                with first._opener.open(request, timeout=30.) as response:
                    status = response.status
                    raw = response.read(_MAX_RPC_BYTES + 1)
            except (HTTPError, URLError, TimeoutError, socket.timeout) as error:
                raise CandidateExecutionError('Candidate batch public-oracle call failed') from error
            if len(raw) > _MAX_RPC_BYTES:
                raise CandidateExecutionError('Candidate batch public-oracle response is too large')
            payload = json.loads(raw)
            if (payload.get('schema') != CANDIDATE_ORACLE_RESPONSE_SCHEMA
                    or payload.get('request_id') != batch_id):
                raise CandidateExecutionError('Candidate batch public-oracle response is malformed')
            if status == 202 and payload.get('pending') is True:
                action = 'poll_many'
                entries = [{'token': e['token'], 'request_id': e['request_id']} for e in entries]
                if delay:
                    time.sleep(delay)
                delay = min(.05, max(.002, delay * 2))
                continue
            rows = payload.get('results', ())
            if status != 200 or [r['request_id'] for r in rows] != [e['request_id'] for e in entries]:
                raise CandidateExecutionError('Candidate batch public-oracle results are malformed')
            values = {}
            failed = False
            for client, row in zip(clients, rows):
                result = row['result']
                timing = {'schema': ORACLE_WAIT_TIMING_SCHEMA,
                    'queue_wait_seconds': result['queue_wait_seconds'] - result['lock_wait_seconds'],
                    'lock_wait_seconds': result['lock_wait_seconds'],
                    **({'group_wait_seconds': result['group_wait_seconds']}
                        if 'group_wait_seconds' in result else {})}
                _record_oracle_response_wait(row['request_id'], timing,
                    rpc_wall_seconds=time.perf_counter() - started, meter=client._solve_meter)
                client.queue_wait_seconds += result['queue_wait_seconds']
                if result.get('error'):
                    failed = True
                else:
                    values[client.case_id] = client.result_codec.decode(result['value'])
            if failed:
                raise CandidateExecutionError('Candidate batch public-oracle call failed')
            return values

    def invoke(
        self,
        case_id: str,
        method: str,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> object:
        if method not in self.methods:
            raise AttributeError(method)
        if case_id != self.case_id:
            raise PermissionError(
                "Candidate public-oracle token is scoped to another Case"
            )
        request_id = uuid4().hex
        body = json.dumps(
            {
                "schema": CANDIDATE_ORACLE_REQUEST_SCHEMA,
                "request_id": request_id,
                "params": {
                    "case_id": case_id,
                    "method": method,
                    "args": self.codec.encode(args),
                    "kwargs": self.codec.encode(dict(kwargs)),
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(body) > _MAX_RPC_BYTES:
            raise CandidateExecutionError("Candidate public-oracle request is too large")
        poll_delay = 0.0
        # The final timing covers this entire logical invocation, not just
        # its last short HTTP poll. Pending replies never debit either ledger.
        rpc_started = time.perf_counter()
        while True:
            request = Request(
                self.endpoint, data=body, method="POST",
                headers={"Content-Type": "application/json",
                         "Content-Length": str(len(body)), _TOKEN_HEADER: self.token},
            )
            try:
                with self._opener.open(request, timeout=30.0) as response:
                    response_status = response.status
                    raw = response.read(_MAX_RPC_BYTES + 1)
            except HTTPError as error:
                response_status = error.code
                with error:
                    raw = error.read(_MAX_RPC_BYTES + 1)
            except (URLError, TimeoutError, socket.timeout) as error:
                raise CandidateExecutionError("Candidate public-oracle call failed") from error
            if len(raw) > _MAX_RPC_BYTES:
                raise CandidateExecutionError("Candidate public-oracle response is too large")
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as error:
                raise CandidateExecutionError("Candidate public-oracle response is invalid") from error
            if (not isinstance(payload, Mapping)
                    or payload.get("schema") != CANDIDATE_ORACLE_RESPONSE_SCHEMA
                    or payload.get("request_id") != request_id):
                raise CandidateExecutionError("Candidate public-oracle response is malformed")
            if response_status == 202:
                if set(payload) != {"schema", "request_id", "pending"} or payload["pending"] is not True:
                    raise CandidateExecutionError("Candidate public-oracle pending response is malformed")
                body = json.dumps({"schema": CANDIDATE_ORACLE_REQUEST_SCHEMA,
                    "request_id": request_id, "action": "poll", "params": {}},
                    separators=(",", ":")).encode("utf-8")
                if poll_delay:
                    time.sleep(poll_delay)
                poll_delay = min(0.05, max(0.002, poll_delay * 2.0))
                continue
            if response_status == 200:
                if (set(payload) != {"schema", "request_id", "result", "timing"}
                        or not isinstance(payload["result"], Mapping)
                        or set(payload["result"]) != {"value"}):
                    raise CandidateExecutionError("Candidate public-oracle response is malformed")
            else:
                if (payload.get("error") != {"code": "oracle_call_failed"}
                        or set(payload) not in (
                            {"schema", "request_id", "error"},
                            {"schema", "request_id", "error", "timing"},
                        )):
                    raise CandidateExecutionError("Candidate public-oracle error response is malformed")
                # Rejected/expired requests have no completed scientific-call
                # timing and never receive speculative wait credit.
                if "timing" not in payload:
                    raise CandidateExecutionError("Candidate public-oracle call failed")
            try:
                _record_oracle_response_wait(
                    request_id, payload["timing"],
                    rpc_wall_seconds=time.perf_counter() - rpc_started,
                    meter=self._solve_meter,
                )
            except ValueError as error:
                raise CandidateExecutionInfrastructureError(
                    "Candidate public-oracle wait timing is invalid"
                ) from error
            # Compatibility reader for the upstream SDK helpers. This mirrors
            # the same validated total; no consumer adds it to the meter again.
            self.queue_wait_seconds += (
                payload["timing"]["queue_wait_seconds"]
                + payload["timing"]["lock_wait_seconds"]
            )
            if response_status != 200:
                raise CandidateExecutionError("Candidate public-oracle call failed")
            return self.result_codec.decode(payload["result"]["value"])


class _RemotePublicSession:
    def __init__(self, client: CandidatePublicSessionClient, case_id: str) -> None:
        self._client = client
        self._case_id = case_id

    def __getattr__(self, name: str):
        if name not in self._client.methods:
            raise AttributeError(name)

        def call(*args: object, **kwargs: object) -> object:
            return self._client.invoke(
                self._case_id,
                name,
                tuple(args),
                dict(kwargs),
            )

        return call

    def close(self) -> None:
        return None


def run_candidate_worker(request_path: str | Path, response_path: str | Path) -> int:
    """Execute one trusted worker request; used by ``candidate_worker``."""

    request_path = Path(request_path).resolve()
    response_path = Path(response_path).resolve()
    executing_candidate = False
    device_metadata: dict[str, object] | None = None
    try:
        payload = _read_bounded_json(request_path, _MAX_RESPONSE_BYTES)
        if payload.get("schema") != CANDIDATE_WORKER_REQUEST_SCHEMA:
            raise CandidateExecutionInfrastructureError(
                "Candidate worker request schema is invalid"
            )
        sidecar_root = payload.get("sidecar_root")
        if not isinstance(sidecar_root, str) or not sidecar_root:
            raise CandidateExecutionInfrastructureError(
                "Candidate worker request has no sidecar root"
            )
        codec = AgentWireCodec(sidecar_root, durable_sidecars=False)
        task = _task_from_payload(payload.get("task"), codec)
        settings = _settings_from_payload(
            payload.get("evaluation_settings"),
            codec,
        )
        from pde_agent_verifier_solver.sdk import resolve_evaluation_program

        evaluation = resolve_evaluation_program(task, dict(settings))
        expected_program_id = payload.get("evaluation_program_id")
        if (
            not isinstance(expected_program_id, str)
            or getattr(evaluation, "program_id", None) != expected_program_id
            or getattr(evaluation, "supports_split_candidate_execution", False)
            is not True
        ):
            raise CandidateExecutionInfrastructureError(
                "Candidate worker EvaluationProgram differs from the Controller"
            )
        artifact = _artifact_from_payload(payload.get("artifact"), codec)
        case = _case_from_payload(payload.get("case"), codec)
        plan = _plan_from_payload(payload.get("plan"), codec)
        oracle_node = payload.get("public_oracle")
        if not isinstance(oracle_node, Mapping):
            raise CandidateExecutionInfrastructureError(
                "Candidate worker public-oracle manifest is invalid"
            )
        execute = getattr(evaluation, "execute_candidate", None)
        if not callable(execute):  # pragma: no cover - capability checked above
            raise CandidateExecutionInfrastructureError(
                "Candidate worker EvaluationProgram has no executor"
            )
        members = payload.get("batch_members")
        if members is not None:
            if not isinstance(members, list) or not members:
                raise CandidateExecutionInfrastructureError("worker batch is empty")
            batch_cases = tuple(_case_from_payload(row["case"], codec) for row in members)
            batch_plans = tuple(_plan_from_payload(row["plan"], codec) for row in members)
            if len({c.case_id for c in batch_cases}) != len(batch_cases):
                raise CandidateExecutionInfrastructureError("worker batch repeats a Case")
            if getattr(evaluation, "supports_batch_candidate_execution", False) is not True:
                raise CandidateExecutionInfrastructureError("Task has no batch executor")
        execution_budget = payload.get("execution_budget")
        budget_kwargs = {}
        if execution_budget is not None:
            if (not isinstance(execution_budget, Mapping) or set(execution_budget) != {
                    "case_charged_wall_seconds"}):
                raise CandidateExecutionInfrastructureError("Candidate worker budget is malformed")
            try:
                started_at = float(os.environ["PDE_CANDIDATE_STARTED_AT_MONOTONIC"])
                budget = execution_budget["case_charged_wall_seconds"]
                if (not math.isfinite(started_at) or started_at < 0
                        or isinstance(budget, bool) or not isinstance(budget, (int, float))
                        or not math.isfinite(budget) or budget <= 0):
                    raise ValueError("invalid Candidate budget or launch clock")
            except (KeyError, TypeError, ValueError) as error:
                raise CandidateExecutionInfrastructureError("Candidate worker budget is malformed") from error
            budget_kwargs = {"budget_seconds": budget, "started_at_monotonic": started_at}
        with bind_candidate_solve_budget(new_case=True, batch_mode=members is not None, **budget_kwargs) as solve_meter:
            device_policy = payload.get("device_policy")
            if device_policy is not None:
                if (not isinstance(device_policy, Mapping) or set(device_policy) != {
                        "schema", "device", "assigned_gpu", "memory_fraction"}
                        or device_policy.get("schema") != "candidate-device-policy-v1"
                        or device_policy.get("device") != "cuda"):
                    raise CandidateExecutionInfrastructureError("Candidate GPU device policy is malformed")
                try:
                    device_metadata = initialize_cuda_device(
                        device_policy["assigned_gpu"], memory_fraction=device_policy["memory_fraction"])
                except Exception as error:
                    raise CandidateExecutionInfrastructureError(
                        "Candidate assigned GPU readiness failed; no CPU fallback") from error
            oracle = CandidatePublicSessionClient(oracle_node)
            factory = oracle.session_factory if oracle.available else None
            try:
                if members is None:
                    with bind_public_session_factory(factory, case, allowed_methods=oracle.methods):
                        executing_candidate = True
                        output = _require_candidate_output(execute(artifact, case, plan))
                else:
                    from pde_agent_verifier_solver.sdk.public_sessions import bind_public_session_batch
                    clients = {c.case_id: CandidatePublicSessionClient(row["public_oracle"])
                               for c, row in zip(batch_cases, members)}
                    factories = {key: client.session_factory if client.available else None
                                 for key, client in clients.items()}
                    with bind_public_session_batch(factories, batch_cases, allowed_methods=oracle.methods,
                            query_many=lambda selected, method, points, limit:
                                CandidatePublicSessionClient.invoke_many(
                                    [clients[c.case_id] for c in selected], method, points,
                                    max_workers=limit)):
                        executing_candidate = True
                        batch_outputs = tuple(evaluation.execute_candidate_batch(artifact, batch_cases, batch_plans))
                        output = batch_outputs[0]
            finally:
                if device_metadata is not None:
                    candidate_error = sys.exc_info()[1]
                    try:
                        synchronize_cuda_device(device_metadata)
                    except Exception as error:
                        if candidate_error is not None:
                            # Keep the original solver failure and also retain
                            # the failed GPU drain rather than replacing it.
                            _attach_execution_error(candidate_error, error)
                            candidate_error.add_note(
                                "GPU synchronization also failed: " + _bounded_diagnostic(str(error), 2048))
                        else:
                            raise CandidateExecutionInfrastructureError(
                                "Candidate GPU synchronization failed") from error
        response = {
            "schema": CANDIDATE_WORKER_RESPONSE_SCHEMA,
            "execution_id": payload.get("execution_id"),
            "case_index": payload.get("case_index"),
            "case_id": case.case_id,
            "status": "completed",
            "output": _candidate_output_to_payload(output, codec),
            "solve_budget_timing": dict(solve_meter.snapshot()),
        }
        if members is not None:
            response["batch_outputs"] = [
                {"case_index": member["index"], "case_id": c.case_id,
                 "output": _candidate_output_to_payload(result, codec)}
                for member, c, result in zip(members, batch_cases, batch_outputs)]
        if device_metadata is not None:
            response["device_execution"] = device_metadata
        _write_private_json(response_path, response, durable=False)
        return 0
    except BaseException as error:
        response = {
            "schema": CANDIDATE_WORKER_RESPONSE_SCHEMA,
            "execution_id": (
                payload.get("execution_id")
                if isinstance(locals().get("payload"), Mapping)
                else None
            ),
            "case_index": (
                payload.get("case_index")
                if isinstance(locals().get("payload"), Mapping)
                else None
            ),
            "case_id": (
                case.case_id
                if isinstance(locals().get("case"), Case)
                else None
            ),
            "status": "failed",
            "error_code": _candidate_error_code(error),
            "error_type": type(error).__name__,
            "error_message": _bounded_diagnostic(str(error)),
            "traceback": _bounded_diagnostic(traceback.format_exc()),
            "solve_budget_timing": (
                dict(solve_meter.snapshot())
                if "solve_meter" in locals()
                else None
            ),
        }
        if executing_candidate and not isinstance(error, CandidateExecutionInfrastructureError):
            diagnostic = candidate_diagnostic(error)
            locator = getattr(getattr(artifact, "program", None), "bundle_locator", None)
            if not diagnostic and isinstance(locator, Path):
                diagnostic = candidate_exception_diagnostic(
                    error, bundle_root=locator if locator.is_dir() else locator.parent,
                )
            if diagnostic:
                response["candidate_diagnostic"] = diagnostic
        if device_metadata is not None:
            response["device_execution"] = device_metadata
        try:
            _write_private_json(response_path, response)
        except BaseException:
            pass
        return 1


def _worker_request_payload(
    *,
    execution_id: str,
    case_index: int,
    task: PythonTaskWorkerSpec,
    evaluation_settings: Mapping[str, object],
    expected_program_id: str,
    artifact: Artifact,
    case: Case,
    plan: EvaluationPlan,
    oracle_manifest: Mapping[str, object],
    codec: AgentWireCodec,
) -> Mapping[str, object]:
    program = artifact.program
    assert isinstance(program, PythonCandidateProgram)
    return {
        "schema": CANDIDATE_WORKER_REQUEST_SCHEMA,
        "execution_id": execution_id,
        "case_index": case_index,
        "sidecar_root": str(codec.sidecar_root),
        "task": {
            "module": task.module,
            "qualname": task.qualname,
            "init_kwargs": codec.encode(dict(task.init_kwargs)),
        },
        "evaluation_settings": _settings_to_payload(evaluation_settings, codec),
        "evaluation_program_id": expected_program_id,
        "artifact": {
            "artifact_id": artifact.artifact_id,
            "kind": artifact.kind.value,
            "name": artifact.name,
            "summary": artifact.summary,
            "usage": artifact.usage,
            "round_index": artifact.round_index,
            "agent_id": artifact.agent_id,
            "parent_artifact_id": artifact.parent_artifact_id,
            "metadata": codec.encode(dict(artifact.metadata)),
            "bundle_locator": str(program.bundle_locator),
            "bundle_tree_sha256": program.tree_sha256,
        },
        "case": _case_to_payload(case, codec),
        "plan": _plan_to_payload(plan, codec),
        "public_oracle": dict(oracle_manifest),
    }


def _task_from_payload(node: object, codec: AgentWireCodec) -> object:
    if not isinstance(node, Mapping) or set(node) != {
        "module",
        "qualname",
        "init_kwargs",
    }:
        raise CandidateExecutionInfrastructureError("Task worker spec is invalid")
    spec = PythonTaskWorkerSpec(
        str(node["module"]),
        str(node["qualname"]),
        codec.decode(node["init_kwargs"]),
    )
    value: object = import_module(spec.module)
    for part in spec.qualname.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise CandidateExecutionInfrastructureError("Task worker target is not callable")
    return value(**dict(spec.init_kwargs))


def _artifact_from_payload(node: object, codec: AgentWireCodec) -> Artifact:
    from pde_agent_verifier_solver.sdk import TaskKind

    if not isinstance(node, Mapping):
        raise CandidateExecutionInfrastructureError("Candidate artifact payload is invalid")
    metadata = codec.decode(node.get("metadata"))
    if not isinstance(metadata, Mapping):
        raise CandidateExecutionInfrastructureError("Candidate metadata is invalid")
    program = PythonCandidateProgram(
        str(node.get("bundle_locator")),
        expected_tree_sha256=str(node.get("bundle_tree_sha256")),
    )
    return Artifact(
        artifact_id=str(node.get("artifact_id")),
        kind=TaskKind(str(node.get("kind"))),
        name=str(node.get("name")),
        program=program,
        summary=str(node.get("summary")),
        usage=str(node.get("usage")),
        round_index=int(node.get("round_index")),
        agent_id=str(node.get("agent_id")),
        parent_artifact_id=(
            None
            if node.get("parent_artifact_id") is None
            else str(node.get("parent_artifact_id"))
        ),
        metadata=dict(metadata),
    )


def _case_to_payload(case: Case, codec: AgentWireCodec) -> Mapping[str, object]:
    return {
        "case_id": case.case_id,
        "source": case.source.value,
        "visibility": case.visibility.value,
        "public_data": codec.encode(case.public_data),
        "round_index": case.round_index,
        "parent_case_id": case.parent_case_id,
        "tags": list(case.tags),
    }


def _case_from_payload(node: object, codec: AgentWireCodec) -> Case:
    if not isinstance(node, Mapping):
        raise CandidateExecutionInfrastructureError("Candidate Case payload is invalid")
    public_data = codec.decode(node.get("public_data"))
    if not isinstance(public_data, Mapping):
        raise CandidateExecutionInfrastructureError("Candidate Case public_data is invalid")
    tags = node.get("tags")
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise CandidateExecutionInfrastructureError("Candidate Case tags are invalid")
    round_index = node.get("round_index")
    if round_index is not None and (
        isinstance(round_index, bool) or not isinstance(round_index, int)
    ):
        raise CandidateExecutionInfrastructureError("Candidate Case round is invalid")
    return Case(
        case_id=str(node.get("case_id")),
        source=CaseSource(str(node.get("source"))),
        visibility=CaseVisibility(str(node.get("visibility"))),
        public_data=dict(public_data),
        round_index=round_index,
        parent_case_id=(
            None
            if node.get("parent_case_id") is None
            else str(node.get("parent_case_id"))
        ),
        tags=tuple(tags),
    )


def _query_to_payload(query: AnswerQuery, codec: AgentWireCodec) -> Mapping[str, object]:
    return {
        "kind": query.kind.value,
        "fields": list(query.fields),
        "resolution": codec.encode(dict(query.resolution)),
        "points": codec.encode(tuple(dict(point) for point in query.points)),
    }


def _query_from_payload(node: object, codec: AgentWireCodec) -> AnswerQuery:
    if not isinstance(node, Mapping):
        raise CandidateExecutionInfrastructureError("Candidate query payload is invalid")
    fields = node.get("fields")
    resolution = codec.decode(node.get("resolution"))
    points = codec.decode(node.get("points"))
    if (
        not isinstance(fields, list)
        or any(not isinstance(field, str) for field in fields)
        or not isinstance(resolution, Mapping)
        or not isinstance(points, tuple)
    ):
        raise CandidateExecutionInfrastructureError("Candidate query payload is invalid")
    return AnswerQuery(
        kind=QueryKind(str(node.get("kind"))),
        fields=tuple(fields),
        resolution=dict(resolution),
        points=tuple(dict(point) for point in points),
    )


def _settings_to_payload(
    settings: Mapping[str, object],
    codec: AgentWireCodec,
) -> list[Mapping[str, object]]:
    """Encode Task settings while preserving typed AnswerQuery values."""

    rows: list[Mapping[str, object]] = []
    for name, value in settings.items():
        if not isinstance(name, str):
            raise CandidateExecutionInfrastructureError(
                "Candidate evaluation setting names must be strings"
            )
        if isinstance(value, AnswerQuery):
            rows.append(
                {
                    "name": name,
                    "kind": "answer_query",
                    "value": _query_to_payload(value, codec),
                }
            )
        else:
            rows.append(
                {
                    "name": name,
                    "kind": "wire",
                    "value": codec.encode(value),
                }
            )
    return rows


def _settings_from_payload(
    node: object,
    codec: AgentWireCodec,
) -> Mapping[str, object]:
    if not isinstance(node, list):
        raise CandidateExecutionInfrastructureError(
            "Candidate evaluation settings are invalid"
        )
    settings: dict[str, object] = {}
    for row in node:
        if not isinstance(row, Mapping) or set(row) != {"name", "kind", "value"}:
            raise CandidateExecutionInfrastructureError(
                "Candidate evaluation setting row is invalid"
            )
        name = row.get("name")
        kind = row.get("kind")
        if not isinstance(name, str) or not name or name in settings:
            raise CandidateExecutionInfrastructureError(
                "Candidate evaluation setting name is invalid"
            )
        if kind == "answer_query":
            settings[name] = _query_from_payload(row.get("value"), codec)
        elif kind == "wire":
            settings[name] = codec.decode(row.get("value"))
        else:
            raise CandidateExecutionInfrastructureError(
                "Candidate evaluation setting kind is invalid"
            )
    return settings


def _plan_to_payload(plan: EvaluationPlan, codec: AgentWireCodec) -> Mapping[str, object]:
    return {
        "phase": plan.phase.value,
        "query": _query_to_payload(plan.query, codec),
        "seed": plan.seed,
        "settings": _settings_to_payload(plan.settings, codec),
    }


def _plan_from_payload(node: object, codec: AgentWireCodec) -> EvaluationPlan:
    if not isinstance(node, Mapping):
        raise CandidateExecutionInfrastructureError("Candidate plan payload is invalid")
    settings = _settings_from_payload(node.get("settings"), codec)
    return EvaluationPlan(
        phase=EvaluationPhase(str(node.get("phase"))),
        query=_query_from_payload(node.get("query"), codec),
        seed=int(node.get("seed")),
        settings=dict(settings),
    )


def _answer_to_payload(answer: AnswerView, codec: AgentWireCodec) -> Mapping[str, object]:
    return {
        "kind": answer.kind.value,
        "fields": codec.encode(
            {name: tuple(values) for name, values in answer.fields.items()}
        ),
        "axes": codec.encode(
            {name: tuple(values) for name, values in answer.axes.items()}
        ),
        "points": codec.encode(tuple(dict(point) for point in answer.points)),
        "shape": list(answer.shape),
    }


def _answer_from_payload(node: object, codec: AgentWireCodec) -> AnswerView:
    if not isinstance(node, Mapping):
        raise CandidateExecutionInfrastructureError("Candidate AnswerView is invalid")
    fields = codec.decode(node.get("fields"))
    axes = codec.decode(node.get("axes"))
    points = codec.decode(node.get("points"))
    shape = node.get("shape")
    if (
        not isinstance(fields, Mapping)
        or not isinstance(axes, Mapping)
        or not isinstance(points, tuple)
        or not isinstance(shape, list)
    ):
        raise CandidateExecutionInfrastructureError("Candidate AnswerView is invalid")
    return AnswerView(
        kind=QueryKind(str(node.get("kind"))),
        fields={name: tuple(values) for name, values in fields.items()},
        axes={name: tuple(values) for name, values in axes.items()},
        points=tuple(dict(point) for point in points),
        shape=tuple(int(value) for value in shape),
    )


def _candidate_output_to_payload(
    output: CandidateEvaluationOutput,
    codec: AgentWireCodec,
) -> Mapping[str, object]:
    return {
        "prediction": _answer_to_payload(output.prediction, codec),
        "metrics": codec.encode(dict(output.metrics)),
    }


def _candidate_output_from_payload(
    node: object,
    codec: AgentWireCodec,
) -> CandidateEvaluationOutput:
    if not isinstance(node, Mapping) or set(node) != {"prediction", "metrics"}:
        raise CandidateExecutionInfrastructureError("Candidate output is invalid")
    metrics = codec.decode(node["metrics"])
    if not isinstance(metrics, Mapping):
        raise CandidateExecutionInfrastructureError("Candidate output metrics are invalid")
    return _require_candidate_output(
        CandidateEvaluationOutput(
            prediction=_answer_from_payload(node["prediction"], codec),
            metrics={str(name): float(value) for name, value in metrics.items()},
        )
    )


def _worker_output_from_payload(
    payload: Mapping[str, object],
    *,
    expected_execution_id: str,
    expected_case_index: int,
    expected_case_id: str,
    codec: AgentWireCodec,
) -> CandidateEvaluationOutput:
    if (
        payload.get("schema") != CANDIDATE_WORKER_RESPONSE_SCHEMA
        or payload.get("execution_id") != expected_execution_id
        or payload.get("case_index") != expected_case_index
        or payload.get("case_id") != expected_case_id
        or payload.get("status") != "completed"
    ):
        raise CandidateExecutionInfrastructureError(
            "Candidate worker response identity is invalid"
        )
    return _candidate_output_from_payload(payload.get("output"), codec)


def _failed_worker_code(
    response_path: Path,
    *,
    execution_id: str,
    case_index: int,
    case_id: str,
    diagnostic: dict | None = None,
) -> str:
    """Validate failure identity and retain candidate-owned debugging evidence."""

    try:
        payload = _read_bounded_json(response_path, _MAX_RESPONSE_BYTES)
    except CandidateExecutionInfrastructureError:
        return "candidate_worker_failed_without_response"
    if (
        payload.get("schema") != CANDIDATE_WORKER_RESPONSE_SCHEMA
        or payload.get("execution_id") != execution_id
        or payload.get("case_index") != case_index
        or payload.get("case_id") != case_id
        or payload.get("status") != "failed"
    ):
        return "candidate_worker_failed_with_invalid_response"
    code = payload.get("error_code")
    if not isinstance(code, str) or not code:
        return "candidate_worker_failed_with_invalid_response"
    if diagnostic is not None:
        diagnostic.update(normalize_candidate_diagnostic(payload.get("candidate_diagnostic")))
    return code


def _validated_cases_and_plans(
    cases: Sequence[Case],
    plans: Sequence[EvaluationPlan],
) -> tuple[tuple[Case, ...], tuple[EvaluationPlan, ...]]:
    prepared_cases = tuple(cases)
    prepared_plans = tuple(plans)
    if not prepared_cases or len(prepared_cases) != len(prepared_plans):
        raise ValueError("Candidate Cases and plans must be nonempty and aligned")
    if any(not isinstance(case, Case) for case in prepared_cases):
        raise TypeError("Candidate execution accepts only Case values")
    if any(not isinstance(plan, EvaluationPlan) for plan in prepared_plans):
        raise TypeError("Candidate execution accepts only EvaluationPlan values")
    return prepared_cases, prepared_plans


def _validate_cancellation_event(value: Event | None) -> None:
    if value is not None and not callable(getattr(value, "is_set", None)):
        raise TypeError("cancellation_event must expose is_set()")


def _validated_development_batch_deadline(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("development_batch_deadline must be a number")
    deadline = float(value)
    if not math.isfinite(deadline) or deadline <= 0.0:
        raise ValueError("development_batch_deadline must be positive and finite")
    return deadline


def _validated_oracle_drain_timeout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("oracle drain timeout must be a number")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout < 0.0:
        raise ValueError("oracle drain timeout must be non-negative and finite")
    return timeout


def _caller_cancelled(value: Event | None) -> bool:
    if value is None:
        return False
    cancelled = value.is_set()
    if not isinstance(cancelled, bool):
        raise TypeError("cancellation_event.is_set() must return bool")
    return cancelled


def _raise_if_candidate_execution_cancelled(
    batch_cancel: Event,
    cancellation_event: Event | None,
    batch_deadline: float | None,
) -> None:
    """Fail one batch without revoking processes owned by another batch."""

    if _caller_cancelled(cancellation_event):
        batch_cancel.set()
        raise CandidateExecutionCancelled(
            "Candidate evaluation was cancelled by its Agent turn"
        )
    if batch_deadline is not None and time.monotonic() >= batch_deadline:
        batch_cancel.set()
        raise CandidateExecutionBatchTimeout(
            "Development Candidate evaluation batch exceeded its hard wall"
        )
    if batch_cancel.is_set():
        raise _CandidatePeerCancelled(
            "Candidate evaluation batch was cancelled after a peer failure"
        )


def _validated_methods(methods: Sequence[str]) -> tuple[str, ...]:
    if isinstance(methods, (str, bytes)):
        raise TypeError("public Candidate methods must be a sequence")
    values = tuple(methods)
    if any(
        not isinstance(name, str)
        or not name.isidentifier()
        or name.startswith("_")
        for name in values
    ):
        raise ValueError("public Candidate methods must be public identifiers")
    if len(values) != len(set(values)):
        raise ValueError("public Candidate methods cannot repeat")
    return values


def _require_candidate_output(value: object) -> CandidateEvaluationOutput:
    if not isinstance(value, CandidateEvaluationOutput):
        raise TypeError(
            "split EvaluationProgram must return CandidateEvaluationOutput"
        )
    if not isinstance(value.prediction, AnswerView):
        raise TypeError("Candidate output prediction must be an AnswerView")
    for name, metric in value.metrics.items():
        if (
            not isinstance(name, str)
            or isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(float(metric))
        ):
            raise ValueError("Candidate output metrics must be finite numbers")
    return value


def _validate_oracle_transport(inline_array_bytes, trusted_results) -> None:
    # Keep a single raw array well below the bounded JSON/base64 RPC envelope.
    # A request/response containing several arrays retains the total RPC limit.
    if (isinstance(inline_array_bytes, bool) or not isinstance(inline_array_bytes, int)
            or not 1 <= inline_array_bytes <= _MAX_RPC_BYTES // 2):
        raise ValueError("oracle_inline_array_bytes must be an integer in [1, 33554432]")
    if type(trusted_results) is not bool:
        raise TypeError("oracle_trusted_results must be boolean")


def _candidate_oracle_handler():
    class Handler(BaseHTTPRequestHandler):
        server: _CandidateOracleHTTPServer

        def do_POST(self) -> None:
            self._request_started = time.perf_counter()
            self._action = None
            request_id: str | None = None
            if self.path != _ORACLE_PATH:
                self._send_error(404, request_id)
                return
            token = self.headers.get(_TOKEN_HEADER, "")
            if not token:
                self._send_error(403, request_id)
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._send_error(400, request_id)
                return
            if length <= 0 or length > _MAX_RPC_BYTES:
                self._send_error(413, request_id)
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if (
                    not isinstance(payload, Mapping)
                    or payload.get("schema") != CANDIDATE_ORACLE_REQUEST_SCHEMA
                    or not isinstance(payload.get("request_id"), str)
                    or not isinstance(payload.get("params"), Mapping)
                ):
                    raise ValueError
                request_id = payload["request_id"]
                action = payload.get("action", "submit")
                self._action = action
                if action in ('submit_many', 'poll_many'):
                    entries = payload['params'].get('requests')
                    if (not isinstance(entries, list) or not 1 <= len(entries) <= 128
                            or any(not isinstance(e, dict) or not isinstance(e.get('token'), str)
                                   or not isinstance(e.get('request_id'), str) for e in entries)
                            or len({e['request_id'] for e in entries}) != len(entries)
                            or token != entries[0]['token']):
                        raise ValueError('Oracle batch envelope is invalid')
                    if action == 'submit_many':
                        self.server.state.submit_many(entries)
                    rows = self.server.state.poll_many(entries)
                    self._send_json(202 if rows is None else 200, {
                        'schema': CANDIDATE_ORACLE_RESPONSE_SCHEMA, 'request_id': request_id,
                        **({'pending': True} if rows is None else {'results': rows})})
                    return
                if action == "submit":
                    self.server.state.submit(token, request_id, payload["params"])
                elif action != "poll":
                    raise ValueError("Candidate oracle action is invalid")
                result = self.server.state.poll(token, request_id)
            except _OracleRequestFailure as error:
                self._send_json(500, {
                    "schema": CANDIDATE_ORACLE_RESPONSE_SCHEMA,
                    "request_id": request_id,
                    "error": {"code": "oracle_call_failed"},
                    "timing": {
                        "schema": ORACLE_WAIT_TIMING_SCHEMA,
                        "queue_wait_seconds": error.queue_wait_seconds - error.lock_wait_seconds,
                        "lock_wait_seconds": error.lock_wait_seconds,
                    },
                })
                return
            except PermissionError:
                self._send_error(403, request_id)
                return
            except Exception:
                self._send_error(500, request_id)
                return
            if result is None:
                self._send_json(202, {"schema": CANDIDATE_ORACLE_RESPONSE_SCHEMA,
                                     "request_id": request_id, "pending": True})
                return
            self._send_json(
                200,
                {
                    "schema": CANDIDATE_ORACLE_RESPONSE_SCHEMA,
                    "request_id": request_id,
                    "result": {"value": result["value"]},
                    "timing": {
                        "schema": ORACLE_WAIT_TIMING_SCHEMA,
                        "queue_wait_seconds": result["queue_wait_seconds"] - result["lock_wait_seconds"],
                        "lock_wait_seconds": result["lock_wait_seconds"],
                        **({"group_wait_seconds": result["group_wait_seconds"]}
                           if "group_wait_seconds" in result else {}),
                    },
                },
            )

        def log_message(self, format: str, *args: object) -> None:
            del format, args

        def _send_error(self, status: int, request_id: str | None) -> None:
            self._send_json(
                status,
                {
                    "schema": CANDIDATE_ORACLE_RESPONSE_SCHEMA,
                    "request_id": request_id,
                    "error": {"code": "oracle_call_failed"},
                },
            )

        def _send_json(self, status: int, payload: Mapping[str, object]) -> None:
            encoding_started = time.perf_counter()
            body = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            if len(body) > _MAX_RPC_BYTES:
                self._send_error(413, payload.get("request_id"))
                return
            encoding_seconds = time.perf_counter() - encoding_started
            send_started = time.perf_counter()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            trace = self.server.state._trace_call
            if trace is not None:
                trace(event='oracle_transport', action=self._action, status=status,
                    response_bytes=len(body), json_encode_seconds=encoding_seconds,
                    send_seconds=time.perf_counter()-send_started,
                    handler_seconds=time.perf_counter()-self._request_started)

    return Handler


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        _cleanup_process_group(process.pid)
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired as error:
            raise CandidateExecutionInfrastructureError(
                "Candidate process group could not be terminated"
            ) from error


def _cleanup_process_group(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError as error:
        raise CandidateExecutionInfrastructureError(
            "Candidate process group ownership could not be confirmed"
        ) from error
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.01)
    return False


def _candidate_call_workspace(retained_root: Path) -> Path:
    """Use private node-local storage when the scheduler supplies it."""
    scratch = os.environ.get("SLURM_TMPDIR")
    if scratch:
        scratch_root = Path(scratch).expanduser().resolve()
        if scratch_root.is_dir() and os.access(scratch_root, os.W_OK | os.X_OK):
            return Path(tempfile.mkdtemp(
                prefix=f"pde-candidate-{retained_root.name}-", dir=scratch_root,
            ))
    retained_root.mkdir(mode=0o700)
    return retained_root


def _retain_candidate_workspace(source: Path, destination: Path) -> None:
    """Persist failure evidence without copying or syncing successful scratch."""
    if source != destination:
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copytree(source, destination, symlinks=True)
    # The transport normally requires visibility only, not crash durability.
    # Failed requests and their sidecars become persistent evidence here.
    for directory, _subdirectories, filenames in os.walk(destination, followlinks=False):
        for filename in filenames:
            path = Path(directory) / filename
            if not stat.S_ISREG(path.lstat().st_mode):
                continue
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write_private_json(
    path: Path, payload: Mapping[str, object], *, durable: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        if durable:
            os.fsync(handle.fileno())


def _read_bounded_json(path: Path, maximum_bytes: int) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise CandidateExecutionInfrastructureError(
            "Candidate worker response is missing or unsafe"
        )
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise CandidateExecutionInfrastructureError(
            "Candidate worker response size is invalid"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CandidateExecutionInfrastructureError(
            "Candidate worker response is not valid JSON"
        ) from error
    if not isinstance(value, Mapping):
        raise CandidateExecutionInfrastructureError(
            "Candidate worker response must be an object"
        )
    return value


def _candidate_error_code(error: BaseException) -> str:
    if isinstance(error, _CandidatePeerCancelled):
        return "candidate_peer_cancelled"
    if isinstance(error, CandidateExecutionBatchTimeout):
        return "candidate_development_batch_timeout"
    if isinstance(error, CandidateExecutionCancelled):
        return "candidate_execution_cancelled"
    if isinstance(error, CandidateExecutionTimeout):
        return "candidate_case_timeout"
    if isinstance(error, CandidateExecutionInfrastructureError):
        return "candidate_execution_infrastructure_failure"
    return "candidate_execution_failed"


def _oracle_thread_stacks() -> Mapping[str, str]:
    """Failure-only Python stacks; no sampling or I/O on successful queries."""
    frames = sys._current_frames()
    return {
        f"{thread.name}:{thread.ident}": _bounded_diagnostic(
            "".join(traceback.format_stack(frames[thread.ident], limit=25)))
        for thread in enumerate_threads()
        if thread.name.startswith("candidate-public-oracle") and thread.ident in frames
    }


def _bounded_diagnostic(value: str, maximum_characters: int = 32_768) -> str:
    if len(value) <= maximum_characters:
        return value
    return value[:maximum_characters] + "\n[diagnostic truncated]"


def _attach_execution_error(
    primary: BaseException,
    additional: BaseException,
) -> None:
    errors = tuple(getattr(primary, "_candidate_cleanup_errors", ())) + (
        additional,
    )
    try:
        setattr(primary, "_candidate_cleanup_errors", errors)
    except BaseException:  # pragma: no cover - unusual immutable exception
        pass
    add_note = getattr(primary, "add_note", None)
    if callable(add_note):
        label = ("Candidate peer cancellation: " if isinstance(additional, _CandidatePeerCancelled)
                 else "Candidate execution cleanup also failed: ")
        add_note(label + f"{type(additional).__name__}: {additional}")


__all__ = [
    "CandidateExecutionAudit",
    "CandidateExecutionBatchTimeout",
    "CandidateExecutionCancelled",
    "CandidateExecutionError",
    "CandidateExecutionInfrastructureError",
    "CandidateExecutionService",
    "CandidateExecutionTimeout",
    "InProcessCandidateExecutionService",
    "PythonTaskWorkerSpec",
    "SubprocessCandidateExecutionService",
]
