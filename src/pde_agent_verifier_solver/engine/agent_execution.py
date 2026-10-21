"""Execution boundary for one Agent turn in the canonical Controller.

The in-process adapter remains the development default. The subprocess adapter
isolates the *Agent turn* in its own workspace and process group and exchanges
versioned JSON files. It deliberately does not claim that later Candidate
execution is sandboxed: nomination decoding and Candidate evaluation still
happen in the Controller process.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
from threading import Condition, Lock
import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Mapping, Protocol, Sequence
from uuid import uuid4

from pde_agent_verifier_solver.sdk.evaluation_devices import EvaluationDevicePool

if TYPE_CHECKING:
    from pde_agent_verifier_solver.sdk import Artifact

    from .runtime import AgentContext, AgentOutcome, AgentProgram


AGENT_WORKER_REQUEST_SCHEMA = "canonical-agent-worker-request-v2"
AGENT_WORKER_RESPONSE_SCHEMA = "canonical-agent-worker-response-v1"
AGENT_NOMINATION_SCHEMA = "canonical-agent-nomination-v1"
AGENT_PRIMARY_NOMINATION_SCHEMA = "canonical-agent-primary-nomination-v1"
AGENT_NOMINATION_POLICIES = ("all", "latest", "single_primary_v1")
AGENT_CAPABILITY_SCHEMA = "canonical-agent-capabilities-v1"
AGENT_EXECUTION_AUDIT_SCHEMA = "canonical-agent-execution-audit-v1"

AGENT_WORKSPACE_ENV = "PDE_AGENT_WORKSPACE"
AGENT_REQUEST_ENV = "PDE_AGENT_REQUEST"
AGENT_RESPONSE_ENV = "PDE_AGENT_RESPONSE"
AGENT_NOMINATIONS_ENV = "PDE_AGENT_NOMINATIONS"
AGENT_EXECUTION_ID_ENV = "PDE_AGENT_EXECUTION_ID"
AGENT_CAMPAIGN_MEMORY_ENV = "PDE_AGENT_CAMPAIGN_MEMORY"
AGENT_CANDIDATE_HISTORY_ENV = "PDE_AGENT_CANDIDATE_HISTORY"
AGENT_ROUND_LOG_ENV = "PDE_AGENT_ROUND_LOG"
AGENT_ROUND_LOG_MAX_BYTES_ENV = "PDE_AGENT_ROUND_LOG_MAX_BYTES"
AGENT_PROTOCOL_MAX_BYTES_ENV = "PDE_AGENT_PROTOCOL_MAX_BYTES"
AGENT_DEADLINE_MONOTONIC_ENV = "PDE_AGENT_DEADLINE_MONOTONIC"

_RESERVED_ENVIRONMENT = frozenset(
    {
        AGENT_WORKSPACE_ENV,
        AGENT_REQUEST_ENV,
        AGENT_RESPONSE_ENV,
        AGENT_NOMINATIONS_ENV,
        AGENT_EXECUTION_ID_ENV,
        AGENT_CAMPAIGN_MEMORY_ENV,
        AGENT_CANDIDATE_HISTORY_ENV,
        AGENT_ROUND_LOG_ENV,
        AGENT_ROUND_LOG_MAX_BYTES_ENV,
        AGENT_PROTOCOL_MAX_BYTES_ENV,
        AGENT_DEADLINE_MONOTONIC_ENV,
        "HOME",
        "TMPDIR",
        "TMP",
        "TEMP",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPYCACHEPREFIX",
        "CUDA_VISIBLE_DEVICES",
    }
)
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")
_INHERITED_ENVIRONMENT = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CPATH",
    "VIRTUAL_ENV",
)


class AgentExecutionService(Protocol):
    """Execute one Agent turn with the Controller's context and time budget."""

    def execute(
        self,
        program: AgentProgram,
        context: AgentContext,
        active_budget_seconds: float,
    ) -> AgentOutcome: ...


class BatchAgentExecutionService(AgentExecutionService, Protocol):
    """Ordered batch extension with an explicit resource reservation lease."""

    def validate_batch(
        self,
        requests: Sequence[AgentExecutionRequest],
    ) -> None: ...

    def execute_many(
        self,
        requests: Sequence[AgentExecutionRequest],
    ) -> tuple[AgentExecutionResult, ...]: ...

    def reserve_batch(
        self,
        requests: Sequence[AgentExecutionRequest],
    ) -> AgentBatchReservation: ...

    def execute_reserved_many(
        self,
        reservation: AgentBatchReservation,
    ) -> tuple[AgentExecutionResult, ...]: ...


class AgentExecutionError(RuntimeError):
    """The isolated Agent turn could not produce a valid result."""


class AgentExecutionTimeout(AgentExecutionError):
    """The isolated Agent turn exhausted its active or hard-wall budget."""


class AgentInfrastructureError(AgentExecutionError):
    """Execution isolation or capability cleanup could not be confirmed."""


class AgentGPUConflict(AgentExecutionError):
    """A same-round GPU assignment overlaps another Agent or the Controller."""


@dataclass(frozen=True)
class AgentExecutionRequest:
    """One ordered Agent turn submitted through the batch execution port.

    ``active_budget_seconds`` keeps its serialized caller-facing name for
    compatibility; the default subprocess service treats it as elapsed time.
    """

    program: AgentProgram
    context: AgentContext
    active_budget_seconds: float | None
    continuation: dict | None = field(default=None, compare=False, repr=False)
    checkpoint: Callable[[], None] | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.active_budget_seconds is None:
            return
        object.__setattr__(
            self,
            "active_budget_seconds",
            _positive_seconds(self.active_budget_seconds, "active_budget_seconds"),
        )


@dataclass(frozen=True)
class AgentExecutionResult:
    """One raw outcome or isolated failure, paired with its ordered request."""

    request: AgentExecutionRequest
    outcome: object | None = None
    error: BaseException | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, AgentExecutionRequest):
            raise TypeError("Agent execution result needs its original request")
        if self.error is not None and not isinstance(self.error, BaseException):
            raise TypeError("Agent execution result error must be an exception")
        if self.error is not None and self.outcome is not None:
            raise ValueError("failed Agent execution cannot also contain an outcome")

    @property
    def succeeded(self) -> bool:
        return self.error is None


class AgentBatchReservation:
    """One live preflight lease spanning Agent start through worker cleanup."""

    def __init__(
        self,
        service: AgentExecutionService,
        requests: Sequence[AgentExecutionRequest],
        *,
        release: Callable[[], None] | None = None,
    ) -> None:
        self._service = service
        self._requests = _ordered_execution_requests(requests)
        self._release = release or (lambda: None)
        self._lock = Lock()
        self._active = True

    def requests_for(
        self,
        service: AgentExecutionService,
    ) -> tuple[AgentExecutionRequest, ...]:
        with self._lock:
            if not self._active:
                raise RuntimeError("Agent batch reservation is no longer active")
            if service is not self._service:
                raise ValueError("Agent batch reservation belongs to another service")
            return self._requests

    def close(self) -> None:
        with self._lock:
            if not self._active:
                return
            try:
                self._release()
            finally:
                self._active = False


def reserve_agent_batch(
    service: AgentExecutionService,
    requests: Sequence[AgentExecutionRequest],
) -> AgentBatchReservation:
    prepared = _ordered_execution_requests(requests)
    reserve = getattr(service, "reserve_batch", None)
    if callable(reserve):
        reservation = reserve(prepared)
        if not isinstance(reservation, AgentBatchReservation):
            raise TypeError("reserve_batch must return an Agent batch reservation")
        reservation.requests_for(service)
        return reservation
    validate_agent_batch(service, prepared)
    return AgentBatchReservation(service, prepared)


def validate_agent_batch(
    service: AgentExecutionService,
    requests: Sequence[AgentExecutionRequest],
) -> tuple[AgentExecutionRequest, ...]:
    """Run an optional service preflight without starting any Agent turn."""

    prepared = _ordered_execution_requests(requests)
    validate = getattr(service, "validate_batch", None)
    if callable(validate):
        validate(prepared)
    return prepared


def execute_agent_batch(
    service: AgentExecutionService,
    requests: Sequence[AgentExecutionRequest],
) -> tuple[AgentExecutionResult, ...]:
    """Preflight and execute an ordered batch exactly once."""

    reservation = reserve_agent_batch(service, requests)
    try:
        results = execute_reserved_agent_batch(service, reservation)
    finally:
        reservation.close()
    _raise_batch_fatal_errors(results)
    return results


def execute_reserved_agent_batch(
    service: AgentExecutionService,
    reservation: AgentBatchReservation,
) -> tuple[AgentExecutionResult, ...]:
    """Execute a live batch reservation without repeating preflight."""

    prepared = reservation.requests_for(service)
    execute_validated_many = getattr(service, "execute_reserved_many", None)
    if callable(execute_validated_many):
        results = tuple(execute_validated_many(reservation))
    else:
        # A custom service written against the original execute-only port keeps
        # working. Its turns are sequential, but one failure still cannot skip
        # the remaining requests.
        results = tuple(_capture_execution(service, request) for request in prepared)
    if len(results) != len(prepared) or any(
        not isinstance(result, AgentExecutionResult)
        for result in results
    ):
        raise TypeError(
            "Agent execute_many must return one AgentExecutionResult per request"
        )
    for request, result in zip(prepared, results):
        if result.request is not request:
            raise ValueError("Agent execute_many changed ordered request identity")
    return results


class NominationDecoder(Protocol):
    """Trusted Controller adapter from a JSON nomination to a Candidate."""

    def __call__(
        self,
        payload: Mapping[str, object],
        context: AgentContext,
        workspace: Path,
    ) -> Artifact: ...


class IncumbentEncoder(Protocol):
    """Trusted adapter that may stage a selected program for the worker."""

    def __call__(
        self,
        incumbent: Artifact,
        workspace: Path,
    ) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class CommandAgentProgram:
    """An Agent implemented as a direct executable command.

    ``nomination_decoder`` and ``incumbent_encoder`` run in the trusted
    Controller process. They are codec adapters, not untrusted Agent code.
    Commands are argv sequences and are never interpreted by a shell.
    ``worker_settings["nomination_policy"]`` defaults to ``"all"``. Opting into
    ``"latest"`` retains the submission journal but returns only the final atomic
    commitment for Official evaluation; it does not limit development attempts.
    ``"single_primary_v1"`` instead evaluates only the explicitly selected saved
    nomination. Later ordinary submissions cannot replace that selection.
    """

    command: tuple[str, ...]
    nomination_decoder: NominationDecoder
    environment: Mapping[str, str] = field(default_factory=dict)
    worker_settings: Mapping[str, object] = field(default_factory=dict)
    incumbent_encoder: IncumbentEncoder | None = None
    gpu_devices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        command = tuple(self.command)
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise ValueError("Agent worker command must contain nonempty argv strings")
        if not callable(self.nomination_decoder):
            raise TypeError("nomination_decoder must be callable")
        environment = dict(self.environment)
        if any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or "=" in key
            or "\x00" in key
            or "\x00" in value
            for key, value in environment.items()
        ):
            raise ValueError("Agent worker environment must contain valid strings")
        reserved = _RESERVED_ENVIRONMENT.intersection(environment)
        if reserved:
            raise ValueError(
                "Agent worker environment cannot override managed values: "
                f"{sorted(reserved)}"
            )
        gpu_devices = _validated_gpu_devices(self.gpu_devices, "gpu_devices")
        if self.worker_settings.get("nomination_policy", "all") not in AGENT_NOMINATION_POLICIES:
            raise ValueError("worker_settings.nomination_policy must be all, latest or single_primary_v1")
        _json_bytes(dict(self.worker_settings))
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "environment", MappingProxyType(environment))
        object.__setattr__(
            self,
            "worker_settings",
            MappingProxyType(dict(self.worker_settings)),
        )
        object.__setattr__(self, "gpu_devices", gpu_devices)

    def run(self, context: AgentContext) -> AgentOutcome:
        del context
        raise RuntimeError(
            "CommandAgentProgram requires SubprocessAgentExecutionService"
        )


class AgentCapabilityLease:
    """One revocable, execution-scoped Agent capability endpoint."""

    def __init__(
        self,
        manifest: Mapping[str, object],
        *,
        revoke: Callable[[], None] | None = None,
        drain: Callable[[], None] | None = None,
        service_seconds: Callable[[], float] | None = None,
    ) -> None:
        _json_bytes(dict(manifest))
        self._manifest = MappingProxyType(dict(manifest))
        self._revoke = revoke or (lambda: None)
        self._drain = drain or (lambda: None)
        self._service_seconds = service_seconds or (lambda: 0.0)
        self._revoked = False
        self._revoking = False
        self._drained = False
        self._draining = False
        self._lifecycle = Condition(Lock())

    @property
    def manifest(self) -> Mapping[str, object]:
        return self._manifest

    @property
    def controller_service_seconds(self) -> float:
        value = float(self._service_seconds())
        if not math.isfinite(value) or value < 0:
            raise AgentExecutionError(
                "capability facade reported invalid controller service time"
            )
        return value

    @property
    def revoked(self) -> bool:
        with self._lifecycle:
            return self._revoked

    @property
    def drained(self) -> bool:
        with self._lifecycle:
            return self._drained

    def revoke(self) -> None:
        with self._lifecycle:
            while self._revoking:
                self._lifecycle.wait()
            if self._revoked:
                return
            self._revoking = True
        try:
            self._revoke()
        except BaseException:
            with self._lifecycle:
                self._revoking = False
                self._lifecycle.notify_all()
            raise
        with self._lifecycle:
            self._revoked = True
            self._revoking = False
            self._lifecycle.notify_all()

    def close(self) -> None:
        with self._lifecycle:
            if self._drained:
                return

        revoke_error: BaseException | None = None
        try:
            self.revoke()
        except BaseException as exc:
            # Draining is a separate containment boundary.  A broken revoke
            # callback must not leave the listener or accepted calls alive.
            revoke_error = exc
        with self._lifecycle:
            while self._draining:
                self._lifecycle.wait()
            if self._drained:
                if revoke_error is not None:
                    raise revoke_error
                return
            self._draining = True
        try:
            self._drain()
        except BaseException as drain_error:
            with self._lifecycle:
                self._draining = False
                self._lifecycle.notify_all()
            if revoke_error is not None:
                revoke_error.add_note(
                    "capability drain also failed: "
                    f"{type(drain_error).__name__}: {drain_error}"
                )
                raise revoke_error from drain_error
            raise
        with self._lifecycle:
            self._drained = True
            self._draining = False
            self._lifecycle.notify_all()
        if revoke_error is not None:
            raise revoke_error


class AgentCapabilityFacade(Protocol):
    """Open the only IPC capabilities granted to one Agent turn.

    The formal facade exposes sampler/evaluator RPC and accounts Controller-side
    service time. The subprocess service never receives the live AgentContext.
    """

    def open(
        self,
        context: AgentContext,
        workspace: Path,
        execution_id: str,
        program: CommandAgentProgram,
    ) -> AgentCapabilityLease: ...


class UnavailableAgentCapabilityFacade:
    """Explicitly disable sampler/evaluator RPC for tests or restricted turns."""

    def open(
        self,
        context: AgentContext,
        workspace: Path,
        execution_id: str,
        program: CommandAgentProgram,
    ) -> AgentCapabilityLease:
        del context, workspace, execution_id, program
        return AgentCapabilityLease(
            {
                "schema": AGENT_CAPABILITY_SCHEMA,
                "sampling": {"available": False},
                "evaluation": {"available": False},
                "public_problem": {"available": False},
            }
        )


@dataclass(frozen=True)
class AgentExecutionAudit:
    schema: str
    execution_id: str
    round_index: int
    agent_id: str
    status: str
    pid: int | None
    return_code: int | None
    started_at_unix: float
    finished_at_unix: float
    wall_seconds: float
    controller_service_seconds: float
    active_seconds: float
    active_budget_seconds: float | None
    hard_wall_seconds: float | None
    workspace: str
    stdout_path: str
    stderr_path: str
    request_path: str
    response_path: str
    nomination_count: int
    rejected_nomination_count: int
    capability_revoked: bool
    capability_drained: bool
    process_group_cleaned: bool
    error: str | None
    agent_process_isolated: bool = True
    candidate_execution_isolated: bool = False
    formal_private_scoring_claimed: bool = False
    time_budget_mode: str = "hard_wall"
    # Preserve wall_seconds as the historical launch-through-cleanup duration.
    # Only the monitored Agent phase is compared with its configured deadline.
    agent_execution_wall_seconds: float = 0.0
    cleanup_wall_seconds: float = 0.0
    nomination_policy: str = "all"
    # Count raw committed journal entries separately from returned nominees.
    # None preserves the distinction between legacy/unreadable evidence and zero.
    nomination_journal_entry_count: int | None = None
    final_nomination_sequence: int | None = None

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)


def _ordered_execution_requests(
    requests: Sequence[AgentExecutionRequest],
) -> tuple[AgentExecutionRequest, ...]:
    if isinstance(requests, (str, bytes)):
        raise TypeError("Agent execution requests must be a sequence")
    prepared = tuple(requests)
    if any(not isinstance(request, AgentExecutionRequest) for request in prepared):
        raise TypeError(
            "Agent execution requests must contain AgentExecutionRequest values"
        )
    return prepared


def _capture_execution(
    service: AgentExecutionService,
    request: AgentExecutionRequest,
) -> AgentExecutionResult:
    try:
        outcome = service.execute(
            request.program,
            request.context,
            request.active_budget_seconds,
        )
    except BaseException as exc:
        return AgentExecutionResult(request=request, error=exc)
    return AgentExecutionResult(request=request, outcome=outcome)


def _capture_future(
    request: AgentExecutionRequest,
    future: Future,
) -> AgentExecutionResult:
    try:
        outcome = future.result()
    except BaseException as exc:
        return AgentExecutionResult(request=request, error=exc)
    return AgentExecutionResult(request=request, outcome=outcome)


def _raise_batch_fatal_errors(
    results: Sequence[AgentExecutionResult],
) -> None:
    """Re-raise turn-fatal errors only after every peer has settled."""

    for result in results:
        if isinstance(result.error, (KeyboardInterrupt, SystemExit)):
            raise result.error
    for result in results:
        if isinstance(result.error, AgentInfrastructureError):
            raise result.error


def _parallel_execution(
    execute_request: Callable[[AgentExecutionRequest], object],
    requests: tuple[AgentExecutionRequest, ...],
) -> tuple[AgentExecutionResult, ...]:
    if not requests:
        return ()
    with ThreadPoolExecutor(
        max_workers=len(requests),
        thread_name_prefix="canonical-agent-turn",
    ) as executor:
        futures = tuple(
            executor.submit(execute_request, request)
            for request in requests
        )
        return tuple(
            _capture_future(request, future)
            for request, future in zip(requests, futures)
        )


class InProcessAgentExecutionService:
    """Development default preserving the existing direct-call behavior."""

    def execute(
        self,
        program: AgentProgram,
        context: AgentContext,
        active_budget_seconds: float,
    ) -> AgentOutcome:
        del active_budget_seconds
        return program.run(context)

    def validate_batch(
        self,
        requests: Sequence[AgentExecutionRequest],
    ) -> None:
        _ordered_execution_requests(requests)

    def execute_many(
        self,
        requests: Sequence[AgentExecutionRequest],
    ) -> tuple[AgentExecutionResult, ...]:
        return execute_agent_batch(self, requests)

    def reserve_batch(
        self,
        requests: Sequence[AgentExecutionRequest],
    ) -> AgentBatchReservation:
        prepared = validate_agent_batch(self, requests)
        return AgentBatchReservation(self, prepared)

    def execute_reserved_many(
        self,
        reservation: AgentBatchReservation,
    ) -> tuple[AgentExecutionResult, ...]:
        prepared = reservation.requests_for(self)
        return _parallel_execution(self._execute_validated_request, prepared)

    def _execute_validated_request(self, request: AgentExecutionRequest) -> object:
        return self.execute(
            request.program,
            request.context,
            request.active_budget_seconds,
        )


class SubprocessAgentExecutionService:
    """Run a command Agent in an execution-scoped workspace and process group.

    This adapter provides a durable request/result protocol, elapsed-time
    deadlines, partial nomination recovery, log capture, and cleanup. It is a
    process boundary, not an OS security sandbox. In particular, the current
    Candidate decoder and later Candidate evaluation are Controller-side.
    """

    def __init__(
        self,
        workspace_root: str | os.PathLike[str],
        *,
        hard_wall_seconds: float | None,
        enforce_active_budget: bool = False,
        capability_facade: AgentCapabilityFacade | None = None,
        controller_gpu_devices: Sequence[str] = (),
        evaluation_device_map: Mapping[str, str] | None = None,
        termination_grace_seconds: float = 2.0,
        poll_interval_seconds: float = 0.05,
        max_protocol_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.hard_wall_seconds = (None if hard_wall_seconds is None else
                                  _positive_seconds(hard_wall_seconds, "hard_wall_seconds"))
        self.allocation_should_pause = None
        if not isinstance(enforce_active_budget, bool):
            raise TypeError("enforce_active_budget must be a boolean")
        # Explicit opt-in only for replaying the former dual-clock protocol.
        # New turns spend one elapsed-time budget, including all tool waits.
        if enforce_active_budget and self.hard_wall_seconds is None:
            raise ValueError("active-clock mode requires a finite outer limit")
        self.enforce_active_budget = enforce_active_budget
        self.termination_grace_seconds = _positive_seconds(
            termination_grace_seconds,
            "termination_grace_seconds",
        )
        self.poll_interval_seconds = _positive_seconds(
            poll_interval_seconds,
            "poll_interval_seconds",
        )
        if isinstance(max_protocol_bytes, bool) or not isinstance(max_protocol_bytes, int):
            raise TypeError("max_protocol_bytes must be an integer")
        if max_protocol_bytes < 1024:
            raise ValueError("max_protocol_bytes must be at least 1024")
        self.max_protocol_bytes = max_protocol_bytes
        self.controller_gpu_devices = _validated_gpu_devices(
            controller_gpu_devices,
            "controller_gpu_devices",
        )
        if evaluation_device_map is not None:
            if not isinstance(evaluation_device_map, Mapping):
                raise TypeError("evaluation_device_map must be a mapping or None")
            tokens = _validated_gpu_devices(tuple(evaluation_device_map), "evaluation_device_map")
            if tokens != tuple(evaluation_device_map):
                raise ValueError("evaluation device tokens cannot contain surrounding whitespace")
            logical = tuple(evaluation_device_map.values())
            if any(not isinstance(value, str) or re.fullmatch(r"cuda:(0|[1-9][0-9]*)", value) is None
                   for value in logical) or len(logical) != len(set(logical)):
                raise ValueError("evaluation_device_map requires distinct logical cuda:N devices")
        self.evaluation_device_map = (
            None if not evaluation_device_map else MappingProxyType(dict(evaluation_device_map))
        )
        if capability_facade is None:
            # Lazy import avoids the capability module's dependency on the
            # execution DTOs while making formal command Agents useful by default.
            from .agent_capability import ControllerAgentCapabilityFacade

            capability_facade = ControllerAgentCapabilityFacade()
        self.capability_facade = capability_facade
        self._audits: list[AgentExecutionAudit] = []
        self._audit_lock = Lock()
        self._gpu_reservation_lock = Lock()
        self._reserved_gpu_devices: dict[str, tuple[str, str]] = {}
        self._quarantined_gpu_devices: dict[str, str] = {}
        self.workspace_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.workspace_root.is_symlink() or not self.workspace_root.is_dir():
            raise ValueError("workspace_root must be a real directory")
        os.chmod(self.workspace_root, 0o700)

    @property
    def audits(self) -> tuple[AgentExecutionAudit, ...]:
        with self._audit_lock:
            return tuple(self._audits)

    @property
    def quarantined_gpu_devices(self) -> tuple[str, ...]:
        """Devices withheld after an unconfirmed worker cleanup."""

        with self._gpu_reservation_lock:
            return tuple(sorted(self._quarantined_gpu_devices))

    def validate_batch(
        self,
        requests: Sequence[AgentExecutionRequest],
    ) -> None:
        prepared = _ordered_execution_requests(requests)
        owners = {
            device: "Controller"
            for device in self.controller_gpu_devices
        }
        for index, request in enumerate(prepared):
            program = request.program
            if not isinstance(program, CommandAgentProgram):
                raise TypeError(
                    "SubprocessAgentExecutionService requires "
                    "CommandAgentProgram values"
                )
            if request.active_budget_seconds is None and self.hard_wall_seconds is not None:
                raise ValueError("completion-driven request requires an uncapped service")
            if (
                self.enforce_active_budget
                and self.hard_wall_seconds < request.active_budget_seconds
            ):
                raise ValueError(
                    "hard_wall_seconds cannot be shorter than "
                    "active_budget_seconds"
                )
            agent = getattr(request.context, "agent", None)
            agent_id = getattr(agent, "agent_id", None)
            owner = (
                f"Agent {agent_id!r}"
                if isinstance(agent_id, str) and agent_id
                else f"Agent request {index}"
            )
            for device in program.gpu_devices:
                previous = owners.get(device)
                if previous is not None:
                    raise AgentGPUConflict(
                        f"GPU device {device!r} is assigned to both "
                        f"{previous} and {owner}"
                    )
                owners[device] = owner

    def reserve_batch(
        self,
        requests: Sequence[AgentExecutionRequest],
    ) -> AgentBatchReservation:
        prepared = _ordered_execution_requests(requests)
        self.validate_batch(prepared)
        reservation_id = uuid4().hex
        allocations: dict[str, str] = {}
        for index, request in enumerate(prepared):
            program = request.program
            assert isinstance(program, CommandAgentProgram)
            agent_id = getattr(request.context.agent, "agent_id", None)
            owner = (
                f"Agent {agent_id!r}"
                if isinstance(agent_id, str) and agent_id
                else f"Agent request {index}"
            )
            allocations.update({device: owner for device in program.gpu_devices})
        with self._gpu_reservation_lock:
            for device, owner in allocations.items():
                quarantine_reason = self._quarantined_gpu_devices.get(device)
                if quarantine_reason is not None:
                    raise AgentGPUConflict(
                        f"GPU device {device!r} is quarantined after an "
                        f"unconfirmed cleanup: {quarantine_reason}"
                    )
                active = self._reserved_gpu_devices.get(device)
                if active is not None:
                    raise AgentGPUConflict(
                        f"GPU device {device!r} is already reserved by "
                        f"{active[1]} and cannot also serve {owner}"
                    )
            self._reserved_gpu_devices.update(
                {
                    device: (reservation_id, owner)
                    for device, owner in allocations.items()
                }
            )

        def release() -> None:
            with self._gpu_reservation_lock:
                for device in allocations:
                    active = self._reserved_gpu_devices.get(device)
                    if active is not None and active[0] == reservation_id:
                        del self._reserved_gpu_devices[device]

        return AgentBatchReservation(self, prepared, release=release)

    @contextmanager
    def reserve_evaluation_devices(self, programs: Sequence[AgentProgram]):
        """Borrow this run's idle GPUs only after all Agent reservations settle.

        Configuration is opt-in and established before CUDA initialization.
        Closing plugin resources precedes release; unconfirmed cleanup keeps
        the devices quarantined against the next Agent round.
        """
        if self.evaluation_device_map is None:
            yield None
            return
        tokens = list(self.controller_gpu_devices)
        for program in programs:
            if not isinstance(program, CommandAgentProgram):
                raise TypeError("evaluation GPU borrowing requires CommandAgentProgram values")
            tokens.extend(program.gpu_devices)
        tokens = tuple(dict.fromkeys(tokens))
        missing = set(tokens) - set(self.evaluation_device_map)
        if missing:
            raise ValueError("evaluation_device_map omits a configured run GPU")
        if not tokens:
            yield None
            return
        pool = EvaluationDevicePool(tuple(self.evaluation_device_map[token] for token in tokens))
        reservation_id = uuid4().hex
        with self._gpu_reservation_lock:
            if self._reserved_gpu_devices:
                raise AgentGPUConflict("evaluation GPUs cannot be borrowed while an Agent or evaluation reservation is active")
            if any(token in self._quarantined_gpu_devices for token in tokens):
                raise AgentGPUConflict("evaluation GPU is quarantined after unconfirmed cleanup")
            self._reserved_gpu_devices.update({
                token: (reservation_id, "Controller evaluation") for token in tokens
            })
        try:
            yield pool
        finally:
            try:
                pool.close()
            except BaseException as error:
                with self._gpu_reservation_lock:
                    self._quarantined_gpu_devices.update({
                        token: "Controller evaluation resource cleanup could not be confirmed"
                        for token in tokens
                    })
                raise AgentInfrastructureError(
                    "Controller evaluation resource cleanup could not be confirmed"
                ) from error
            finally:
                with self._gpu_reservation_lock:
                    for token in tokens:
                        active = self._reserved_gpu_devices.get(token)
                        if active is not None and active[0] == reservation_id:
                            del self._reserved_gpu_devices[token]

    def execute_many(
        self,
        requests: Sequence[AgentExecutionRequest],
    ) -> tuple[AgentExecutionResult, ...]:
        return execute_agent_batch(self, requests)

    def execute_reserved_many(
        self,
        reservation: AgentBatchReservation,
    ) -> tuple[AgentExecutionResult, ...]:
        prepared = reservation.requests_for(self)
        results = _parallel_execution(self._execute_validated_request, prepared)
        quarantined: dict[str, str] = {}
        for request, result in zip(prepared, results):
            if not isinstance(result.error, AgentInfrastructureError):
                continue
            program = request.program
            assert isinstance(program, CommandAgentProgram)
            quarantined.update(
                {
                    device: str(result.error)
                    for device in program.gpu_devices
                }
            )
        if quarantined:
            with self._gpu_reservation_lock:
                self._quarantined_gpu_devices.update(quarantined)
        return results

    def _execute_validated_request(self, request: AgentExecutionRequest) -> object:
        return self._execute(
            request.program,
            request.context,
            request.active_budget_seconds,
            continuation=request.continuation, checkpoint=request.checkpoint,
        )

    def execute(
        self,
        program: AgentProgram,
        context: AgentContext,
        active_budget_seconds: float,
    ) -> AgentOutcome:
        request = AgentExecutionRequest(program, context, active_budget_seconds)
        result = execute_agent_batch(self, (request,))[0]
        if result.error is not None:
            raise result.error
        return result.outcome

    def execute_continuable(self, program, context, continuation, checkpoint):
        """Continue one natural-completion turn across allocation boundaries."""
        if self.hard_wall_seconds is not None:
            raise ValueError("continuable turns require an uncapped service")
        request = AgentExecutionRequest(program, context, None, continuation, checkpoint)
        result = execute_agent_batch(self, (request,))[0]
        if result.error is not None:
            raise result.error
        return result.outcome

    def _execute(
        self,
        program: AgentProgram,
        context: AgentContext,
        active_budget_seconds: float | None,
        *, continuation=None, checkpoint=None,
    ) -> AgentOutcome:
        single_request = AgentExecutionRequest(
            program,
            context,
            active_budget_seconds,
        )
        assert isinstance(program, CommandAgentProgram)
        active_budget = single_request.active_budget_seconds

        execution_id = uuid4().hex
        resumed = continuation is not None and bool(continuation)
        if resumed:
            workspace = Path(continuation["workspace"])
            if workspace.is_symlink() or workspace.parent.resolve() != self.workspace_root or not workspace.is_dir():
                raise AgentInfrastructureError("invalid continuation workspace")
            execution_id = continuation["execution_id"]
            previous = Path(continuation["audit_path"])
            if previous.is_symlink() or not previous.resolve().is_relative_to(workspace):
                raise AgentInfrastructureError("invalid continuation audit path")
            audit = json.loads(previous.read_text())
            if (audit["execution_id"] != execution_id or audit["status"] != "allocation_paused"
                    or not all(audit[k] for k in ("capability_revoked", "capability_drained", "process_group_cleaned"))):
                raise AgentInfrastructureError("continuation requires a clean allocation pause")
            archive = previous.parent
            for name in ("public_resources", "incumbent", "CANONICAL_AGENT_PROMPT.md",
                         "codex-last-message.txt", "codex-auth-lifecycle.json", "codex-recovery.json"):
                item = workspace / name
                if item.exists() or item.is_symlink():
                    # Moving a directory to another parent updates '..' and can
                    # require owner-write even when both parents are writable.
                    mode = stat.S_IMODE(item.lstat().st_mode)
                    directory = item.is_dir() and not item.is_symlink()
                    if directory:
                        item.chmod(mode | stat.S_IWUSR)
                    try:
                        item.rename(archive / name)
                    finally:
                        if directory:
                            (item if item.exists() else archive / name).chmod(mode)
        else:
            workspace = self._create_workspace(context, execution_id)
        segment = int(continuation.get("segment", 0)) + 1 if continuation is not None else 0
        protocol_root = workspace / "segments" / f"{segment:04d}" if continuation is not None else workspace
        protocol_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        request_path = protocol_root / "request.json"
        response_path = protocol_root / "response.json"
        nominations_path = workspace / "nominations"
        stdout_path = protocol_root / "stdout.log"
        stderr_path = protocol_root / "stderr.log"
        audit_path = protocol_root / "audit.json"
        if continuation is not None:
            continuation.update(workspace=str(workspace), execution_id=execution_id,
                                segment=segment, audit_path=str(audit_path))
            if checkpoint:
                checkpoint()
        campaign_memory_path = workspace / "CAMPAIGN_MEMORY.md"
        candidate_history_path = workspace / "CANDIDATE_HISTORY.json"
        round_log_path = workspace / "ROUND_LOG.md"
        nominations_path.mkdir(mode=0o700, exist_ok=resumed)
        (workspace / "home").mkdir(mode=0o700, exist_ok=resumed)
        (workspace / "tmp").mkdir(mode=0o700, exist_ok=resumed)
        (workspace / "pycache").mkdir(mode=0o700, exist_ok=resumed)

        execution_started_at_unix = time.time()
        worker_started_at_unix: float | None = None
        worker_finished_at_unix: float | None = None
        worker_started: float | None = None
        worker_finished: float | None = None
        worker_execution_finished: float | None = None
        lease: AgentCapabilityLease | None = None
        process: subprocess.Popen[bytes] | None = None
        status = "setup_failed"
        error: str | None = None
        return_code: int | None = None
        controller_service_seconds = 0.0
        capability_revoked = False
        capability_drained = False
        capability_revoke_attempted = False
        capability_close_attempted = False
        process_group_cleaned = True
        artifacts: tuple[Artifact, ...] = ()
        observations: tuple[str, ...] = ()
        rejected_nominations = 0
        control_flow_error: KeyboardInterrupt | SystemExit | None = None
        infrastructure_error: AgentInfrastructureError | None = None
        initial_round_log = ""
        round_log = ""

        try:
            from .memory import agent_round_log_template

            initial_round_log = agent_round_log_template(
                context.round_plan.round_index,
                context.agent.name,
                incumbent_name=(
                    context.incumbent.name if context.incumbent is not None else None
                ),
            )
            if not resumed:
                _write_text_new(campaign_memory_path, context.memory.markdown)
            os.chmod(campaign_memory_path, 0o400)
            if not resumed:
                _write_json_new(candidate_history_path, self._candidate_history_document(context))
            os.chmod(candidate_history_path, 0o400)
            if not resumed:
                _write_text_new(round_log_path, initial_round_log)
            lease = self.capability_facade.open(
                context,
                workspace,
                execution_id,
                program,
            )
            if not isinstance(lease, AgentCapabilityLease):
                raise TypeError("capability facade must return AgentCapabilityLease")
            request = self._build_request(
                program,
                context,
                execution_id,
                workspace,
                active_budget,
                lease.manifest,
            )
            _write_json_new(request_path, request)

            environment = self._worker_environment(
                program,
                workspace,
                execution_id,
                request_path,
                response_path,
                nominations_path,
            )
            with stdout_path.open("xb") as stdout_handle, stderr_path.open("xb") as stderr_handle:
                # Start the one elapsed-time clock before launching the worker.
                worker_started_at_unix = time.time()
                worker_started = time.monotonic()
                # Native reconnects share the original Agent elapsed deadline.
                wall_limit = self._wall_budget(active_budget)
                if wall_limit is not None:
                    environment[AGENT_DEADLINE_MONOTONIC_ENV] = str(worker_started + wall_limit)
                process = subprocess.Popen(
                    program.command,
                    cwd=workspace,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    start_new_session=True,
                    shell=False,
                )
                status, controller_service_seconds = self._wait(
                    process,
                    lease,
                    worker_started,
                    active_budget,
                    **({"progress": _native_recovery_observer(workspace, context.execution_progress)}
                       if context.execution_progress is not None else {}),
                )
                worker_execution_finished = time.monotonic()

            if status != "completed":
                capability_revoke_attempted = True
                try:
                    lease.revoke()
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as exc:
                    raise AgentInfrastructureError(
                        "Agent capability revocation failed"
                    ) from exc
                capability_revoked = lease.revoked
                process_group_cleaned = self._terminate_process_group(process)
                capability_close_attempted = True
                try:
                    lease.close()
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as exc:
                    raise AgentInfrastructureError(
                        "Agent capability drain failed"
                    ) from exc
                capability_drained = lease.drained
            else:
                return_code = process.returncode
                capability_revoke_attempted = True
                capability_close_attempted = True
                try:
                    lease.close()
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as exc:
                    raise AgentInfrastructureError(
                        "Agent capability drain failed"
                    ) from exc
                capability_revoked = lease.revoked
                capability_drained = lease.drained
                process_group_cleaned = self._terminate_remaining_process_group(process)

            return_code = process.returncode
            worker_finished = time.monotonic()
            worker_finished_at_unix = time.time()
            if not process_group_cleaned:
                raise AgentExecutionError(
                    "Agent process-group cleanup could not be confirmed"
                )
            if status == "allocation_paused" and program.worker_settings.get("continuation_backend") == "canonical_codex_v1":
                receipt = protocol_root / "native_shutdown.json"
                try:
                    proof = json.loads(receipt.read_text())
                    confirmed = (not receipt.is_symlink() and proof.get("process_group_cleaned") is True
                                 and proof.get("execution_id") == execution_id
                                 and proof.get("request_path") == str(request_path))
                except (OSError, ValueError):
                    confirmed = False
                if not confirmed:
                    raise AgentInfrastructureError("native session cleanup unconfirmed; refusing allocation resume")
            response = None if status == "allocation_paused" else self._read_response(response_path, execution_id)
            observations = response["observations"] if response is not None else ()
            if status != "allocation_paused":
                artifacts, rejected_nominations = self._load_nominations(
                    nominations_path, execution_id, program, context, workspace)


            if status == "completed":
                if return_code != 0:
                    status = "worker_failed"
                    error = f"Agent worker exited with status {return_code}"
                elif response is None:
                    status = "protocol_error"
                    error = "Agent worker did not atomically publish response.json"
                elif response["status"] != "completed":
                    status = "worker_failed"
                    error = response.get("error") or "Agent worker reported failure"
                elif rejected_nominations:
                    status = "protocol_error"
                    error = (
                        f"Agent worker published {rejected_nominations} invalid nomination(s)"
                    )
            elif status != "allocation_paused":
                error = (
                    "Agent worker exceeded its active-time budget"
                    if status == "active_timeout"
                    else "Agent worker exceeded its hard-wall deadline"
                )
        except BaseException as exc:
            if worker_started is not None and worker_execution_finished is None:
                worker_execution_finished = time.monotonic()
            error = f"{type(exc).__name__}: {exc}"
            status = "execution_error"
            if isinstance(exc, AgentInfrastructureError):
                infrastructure_error = exc
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                control_flow_error = exc
            if lease is not None and not lease.revoked:
                capability_revoke_attempted = True
                try:
                    lease.revoke()
                except BaseException as revoke_exc:
                    error += f"; capability revoke failed: {revoke_exc}"
                    if isinstance(revoke_exc, (KeyboardInterrupt, SystemExit)):
                        control_flow_error = control_flow_error or revoke_exc
                    else:
                        infrastructure_error = AgentInfrastructureError(
                            "Agent capability revocation failed"
                        )
                finally:
                    capability_revoked = lease.revoked
            if process is not None:
                try:
                    process_group_cleaned = self._terminate_process_group(process)
                    return_code = process.returncode
                    if not process_group_cleaned:
                        infrastructure_error = AgentInfrastructureError(
                            "Agent process-group cleanup could not be confirmed"
                        )
                except BaseException as cleanup_exc:
                    process_group_cleaned = False
                    error += f"; process cleanup failed: {cleanup_exc}"
                    if isinstance(cleanup_exc, (KeyboardInterrupt, SystemExit)):
                        control_flow_error = control_flow_error or cleanup_exc
                    else:
                        infrastructure_error = AgentInfrastructureError(
                            "Agent process-group cleanup failed"
                        )
            if lease is not None and not lease.drained:
                capability_close_attempted = True
                try:
                    lease.close()
                except BaseException as drain_exc:
                    error += f"; capability drain failed: {drain_exc}"
                    if isinstance(drain_exc, (KeyboardInterrupt, SystemExit)):
                        control_flow_error = control_flow_error or drain_exc
                    else:
                        infrastructure_error = AgentInfrastructureError(
                            "Agent capability drain failed"
                        )
                finally:
                    capability_revoked = lease.revoked
                    capability_drained = lease.drained
            if process is not None and worker_started is not None:
                worker_finished = time.monotonic()
                worker_finished_at_unix = time.time()
            if process is not None and process_group_cleaned:
                try:
                    artifacts, rejected_nominations = self._load_nominations(
                        nominations_path,
                        execution_id,
                        program,
                        context,
                        workspace,
                    )
                except BaseException as nomination_exc:
                    error += f"; nomination recovery failed: {nomination_exc}"
                    if isinstance(nomination_exc, (KeyboardInterrupt, SystemExit)):
                        control_flow_error = control_flow_error or nomination_exc
            else:
                error += (
                    "; nomination recovery skipped because process-group "
                    "cleanup was not confirmed"
                )

        # ROUND_LOG.md is a live notebook. Once the worker is gone, preserve
        # its last complete contents even when the turn timed out or failed.
        if process is None or process_group_cleaned:
            try:
                recovered_round_log = _read_text_file(
                    round_log_path,
                    self.max_protocol_bytes,
                )
            except AgentExecutionError as log_error:
                if round_log_path.exists():
                    observations = (
                        *observations,
                        f"ROUND_LOG.md could not be read: {log_error}",
                    )
            else:
                if recovered_round_log.strip() != initial_round_log.strip():
                    round_log = recovered_round_log

        if lease is not None:
            # A close callback can drain successfully and then re-raise an
            # earlier revoke failure.  Always snapshot the lease itself so
            # the immutable audit records the containment work that happened.
            capability_revoked = lease.revoked
            capability_drained = lease.drained

        if infrastructure_error is not None:
            status = "infrastructure_error"

        audited_at_unix = time.time()
        if worker_started is None:
            wall_seconds = 0.0
            audit_started_at_unix = audited_at_unix
            audit_finished_at_unix = audited_at_unix
        else:
            if worker_finished is None:
                worker_finished = time.monotonic()
                worker_finished_at_unix = time.time()
            wall_seconds = max(0.0, worker_finished - worker_started)
            audit_started_at_unix = (
                worker_started_at_unix
                if worker_started_at_unix is not None
                else execution_started_at_unix
            )
            audit_finished_at_unix = (
                worker_finished_at_unix
                if worker_finished_at_unix is not None
                else audited_at_unix
            )
        if lease is not None:
            try:
                controller_service_seconds = max(
                    controller_service_seconds,
                    lease.controller_service_seconds,
                )
            except BaseException as exc:
                error = f"{error}; service clock failed: {exc}" if error else str(exc)
                status = "execution_error"
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    control_flow_error = control_flow_error or exc
        active_seconds = max(0.0, wall_seconds - controller_service_seconds)
        agent_execution_wall_seconds = (
            0.0 if worker_started is None else min(
                wall_seconds,
                max(0.0, (worker_execution_finished or worker_finished) - worker_started),
            )
        )
        nomination_policy = program.worker_settings.get("nomination_policy", "all")
        nomination_journal_entry_count = None
        if process_group_cleaned:
            try:
                nomination_journal_entry_count = len(tuple(nominations_path.glob("nomination-*.json")))
            except OSError:
                pass  # Preserve the primary execution error if evidence is unavailable.
        selected_sequence = None
        if artifacts and not rejected_nominations:
            if nomination_policy == "latest":
                selected_sequence = nomination_journal_entry_count
            elif nomination_policy == "single_primary_v1":
                selected_sequence = read_agent_primary_nomination(
                    nominations_path, execution_id, max_bytes=self.max_protocol_bytes)
        audit = AgentExecutionAudit(
            schema=AGENT_EXECUTION_AUDIT_SCHEMA,
            execution_id=execution_id,
            round_index=context.round_plan.round_index,
            agent_id=context.agent.agent_id,
            status=status,
            pid=process.pid if process is not None else None,
            return_code=return_code,
            started_at_unix=audit_started_at_unix,
            finished_at_unix=audit_finished_at_unix,
            wall_seconds=wall_seconds,
            controller_service_seconds=controller_service_seconds,
            active_seconds=active_seconds,
            active_budget_seconds=active_budget if self.enforce_active_budget else None,
            hard_wall_seconds=self._wall_budget(active_budget),
            workspace=str(workspace),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            request_path=str(request_path),
            response_path=str(response_path),
            nomination_count=len(artifacts),
            rejected_nomination_count=rejected_nominations,
            nomination_policy=nomination_policy,
            nomination_journal_entry_count=nomination_journal_entry_count,
            final_nomination_sequence=selected_sequence,
            capability_revoked=capability_revoked,
            capability_drained=capability_drained,
            process_group_cleaned=process_group_cleaned,
            error=error,
            time_budget_mode="completion" if active_budget is None else "active_and_hard_wall" if self.enforce_active_budget else "hard_wall",
            agent_execution_wall_seconds=agent_execution_wall_seconds,
            cleanup_wall_seconds=max(0.0, wall_seconds - agent_execution_wall_seconds),
        )
        audit_persistence_error: BaseException | None = None
        try:
            _write_json_new(audit_path, audit.to_payload())
            os.chmod(audit_path, 0o400)
        except BaseException as exc:
            # Audit durability is important, but an audit I/O failure must not
            # hide a stronger containment failure.  The caller must still see
            # AgentInfrastructureError so its GPU reservation is quarantined.
            audit_persistence_error = exc
        else:
            with self._audit_lock:
                self._audits.append(audit)

        if control_flow_error is not None:
            if audit_persistence_error is not None:
                control_flow_error.add_note(
                    "Agent execution audit persistence also failed: "
                    f"{type(audit_persistence_error).__name__}: "
                    f"{audit_persistence_error}"
                )
            setattr(control_flow_error, "round_log", round_log)
            raise control_flow_error
        if infrastructure_error is not None:
            if audit_persistence_error is not None:
                infrastructure_error.add_note(
                    "Agent execution audit persistence also failed: "
                    f"{type(audit_persistence_error).__name__}: "
                    f"{audit_persistence_error}"
                )
            setattr(infrastructure_error, "round_log", round_log)
            raise infrastructure_error
        if audit_persistence_error is not None:
            setattr(audit_persistence_error, "round_log", round_log)
            raise audit_persistence_error
        if rejected_nominations or status == "protocol_error":
            protocol_error = AgentExecutionError(
                error or "Agent worker protocol failed"
            )
            protocol_error.round_log = round_log
            raise protocol_error
        if status == "allocation_paused":
            from .campaign_continuation import CampaignPaused
            raise CampaignPaused(stage="agent_continuation")
        recoverable_statuses = {
            "completed",
            "active_timeout",
            "hard_wall_timeout",
            "worker_failed",
        }
        if artifacts and status in recoverable_statuses:
            if status != "completed":
                observations = (
                    *observations,
                    f"Execution ended with {status}; preserved "
                    f"{len(artifacts)} committed nomination(s).",
                )
            from .runtime import AgentOutcome

            return AgentOutcome(
                artifacts[0],
                observations,
                artifacts=artifacts[1:],
                round_log=round_log,
                execution_status=(
                    "completed"
                    if status == "completed"
                    else "timed_out"
                    if status in {"active_timeout", "hard_wall_timeout"}
                    else "failed"
                ),
            )
        if status == "completed":
            from .runtime import AgentOutcome

            return AgentOutcome(None, observations, round_log=round_log)
        if status in {"active_timeout", "hard_wall_timeout"}:
            timeout = AgentExecutionTimeout(error or status)
            timeout.round_log = round_log
            raise timeout
        execution_error = AgentExecutionError(error or status)
        execution_error.round_log = round_log
        raise execution_error

    def _create_workspace(self, context: AgentContext, execution_id: str) -> Path:
        component = _SAFE_COMPONENT.sub("-", context.agent.agent_id).strip(".-") or "agent"
        workspace = self.workspace_root / (
            f"round-{context.round_plan.round_index:04d}-{component}-{execution_id}"
        )
        workspace.mkdir(mode=0o700)
        return workspace

    def _build_request(
        self,
        program: CommandAgentProgram,
        context: AgentContext,
        execution_id: str,
        workspace: Path,
        active_budget: float,
        capabilities: Mapping[str, object],
    ) -> dict[str, object]:
        from .agent_wire import AgentWireCodec

        policy = context.round_plan.development_sampling
        wire = capabilities.get("wire")
        wire = wire if isinstance(wire, Mapping) else {}
        inline_array_bytes = wire.get("inline_array_bytes", 64 * 1024)
        max_sidecar_bytes = wire.get(
            "max_sidecar_bytes",
            2 * 1024 * 1024 * 1024,
        )
        codec = AgentWireCodec(
            workspace / "capability-arrays",
            inline_array_bytes=(
                inline_array_bytes
                if isinstance(inline_array_bytes, int)
                and not isinstance(inline_array_bytes, bool)
                else 64 * 1024
            ),
            max_sidecar_bytes=(
                max_sidecar_bytes
                if isinstance(max_sidecar_bytes, int)
                and not isinstance(max_sidecar_bytes, bool)
                else 2 * 1024 * 1024 * 1024
            ),
        )
        incumbent: dict[str, object] | None = None
        if context.incumbent is not None:
            incumbent = {
                "artifact_id": context.incumbent.artifact_id,
                "kind": context.incumbent.kind.value,
                "name": context.incumbent.name,
                "summary": context.incumbent.summary,
                "usage": context.incumbent.usage,
                "round_index": context.incumbent.round_index,
                "agent_id": context.incumbent.agent_id,
                "parent_artifact_id": context.incumbent.parent_artifact_id,
                "metadata": codec.encode(dict(context.incumbent.metadata)),
                "program": None,
                "program_available": False,
            }
            if program.incumbent_encoder is not None:
                encoded = dict(program.incumbent_encoder(context.incumbent, workspace))
                incumbent["program"] = _json_value(encoded)
                incumbent["program_available"] = True
        public_problem = _stage_public_problem_bundle(
            context.public_problem_bundle,
            workspace,
        )
        request = {
            "schema": AGENT_WORKER_REQUEST_SCHEMA,
            "execution_id": execution_id,
            "round": {
                "round_index": context.round_plan.round_index,
                "active_budget_seconds": active_budget if self.enforce_active_budget else None,
                "hard_wall_seconds": self._wall_budget(active_budget),
                "time_budget_mode": "completion" if active_budget is None else "active_and_hard_wall" if self.enforce_active_budget else "hard_wall",
                "development_sampling": {
                    "strategy": policy.strategy.value,
                    "max_cases": policy.max_cases_per_agent_per_round,
                    "allowed_sources": [source.value for source in policy.allowed_sources],
                    "hide_case_types": policy.hide_case_types,
                },
            },
            "agent": {
                "agent_id": context.agent.agent_id,
                "name": context.agent.name,
                "prompt": context.prompt,
                "seed": context.seed,
            },
            "incumbent": incumbent,
            "campaign_memory": {
                "format": "markdown",
                "completed_rounds": len(context.memory.rounds),
                "path": "CAMPAIGN_MEMORY.md",
            },
            "candidate_history": {
                "format": "json",
                "candidate_count": len(context.candidate_history.nodes),
                "path": "CANDIDATE_HISTORY.json",
            },
            "round_log": {
                "format": "markdown",
                "path": "ROUND_LOG.md",
                "max_bytes": self.max_protocol_bytes,
                "instruction": (
                    "Use write_agent_round_log() to save a concise final research "
                    "record before finishing. Earlier checkpoints are available; "
                    "the Controller preserves the last complete version for the next round."
                ),
            },
            "issued_cases": [
                {
                    "case_id": case.case_id,
                    "source": case.source.value,
                    "visibility": case.visibility.value,
                    "public_data": codec.encode(dict(case.public_data)),
                    "round_index": case.round_index,
                    "parent_case_id": case.parent_case_id,
                    "tags": list(case.tags),
                }
                for case in context.cases
            ],
            "public_problem": public_problem,
            "candidate_execution": {
                "case_charged_wall_seconds": context.candidate_solve_budget_seconds,
                "budget_scope": "entire_candidate_worker_including_prepare_and_query",
                "excluded_time": "controller_metered_queue_and_lock_wait_only",
                "oracle_numerical_execution_is_charged": True,
            },
            "capabilities": _json_value(dict(capabilities)),
            "worker_settings": _json_value(dict(program.worker_settings)),
        }
        encoded = _json_bytes(request)
        if len(encoded) > self.max_protocol_bytes:
            raise AgentExecutionError("Agent worker request exceeds protocol size limit")
        return request

    @staticmethod
    def _candidate_history_document(context: AgentContext) -> dict[str, object]:
        return {
            "schema": "canonical-candidate-history-v1",
            "candidates": [
                {
                    "node_id": node.node_id,
                    "round_index": node.round_index,
                    "agent_id": node.agent_id,
                    "artifact_id": node.artifact_id,
                    "parent_node_id": node.parent_node_id,
                    "title": node.title,
                    "summary": node.summary,
                    "official_distributions": {
                        name: {
                            "case_count": summary.case_count,
                            "mean": summary.mean,
                            "cvar": summary.cvar,
                            "maximum": summary.maximum,
                            "metrics": dict(summary.metrics),
                        }
                        for name, summary in node.official_distributions.items()
                    },
                    "failures": list(node.failures),
                    "status": node.status,
                    "selected": node.selected,
                }
                for node in context.candidate_history.nodes
            ],
        }

    def _worker_environment(
        self,
        program: CommandAgentProgram,
        workspace: Path,
        execution_id: str,
        request_path: Path,
        response_path: Path,
        nominations_path: Path,
    ) -> dict[str, str]:
        # Do not leak credentials and scheduler internals from the Controller.
        # Deployments can add any required library variables explicitly on the
        # CommandAgentProgram.
        environment = {
            name: os.environ[name]
            for name in _INHERITED_ENVIRONMENT
            if name in os.environ
        }
        environment.update(program.environment)
        gpu_policy = program.worker_settings.get("agent_gpu_execution")
        if (isinstance(gpu_policy, Mapping)
                and gpu_policy.get("protocol") == "shared_gpu_v1"
                and gpu_policy.get("required") is True):
            # Only the explicit native GPU gate needs this Controller-owned
            # allocation context. Never invent it or accept a program override;
            # the assigned CUDA_VISIBLE_DEVICES reservation remains below.
            for name in ("SLURM_JOB_ID", "CUDA_DEVICE_ORDER"):
                environment.pop(name, None)
                if name in os.environ:
                    environment[name] = os.environ[name]
        environment.update(
            {
                AGENT_WORKSPACE_ENV: str(workspace),
                AGENT_REQUEST_ENV: str(request_path),
                AGENT_RESPONSE_ENV: str(response_path),
                AGENT_NOMINATIONS_ENV: str(nominations_path),
                AGENT_EXECUTION_ID_ENV: execution_id,
                AGENT_CAMPAIGN_MEMORY_ENV: str(workspace / "CAMPAIGN_MEMORY.md"),
                AGENT_CANDIDATE_HISTORY_ENV: str(
                    workspace / "CANDIDATE_HISTORY.json"
                ),
                AGENT_ROUND_LOG_ENV: str(workspace / "ROUND_LOG.md"),
                AGENT_ROUND_LOG_MAX_BYTES_ENV: str(self.max_protocol_bytes),
                AGENT_PROTOCOL_MAX_BYTES_ENV: str(self.max_protocol_bytes),
                "HOME": str(workspace / "home"),
                "TMPDIR": str(workspace / "tmp"),
                "TMP": str(workspace / "tmp"),
                "TEMP": str(workspace / "tmp"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPYCACHEPREFIX": str(workspace / "pycache"),
                "CUDA_VISIBLE_DEVICES": ",".join(program.gpu_devices),
            }
        )
        return environment

    def _wall_budget(self, requested_seconds: float) -> float:
        """Round elapsed-time limit, capped by the service's outer deadline.

        The execution port keeps its historical argument name for callers,
        but by default that argument now means total elapsed time.
        """
        limits = [v for v in (self.hard_wall_seconds, requested_seconds) if v is not None]
        return self.hard_wall_seconds if self.enforce_active_budget else min(limits) if limits else None

    def _wait(
        self,
        process: subprocess.Popen[bytes],
        lease: AgentCapabilityLease,
        started: float,
        active_budget: float,
        *, progress: Callable[[], None] | None = None,
    ) -> tuple[str, float]:
        service_seconds = 0.0
        hard_wall = self._wall_budget(active_budget)
        while True:
            return_code = process.poll()
            now = time.monotonic()
            wall_seconds = now - started
            service_seconds = max(service_seconds, lease.controller_service_seconds)
            active_seconds = max(0.0, wall_seconds - service_seconds)
            if progress is not None:
                progress()
            if return_code is not None:
                return "completed", service_seconds
            if self.allocation_should_pause is not None and self.allocation_should_pause():
                return "allocation_paused", service_seconds
            if hard_wall is not None and wall_seconds >= hard_wall:
                return "hard_wall_timeout", service_seconds
            if self.enforce_active_budget and active_seconds >= active_budget:
                return "active_timeout", service_seconds
            time.sleep(
                min(
                    self.poll_interval_seconds,
                    hard_wall - wall_seconds if hard_wall is not None else math.inf,
                    active_budget - active_seconds if self.enforce_active_budget else math.inf,
                )
            )

    def _terminate_process_group(self, process: subprocess.Popen[bytes]) -> bool:
        pgid = process.pid
        if not _process_group_exists(pgid):
            process.poll()
            return True
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            process.poll()
            return True
        deadline = time.monotonic() + self.termination_grace_seconds
        while time.monotonic() < deadline:
            process.poll()
            if not _process_group_exists(pgid):
                return True
            time.sleep(min(self.poll_interval_seconds, deadline - time.monotonic()))
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=self.termination_grace_seconds)
        except subprocess.TimeoutExpired:
            return False
        deadline = time.monotonic() + self.termination_grace_seconds
        while time.monotonic() < deadline:
            if not _process_group_exists(pgid):
                return True
            time.sleep(min(self.poll_interval_seconds, deadline - time.monotonic()))
        return not _process_group_exists(pgid)

    def _terminate_remaining_process_group(
        self,
        process: subprocess.Popen[bytes],
    ) -> bool:
        if not _process_group_exists(process.pid):
            return True
        return self._terminate_process_group(process)

    def _read_response(
        self,
        response_path: Path,
        execution_id: str,
    ) -> dict[str, object] | None:
        if not response_path.exists():
            return None
        payload = _read_json_object(response_path, self.max_protocol_bytes)
        if payload.get("schema") != AGENT_WORKER_RESPONSE_SCHEMA:
            raise AgentExecutionError("Agent worker response has an unknown schema")
        if payload.get("execution_id") != execution_id:
            raise AgentExecutionError("Agent worker response execution_id does not match")
        status = payload.get("status")
        if status not in {"completed", "failed"}:
            raise AgentExecutionError("Agent worker response status is invalid")
        observations = payload.get("observations", [])
        if not isinstance(observations, list) or any(
            not isinstance(value, str) for value in observations
        ):
            raise AgentExecutionError("Agent worker response observations are invalid")
        error = payload.get("error")
        if error is not None and not isinstance(error, str):
            raise AgentExecutionError("Agent worker response error is invalid")
        return {
            "status": status,
            "observations": tuple(observations),
            "error": error,
        }

    def _load_nominations(
        self,
        nominations_path: Path,
        execution_id: str,
        program: CommandAgentProgram,
        context: AgentContext,
        workspace: Path,
    ) -> tuple[tuple[Artifact, ...], int]:
        from pde_agent_verifier_solver.sdk import Artifact

        artifacts: list[Artifact] = []
        rejected = 0
        policy = program.worker_settings.get("nomination_policy", "all")
        latest_only = policy == "latest"
        primary_only = policy == "single_primary_v1"
        selected_sequence = None
        if primary_only:
            try:
                selected_sequence = read_agent_primary_nomination(
                    nominations_path, execution_id, max_bytes=self.max_protocol_bytes)
            except Exception:
                return (), 1
        latest_body: dict[str, object] | None = None
        entry_count = 0
        for expected_sequence, path in enumerate(
            sorted(nominations_path.glob("nomination-*.json")),
            start=1,
        ):
            entry_count = expected_sequence
            try:
                expected_name = f"nomination-{expected_sequence:08d}.json"
                if path.name != expected_name:
                    raise AgentExecutionError(
                        "nomination journal sequence is not contiguous"
                    )
                payload = _read_json_object(path, self.max_protocol_bytes)
                if payload.get("schema") != AGENT_NOMINATION_SCHEMA:
                    raise AgentExecutionError("unknown nomination schema")
                if payload.get("execution_id") != execution_id:
                    raise AgentExecutionError("nomination execution_id does not match")
                if payload.get("sequence") != expected_sequence:
                    raise AgentExecutionError(
                        "nomination payload sequence does not match its journal entry"
                    )
                body = payload.get("payload")
                if not isinstance(body, dict):
                    raise AgentExecutionError("nomination payload must be an object")
                if primary_only:
                    if expected_sequence == selected_sequence:
                        latest_body = body
                elif latest_only:
                    # Decode only the Agent's final commitment. Decoding can freeze
                    # bundles or allocate candidate identities, so decoding history
                    # first and slicing the result would have observable side effects.
                    latest_body = body
                else:
                    artifact = program.nomination_decoder(body, context, workspace)
                    if not isinstance(artifact, Artifact):
                        raise TypeError("nomination_decoder must return Artifact")
                    artifacts.append(artifact)
            except Exception:
                rejected += 1
        if primary_only and entry_count and selected_sequence is None:
            # Saved versions alone do not authorize an implicit final choice.
            rejected += 1
        if (latest_only or primary_only) and latest_body is not None and not rejected:
            try:
                artifact = program.nomination_decoder(latest_body, context, workspace)
                if not isinstance(artifact, Artifact):
                    raise TypeError("nomination_decoder must return Artifact")
                artifacts.append(artifact)
            except Exception:
                # An invalid final commitment cannot silently revive an earlier one.
                rejected += 1
        return tuple(artifacts), rejected


def load_agent_worker_request(
    path: str | os.PathLike[str] | None = None,
    *,
    max_bytes: int | None = None,
) -> dict[str, object]:
    """Load and minimally validate a Controller-authored worker request."""

    request_path = Path(path or _required_environment(AGENT_REQUEST_ENV))
    if max_bytes is None:
        raw_limit = os.environ.get(
            AGENT_PROTOCOL_MAX_BYTES_ENV,
            str(8 * 1024 * 1024),
        )
        try:
            resolved_max_bytes = int(raw_limit)
        except ValueError as exc:
            raise AgentExecutionError(
                "Agent worker protocol size limit is invalid"
            ) from exc
    else:
        resolved_max_bytes = max_bytes
    if (
        isinstance(resolved_max_bytes, bool)
        or not isinstance(resolved_max_bytes, int)
        or resolved_max_bytes < 1
    ):
        raise AgentExecutionError("Agent worker protocol size limit is invalid")
    payload = _read_json_object(request_path, resolved_max_bytes)
    if payload.get("schema") != AGENT_WORKER_REQUEST_SCHEMA:
        raise AgentExecutionError("Agent worker request has an unknown schema")
    expected_execution_id = os.environ.get(AGENT_EXECUTION_ID_ENV)
    if expected_execution_id is not None and payload.get("execution_id") != expected_execution_id:
        raise AgentExecutionError("Agent worker request execution_id does not match")
    return payload


def submit_agent_nomination(
    payload: Mapping[str, object],
    *,
    sequence: int,
    directory: str | os.PathLike[str] | None = None,
    execution_id: str | None = None,
) -> Path:
    """Atomically commit one JSON nomination before the Agent deadline."""

    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
        raise ValueError("nomination sequence must be a positive integer")
    nominations = Path(directory or _required_environment(AGENT_NOMINATIONS_ENV))
    expected_execution_id = execution_id or _required_environment(AGENT_EXECUTION_ID_ENV)
    return _write_json_new(
        nominations / f"nomination-{sequence:08d}.json",
        {
            "schema": AGENT_NOMINATION_SCHEMA,
            "execution_id": expected_execution_id,
            "sequence": sequence,
            "payload": _json_value(dict(payload)),
        },
    )


def read_agent_primary_nomination(
    directory: Path, execution_id: str, *, max_bytes: int = 8 * 1024 * 1024,
) -> int | None:
    """Read an explicit choice bound to an existing nomination in this turn."""
    path = directory / "primary.json"
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    choice = _read_json_object(path, max_bytes)
    sequence = choice.get("sequence")
    if (set(choice) != {"schema", "execution_id", "sequence"}
            or choice.get("schema") != AGENT_PRIMARY_NOMINATION_SCHEMA
            or choice.get("execution_id") != execution_id
            or type(sequence) is not int or sequence < 1):
        raise AgentExecutionError("invalid explicit primary nomination")
    _validate_primary_target(directory, execution_id, sequence, max_bytes=max_bytes)
    return sequence


def _validate_primary_target(directory, execution_id, sequence, *, max_bytes):
    nominee = _read_json_object(directory / f"nomination-{sequence:08d}.json", max_bytes)
    if (nominee.get("schema") != AGENT_NOMINATION_SCHEMA
            or nominee.get("execution_id") != execution_id
            or type(nominee.get("sequence")) is not int
            or nominee["sequence"] != sequence
            or not isinstance(nominee.get("payload"), dict)):
        raise AgentExecutionError("primary target is not a nomination from this turn")


def select_agent_primary_nomination(
    sequence: int, *, directory: str | os.PathLike[str] | None = None,
    execution_id: str | None = None,
) -> Path:
    """Atomically select an already committed snapshot; ordinary submits do not select."""
    if type(sequence) is not int or sequence < 1:
        raise ValueError("primary sequence must be a positive integer")
    directory = Path(directory or _required_environment(AGENT_NOMINATIONS_ENV))
    execution_id = execution_id or _required_environment(AGENT_EXECUTION_ID_ENV)
    _validate_primary_target(directory, execution_id, sequence, max_bytes=8 * 1024 * 1024)
    path = directory / "primary.json"
    temporary = directory / f".primary-{uuid4().hex}.partial"
    try:
        _write_json_new(temporary, {
            "schema": AGENT_PRIMARY_NOMINATION_SCHEMA,
            "execution_id": execution_id, "sequence": sequence,
        })
        os.replace(temporary, path)
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_agent_worker_response(
    *,
    status: str = "completed",
    observations: Sequence[str] = (),
    error: str | None = None,
    path: str | os.PathLike[str] | None = None,
    execution_id: str | None = None,
) -> Path:
    """Atomically publish the final worker response (including zero nominations)."""

    if status not in {"completed", "failed"}:
        raise ValueError("worker response status must be completed or failed")
    observations = tuple(observations)
    if any(not isinstance(value, str) for value in observations):
        raise TypeError("worker observations must be strings")
    if error is not None and not isinstance(error, str):
        raise TypeError("worker response error must be a string or None")
    response_path = Path(path or _required_environment(AGENT_RESPONSE_ENV))
    expected_execution_id = execution_id or _required_environment(AGENT_EXECUTION_ID_ENV)
    return _write_json_new(
        response_path,
        {
            "schema": AGENT_WORKER_RESPONSE_SCHEMA,
            "execution_id": expected_execution_id,
            "status": status,
            "observations": list(observations),
            "error": error,
        },
    )


def write_agent_round_log(
    markdown: str,
) -> Path:
    """Atomically replace the Controller-managed log for this Agent turn."""

    if not isinstance(markdown, str):
        raise TypeError("Agent round log must be Markdown text")
    if not markdown.strip():
        raise ValueError("Agent round log cannot be empty")
    encoded = markdown.encode("utf-8")
    raw_limit = os.environ.get(
        AGENT_ROUND_LOG_MAX_BYTES_ENV,
        str(8 * 1024 * 1024),
    )
    try:
        max_bytes = int(raw_limit)
    except ValueError as exc:
        raise AgentExecutionError("Agent round log size limit is invalid") from exc
    if max_bytes < 1 or len(encoded) > max_bytes:
        raise AgentExecutionError("Agent round log exceeds size limit")
    round_log_path = Path(_required_environment(AGENT_ROUND_LOG_ENV))
    temporary = round_log_path.parent / (
        f".{round_log_path.name}.{uuid4().hex}.partial"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, round_log_path)
        directory_descriptor = os.open(round_log_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return round_log_path


def _native_recovery_observer(
    workspace: Path, emit: Callable[[str, Mapping[str, object]], None],
) -> Callable[[], None]:
    """Project the existing native retry audit into progress, never execution decisions."""
    previous = None
    next_read = 0.0

    def observe() -> None:
        nonlocal previous, next_read
        now = time.monotonic()
        if now < next_read:
            return
        next_read = now + 1.0
        try:
            record = _read_json_object(workspace / "codex-recovery.json", 1024 * 1024)
        except (AgentExecutionError, OSError, ValueError):
            return
        if record.get("schema") != "canonical_codex_same_session_recovery_v1":
            return
        attempt = record.get("current_attempt")
        if type(attempt) is not int or attempt < 1:
            return
        status = record.get("status")
        state = "recovering" if status == "retry" else "resumed" if status == "running" and attempt > 1 else None
        if state is None or previous == (attempt, state):
            return
        # No provider errors, file paths or conversation text enter the panel.
        try:
            emit(state, {"attempt": attempt})
        except Exception:
            # A display subscriber must not interrupt the research process.
            return
        previous = (attempt, state)

    return observe


def _positive_seconds(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _validated_gpu_devices(
    values: Sequence[str],
    name: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of device identifiers")
    raw_devices = tuple(values)
    if any(
        not isinstance(device, str)
        or not device.strip()
        or "," in device
        or "\x00" in device
        for device in raw_devices
    ):
        raise ValueError(f"{name} must contain nonempty identifiers")
    devices = tuple(device.strip() for device in raw_devices)
    if len(devices) != len(set(devices)):
        raise ValueError(f"{name} cannot repeat a device")
    return devices


def _stage_public_problem_bundle(bundle, workspace: Path) -> dict[str, object] | None:
    """Materialize only immutable Agent-public resources for one turn."""

    if bundle is None:
        return None
    from pde_agent_verifier_solver.sdk import PublicProblemBundle

    if not isinstance(bundle, PublicProblemBundle):
        raise TypeError("AgentContext public_problem_bundle is invalid")
    destination = bundle.materialize(workspace / "public_resources")
    # Files are already 0444. Remove directory write bits too so the staged
    # tree is conventionally read-only inside the execution workspace. This is
    # an integrity boundary, not a claim of hostile-code OS sandboxing.
    directories = sorted(
        (path for path in destination.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in (*directories, destination):
        os.chmod(directory, 0o555)
    relative = destination.relative_to(workspace).as_posix()
    return {
        "resources_path": relative,
        "manifest_path": f"{relative}/manifest.json",
        "manifest": bundle.manifest(),
    }


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("protocol JSON cannot contain non-finite numbers")
        return value
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Mapping):
        converted: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("protocol JSON object keys must be strings")
            converted[key] = _json_value(item)
        return converted
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise TypeError(f"value of type {type(value).__name__} is not protocol JSON")


def _json_bytes(payload: object) -> bytes:
    return json.dumps(
        _json_value(payload),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_json_new(path: Path, payload: object) -> Path:
    """Atomically create ``path`` without replacing an existing commit."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = _json_bytes(payload)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.partial"
    file_descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


def _write_text_new(path: Path, value: str) -> Path:
    """Atomically create a UTF-8 text file without replacing an existing file."""

    if not isinstance(value, str):
        raise TypeError("text file contents must be text")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = value.encode("utf-8")
    temporary = path.parent / f".{path.name}.{uuid4().hex}.partial"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


def _read_text_file(path: Path, max_bytes: int) -> str:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError as exc:
        raise AgentExecutionError(f"text file is missing: {path.name}") from exc
    except OSError as exc:
        raise AgentExecutionError(f"text file cannot be opened: {path.name}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise AgentExecutionError(
                f"text path is not a regular file: {path.name}"
            )
        if metadata.st_size > max_bytes:
            raise AgentExecutionError(f"text file exceeds size limit: {path.name}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            encoded = handle.read(max_bytes + 1)
        if len(encoded) > max_bytes:
            raise AgentExecutionError(f"text file exceeds size limit: {path.name}")
        return encoded.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise AgentExecutionError(f"text file is invalid UTF-8: {path.name}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_json_object(path: Path, max_bytes: int) -> dict[str, object]:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError as exc:
        raise AgentExecutionError(f"protocol file is missing: {path.name}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise AgentExecutionError(
                f"protocol path is not a regular file: {path.name}"
            )
        if metadata.st_size > max_bytes:
            raise AgentExecutionError(
                f"protocol file exceeds size limit: {path.name}"
            )
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            encoded = handle.read(max_bytes + 1)
        if len(encoded) > max_bytes:
            raise AgentExecutionError(
                f"protocol file exceeds size limit: {path.name}"
            )
        payload = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AgentExecutionError(f"protocol file is invalid JSON: {path.name}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(payload, dict):
        raise AgentExecutionError(f"protocol file must contain an object: {path.name}")
    return payload


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise AgentExecutionError(f"required Agent worker environment is missing: {name}")
    return value


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


__all__ = [
    "AGENT_CAMPAIGN_MEMORY_ENV",
    "AGENT_CANDIDATE_HISTORY_ENV",
    "AGENT_CAPABILITY_SCHEMA",
    "AGENT_EXECUTION_AUDIT_SCHEMA",
    "AGENT_EXECUTION_ID_ENV",
    "AGENT_DEADLINE_MONOTONIC_ENV",
    "AGENT_NOMINATION_SCHEMA",
    "AGENT_NOMINATIONS_ENV",
    "AGENT_PROTOCOL_MAX_BYTES_ENV",
    "AGENT_REQUEST_ENV",
    "AGENT_RESPONSE_ENV",
    "AGENT_ROUND_LOG_ENV",
    "AGENT_ROUND_LOG_MAX_BYTES_ENV",
    "AGENT_WORKER_REQUEST_SCHEMA",
    "AGENT_WORKER_RESPONSE_SCHEMA",
    "AGENT_WORKSPACE_ENV",
    "AgentCapabilityFacade",
    "AgentCapabilityLease",
    "AgentBatchReservation",
    "AgentExecutionAudit",
    "AgentExecutionError",
    "AgentInfrastructureError",
    "AgentExecutionRequest",
    "AgentExecutionResult",
    "AgentExecutionService",
    "AgentExecutionTimeout",
    "AgentGPUConflict",
    "BatchAgentExecutionService",
    "CommandAgentProgram",
    "InProcessAgentExecutionService",
    "IncumbentEncoder",
    "NominationDecoder",
    "SubprocessAgentExecutionService",
    "UnavailableAgentCapabilityFacade",
    "execute_agent_batch",
    "execute_reserved_agent_batch",
    "load_agent_worker_request",
    "reserve_agent_batch",
    "submit_agent_nomination",
    "write_agent_round_log",
    "validate_agent_batch",
    "write_agent_worker_response",
]
