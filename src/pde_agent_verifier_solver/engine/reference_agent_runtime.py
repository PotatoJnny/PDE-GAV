"""PDE-neutral filesystem/subprocess adapter for reference macro Agent runs.

This module is deliberately not an isolation boundary. It gives a local Agent
subprocess a public workspace and a budgeted loopback RPC backed by the existing
DevelopmentMacroEvaluator. Trial and finalized candidate modules execute in the
controller process. The Agent subprocess also shares the controller node and
operating-system user. Consequently every API here remains
reference_in_process evidence and can never enable formal_isolated claims.

The formal candidate bundle ABI is a Python entrypoint exporting
prepare_solution(case, solve_budget=None), whose returned handle implements
handle.evaluate(points_query) -> AnswerView. Legacy predict_macros remains an
explicitly timing-unassured compatibility path. Bundles may also contain helper
modules and relative assets. PDE-specific case generation, queries, macro
definitions, quadrature, gates, and metrics remain outside this module.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from math import isfinite
from numbers import Complex, Real
import os
from pathlib import Path
from queue import Queue
import re
import secrets
import signal
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from threading import Condition, Lock, Thread
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    CandidateGuidanceSnapshot,
    CandidateGuidedSample,
    CandidateGuidedSampleBatch,
    Case,
    CaseSource,
    CaseVisibility,
    GuidedSampleSpec,
    PublicProblemBundle,
    QueryKind,
)
from pde_agent_verifier_solver.sdk.candidate_bundle import (
    CANDIDATE_BUNDLE_MANIFEST,
    CandidateBundle,
    CandidateBundleError,
    CandidateBundleLimits,
    freeze_candidate_bundle,
    inspect_candidate_bundle,
)
from pde_agent_verifier_solver.sdk.execution_boundary import (
    AgentBundleSubmission,
    ControllerOwnedBundleAttestor,
    ExecutionAssurance,
    candidate_ref_from_attestation,
    require_execution_assurance,
)
from pde_agent_verifier_solver.sdk.macro_loop import (
    CandidateRef,
    CandidateSubmission,
)

from .macro_closed_loop import (
    MacroAgentContext,
    MacroAgentOutcome,
    SearchLeaderboardEntry,
)
from .guidance_controller import DevelopmentGuidanceOracle
from .development_session import (
    DevelopmentCreditExceeded,
    DevelopmentRelease,
    DevelopmentSession,
    DevelopmentSessionError,
)
from .macro_evaluator import (
    CaseCreditExceeded,
    DevelopmentMacroEvaluator,
    DevelopmentMacroResponse,
)
from .candidate_case import candidate_facing_case


REFERENCE_CONTEXT_SCHEMA = "reference-macro-agent-context-v1"
REFERENCE_SUBMISSION_SCHEMA = "reference-macro-agent-submission-v1"
REFERENCE_EVALUATION_REQUEST_SCHEMA = "reference-macro-evaluation-request-v1"
REFERENCE_EVALUATION_RESPONSE_SCHEMA = "reference-macro-evaluation-response-v1"
REFERENCE_EVALUATION_ERROR_SCHEMA = "reference-macro-evaluation-error-v1"
REFERENCE_SCREEN_REQUEST_SCHEMA = "reference-macro-screen-request-v1"
REFERENCE_SCREEN_RESPONSE_SCHEMA = "macro-development-screen-response-v2"
REFERENCE_SCREEN_SUBMIT_REQUEST_SCHEMA = (
    "reference-macro-screen-submit-request-v1"
)
REFERENCE_SCREEN_SUBMIT_RESPONSE_SCHEMA = (
    "reference-macro-screen-submit-response-v1"
)
REFERENCE_SCREEN_POLL_REQUEST_SCHEMA = "reference-macro-screen-poll-request-v1"
REFERENCE_SCREEN_POLL_RESPONSE_SCHEMA = "reference-macro-screen-poll-response-v1"
REFERENCE_RELEASE_REQUEST_SCHEMA = "reference-macro-release-request-v1"
REFERENCE_RELEASE_RESPONSE_SCHEMA = "reference-macro-release-response-v1"
REFERENCE_NUMERICAL_ORACLE_REQUEST_SCHEMA = (
    "reference-numerical-oracle-request-v1"
)
REFERENCE_NUMERICAL_ORACLE_RESPONSE_SCHEMA = (
    "reference-numerical-oracle-response-v1"
)
REFERENCE_GUIDANCE_ORACLE_REQUEST_SCHEMA = (
    "reference-guidance-oracle-request-v1"
)
REFERENCE_GUIDANCE_ORACLE_RESPONSE_SCHEMA = (
    "reference-guidance-oracle-response-v2"
)
REFERENCE_GUIDANCE_WORKER_CAPABILITY_SCHEMA = (
    "reference-guidance-worker-capability-v2"
)
REFERENCE_RELEASE_MANIFEST_SCHEMA = "reference-macro-release-manifest-v1"
REFERENCE_DEVELOPMENT_LEDGER_SCHEMA = "reference-macro-development-ledger-v1"

_PUBLIC_SCREEN_SESSION_ERRORS = frozenset(
    {
        "incumbent cannot screen against itself",
        "screen candidate parent must be the session incumbent",
        "screen evidence must advance by exactly one configured stage",
        "candidate did not clear the continuation mean-ratio gate",
    }
)

_SAFE_PRIVATE_SCREEN_EXCEPTION_TYPES = frozenset(
    {
        ArithmeticError,
        IndexError,
        KeyError,
        MemoryError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    }
)

_EVALUATOR_CAPABILITY_ACTIVE = "active"
_EVALUATOR_CAPABILITY_EXPIRED = "expired"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_BUNDLE_BYTES = 512 * 1024 * 1024
_CANDIDATE_ARCHIVE_MAGIC = b"PDE-AGENT-CANDIDATE-ZIP-V1\0"
_CANDIDATE_IMPORT_LOCK = Lock()
_MAX_RPC_BYTES = 2 * 1024 * 1024
_MAX_CONTEXT_JSON_BYTES = 8 * 1024 * 1024
_MAX_SUBMISSION_BYTES = 64 * 1024
_MAX_CANDIDATE_ID_CHARS = 256
_MAX_OBSERVATIONS = 64
_MAX_OBSERVATION_CHARS = 4096
_MAX_PATH_CHARS = 4096
_MAX_RELEASE_ROOTS_PER_REQUEST = 32
_MAX_ASYNC_SCREEN_INFLIGHT = 2
_REFERENCE_AGENT_WAIT_POLL_SECONDS = 0.1
_SAFE_INHERITED_ENVIRONMENT = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TMPDIR",
)
_RESERVED_ENVIRONMENT = frozenset(
    {
        "PDE_MACRO_WORKSPACE",
        "PDE_MACRO_CONTEXT",
        "PDE_MACRO_PROMPT",
        "PDE_MACRO_SUBMISSION",
    }
)
_REFERENCE_LIMITATIONS = (
    "Agent subprocess shares the controller node and operating-system user",
    "loopback evaluator RPC is a workflow boundary, not a security boundary",
    "trial and finalized candidate modules execute in the controller process",
    "read-only file modes do not protect against a hostile same-user process",
    "a client timeout cannot stop candidate code already running in-process",
)


class ReferenceAgentRuntimeError(RuntimeError):
    """A reference workspace, bundle, launcher, or submission is invalid."""


class ReferenceEvaluatorClientError(ReferenceAgentRuntimeError):
    """A sanitized error returned by the local development evaluator."""

    def __init__(self, code: str, message: str, *, http_status: int | None = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.http_status = http_status


class _PrivateScreenExecutionError(RuntimeError):
    """An internal oracle failure whose details must not cross the RPC."""

    def __init__(self, exception: BaseException) -> None:
        exception_type = type(exception)
        self.exception_type = (
            exception_type.__name__
            if exception_type in _SAFE_PRIVATE_SCREEN_EXCEPTION_TYPES
            else "InternalError"
        )
        super().__init__(self.exception_type)


class _ReferenceAsyncScreenQueueFull(RuntimeError):
    """Raised before an async submit would exceed its bounded inflight cap."""


class _ReferenceAsyncScreenTaskFailed(RuntimeError):
    def __init__(self, error: Mapping[str, object]) -> None:
        self.code = str(error.get("code", "screen_failed"))
        self.message = str(
            error.get("message", "local candidate screen failed")
        )
        status = error.get("http_status")
        self.http_status = status if isinstance(status, int) else 500
        super().__init__(self.code)


class ReferenceAgentLaunchError(ReferenceAgentRuntimeError):
    """The injected Agent launcher did not complete successfully."""

    def __init__(
        self,
        message: str,
        *,
        audit: "ReferenceAgentLaunchAudit | None" = None,
    ) -> None:
        super().__init__(message)
        self.audit = audit


@dataclass(frozen=True)
class ReferenceAgentLaunchAudit:
    """Controller-private timing/failure record for one Agent subprocess."""

    status: str
    returncode: int | None
    wall_time_seconds: float
    controller_service_seconds: float
    controller_service_provisional_seconds: float
    agent_active_wall_seconds: float
    active_timeout_seconds: float | None
    hard_timeout_seconds: float | None
    observed_monotonic_seconds: float = field(repr=False, compare=False)

    def payload(self) -> Mapping[str, object]:
        return {
            "schema": "reference-agent-launch-audit-v1",
            "status": self.status,
            "returncode": self.returncode,
            "clock": "monotonic",
            "wall_time_seconds": self.wall_time_seconds,
            "controller_service_seconds": self.controller_service_seconds,
            "controller_service_committed_seconds": (
                self.controller_service_seconds
                - self.controller_service_provisional_seconds
            ),
            "controller_service_provisional_seconds": (
                self.controller_service_provisional_seconds
            ),
            "agent_active_wall_seconds": self.agent_active_wall_seconds,
            "active_timeout_seconds": self.active_timeout_seconds,
            "hard_timeout_seconds": self.hard_timeout_seconds,
            "controller_service_time_excluded_from_active_budget": True,
            "controller_service_accounting_finalized": (
                self.controller_service_provisional_seconds == 0.0
            ),
            "in_process_evaluator_preemption_guaranteed": False,
        }


class ReferenceControllerServiceClock:
    """Thread-safe union clock for one serialized local evaluator service.

    A service interval is provisional while it is in flight. The caller
    commits it only when the request consumed development credits; rejected,
    cached, or failed requests therefore cannot buy additional Agent time.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._committed_intervals: list[tuple[float, float]] = []
        self._committed_seconds = 0.0
        self._active_token: object | None = None
        self._active_started: float | None = None

    def begin_service(self) -> object:
        now = time.monotonic()
        token = object()
        with self._lock:
            if self._active_token is not None:
                raise RuntimeError("controller service intervals cannot overlap")
            self._active_token = token
            self._active_started = now
        return token

    def end_service(self, token: object, *, credit: bool) -> None:
        if not isinstance(credit, bool):
            raise TypeError("controller service credit decision must be bool")
        now = time.monotonic()
        with self._lock:
            if token is not self._active_token or self._active_started is None:
                raise RuntimeError("controller service token is not active")
            elapsed = now - self._active_started
            if elapsed < 0.0 or not isfinite(elapsed):
                raise RuntimeError("controller service timing is inconsistent")
            if credit:
                self._committed_intervals.append((self._active_started, now))
                self._committed_seconds += elapsed
            self._active_token = None
            self._active_started = None

    def accounting_seconds(
        self,
        *,
        now: float | None = None,
    ) -> tuple[float, float]:
        """Return committed and currently provisional service through ``now``."""

        current = time.monotonic() if now is None else float(now)
        if not isfinite(current):
            raise ValueError("controller service clock needs finite time")
        with self._lock:
            if (
                not self._committed_intervals
                or current >= self._committed_intervals[-1][1]
            ):
                committed = self._committed_seconds
            else:
                committed = sum(
                    max(0.0, min(ended, current) - started)
                    for started, ended in self._committed_intervals
                    if current > started
                )
            provisional = 0.0
            if self._active_started is not None and current > self._active_started:
                provisional = current - self._active_started
            return committed, provisional

    def creditable_seconds(self, *, now: float | None = None) -> float:
        committed, provisional = self.accounting_seconds(now=now)
        return committed + provisional


# Compatibility names are exact aliases, not a second RPC-specific DTO.
ReferenceGuidanceSnapshotIdentity = CandidateGuidanceSnapshot
ReferenceGuidedSample = CandidateGuidedSample
ReferenceGuidedSampleBatch = CandidateGuidedSampleBatch


@dataclass(frozen=True)
class ReferencePublicWorkspace:
    """Agent-readable reconstruction of one public development workspace."""

    context_path: Path
    run_id: str
    round_index: int
    agent_id: str
    prompt: str
    cases: tuple[Case, ...]
    queries: tuple[AnswerQuery, ...]
    completed_answers: tuple[AnswerView | None, ...]
    cohort_id: str
    cohort_sha256: str
    contract_id: str
    incumbent_path: Path
    incumbent_sha256: str
    evaluator_endpoint: str
    remaining_case_credits: int | None
    remaining_root_credits: int | None
    screen_timeout_seconds: float | None = None
    release_endpoint: str | None = None
    remaining_release_root_credits: int | None = None
    max_release_roots_per_request: int | None = None
    screen_submit_endpoint: str | None = None
    screen_poll_endpoint: str | None = None
    max_async_screen_inflight: int | None = None
    # Legacy read-only compatibility. New workspaces never publish or call it.
    finish_endpoint: str | None = None
    search_contract: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    prior_leaderboard: tuple[Mapping[str, object], ...] = ()
    prior_candidates: tuple[Mapping[str, object], ...] = ()
    public_resources_path: Path | None = None
    public_problem_manifest_path: Path | None = None
    public_problem_manifest: Mapping[str, object] | None = None
    numerical_oracle_endpoint: str | None = None
    equation_oracle_methods: tuple[str, ...] = ()
    development_solution_oracle_methods: tuple[str, ...] = ()
    guidance_oracle_endpoint: str | None = None
    guidance_snapshot_sha256: str | None = None
    guidance_solution_methods: tuple[str, ...] = ()
    guidance_unlimited_fresh_inference: bool = False
    open_search: bool = False
    screen_root_request_capacity: int | None = None
    release_root_request_capacity: int | None = None
    recommended_screen_root_counts: tuple[int, ...] = ()
    observational_usage: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True)
class ReferenceCompletedAnswerRelease:
    """One immutable, independently loaded completed-answer release."""

    manifest_path: Path
    release_index: int
    root_count: int
    member_case_count: int
    cohort_id: str
    cohort_sha256: str
    contract_id: str
    cases: tuple[Case, ...]
    queries: tuple[AnswerQuery, ...]
    answers: tuple[AnswerView, ...]
    charged_root_credits: int
    cumulative_charged_root_credits: int
    remaining_root_credits: int


@dataclass
class _ReferenceAsyncScreenTask:
    task_id: str
    candidate: CandidateSubmission
    requested_root_count: int
    status: str = "queued"
    response: Mapping[str, object] | None = None
    error: Mapping[str, object] | None = None


@dataclass(frozen=True)
class ReferenceAgentLaunchRequest:
    """Public paths plus controller-only lifecycle hooks for one launch."""

    workspace: Path
    context_path: Path
    prompt_path: Path
    submission_path: Path
    controller_service_clock: ReferenceControllerServiceClock | None = None
    revoke_evaluator_capability: Callable[[], None] | None = None


class ReferenceAgentLauncher(Protocol):
    """Injected launcher for Codex, another Agent, or a deterministic test."""

    def launch(
        self, request: ReferenceAgentLaunchRequest
    ) -> ReferenceAgentLaunchAudit | None: ...


@dataclass(frozen=True)
class SubprocessReferenceAgentLauncher:
    """Run an arbitrary injected command without shell interpretation.

    Every launch owns a new process group. A hard timeout terminates that
    whole group, rather than only the command's immediate process.
    """

    command: tuple[str, ...]
    environment: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float | None = None
    hard_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        command = tuple(self.command)
        if not command or any(
            not isinstance(part, str) or not part for part in command
        ):
            raise ValueError("reference Agent command needs nonempty argv strings")
        environment = dict(self.environment)
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            for key, value in environment.items()
        ):
            raise ValueError("reference Agent environment must contain strings")
        overlap = _RESERVED_ENVIRONMENT.intersection(environment)
        if overlap:
            raise ValueError(
                "reference Agent environment cannot override reserved names: "
                f"{sorted(overlap)}"
            )
        if self.timeout_seconds is not None and (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, Real)
            or not isfinite(float(self.timeout_seconds))
            or float(self.timeout_seconds) <= 0.0
        ):
            raise ValueError("reference Agent timeout must be finite and positive")
        if self.hard_timeout_seconds is not None and (
            isinstance(self.hard_timeout_seconds, bool)
            or not isinstance(self.hard_timeout_seconds, Real)
            or not isfinite(float(self.hard_timeout_seconds))
            or float(self.hard_timeout_seconds) <= 0.0
        ):
            raise ValueError(
                "reference Agent hard timeout must be finite and positive"
            )
        if self.timeout_seconds is None and self.hard_timeout_seconds is not None:
            raise ValueError("a hard timeout requires an active-time timeout")
        if (
            self.timeout_seconds is not None
            and self.hard_timeout_seconds is not None
            and float(self.hard_timeout_seconds) < float(self.timeout_seconds)
        ):
            raise ValueError(
                "reference Agent hard timeout cannot be below active timeout"
            )
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "environment", MappingProxyType(environment))

    def launch(self, request: ReferenceAgentLaunchRequest) -> ReferenceAgentLaunchAudit:
        if not isinstance(request, ReferenceAgentLaunchRequest):
            raise TypeError("reference Agent launcher needs a launch request")
        service_clock = request.controller_service_clock
        if service_clock is not None and not isinstance(
            service_clock, ReferenceControllerServiceClock
        ):
            raise TypeError("reference Agent request has an invalid service clock")
        revoke_capability = request.revoke_evaluator_capability
        if revoke_capability is not None and not callable(revoke_capability):
            raise TypeError("reference Agent request has an invalid revoke callback")

        capability_revoked = False

        def revoke() -> None:
            nonlocal capability_revoked
            if revoke_capability is not None and not capability_revoked:
                revoke_capability()
                capability_revoked = True

        environment = {
            key: os.environ[key]
            for key in _SAFE_INHERITED_ENVIRONMENT
            if key in os.environ
        }
        environment.update(self.environment)
        environment.update(
            {
                "PDE_MACRO_WORKSPACE": str(request.workspace),
                "PDE_MACRO_CONTEXT": str(request.context_path),
                "PDE_MACRO_PROMPT": str(request.prompt_path),
                "PDE_MACRO_SUBMISSION": str(request.submission_path),
            }
        )
        stdout_path = request.workspace / "agent.stdout.log"
        stderr_path = request.workspace / "agent.stderr.log"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        stdout_fd = os.open(stdout_path, flags, 0o600)
        try:
            stderr_fd = os.open(stderr_path, flags, 0o600)
        except BaseException:
            os.close(stdout_fd)
            raise
        started = time.monotonic()
        try:
            process = subprocess.Popen(
                self.command,
                cwd=request.workspace,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_fd,
                stderr=stderr_fd,
                start_new_session=True,
            )
        finally:
            os.close(stdout_fd)
            os.close(stderr_fd)
        active_timeout = (
            None if self.timeout_seconds is None else float(self.timeout_seconds)
        )
        hard_timeout = (
            float(self.hard_timeout_seconds)
            if self.hard_timeout_seconds is not None
            else active_timeout
        )
        try:
            returncode, audit = _wait_for_reference_agent(
                process,
                started=started,
                active_timeout_seconds=active_timeout,
                hard_timeout_seconds=hard_timeout,
                service_clock=service_clock,
            )
        except _ReferenceAgentDeadlineExpired as exc:
            revoke()
            _terminate_reference_agent_process_group(process)
            raise ReferenceAgentLaunchError(
                (
                    "reference Agent subprocess exceeded its hard wall time budget"
                    if exc.audit.status == "hard_wall_timeout"
                    else "reference Agent subprocess exceeded its time budget "
                    "(active-time budget)"
                ),
                audit=exc.audit,
            ) from exc
        except BaseException:
            revoke()
            _terminate_reference_agent_process_group(process)
            raise
        revoke()
        if _reference_agent_process_group_exists(process.pid):
            _terminate_reference_agent_process_group(process)
        if returncode != 0:
            raise ReferenceAgentLaunchError(
                "reference Agent subprocess exited with code "
                f"{returncode}",
                audit=audit,
            )
        return audit


class _ReferenceAgentDeadlineExpired(RuntimeError):
    def __init__(self, audit: ReferenceAgentLaunchAudit) -> None:
        super().__init__(audit.status)
        self.audit = audit


def _reference_agent_timing(
    *,
    started: float,
    current: float,
    service_clock: ReferenceControllerServiceClock | None,
) -> tuple[float, float, float, float]:
    wall = current - started
    if wall < 0.0 or not isfinite(wall):
        raise RuntimeError("reference Agent subprocess timing is inconsistent")
    committed, provisional = (
        (0.0, 0.0)
        if service_clock is None
        else service_clock.accounting_seconds(now=current)
    )
    service = committed + provisional
    active = max(0.0, wall - service)
    return wall, service, provisional, active


def _launch_audit(
    *,
    status: str,
    returncode: int | None,
    started: float,
    current: float,
    active_timeout_seconds: float | None,
    hard_timeout_seconds: float | None,
    service_clock: ReferenceControllerServiceClock | None,
) -> ReferenceAgentLaunchAudit:
    wall, service, provisional, active = _reference_agent_timing(
        started=started,
        current=current,
        service_clock=service_clock,
    )
    return ReferenceAgentLaunchAudit(
        status=status,
        returncode=returncode,
        wall_time_seconds=wall,
        controller_service_seconds=service,
        controller_service_provisional_seconds=provisional,
        agent_active_wall_seconds=active,
        active_timeout_seconds=active_timeout_seconds,
        hard_timeout_seconds=hard_timeout_seconds,
        observed_monotonic_seconds=current,
    )


def _reference_agent_deadline_status(
    audit: ReferenceAgentLaunchAudit,
    *,
    active_timeout_seconds: float | None,
    hard_timeout_seconds: float | None,
) -> str | None:
    hard_expired = (
        hard_timeout_seconds is not None
        and audit.wall_time_seconds >= hard_timeout_seconds
    )
    active_expired = (
        active_timeout_seconds is not None
        and audit.agent_active_wall_seconds >= active_timeout_seconds
    )
    if hard_expired and (
        active_timeout_seconds is None
        or hard_timeout_seconds != active_timeout_seconds
    ):
        return "hard_wall_timeout"
    if active_expired:
        return "active_time_timeout"
    if hard_expired:
        return "hard_wall_timeout"
    return None


def _finalize_reference_agent_launch_audit(
    audit: ReferenceAgentLaunchAudit,
    service_clock: ReferenceControllerServiceClock | None,
) -> ReferenceAgentLaunchAudit:
    """Resolve provisional service after the RPC has drained its handlers."""

    if service_clock is None:
        return audit
    committed, provisional = service_clock.accounting_seconds(
        now=audit.observed_monotonic_seconds
    )
    if provisional != 0.0:
        raise RuntimeError("controller service clock was not drained")
    finalized = replace(
        audit,
        controller_service_seconds=committed,
        controller_service_provisional_seconds=0.0,
        agent_active_wall_seconds=max(0.0, audit.wall_time_seconds - committed),
    )
    deadline_status = _reference_agent_deadline_status(
        finalized,
        active_timeout_seconds=finalized.active_timeout_seconds,
        hard_timeout_seconds=finalized.hard_timeout_seconds,
    )
    return (
        finalized
        if deadline_status is None
        else replace(finalized, status=deadline_status)
    )


def _wait_for_reference_agent(
    process: subprocess.Popen[bytes],
    *,
    started: float,
    active_timeout_seconds: float | None,
    hard_timeout_seconds: float | None,
    service_clock: ReferenceControllerServiceClock | None,
) -> tuple[int, ReferenceAgentLaunchAudit]:
    while True:
        returncode = process.poll()
        now = time.monotonic()
        audit = _launch_audit(
            status="running",
            returncode=returncode,
            started=started,
            current=now,
            active_timeout_seconds=active_timeout_seconds,
            hard_timeout_seconds=hard_timeout_seconds,
            service_clock=service_clock,
        )
        deadline_status = _reference_agent_deadline_status(
            audit,
            active_timeout_seconds=active_timeout_seconds,
            hard_timeout_seconds=hard_timeout_seconds,
        )
        if deadline_status is not None:
            raise _ReferenceAgentDeadlineExpired(
                replace(audit, status=deadline_status)
            )
        if returncode is not None:
            status = "completed" if returncode == 0 else "nonzero_exit"
            return returncode, replace(audit, status=status)

        remaining = [_REFERENCE_AGENT_WAIT_POLL_SECONDS]
        if hard_timeout_seconds is not None:
            remaining.append(
                max(0.0, hard_timeout_seconds - audit.wall_time_seconds)
            )
        if active_timeout_seconds is not None:
            remaining.append(
                max(
                    0.0,
                    active_timeout_seconds - audit.agent_active_wall_seconds,
                )
            )
        wait_seconds = min(remaining)
        if wait_seconds <= 0.0:
            continue
        try:
            returncode = process.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired:
            continue

        finished = time.monotonic()
        finished_audit = _launch_audit(
            status="completed" if returncode == 0 else "nonzero_exit",
            returncode=returncode,
            started=started,
            current=finished,
            active_timeout_seconds=active_timeout_seconds,
            hard_timeout_seconds=hard_timeout_seconds,
            service_clock=service_clock,
        )
        deadline_status = _reference_agent_deadline_status(
            finished_audit,
            active_timeout_seconds=active_timeout_seconds,
            hard_timeout_seconds=hard_timeout_seconds,
        )
        if deadline_status is not None:
            raise _ReferenceAgentDeadlineExpired(
                replace(finished_audit, status=deadline_status)
            )
        return returncode, finished_audit


def _terminate_reference_agent_process_group(
    process: subprocess.Popen[bytes],
    *,
    grace_seconds: float = 5.0,
) -> None:
    """Terminate a launch-owned process group, escalating after a short grace."""

    process_group_id = process.pid
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while _reference_agent_process_group_exists(process_group_id):
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        if process.poll() is None:
            try:
                process.wait(timeout=min(0.05, remaining))
            except subprocess.TimeoutExpired:
                pass
        else:
            time.sleep(min(0.05, remaining))
    if not _reference_agent_process_group_exists(process_group_id):
        return
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        # SIGKILL cannot be handled. Any remaining group members have been
        # signalled even if an unrelated parent has not reaped them yet.
        pass


def _reference_agent_process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class ReferenceFilesystemBundleAttestor(ControllerOwnedBundleAttestor):
    """Best-effort immutable copy for Python candidate bundles.

    Single-file identity uses exact source bytes; multi-file identity uses a
    canonical archive over the entrypoint and declared assets. A controller
    copy is owner-readable and verified whenever loaded. This is deterministic
    provenance, not same-user tamper resistance.
    """

    def __init__(
        self,
        storage_root: Path | str,
        *,
        max_bundle_bytes: int = _MAX_BUNDLE_BYTES,
    ) -> None:
        if (
            isinstance(max_bundle_bytes, bool)
            or not isinstance(max_bundle_bytes, int)
            or max_bundle_bytes <= 0
        ):
            raise ValueError("max_bundle_bytes must be a positive integer")
        self.storage_root = Path(storage_root).resolve()
        _ensure_private_directory(self.storage_root)
        self.max_bundle_bytes = max_bundle_bytes

    @property
    def attestor_id(self) -> str:
        return "reference-filesystem-candidate-bundle-v2"

    def _read_untrusted_bundle(self, bundle_uri: str) -> bytes:
        return _read_candidate_bundle_payload(
            _local_bundle_path(bundle_uri),
            max_bytes=self.max_bundle_bytes,
        )

    def _seal_verified_bundle(self, bundle_sha256: str, payload: bytes) -> str:
        if payload.startswith(_CANDIDATE_ARCHIVE_MAGIC):
            return _seal_reference_candidate_archive(
                self.storage_root, bundle_sha256, payload, self.max_bundle_bytes
            )
        destination = self.storage_root / f"{bundle_sha256}.py"
        try:
            descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o400,
            )
        except FileExistsError:
            existing = _read_python_bundle(
                destination,
                max_bytes=self.max_bundle_bytes,
            )
            if existing != payload:
                raise ReferenceAgentRuntimeError(
                    "digest-named sealed bundle contains different bytes"
                )
        else:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(destination, 0o400)
        return str(destination)


class _LoadedPythonMacroProgram:
    def __init__(
        self,
        predict: Callable[..., AnswerView] | None,
        prepare_solution: Callable[..., object] | None = None,
    ) -> None:
        if predict is None and prepare_solution is None:
            raise TypeError("candidate has no supported solution ABI")
        self._predict = predict
        self._prepare_solution = prepare_solution
        self._controller_loaded_two_phase_abi = (
            prepare_solution is not None
        )

    def predict_macros(
        self,
        case: Case,
        query: AnswerQuery,
        budget: float | None = None,
    ) -> AnswerView:
        if self._predict is None:
            raise TypeError(
                "candidate does not export legacy callable predict_macros"
            )
        return self._predict(case, query, budget=budget)

    def __getattr__(self, name: str) -> object:
        # Defining prepare_solution on the class would make every legacy bundle
        # look formal to the evaluator's callable capability check.
        if name != "prepare_solution" or self._prepare_solution is None:
            raise AttributeError(name)
        return self._prepare_solution


class _LazyPythonMacroProgram:
    """Delay module execution until the evaluator has accepted the request."""

    _deferred_solution_abi = True

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str | None = None,
    ) -> None:
        self._path = path
        self._expected_sha256 = expected_sha256
        self._program: _LoadedPythonMacroProgram | None = None
        self._lock = Lock()

    def _loaded(self) -> _LoadedPythonMacroProgram:
        with self._lock:
            if self._program is None:
                self._program = load_python_macro_candidate(
                    self._path,
                    expected_sha256=self._expected_sha256,
                )
            return self._program

    def predict_macros(
        self,
        case: Case,
        query: AnswerQuery,
        budget: float | None = None,
    ) -> AnswerView:
        return self._loaded().predict_macros(case, query, budget=budget)

    def __getattr__(self, name: str) -> object:
        if name != "prepare_solution":
            raise AttributeError(name)
        try:
            return getattr(self._loaded(), name)
        except AttributeError:
            raise AttributeError(name) from None


def python_bundle_sha256(bundle_uri: Path | str) -> str:
    """Return the controller identity of a valid Python candidate bundle."""

    payload = _read_candidate_bundle_payload(
        _local_bundle_path(str(bundle_uri)),
        max_bytes=_MAX_BUNDLE_BYTES,
    )
    return hashlib.sha256(payload).hexdigest()


def load_python_macro_candidate(
    bundle_uri: Path | str,
    *,
    expected_sha256: str | None = None,
    deferred: bool = False,
):
    """Bind a verified source bundle, optionally deferring candidate execution."""

    if not isinstance(deferred, bool):
        raise TypeError("deferred must be bool")
    path = _local_bundle_path(str(bundle_uri))
    if deferred:
        if expected_sha256 is not None:
            _require_sha256(expected_sha256, "expected candidate")
        return _LazyPythonMacroProgram(
            path,
            expected_sha256=expected_sha256,
        )

    payload, bundle = _read_candidate_bundle_for_runtime(path)
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, "expected candidate")
        if digest != expected_sha256:
            raise ReferenceAgentRuntimeError(
                "candidate bundle changed after its identity was recorded"
            )
    if bundle is not None:
        return _load_multifile_python_candidate(bundle, digest)
    namespace: dict[str, object] = {
        "__file__": str(path),
        "__name__": f"_reference_macro_candidate_{digest}",
        "__package__": None,
    }
    try:
        code = compile(payload, str(path), "exec")
        exec(code, namespace)
    except (KeyboardInterrupt, GeneratorExit):
        raise
    except BaseException as exc:
        raise ReferenceAgentRuntimeError(
            f"candidate bundle could not be loaded: {type(exc).__name__}"
        ) from exc
    predict = namespace.get("predict_macros")
    prepare_solution = namespace.get("prepare_solution")
    if not callable(predict) and not callable(prepare_solution):
        raise ReferenceAgentRuntimeError(
            "candidate bundle must export callable prepare_solution or predict_macros"
        )
    return _LoadedPythonMacroProgram(
        predict if callable(predict) else None,
        prepare_solution if callable(prepare_solution) else None,
    )


def _reference_candidate_submission(
    ref: CandidateRef,
) -> CandidateSubmission:
    """Bind one sealed candidate and its controller-owned fresh loader."""

    if not isinstance(ref, CandidateRef):
        raise TypeError("reference candidate submission needs CandidateRef")
    if ref.bundle_uri is None:
        raise ReferenceAgentRuntimeError("sealed candidate URI is absent")
    bundle_uri = ref.bundle_uri
    bundle_sha256 = ref.bundle_sha256

    def fresh_program():
        return load_python_macro_candidate(
            bundle_uri,
            expected_sha256=bundle_sha256,
            deferred=True,
        )

    return CandidateSubmission(
        ref,
        fresh_program(),
        program_factory=fresh_program,
    )


def _load_multifile_python_candidate(
    bundle: CandidateBundle,
    digest: str,
) -> _LoadedPythonMacroProgram:
    """Load one bundle as a digest-namespaced package.

    Relative imports are isolated by the synthetic package name. The bundle
    root and entrypoint directory are on sys.path only while module preparation
    runs; scoring remains a call to the resulting predict function.
    """

    entrypoint = bundle.entrypoint_path
    # A digest identifies immutable bytes, not one runtime. A fresh namespace
    # prevents relative-import module state from crossing formal cohorts.
    module_name = (
        f"_reference_macro_candidate_{digest}_{secrets.token_hex(16)}"
    )
    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            entrypoint,
            submodule_search_locations=[str(entrypoint.parent)],
        )
        if spec is None or spec.loader is None:
            raise ImportError("candidate entrypoint has no Python loader")
        module = importlib.util.module_from_spec(spec)
    except BaseException as exc:
        raise ReferenceAgentRuntimeError(
            f"candidate bundle could not be loaded: {type(exc).__name__}"
        ) from exc

    local_top_levels = {
        member.path.split("/", 1)[0].removesuffix(".py")
        for member in bundle.members
        if member.path.endswith(".py")
    }
    with _CANDIDATE_IMPORT_LOCK:
        original_path = list(sys.path)
        original_bytecode = sys.dont_write_bytecode
        saved_modules = {
            name: sys.modules.pop(name)
            for name in tuple(sys.modules)
            if name.split(".", 1)[0] in local_top_levels
        }
        sys.modules[module_name] = module
        search_paths = tuple(
            dict.fromkeys((str(entrypoint.parent), str(bundle.root)))
        )
        sys.path[:0] = list(search_paths)
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        except (KeyboardInterrupt, GeneratorExit):
            _discard_candidate_imports(module_name, local_top_levels, saved_modules)
            raise
        except BaseException as exc:
            _discard_candidate_imports(module_name, local_top_levels, saved_modules)
            raise ReferenceAgentRuntimeError(
                f"candidate bundle could not be loaded: {type(exc).__name__}"
            ) from exc
        finally:
            sys.path[:] = original_path
            sys.dont_write_bytecode = original_bytecode
        # Keep the imported objects reachable through the returned callables,
        # but do not leave a phase-discoverable package registry behind.  A
        # later formal runtime must not be able to find an earlier runtime by
        # scanning ``sys.modules``.
        _discard_candidate_imports(module_name, local_top_levels, saved_modules)

    predict = getattr(module, "predict_macros", None)
    prepare_solution = getattr(module, "prepare_solution", None)
    if not callable(predict) and not callable(prepare_solution):
        raise ReferenceAgentRuntimeError(
            "candidate bundle must export callable prepare_solution or predict_macros"
        )
    return _LoadedPythonMacroProgram(
        predict if callable(predict) else None,
        prepare_solution if callable(prepare_solution) else None,
    )


def _discard_candidate_imports(
    module_name: str,
    local_top_levels: set[str],
    saved_modules: Mapping[str, object],
) -> None:
    for name in tuple(sys.modules):
        if (
            name == module_name
            or name.startswith(module_name + ".")
            or name.split(".", 1)[0] in local_top_levels
        ):
            sys.modules.pop(name, None)
    sys.modules.update(saved_modules)

class _PublicValueCodec:
    """Tagged JSON metadata plus digest-named NPY files for public arrays."""

    def __init__(self, arrays_root: Path, *, create: bool = True) -> None:
        self.arrays_root = arrays_root
        if create:
            _ensure_private_directory(self.arrays_root)
        elif self.arrays_root.is_symlink() or not self.arrays_root.is_dir():
            raise ReferenceAgentRuntimeError(
                "public arrays directory is unavailable"
            )
        try:
            import numpy as np
        except ImportError:
            self.np = None
        else:
            self.np = np

    def encode(self, value: object) -> Mapping[str, object]:
        np = self.np
        if np is not None and isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError("public numpy arrays cannot use object dtype")
            if value.dtype.kind in {"f", "c"} and not bool(np.all(np.isfinite(value))):
                raise ValueError("public numpy arrays must contain finite values")
            stream = io.BytesIO()
            np.save(stream, value, allow_pickle=False)
            payload = stream.getvalue()
            digest = hashlib.sha256(payload).hexdigest()
            relative = Path("arrays") / f"{digest}.npy"
            destination = self.arrays_root.parent / relative
            if destination.exists():
                existing = _read_bounded_regular_file(
                    destination,
                    max_bytes=len(payload),
                    label="public array",
                )
                if existing != payload:
                    raise ReferenceAgentRuntimeError(
                        "digest-named public array contains different bytes"
                    )
            else:
                descriptor = os.open(
                    destination,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o400,
                )
                with os.fdopen(descriptor, "wb") as output:
                    output.write(payload)
                    output.flush()
                    os.fsync(output.fileno())
            os.chmod(destination, 0o400)
            return {
                "type": "ndarray",
                "path": relative.as_posix(),
                "sha256": digest,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
        if np is not None and isinstance(value, np.generic):
            return self.encode(value.item())
        if isinstance(value, Enum):
            return self.encode(value.value)
        if value is None or isinstance(value, (str, bool, int)):
            return {"type": "scalar", "value": value}
        if isinstance(value, Real):
            converted = float(value)
            if not isfinite(converted):
                raise ValueError("public floating-point values must be finite")
            return {"type": "scalar", "value": converted}
        if isinstance(value, Complex):
            real = float(value.real)
            imaginary = float(value.imag)
            if not isfinite(real) or not isfinite(imaginary):
                raise ValueError("public complex values must be finite")
            return {
                "type": "complex",
                "real": real,
                "imag": imaginary,
            }
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) or not key for key in value):
                raise TypeError("public mapping keys must be nonempty strings")
            return {
                "type": "mapping",
                "items": {
                    key: self.encode(value[key])
                    for key in sorted(value)
                },
            }
        if isinstance(value, tuple):
            return {
                "type": "tuple",
                "items": [self.encode(item) for item in value],
            }
        if isinstance(value, list):
            return {
                "type": "list",
                "items": [self.encode(item) for item in value],
            }
        raise TypeError(f"unsupported public value: {type(value).__name__}")

    def decode(self, node: object, *, mmap_arrays: bool) -> object:
        if not isinstance(node, Mapping):
            raise ReferenceAgentRuntimeError("public value node must be an object")
        kind = node.get("type")
        if kind == "scalar":
            if set(node) != {"type", "value"}:
                raise ReferenceAgentRuntimeError("invalid scalar public value")
            return node["value"]
        if kind == "complex":
            if set(node) != {"type", "real", "imag"}:
                raise ReferenceAgentRuntimeError("invalid complex public value")
            real, imaginary = node["real"], node["imag"]
            if any(
                isinstance(value, bool) or not isinstance(value, Real)
                for value in (real, imaginary)
            ):
                raise ReferenceAgentRuntimeError("invalid complex public value")
            converted = complex(float(real), float(imaginary))
            if not isfinite(converted.real) or not isfinite(converted.imag):
                raise ReferenceAgentRuntimeError("invalid complex public value")
            return converted
        if kind in {"mapping", "tuple", "list"}:
            if set(node) != {"type", "items"}:
                raise ReferenceAgentRuntimeError("invalid container public value")
            items = node["items"]
            if kind == "mapping":
                if not isinstance(items, Mapping):
                    raise ReferenceAgentRuntimeError(
                        "public mapping items must be an object"
                    )
                return {
                    str(key): self.decode(value, mmap_arrays=mmap_arrays)
                    for key, value in items.items()
                }
            if not isinstance(items, list):
                raise ReferenceAgentRuntimeError(
                    "public sequence items must be a list"
                )
            values = [
                self.decode(item, mmap_arrays=mmap_arrays)
                for item in items
            ]
            return tuple(values) if kind == "tuple" else values
        if kind == "ndarray":
            if set(node) - {"sha256"} != {"type", "path", "dtype", "shape"}:
                raise ReferenceAgentRuntimeError("invalid ndarray public value")
            if self.np is None:
                raise ReferenceAgentRuntimeError(
                    "numpy is required to load public array files"
                )
            relative = node["path"]
            if not isinstance(relative, str):
                raise ReferenceAgentRuntimeError("public array path must be a string")
            path = _resolve_within(self.arrays_root.parent, relative)
            if path.parent != self.arrays_root or path.suffix != ".npy":
                raise ReferenceAgentRuntimeError("public array path is invalid")
            array = self.np.load(
                path,
                allow_pickle=False,
                mmap_mode="r" if mmap_arrays else None,
            )
            if str(array.dtype) != node["dtype"] or list(array.shape) != node["shape"]:
                raise ReferenceAgentRuntimeError("public array metadata changed")
            return array
        raise ReferenceAgentRuntimeError(f"unknown public value type: {kind!r}")


def _reference_public_case_payload(
    case: Case,
    codec: _PublicValueCodec,
) -> Mapping[str, object]:
    case = candidate_facing_case(case)
    return {
        "case_id": case.case_id,
        "source": case.source.value,
        "visibility": case.visibility.value,
        "public_data": codec.encode(case.public_data),
        "round_index": case.round_index,
        "parent_case_id": case.parent_case_id,
        "tags": list(case.tags),
    }


def _reference_public_case_from_payload(
    payload: object,
    *,
    codec: _PublicValueCodec,
    mmap_arrays: bool,
) -> Case:
    if not isinstance(payload, Mapping):
        raise ReferenceAgentRuntimeError("guided public case must be an object")
    expected = {
        "case_id",
        "source",
        "visibility",
        "public_data",
        "round_index",
        "parent_case_id",
        "tags",
    }
    if set(payload) != expected:
        raise ReferenceAgentRuntimeError(
            "guided public case fields do not match the schema"
        )
    try:
        case = Case(
            case_id=_required_string(payload, "case_id"),
            source=CaseSource(_required_string(payload, "source")),
            visibility=CaseVisibility(_required_string(payload, "visibility")),
            public_data=codec.decode(
                payload.get("public_data"),
                mmap_arrays=mmap_arrays,
            ),
            round_index=payload.get("round_index"),
            parent_case_id=payload.get("parent_case_id"),
            tags=tuple(payload.get("tags", ())),
        )
    except (TypeError, ValueError) as exc:
        raise ReferenceAgentRuntimeError(
            "guided public case is invalid"
        ) from exc
    if case.visibility is not CaseVisibility.DEVELOPMENT:
        raise ReferenceAgentRuntimeError(
            "external guidance may expose only DEVELOPMENT cases"
        )
    return case


def _reference_guided_batch_payload(
    batch: CandidateGuidedSampleBatch,
    codec: _PublicValueCodec,
) -> Mapping[str, object]:
    """Serialize only public cases and opaque ids, never answer locators."""

    if not isinstance(batch, CandidateGuidedSampleBatch):
        raise TypeError("guidance sample returned an invalid batch")
    samples = batch.samples
    rows = [
        {
            "sample_id": sample.sample_id,
            "case": _reference_public_case_payload(sample.case, codec),
            "answer_to_case_derivation_commitment_sha256": (
                sample.answer_to_case_derivation_commitment_sha256
            ),
        }
        for sample in samples
    ]
    return {
        "snapshot_sha256": batch.snapshot_sha256,
        "sample_settings_commitment_sha256": (
            batch.sample_settings_commitment_sha256
        ),
        "samples": rows,
        "campaign_guided_generation_wall_seconds": (
            batch.campaign_guided_generation_wall_seconds
        ),
    }


def _reference_guided_batch_from_payload(
    payload: object,
    *,
    codec: _PublicValueCodec,
    mmap_arrays: bool,
) -> CandidateGuidedSampleBatch:
    if not isinstance(payload, Mapping):
        raise ReferenceAgentRuntimeError("guided sample batch must be an object")
    expected = {
        "snapshot_sha256",
        "sample_settings_commitment_sha256",
        "samples",
        "campaign_guided_generation_wall_seconds",
    }
    if set(payload) != expected:
        raise ReferenceAgentRuntimeError(
            "guided sample batch fields do not match the schema"
        )
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ReferenceAgentRuntimeError(
            "guided sample batch must contain public samples"
        )
    samples = []
    for raw in raw_samples:
        if not isinstance(raw, Mapping) or set(raw) != {
            "sample_id",
            "case",
            "answer_to_case_derivation_commitment_sha256",
        }:
            raise ReferenceAgentRuntimeError(
                "guided sample fields do not match the schema"
            )
        samples.append(
            CandidateGuidedSample(
                sample_id=_required_string(raw, "sample_id"),
                case=_reference_public_case_from_payload(
                    raw.get("case"),
                    codec=codec,
                    mmap_arrays=mmap_arrays,
                ),
                answer_to_case_derivation_commitment_sha256=_required_sha256(
                    raw, "answer_to_case_derivation_commitment_sha256"
                ),
            )
        )
    elapsed = payload.get("campaign_guided_generation_wall_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, Real)
        or not isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise ReferenceAgentRuntimeError(
            "guided generation wall time must be finite and nonnegative"
        )
    return CandidateGuidedSampleBatch(
        snapshot_sha256=_required_sha256(payload, "snapshot_sha256"),
        sample_settings_commitment_sha256=_required_sha256(
            payload, "sample_settings_commitment_sha256"
        ),
        samples=tuple(samples),
        campaign_guided_generation_wall_seconds=float(elapsed),
    )


def _decode_reference_public_tasks(
    rows: object,
    *,
    codec: _PublicValueCodec,
    mmap_arrays: bool,
    require_answers: bool,
) -> tuple[
    tuple[Case, ...],
    tuple[AnswerQuery, ...],
    tuple[AnswerView | None, ...],
]:
    if not isinstance(rows, list) or not rows:
        raise ReferenceAgentRuntimeError("reference workspace has no public tasks")
    cases, queries, answers = [], [], []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ReferenceAgentRuntimeError("public task must be an object")
        case_payload, query_payload = row.get("case"), row.get("query")
        if not isinstance(case_payload, Mapping) or not isinstance(
            query_payload, Mapping
        ):
            raise ReferenceAgentRuntimeError("public task is missing case or query")
        cases.append(
            Case(
                case_id=_required_string(case_payload, "case_id"),
                source=CaseSource(_required_string(case_payload, "source")),
                visibility=CaseVisibility(
                    _required_string(case_payload, "visibility")
                ),
                public_data=codec.decode(
                    case_payload.get("public_data"),
                    mmap_arrays=mmap_arrays,
                ),
                round_index=case_payload.get("round_index"),
                parent_case_id=case_payload.get("parent_case_id"),
                tags=tuple(case_payload.get("tags", ())),
            )
        )
        answer_payload = row.get("completed_answer")
        points_redacted = query_payload.get("points_redacted", False)
        if not isinstance(points_redacted, bool):
            raise ReferenceAgentRuntimeError(
                "public query points_redacted must be bool"
            )
        if points_redacted:
            if (
                answer_payload is not None
                or query_payload.get("kind") != QueryKind.POINTS.value
                or query_payload.get("points") != []
            ):
                raise ReferenceAgentRuntimeError(
                    "only unreleased POINTS queries may redact coordinates"
                )
            point_count = query_payload.get("point_count")
            if (
                isinstance(point_count, bool)
                or not isinstance(point_count, int)
                or point_count <= 0
            ):
                raise ReferenceAgentRuntimeError(
                    "redacted POINTS query needs a positive point_count"
                )
            public_points = tuple({} for _index in range(point_count))
        else:
            if "point_count" in query_payload:
                raise ReferenceAgentRuntimeError(
                    "unredacted query cannot declare point_count"
                )
            public_points = tuple(query_payload.get("points", ()))
        queries.append(
            AnswerQuery(
                kind=QueryKind(_required_string(query_payload, "kind")),
                fields=tuple(query_payload.get("fields", ())),
                resolution=dict(query_payload.get("resolution", {})),
                points=public_points,
            )
        )
        if answer_payload is None and require_answers:
            raise ReferenceAgentRuntimeError(
                "completed-answer release is missing an answer"
            )
        answers.append(
            None
            if answer_payload is None
            else _answer_view_from_payload(
                answer_payload,
                codec=codec,
                mmap_arrays=mmap_arrays,
            )
        )
    return tuple(cases), tuple(queries), tuple(answers)


def _load_public_search_contract(payload: object) -> Mapping[str, object]:
    if payload is None:
        return MappingProxyType({})
    if not isinstance(payload, Mapping):
        raise ReferenceAgentRuntimeError("search_contract must be an object")
    open_search_fields = {
        "leaderboard_limit",
        "cross_share_after_round",
        "counts_are_aggregate_only",
        "official_nomination_limit",
        "development_screens_are_official_nominations",
    }
    legacy_finish_fields = {
        "finish_gate_available",
        "min_distinct_screened_candidates",
        "min_full_stage_candidates",
        "min_released_roots",
        "require_final_full_stage",
        "leaderboard_limit",
        "cross_share_after_round",
        "counts_are_aggregate_only",
    }
    fields = set(payload)
    if fields not in (open_search_fields, legacy_finish_fields):
        raise ReferenceAgentRuntimeError(
            "search_contract fields do not match the schema"
        )
    if payload.get("counts_are_aggregate_only") is not True:
        raise ReferenceAgentRuntimeError(
            "search_contract must declare aggregate-only counts"
        )
    parsed = dict(payload)
    if fields == open_search_fields:
        if payload.get("development_screens_are_official_nominations") is not False:
            raise ReferenceAgentRuntimeError(
                "development screens must remain numerical feedback only"
            )
        parsed["official_nomination_limit"] = _required_positive_int(
            payload, "official_nomination_limit"
        )
        if parsed["official_nomination_limit"] != 1:
            raise ReferenceAgentRuntimeError(
                "each Agent must have exactly one official nomination"
            )
        integer_names = ("leaderboard_limit", "cross_share_after_round")
    else:
        for name in ("finish_gate_available", "require_final_full_stage"):
            if not isinstance(payload.get(name), bool):
                raise ReferenceAgentRuntimeError(
                    f"search_contract {name} must be bool"
                )
        integer_names = (
            "min_distinct_screened_candidates",
            "min_full_stage_candidates",
            "min_released_roots",
            "leaderboard_limit",
            "cross_share_after_round",
        )
    for name in integer_names:
        parsed[name] = _required_nonnegative_int(payload, name)
    return MappingProxyType(parsed)


def _optional_finite_number(payload: Mapping[str, object], name: str) -> float | None:
    value = payload.get(name)
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(float(value))
    ):
        raise ReferenceAgentRuntimeError(f"{name} must be finite or null")
    return float(value)


def _load_public_prior_leaderboard(
    payload: object,
) -> tuple[Mapping[str, object], ...]:
    if payload is None:
        return ()
    if not isinstance(payload, list):
        raise ReferenceAgentRuntimeError("prior_leaderboard must be a list")
    expected = {
        "rank",
        "candidate_id",
        "bundle_sha256",
        "agent_id",
        "round_index",
        "eligible",
        "loss",
        "guardrail_passed",
        "roles",
    }
    loss_expected = {
        "mean",
        "cvar",
        "max",
        "generated_mean",
        "perturbed_mean",
    }
    rows: list[Mapping[str, object]] = []
    for raw in payload:
        if not isinstance(raw, Mapping) or set(raw) != expected:
            raise ReferenceAgentRuntimeError(
                "prior_leaderboard row fields do not match the schema"
            )
        loss = raw.get("loss")
        if not isinstance(loss, Mapping) or set(loss) != loss_expected:
            raise ReferenceAgentRuntimeError(
                "prior_leaderboard loss fields do not match the schema"
            )
        eligible = raw.get("eligible")
        guardrail = raw.get("guardrail_passed")
        roles = raw.get("roles")
        if not isinstance(eligible, bool):
            raise ReferenceAgentRuntimeError(
                "prior_leaderboard eligible must be bool"
            )
        if guardrail is not None and not isinstance(guardrail, bool):
            raise ReferenceAgentRuntimeError(
                "prior_leaderboard guardrail must be bool or null"
            )
        if not isinstance(roles, list) or any(
            not isinstance(role, str) or not role for role in roles
        ):
            raise ReferenceAgentRuntimeError(
                "prior_leaderboard roles must be nonempty strings"
            )
        parsed_loss = MappingProxyType(
            {
                name: _optional_finite_number(loss, name)
                for name in loss_expected
            }
        )
        rows.append(
            MappingProxyType(
                {
                    "rank": _required_positive_int(raw, "rank"),
                    "candidate_id": _required_string(raw, "candidate_id"),
                    "bundle_sha256": _required_sha256(raw, "bundle_sha256"),
                    "agent_id": _required_string(raw, "agent_id"),
                    "round_index": _required_nonnegative_int(
                        raw, "round_index"
                    ),
                    "eligible": eligible,
                    "loss": parsed_loss,
                    "guardrail_passed": guardrail,
                    "roles": tuple(roles),
                }
            )
        )
    return tuple(rows)


def _load_public_prior_candidates(
    payload: object,
    *,
    workspace: Path,
) -> tuple[Mapping[str, object], ...]:
    if payload is None:
        return ()
    if not isinstance(payload, list):
        raise ReferenceAgentRuntimeError("prior_candidates must be a list")
    expected = {
        "candidate_id",
        "bundle_sha256",
        "agent_id",
        "round_index",
        "path",
    }
    rows: list[Mapping[str, object]] = []
    for raw in payload:
        if not isinstance(raw, Mapping) or set(raw) != expected:
            raise ReferenceAgentRuntimeError(
                "prior_candidates row fields do not match the schema"
            )
        digest = _required_sha256(raw, "bundle_sha256")
        staged_path = _resolve_agent_python_file(
            workspace,
            workspace / _required_string(raw, "path"),
        )
        rows.append(
            MappingProxyType(
                {
                    "candidate_id": _required_string(raw, "candidate_id"),
                    "bundle_sha256": digest,
                    "agent_id": _required_string(raw, "agent_id"),
                    "round_index": _required_nonnegative_int(
                        raw, "round_index"
                    ),
                    "path": staged_path,
                }
            )
        )
    return tuple(rows)



def _load_reference_public_problem(
    payload: object,
    *,
    workspace: Path,
) -> tuple[Path | None, Path | None, Mapping[str, object] | None]:
    if payload is None:
        return None, None, None
    if not isinstance(payload, Mapping):
        raise ReferenceAgentRuntimeError("public_problem must be an object or null")
    expected = {
        "bundle_id",
        "bundle_sha256",
        "equation_id",
        "resources_path",
        "manifest_path",
    }
    if set(payload) != expected:
        raise ReferenceAgentRuntimeError(
            "public_problem fields do not match the schema"
        )
    resources = _resolve_within(
        workspace,
        _required_string(payload, "resources_path"),
    )
    manifest_path = _resolve_within(
        workspace,
        _required_string(payload, "manifest_path"),
    )
    if (
        resources.name != "public_resources"
        or resources.is_symlink()
        or not resources.is_dir()
        or manifest_path != resources / "manifest.json"
    ):
        raise ReferenceAgentRuntimeError(
            "public problem resources are outside the immutable bundle tree"
        )
    manifest = _read_json_object(
        manifest_path,
        max_bytes=_MAX_CONTEXT_JSON_BYTES,
    )
    if manifest.get("schema") != "pde-public-problem-bundle-v1":
        raise ReferenceAgentRuntimeError("unknown public problem manifest schema")
    for name in ("bundle_id", "bundle_sha256", "equation_id"):
        if manifest.get(name) != payload.get(name):
            raise ReferenceAgentRuntimeError(
                f"public problem {name} differs from its context binding"
            )
    return resources, manifest_path, MappingProxyType(dict(manifest))


def _load_reference_numerical_oracle(
    payload: object,
    *,
    manifest: Mapping[str, object] | None,
) -> tuple[str | None, tuple[str, ...], tuple[str, ...]]:
    if payload is None:
        declared = () if manifest is None else tuple(
            manifest.get("oracle_methods", ())
        ) + tuple(manifest.get("development_solution_oracle_methods", ()))
        if declared:
            raise ReferenceAgentRuntimeError(
                "public problem declares numerical methods without an RPC"
            )
        return None, (), ()
    if not isinstance(payload, Mapping):
        raise ReferenceAgentRuntimeError(
            "numerical_oracle must be an object or null"
        )
    endpoint = _required_string(payload, "endpoint")
    status = _required_string(payload, "capability_status")
    if status not in {
        _EVALUATOR_CAPABILITY_ACTIVE,
        _EVALUATOR_CAPABILITY_EXPIRED,
    }:
        raise ReferenceAgentRuntimeError(
            "numerical oracle capability status is invalid"
        )
    if payload.get("coordinate_policy") != "agent_selected":
        raise ReferenceAgentRuntimeError(
            "numerical oracle coordinates must be Agent-selected"
        )
    if payload.get("session_lifecycle") != "fresh_per_call":
        raise ReferenceAgentRuntimeError(
            "numerical oracle session lifecycle is invalid"
        )
    if payload.get("fixed_lifetime_or_sample_quota") is not False:
        raise ReferenceAgentRuntimeError(
            "numerical oracle must not advertise a fixed campaign quota"
        )

    def methods(name: str) -> tuple[str, ...]:
        raw = payload.get(name)
        if not isinstance(raw, list) or any(
            not isinstance(method, str)
            or not method.isidentifier()
            or method.startswith("_")
            for method in raw
        ):
            raise ReferenceAgentRuntimeError(
                f"numerical oracle {name} is invalid"
            )
        values = tuple(raw)
        if len(values) != len(set(values)):
            raise ReferenceAgentRuntimeError(
                f"numerical oracle {name} contains duplicates"
            )
        return values

    equation = methods("equation_methods")
    completed = methods("development_solution_methods")
    if set(equation) & set(completed):
        raise ReferenceAgentRuntimeError(
            "numerical oracle namespaces must have disjoint methods"
        )
    if manifest is None:
        raise ReferenceAgentRuntimeError(
            "numerical oracle requires a public problem manifest"
        )

    def declared(name: str) -> set[str]:
        rows = manifest.get(name)
        if not isinstance(rows, list) or any(
            not isinstance(row, Mapping) or not isinstance(row.get("name"), str)
            for row in rows
        ):
            raise ReferenceAgentRuntimeError(
                "public problem oracle manifest is invalid"
            )
        return {str(row["name"]) for row in rows}

    if set(equation) != declared("oracle_methods") or set(completed) != declared(
        "development_solution_oracle_methods"
    ):
        raise ReferenceAgentRuntimeError(
            "numerical RPC allowlists differ from the public problem manifest"
        )
    return endpoint, equation, completed


def _load_reference_guidance_oracle(
    payload: object,
) -> tuple[str | None, str | None, tuple[str, ...], bool]:
    if payload is None:
        return None, None, (), False
    if not isinstance(payload, Mapping):
        raise ReferenceAgentRuntimeError(
            "guidance_oracle must be an object or null"
        )
    status = _required_string(payload, "capability_status")
    if status not in {
        _EVALUATOR_CAPABILITY_ACTIVE,
        _EVALUATOR_CAPABILITY_EXPIRED,
    }:
        raise ReferenceAgentRuntimeError(
            "guidance Oracle capability status is invalid"
        )
    if payload.get("actions") != [
        "sample",
        "query_completed_solution",
    ]:
        raise ReferenceAgentRuntimeError(
            "guidance Oracle actions are invalid"
        )
    if payload.get("unlimited_fresh_inference") is not True:
        raise ReferenceAgentRuntimeError(
            "guidance Oracle must expose unlimited fresh inference"
        )
    if payload.get("fixed_lifetime_or_sample_quota") is not False:
        raise ReferenceAgentRuntimeError(
            "guidance Oracle cannot advertise a fixed campaign quota"
        )
    if payload.get("training_available") is not False:
        raise ReferenceAgentRuntimeError(
            "Agent guidance must not expose training"
        )
    if payload.get("hidden_resolver_available") is not False:
        raise ReferenceAgentRuntimeError(
            "Agent guidance must not expose a hidden resolver"
        )
    if payload.get("answer_locator_exposed") is not False:
        raise ReferenceAgentRuntimeError(
            "Agent guidance must not expose answer locators"
        )
    raw_methods = payload.get("development_solution_methods")
    if not isinstance(raw_methods, list) or any(
        not isinstance(method, str)
        or not method.isidentifier()
        or method.startswith("_")
        for method in raw_methods
    ):
        raise ReferenceAgentRuntimeError(
            "guidance completed-solution method allowlist is invalid"
        )
    methods = tuple(raw_methods)
    if not methods or len(methods) != len(set(methods)):
        raise ReferenceAgentRuntimeError(
            "guidance completed-solution methods must be unique"
        )
    return (
        _required_string(payload, "endpoint"),
        _required_sha256(payload, "snapshot_sha256"),
        methods,
        True,
    )

def load_reference_public_workspace(
    context_path: Path | str,
    *,
    mmap_arrays: bool = True,
) -> ReferencePublicWorkspace:
    """Load the public Case/query workspace without controller targets."""

    path = Path(context_path).resolve()
    payload = _read_json_object(path)
    if payload.get("schema") != REFERENCE_CONTEXT_SCHEMA:
        raise ReferenceAgentRuntimeError("unknown reference workspace schema")
    workspace = path.parent
    codec = _PublicValueCodec(workspace / "arrays", create=False)
    cases, queries, completed_answers = _decode_reference_public_tasks(
        payload.get("public_tasks"),
        codec=codec,
        mmap_arrays=mmap_arrays,
        require_answers=False,
    )
    cohort = payload.get("cohort")
    incumbent = payload.get("incumbent")
    evaluator = payload.get("evaluator")
    if not all(isinstance(value, Mapping) for value in (cohort, incumbent, evaluator)):
        raise ReferenceAgentRuntimeError("reference workspace metadata is incomplete")
    incumbent_path = _resolve_within(
        workspace,
        _required_string(incumbent, "path"),
    )
    ordered_ids = tuple(cohort.get("ordered_case_ids", ()))
    if ordered_ids != tuple(case.case_id for case in cases):
        raise ReferenceAgentRuntimeError("public task ordering changed")
    credit_fields = {
        "remaining_case_credits",
        "remaining_root_credits",
        "screen_root_request_capacity",
    } & set(evaluator)
    if len(credit_fields) != 1:
        raise ReferenceAgentRuntimeError(
            "workspace must expose exactly one development resource contract"
        )
    open_search = "screen_root_request_capacity" in evaluator
    strong_session = open_search or "remaining_root_credits" in evaluator
    if open_search:
        if evaluator.get("search_mode") != "open":
            raise ReferenceAgentRuntimeError(
                "open-search evaluator metadata is incomplete"
            )
        if evaluator.get("fixed_lifetime_root_quota") is not False:
            raise ReferenceAgentRuntimeError(
                "open search cannot advertise a lifetime root quota"
            )
    finish_endpoint = (
        _required_string(evaluator, "finish_endpoint")
        if "finish_endpoint" in evaluator
        else None
    )
    if finish_endpoint is not None and not strong_session:
        raise ReferenceAgentRuntimeError(
            "finish endpoint is only valid for a strong development session"
        )
    search_contract = _load_public_search_contract(
        payload.get("search_contract")
    )
    if (
        "finish_gate_available" in search_contract
        and search_contract["finish_gate_available"]
        != (finish_endpoint is not None)
    ):
        raise ReferenceAgentRuntimeError(
            "search contract and finish capability disagree"
        )
    prior_leaderboard = _load_public_prior_leaderboard(
        payload.get("prior_leaderboard")
    )
    prior_candidates = _load_public_prior_candidates(
        payload.get("prior_candidates"), workspace=workspace
    )
    if open_search:
        recommended = evaluator.get("recommended_screen_root_counts")
        usage = evaluator.get("observational_usage")
        if not isinstance(recommended, list) or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in recommended
        ):
            raise ReferenceAgentRuntimeError(
                "recommended open-search request sizes are invalid"
            )
        if not isinstance(usage, Mapping) or any(
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for name, value in usage.items()
        ):
            raise ReferenceAgentRuntimeError(
                "open-search observational usage is invalid"
            )
        recommended_screen_root_counts = tuple(recommended)
        observational_usage = MappingProxyType(dict(usage))
    else:
        recommended_screen_root_counts = ()
        observational_usage = MappingProxyType({})
    resources_path, manifest_path, public_manifest = (
        _load_reference_public_problem(
            payload.get("public_problem"),
            workspace=workspace,
        )
    )
    oracle_endpoint, equation_methods, development_solution_methods = (
        _load_reference_numerical_oracle(
            payload.get("numerical_oracle"),
            manifest=public_manifest,
        )
    )
    (
        guidance_endpoint,
        guidance_snapshot,
        guidance_solution_methods,
        guidance_unlimited,
    ) = _load_reference_guidance_oracle(payload.get("guidance_oracle"))
    return ReferencePublicWorkspace(
        context_path=path,
        run_id=_required_string(payload, "run_id"),
        round_index=_required_nonnegative_int(payload, "round_index"),
        agent_id=_required_string(payload, "agent_id"),
        prompt=_required_string(payload, "prompt"),
        cases=tuple(cases),
        queries=tuple(queries),
        completed_answers=tuple(completed_answers),
        cohort_id=_required_string(cohort, "cohort_id"),
        cohort_sha256=_required_sha256(cohort, "cohort_sha256"),
        contract_id=_required_string(cohort, "contract_id"),
        incumbent_path=incumbent_path,
        incumbent_sha256=_required_sha256(incumbent, "bundle_sha256"),
        evaluator_endpoint=_required_string(evaluator, "endpoint"),
        remaining_case_credits=(
            _required_nonnegative_int(evaluator, "remaining_case_credits")
            if "remaining_case_credits" in evaluator
            else None
        ),
        remaining_root_credits=(
            _required_nonnegative_int(evaluator, "remaining_root_credits")
            if "remaining_root_credits" in evaluator
            else None
        ),
        screen_timeout_seconds=(
            _required_positive_float(evaluator, "screen_timeout_seconds")
            if "screen_timeout_seconds" in evaluator
            else None
        ),
        release_endpoint=(
            _required_string(evaluator, "release_endpoint")
            if "release_endpoint" in evaluator
            else None
        ),
        remaining_release_root_credits=(
            _required_nonnegative_int(
                evaluator, "remaining_release_root_credits"
            )
            if "remaining_release_root_credits" in evaluator
            else None
        ),
        max_release_roots_per_request=(
            _required_positive_int(evaluator, "max_release_roots_per_request")
            if "max_release_roots_per_request" in evaluator
            else None
        ),
        screen_submit_endpoint=(
            _required_string(evaluator, "screen_submit_endpoint")
            if "screen_submit_endpoint" in evaluator
            else None
        ),
        screen_poll_endpoint=(
            _required_string(evaluator, "screen_poll_endpoint")
            if "screen_poll_endpoint" in evaluator
            else None
        ),
        max_async_screen_inflight=(
            _required_positive_int(evaluator, "max_async_screen_inflight")
            if "max_async_screen_inflight" in evaluator
            else None
        ),
        finish_endpoint=finish_endpoint,
        search_contract=search_contract,
        prior_leaderboard=prior_leaderboard,
        prior_candidates=prior_candidates,
        public_resources_path=resources_path,
        public_problem_manifest_path=manifest_path,
        public_problem_manifest=public_manifest,
        numerical_oracle_endpoint=oracle_endpoint,
        equation_oracle_methods=equation_methods,
        development_solution_oracle_methods=development_solution_methods,
        guidance_oracle_endpoint=guidance_endpoint,
        guidance_snapshot_sha256=guidance_snapshot,
        guidance_solution_methods=guidance_solution_methods,
        guidance_unlimited_fresh_inference=guidance_unlimited,
        open_search=open_search,
        screen_root_request_capacity=(
            _required_positive_int(evaluator, "screen_root_request_capacity")
            if open_search
            else None
        ),
        release_root_request_capacity=(
            _required_nonnegative_int(evaluator, "release_root_request_capacity")
            if open_search
            else None
        ),
        recommended_screen_root_counts=recommended_screen_root_counts,
        observational_usage=observational_usage,
    )


def load_reference_completed_answer_release(
    manifest_path: Path | str,
    *,
    mmap_arrays: bool = True,
) -> ReferenceCompletedAnswerRelease:
    """Load one immutable on-demand release without consulting private state."""

    path = Path(manifest_path).resolve()
    release_root = path.parent
    releases_root = release_root.parent
    workspace = releases_root.parent
    if path.name != "manifest.json" or releases_root.name != "releases":
        raise ReferenceAgentRuntimeError(
            "release manifest must be under workspace/releases"
        )
    try:
        relative_manifest = path.relative_to(workspace)
    except ValueError as exc:
        raise ReferenceAgentRuntimeError(
            "release manifest is outside its workspace"
        ) from exc
    if _resolve_within(workspace, relative_manifest.as_posix()) != path:
        raise ReferenceAgentRuntimeError("release manifest path is invalid")

    payload = _read_json_object(path, max_bytes=_MAX_CONTEXT_JSON_BYTES)
    if payload.get("schema") != REFERENCE_RELEASE_MANIFEST_SCHEMA:
        raise ReferenceAgentRuntimeError("unknown completed-answer release schema")
    cohort = payload.get("cohort")
    accounting = payload.get("accounting")
    if not isinstance(cohort, Mapping) or not isinstance(accounting, Mapping):
        raise ReferenceAgentRuntimeError(
            "completed-answer release metadata is incomplete"
        )
    codec = _PublicValueCodec(release_root / "arrays", create=False)
    cases, queries, raw_answers = _decode_reference_public_tasks(
        payload.get("public_tasks"),
        codec=codec,
        mmap_arrays=mmap_arrays,
        require_answers=True,
    )
    answers = tuple(answer for answer in raw_answers if answer is not None)
    ordered_ids = tuple(cohort.get("ordered_case_ids", ()))
    if ordered_ids != tuple(case.case_id for case in cases):
        raise ReferenceAgentRuntimeError("release task ordering changed")
    member_case_count = _required_positive_int(payload, "member_case_count")
    if member_case_count != len(cases):
        raise ReferenceAgentRuntimeError("release member case count changed")
    return ReferenceCompletedAnswerRelease(
        manifest_path=path,
        release_index=_required_positive_int(payload, "release_index"),
        root_count=_required_positive_int(payload, "root_count"),
        member_case_count=member_case_count,
        cohort_id=_required_string(cohort, "cohort_id"),
        cohort_sha256=_required_sha256(cohort, "cohort_sha256"),
        contract_id=_required_string(cohort, "contract_id"),
        cases=tuple(cases),
        queries=tuple(queries),
        answers=tuple(answers),
        charged_root_credits=_required_positive_int(
            accounting, "charged_root_credits"
        ),
        cumulative_charged_root_credits=_required_positive_int(
            accounting, "cumulative_charged_root_credits"
        ),
        remaining_root_credits=_required_nonnegative_int(
            accounting, "remaining_root_credits"
        ),
    )


def write_reference_candidate_submission(
    context_path: Path | str,
    *,
    candidate_id: str,
    bundle_path: Path | str,
    observations: Sequence[str] = (),
) -> Path:
    """Write the Agent's official nomination artifact for controller handoff."""

    observation_rows = _validated_submission_fields(candidate_id, observations)
    context = Path(context_path).resolve()
    payload = _read_json_object(context)
    if payload.get("schema") != REFERENCE_CONTEXT_SCHEMA:
        raise ReferenceAgentRuntimeError("unknown reference workspace schema")
    workspace = context.parent
    bundle = Path(bundle_path)
    if not bundle.is_absolute():
        bundle = workspace / bundle
    bundle = _resolve_agent_python_file(workspace, bundle)
    submission_path = _resolve_within(
        workspace,
        _required_string(payload, "submission_path"),
    )
    submission = {
        "schema": REFERENCE_SUBMISSION_SCHEMA,
        "candidate_id": candidate_id,
        "bundle_path": bundle.relative_to(workspace).as_posix(),
        "observations": list(observation_rows),
    }
    _write_json_exclusive(
        submission_path,
        submission,
        max_bytes=_MAX_SUBMISSION_BYTES,
    )
    return submission_path


def _reference_rpc_context(
    context_path: Path | str,
) -> tuple[Path, Path, Mapping[str, object]]:
    context = Path(context_path).resolve()
    metadata = _read_json_object(context)
    if metadata.get("schema") != REFERENCE_CONTEXT_SCHEMA:
        raise ReferenceAgentRuntimeError("unknown reference workspace schema")
    evaluator = metadata.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise ReferenceAgentRuntimeError("workspace evaluator metadata is missing")
    return context, context.parent, evaluator


def evaluate_reference_candidate(
    context_path: Path | str,
    *,
    candidate_path: Path | str,
    candidate_id: str,
    case_ids: Sequence[str] | None = None,
    return_finite_truth: bool = False,
    timeout_seconds: float = 300.0,
) -> Mapping[str, object]:
    """Call the budgeted loopback evaluator in a public workspace."""

    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("candidate_id cannot be empty")
    if not isinstance(return_finite_truth, bool):
        raise TypeError("return_finite_truth must be bool")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, Real)
        or not isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0.0
    ):
        raise ValueError("evaluator timeout must be finite and positive")
    context, workspace, evaluator = _reference_rpc_context(context_path)
    candidate = Path(candidate_path)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    candidate = _resolve_agent_python_file(workspace, candidate)
    requested_ids = None if case_ids is None else tuple(case_ids)
    if requested_ids is not None and any(
        not isinstance(case_id, str) or not case_id.strip()
        for case_id in requested_ids
    ):
        raise ValueError("case_ids must contain nonempty strings")
    request_payload = {
        "schema": REFERENCE_EVALUATION_REQUEST_SCHEMA,
        "candidate_id": candidate_id,
        "candidate_path": candidate.relative_to(workspace).as_posix(),
        "case_ids": list(requested_ids) if requested_ids is not None else None,
        "return_finite_truth": return_finite_truth,
    }
    return _call_reference_rpc(
        endpoint=_required_string(evaluator, "endpoint"),
        token=_required_string(evaluator, "token"),
        payload=request_payload,
        timeout_seconds=float(timeout_seconds),
        expected_schema=REFERENCE_EVALUATION_RESPONSE_SCHEMA,
    )



def query_reference_public_oracle(
    context_path: Path | str,
    *,
    namespace: str,
    case_id: str,
    method: str,
    args: Sequence[object] = (),
    kwargs: Mapping[str, object] | None = None,
    timeout_seconds: float = 300.0,
    mmap_arrays: bool = False,
) -> object:
    """Call one allowlisted numerical method at Agent-selected coordinates.

    ``namespace="equation"`` exposes only equation/public-data methods.
    ``namespace="development_solution"`` is a distinct capability that the
    controller accepts only for a registered DEVELOPMENT case. Each RPC opens
    and closes a fresh backend session; there is no client-side query quota.
    """

    if namespace not in {"equation", "development_solution"}:
        raise ValueError("unknown numerical oracle namespace")
    for label, value in (("case_id", case_id), ("method", method)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"numerical oracle {label} cannot be empty")
    if not method.isidentifier() or method.startswith("_"):
        raise ValueError("numerical oracle method is not a public identifier")
    if isinstance(args, (str, bytes)) or not isinstance(args, Sequence):
        raise TypeError("numerical oracle args must be a sequence")
    call_kwargs = {} if kwargs is None else kwargs
    if not isinstance(call_kwargs, Mapping) or any(
        not isinstance(name, str) or not name
        for name in call_kwargs
    ):
        raise TypeError("numerical oracle kwargs need nonempty string keys")
    if not isinstance(mmap_arrays, bool):
        raise TypeError("mmap_arrays must be bool")

    context, workspace, evaluator = _reference_rpc_context(context_path)
    metadata = _read_json_object(context)
    oracle = metadata.get("numerical_oracle")
    if not isinstance(oracle, Mapping):
        raise ReferenceAgentRuntimeError(
            "this workspace does not expose a numerical oracle"
        )
    methods_field = (
        "equation_methods"
        if namespace == "equation"
        else "development_solution_methods"
    )
    allowed = oracle.get(methods_field)
    if not isinstance(allowed, list) or method not in allowed:
        raise ReferenceAgentRuntimeError(
            "numerical oracle method is not allowlisted in this namespace"
        )
    codec = _PublicValueCodec(workspace / "arrays", create=False)
    response = _call_reference_rpc(
        endpoint=_required_string(oracle, "endpoint"),
        token=_required_string(evaluator, "token"),
        payload={
            "schema": REFERENCE_NUMERICAL_ORACLE_REQUEST_SCHEMA,
            "namespace": namespace,
            "case_id": case_id,
            "method": method,
            "args": codec.encode(tuple(args)),
            "kwargs": codec.encode(dict(call_kwargs)),
        },
        timeout_seconds=_reference_rpc_timeout(
            timeout_seconds,
            evaluator=oracle,
            metadata_field="timeout_seconds",
            fallback_seconds=300.0,
            label="numerical oracle timeout",
        ),
        expected_schema=REFERENCE_NUMERICAL_ORACLE_RESPONSE_SCHEMA,
    )
    expected = {"schema", "namespace", "case_id", "method", "result"}
    if set(response) != expected:
        raise ReferenceEvaluatorClientError(
            "malformed_evaluator_response",
            "local numerical oracle returned a malformed response",
        )
    if (
        response.get("namespace") != namespace
        or response.get("case_id") != case_id
        or response.get("method") != method
    ):
        raise ReferenceEvaluatorClientError(
            "malformed_evaluator_response",
            "local numerical oracle response identity changed",
        )
    return codec.decode(response.get("result"), mmap_arrays=mmap_arrays)


# Compatibility alias for the implementation name used during integration.
call_reference_numerical_oracle = query_reference_public_oracle


def sample_reference_guidance(
    context_path: Path | str,
    *,
    spec: GuidedSampleSpec,
    timeout_seconds: float = 300.0,
    mmap_arrays: bool = False,
) -> CandidateGuidedSampleBatch:
    """Draw fresh inference-only guided DEVELOPMENT cases.

    The Agent chooses sample_count and settings. The controller supplies fresh
    randomness on every call. This helper has no lifetime/sample quota and
    exposes neither training nor a hidden answer resolver/locator.
    """

    if not isinstance(spec, GuidedSampleSpec):
        raise TypeError("guidance sample needs a GuidedSampleSpec")
    if not isinstance(mmap_arrays, bool):
        raise TypeError("mmap_arrays must be bool")
    context, workspace, evaluator = _reference_rpc_context(context_path)
    metadata = _read_json_object(context)
    oracle = metadata.get("guidance_oracle")
    if not isinstance(oracle, Mapping):
        raise ReferenceAgentRuntimeError(
            "this workspace does not expose a guidance Oracle"
        )
    codec = _PublicValueCodec(workspace / "arrays", create=False)
    response = _call_reference_rpc(
        endpoint=_required_string(oracle, "endpoint"),
        token=_required_string(evaluator, "token"),
        payload={
            "schema": REFERENCE_GUIDANCE_ORACLE_REQUEST_SCHEMA,
            "action": "sample",
            "sample_count": spec.sample_count,
            "settings": codec.encode(dict(spec.settings)),
        },
        timeout_seconds=_reference_rpc_timeout(
            timeout_seconds,
            evaluator=oracle,
            metadata_field="timeout_seconds",
            fallback_seconds=300.0,
            label="guidance Oracle timeout",
        ),
        expected_schema=REFERENCE_GUIDANCE_ORACLE_RESPONSE_SCHEMA,
    )
    if set(response) != {"schema", "action", "batch"} or response.get(
        "action"
    ) != "sample":
        raise ReferenceEvaluatorClientError(
            "malformed_evaluator_response",
            "local guidance Oracle returned a malformed sample response",
        )
    batch = _reference_guided_batch_from_payload(
        response.get("batch"),
        codec=codec,
        mmap_arrays=mmap_arrays,
    )
    if batch.sample_settings_commitment_sha256 != spec.settings_sha256:
        raise ReferenceEvaluatorClientError(
            "malformed_evaluator_response",
            "local guidance Oracle changed the requested settings",
        )
    if len(batch.samples) != spec.sample_count:
        raise ReferenceEvaluatorClientError(
            "malformed_evaluator_response",
            "local guidance Oracle changed the requested sample count",
        )
    return batch


def query_reference_guided_completed_solution(
    context_path: Path | str,
    *,
    sample_id: str,
    method: str,
    args: Sequence[object] = (),
    kwargs: Mapping[str, object] | None = None,
    timeout_seconds: float = 300.0,
    mmap_arrays: bool = False,
) -> object:
    """Query one issued guided sample at Agent-selected coordinates.

    Only the public allowlist is callable. The opaque sample_id is resolved
    inside the controller; no answer locator or filesystem path crosses the
    loopback boundary.
    """

    for label, value in (("sample_id", sample_id), ("method", method)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"guided solution {label} cannot be empty")
    if not method.isidentifier() or method.startswith("_"):
        raise ValueError("guided solution method is not a public identifier")
    if isinstance(args, (str, bytes)) or not isinstance(args, Sequence):
        raise TypeError("guided solution args must be a sequence")
    call_kwargs = {} if kwargs is None else kwargs
    if not isinstance(call_kwargs, Mapping) or any(
        not isinstance(name, str) or not name for name in call_kwargs
    ):
        raise TypeError("guided solution kwargs need nonempty string keys")
    if not isinstance(mmap_arrays, bool):
        raise TypeError("mmap_arrays must be bool")

    context, workspace, evaluator = _reference_rpc_context(context_path)
    metadata = _read_json_object(context)
    oracle = metadata.get("guidance_oracle")
    if not isinstance(oracle, Mapping):
        raise ReferenceAgentRuntimeError(
            "this workspace does not expose a guidance Oracle"
        )
    allowed = oracle.get("development_solution_methods")
    if not isinstance(allowed, list) or method not in allowed:
        raise ReferenceAgentRuntimeError(
            "guided completed-solution method is not allowlisted"
        )
    codec = _PublicValueCodec(workspace / "arrays", create=False)
    response = _call_reference_rpc(
        endpoint=_required_string(oracle, "endpoint"),
        token=_required_string(evaluator, "token"),
        payload={
            "schema": REFERENCE_GUIDANCE_ORACLE_REQUEST_SCHEMA,
            "action": "query_completed_solution",
            "sample_id": sample_id,
            "method": method,
            "args": codec.encode(tuple(args)),
            "kwargs": codec.encode(dict(call_kwargs)),
        },
        timeout_seconds=_reference_rpc_timeout(
            timeout_seconds,
            evaluator=oracle,
            metadata_field="timeout_seconds",
            fallback_seconds=300.0,
            label="guidance Oracle timeout",
        ),
        expected_schema=REFERENCE_GUIDANCE_ORACLE_RESPONSE_SCHEMA,
    )
    expected = {"schema", "action", "sample_id", "method", "result"}
    if set(response) != expected or (
        response.get("action") != "query_completed_solution"
        or response.get("sample_id") != sample_id
        or response.get("method") != method
    ):
        raise ReferenceEvaluatorClientError(
            "malformed_evaluator_response",
            "local guidance Oracle returned a malformed solution response",
        )
    return codec.decode(response.get("result"), mmap_arrays=mmap_arrays)

def release_reference_completed_answers(
    context_path: Path | str,
    *,
    root_count: int,
    timeout_seconds: float = 300.0,
) -> Mapping[str, object]:
    """Request a bounded fresh completed-answer release into the workspace."""

    if (
        isinstance(root_count, bool)
        or not isinstance(root_count, int)
        or root_count <= 0
        or root_count > _MAX_RELEASE_ROOTS_PER_REQUEST
    ):
        raise ValueError(
            "release root_count must be in "
            f"[1, {_MAX_RELEASE_ROOTS_PER_REQUEST}]"
        )
    context, _workspace, evaluator = _reference_rpc_context(context_path)
    effective_timeout = _reference_rpc_timeout(
        timeout_seconds,
        evaluator=evaluator,
        metadata_field="release_timeout_seconds",
        fallback_seconds=300.0,
        label="release timeout",
    )
    return _call_reference_rpc(
        endpoint=_required_string(evaluator, "release_endpoint"),
        token=_required_string(evaluator, "token"),
        payload={
            "schema": REFERENCE_RELEASE_REQUEST_SCHEMA,
            "root_count": root_count,
        },
        timeout_seconds=effective_timeout,
        expected_schema=REFERENCE_RELEASE_RESPONSE_SCHEMA,
    )


def submit_reference_candidate_screen(
    context_path: Path | str,
    *,
    candidate_path: Path | str,
    candidate_id: str,
    requested_root_count: int,
    timeout_seconds: float = 30.0,
) -> Mapping[str, object]:
    """Seal candidate bytes and enqueue one bounded asynchronous screen."""

    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("candidate_id cannot be empty")
    if (
        isinstance(requested_root_count, bool)
        or not isinstance(requested_root_count, int)
        or requested_root_count <= 0
    ):
        raise ValueError("requested_root_count must be a positive integer")
    context, workspace, evaluator = _reference_rpc_context(context_path)
    effective_timeout = _reference_rpc_timeout(
        timeout_seconds,
        evaluator=evaluator,
        metadata_field="screen_submit_timeout_seconds",
        fallback_seconds=30.0,
        label="screen submit timeout",
    )
    candidate = Path(candidate_path)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    candidate = _resolve_agent_python_file(workspace, candidate)
    return _call_reference_rpc(
        endpoint=_required_string(evaluator, "screen_submit_endpoint"),
        token=_required_string(evaluator, "token"),
        payload={
            "schema": REFERENCE_SCREEN_SUBMIT_REQUEST_SCHEMA,
            "candidate_id": candidate_id,
            "candidate_path": candidate.relative_to(workspace).as_posix(),
            "requested_root_count": requested_root_count,
        },
        timeout_seconds=effective_timeout,
        expected_schema=REFERENCE_SCREEN_SUBMIT_RESPONSE_SCHEMA,
    )


def poll_reference_candidate_screen(
    context_path: Path | str,
    *,
    task_id: str,
    timeout_seconds: float = 30.0,
) -> Mapping[str, object]:
    """Poll one previously accepted asynchronous screen task."""

    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("screen task_id cannot be empty")
    context, _workspace, evaluator = _reference_rpc_context(context_path)
    effective_timeout = _reference_rpc_timeout(
        timeout_seconds,
        evaluator=evaluator,
        metadata_field="screen_poll_timeout_seconds",
        fallback_seconds=30.0,
        label="screen poll timeout",
    )
    return _call_reference_rpc(
        endpoint=_required_string(evaluator, "screen_poll_endpoint"),
        token=_required_string(evaluator, "token"),
        payload={
            "schema": REFERENCE_SCREEN_POLL_REQUEST_SCHEMA,
            "task_id": task_id,
        },
        timeout_seconds=effective_timeout,
        expected_schema=REFERENCE_SCREEN_POLL_RESPONSE_SCHEMA,
    )


def screen_reference_candidate(
    context_path: Path | str,
    *,
    candidate_path: Path | str,
    candidate_id: str,
    requested_root_count: int,
    timeout_seconds: float | None = None,
) -> Mapping[str, object]:
    """Screen one trial through the original blocking helper ABI."""

    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("candidate_id cannot be empty")
    if (
        isinstance(requested_root_count, bool)
        or not isinstance(requested_root_count, int)
        or requested_root_count <= 0
    ):
        raise ValueError("requested_root_count must be a positive integer")
    context, workspace, evaluator = _reference_rpc_context(context_path)
    effective_timeout = _reference_rpc_timeout(
        timeout_seconds,
        evaluator=evaluator,
        metadata_field="screen_timeout_seconds",
        fallback_seconds=300.0,
        label="screen timeout",
    )
    candidate = Path(candidate_path)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    candidate = _resolve_agent_python_file(workspace, candidate)
    return _call_reference_rpc(
        endpoint=_required_string(evaluator, "screen_endpoint"),
        token=_required_string(evaluator, "token"),
        payload={
            "schema": REFERENCE_SCREEN_REQUEST_SCHEMA,
            "candidate_id": candidate_id,
            "candidate_path": candidate.relative_to(workspace).as_posix(),
            "requested_root_count": requested_root_count,
        },
        timeout_seconds=effective_timeout,
        expected_schema=(
            REFERENCE_SCREEN_RESPONSE_SCHEMA,
            "macro-development-screen-response-v1",
        ),
    )


def _reference_rpc_timeout(
    requested: float | None,
    *,
    evaluator: Mapping[str, object],
    metadata_field: str,
    fallback_seconds: float,
    label: str,
) -> float:
    value: object = (
        evaluator.get(metadata_field, fallback_seconds)
        if requested is None
        else requested
    )
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _call_reference_rpc(
    *,
    endpoint: str,
    token: str,
    payload: Mapping[str, object],
    timeout_seconds: float,
    expected_schema: str | Sequence[str],
) -> Mapping[str, object]:
    body = _json_bytes(payload)
    request = Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            "X-PDE-Macro-Token": token,
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            error_payload = json.loads(exc.read().decode("utf-8"))
            error = error_payload["error"]
            code = str(error["code"])
            message = str(error["message"])
        except BaseException:
            code = "malformed_evaluator_error"
            message = "local evaluator returned a malformed error"
        raise ReferenceEvaluatorClientError(
            code,
            message,
            http_status=exc.code,
        ) from exc
    except URLError as exc:
        raise ReferenceEvaluatorClientError(
            "evaluator_unavailable",
            "local development evaluator is unavailable",
        ) from exc
    except TimeoutError as exc:
        raise ReferenceEvaluatorClientError(
            "evaluator_timeout",
            "local development evaluator timed out",
        ) from exc
    expected = (
        (expected_schema,)
        if isinstance(expected_schema, str)
        else tuple(expected_schema)
    )
    if not isinstance(result, Mapping) or result.get("schema") not in expected:
        raise ReferenceEvaluatorClientError(
            "malformed_evaluator_response",
            "local evaluator returned a malformed response",
        )
    return result



def _validated_reference_oracle_methods(
    methods: Sequence[str],
    *,
    label: str,
    factory: Callable[[Case], object] | None,
) -> tuple[str, ...]:
    if isinstance(methods, (str, bytes)) or not isinstance(methods, Sequence):
        raise TypeError(f"{label} numerical Oracle methods must be a sequence")
    values = tuple(methods)
    if any(
        not isinstance(method, str)
        or not method.isidentifier()
        or method.startswith("_")
        for method in values
    ):
        raise ValueError(
            f"{label} numerical Oracle methods need public identifiers"
        )
    if len(values) != len(set(values)):
        raise ValueError(f"{label} numerical Oracle methods must be unique")
    if (factory is None) != (not values):
        raise ValueError(
            f"{label} numerical Oracle factory and allowlist must appear together"
        )
    if factory is not None and not callable(factory):
        raise TypeError(f"{label} numerical Oracle factory must be callable")
    return values


def _dispatch_reference_guidance_oracle(
    oracle: DevelopmentGuidanceOracle,
    payload: object,
    *,
    codec: _PublicValueCodec,
) -> Mapping[str, object]:
    """Run the two-method guidance ABI without exposing controller internals."""

    if not isinstance(oracle, DevelopmentGuidanceOracle):
        raise TypeError(
            "guidance RPC needs a DevelopmentGuidanceOracle"
        )
    if not isinstance(payload, Mapping):
        raise ValueError("guidance Oracle request must be an object")
    if payload.get("schema") != REFERENCE_GUIDANCE_ORACLE_REQUEST_SCHEMA:
        raise ValueError("unknown guidance Oracle request schema")
    action = _required_string(payload, "action")
    if action == "sample":
        if set(payload) != {
            "schema",
            "action",
            "sample_count",
            "settings",
        }:
            raise ValueError(
                "guidance sample request fields do not match the schema"
            )
        sample_count = payload.get("sample_count")
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count <= 0
        ):
            raise ValueError(
                "guided sample_count must be a positive integer"
            )
        settings = codec.decode(
            payload.get("settings"),
            mmap_arrays=False,
        )
        if not isinstance(settings, dict):
            raise ValueError(
                "guided sample settings must be a mapping"
            )
        batch = oracle.sample(GuidedSampleSpec(sample_count, settings))
        return {
            "schema": REFERENCE_GUIDANCE_ORACLE_RESPONSE_SCHEMA,
            "action": action,
            "batch": _reference_guided_batch_payload(batch, codec),
        }
    if action != "query_completed_solution":
        raise ValueError("unknown guidance Oracle action")
    if set(payload) != {
        "schema",
        "action",
        "sample_id",
        "method",
        "args",
        "kwargs",
    }:
        raise ValueError(
            "guided solution request fields do not match the schema"
        )
    sample_id = _required_string(payload, "sample_id")
    method = _required_string(payload, "method")
    if method not in oracle.available_solution_methods:
        raise PermissionError("guided solution method is not allowlisted")
    args = codec.decode(payload.get("args"), mmap_arrays=False)
    kwargs = codec.decode(payload.get("kwargs"), mmap_arrays=False)
    if (
        not isinstance(args, tuple)
        or not isinstance(kwargs, dict)
        or any(
            not isinstance(name, str) or not name
            for name in kwargs
        )
    ):
        raise ValueError("guided solution arguments are invalid")
    try:
        result = oracle.query_completed_solution(
            sample_id,
            method,
            *args,
            **kwargs,
        )
    except KeyError as exc:
        raise PermissionError(
            "guided solution sample is not available"
        ) from exc
    return {
        "schema": REFERENCE_GUIDANCE_ORACLE_RESPONSE_SCHEMA,
        "action": action,
        "sample_id": sample_id,
        "method": method,
        "result": codec.encode(result),
    }

class ReferenceDevelopmentEvaluatorRPC:
    """Loopback JSON RPC for either legacy evaluation or fresh screening."""

    def __init__(
        self,
        evaluator: DevelopmentMacroEvaluator | DevelopmentSession,
        *,
        workspace: Path | str,
        bundle_attestor: ReferenceFilesystemBundleAttestor | None = None,
        round_index: int | None = None,
        agent_id: str | None = None,
        parent_sha256: str | None = None,
        controller_service_clock: ReferenceControllerServiceClock | None = None,
        oracle_cases: Sequence[Case] = (),
        public_session_factory: Callable[[Case], object] | None = None,
        public_session_methods: Sequence[str] = (),
        development_solution_session_factory: (
            Callable[[Case], object] | None
        ) = None,
        development_solution_session_methods: Sequence[str] = (),
        guidance_oracle: DevelopmentGuidanceOracle | None = None,
    ) -> None:
        if not isinstance(
            evaluator,
            (DevelopmentMacroEvaluator, DevelopmentSession),
        ):
            raise TypeError("reference RPC needs a supported development evaluator")
        if controller_service_clock is not None and not isinstance(
            controller_service_clock, ReferenceControllerServiceClock
        ):
            raise TypeError("reference RPC has an invalid controller service clock")

        if isinstance(evaluator, DevelopmentSession):
            if not isinstance(bundle_attestor, ReferenceFilesystemBundleAttestor):
                raise TypeError("fresh screen RPC needs a filesystem bundle attestor")
            if (
                isinstance(round_index, bool)
                or not isinstance(round_index, int)
                or round_index < 0
            ):
                raise ValueError("fresh screen RPC needs a nonnegative round")
            if not isinstance(agent_id, str) or not agent_id.strip():
                raise ValueError("fresh screen RPC needs a nonempty agent_id")
            _require_sha256(parent_sha256, "fresh screen parent")

        self.evaluator = evaluator
        self.workspace = Path(workspace).resolve()
        self.token = secrets.token_urlsafe(32)
        handler = _reference_handler()
        self._server = _ReferenceHTTPServer(
            ("127.0.0.1", 0),
            handler,
            evaluator=evaluator,
            workspace=self.workspace,
            token=self.token,
            bundle_attestor=bundle_attestor,
            round_index=round_index,
            agent_id=agent_id,
            parent_sha256=parent_sha256,
            controller_service_clock=controller_service_clock,
            oracle_cases=oracle_cases,
            public_session_factory=public_session_factory,
            public_session_methods=public_session_methods,
            development_solution_session_factory=(
                development_solution_session_factory
            ),
            development_solution_session_methods=(
                development_solution_session_methods
            ),
            guidance_oracle=guidance_oracle,
        )
        host, port = self._server.server_address
        self.endpoint = f"http://{host}:{port}/evaluate"
        self.screen_endpoint = f"http://{host}:{port}/screen"
        self.screen_submit_endpoint = f"http://{host}:{port}/screen/submit"
        self.screen_poll_endpoint = f"http://{host}:{port}/screen/poll"
        self.release_endpoint = f"http://{host}:{port}/release"
        self.numerical_oracle_endpoint = f"http://{host}:{port}/oracle"
        self.guidance_oracle_endpoint = f"http://{host}:{port}/guidance"
        self._thread: Thread | None = None

    @property
    def execution_assurance(self) -> ExecutionAssurance:
        return ExecutionAssurance.REFERENCE_IN_PROCESS

    @property
    def formal_claims_allowed(self) -> bool:
        return False

    @property
    def limitations(self) -> tuple[str, ...]:
        return _REFERENCE_LIMITATIONS

    def __enter__(self):
        if self._thread is not None:
            raise RuntimeError("reference evaluator RPC is single-use")
        self._thread = Thread(
            target=self._server.serve_forever,
            name="reference-macro-evaluator",
            daemon=True,
        )
        self._thread.start()
        return self

    def revoke(self) -> None:
        self._server.revoke()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.revoke()
        self._server.shutdown()
        self._server.server_close()
        self._server.drain_async_screens()
        assert self._thread is not None
        self._thread.join(timeout=5.0)



class _ReferenceGuidanceHTTPServer(ThreadingHTTPServer):
    """Parent-owned inference-only guidance endpoint for one GPU worker."""

    daemon_threads = False
    block_on_close = True

    def __init__(
        self,
        address,
        handler,
        *,
        oracle: DevelopmentGuidanceOracle,
        workspace: Path,
        token: str,
    ) -> None:
        super().__init__(address, handler)
        if not isinstance(oracle, DevelopmentGuidanceOracle):
            raise TypeError(
                "worker guidance RPC needs a DevelopmentGuidanceOracle"
            )
        self.oracle = oracle
        self.workspace = workspace
        self.token = token
        self._call_lock = Lock()
        self._lifecycle_lock = Lock()
        self._accepting = True

    def call(self, payload: object) -> Mapping[str, object]:
        codec = _PublicValueCodec(self.workspace / "arrays", create=False)
        with self._call_lock:
            self.require_accepting()
            return _dispatch_reference_guidance_oracle(
                self.oracle,
                payload,
                codec=codec,
            )

    def revoke(self) -> None:
        with self._lifecycle_lock:
            self._accepting = False

    def require_accepting(self) -> None:
        with self._lifecycle_lock:
            if not self._accepting:
                raise ReferenceAgentRuntimeError(
                    "worker guidance capability is expired"
                )


def _reference_guidance_handler():
    class Handler(BaseHTTPRequestHandler):
        server: _ReferenceGuidanceHTTPServer

        def do_POST(self) -> None:
            if self.path != "/guidance":
                self._error(404, "not_found", "unknown guidance route")
                return
            if self.headers.get("X-PDE-Macro-Token") != self.server.token:
                self._error(403, "permission_denied", "invalid guidance token")
                return
            raw_length = self.headers.get("Content-Length", "")
            try:
                length = int(raw_length)
            except ValueError:
                self._error(400, "invalid_request", "invalid content length")
                return
            if length <= 0 or length > _MAX_RPC_BYTES:
                self._error(400, "invalid_request", "invalid request size")
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                response = self.server.call(payload)
            except PermissionError:
                self._error(
                    403,
                    "guidance_access_denied",
                    "guidance Oracle access was denied",
                )
                return
            except (
                KeyError,
                ValueError,
                TypeError,
                ReferenceAgentRuntimeError,
            ) as exc:
                self._error(400, "invalid_request", str(exc))
                return
            except BaseException as exc:
                self._error(
                    500,
                    "guidance_failed",
                    f"guidance Oracle call failed: {type(exc).__name__}",
                )
                return
            self._send(200, response)

        def _error(self, status: int, code: str, message: str) -> None:
            self._send(
                status,
                {
                    "schema": REFERENCE_EVALUATION_ERROR_SCHEMA,
                    "error": {"code": code, "message": message},
                },
            )

        def _send(self, status: int, payload: Mapping[str, object]) -> None:
            body = _json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args) -> None:
            return

    return Handler


class ReferenceGuidanceOracleRPC:
    """Host exactly one Agent's frozen development guidance in the parent."""

    def __init__(
        self,
        oracle: DevelopmentGuidanceOracle,
        *,
        workspace: Path | str,
    ) -> None:
        if not isinstance(oracle, DevelopmentGuidanceOracle):
            raise TypeError(
                "worker guidance RPC needs a DevelopmentGuidanceOracle"
            )
        self.oracle = oracle
        self.workspace = Path(workspace).resolve()
        _ensure_private_directory(self.workspace)
        _PublicValueCodec(self.workspace / "arrays")
        self.token = secrets.token_urlsafe(32)
        self._server = _ReferenceGuidanceHTTPServer(
            ("127.0.0.1", 0),
            _reference_guidance_handler(),
            oracle=oracle,
            workspace=self.workspace,
            token=self.token,
        )
        host, port = self._server.server_address
        self.endpoint = f"http://{host}:{port}/guidance"
        self._thread: Thread | None = None
        self._closed = False

    def worker_capability(self) -> Mapping[str, object]:
        if self._closed:
            raise RuntimeError("worker guidance RPC is closed")
        return {
            "schema": REFERENCE_GUIDANCE_WORKER_CAPABILITY_SCHEMA,
            "endpoint": self.endpoint,
            "token": self.token,
            "snapshot_sha256": self.oracle.snapshot.snapshot_sha256,
            "campaign_guidance_training_wall_seconds": (
                self.oracle.snapshot.campaign_guidance_training_wall_seconds
            ),
            "development_solution_methods": list(
                self.oracle.available_solution_methods
            ),
        }

    def __enter__(self):
        if self._thread is not None or self._closed:
            raise RuntimeError("worker guidance RPC is single-use")
        self._thread = Thread(
            target=self._server.serve_forever,
            name="reference-worker-guidance-oracle",
            daemon=True,
        )
        self._thread.start()
        return self

    def revoke(self) -> None:
        self._server.revoke()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.revoke()
            if self._thread is not None:
                self._server.shutdown()
            self._server.server_close()
            if self._thread is not None:
                self._thread.join(timeout=5.0)
                if self._thread.is_alive():
                    raise RuntimeError(
                        "worker guidance RPC thread did not stop"
                    )
        finally:
            self.oracle.close()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class ReferenceGuidanceOracleProxy(DevelopmentGuidanceOracle):
    """Worker-side inference-only proxy; it has no training/resolver API."""

    def __init__(
        self,
        *,
        endpoint: str,
        token: str,
        transport_workspace: Path,
        snapshot_sha256: str,
        campaign_guidance_training_wall_seconds: float,
        development_solution_methods: Sequence[str],
        timeout_seconds: float = 300.0,
    ) -> None:
        parsed = urlparse(endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
            or parsed.path != "/guidance"
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ReferenceAgentRuntimeError(
                "worker guidance endpoint must be loopback HTTP"
            )
        if not isinstance(token, str) or not token:
            raise ReferenceAgentRuntimeError(
                "worker guidance capability token is invalid"
            )
        _require_sha256(snapshot_sha256, "worker guidance snapshot")
        methods = tuple(development_solution_methods)
        if not methods or any(
            not isinstance(method, str)
            or not method.isidentifier()
            or method.startswith("_")
            for method in methods
        ) or len(methods) != len(set(methods)):
            raise ReferenceAgentRuntimeError(
                "worker guidance method allowlist is invalid"
            )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, Real)
            or not isfinite(float(timeout_seconds))
            or float(timeout_seconds) <= 0.0
        ):
            raise ValueError(
                "worker guidance timeout must be finite and positive"
            )
        self._endpoint = endpoint
        self._token = token
        self._transport_workspace = Path(transport_workspace).resolve()
        self._snapshot = CandidateGuidanceSnapshot(
            snapshot_sha256,
            campaign_guidance_training_wall_seconds,
        )
        self._methods = methods
        self._timeout_seconds = float(timeout_seconds)
        self._lock = Lock()
        self._operation_lock = Lock()
        self._closed = False
        self._call_count = 0
        self._generation_wall_seconds = 0.0
        self._solution_query_wall_seconds = 0.0

    @classmethod
    def from_worker_capability(
        cls,
        payload: object,
        *,
        worker_root: Path | str,
        timeout_seconds: float = 300.0,
    ) -> ReferenceGuidanceOracleProxy | None:
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise ReferenceAgentRuntimeError(
                "worker guidance capability must be an object or null"
            )
        expected = {
            "schema",
            "endpoint",
            "token",
            "snapshot_sha256",
            "campaign_guidance_training_wall_seconds",
            "development_solution_methods",
        }
        if set(payload) != expected or payload.get(
            "schema"
        ) != REFERENCE_GUIDANCE_WORKER_CAPABILITY_SCHEMA:
            raise ReferenceAgentRuntimeError(
                "worker guidance capability fields changed"
            )
        methods = payload.get("development_solution_methods")
        if not isinstance(methods, list):
            raise ReferenceAgentRuntimeError(
                "worker guidance method allowlist is invalid"
            )
        root = Path(worker_root).resolve()
        transport = (root / "guidance_transport").resolve()
        if transport.parent != root:
            raise ReferenceAgentRuntimeError(
                "worker guidance transport path escaped its worker"
            )
        return cls(
            endpoint=_required_string(payload, "endpoint"),
            token=_required_string(payload, "token"),
            transport_workspace=transport,
            snapshot_sha256=_required_sha256(
                payload, "snapshot_sha256"
            ),
            campaign_guidance_training_wall_seconds=(
                _required_nonnegative_float(
                    payload, "campaign_guidance_training_wall_seconds"
                )
            ),
            development_solution_methods=methods,
            timeout_seconds=timeout_seconds,
        )

    @property
    def snapshot(self) -> CandidateGuidanceSnapshot:
        return self._snapshot

    @property
    def available_solution_methods(self) -> tuple[str, ...]:
        return self._methods

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count

    @property
    def campaign_guided_generation_wall_seconds(self) -> float:
        with self._lock:
            return self._generation_wall_seconds

    @property
    def campaign_solution_query_wall_seconds(self) -> float:
        with self._lock:
            return self._solution_query_wall_seconds

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def _codec(self) -> _PublicValueCodec:
        return _PublicValueCodec(
            self._transport_workspace / "arrays",
            create=False,
        )

    def sample(self, spec: GuidedSampleSpec) -> CandidateGuidedSampleBatch:
        with self._operation_lock:
            return self._sample_open(spec)

    def _sample_open(
        self, spec: GuidedSampleSpec
    ) -> CandidateGuidedSampleBatch:
        if not isinstance(spec, GuidedSampleSpec):
            raise TypeError("guidance sample needs a GuidedSampleSpec")
        with self._lock:
            if self._closed:
                raise RuntimeError("worker guidance proxy is closed")
            self._call_count += 1
        codec = self._codec()
        response = _call_reference_rpc(
            endpoint=self._endpoint,
            token=self._token,
            payload={
                "schema": REFERENCE_GUIDANCE_ORACLE_REQUEST_SCHEMA,
                "action": "sample",
                "sample_count": spec.sample_count,
                "settings": codec.encode(dict(spec.settings)),
            },
            timeout_seconds=self._timeout_seconds,
            expected_schema=REFERENCE_GUIDANCE_ORACLE_RESPONSE_SCHEMA,
        )
        if set(response) != {"schema", "action", "batch"} or response.get(
            "action"
        ) != "sample":
            raise ReferenceEvaluatorClientError(
                "malformed_evaluator_response",
                "worker guidance returned a malformed sample response",
            )
        batch = _reference_guided_batch_from_payload(
            response.get("batch"),
            codec=codec,
            mmap_arrays=False,
        )
        if (
            batch.sample_settings_commitment_sha256
            != spec.settings_sha256
            or len(batch.samples) != spec.sample_count
            or batch.snapshot_sha256 != self.snapshot.snapshot_sha256
        ):
            raise ReferenceEvaluatorClientError(
                "malformed_evaluator_response",
                "worker guidance changed the requested sample identity",
            )
        with self._lock:
            self._generation_wall_seconds += (
                batch.campaign_guided_generation_wall_seconds
            )
        return batch

    def query_completed_solution(
        self,
        sample_id: str,
        method: str,
        /,
        *args: object,
        **kwargs: object,
    ) -> object:
        with self._operation_lock:
            return self._query_completed_solution_open(
                sample_id,
                method,
                *args,
                **kwargs,
            )

    def _query_completed_solution_open(
        self,
        sample_id: str,
        method: str,
        /,
        *args: object,
        **kwargs: object,
    ) -> object:
        with self._lock:
            if self._closed:
                raise RuntimeError("worker guidance proxy is closed")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("guided sample_id cannot be empty")
        if method not in self._methods:
            raise PermissionError(
                "guided solution method is not allowlisted"
            )
        started = time.perf_counter()
        try:
            return self._query_completed_solution_rpc(
                sample_id,
                method,
                *args,
                **kwargs,
            )
        finally:
            elapsed = time.perf_counter() - started
            with self._lock:
                self._solution_query_wall_seconds += elapsed

    def _query_completed_solution_rpc(
        self,
        sample_id: str,
        method: str,
        /,
        *args: object,
        **kwargs: object,
    ) -> object:
        codec = self._codec()
        response = _call_reference_rpc(
            endpoint=self._endpoint,
            token=self._token,
            payload={
                "schema": REFERENCE_GUIDANCE_ORACLE_REQUEST_SCHEMA,
                "action": "query_completed_solution",
                "sample_id": sample_id,
                "method": method,
                "args": codec.encode(tuple(args)),
                "kwargs": codec.encode(dict(kwargs)),
            },
            timeout_seconds=self._timeout_seconds,
            expected_schema=REFERENCE_GUIDANCE_ORACLE_RESPONSE_SCHEMA,
        )
        expected = {"schema", "action", "sample_id", "method", "result"}
        if set(response) != expected or (
            response.get("action") != "query_completed_solution"
            or response.get("sample_id") != sample_id
            or response.get("method") != method
        ):
            raise ReferenceEvaluatorClientError(
                "malformed_evaluator_response",
                "worker guidance returned a malformed solution response",
            )
        return codec.decode(response.get("result"), mmap_arrays=False)

    def close(self) -> None:
        """Drop this worker capability without releasing its parent Oracle."""

        with self._operation_lock:
            with self._lock:
                self._closed = True


def _sanitized_async_screen_error(
    exception: BaseException,
) -> Mapping[str, object]:
    """Map worker failures to the same bounded public errors as sync screening."""

    if isinstance(exception, DevelopmentCreditExceeded):
        status, code, message = (
            409,
            "root_credit_exceeded",
            "screen root credit limit exceeded",
        )
    elif (
        isinstance(exception, DevelopmentSessionError)
        and str(exception) in _PUBLIC_SCREEN_SESSION_ERRORS
    ):
        status, code, message = 409, "screen_rejected", str(exception)
    elif isinstance(exception, _PrivateScreenExecutionError):
        status, code, message = (
            500,
            "screen_failed",
            f"screen_failed:{exception.exception_type}",
        )
    else:
        status, code, message = (
            500,
            "screen_failed",
            "screen_failed:InternalError",
        )
    return {
        "code": code,
        "message": message,
        "http_status": status,
    }


class _ReferenceHTTPServer(ThreadingHTTPServer):
    # Accepted evaluations mutate the shared session that is serialized after
    # this server closes. Drain handlers and accepted async work before ledger.
    daemon_threads = False
    block_on_close = True

    def __init__(
        self,
        address,
        handler,
        *,
        evaluator: DevelopmentMacroEvaluator | DevelopmentSession,
        workspace: Path,
        token: str,
        bundle_attestor: ReferenceFilesystemBundleAttestor | None,
        round_index: int | None,
        agent_id: str | None,
        parent_sha256: str | None,
        controller_service_clock: ReferenceControllerServiceClock | None,
        oracle_cases: Sequence[Case],
        public_session_factory: Callable[[Case], object] | None,
        public_session_methods: Sequence[str],
        development_solution_session_factory: Callable[[Case], object] | None,
        development_solution_session_methods: Sequence[str],
        guidance_oracle: DevelopmentGuidanceOracle | None,
    ) -> None:
        super().__init__(address, handler)
        self.evaluator = evaluator
        self.workspace = workspace
        self.token = token
        self.bundle_attestor = bundle_attestor
        self.round_index = round_index
        self.agent_id = agent_id
        self.parent_sha256 = parent_sha256
        self.evaluation_lock = Lock()
        self._attestation_lock = Lock()
        self.controller_service_clock = controller_service_clock
        self._lifecycle_lock = Lock()
        self._accepting = True
        self._screen_inflight = 0
        self._screen_tasks: dict[str, _ReferenceAsyncScreenTask] = {}
        self._screen_condition = Condition(Lock())
        self._screen_queue: Queue[_ReferenceAsyncScreenTask | None] | None = None
        self._screen_worker: Thread | None = None
        self.public_session_factory = public_session_factory
        self.public_session_methods = _validated_reference_oracle_methods(
            public_session_methods,
            label="equation",
            factory=public_session_factory,
        )
        self.development_solution_session_factory = (
            development_solution_session_factory
        )
        self.development_solution_session_methods = (
            _validated_reference_oracle_methods(
                development_solution_session_methods,
                label="development solution",
                factory=development_solution_session_factory,
            )
        )
        if set(self.public_session_methods) & set(
            self.development_solution_session_methods
        ):
            raise ValueError(
                "equation and development solution Oracle methods must be disjoint"
            )
        if guidance_oracle is not None and not isinstance(
            guidance_oracle, DevelopmentGuidanceOracle
        ):
            raise TypeError(
                "reference RPC guidance must be a DevelopmentGuidanceOracle"
            )
        self.guidance_oracle = guidance_oracle
        self._oracle_cases: dict[str, Case] = {}
        self.replace_oracle_cases(oracle_cases)
        if isinstance(evaluator, DevelopmentSession):
            self._screen_queue = Queue(maxsize=_MAX_ASYNC_SCREEN_INFLIGHT)
            self._screen_worker = Thread(
                target=self._run_screen_worker,
                name="reference-macro-screen-worker",
                daemon=False,
            )
            self._screen_worker.start()

    @property
    def numerical_oracle_available(self) -> bool:
        return bool(
            self.public_session_methods
            or self.development_solution_session_methods
        )

    @property
    def guidance_oracle_available(self) -> bool:
        return self.guidance_oracle is not None

    def replace_oracle_cases(self, cases: Sequence[Case]) -> None:
        """Atomically bind the one currently resident development batch."""

        if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence):
            raise TypeError("numerical Oracle cases must be a sequence")
        replacement: dict[str, Case] = {}
        for case in cases:
            if not isinstance(case, Case):
                raise TypeError("numerical Oracle cases must contain Case values")
            candidate_id = candidate_facing_case(case).case_id
            if candidate_id in replacement:
                raise ReferenceAgentRuntimeError(
                    "numerical Oracle case ids must be unique"
                )
            replacement[candidate_id] = case
        self._oracle_cases = replacement

    def call_numerical_oracle(
        self,
        payload: object,
    ) -> Mapping[str, object]:
        if not isinstance(payload, Mapping):
            raise ValueError("numerical Oracle request must be an object")
        expected = {"schema", "namespace", "case_id", "method", "args", "kwargs"}
        if set(payload) != expected:
            raise ValueError(
                "numerical Oracle request fields do not match the schema"
            )
        if payload.get("schema") != REFERENCE_NUMERICAL_ORACLE_REQUEST_SCHEMA:
            raise ValueError("unknown numerical Oracle request schema")
        namespace = _required_string(payload, "namespace")
        case_id = _required_string(payload, "case_id")
        method_name = _required_string(payload, "method")
        if namespace == "equation":
            factory = self.public_session_factory
            methods = self.public_session_methods
        elif namespace == "development_solution":
            factory = self.development_solution_session_factory
            methods = self.development_solution_session_methods
        else:
            raise ValueError("unknown numerical Oracle namespace")
        if method_name not in methods or factory is None:
            raise PermissionError(
                "numerical Oracle method is not allowlisted in this namespace"
            )
        codec = _PublicValueCodec(self.workspace / "arrays", create=False)
        args = codec.decode(payload.get("args"), mmap_arrays=False)
        kwargs = codec.decode(payload.get("kwargs"), mmap_arrays=False)
        if not isinstance(args, tuple) or not isinstance(kwargs, dict) or any(
            not isinstance(name, str) or not name for name in kwargs
        ):
            raise ValueError("numerical Oracle arguments are invalid")

        with self.evaluation_lock:
            self.require_accepting()
            case = self._oracle_cases.get(case_id)
            if case is None:
                raise PermissionError(
                    "numerical Oracle case is not registered for development"
                )
            if (
                namespace == "development_solution"
                and case.visibility is not CaseVisibility.DEVELOPMENT
            ):
                raise PermissionError(
                    "completed-solution Oracle access is DEVELOPMENT-only"
                )
            clock = self.controller_service_clock
            service_token = None if clock is None else clock.begin_service()
            try:
                session = factory(case)
                if session is None:
                    raise TypeError("numerical Oracle session factory returned None")
                try:
                    method = getattr(session, method_name, None)
                    if not callable(method):
                        raise TypeError(
                            "numerical Oracle session is missing its declared method"
                        )
                    result = method(*args, **kwargs)
                finally:
                    close = getattr(session, "close", None)
                    if callable(close):
                        close()
                result_node = codec.encode(result)
            finally:
                if clock is not None and service_token is not None:
                    clock.end_service(service_token, credit=True)
        return {
            "schema": REFERENCE_NUMERICAL_ORACLE_RESPONSE_SCHEMA,
            "namespace": namespace,
            "case_id": case_id,
            "method": method_name,
            "result": result_node,
        }

    def call_guidance_oracle(
        self,
        payload: object,
    ) -> Mapping[str, object]:
        oracle = self.guidance_oracle
        if oracle is None:
            raise ReferenceAgentRuntimeError("guidance Oracle is unavailable")
        codec = _PublicValueCodec(self.workspace / "arrays", create=False)
        with self.evaluation_lock:
            self.require_accepting()
            clock = self.controller_service_clock
            service_token = None if clock is None else clock.begin_service()
            try:
                return _dispatch_reference_guidance_oracle(
                    oracle,
                    payload,
                    codec=codec,
                )
            finally:
                if clock is not None and service_token is not None:
                    clock.end_service(service_token, credit=True)

    def revoke(self) -> None:
        with self._lifecycle_lock:
            self._accepting = False

    def require_accepting(self) -> None:
        with self._lifecycle_lock:
            if not self._accepting:
                raise ReferenceAgentRuntimeError(
                    "reference evaluator capability is expired"
                )

    def _reserve_screen_slot(self) -> None:
        with self._lifecycle_lock:
            if not self._accepting:
                raise ReferenceAgentRuntimeError(
                    "reference evaluator capability is expired"
                )
            if self._screen_inflight >= _MAX_ASYNC_SCREEN_INFLIGHT:
                raise _ReferenceAsyncScreenQueueFull
            self._screen_inflight += 1

    def _release_screen_slot(self) -> None:
        with self._lifecycle_lock:
            self._screen_inflight -= 1
            if self._screen_inflight < 0:
                raise RuntimeError("async screen accounting became negative")

    def submit_screen(
        self,
        payload: object,
        *,
        request_schema: str,
    ) -> Mapping[str, object]:
        evaluator = self.evaluator
        if not isinstance(evaluator, DevelopmentSession):
            raise ReferenceAgentRuntimeError("fresh screen is unavailable")
        if not isinstance(payload, Mapping):
            raise ValueError("screen request must be an object")
        allowed = {
            "schema",
            "candidate_id",
            "candidate_path",
            "requested_root_count",
        }
        if set(payload) != allowed:
            raise ValueError("screen request fields do not match the schema")
        if payload.get("schema") != request_schema:
            raise ValueError("unknown screen request schema")
        raw_candidate_id = _required_string(payload, "candidate_id")
        candidate_id = _canonical_candidate_id(
            raw_candidate_id,
            round_index=self.round_index,
            agent_id=self.agent_id,
        )
        candidate_path = _resolve_agent_python_file(
            self.workspace,
            self.workspace / _required_string(payload, "candidate_path"),
        )
        root_count = payload.get("requested_root_count")
        if (
            isinstance(root_count, bool)
            or not isinstance(root_count, int)
            or root_count <= 0
        ):
            raise ValueError(
                "requested_root_count must be a positive integer"
            )
        if evaluator.open_search:
            if root_count > evaluator.screen_root_credit_limit:
                raise ValueError(
                    "requested_root_count exceeds the per-request capacity"
                )
        elif root_count not in evaluator.screen_root_stages:
            raise ValueError(
                "requested_root_count is not a configured stage"
            )
        attestor = self.bundle_attestor
        round_index = self.round_index
        agent_id = self.agent_id
        parent_sha256 = self.parent_sha256
        if (
            not isinstance(attestor, ReferenceFilesystemBundleAttestor)
            or not isinstance(round_index, int)
            or not isinstance(agent_id, str)
            or not isinstance(parent_sha256, str)
        ):
            raise ReferenceAgentRuntimeError(
                "fresh screen controller identity is unavailable"
            )

        self._reserve_screen_slot()
        try:
            # Attestation is deliberately part of submit, before enqueue: later
            # candidate-file edits cannot change the accepted task.
            with self._attestation_lock:
                attestation = attestor.attest(
                    AgentBundleSubmission(candidate_id, str(candidate_path)),
                    round_index=round_index,
                    agent_id=agent_id,
                    parent_sha256=parent_sha256,
                )
            candidate_ref = candidate_ref_from_attestation(attestation)
            task = _ReferenceAsyncScreenTask(
                task_id=secrets.token_urlsafe(18),
                candidate=_reference_candidate_submission(candidate_ref),
                requested_root_count=root_count,
            )
            queue = self._screen_queue
            if queue is None:
                raise RuntimeError("async screen worker is unavailable")
            with self._screen_condition:
                self._screen_tasks[task.task_id] = task
                self._screen_condition.notify_all()
            try:
                queue.put_nowait(task)
            except BaseException:
                with self._screen_condition:
                    self._screen_tasks.pop(task.task_id, None)
                raise
        except BaseException:
            self._release_screen_slot()
            raise
        return {
            "schema": REFERENCE_SCREEN_SUBMIT_RESPONSE_SCHEMA,
            "task_id": task.task_id,
            "status": task.status,
            "candidate_sha256": task.candidate.ref.bundle_sha256,
            "requested_root_count": root_count,
        }

    def poll_screen(self, payload: object) -> Mapping[str, object]:
        if not isinstance(payload, Mapping):
            raise ValueError("screen poll request must be an object")
        if set(payload) != {"schema", "task_id"}:
            raise ValueError("screen poll fields do not match the schema")
        if payload.get("schema") != REFERENCE_SCREEN_POLL_REQUEST_SCHEMA:
            raise ValueError("unknown screen poll request schema")
        task_id = _required_string(payload, "task_id")
        with self._screen_condition:
            task = self._screen_tasks.get(task_id)
            if task is None:
                raise ReferenceAgentRuntimeError("unknown async screen task")
            return self._screen_poll_payload(task)

    def wait_screen(self, task_id: str) -> Mapping[str, object]:
        with self._screen_condition:
            while True:
                task = self._screen_tasks.get(task_id)
                if task is None:
                    raise ReferenceAgentRuntimeError("unknown async screen task")
                if task.status in {"completed", "failed"}:
                    return self._screen_poll_payload(task)
                self._screen_condition.wait()

    @staticmethod
    def _screen_poll_payload(
        task: _ReferenceAsyncScreenTask,
    ) -> Mapping[str, object]:
        return {
            "schema": REFERENCE_SCREEN_POLL_RESPONSE_SCHEMA,
            "task_id": task.task_id,
            "status": task.status,
            "candidate_sha256": task.candidate.ref.bundle_sha256,
            "requested_root_count": task.requested_root_count,
            "result": task.response,
            "error": task.error,
        }

    def _run_screen_worker(self) -> None:
        queue = self._screen_queue
        evaluator = self.evaluator
        if queue is None or not isinstance(evaluator, DevelopmentSession):
            return
        while True:
            task = queue.get()
            try:
                if task is None:
                    return
                with self._screen_condition:
                    task.status = "running"
                    self._screen_condition.notify_all()
                charged_before = evaluator.screen_charged_root_credits
                clock = self.controller_service_clock
                service_token = None
                try:
                    with self.evaluation_lock:
                        service_token = None if clock is None else clock.begin_service()
                        try:
                            try:
                                response = evaluator.screen(
                                    task.candidate,
                                    requested_root_count=task.requested_root_count,
                                )
                            except DevelopmentCreditExceeded:
                                raise
                            except DevelopmentSessionError as exc:
                                if str(exc) in _PUBLIC_SCREEN_SESSION_ERRORS:
                                    raise
                                raise _PrivateScreenExecutionError(exc) from exc
                            except BaseException as exc:
                                raise _PrivateScreenExecutionError(exc) from exc
                            task.response = response.public_payload()
                        finally:
                            if clock is not None and service_token is not None:
                                clock.end_service(
                                    service_token,
                                    credit=(
                                        evaluator.screen_charged_root_credits
                                        > charged_before
                                    ),
                                )
                except BaseException as exc:
                    task.error = _sanitized_async_screen_error(exc)
                    task.status = "failed"
                else:
                    task.status = "completed"
                with self._screen_condition:
                    self._screen_condition.notify_all()
                self._release_screen_slot()
            finally:
                queue.task_done()

    def drain_async_screens(self) -> None:
        queue = self._screen_queue
        worker = self._screen_worker
        if queue is None or worker is None:
            return
        queue.join()
        queue.put(None)
        worker.join(timeout=30.0)
        if worker.is_alive():
            raise RuntimeError("async screen worker did not drain")
        self._screen_queue = None
        self._screen_worker = None



def _reference_handler():
    class Handler(BaseHTTPRequestHandler):
        server: _ReferenceHTTPServer

        def do_POST(self) -> None:
            routes = {
                "/evaluate",
                "/screen",
                "/screen/submit",
                "/screen/poll",
                "/release",
                "/oracle",
                "/guidance",
            }
            if self.path not in routes:
                self._error(404, "not_found", "unknown local evaluator route")
                return
            if (
                self.path == "/evaluate"
                and not isinstance(
                    self.server.evaluator,
                    DevelopmentMacroEvaluator,
                )
            ):
                self._error(404, "not_found", "legacy evaluation is unavailable")
                return
            if (
                self.path not in {"/evaluate", "/oracle", "/guidance"}
                and not isinstance(self.server.evaluator, DevelopmentSession)
            ):
                self._error(404, "not_found", "fresh development is unavailable")
                return
            if (
                self.path == "/oracle"
                and not self.server.numerical_oracle_available
            ):
                self._error(404, "not_found", "numerical Oracle is unavailable")
                return
            if (
                self.path == "/guidance"
                and not self.server.guidance_oracle_available
            ):
                self._error(404, "not_found", "guidance Oracle is unavailable")
                return
            if self.headers.get("X-PDE-Macro-Token") != self.server.token:
                self._error(403, "permission_denied", "invalid evaluator token")
                return
            raw_length = self.headers.get("Content-Length", "")
            try:
                length = int(raw_length)
            except ValueError:
                self._error(400, "invalid_request", "invalid content length")
                return
            if length <= 0 or length > _MAX_RPC_BYTES:
                self._error(400, "invalid_request", "invalid request size")
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if self.path == "/evaluate":
                    response = self._evaluate(payload)
                elif self.path == "/screen":
                    response = self._screen(payload)
                elif self.path == "/screen/submit":
                    response = self.server.submit_screen(
                        payload,
                        request_schema=REFERENCE_SCREEN_SUBMIT_REQUEST_SCHEMA,
                    )
                elif self.path == "/screen/poll":
                    response = self.server.poll_screen(payload)
                elif self.path == "/oracle":
                    response = self.server.call_numerical_oracle(payload)
                elif self.path == "/guidance":
                    response = self.server.call_guidance_oracle(payload)
                else:
                    response = self._release(payload)
            except CaseCreditExceeded as exc:
                self._error(409, "case_credit_exceeded", str(exc))
                return
            except _ReferenceAsyncScreenQueueFull:
                self._error(
                    429,
                    "screen_queue_full",
                    "async screen inflight limit reached",
                )
                return
            except _ReferenceAsyncScreenTaskFailed as exc:
                self._error(exc.http_status, exc.code, exc.message)
                return
            except _PrivateScreenExecutionError as exc:
                self._error(
                    500,
                    "release_failed",
                    f"release_failed:{exc.exception_type}",
                )
                return
            except DevelopmentCreditExceeded:
                code = (
                    "release_root_credit_exceeded"
                    if self.path == "/release"
                    else "root_credit_exceeded"
                )
                message = (
                    "completed-answer release root credit limit exceeded"
                    if self.path == "/release"
                    else "screen root credit limit exceeded"
                )
                self._error(409, code, message)
                return
            except DevelopmentSessionError as exc:
                self._error(409, "screen_rejected", str(exc))
                return
            except PermissionError:
                if self.path == "/oracle":
                    self._error(
                        403,
                        "oracle_access_denied",
                        "numerical Oracle access was denied",
                    )
                elif self.path == "/guidance":
                    self._error(
                        403,
                        "guidance_access_denied",
                        "guidance Oracle access was denied",
                    )
                else:
                    self._error(
                        403,
                        "finite_truth_disabled",
                        "finite macro truth views are disabled",
                    )
                return
            except (KeyError, ValueError, TypeError, ReferenceAgentRuntimeError) as exc:
                self._error(400, "invalid_request", str(exc))
                return
            except BaseException as exc:
                self._error(
                    500,
                    "evaluation_failed",
                    f"local candidate evaluation failed: {type(exc).__name__}",
                )
                return
            self._send(200, response)

        def _evaluate(self, payload: object) -> Mapping[str, object]:
            evaluator = self.server.evaluator
            if not isinstance(evaluator, DevelopmentMacroEvaluator):
                raise ReferenceAgentRuntimeError("legacy evaluator is unavailable")
            if not isinstance(payload, Mapping):
                raise ValueError("evaluation request must be an object")
            allowed = {
                "schema",
                "candidate_id",
                "candidate_path",
                "case_ids",
                "return_finite_truth",
            }
            if set(payload) != allowed:
                raise ValueError("evaluation request fields do not match the schema")
            if payload.get("schema") != REFERENCE_EVALUATION_REQUEST_SCHEMA:
                raise ValueError("unknown evaluation request schema")
            candidate_path = _resolve_agent_python_file(
                self.server.workspace,
                self.server.workspace
                / _required_string(payload, "candidate_path"),
            )
            case_ids = payload.get("case_ids")
            if case_ids is not None and not isinstance(case_ids, list):
                raise ValueError("case_ids must be a list or null")
            return_truth = payload.get("return_finite_truth")
            if not isinstance(return_truth, bool):
                raise TypeError("return_finite_truth must be bool")
            program = _LazyPythonMacroProgram(candidate_path)
            with self.server.evaluation_lock:
                self.server.require_accepting()
                charged_before = evaluator.charged_case_credits
                clock = self.server.controller_service_clock
                service_token = None if clock is None else clock.begin_service()
                try:
                    response = evaluator.evaluate(
                        _required_string(payload, "candidate_id"),
                        program,
                        case_ids=case_ids,
                        return_finite_truth=return_truth,
                    )
                    return _development_response_payload(response)
                finally:
                    if clock is not None and service_token is not None:
                        clock.end_service(
                            service_token,
                            credit=(
                                evaluator.charged_case_credits > charged_before
                            ),
                        )

        def _screen(self, payload: object) -> Mapping[str, object]:
            submitted = self.server.submit_screen(
                payload,
                request_schema=REFERENCE_SCREEN_REQUEST_SCHEMA,
            )
            task_id = _required_string(submitted, "task_id")
            completed = self.server.wait_screen(task_id)
            if completed.get("status") == "failed":
                error = completed.get("error")
                if not isinstance(error, Mapping):
                    raise _ReferenceAsyncScreenTaskFailed({})
                raise _ReferenceAsyncScreenTaskFailed(error)
            result = completed.get("result")
            if not isinstance(result, Mapping):
                raise ReferenceAgentRuntimeError(
                    "async screen completed without a result"
                )
            return result

        def _release(self, payload: object) -> Mapping[str, object]:
            evaluator = self.server.evaluator
            if not isinstance(evaluator, DevelopmentSession):
                raise ReferenceAgentRuntimeError(
                    "completed-answer release is unavailable"
                )
            if not isinstance(payload, Mapping):
                raise ValueError("release request must be an object")
            if set(payload) != {"schema", "root_count"}:
                raise ValueError("release request fields do not match the schema")
            if payload.get("schema") != REFERENCE_RELEASE_REQUEST_SCHEMA:
                raise ValueError("unknown release request schema")
            root_count = payload.get("root_count")
            if (
                isinstance(root_count, bool)
                or not isinstance(root_count, int)
                or root_count <= 0
                or root_count > _MAX_RELEASE_ROOTS_PER_REQUEST
            ):
                raise ValueError(
                    "release root_count must be in "
                    f"[1, {_MAX_RELEASE_ROOTS_PER_REQUEST}]"
                )
            with self.server.evaluation_lock:
                self.server.require_accepting()
                charged_before = evaluator.release_charged_root_credits
                clock = self.server.controller_service_clock
                service_token = None if clock is None else clock.begin_service()
                try:
                    try:
                        released = evaluator.release(root_count)
                        manifest = _write_reference_release_manifest(
                            self.server.workspace,
                            released,
                        )
                        self.server.replace_oracle_cases(
                            _reference_controller_cases_for_evaluator(
                                evaluator,
                                released.cases,
                            )
                        )
                        if not self.server.numerical_oracle_available:
                            evaluator.discard_latest_release()
                        return manifest
                    except DevelopmentCreditExceeded:
                        raise
                    except BaseException as exc:
                        raise _PrivateScreenExecutionError(exc) from exc
                finally:
                    if clock is not None and service_token is not None:
                        clock.end_service(
                            service_token,
                            credit=(
                                evaluator.release_charged_root_credits
                                > charged_before
                            ),
                        )

        def _error(self, status: int, code: str, message: str) -> None:
            self._send(
                status,
                {
                    "schema": REFERENCE_EVALUATION_ERROR_SCHEMA,
                    "error": {"code": code, "message": message},
                },
            )

        def _send(self, status: int, payload: Mapping[str, object]) -> None:
            try:
                body = _json_bytes(payload)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def _reference_controller_cases_for_evaluator(
    evaluator: DevelopmentMacroEvaluator | DevelopmentSession,
    candidate_cases: Sequence[Case],
) -> tuple[Case, ...]:
    """Recover issued Cases without crossing them into the Agent payload."""

    if isinstance(evaluator, DevelopmentSession):
        controller_cases = evaluator._controller_release_cases()
    elif isinstance(evaluator, DevelopmentMacroEvaluator):
        controller_cases = evaluator._cohort._controller_cases()
    else:  # pragma: no cover - MacroAgentContext/RPC typing guards this path
        raise TypeError("unsupported reference development evaluator")

    candidate_ids = tuple(case.case_id for case in candidate_cases)
    projected_ids = tuple(
        candidate_facing_case(case).case_id for case in controller_cases
    )
    if candidate_ids != projected_ids:
        raise ReferenceAgentRuntimeError(
            "candidate Case snapshots do not match the controller cohort"
        )
    return controller_cases


def _reference_controller_oracle_cases(
    context: MacroAgentContext,
) -> tuple[Case, ...]:
    """Bind opaque context IDs to the matching controller-private Cases."""

    return _reference_controller_cases_for_evaluator(
        context.evaluator,
        context.cases,
    )


class ReferenceFilesystemMacroAgentProgram:
    """Stage, launch, attest, and load one candidate for a macro-loop round."""

    def __init__(
        self,
        *,
        run_root: Path | str,
        run_id: str,
        launcher: ReferenceAgentLauncher,
        bundle_attestor: ReferenceFilesystemBundleAttestor | None = None,
        screen_timeout_seconds: float = 300.0,
    ) -> None:
        boundary = require_execution_assurance(
            ExecutionAssurance.REFERENCE_IN_PROCESS
        )
        if boundary.formal_claims_allowed:
            raise RuntimeError("reference runtime unexpectedly enabled formal claims")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("reference runtime run_id cannot be empty")
        if not callable(getattr(launcher, "launch", None)):
            raise TypeError("reference runtime needs an injected Agent launcher")
        if (
            isinstance(screen_timeout_seconds, bool)
            or not isinstance(screen_timeout_seconds, Real)
            or not isfinite(float(screen_timeout_seconds))
            or float(screen_timeout_seconds) <= 0.0
        ):
            raise ValueError("screen timeout must be finite and positive")
        self.screen_timeout_seconds = float(screen_timeout_seconds)
        self.run_root = Path(run_root).resolve()
        _ensure_private_directory(self.run_root)
        self.run_id = run_id
        self.launcher = launcher
        self.bundle_attestor = bundle_attestor or ReferenceFilesystemBundleAttestor(
            self.run_root / "_controller" / "sealed_python_bundles"
        )

    @property
    def execution_assurance(self) -> ExecutionAssurance:
        return ExecutionAssurance.REFERENCE_IN_PROCESS

    @property
    def formal_claims_allowed(self) -> bool:
        return False

    @property
    def limitations(self) -> tuple[str, ...]:
        return _REFERENCE_LIMITATIONS

    def run(self, context: MacroAgentContext) -> MacroAgentOutcome:
        if not isinstance(context, MacroAgentContext):
            raise TypeError("reference runtime needs a MacroAgentContext")
        if context.round_index <= 0:
            raise ValueError("reference Agent rounds must be positive")
        if context.incumbent.bundle_uri is None:
            raise ReferenceAgentRuntimeError(
                "incumbent needs a readable candidate bundle_uri"
            )
        workspace = self._workspace_path(context)
        _ensure_private_directory(workspace.parent)
        workspace.mkdir(mode=0o700, exist_ok=False)
        incumbent_dir = workspace / "incumbent"
        incumbent_path = _stage_reference_candidate_bundle(
            context.incumbent.bundle_uri,
            incumbent_dir,
            expected_sha256=context.incumbent.bundle_sha256,
        )
        shared_candidate_payload = _stage_reference_shared_candidates(
            workspace,
            context.shared_candidates,
            incumbent_sha256=context.incumbent.bundle_sha256,
        )

        prompt_path = workspace / "PROMPT.md"
        _write_bytes_exclusive(
            prompt_path,
            context.prompt.encode("utf-8"),
            mode=0o400,
        )
        submission_path = workspace / "submission.json"
        context_path = workspace / "context.json"

        context_payload: Mapping[str, object] | None = None
        service_clock = ReferenceControllerServiceClock()
        launch_audit: ReferenceAgentLaunchAudit | None = None
        launch_failure: Mapping[str, object] | None = None
        try:
            with ReferenceDevelopmentEvaluatorRPC(
                context.evaluator,
                workspace=workspace,
                bundle_attestor=self.bundle_attestor,
                round_index=context.round_index,
                agent_id=context.agent.agent_id,
                parent_sha256=context.incumbent.bundle_sha256,
                controller_service_clock=service_clock,
                oracle_cases=_reference_controller_oracle_cases(context),
                public_session_factory=context.public_session_factory,
                public_session_methods=context.public_session_methods,
                development_solution_session_factory=(
                    context.development_solution_session_factory
                ),
                development_solution_session_methods=(
                    context.development_solution_session_methods
                ),
                guidance_oracle=context.guidance_oracle,
            ) as rpc:
                context_payload = _write_reference_context(
                    context_path,
                    run_id=self.run_id,
                    context=context,
                    incumbent_path=incumbent_path,
                    submission_path=submission_path,
                    endpoint=rpc.endpoint,
                    screen_endpoint=rpc.screen_endpoint,
                    screen_submit_endpoint=rpc.screen_submit_endpoint,
                    screen_poll_endpoint=rpc.screen_poll_endpoint,
                    release_endpoint=rpc.release_endpoint,
                    numerical_oracle_endpoint=rpc.numerical_oracle_endpoint,
                    guidance_oracle_endpoint=rpc.guidance_oracle_endpoint,
                    shared_candidates=shared_candidate_payload,
                    screen_timeout_seconds=self.screen_timeout_seconds,
                    token=rpc.token,
                )
                try:
                    returned_audit = self.launcher.launch(
                        ReferenceAgentLaunchRequest(
                            workspace=workspace,
                            context_path=context_path,
                            prompt_path=prompt_path,
                            submission_path=submission_path,
                            controller_service_clock=service_clock,
                            revoke_evaluator_capability=rpc.revoke,
                        )
                    )
                    if returned_audit is not None and not isinstance(
                        returned_audit, ReferenceAgentLaunchAudit
                    ):
                        raise TypeError("reference launcher returned an invalid audit")
                    launch_audit = returned_audit
                except BaseException as exc:
                    if isinstance(exc, ReferenceAgentLaunchError):
                        launch_audit = exc.audit
                    launch_failure = {"exception_type": type(exc).__name__}
                    raise
        finally:
            active_exception = sys.exception()
            deferred_launch_error: ReferenceAgentLaunchError | None = None
            cleanup_errors: list[tuple[str, BaseException]] = []

            if launch_audit is not None:
                try:
                    preliminary_status = launch_audit.status
                    launch_audit = _finalize_reference_agent_launch_audit(
                        launch_audit,
                        service_clock,
                    )
                    if isinstance(active_exception, ReferenceAgentLaunchError):
                        active_exception.audit = launch_audit
                    elif (
                        active_exception is None
                        and preliminary_status == "completed"
                        and launch_audit.status
                        in {"active_time_timeout", "hard_wall_timeout"}
                    ):
                        deferred_launch_error = ReferenceAgentLaunchError(
                            "reference Agent subprocess exceeded its finalized "
                            "time budget",
                            audit=launch_audit,
                        )
                        launch_failure = {
                            "exception_type": "ReferenceAgentLaunchError"
                        }
                except BaseException as exc:
                    cleanup_errors.append(("launch audit finalization", exc))

            if context_payload is not None:
                try:
                    _expire_reference_evaluator_capability(
                        context_path,
                        context_payload,
                    )
                except BaseException as exc:
                    cleanup_errors.append(("evaluator capability expiration", exc))

            if isinstance(context.evaluator, DevelopmentSession):
                try:
                    _write_development_ledger(
                        workspace / "development_ledger.json",
                        context.evaluator,
                    )
                except BaseException as exc:
                    cleanup_errors.append(("development ledger write", exc))

            if launch_audit is not None or launch_failure is not None:
                try:
                    _write_reference_agent_launch_audit(
                        workspace / "agent_launch_audit.json",
                        audit=launch_audit,
                        failure=launch_failure,
                    )
                except BaseException as exc:
                    cleanup_errors.append(("launch audit write", exc))

            preserved_exception = active_exception or deferred_launch_error
            if cleanup_errors:
                if preserved_exception is not None:
                    for label, cleanup_error in cleanup_errors:
                        preserved_exception.add_note(
                            f"{label} failed: {type(cleanup_error).__name__}"
                        )
                else:
                    _label, primary_cleanup_error = cleanup_errors[0]
                    for label, cleanup_error in cleanup_errors[1:]:
                        primary_cleanup_error.add_note(
                            f"{label} failed: {type(cleanup_error).__name__}"
                        )
                    raise primary_cleanup_error
            if deferred_launch_error is not None:
                raise deferred_launch_error

        submission, observations = _read_agent_submission(
            submission_path,
            workspace=workspace,
        )
        canonical_submission = AgentBundleSubmission(
            candidate_id=_canonical_candidate_id(
                submission.candidate_id,
                round_index=context.round_index,
                agent_id=context.agent.agent_id,
            ),
            bundle_uri=submission.bundle_uri,
        )
        attestation = self.bundle_attestor.attest(
            canonical_submission,
            round_index=context.round_index,
            agent_id=context.agent.agent_id,
            parent_sha256=context.incumbent.bundle_sha256,
        )
        candidate_ref = candidate_ref_from_attestation(attestation)
        return MacroAgentOutcome(
            candidate=_reference_candidate_submission(candidate_ref),
            observations=observations,
        )

    def _workspace_path(self, context: MacroAgentContext) -> Path:
        identity = hashlib.sha256(
            f"{self.run_id}\0{context.round_index}\0{context.agent.agent_id}".encode(
                "utf-8"
            )
        ).hexdigest()[:12]
        return (
            self.run_root
            / f"round-{context.round_index:04d}"
            / f"agent-{identity}"
        )


def _stage_reference_shared_candidates(
    workspace: Path,
    candidates: Sequence[CandidateRef],
    *,
    incumbent_sha256: str,
) -> tuple[Mapping[str, object], ...]:
    """Copy prior common-cohort candidates without exposing private evidence."""

    _require_sha256(incumbent_sha256, "incumbent")
    rows: list[Mapping[str, object]] = []
    seen = {incumbent_sha256}
    prior_dir: Path | None = None
    for candidate in candidates:
        if not isinstance(candidate, CandidateRef):
            raise TypeError("shared candidate must be a CandidateRef")
        if candidate.bundle_sha256 in seen:
            continue
        if candidate.bundle_uri is None:
            raise ReferenceAgentRuntimeError(
                "shared candidate needs a readable candidate bundle_uri"
            )
        source_path = _local_bundle_path(candidate.bundle_uri)
        payload, bundle = _read_candidate_bundle_for_runtime(
            source_path,
        )
        if prior_dir is None:
            prior_dir = workspace / "prior_candidates"
            prior_dir.mkdir(mode=0o700)
        if bundle is None:
            staged_path = prior_dir / (
                f"candidate-{len(rows) + 1:02d}-{candidate.bundle_sha256[:12]}.py"
            )
            _write_bytes_exclusive(staged_path, payload, mode=0o400)
        else:
            staged_root = prior_dir / (
                f"candidate-{len(rows) + 1:02d}-{candidate.bundle_sha256[:12]}"
            )
            staged_path = _stage_reference_candidate_bundle(
                source_path,
                staged_root,
                expected_sha256=candidate.bundle_sha256,
            )
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "bundle_sha256": candidate.bundle_sha256,
                "agent_id": candidate.agent_id,
                "round_index": candidate.round_index,
                "path": staged_path.relative_to(workspace).as_posix(),
            }
        )
        seen.add(candidate.bundle_sha256)
    return tuple(rows)


def _public_search_leaderboard(
    entries: Sequence[SearchLeaderboardEntry],
) -> list[Mapping[str, object]]:
    """Serialize only common-cohort aggregate standings."""

    rows: list[Mapping[str, object]] = []
    for entry in entries:
        if not isinstance(entry, SearchLeaderboardEntry):
            raise TypeError("prior leaderboard entry has an invalid type")
        rows.append(
            {
                "rank": entry.rank,
                "candidate_id": entry.candidate.candidate_id,
                "bundle_sha256": entry.candidate.bundle_sha256,
                "agent_id": entry.agent_id,
                "round_index": entry.candidate.round_index,
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
        )
    return rows


def _reference_public_tasks(
    cases: Sequence[Case],
    queries: Sequence[AnswerQuery],
    answers: Sequence[AnswerView | None],
    codec: _PublicValueCodec,
) -> list[Mapping[str, object]]:
    if not len(cases) == len(queries) == len(answers):
        raise ReferenceAgentRuntimeError(
            "public Case/query/completed-answer counts do not match"
        )
    rows = []
    for case, query, answer in zip(cases, queries, answers):
        redact_points = (
            answer is None and query.kind is QueryKind.POINTS
        )
        query_payload = {
            "kind": query.kind.value,
            "fields": list(query.fields),
            "resolution": dict(query.resolution),
            "points": (
                []
                if redact_points
                else [dict(point) for point in query.points]
            ),
        }
        if redact_points:
            query_payload.update(
                {
                    "points_redacted": True,
                    "point_count": len(query.points),
                }
            )
        rows.append(
            {
                "case": _reference_public_case_payload(case, codec),
                "query": query_payload,
                "completed_answer": (
                    None
                    if answer is None
                    else _completed_answer_payload(answer, codec)
                ),
            }
        )
    return rows



def _reference_cli(context_name: str, command: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pde_agent_verifier_solver.engine.reference_agent_runtime",
        command,
        "--context",
        context_name,
    ]



def _stage_reference_public_problem(
    workspace: Path,
    context: MacroAgentContext,
    *,
    numerical_oracle_endpoint: str,
) -> tuple[Mapping[str, object] | None, Mapping[str, object] | None]:
    bundle = context.public_problem_bundle
    equation_methods = _validated_reference_oracle_methods(
        context.public_session_methods,
        label="equation",
        factory=context.public_session_factory,
    )
    development_methods = _validated_reference_oracle_methods(
        context.development_solution_session_methods,
        label="development solution",
        factory=context.development_solution_session_factory,
    )
    if set(equation_methods) & set(development_methods):
        raise ReferenceAgentRuntimeError(
            "equation and completed-solution Oracle methods overlap"
        )
    if bundle is None:
        if equation_methods or development_methods:
            raise ReferenceAgentRuntimeError(
                "numerical Oracle methods require a PublicProblemBundle"
            )
        return None, None
    if not isinstance(bundle, PublicProblemBundle):
        raise TypeError("public_problem_bundle must be a PublicProblemBundle")
    declared_equation = tuple(
        method.name for method in bundle.oracle_methods
    )
    declared_development = tuple(
        method.name for method in bundle.development_solution_oracle_methods
    )
    if set(equation_methods) != set(declared_equation) or set(
        development_methods
    ) != set(declared_development):
        raise ReferenceAgentRuntimeError(
            "runtime numerical Oracle allowlists differ from the public bundle"
        )

    resources = bundle.materialize(workspace / "public_resources")
    for member in sorted(
        resources.rglob("*"),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        member.chmod(0o500 if member.is_dir() else 0o400)
    resources.chmod(0o500)
    public_problem = {
        "bundle_id": bundle.bundle_id,
        "bundle_sha256": bundle.bundle_sha256,
        "equation_id": bundle.equation_id,
        "resources_path": resources.relative_to(workspace).as_posix(),
        "manifest_path": (
            (resources / "manifest.json").relative_to(workspace).as_posix()
        ),
    }
    if not equation_methods and not development_methods:
        return public_problem, None
    numerical_oracle = {
        "endpoint": numerical_oracle_endpoint,
        "capability_status": _EVALUATOR_CAPABILITY_ACTIVE,
        "equation_methods": list(equation_methods),
        "development_solution_methods": list(development_methods),
        "coordinate_policy": "agent_selected",
        "session_lifecycle": "fresh_per_call",
        "completed_solution_visibility": "development_only",
        "fixed_lifetime_or_sample_quota": False,
        "limited_only_by": [
            "method_request_constraints",
            "hard_wall_time",
            "concurrency",
            "backend_compute_capacity",
        ],
        "client_helper": (
            "pde_agent_verifier_solver.engine.reference_agent_runtime."
            "query_reference_public_oracle"
        ),
    }
    return public_problem, numerical_oracle


def _stage_reference_guidance_oracle(
    context: MacroAgentContext,
    *,
    guidance_oracle_endpoint: str,
) -> Mapping[str, object] | None:
    oracle = context.guidance_oracle
    if oracle is None:
        return None
    if not isinstance(oracle, DevelopmentGuidanceOracle):
        raise TypeError(
            "MacroAgentContext guidance_oracle must be a "
            "DevelopmentGuidanceOracle"
        )
    methods = tuple(oracle.available_solution_methods)
    if not methods or any(
        not isinstance(method, str)
        or not method.isidentifier()
        or method.startswith("_")
        for method in methods
    ) or len(methods) != len(set(methods)):
        raise ReferenceAgentRuntimeError(
            "guidance completed-solution allowlist is invalid"
        )
    return {
        "endpoint": guidance_oracle_endpoint,
        "capability_status": _EVALUATOR_CAPABILITY_ACTIVE,
        "snapshot_sha256": oracle.snapshot.snapshot_sha256,
        "actions": ["sample", "query_completed_solution"],
        "development_solution_methods": list(methods),
        "unlimited_fresh_inference": True,
        "fixed_lifetime_or_sample_quota": False,
        "fresh_randomness": "controller_seed_and_nonce_per_sample_call",
        "training_available": False,
        "hidden_resolver_available": False,
        "answer_locator_exposed": False,
        "sample_response": "public_case_and_opaque_sample_id_only",
        "completed_solution_scope": "issued_development_samples_only",
        "limited_only_by": [
            "per_request_backend_constraints",
            "inference_wall_time",
            "hard_wall_time",
            "concurrency",
            "backend_compute_capacity",
        ],
        "sample_client_helper": (
            "pde_agent_verifier_solver.engine.reference_agent_runtime."
            "sample_reference_guidance"
        ),
        "solution_client_helper": (
            "pde_agent_verifier_solver.engine.reference_agent_runtime."
            "query_reference_guided_completed_solution"
        ),
    }

def _write_reference_context(
    path: Path,
    *,
    run_id: str,
    context: MacroAgentContext,
    incumbent_path: Path,
    submission_path: Path,
    endpoint: str,
    screen_endpoint: str,
    screen_submit_endpoint: str,
    screen_poll_endpoint: str,
    release_endpoint: str,
    numerical_oracle_endpoint: str,
    guidance_oracle_endpoint: str,
    shared_candidates: Sequence[Mapping[str, object]],
    screen_timeout_seconds: float,
    token: str,
) -> Mapping[str, object]:
    workspace = path.parent
    cases = context.evaluator.public_cases
    queries = context.evaluator.public_queries
    strong_session = isinstance(context.evaluator, DevelopmentSession)
    completed_answers: tuple[AnswerView | None, ...] = (
        context.evaluator.public_answers
        if strong_session
        else (None,) * len(cases)
    )
    if tuple(case.case_id for case in cases) != tuple(
        case.case_id for case in context.cases
    ):
        raise ReferenceAgentRuntimeError(
            "MacroAgentContext cases do not match the evaluator cohort"
        )
    if not (
        len(cases) == len(queries) == len(completed_answers)
    ):
        raise ReferenceAgentRuntimeError(
            "public Case/query/completed-answer counts do not match"
        )
    if strong_session and any(answer is None for answer in completed_answers):
        raise ReferenceAgentRuntimeError(
            "fresh development releases need every completed answer"
        )
    public_problem, numerical_oracle = _stage_reference_public_problem(
        workspace,
        context,
        numerical_oracle_endpoint=numerical_oracle_endpoint,
    )
    if numerical_oracle is not None:
        numerical_oracle = dict(numerical_oracle)
        numerical_oracle["cli"] = _reference_cli(path.name, "oracle")
    guidance_oracle = _stage_reference_guidance_oracle(
        context,
        guidance_oracle_endpoint=guidance_oracle_endpoint,
    )
    if guidance_oracle is not None:
        guidance_oracle = dict(guidance_oracle)
        guidance_oracle["sample_cli"] = _reference_cli(
            path.name, "guidance-sample"
        )
        guidance_oracle["solution_cli"] = _reference_cli(
            path.name, "guidance-solution"
        )
    codec = _PublicValueCodec(workspace / "arrays")
    tasks = _reference_public_tasks(
        cases,
        queries,
        completed_answers,
        codec,
    )
    boundary = require_execution_assurance(
        ExecutionAssurance.REFERENCE_IN_PROCESS
    )
    evaluator_payload = {
        "endpoint": endpoint,
        "token": token,
        "capability_status": _EVALUATOR_CAPABILITY_ACTIVE,
        "remaining_case_credits": context.evaluator.remaining_case_credits,
        "cli": _reference_cli(path.name, "evaluate"),
    }
    if strong_session:
        evaluator_payload.pop("remaining_case_credits")
        open_search = context.evaluator.open_search
        release_capacity = context.evaluator.release_root_credit_limit
        evaluator_payload.update(
            {
                "screen_endpoint": screen_endpoint,
                "screen_submit_endpoint": screen_submit_endpoint,
                "screen_poll_endpoint": screen_poll_endpoint,
                "release_endpoint": release_endpoint,
                "screen_timeout_seconds": screen_timeout_seconds,
                "statistical_unit": "independent_root",
                "max_release_roots_per_request": (
                    min(_MAX_RELEASE_ROOTS_PER_REQUEST, release_capacity)
                    if open_search
                    else _MAX_RELEASE_ROOTS_PER_REQUEST
                ),
                "max_async_screen_inflight": _MAX_ASYNC_SCREEN_INFLIGHT,
                "capabilities": {
                    "on_demand_completed_answer_release": True,
                    "async_screen_submit_poll": True,
                    "submit_attests_sealed_candidate_bytes": True,
                    "screen_worker_count": 1,
                    "development_screen_feedback": True,
                    "development_screens_are_official_nominations": False,
                    "fixed_lifetime_root_quota": not open_search,
                },
                "cli": _reference_cli(path.name, "screen"),
                "release_cli": _reference_cli(path.name, "release"),
                "screen_submit_cli": _reference_cli(
                    path.name, "screen-submit"
                ),
                "screen_poll_cli": _reference_cli(path.name, "screen-poll"),
            }
        )
        if open_search:
            evaluator_payload.update(
                {
                    "search_mode": "open",
                    "credit_scope": "per_request_capacity",
                    "fixed_lifetime_root_quota": False,
                    "screen_root_request_capacity": (
                        context.evaluator.screen_root_credit_limit
                    ),
                    "release_root_request_capacity": release_capacity,
                    "recommended_screen_root_counts": list(
                        context.evaluator.screen_root_stages
                    ),
                    "observational_usage": {
                        "screen_requests": len(context.evaluator.screen_ledger),
                        "screen_evaluated_roots": (
                            context.evaluator.screen_charged_root_credits
                        ),
                        "release_requests": len(context.evaluator.release_ledger),
                        "release_sampled_roots": (
                            context.evaluator.release_charged_root_credits
                        ),
                    },
                }
            )
        else:
            evaluator_payload.update(
                {
                    "credit_scope": "per_agent_per_round",
                    "remaining_root_credits": (
                        context.evaluator.screen_remaining_root_credits
                    ),
                    "remaining_release_root_credits": (
                        context.evaluator.release_remaining_root_credits
                    ),
                }
            )
    payload = {
        "schema": REFERENCE_CONTEXT_SCHEMA,
        "assurance": boundary.assurance.value,
        "formal_claims_allowed": False,
        "limitations": list(boundary.limitations + _REFERENCE_LIMITATIONS),
        "run_id": run_id,
        "round_index": context.round_index,
        "agent_id": context.agent.agent_id,
        "agent_name": context.agent.name,
        "prompt": context.prompt,
        "cohort": {
            "cohort_id": context.evaluator.cohort_id,
            "cohort_sha256": context.evaluator.cohort_sha256,
            "contract_id": context.evaluator.contract_id,
            "ordered_case_ids": [case.case_id for case in cases],
        },
        "public_tasks": tasks,
        "public_problem": public_problem,
        "numerical_oracle": numerical_oracle,
        "guidance_oracle": guidance_oracle,
        "incumbent": {
            "candidate_id": context.incumbent.candidate_id,
            "bundle_sha256": context.incumbent.bundle_sha256,
            "path": incumbent_path.relative_to(workspace).as_posix(),
        },
        "prior_handoff": _public_handoff_payload(context.prior_handoff),
        "search_contract": {
            "leaderboard_limit": context.search_policy.leaderboard_limit,
            "cross_share_after_round": (
                context.search_policy.cross_share_after_round
            ),
            "counts_are_aggregate_only": True,
            "official_nomination_limit": 1,
            "development_screens_are_official_nominations": False,
        },
        "prior_leaderboard": _public_search_leaderboard(
            context.prior_leaderboard
        ),
        "prior_candidates": [dict(row) for row in shared_candidates],
        "evaluator": evaluator_payload,
        "submission_path": submission_path.relative_to(workspace).as_posix(),
    }
    _write_json_exclusive(
        path,
        payload,
        mode=0o400,
        max_bytes=_MAX_CONTEXT_JSON_BYTES,
    )
    if (
        isinstance(context.evaluator, DevelopmentSession)
        and numerical_oracle is None
    ):
        context.evaluator.discard_latest_release()
    return payload


def _expire_reference_evaluator_capability(
    path: Path,
    active_payload: Mapping[str, object],
) -> None:
    """Publish a non-sensitive, auditable context after the RPC is closed."""

    payload = dict(active_payload)
    evaluator = payload.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise ReferenceAgentRuntimeError(
            "reference workspace evaluator metadata is missing"
        )
    terminal_evaluator = dict(evaluator)
    terminal_evaluator.pop("token", None)
    terminal_evaluator["capability_status"] = _EVALUATOR_CAPABILITY_EXPIRED
    payload["evaluator"] = terminal_evaluator
    numerical_oracle = payload.get("numerical_oracle")
    if numerical_oracle is not None:
        if not isinstance(numerical_oracle, Mapping):
            raise ReferenceAgentRuntimeError(
                "reference numerical Oracle metadata is invalid"
            )
        terminal_oracle = dict(numerical_oracle)
        terminal_oracle["capability_status"] = _EVALUATOR_CAPABILITY_EXPIRED
        payload["numerical_oracle"] = terminal_oracle
    guidance_oracle = payload.get("guidance_oracle")
    if guidance_oracle is not None:
        if not isinstance(guidance_oracle, Mapping):
            raise ReferenceAgentRuntimeError(
                "reference guidance Oracle metadata is invalid"
            )
        terminal_guidance = dict(guidance_oracle)
        terminal_guidance["capability_status"] = (
            _EVALUATOR_CAPABILITY_EXPIRED
        )
        payload["guidance_oracle"] = terminal_guidance
    _replace_json_atomically(
        path,
        payload,
        mode=0o400,
        max_bytes=_MAX_CONTEXT_JSON_BYTES,
    )


def _write_reference_agent_launch_audit(
    path: Path,
    *,
    audit: ReferenceAgentLaunchAudit | None,
    failure: Mapping[str, object] | None,
) -> None:
    """Persist bounded timing and a sanitized launch failure classification."""

    payload = (
        dict(audit.payload())
        if audit is not None
        else {
            "schema": "reference-agent-launch-audit-v1",
            "status": "launcher_failed_before_timing",
            "controller_service_time_excluded_from_active_budget": True,
            "in_process_evaluator_preemption_guaranteed": False,
        }
    )
    payload["failure"] = (
        None
        if failure is None
        else {"exception_type": str(failure.get("exception_type", "InternalError"))}
    )
    _replace_json_atomically(
        path,
        payload,
        mode=0o400,
        max_bytes=_MAX_SUBMISSION_BYTES,
    )


def _write_development_ledger(
    path: Path,
    session: DevelopmentSession,
) -> None:
    """Persist aggregate interaction decisions without cases or truth."""

    payload = {
        "schema": REFERENCE_DEVELOPMENT_LEDGER_SCHEMA,
        "statistical_unit": "independent_root",
        "release_ledger": [
            _development_ledger_entry_payload(entry)
            for entry in session.release_ledger
        ],
        "screen_ledger": [
            _development_ledger_entry_payload(entry)
            for entry in session.screen_ledger
        ],
    }
    if session.open_search:
        payload.update(
            {
                "resource_scope": "per_request_capacity",
                "fixed_lifetime_root_quota": False,
                "request_capacity": {
                    "release_roots": session.release_root_credit_limit,
                    "screen_roots": session.screen_root_credit_limit,
                },
                "observational_usage": {
                    "release_sampled_roots": (
                        session.release_charged_root_credits
                    ),
                    "screen_evaluated_roots": (
                        session.screen_charged_root_credits
                    ),
                },
            }
        )
    else:
        payload.update(
            {
                "credit_scope": "per_agent_per_round",
                "root_credits": {
                    "release_charged": session.release_charged_root_credits,
                    "release_remaining": session.release_remaining_root_credits,
                    "screen_charged": session.screen_charged_root_credits,
                    "screen_remaining": session.screen_remaining_root_credits,
                },
            }
        )
    _write_json_exclusive(
        path,
        payload,
        mode=0o400,
        max_bytes=_MAX_CONTEXT_JSON_BYTES,
    )


def _development_ledger_entry_payload(entry) -> Mapping[str, object]:
    draw = entry.draw
    return {
        "request_index": entry.request_index,
        "charged_root_credits": entry.charged_root_credits,
        "candidate_sha256": entry.candidate_sha256,
        "requested_root_count": entry.requested_root_count,
        "previous_root_count": entry.previous_root_count,
        "new_root_count": entry.new_root_count,
        "cached": entry.cached,
        "draw": (
            None
            if draw is None
            else {
                "draw_index": draw.draw_index,
                "kind": draw.kind.value,
                "root_count": draw.root_count,
                "member_case_count": draw.member_case_count,
                "cohort_id": draw.cohort_id,
                "cohort_sha256": draw.cohort_sha256,
            }
        ),
    }


def _public_handoff_payload(handoff) -> Mapping[str, object] | None:
    if handoff is None:
        return None
    return {
        "round_index": handoff.round_index,
        "selected_sha256": handoff.selected.bundle_sha256,
        "promoted": handoff.promoted,
        "challenger_sha256": (
            handoff.challenger.bundle_sha256
            if handoff.challenger is not None
            else None
        ),
        "decision_reason": handoff.decision_reason,
        "feedback": {
            "previous_incumbent_sha256": (
                handoff.feedback.previous_incumbent_sha256
            ),
            "selected_sha256": handoff.feedback.selected_sha256,
            "promoted": handoff.feedback.promoted,
            "decision_reason": handoff.feedback.decision_reason,
            "evidence_case_count": handoff.feedback.evidence_case_count,
            "confidence_group_size": (
                handoff.feedback.confidence_group_size
            ),
            "paired_mean_improvement": (
                handoff.feedback.paired_mean_improvement
            ),
            "paired_confidence_lower": (
                handoff.feedback.paired_confidence_lower
            ),
            "paired_win_rate": handoff.feedback.paired_win_rate,
        },
        "previous_handoff_sha256": handoff.previous_handoff_sha256,
        "handoff_sha256": handoff.handoff_sha256,
    }


def _canonical_candidate_id(
    raw_candidate_id: object,
    *,
    round_index: object,
    agent_id: object,
) -> str:
    if (
        not isinstance(raw_candidate_id, str)
        or not raw_candidate_id.strip()
        or len(raw_candidate_id) > _MAX_CANDIDATE_ID_CHARS
    ):
        raise ReferenceAgentRuntimeError("candidate_id is empty or too long")
    if (
        isinstance(round_index, bool)
        or not isinstance(round_index, int)
        or round_index < 0
    ):
        raise ReferenceAgentRuntimeError("candidate round is invalid")
    if not isinstance(agent_id, str) or not agent_id.strip():
        raise ReferenceAgentRuntimeError("candidate agent_id is invalid")
    return f"{agent_id}:r{round_index}:{raw_candidate_id}"


def _validated_submission_fields(
    candidate_id: object,
    observations: object,
) -> tuple[str, ...]:
    if (
        not isinstance(candidate_id, str)
        or not candidate_id.strip()
        or len(candidate_id) > _MAX_CANDIDATE_ID_CHARS
    ):
        raise ReferenceAgentRuntimeError("candidate_id is empty or too long")
    if not isinstance(observations, (tuple, list)):
        raise ReferenceAgentRuntimeError("observations must be a sequence of strings")
    if len(observations) > _MAX_OBSERVATIONS:
        raise ReferenceAgentRuntimeError("too many submission observations")
    rows = tuple(observations)
    if any(
        not isinstance(value, str) or len(value) > _MAX_OBSERVATION_CHARS
        for value in rows
    ):
        raise ReferenceAgentRuntimeError(
            "submission observation is invalid or too long"
        )
    return rows


def _read_agent_submission(
    path: Path,
    *,
    workspace: Path,
) -> tuple[AgentBundleSubmission, tuple[str, ...]]:
    payload = _read_json_object(path, max_bytes=_MAX_SUBMISSION_BYTES)
    expected = {"schema", "candidate_id", "bundle_path", "observations"}
    if set(payload) != expected:
        raise ReferenceAgentRuntimeError(
            "submission fields do not match the one-candidate schema"
        )
    if payload.get("schema") != REFERENCE_SUBMISSION_SCHEMA:
        raise ReferenceAgentRuntimeError("unknown reference submission schema")
    candidate_id = _required_string(payload, "candidate_id")
    observations = _validated_submission_fields(
        candidate_id,
        payload.get("observations"),
    )
    bundle_path = _required_string(payload, "bundle_path")
    if len(bundle_path) > _MAX_PATH_CHARS:
        raise ReferenceAgentRuntimeError("submission bundle_path is too long")
    bundle = _resolve_agent_python_file(
        workspace,
        workspace / bundle_path,
    )
    return (
        AgentBundleSubmission(
            candidate_id=candidate_id,
            bundle_uri=str(bundle),
        ),
        observations,
    )


def _development_response_payload(
    response: DevelopmentMacroResponse,
) -> Mapping[str, object]:
    return {
        "schema": REFERENCE_EVALUATION_RESPONSE_SCHEMA,
        "candidate_id": response.candidate_id,
        "cohort_id": response.cohort_id,
        "cohort_sha256": response.cohort_sha256,
        "contract_id": response.contract_id,
        "charged_case_credits": response.charged_case_credits,
        "remaining_case_credits": response.remaining_case_credits,
        "cases": [
            {
                "score": {
                    "case_id": row.score.case_id,
                    "contract_id": row.score.contract_id,
                    "status": row.score.status,
                    "primary_error": row.score.primary_error,
                    "metrics": dict(row.score.metrics),
                    "failures": list(row.score.failures),
                },
                "finite_truth": (
                    _answer_view_payload(row.finite_truth)
                    if row.finite_truth is not None
                    else None
                ),
            }
            for row in response.cases
        ],
    }


def _answer_view_payload(view: AnswerView) -> Mapping[str, object]:
    return {
        "kind": view.kind.value,
        "fields": {
            name: [float(value) for value in values]
            for name, values in view.fields.items()
        },
        "axes": {
            name: [float(value) for value in values]
            for name, values in view.axes.items()
        },
        "points": [dict(point) for point in view.points],
        "shape": list(view.shape),
    }


def _write_reference_release_manifest(
    workspace: Path,
    release: DevelopmentRelease,
) -> Mapping[str, object]:
    """Publish one release in a self-contained immutable workspace directory."""

    if not isinstance(release, DevelopmentRelease):
        raise TypeError("release manifest needs a DevelopmentRelease")
    if not (
        len(release.cases) == len(release.queries) == len(release.answers)
        == release.member_case_count
    ):
        raise ReferenceAgentRuntimeError(
            "release Case/query/completed-answer counts do not match"
        )
    releases_root = workspace / "releases"
    _ensure_private_directory(releases_root)
    release_root = releases_root / (
        f"release-{release.release_index:04d}-"
        f"{release.cohort_sha256[:12]}"
    )
    release_root.mkdir(mode=0o700, exist_ok=False)
    codec = _PublicValueCodec(release_root / "arrays")
    tasks = _reference_public_tasks(
        release.cases,
        release.queries,
        release.answers,
        codec,
    )
    manifest = {
        "schema": REFERENCE_RELEASE_MANIFEST_SCHEMA,
        "release_index": release.release_index,
        "root_count": release.root_count,
        "member_case_count": release.member_case_count,
        "cohort": {
            "cohort_id": release.cohort_id,
            "cohort_sha256": release.cohort_sha256,
            "contract_id": release.contract_id,
            "ordered_case_ids": [case.case_id for case in release.cases],
        },
        "public_tasks": tasks,
        "accounting": {
            "charged_root_credits": release.charged_root_credits,
            "cumulative_charged_root_credits": (
                release.cumulative_charged_root_credits
            ),
            "remaining_root_credits": release.remaining_root_credits,
        },
        "private_seeds_included": False,
    }
    manifest_path = release_root / "manifest.json"
    _write_json_exclusive(
        manifest_path,
        manifest,
        mode=0o400,
        max_bytes=_MAX_CONTEXT_JSON_BYTES,
    )
    return {
        "schema": REFERENCE_RELEASE_RESPONSE_SCHEMA,
        "release_index": release.release_index,
        "root_count": release.root_count,
        "member_case_count": release.member_case_count,
        "manifest_path": manifest_path.relative_to(workspace).as_posix(),
        "cohort_id": release.cohort_id,
        "cohort_sha256": release.cohort_sha256,
        "contract_id": release.contract_id,
        "accounting": {
            "charged_root_credits": release.charged_root_credits,
            "cumulative_charged_root_credits": (
                release.cumulative_charged_root_credits
            ),
            "remaining_root_credits": release.remaining_root_credits,
        },
        "private_seeds_included": False,
    }


def _completed_answer_payload(
    view: AnswerView,
    codec: _PublicValueCodec,
) -> Mapping[str, object]:
    np = codec.np
    if np is None:
        raise ReferenceAgentRuntimeError(
            "numpy is required to publish completed answers"
        )

    def encode_values(values: object) -> Mapping[str, object]:
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError("completed answer values must be one-dimensional")
        return codec.encode(array)

    return {
        "kind": view.kind.value,
        "fields": {
            name: encode_values(values)
            for name, values in view.fields.items()
        },
        "axes": {
            name: encode_values(values)
            for name, values in view.axes.items()
        },
        "points": [dict(point) for point in view.points],
        "shape": list(view.shape),
    }


def _answer_view_from_payload(
    payload: object,
    *,
    codec: _PublicValueCodec,
    mmap_arrays: bool,
) -> AnswerView:
    if not isinstance(payload, Mapping):
        raise ReferenceAgentRuntimeError(
            "completed_answer must be an object or null"
        )
    expected = {"kind", "fields", "axes", "points", "shape"}
    if set(payload) != expected:
        raise ReferenceAgentRuntimeError(
            "completed_answer fields do not match the schema"
        )
    fields = payload.get("fields")
    axes = payload.get("axes")
    points = payload.get("points")
    shape = payload.get("shape")
    if (
        not isinstance(fields, Mapping)
        or not isinstance(axes, Mapping)
        or not isinstance(points, list)
        or not isinstance(shape, list)
        or any(not isinstance(point, Mapping) for point in points)
    ):
        raise ReferenceAgentRuntimeError("completed_answer values are invalid")

    def decode_values(node: object) -> tuple[float, ...]:
        values = codec.decode(node, mmap_arrays=mmap_arrays)
        try:
            return tuple(float(value) for value in values)
        except (TypeError, ValueError) as exc:
            raise ReferenceAgentRuntimeError(
                "completed_answer array is invalid"
            ) from exc

    try:
        return AnswerView(
            kind=QueryKind(_required_string(payload, "kind")),
            fields={
                str(name): decode_values(node)
                for name, node in fields.items()
            },
            axes={
                str(name): decode_values(node)
                for name, node in axes.items()
            },
            points=tuple(dict(point) for point in points),
            shape=tuple(int(value) for value in shape),
        )
    except (TypeError, ValueError) as exc:
        raise ReferenceAgentRuntimeError(
            "completed_answer values are invalid"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pde_agent_verifier_solver.engine.reference_agent_runtime"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser(
        "evaluate",
        help="Evaluate a workspace candidate through the budgeted local RPC",
    )
    evaluate.add_argument("--context", required=True)
    evaluate.add_argument("--candidate", required=True)
    evaluate.add_argument("--candidate-id", required=True)
    evaluate.add_argument("--case-id", action="append", default=[])
    evaluate.add_argument("--return-finite-truth", action="store_true")
    evaluate.add_argument("--timeout", type=float, default=300.0)

    def add_screen_arguments(command) -> None:
        command.add_argument("--context", required=True)
        command.add_argument("--candidate", required=True)
        command.add_argument("--candidate-id", required=True)
        command.add_argument(
            "--requested-root-count",
            required=True,
            type=int,
        )

    screen = commands.add_parser(
        "screen",
        help="Synchronously screen a sealed trial on fresh private roots",
    )
    add_screen_arguments(screen)
    screen.add_argument("--timeout", type=float, default=None)
    screen_submit = commands.add_parser(
        "screen-submit",
        help="Seal and enqueue a bounded asynchronous screen",
    )
    add_screen_arguments(screen_submit)
    screen_submit.add_argument("--timeout", type=float, default=30.0)
    screen_poll = commands.add_parser(
        "screen-poll",
        help="Poll a previously accepted asynchronous screen",
    )
    screen_poll.add_argument("--context", required=True)
    screen_poll.add_argument("--task-id", required=True)
    screen_poll.add_argument("--timeout", type=float, default=30.0)
    release = commands.add_parser(
        "release",
        help="Release fresh completed answers under the independent root budget",
    )
    release.add_argument("--context", required=True)
    release.add_argument("--root-count", required=True, type=int)
    release.add_argument("--timeout", type=float, default=300.0)
    oracle = commands.add_parser(
        "oracle",
        help="Query an allowlisted public numerical Oracle method",
    )
    oracle.add_argument("--context", required=True)
    oracle.add_argument(
        "--namespace",
        required=True,
        choices=("equation", "development_solution"),
    )
    oracle.add_argument("--case-id", required=True)
    oracle.add_argument("--method", required=True)
    oracle.add_argument("--args-json", type=json.loads, default=[])
    oracle.add_argument("--kwargs-json", type=json.loads, default={})
    oracle.add_argument("--timeout", type=float, default=300.0)
    guidance_sample = commands.add_parser(
        "guidance-sample",
        help="Draw fresh inference-only guided DEVELOPMENT cases",
    )
    guidance_sample.add_argument("--context", required=True)
    guidance_sample.add_argument("--sample-count", required=True, type=int)
    guidance_sample.add_argument("--settings-json", type=json.loads, default={})
    guidance_sample.add_argument("--timeout", type=float, default=300.0)
    guidance_solution = commands.add_parser(
        "guidance-solution",
        help="Query one issued guided completed solution",
    )
    guidance_solution.add_argument("--context", required=True)
    guidance_solution.add_argument("--sample-id", required=True)
    guidance_solution.add_argument("--method", required=True)
    guidance_solution.add_argument("--args-json", type=json.loads, default=[])
    guidance_solution.add_argument("--kwargs-json", type=json.loads, default={})
    guidance_solution.add_argument("--timeout", type=float, default=300.0)

    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            result = evaluate_reference_candidate(
                args.context,
                candidate_path=args.candidate,
                candidate_id=args.candidate_id,
                case_ids=args.case_id or None,
                return_finite_truth=args.return_finite_truth,
                timeout_seconds=args.timeout,
            )
        elif args.command == "screen":
            result = screen_reference_candidate(
                args.context,
                candidate_path=args.candidate,
                candidate_id=args.candidate_id,
                requested_root_count=args.requested_root_count,
                timeout_seconds=args.timeout,
            )
        elif args.command == "screen-submit":
            result = submit_reference_candidate_screen(
                args.context,
                candidate_path=args.candidate,
                candidate_id=args.candidate_id,
                requested_root_count=args.requested_root_count,
                timeout_seconds=args.timeout,
            )
        elif args.command == "screen-poll":
            result = poll_reference_candidate_screen(
                args.context,
                task_id=args.task_id,
                timeout_seconds=args.timeout,
            )
        elif args.command == "release":
            result = release_reference_completed_answers(
                args.context,
                root_count=args.root_count,
                timeout_seconds=args.timeout,
            )
        elif args.command == "guidance-sample":
            batch = sample_reference_guidance(
                args.context,
                spec=GuidedSampleSpec(
                    args.sample_count,
                    args.settings_json,
                ),
                timeout_seconds=args.timeout,
            )
            context_path = Path(args.context).resolve()
            codec = _PublicValueCodec(
                context_path.parent / "arrays",
                create=False,
            )
            result = {
                "schema": REFERENCE_GUIDANCE_ORACLE_RESPONSE_SCHEMA,
                "action": "sample",
                "batch": _reference_guided_batch_payload(batch, codec),
            }
        elif args.command == "guidance-solution":
            value = query_reference_guided_completed_solution(
                args.context,
                sample_id=args.sample_id,
                method=args.method,
                args=args.args_json,
                kwargs=args.kwargs_json,
                timeout_seconds=args.timeout,
            )
            context_path = Path(args.context).resolve()
            result = {
                "schema": REFERENCE_GUIDANCE_ORACLE_RESPONSE_SCHEMA,
                "action": "query_completed_solution",
                "sample_id": args.sample_id,
                "method": args.method,
                "result": _PublicValueCodec(
                    context_path.parent / "arrays",
                    create=False,
                ).encode(value),
            }
        else:
            value = query_reference_public_oracle(
                args.context,
                namespace=args.namespace,
                case_id=args.case_id,
                method=args.method,
                args=args.args_json,
                kwargs=args.kwargs_json,
                timeout_seconds=args.timeout,
            )
            context_path = Path(args.context).resolve()
            result = {
                "schema": REFERENCE_NUMERICAL_ORACLE_RESPONSE_SCHEMA,
                "namespace": args.namespace,
                "case_id": args.case_id,
                "method": args.method,
                "result": _PublicValueCodec(
                    context_path.parent / "arrays",
                    create=False,
                ).encode(value),
            }
    except ReferenceEvaluatorClientError as exc:
        print(
            json.dumps(
                {
                    "schema": REFERENCE_EVALUATION_ERROR_SCHEMA,
                    "error": {"code": exc.code, "message": exc.message},
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


def _candidate_bundle_limits(max_bytes: int) -> CandidateBundleLimits:
    return CandidateBundleLimits(
        max_total_bytes=max_bytes,
        max_file_bytes=max_bytes,
    )


def _is_multifile_candidate_locator(path: Path) -> bool:
    if path.is_dir() or path.name == CANDIDATE_BUNDLE_MANIFEST:
        return True
    manifest = path.parent / CANDIDATE_BUNDLE_MANIFEST
    return manifest.exists() or manifest.is_symlink()


def _inspect_runtime_candidate_bundle(
    path: Path,
    *,
    max_bytes: int,
) -> CandidateBundle:
    try:
        return inspect_candidate_bundle(
            path,
            limits=_candidate_bundle_limits(max_bytes),
        )
    except (CandidateBundleError, FileNotFoundError, OSError) as exc:
        raise ReferenceAgentRuntimeError(
            f"candidate bundle is invalid: {exc}"
        ) from exc


def _read_candidate_bundle_for_runtime(
    path: Path,
    *,
    max_bytes: int = _MAX_BUNDLE_BYTES,
) -> tuple[bytes, CandidateBundle | None]:
    if not _is_multifile_candidate_locator(path):
        return _read_python_bundle(path, max_bytes=max_bytes), None
    bundle = _inspect_runtime_candidate_bundle(path, max_bytes=max_bytes)
    return _canonical_candidate_archive(bundle, max_bytes=max_bytes), bundle


def _read_candidate_bundle_payload(
    path: Path,
    *,
    max_bytes: int = _MAX_BUNDLE_BYTES,
) -> bytes:
    payload, _bundle = _read_candidate_bundle_for_runtime(
        path,
        max_bytes=max_bytes,
    )
    return payload


def _canonical_candidate_archive(
    bundle: CandidateBundle,
    *,
    max_bytes: int,
) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=False,
    ) as archive:
        manifest = (
            json.dumps(
                bundle.normalized_manifest(),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        _write_canonical_archive_member(
            archive,
            CANDIDATE_BUNDLE_MANIFEST,
            manifest,
        )
        for member in bundle.members:
            source = bundle.root.joinpath(*member.path.split("/"))
            payload = _read_bounded_regular_file(
                source,
                max_bytes=max(1, member.byte_length),
                label=f"candidate bundle member {member.path}",
            )
            if len(payload) != member.byte_length:
                raise ReferenceAgentRuntimeError(
                    f"candidate bundle member changed while archived: {member.path}"
                )
            _write_canonical_archive_member(archive, member.path, payload)
    payload = _CANDIDATE_ARCHIVE_MAGIC + stream.getvalue()
    if len(payload) > max_bytes:
        raise ReferenceAgentRuntimeError(
            f"candidate bundle archive exceeds {max_bytes} bytes"
        )
    return payload


def _write_canonical_archive_member(
    archive: zipfile.ZipFile,
    name: str,
    payload: bytes,
) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o444) << 16
    archive.writestr(info, payload)


def _seal_reference_candidate_archive(
    storage_root: Path,
    bundle_sha256: str,
    payload: bytes,
    max_bytes: int,
) -> str:
    if len(payload) > max_bytes:
        raise ReferenceAgentRuntimeError(
            f"candidate bundle archive exceeds {max_bytes} bytes"
        )
    with tempfile.TemporaryDirectory(
        prefix=".candidate-archive-",
        dir=storage_root,
    ) as temporary:
        unpacked = Path(temporary)
        _extract_candidate_archive(payload, unpacked, max_bytes=max_bytes)
        source = _inspect_runtime_candidate_bundle(
            unpacked,
            max_bytes=max_bytes,
        )
        if _canonical_candidate_archive(source, max_bytes=max_bytes) != payload:
            raise ReferenceAgentRuntimeError(
                "candidate archive is not in canonical form"
            )
        try:
            frozen = freeze_candidate_bundle(
                source,
                storage_root / "bundle_trees",
                limits=_candidate_bundle_limits(max_bytes),
            )
        except (CandidateBundleError, FileNotFoundError, OSError) as exc:
            raise ReferenceAgentRuntimeError(
                f"candidate bundle could not be frozen: {exc}"
            ) from exc
    return str(frozen.root)


def _extract_candidate_archive(
    payload: bytes,
    destination: Path,
    *,
    max_bytes: int,
) -> None:
    if not payload.startswith(_CANDIDATE_ARCHIVE_MAGIC):
        raise ReferenceAgentRuntimeError("candidate archive magic is invalid")
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload[len(_CANDIDATE_ARCHIVE_MAGIC) :]))
    except zipfile.BadZipFile as exc:
        raise ReferenceAgentRuntimeError("candidate archive is invalid") from exc
    seen: set[str] = set()
    total_bytes = 0
    with archive:
        for info in archive.infolist():
            name = info.filename
            parts = name.split("/")
            if (
                not name
                or name.startswith("/")
                or "\\" in name
                or any(part in {"", ".", ".."} for part in parts)
                or info.is_dir()
                or info.flag_bits & 0x1
                or info.compress_type != zipfile.ZIP_STORED
                or name in seen
            ):
                raise ReferenceAgentRuntimeError(
                    f"candidate archive member is unsafe: {name!r}"
                )
            seen.add(name)
            total_bytes += info.file_size
            if total_bytes > max_bytes:
                raise ReferenceAgentRuntimeError(
                    f"candidate archive contents exceed {max_bytes} bytes"
                )
            member_payload = archive.read(info)
            target = destination.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_bytes_exclusive(target, member_payload, mode=0o600)
    if CANDIDATE_BUNDLE_MANIFEST not in seen:
        raise ReferenceAgentRuntimeError(
            "candidate archive lacks its normalized manifest"
        )


def _stage_reference_candidate_bundle(
    bundle_uri: Path | str,
    destination_root: Path,
    *,
    expected_sha256: str,
    legacy_filename: str = "candidate.py",
    max_bytes: int = _MAX_BUNDLE_BYTES,
) -> Path:
    """Stage an exact candidate tree without adding controller-owned assets."""

    _require_sha256(expected_sha256, "candidate bundle")
    source_path = _local_bundle_path(str(bundle_uri))
    payload, bundle = _read_candidate_bundle_for_runtime(
        source_path,
        max_bytes=max_bytes,
    )
    if destination_root.exists():
        if destination_root.is_symlink() or not destination_root.is_dir():
            raise ReferenceAgentRuntimeError(
                "candidate staging root must be a regular directory"
            )
        if any(destination_root.iterdir()):
            raise ReferenceAgentRuntimeError("candidate staging root is not empty")
    else:
        destination_root.mkdir(mode=0o700, parents=False)

    if bundle is None:
        destination = destination_root / legacy_filename
        _write_bytes_exclusive(destination, payload, mode=0o400)
        return destination

    for member in bundle.members:
        source = bundle.root.joinpath(*member.path.split("/"))
        member_payload = _read_bounded_regular_file(
            source,
            max_bytes=max(1, member.byte_length),
            label=f"candidate bundle member {member.path}",
        )
        if len(member_payload) != member.byte_length:
            raise ReferenceAgentRuntimeError(
                f"candidate bundle member changed while staged: {member.path}"
            )
        destination = destination_root.joinpath(*member.path.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_bytes_exclusive(destination, member_payload, mode=0o400)
    manifest = (
        json.dumps(
            bundle.normalized_manifest(),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _write_bytes_exclusive(
        destination_root / CANDIDATE_BUNDLE_MANIFEST,
        manifest,
        mode=0o400,
    )
    staged_entrypoint = destination_root.joinpath(*bundle.entrypoint.split("/"))
    directories = [destination_root]
    directories.extend(
        path for path in destination_root.rglob("*") if path.is_dir()
    )
    for directory in sorted(
        directories,
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        directory.chmod(0o500)
    return staged_entrypoint

def _read_python_bundle(
    path: Path,
    *,
    max_bytes: int = _MAX_BUNDLE_BYTES,
) -> bytes:
    if path.suffix != ".py":
        raise ReferenceAgentRuntimeError("candidate bundle must be one .py file")
    return _read_bounded_regular_file(
        path,
        max_bytes=max_bytes,
        label="candidate bundle",
        allow_empty=False,
    )


def _read_bounded_regular_file(
    path: Path,
    *,
    max_bytes: int,
    label: str,
    allow_empty: bool = True,
) -> bytes:
    """Read one stable regular-file descriptor without following a leaf link."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("bounded file size must be a positive integer")
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise ReferenceAgentRuntimeError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ReferenceAgentRuntimeError(
            f"{label} must be a regular, non-symlink file"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReferenceAgentRuntimeError(f"{label} cannot be opened safely") from exc
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ReferenceAgentRuntimeError(f"{label} changed before it was opened")
        payload = stream.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ReferenceAgentRuntimeError(f"{label} exceeds {max_bytes} bytes")
    if not payload and not allow_empty:
        raise ReferenceAgentRuntimeError(f"{label} cannot be empty")
    return payload


def _local_bundle_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme not in {"", "file"}:
        raise ReferenceAgentRuntimeError(
            "reference candidate bundle_uri must be a local path or file URI"
        )
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            raise ReferenceAgentRuntimeError("remote file bundle URIs are unsupported")
        path = Path(unquote(parsed.path))
    else:
        path = Path(uri)
    return Path(os.path.abspath(os.fspath(path)))


def _resolve_agent_python_file(workspace: Path, path: Path) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    resolved = absolute.resolve()
    try:
        resolved.relative_to(workspace.resolve())
    except ValueError as exc:
        raise ReferenceAgentRuntimeError(
            "Agent candidate must stay inside its public workspace"
        ) from exc
    _read_candidate_bundle_payload(absolute)
    return absolute


def _resolve_within(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ReferenceAgentRuntimeError("workspace path cannot be empty")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ReferenceAgentRuntimeError("workspace path escaped its root") from exc
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _ensure_private_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise ReferenceAgentRuntimeError(
            f"cannot create private directory: {path}"
        ) from exc
    if path.is_symlink() or not path.is_dir():
        raise ReferenceAgentRuntimeError(f"private path is not a directory: {path}")
    os.chmod(path, 0o700)


def _write_bytes_exclusive(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    _ensure_private_directory(path.parent)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, mode)


def _write_json_exclusive(
    path: Path,
    payload: Mapping[str, object],
    *,
    mode: int = 0o600,
    max_bytes: int | None = None,
) -> None:
    encoded = _json_bytes(payload) + b"\n"
    if max_bytes is not None:
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes <= 0
        ):
            raise ValueError("bounded JSON size must be a positive integer")
        if len(encoded) > max_bytes:
            raise ReferenceAgentRuntimeError(
                f"JSON payload exceeds {max_bytes} bytes"
            )
    _write_bytes_exclusive(path, encoded, mode=mode)


def _replace_json_atomically(
    path: Path,
    payload: Mapping[str, object],
    *,
    mode: int,
    max_bytes: int,
) -> None:
    """Replace one controller-owned JSON file without exposing a partial write."""

    encoded = _json_bytes(payload) + b"\n"
    if len(encoded) > max_bytes:
        raise ReferenceAgentRuntimeError(
            f"JSON payload exceeds {max_bytes} bytes"
        )
    _ensure_private_directory(path.parent)
    temporary = path.with_name(
        f".{path.name}.{secrets.token_hex(16)}.tmp"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, mode)
    replaced = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        replaced = True
    finally:
        if not replaced:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _read_json_object(
    path: Path,
    *,
    max_bytes: int = _MAX_CONTEXT_JSON_BYTES,
) -> Mapping[str, object]:
    try:
        encoded = _read_bounded_regular_file(
            path,
            max_bytes=max_bytes,
            label="JSON payload",
            allow_empty=False,
        )
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReferenceAgentRuntimeError(f"cannot read JSON object: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ReferenceAgentRuntimeError(f"JSON payload must be an object: {path}")
    return payload


def _required_string(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ReferenceAgentRuntimeError(f"{name} must be a nonempty string")
    return value


def _required_nonnegative_int(payload: Mapping[str, object], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReferenceAgentRuntimeError(f"{name} must be a nonnegative integer")
    return value


def _required_positive_int(payload: Mapping[str, object], name: str) -> int:
    value = _required_nonnegative_int(payload, name)
    if value <= 0:
        raise ReferenceAgentRuntimeError(
            f"{name} must be a positive integer"
        )
    return value


def _required_nonnegative_float(
    payload: Mapping[str, object], name: str
) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ReferenceAgentRuntimeError(f"{name} must be a nonnegative number")
    if not isfinite(float(value)) or float(value) < 0.0:
        raise ReferenceAgentRuntimeError(f"{name} must be a nonnegative number")
    return float(value)


def _required_positive_float(
    payload: Mapping[str, object], name: str
) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ReferenceAgentRuntimeError(f"{name} must be a positive number")
    if not isfinite(float(value)) or float(value) <= 0.0:
        raise ReferenceAgentRuntimeError(f"{name} must be a positive number")
    return float(value)


def _required_sha256(payload: Mapping[str, object], name: str) -> str:
    value = _required_string(payload, name)
    _require_sha256(value, name)
    return value


def _require_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} sha256 must be 64 lowercase hex characters")


if __name__ == "__main__":
    raise SystemExit(main())
