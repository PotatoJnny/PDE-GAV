"""PDE-neutral loopback capabilities for a canonical subprocess Agent turn."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import os
from pathlib import Path
import secrets
import select
import sys
import socket
from threading import Condition, Event, Lock, Thread
import time
from typing import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Artifact,
    Case,
    CaseAccessKind,
    CaseSource,
    CaseVisibility,
    PairedSampleBatch,
    PairedSampleRequest,
    PerturbCasesRequest,
    QueryKind,
    SampleBatch,
    SampleCasesRequest,
    SampleSource,
)

from .agent_execution import (
    AGENT_CAPABILITY_SCHEMA,
    AGENT_WORKSPACE_ENV,
    AgentCapabilityLease,
    AgentExecutionError,
    CommandAgentProgram,
)
from .agent_wire import AgentWireCodec, AgentWireCodecError
from .evaluator import (
    EvaluationCaseResult,
    EvaluationRequest,
    EvaluationResponse,
    evaluation_request_cancellation,
)


from .candidate_execution import (
    CandidateExecutionCancelled, CandidateExecutionInfrastructureError,
    CandidateExecutionError, CandidateExecutionTimeout, CandidateExecutionBatchTimeout,
)
from .python_candidate import PythonCandidateError
from .candidate_diagnostics import candidate_diagnostic, normalize_candidate_diagnostic
from pde_agent_verifier_solver.sdk.candidate_bundle import CandidateBundleError


AGENT_CAPABILITY_RPC_REQUEST_SCHEMA = "canonical-agent-capability-request-v1"
AGENT_CAPABILITY_RPC_RESPONSE_SCHEMA = "canonical-agent-capability-response-v1"
AGENT_CAPABILITY_RPC_ERROR_SCHEMA = "canonical-agent-capability-error-v1"
_TOKEN_HEADER = "X-PDE-Agent-Token"
_LOGGER = logging.getLogger(__name__)


class AgentCapabilityClientError(AgentExecutionError):
    """A worker-side call to its Controller-owned capability failed."""

    def __init__(self, code: str, message: str, *, http_status: int | None = None,
                 request_id: str | None = None, details: Mapping[str, object] | None = None):
        self.code = code
        self.http_status = http_status
        self.request_id = request_id
        self.details = dict(details or {})
        interrupted = self.details.get("interrupted_candidate", {})
        self.diagnostic = normalize_candidate_diagnostic(
            self.details.get("diagnostic") or (
                interrupted.get("diagnostic") if isinstance(interrupted, Mapping) else None
            )
        )
        if self.diagnostic:
            message = f"{self.diagnostic['exception_type']}: {self.diagnostic['message']}\n{message}"
        super().__init__(message)


class ControllerAgentCapabilityFacade:
    """Open one token-scoped loopback facade over an ``AgentContext``.

    The facade delegates to the exact sampler and evaluator already placed in
    the canonical context. It does not reproduce their PDE, Case budget,
    coordinate, or truth-disclosure rules.
    """

    def __init__(
        self,
        *,
        max_rpc_bytes: int = 64 * 1024 * 1024,
        inline_array_bytes: int = 64 * 1024,
        max_sidecar_bytes: int = 2 * 1024 * 1024 * 1024,
        drain_timeout_seconds: float = 30.0,
    ) -> None:
        if isinstance(max_rpc_bytes, bool) or not isinstance(max_rpc_bytes, int):
            raise TypeError("max_rpc_bytes must be an integer")
        if max_rpc_bytes < 1024:
            raise ValueError("max_rpc_bytes must be at least 1024")
        for value, label in (
            (inline_array_bytes, "inline_array_bytes"),
            (max_sidecar_bytes, "max_sidecar_bytes"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        if inline_array_bytes > max_sidecar_bytes:
            raise ValueError("inline_array_bytes cannot exceed max_sidecar_bytes")
        if (
            isinstance(drain_timeout_seconds, bool)
            or not isinstance(drain_timeout_seconds, (int, float))
            or not math.isfinite(float(drain_timeout_seconds))
            or float(drain_timeout_seconds) <= 0
        ):
            raise ValueError("drain_timeout_seconds must be positive and finite")
        self.max_rpc_bytes = max_rpc_bytes
        self.inline_array_bytes = inline_array_bytes
        self.max_sidecar_bytes = max_sidecar_bytes
        self.drain_timeout_seconds = float(drain_timeout_seconds)

    def open(
        self,
        context,
        workspace: Path,
        execution_id: str,
        program: CommandAgentProgram,
    ) -> AgentCapabilityLease:
        if not isinstance(program, CommandAgentProgram):
            raise TypeError("Controller capability needs a CommandAgentProgram")
        host = _ControllerCapabilityHost(
            context=context,
            workspace=workspace,
            execution_id=execution_id,
            program=program,
            max_rpc_bytes=self.max_rpc_bytes,
            inline_array_bytes=self.inline_array_bytes,
            max_sidecar_bytes=self.max_sidecar_bytes,
            drain_timeout_seconds=self.drain_timeout_seconds,
        )
        host.start()
        return AgentCapabilityLease(
            host.manifest,
            revoke=host.revoke,
            drain=host.drain,
            service_seconds=host.controller_service_seconds,
        )


class ControllerAgentCapabilityClient:
    """Worker-side client for the capabilities in its serialized request."""

    def __init__(
        self,
        capability: Mapping[str, object],
        *,
        workspace: str | Path | None = None,
        timeout_seconds: float = 300.0,
        max_rpc_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if not isinstance(capability, Mapping):
            raise AgentCapabilityClientError(
                "invalid_capability",
                "Agent capability manifest must be an object",
            )
        if capability.get("schema") != AGENT_CAPABILITY_SCHEMA:
            raise AgentCapabilityClientError(
                "invalid_capability",
                "Agent capability manifest has an unknown schema",
            )
        endpoint = capability.get("endpoint")
        token = capability.get("token")
        _validate_loopback_endpoint(endpoint)
        if not isinstance(token, str) or not token:
            raise AgentCapabilityClientError(
                "invalid_capability",
                "Agent capability token is invalid",
            )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or float(timeout_seconds) <= 0
        ):
            raise ValueError("capability timeout must be positive and finite")
        if isinstance(max_rpc_bytes, bool) or not isinstance(max_rpc_bytes, int):
            raise TypeError("max_rpc_bytes must be an integer")
        if max_rpc_bytes < 1024:
            raise ValueError("max_rpc_bytes must be at least 1024")
        self._endpoint = endpoint
        self._token = token
        self._timeout_seconds = float(timeout_seconds)
        self._max_rpc_bytes = max_rpc_bytes
        self._sampling_available = _availability(capability, "sampling")
        self._evaluation_available = _availability(capability, "evaluation")
        self._public_equation_methods = _oracle_capability_methods(
            capability,
            "equation_oracle",
        )
        self._development_solution_methods = _oracle_capability_methods(
            capability,
            "development_solution_oracle",
        )
        wire = capability.get("wire")
        if not isinstance(wire, Mapping):
            raise AgentCapabilityClientError(
                "invalid_capability",
                "Agent capability has no wire codec declaration",
            )
        _exact_keys(
            wire,
            {
                "codec",
                "sidecar_directory",
                "inline_array_bytes",
                "max_sidecar_bytes",
                "max_rpc_bytes",
            },
            "wire capability",
        )
        if wire.get("codec") != "safe-json-npy-v1":
            raise AgentCapabilityClientError(
                "invalid_capability",
                "Agent capability wire codec is unsupported",
            )
        sidecar_directory = wire.get("sidecar_directory")
        if sidecar_directory != "capability-arrays":
            raise AgentCapabilityClientError(
                "invalid_capability",
                "Agent capability sidecar directory is invalid",
            )
        workspace_value = workspace or os.environ.get(AGENT_WORKSPACE_ENV)
        if workspace_value is None:
            raise AgentCapabilityClientError(
                "invalid_capability",
                "Agent capability client needs its execution workspace",
            )
        workspace_path = Path(workspace_value).resolve()
        sidecar_root = (workspace_path / sidecar_directory).resolve()
        if sidecar_root.parent != workspace_path:
            raise AgentCapabilityClientError(
                "invalid_capability",
                "Agent capability sidecar escaped its execution workspace",
            )
        declared_rpc_bytes = _positive_manifest_int(wire, "max_rpc_bytes")
        declared_inline_bytes = _positive_manifest_int(wire, "inline_array_bytes")
        declared_sidecar_bytes = _positive_manifest_int(wire, "max_sidecar_bytes")
        self._max_rpc_bytes = min(max_rpc_bytes, declared_rpc_bytes)
        self._codec = AgentWireCodec(
            sidecar_root,
            inline_array_bytes=declared_inline_bytes,
            max_sidecar_bytes=declared_sidecar_bytes,
        )
        self._opener = build_opener(ProxyHandler({}))

    @classmethod
    def from_request(
        cls,
        request: Mapping[str, object],
        **kwargs,
    ) -> ControllerAgentCapabilityClient:
        capability = request.get("capabilities")
        if not isinstance(capability, Mapping):
            raise AgentCapabilityClientError(
                "invalid_capability",
                "Agent request has no capability manifest",
            )
        return cls(capability, **kwargs)

    def current_cases(self) -> tuple[Case, ...]:
        result = self._call("current_cases", {})
        return _case_sequence(result.get("cases"), self._codec)

    def current_status(self) -> tuple[tuple[Case, ...], list[Mapping[str, object]]]:
        result = self._call("current_cases", {})
        return _case_sequence(result.get("cases"), self._codec), result.get("evaluations", [])

    def evaluation_status(self, request_id: str) -> Mapping[str, object]:
        return self._call("evaluation_status", {"request_id": request_id}, timeout_seconds=2.0)

    def cancel_evaluation(
        self, request_id: str, *, wait_seconds: float = 30.0,
    ) -> Mapping[str, object]:
        if not math.isfinite(wait_seconds) or not 0 <= wait_seconds <= 30:
            raise ValueError("cancellation wait must be between 0 and 30 seconds")
        deadline = time.monotonic() + wait_seconds
        result = self._call("cancel_evaluation", {"request_id": request_id}, timeout_seconds=2.0)
        while not result.get("finished") and time.monotonic() < deadline:
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
            result = self.evaluation_status(request_id)
        return result

    def _cancel_interrupted_evaluation(self, request_id: str) -> None:
        try:
            result = self.cancel_evaluation(request_id)
        except BaseException:
            result = {"state": "cancellation_unconfirmed", "finished": False}
        print(json.dumps({"evaluation_request_id": request_id, "cancellation": result}),
              file=sys.stderr, flush=True)

    def sample_cases(self, source: SampleSource, count: int) -> SampleBatch:
        self._require_sampling()
        if not isinstance(source, SampleSource):
            raise TypeError("sample source must be a SampleSource")
        result = self._call(
            "sample",
            {"source": source.value, "count": count},
        )
        return _sample_batch_from_payload(result, self._codec)

    def sample_base(self, count: int) -> SampleBatch:
        return self.sample_cases(SampleSource.BASE, count)

    def sample_guided(self, count: int) -> SampleBatch:
        return self.sample_cases(SampleSource.GUIDED, count)

    def sample_paired_cases(
        self,
        source: SampleSource,
        count: int,
        *,
        max_attempts_per_root: int = 16,
    ) -> PairedSampleBatch:
        self._require_sampling()
        if not isinstance(source, SampleSource):
            raise TypeError("paired sample source must be a SampleSource")
        result = self._call(
            "sample_paired",
            {
                "source": source.value,
                "count": count,
                "max_attempts_per_root": max_attempts_per_root,
            },
        )
        roots = _case_sequence(result.get("roots"), self._codec)
        children = _case_sequence(result.get("children"), self._codec)
        return PairedSampleBatch(
            roots=roots,
            children=children,
            remaining=(None if "remaining" in result and result["remaining"] is None else
                       _nonnegative_int(result.get("remaining"), "remaining")),
            # Retry/audit details stay Controller-private.
            attempts=(),
        )

    def perturb_cases(self, parent_case_ids: Sequence[str]) -> SampleBatch:
        self._require_sampling()
        result = self._call(
            "perturb",
            {"parent_case_ids": list(parent_case_ids)},
        )
        return _sample_batch_from_payload(result, self._codec)

    def evaluate(
        self,
        candidate: Mapping[str, object],
        *,
        case_ids: Sequence[str],
        query: AnswerQuery | None = None,
        return_truth: bool = False,
    ) -> EvaluationResponse:
        return self.evaluate_many(
            (candidate,),
            case_ids=case_ids,
            query=query,
            return_truth=return_truth,
        )[0]

    def evaluate_many(
        self,
        candidates: Sequence[Mapping[str, object]],
        *,
        case_ids: Sequence[str],
        query: AnswerQuery | None = None,
        return_truth: bool = False,
    ) -> tuple[EvaluationResponse, ...]:
        self._require_evaluation()
        candidates = tuple(candidates)
        if not candidates or any(not isinstance(item, Mapping) for item in candidates):
            raise ValueError("evaluation needs one or more Candidate payloads")
        if not isinstance(return_truth, bool):
            raise TypeError("return_truth must be boolean")
        result = self._call(
            "evaluate",
            {
                "candidates": [self._codec.encode(dict(item)) for item in candidates],
                "case_ids": list(case_ids),
                "query": None if query is None else _query_to_payload(query),
                "return_truth": return_truth,
            },
        )
        responses = result.get("responses")
        if not isinstance(responses, list) or len(responses) != len(candidates):
            raise AgentCapabilityClientError(
                "malformed_response",
                "Controller returned the wrong number of evaluation responses",
            )
        return tuple(
            _evaluation_from_payload(item, self._codec)
            for item in responses
        )

    def invoke_public_problem(
        self,
        case_id: str,
        method: str,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Call one equation-public method for an issued opaque Case id."""

        return self._invoke_case_session(
            "public_equation_oracle",
            self._public_equation_methods,
            case_id,
            method,
            args,
            kwargs,
        )

    def invoke_development_solution(
        self,
        case_id: str,
        method: str,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Call one explicitly authorized Development-solution method."""

        return self._invoke_case_session(
            "development_solution_oracle",
            self._development_solution_methods,
            case_id,
            method,
            args,
            kwargs,
        )

    def _invoke_case_session(
        self,
        operation: str,
        allowed_methods: tuple[str, ...],
        case_id: str,
        method: str,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> object:
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("case_id must be a nonempty opaque id")
        if method not in allowed_methods:
            raise AgentCapabilityClientError(
                "capability_unavailable",
                f"case-session method {method!r} is not available in this Agent turn",
            )
        result = self._call(
            operation,
            {
                "case_id": case_id,
                "method": method,
                "args": self._codec.encode(args),
                "kwargs": self._codec.encode(dict(kwargs)),
            },
        )
        if set(result) != {"value"}:
            raise AgentCapabilityClientError(
                "malformed_response",
                "Controller returned a malformed case-session response",
            )
        return self._codec.decode(result["value"])

    def _require_sampling(self) -> None:
        if not self._sampling_available:
            raise AgentCapabilityClientError(
                "capability_unavailable",
                "sampling is unavailable in this Agent turn",
            )

    def _require_evaluation(self) -> None:
        if not self._evaluation_available:
            raise AgentCapabilityClientError(
                "capability_unavailable",
                "evaluation is unavailable in this Agent turn",
            )

    def _call(
        self,
        operation: str,
        params: Mapping[str, object],
        *,
        timeout_seconds: float | None = None,
    ) -> Mapping[str, object]:
        request_id = uuid4().hex
        body = _json_bytes(
            {
                "schema": AGENT_CAPABILITY_RPC_REQUEST_SCHEMA,
                "request_id": request_id,
                "operation": operation,
                "params": dict(params),
            }
        )
        if len(body) > self._max_rpc_bytes:
            raise AgentCapabilityClientError(
                "request_too_large",
                "Agent capability request exceeds the size limit",
            )
        request = Request(
            self._endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                _TOKEN_HEADER: self._token,
            },
        )
        try:
            with self._opener.open(
                request, timeout=(self._timeout_seconds if timeout_seconds is None else timeout_seconds),
            ) as response:
                result = _read_http_json(response, self._max_rpc_bytes)
        except (KeyboardInterrupt, SystemExit):
            if operation == "evaluate":
                self._cancel_interrupted_evaluation(request_id)
            raise
        except HTTPError as exc:
            details = {}
            try:
                payload = _read_http_json(exc, self._max_rpc_bytes)
                error = payload["error"]
                code = str(error["code"])
                message = str(error["message"])
                if isinstance(error.get("details"), Mapping):
                    details = dict(error["details"])
            except Exception:
                code = "malformed_error"
                message = "Controller returned a malformed capability error"
            raise AgentCapabilityClientError(
                code,
                message,
                http_status=exc.code, request_id=request_id, details=details,
            ) from exc
        except (URLError, TimeoutError, socket.timeout) as exc:
            if operation == "evaluate":
                self._cancel_interrupted_evaluation(request_id)
            raise AgentCapabilityClientError(
                "capability_unavailable",
                "Controller capability is unavailable",
            ) from exc
        if (
            result.get("schema") != AGENT_CAPABILITY_RPC_RESPONSE_SCHEMA
            or result.get("request_id") != request_id
            or not isinstance(result.get("result"), Mapping)
        ):
            raise AgentCapabilityClientError(
                "malformed_response",
                "Controller returned a malformed capability response",
            )
        return result["result"]


class _ControllerServiceClock:
    """Monotonic union clock; RPC execution is serialized by the caller."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._committed_seconds = 0.0
        self._active_started: float | None = None

    def begin(self) -> None:
        with self._lock:
            if self._active_started is not None:
                raise RuntimeError("Controller capability calls cannot overlap")
            self._active_started = time.monotonic()

    def end(self) -> None:
        now = time.monotonic()
        with self._lock:
            if self._active_started is None:
                raise RuntimeError("Controller capability clock is not active")
            self._committed_seconds += max(0.0, now - self._active_started)
            self._active_started = None

    def seconds(self) -> float:
        now = time.monotonic()
        with self._lock:
            provisional = (
                0.0
                if self._active_started is None
                else max(0.0, now - self._active_started)
            )
            return self._committed_seconds + provisional


class _ControllerCapabilityState:
    def __init__(
        self,
        *,
        context,
        workspace: Path,
        program: CommandAgentProgram,
        codec: AgentWireCodec,
    ) -> None:
        self.context = context
        self.workspace = workspace
        self.program = program
        self.codec = codec
        self.clock = _ControllerServiceClock()
        self._operation_lock = Lock()
        self._status_lock = Lock()
        self._requests: dict[str, dict[str, object]] = {}
        self._cleanup_failed = False
        self._lifecycle = Condition(Lock())
        self._accepting = True
        self._active_calls = 0

    def call(
        self, operation: str, params: Mapping[str, object], *,
        request_id: str | None = None, disconnected: Event | None = None,
    ) -> Mapping[str, object]:
        record = None
        with self._lifecycle:
            if not self._accepting:
                raise PermissionError("Agent capability has expired")
            if operation == "evaluate":
                key = request_id or uuid4().hex
                if len(key) > 128 or key in self._requests:
                    raise ValueError("invalid or duplicate evaluation request ID")
                record = {"request_id": key, "state": "queued", "finished": False,
                          "cleanup_confirmed": False, "cancel": disconnected or Event()}
                self._requests[key] = record
                # Retain recent terminal receipts; never discard active work.
                terminal = [key for key, row in self._requests.items() if row["finished"]]
                for key in terminal[:-128]:
                    del self._requests[key]
            self._active_calls += 1
        try:
            if operation in {"cancel_evaluation", "evaluation_status"}:
                _exact_keys(params, {"request_id"}, operation)
                return self.request_status(
                    _required_string(params, "request_id"), cancel=operation == "cancel_evaluation",
                )
            lock = self._status_lock if operation == "current_cases" else self._operation_lock
            while not lock.acquire(timeout=0.05):
                self._check_dispatch(record, scientific=operation != "current_cases")
            try:
                self._check_dispatch(record, scientific=operation != "current_cases")
                if record is not None:
                    with self._lifecycle:
                        record["state"] = "running"
                if operation == "current_cases":
                    # Evaluator.cases takes its own short state lock. Reads neither
                    # wait for numerical work nor change the scientific service clock.
                    return self._dispatch(operation, params)
                self.clock.begin()
                try:
                    if record is None:
                        return self._dispatch(operation, params)
                    with evaluation_request_cancellation(record["cancel"]):
                        result = self._dispatch(operation, params)
                    with self._lifecycle:
                        if record["cancel"].is_set():
                            raise CandidateExecutionCancelled("Evaluation request cancelled")
                        record.update(state="completed", finished=True, cleanup_confirmed=True)
                    return result
                finally:
                    self.clock.end()
            finally:
                lock.release()
        except BaseException as exc:
            if record is not None:
                cleanup_failed = _request_cleanup_failed(exc)
                progress = getattr(exc, "evaluation_progress", None)
                diagnostic = candidate_diagnostic(exc)
                if progress is None and diagnostic:
                    progress = {"complete": False, "diagnostic": diagnostic}
                    exc.evaluation_progress = progress
                with self._lifecycle:
                    if progress is not None:
                        record["evaluation_progress"] = progress
                    if cleanup_failed:
                        self._cleanup_failed = True
                    if not record["finished"]:
                        record.update(
                            state=("cleanup_failed" if cleanup_failed else
                                   "cancelled" if isinstance(exc, CandidateExecutionCancelled) else "failed"),
                            finished=True, cleanup_confirmed=not cleanup_failed,
                        )
            raise
        finally:
            with self._lifecycle:
                self._active_calls -= 1
                self._lifecycle.notify_all()

    def _check_dispatch(self, record, *, scientific: bool) -> None:
        with self._lifecycle:
            if scientific and self._cleanup_failed:
                raise CandidateExecutionInfrastructureError("Agent evaluation resources require cleanup")
            if record is not None and record["cancel"].is_set():
                raise CandidateExecutionCancelled("Evaluation request cancelled before dispatch")
            if not self._accepting:
                raise PermissionError("Agent capability has expired")

    def request_status(self, request_id: str, *, cancel: bool = False) -> Mapping[str, object]:
        with self._lifecycle:
            record = self._requests.get(request_id)
            if record is None:
                return {"request_id": request_id, "state": "unknown", "finished": False,
                        "cleanup_confirmed": False}
            if cancel and not record["finished"]:
                record["cancel"].set()
            return self._public_request(record)

    @staticmethod
    def _public_request(record) -> Mapping[str, object]:
        result = {key: record[key] for key in
                  ("request_id", "state", "finished", "cleanup_confirmed")}
        if "evaluation_progress" in record:
            result["evaluation_progress"] = record["evaluation_progress"]
        if not record["finished"] and record["cancel"].is_set():
            result["state"] = "cancelling"
        return result

    def revoke(self) -> None:
        transitioned = False
        with self._lifecycle:
            transitioned = self._accepting
            self._accepting = False
            for record in self._requests.values():
                if not record["finished"]:
                    record["cancel"].set()
            self._lifecycle.notify_all()
        if transitioned:
            cancel = getattr(
                self.context.evaluator,
                "cancel_pending_evaluations",
                None,
            )
            if callable(cancel):
                cancel()

    def wait_drained(self, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + timeout_seconds
        with self._lifecycle:
            while self._active_calls:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._lifecycle.wait(remaining)
            return True

    def _dispatch(
        self,
        operation: str,
        params: Mapping[str, object],
    ) -> Mapping[str, object]:
        if operation == "current_cases":
            _exact_keys(params, set(), "current_cases")
            cases = tuple(self.context.cases)
            with self._lifecycle:
                evaluations = [self._public_request(row) for row in self._requests.values()]
            return {"cases": [_case_to_payload(case, self.codec) for case in cases],
                    "evaluations": evaluations}
        if operation == "sample":
            _exact_keys(params, {"source", "count"}, "sample")
            sampler = self._sampler()
            request = SampleCasesRequest(
                SampleSource(_required_string(params, "source")),
                _positive_int(params.get("count"), "sample count"),
            )
            return _sample_batch_to_payload(
                sampler.sample_cases(request),
                self.codec,
            )
        if operation == "sample_paired":
            _exact_keys(
                params,
                {"source", "count", "max_attempts_per_root"},
                "sample_paired",
            )
            sampler = self._sampler()
            request = PairedSampleRequest(
                source=SampleSource(_required_string(params, "source")),
                count=_positive_int(params.get("count"), "paired sample count"),
                max_attempts_per_root=_positive_int(
                    params.get("max_attempts_per_root"),
                    "paired sample maximum attempts",
                ),
            )
            batch = sampler.sample_paired_cases(request)
            return {
                "roots": [
                    _case_to_payload(case, self.codec) for case in batch.roots
                ],
                "children": [
                    _case_to_payload(case, self.codec) for case in batch.children
                ],
                "remaining": batch.remaining,
            }
        if operation == "perturb":
            _exact_keys(params, {"parent_case_ids"}, "perturb")
            sampler = self._sampler()
            parent_case_ids = _string_sequence(
                params.get("parent_case_ids"),
                "parent_case_ids",
            )
            return _sample_batch_to_payload(
                sampler.perturb_cases(PerturbCasesRequest(parent_case_ids)),
                self.codec,
            )
        if operation == "evaluate":
            return self._evaluate(params)
        if operation == "public_equation_oracle":
            return self._case_session(
                params,
                CaseAccessKind.PUBLIC_PROBLEM,
            )
        if operation == "development_solution_oracle":
            return self._case_session(
                params,
                CaseAccessKind.DEVELOPMENT_SOLUTION,
            )
        raise ValueError(f"unknown Agent capability operation: {operation}")

    def _sampler(self):
        if self.context.sampler is None:
            raise PermissionError("sampling is unavailable in this Agent turn")
        return self.context.sampler

    def _evaluate(self, params: Mapping[str, object]) -> Mapping[str, object]:
        _exact_keys(
            params,
            {"candidates", "case_ids", "query", "return_truth"},
            "evaluate",
        )
        candidate_nodes = params.get("candidates")
        if not isinstance(candidate_nodes, list) or not candidate_nodes:
            raise ValueError("evaluation needs one or more Candidate payloads")
        candidates = []
        for node in candidate_nodes:
            payload = self.codec.decode(node)
            if not isinstance(payload, Mapping):
                raise ValueError("Candidate payload must decode to an object")
            artifact = self.program.nomination_decoder(
                payload,
                self.context,
                self.workspace,
            )
            if not isinstance(artifact, Artifact):
                raise TypeError("nomination_decoder must return Artifact")
            candidates.append(artifact)
        case_ids = _string_sequence(params.get("case_ids"), "case_ids")
        query_node = params.get("query")
        query = None if query_node is None else _query_from_payload(query_node)
        return_truth = params.get("return_truth")
        if not isinstance(return_truth, bool):
            raise TypeError("return_truth must be boolean")
        responses = self.context.evaluator.evaluate_many(
            tuple(candidates),
            EvaluationRequest(
                case_ids=case_ids,
                query=query,
                return_truth=return_truth,
            ),
        )
        return {
            "responses": [
                _evaluation_to_payload(response, self.codec)
                for response in responses
            ]
        }

    def _case_session(
        self,
        params: Mapping[str, object],
        access: CaseAccessKind,
    ) -> Mapping[str, object]:
        _exact_keys(
            params,
            {"case_id", "method", "args", "kwargs"},
            access.value,
        )
        args = self.codec.decode(params.get("args"))
        kwargs = self.codec.decode(params.get("kwargs"))
        if not isinstance(args, tuple):
            raise ValueError("case-session args must decode to a tuple")
        if not isinstance(kwargs, Mapping) or any(
            not isinstance(name, str) for name in kwargs
        ):
            raise ValueError("case-session kwargs must decode to an object")
        method = _required_string(params, "method")
        if method not in _authorized_case_session_methods(self.context, access):
            raise PermissionError(
                f"case-session method {method!r} is not authorized"
            )
        value = self.context.evaluator.invoke_case_session(
            _required_string(params, "case_id"),
            access,
            method,
            args,
            dict(kwargs),
        )
        return {"value": self.codec.encode(value)}


class _CapabilityHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = True

    def __init__(self, address, handler, *, state, token, max_rpc_bytes):
        super().__init__(address, handler)
        self.state = state
        self.token = token
        self.max_rpc_bytes = max_rpc_bytes


class _ControllerCapabilityHost:
    def __init__(
        self,
        *,
        context,
        workspace: Path,
        execution_id: str,
        program: CommandAgentProgram,
        max_rpc_bytes: int,
        inline_array_bytes: int,
        max_sidecar_bytes: int,
        drain_timeout_seconds: float,
    ) -> None:
        self.codec = AgentWireCodec(
            workspace / "capability-arrays",
            inline_array_bytes=inline_array_bytes,
            max_sidecar_bytes=max_sidecar_bytes,
        )
        self.state = _ControllerCapabilityState(
            context=context,
            workspace=workspace,
            program=program,
            codec=self.codec,
        )
        self.token = secrets.token_urlsafe(32)
        self.max_rpc_bytes = max_rpc_bytes
        self.drain_timeout_seconds = drain_timeout_seconds
        self._server = _CapabilityHTTPServer(
            ("127.0.0.1", 0),
            _capability_handler(),
            state=self.state,
            token=self.token,
            max_rpc_bytes=max_rpc_bytes,
        )
        host, port = self._server.server_address
        self.endpoint = f"http://{host}:{port}/v1/agent-capability"
        evaluator = context.evaluator
        candidate_invocation_process = getattr(
            evaluator,
            "candidate_invocation_process",
            "controller",
        )
        if candidate_invocation_process != "bounded_same_user_subprocess":
            candidate_invocation_process = "controller"
        candidate_evaluation_process = (
            "split_subprocess_invocation_controller_scoring"
            if candidate_invocation_process == "bounded_same_user_subprocess"
            else "controller"
        )
        fixed_query = evaluator.fixed_query
        bundle = getattr(context, "public_problem_bundle", None)
        public_methods = _authorized_case_session_methods(
            context,
            CaseAccessKind.PUBLIC_PROBLEM,
        )
        development_methods = _authorized_case_session_methods(
            context,
            CaseAccessKind.DEVELOPMENT_SOLUTION,
        )
        self.manifest = {
            "schema": AGENT_CAPABILITY_SCHEMA,
            "transport": "loopback_http_json",
            "endpoint": self.endpoint,
            "token": self.token,
            "execution_id": execution_id,
            "request_schema": AGENT_CAPABILITY_RPC_REQUEST_SCHEMA,
            "response_schema": AGENT_CAPABILITY_RPC_RESPONSE_SCHEMA,
            "wire": {
                "codec": "safe-json-npy-v1",
                "sidecar_directory": "capability-arrays",
                "inline_array_bytes": inline_array_bytes,
                "max_sidecar_bytes": max_sidecar_bytes,
                "max_rpc_bytes": max_rpc_bytes,
            },
            "sampling": {
                "available": context.sampler is not None,
                "operations": ["sample", "sample_paired", "perturb"],
            },
            "evaluation": {
                "available": True,
                "operations": ["evaluate", "cancel_evaluation", "evaluation_status"],
                "coordinate_mode": evaluator.development_coordinate_mode.value,
                "development_point_count": evaluator.development_point_count,
                "fixed_query": (
                    None if fixed_query is None else _query_to_payload(fixed_query)
                ),
                "allow_grid_truth": evaluator.allow_grid_truth,
                "allow_point_truth": evaluator.allow_point_truth,
            },
            "current_cases": {"available": True},
            "public_problem": {
                "available": bundle is not None,
                "bundle_id": None if bundle is None else bundle.bundle_id,
                "bundle_sha256": (
                    None if bundle is None else bundle.bundle_sha256
                ),
                "equation_oracle": {
                    "available": bool(public_methods),
                    "operation": "public_equation_oracle",
                    "methods": list(public_methods),
                },
                "development_solution_oracle": {
                    "available": bool(development_methods),
                    "operation": "development_solution_oracle",
                    "methods": list(development_methods),
                    "scope": "development_only_user_authorized",
                },
            },
            "assurance": {
                "loopback_only": True,
                "per_turn_token": True,
                "live_agent_context_shared": False,
                "candidate_decoder_process": "controller",
                "candidate_binding_process": "controller",
                "candidate_invocation_process": candidate_invocation_process,
                "candidate_scoring_process": "controller",
                "candidate_evaluation_process": candidate_evaluation_process,
                "hostile_filesystem_isolation": False,
                "hostile_candidate_sandbox": False,
            },
        }
        self._thread = Thread(
            target=self._server.serve_forever,
            name=f"agent-capability-{execution_id[:8]}",
            daemon=True,
        )
        self._started = False
        self._drained = False
        self._close_lock = Lock()

    def start(self) -> None:
        with self._close_lock:
            if self._started or self._drained:
                raise RuntimeError("Agent capability host is single-use")
            self._started = True
            self._thread.start()

    def revoke(self) -> None:
        self.state.revoke()

    def drain(self) -> None:
        with self._close_lock:
            if self._drained:
                return
            self.state.revoke()
            if self._started:
                self._server.shutdown()
            self._server.server_close()
            if self._started:
                self._thread.join(timeout=self.drain_timeout_seconds)
                if self._thread.is_alive():
                    raise AgentExecutionError(
                        "Agent capability listener did not stop during drain"
                    )
            if not self.state.wait_drained(self.drain_timeout_seconds):
                raise AgentExecutionError(
                    "Agent capability calls did not finish during drain"
                )
            if self.state._cleanup_failed:
                raise AgentExecutionError("Agent evaluation resource cleanup was not confirmed")
            # Publish completion only after the listener and every accepted
            # call are actually gone.  A timed-out drain remains retryable.
            self._drained = True

    def controller_service_seconds(self) -> float:
        return self.state.clock.seconds()


def _public_operation_error(error, operation):
    """Classify known failures without exporting private exception text."""
    if operation != "evaluate":
        return 500, "operation_failed", "Controller capability operation failed"
    if _request_cleanup_failed(error) or isinstance(error, CandidateExecutionInfrastructureError):
        return 500, "evaluation_infrastructure_failed", (
            "Controller could not complete evaluation or confirm resource cleanup; retain the request ID for diagnosis."
        )
    if isinstance(error, (PythonCandidateError, CandidateBundleError)):
        return 422, "candidate_bundle_invalid", (
            "Candidate bundle is invalid; check the manifest, declared files, entrypoint and Python imports. "
            "Remove undeclared backup files such as .orig or .rej."
        )
    if isinstance(error, CandidateExecutionBatchTimeout):
        return 408, "evaluation_batch_timeout", (
            "Development evaluation exceeded its batch wall limit; completed results are retained. "
            "Use a smaller batch or a faster candidate for the remaining cases."
        )
    if isinstance(error, CandidateExecutionTimeout):
        return 408, "candidate_timeout", (
            "Candidate exceeded its solve budget; check runtime and numerical resolution before retrying remaining cases."
        )
    if isinstance(error, CandidateExecutionError):
        return 422, "candidate_execution_failed", (
            "Candidate execution failed; check Python syntax, bundled dependencies, output shapes and finite values."
        )
    if getattr(error, "_hidden_evaluation_stage", None) in {
        "score_candidate", "evaluate_candidate_and_score", "aggregate_candidate",
    }:
        # A scorer exception alone does not establish candidate responsibility.
        return 500, "evaluation_scoring_failed", (
            "Evaluation could not score or aggregate the candidate output. Check output shapes and finite values; "
            "a Controller or Task error is also possible. Retain the request ID for diagnosis."
        )
    return 500, "operation_failed", "Controller capability operation failed"


def _capability_handler():
    class Handler(BaseHTTPRequestHandler):
        server: _CapabilityHTTPServer

        def do_POST(self) -> None:
            request_id: str | None = None
            if self.path != "/v1/agent-capability":
                self._error(404, "not_found", "unknown Agent capability route")
                return
            supplied_token = self.headers.get(_TOKEN_HEADER, "")
            if not secrets.compare_digest(supplied_token, self.server.token):
                self._error(403, "permission_denied", "invalid Agent capability token")
                return
            if self.headers.get_content_type() != "application/json":
                self._error(415, "invalid_request", "content type must be application/json")
                return
            raw_length = self.headers.get("Content-Length", "")
            try:
                length = int(raw_length)
            except ValueError:
                self._error(400, "invalid_request", "content length is invalid")
                return
            if length <= 0 or length > self.server.max_rpc_bytes:
                self._error(413, "invalid_request", "request size is invalid")
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                request_id, operation, params = _parse_rpc_request(payload)
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                self._error(
                    400,
                    "invalid_request",
                    "Agent capability request envelope is invalid",
                    request_id=request_id,
                )
                return

            disconnected, watcher_stop = Event(), Event()
            watcher = None
            if operation == "evaluate":
                watcher = Thread(
                    target=_watch_request_disconnect,
                    args=(self.connection, disconnected, watcher_stop),
                    name=f"agent-evaluation-disconnect-{request_id[:8]}", daemon=True,
                )
                watcher.start()
            try:
                result = self.server.state.call(
                    operation, params, request_id=request_id, disconnected=disconnected,
                )
            except CandidateExecutionCancelled as exc:
                failed_cleanup = _request_cleanup_failed(exc)
                if failed_cleanup:
                    _LOGGER.exception("Controller evaluation cancellation cleanup failed")
                self._error(
                    500 if failed_cleanup else 409,
                    "evaluation_cleanup_failed" if failed_cleanup else "evaluation_cancelled",
                    "Evaluation cleanup failed" if failed_cleanup else "Evaluation request cancelled",
                    request_id=request_id,
                    details=getattr(exc, "evaluation_progress", None),
                )
                return
            except PermissionError:
                self._error(
                    403,
                    "access_denied",
                    "Agent capability operation is not authorized",
                    request_id=request_id,
                )
                return
            except Exception as exc:
                # Trusted Controller/Task/plugin exceptions may contain raw
                # Case ids, truth locators, paths, or backend details.  Keep
                # them in the Controller log and expose only a stable boundary.
                _LOGGER.exception(
                    "Controller capability operation %r failed",
                    operation,
                    exc_info=exc,
                )
                status, code, message = _public_operation_error(exc, operation)
                self._error(
                    status, code, message, request_id=request_id,
                    details=getattr(exc, "evaluation_progress", None) if operation == "evaluate" else None,
                )
                return
            finally:
                watcher_stop.set()
                if watcher is not None:
                    watcher.join(timeout=1.0)
            self._send(
                200,
                {
                    "schema": AGENT_CAPABILITY_RPC_RESPONSE_SCHEMA,
                    "request_id": request_id,
                    "result": result,
                },
            )

        def _error(
            self,
            status: int,
            code: str,
            message: str,
            *,
            request_id: str | None = None,
            details: Mapping[str, object] | None = None,
        ) -> None:
            self._send(
                status,
                {
                    "schema": AGENT_CAPABILITY_RPC_ERROR_SCHEMA,
                    "request_id": request_id,
                    "error": {"code": code, "message": message,
                              **({"details": dict(details)} if details is not None else {})},
                },
            )

        def _send(self, status: int, payload: Mapping[str, object]) -> None:
            body = _json_bytes(payload)
            if len(body) > self.server.max_rpc_bytes:
                status = 500
                body = _json_bytes(
                    {
                        "schema": AGENT_CAPABILITY_RPC_ERROR_SCHEMA,
                        "request_id": payload.get("request_id"),
                        "error": {
                            "code": "response_too_large",
                            "message": (
                                "Controller capability response exceeds "
                                "the declared limit"
                            ),
                        },
                    }
                )
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # Work has settled already. The disconnect observer handles
                # cancellation while the request is queued or executing.
                pass

        def log_message(self, format, *args) -> None:
            del format, args

    return Handler


def _watch_request_disconnect(connection, cancelled: Event, stop: Event) -> None:
    while not stop.is_set():
        try:
            readable, _, _ = select.select([connection], [], [], 0.0)
            if readable and connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b"":
                cancelled.set()
                return
        except (OSError, ValueError):
            if not stop.is_set():
                cancelled.set()
            return
        # A completed request wakes this wait immediately: no fixed latency is
        # added to normal evaluation replies and extra bytes cannot busy-spin.
        stop.wait(0.05)


def _request_cleanup_failed(error: BaseException) -> bool:
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, CandidateExecutionInfrastructureError):
            return True
        if getattr(current, "_pde_session_cleanup_errors", ()):
            return True
        pending.extend(item for item in (current.__cause__, current.__context__) if item is not None)
        pending.extend(getattr(current, "_candidate_cleanup_errors", ()))
    return False


def _parse_rpc_request(payload: object) -> tuple[str, str, Mapping[str, object]]:
    if not isinstance(payload, Mapping):
        raise ValueError("Agent capability request must be an object")
    _exact_keys(
        payload,
        {"schema", "request_id", "operation", "params"},
        "request",
    )
    if payload.get("schema") != AGENT_CAPABILITY_RPC_REQUEST_SCHEMA:
        raise ValueError("Agent capability request has an unknown schema")
    request_id = _required_string(payload, "request_id")
    operation = _required_string(payload, "operation")
    params = payload.get("params")
    if not isinstance(params, Mapping):
        raise ValueError("Agent capability params must be an object")
    return request_id, operation, params


def _case_to_payload(
    case: Case,
    codec: AgentWireCodec,
) -> Mapping[str, object]:
    return {
        "case_id": case.case_id,
        "source": case.source.value,
        "visibility": case.visibility.value,
        "public_data": codec.encode(case.public_data),
        "round_index": case.round_index,
        "parent_case_id": case.parent_case_id,
        "tags": list(case.tags),
    }


def _case_from_payload(payload: object, codec: AgentWireCodec) -> Case:
    if not isinstance(payload, Mapping):
        raise AgentCapabilityClientError("malformed_response", "Case must be an object")
    _exact_keys(
        payload,
        {
            "case_id",
            "source",
            "visibility",
            "public_data",
            "round_index",
            "parent_case_id",
            "tags",
        },
        "Case",
    )
    public_data = codec.decode(payload["public_data"])
    if not isinstance(public_data, Mapping):
        raise AgentCapabilityClientError(
            "malformed_response",
            "Case public_data must be an object",
        )
    tags = _string_sequence(payload.get("tags"), "Case tags", allow_empty=True)
    round_index = payload.get("round_index")
    if round_index is not None:
        round_index = _nonnegative_int(round_index, "Case round_index")
    parent_case_id = payload.get("parent_case_id")
    if parent_case_id is not None and (
        not isinstance(parent_case_id, str) or not parent_case_id
    ):
        raise AgentCapabilityClientError(
            "malformed_response",
            "Case parent_case_id is invalid",
        )
    return Case(
        case_id=_required_string(payload, "case_id"),
        source=CaseSource(_required_string(payload, "source")),
        visibility=CaseVisibility(_required_string(payload, "visibility")),
        public_data=public_data,
        round_index=round_index,
        parent_case_id=parent_case_id,
        tags=tags,
    )


def _case_sequence(value: object, codec: AgentWireCodec) -> tuple[Case, ...]:
    if not isinstance(value, list):
        raise AgentCapabilityClientError(
            "malformed_response",
            "Controller Case collection must be a list",
        )
    return tuple(_case_from_payload(item, codec) for item in value)


def _sample_batch_to_payload(
    batch: SampleBatch,
    codec: AgentWireCodec,
) -> Mapping[str, object]:
    return {
        "cases": [_case_to_payload(case, codec) for case in batch.cases],
        "remaining": batch.remaining,
    }


def _sample_batch_from_payload(
    payload: Mapping[str, object],
    codec: AgentWireCodec,
) -> SampleBatch:
    return SampleBatch(
        cases=_case_sequence(payload.get("cases"), codec),
        remaining=(None if "remaining" in payload and payload["remaining"] is None else
                   _nonnegative_int(payload.get("remaining"), "remaining")),
    )


def _query_to_payload(query: AnswerQuery) -> Mapping[str, object]:
    if not isinstance(query, AnswerQuery):
        raise TypeError("query must be an AnswerQuery")
    return {
        "kind": query.kind.value,
        "fields": list(query.fields),
        "resolution": dict(query.resolution),
        "points": [dict(point) for point in query.points],
    }


def _query_from_payload(payload: object) -> AnswerQuery:
    if not isinstance(payload, Mapping):
        raise ValueError("evaluation query must be an object")
    _exact_keys(payload, {"kind", "fields", "resolution", "points"}, "query")
    fields = _string_sequence(payload.get("fields"), "query fields")
    if len(set(fields)) != len(fields):
        raise ValueError("query fields cannot repeat")
    resolution_node = payload.get("resolution")
    if not isinstance(resolution_node, Mapping) or any(
        not isinstance(key, str)
        or not key
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        for key, value in resolution_node.items()
    ):
        raise ValueError("query resolution is invalid")
    points_node = payload.get("points")
    if not isinstance(points_node, list):
        raise ValueError("query points must be a list")
    points = []
    for point in points_node:
        if not isinstance(point, Mapping) or any(
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for key, value in point.items()
        ):
            raise ValueError("query point is invalid")
        points.append({key: float(value) for key, value in point.items()})
    kind = QueryKind(_required_string(payload, "kind"))
    if kind is QueryKind.GRID:
        if not resolution_node or points:
            raise ValueError("grid query needs resolution and no explicit points")
    elif resolution_node or not points:
        raise ValueError("points query needs points and no grid resolution")
    return AnswerQuery(
        kind=kind,
        fields=fields,
        resolution={key: int(value) for key, value in resolution_node.items()},
        points=tuple(points),
    )


def _answer_to_payload(
    answer: AnswerView | None,
    codec: AgentWireCodec,
) -> object:
    if answer is None:
        return None
    return {
        "kind": answer.kind.value,
        # AnswerView uses tuples for its public ABI. Convert only these dense
        # numerical vectors to arrays so large grids use bounded NPY sidecars;
        # the client restores the tuple ABI below.
        "fields": codec.encode(_dense_numeric_mapping(answer.fields)),
        "axes": codec.encode(_dense_numeric_mapping(answer.axes)),
        "points": codec.encode(answer.points),
        "shape": list(answer.shape),
    }


def _answer_from_payload(
    payload: object,
    codec: AgentWireCodec,
) -> AnswerView | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise AgentCapabilityClientError(
            "malformed_response",
            "Answer view must be an object",
        )
    _exact_keys(payload, {"kind", "fields", "axes", "points", "shape"}, "Answer")
    fields = codec.decode(payload["fields"])
    axes = codec.decode(payload["axes"])
    points = codec.decode(payload["points"])
    shape = payload["shape"]
    if not isinstance(fields, Mapping) or not isinstance(axes, Mapping):
        raise AgentCapabilityClientError(
            "malformed_response",
            "Answer fields and axes must be objects",
        )
    if not isinstance(points, tuple) or any(not isinstance(point, Mapping) for point in points):
        raise AgentCapabilityClientError(
            "malformed_response",
            "Answer points are invalid",
        )
    if not isinstance(shape, list) or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 0
        for size in shape
    ):
        raise AgentCapabilityClientError(
            "malformed_response",
            "Answer shape is invalid",
        )
    return AnswerView(
        kind=QueryKind(_required_string(payload, "kind")),
        fields={key: tuple(value) for key, value in fields.items()},
        axes={key: tuple(value) for key, value in axes.items()},
        points=tuple(points),
        shape=tuple(shape),
    )


def _evaluation_to_payload(
    response: EvaluationResponse,
    codec: AgentWireCodec,
) -> Mapping[str, object]:
    return {
        "artifact_id": response.artifact_id,
        "primary_error": response.primary_error,
        "metrics": dict(response.metrics),
        "cases": [
            {
                "case_id": item.case_id,
                "primary_error": item.primary_error,
                "metrics": dict(item.metrics),
                "prediction": _answer_to_payload(item.prediction, codec),
                "truth": _answer_to_payload(item.truth, codec),
            }
            for item in response.cases
        ],
    }


def _evaluation_from_payload(
    payload: object,
    codec: AgentWireCodec,
) -> EvaluationResponse:
    if not isinstance(payload, Mapping):
        raise AgentCapabilityClientError(
            "malformed_response",
            "evaluation response must be an object",
        )
    _exact_keys(
        payload,
        {"artifact_id", "primary_error", "metrics", "cases"},
        "evaluation response",
    )
    cases_node = payload.get("cases")
    if not isinstance(cases_node, list):
        raise AgentCapabilityClientError(
            "malformed_response",
            "evaluation cases must be a list",
        )
    cases = []
    for item in cases_node:
        if not isinstance(item, Mapping):
            raise AgentCapabilityClientError(
                "malformed_response",
                "evaluation Case result must be an object",
            )
        _exact_keys(
            item,
            {"case_id", "primary_error", "metrics", "prediction", "truth"},
            "evaluation Case result",
        )
        cases.append(
            EvaluationCaseResult(
                case_id=_required_string(item, "case_id"),
                primary_error=_finite_float(item.get("primary_error"), "primary_error"),
                metrics=_float_mapping(item.get("metrics"), "Case metrics"),
                prediction=_answer_from_payload(item.get("prediction"), codec),
                truth=_answer_from_payload(item.get("truth"), codec),
            )
        )
    return EvaluationResponse(
        artifact_id=_required_string(payload, "artifact_id"),
        primary_error=_finite_float(payload.get("primary_error"), "primary_error"),
        metrics=_float_mapping(payload.get("metrics"), "metrics"),
        cases=tuple(cases),
    )


def _dense_numeric_mapping(
    values: Mapping[str, Sequence[float]],
) -> Mapping[str, object]:
    try:
        import numpy as np
    except ImportError:
        return values
    return {
        name: np.asarray(tuple(items))
        for name, items in values.items()
    }


def _json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AgentWireCodecError("Agent capability payload is not finite JSON") from exc


def _read_http_json(response, max_bytes: int) -> Mapping[str, object]:
    body = response.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise AgentCapabilityClientError(
            "response_too_large",
            "Controller capability response exceeds the size limit",
        )
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AgentCapabilityClientError(
            "malformed_response",
            "Controller capability response is invalid JSON",
        ) from exc
    if not isinstance(payload, Mapping):
        raise AgentCapabilityClientError(
            "malformed_response",
            "Controller capability response must be an object",
        )
    return payload


def _validate_loopback_endpoint(value: object) -> None:
    if not isinstance(value, str):
        raise AgentCapabilityClientError(
            "invalid_capability",
            "Agent capability endpoint is invalid",
        )
    parsed = urlparse(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
        or parsed.path != "/v1/agent-capability"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise AgentCapabilityClientError(
            "invalid_capability",
            "Agent capability endpoint must be loopback HTTP",
        )


def _availability(capability: Mapping[str, object], name: str) -> bool:
    value = capability.get(name)
    return isinstance(value, Mapping) and value.get("available") is True


def _oracle_capability_methods(
    capability: Mapping[str, object],
    name: str,
) -> tuple[str, ...]:
    public_problem = capability.get("public_problem")
    if public_problem is None:
        return ()
    if not isinstance(public_problem, Mapping):
        raise AgentCapabilityClientError(
            "invalid_capability",
            "Agent public-problem capability is invalid",
        )
    if public_problem.get("available") is not True:
        return ()
    oracle = public_problem.get(name)
    if not isinstance(oracle, Mapping):
        raise AgentCapabilityClientError(
            "invalid_capability",
            f"Agent {name} capability is invalid",
        )
    methods = oracle.get("methods")
    if not isinstance(methods, list) or any(
        not isinstance(method, str)
        or not method.isidentifier()
        or method.startswith("_")
        for method in methods
    ):
        raise AgentCapabilityClientError(
            "invalid_capability",
            f"Agent {name} methods are invalid",
        )
    result = tuple(methods)
    if len(result) != len(set(result)):
        raise AgentCapabilityClientError(
            "invalid_capability",
            f"Agent {name} methods contain duplicates",
        )
    if (oracle.get("available") is True) != bool(result):
        raise AgentCapabilityClientError(
            "invalid_capability",
            f"Agent {name} availability is inconsistent",
        )
    return result


def _authorized_case_session_methods(
    context,
    access: CaseAccessKind,
) -> tuple[str, ...]:
    bundle = getattr(context, "public_problem_bundle", None)
    if bundle is None:
        return ()
    declarations = (
        bundle.oracle_methods
        if access is CaseAccessKind.PUBLIC_PROBLEM
        else bundle.development_solution_oracle_methods
    )
    resolver = getattr(context.evaluator, "case_session_methods", None)
    if not callable(resolver):
        return ()
    runtime_methods = frozenset(resolver(access))
    return tuple(
        method.name for method in declarations if method.name in runtime_methods
    )


def _positive_manifest_int(payload: Mapping[str, object], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AgentCapabilityClientError(
            "invalid_capability",
            f"Agent capability {name} is invalid",
        )
    return value


def _required_string(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _string_sequence(
    value: object,
    label: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{label} must contain nonempty strings")
    result = tuple(value)
    if not allow_empty and not result:
        raise ValueError(f"{label} cannot be empty")
    return result


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AgentCapabilityClientError(
            "malformed_response",
            f"{label} must be a nonnegative integer",
        )
    return value


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AgentCapabilityClientError(
            "malformed_response",
            f"{label} must be numeric",
        )
    converted = float(value)
    if not math.isfinite(converted):
        raise AgentCapabilityClientError(
            "malformed_response",
            f"{label} must be finite",
        )
    return converted


def _float_mapping(value: object, label: str) -> Mapping[str, float]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise AgentCapabilityClientError(
            "malformed_response",
            f"{label} must be an object",
        )
    return {
        key: _finite_float(item, f"{label}.{key}")
        for key, item in value.items()
    }


def _exact_keys(payload: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{label} fields do not match the protocol")


__all__ = [
    "AGENT_CAPABILITY_RPC_ERROR_SCHEMA",
    "AGENT_CAPABILITY_RPC_REQUEST_SCHEMA",
    "AGENT_CAPABILITY_RPC_RESPONSE_SCHEMA",
    "AgentCapabilityClientError",
    "ControllerAgentCapabilityClient",
    "ControllerAgentCapabilityFacade",
]
