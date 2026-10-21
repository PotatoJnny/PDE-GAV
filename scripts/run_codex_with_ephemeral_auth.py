#!/usr/bin/env python
from __future__ import annotations

import argparse
import ctypes
import json
from math import isfinite
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from uuid import UUID


_EVENT_MAX_BYTES = 64 * 1024
_TRANSIENT_FAILURES = frozenset({
    "model at capacity", "connection interrupted", "request timed out",
    "server unavailable", "rate limit exceeded", "native response stalled",
})


def _safe_failure(value: object) -> str | None:
    """Return only fixed labels; native errors may echo prompts or credentials."""
    if isinstance(value, dict):
        value = " ".join(
            item for key in ("code", "type", "message")
            if isinstance(item := value.get(key), str)
        )
    if not isinstance(value, str):
        return None
    message = value[:_EVENT_MAX_BYTES].lower()
    if any(text in message for text in (
        "content_policy", "usage policy", "policy rejection", "rejected by policy",
        "violating our usage policy", "violation of our usage policy",
    )):
        return "prompt rejected by policy (not automatically retryable)"
    if any(text in message for text in ("invalid prompt", "invalid_prompt", "invalid request", "invalid_request")):
        return "invalid prompt / request (not automatically retryable)"
    for phrases, label in (
        (("context_length_exceeded", "context window", "maximum context length"), "context limit exceeded"),
        (("invalid_api_key", "incorrect api key", "authentication", "unauthorized",
          "invalid token", "expired token", "token expired", "refresh token", "forbidden"), "authentication rejected"),
        (("permission_denied", "permission denied"), "permission denied"),
        (("certificate verify failed", "invalid certificate"), "TLS certificate rejected"),
        (("quota", "usage limit", "usage_limit", "billing", "account_deactivated", "account deactivated"), "account quota exhausted"),
        (("model is at capacity", "model at capacity", "model_capacity", "model_overloaded", "overloaded_error", "temporarily overloaded"), "model at capacity"),
        (("rate_limit_exceeded", "rate limit", "too many requests", "http 429", "status 429"), "rate limit exceeded"),
        (("connection reset", "connection refused", "stream disconnected", "connection closed",
          "error sending request", "network error", "network is unreachable", "temporary failure in name resolution",
          "failed to connect", "unexpected eof", "stream ended before", "stream closed before"), "connection interrupted"),
        (("timed out", "timeout"), "request timed out"),
        (("internal server error", "server_error", "service unavailable", "bad gateway",
          "gateway timeout", "temporarily unavailable", "http 500", "http 502", "http 503", "http 504",
          "status 500", "status 502", "status 503", "status 504"), "server unavailable"),
        (("no space left on device",), "no space left on device"),
    ):
        if any(phrase in message for phrase in phrases):
            return label
    return None


class _ToolEventEnvelope:
    """Retain tool lifecycle fields when a JSONL event contains huge output.

    Only bounded keys and the four lifecycle strings are retained. Quoted tool
    output is skipped byte by byte, so embedded JSON cannot close another tool.
    Normal-sized events continue through the ordinary JSON decoder below.
    """

    def __init__(self) -> None:
        self.stack = []
        self.started = False
        self.valid = True
        self.quoted = False
        self.escaped = False
        self.capture = False
        self.key_token = False
        self.token = bytearray()
        self.target = None
        self.kind = None
        self.item = {}

    def feed(self, data: bytes) -> None:
        for char in data:
            if not self.valid:
                return
            if self.quoted:
                if char == 34 and not self.escaped:
                    self.quoted = False
                    if self.capture:
                        try:
                            value = json.loads(b'"' + self.token + b'"')
                        except (ValueError, UnicodeDecodeError):
                            self.valid = False
                            return
                        if self.key_token:
                            self.stack[-1]["key"] = value
                        elif self.target == "event_type":
                            self.kind = value
                        elif self.target is not None:
                            self.item[self.target] = value
                    if self.key_token:
                        self.stack[-1]["state"] = "colon"
                    continue
                if self.capture:
                    if len(self.token) >= 1024:
                        # Overlong fields cannot be tool IDs/types/statuses.
                        self.capture = False
                    else:
                        self.token.append(char)
                if self.escaped:
                    self.escaped = False
                elif char == 92:
                    self.escaped = True
                continue
            if char in b" \t\r\n":
                continue
            top = self.stack[-1] if self.stack else None
            if char == 34:
                self.key_token = top is not None and top["state"] == "key"
                self.target = None
                if self.key_token:
                    top["key"] = None
                elif top is not None and top["state"] == "value":
                    if top["path"] == () and top["key"] == "type":
                        self.target = "event_type"
                    elif top["path"] == ("item",) and top["key"] in {"id", "type", "status"}:
                        self.target = top["key"]
                    top["state"] = "after"
                else:
                    self.valid = False
                    return
                self.quoted = True
                self.escaped = False
                self.capture = self.key_token or self.target is not None
                self.token.clear()
            elif char in b"{[":
                path = None
                if top is None:
                    if self.started or char != 123:
                        self.valid = False
                        return
                    self.started = True
                    path = ()
                elif top["state"] == "value":
                    if top["path"] == () and top["key"] == "item" and char == 123:
                        path = ("item",)
                    top["state"] = "after"
                else:
                    self.valid = False
                    return
                if len(self.stack) >= 64:
                    self.valid = False
                    return
                self.stack.append({"kind": char, "path": path, "key": None,
                                   "state": "key" if char == 123 else "value"})
            elif char in b"}]":
                if top is None or (top["kind"], char) not in {(123, 125), (91, 93)}:
                    self.valid = False
                    return
                self.stack.pop()
            elif char == 58:
                if top is None or top["state"] != "colon":
                    self.valid = False
                    return
                top["state"] = "value"
            elif char == 44:
                if top is None or top["state"] != "after":
                    self.valid = False
                    return
                top["state"] = "key" if top["kind"] == 123 else "value"
                top["key"] = None
            elif top is not None and top["state"] in {"value", "after"}:
                # Scalars elsewhere in the envelope carry no lifecycle string.
                # Their characters cannot alter nesting or escaped-string state.
                if char not in b"-+0123456789.eEtruefalsnul":
                    self.valid = False
                    return
                top["state"] = "after"
            else:
                self.valid = False
                return

    def event(self):
        if (self.valid and self.started and not self.stack and not self.quoted
                and self.kind in {"item.started", "item.updated", "item.completed"}
                and isinstance(self.item.get("id"), str)):
            return {"type": self.kind, "item": self.item}
        return None


class _NativeEvents:
    """Bounded JSONL inspection, independent of verbatim stdout forwarding."""

    def __init__(self) -> None:
        self.pending = bytearray()
        self.discarding = False
        self.envelope = None
        self.failure: str | None = None
        self.thread_id: str | None = None
        self.retryable = False
        self.retry_blocked = False
        self.session_identity_mismatch = False
        self.last_activity = time.monotonic()
        self.active_tools: set[str] = set()
        self._lock = threading.RLock()

    def response_stalled(self, timeout_seconds: float, now: float | None = None) -> bool:
        """A silent tool/evaluation is governed by its own/original deadline."""
        with self._lock:
            return (not self.active_tools
                    and (time.monotonic() if now is None else now) - self.last_activity >= timeout_seconds)

    def feed(self, chunk: bytes) -> None:
        for piece in chunk.splitlines(keepends=True):
            finished = piece.endswith(b"\n")
            if not self.discarding:
                if len(self.pending) + len(piece) > _EVENT_MAX_BYTES:
                    self.envelope = _ToolEventEnvelope()
                    self.envelope.feed(self.pending)
                    self.pending.clear()
                    self.discarding = True
                else:
                    self.pending.extend(piece)
            if self.discarding and self.envelope is not None:
                self.envelope.feed(piece)
            if finished:
                if not self.discarding:
                    self._event(bytes(self.pending))
                elif self.envelope is not None:
                    event = self.envelope.event()
                    if event is not None:
                        self._event(json.dumps(event).encode())
                self.pending.clear()
                self.discarding = False
                self.envelope = None

    def finish(self) -> None:
        if self.pending and not self.discarding:
            self._event(bytes(self.pending))
        elif self.envelope is not None:
            event = self.envelope.event()
            if event is not None:
                self._event(json.dumps(event).encode())
        if self.session_identity_mismatch:
            self.failure = "native resume session identity mismatch"
            self.retryable = False
            self.retry_blocked = True

    def _event(self, line: bytes) -> None:
        with self._lock:
            self._inspect_event(line)

    def _inspect_event(self, line: bytes) -> None:
        try:
            event = json.loads(line)
        except (ValueError, UnicodeDecodeError, RecursionError):
            return
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        if not isinstance(kind, str):
            return
        self.last_activity = time.monotonic()
        item = event.get("item")
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            identifier = item["id"]
            if kind == "item.completed":
                self.active_tools.discard(identifier)
            elif kind in {"item.started", "item.updated"} and item.get("type") in {
                "command_execution", "mcp_tool_call", "web_search", "collab_tool_call",
                "tool_call", "file_change",
            }:
                if item.get("status") in {"completed", "failed", "cancelled"}:
                    self.active_tools.discard(identifier)
                else:
                    self.active_tools.add(identifier)
        if kind == "thread.started":
            value = event.get("thread_id")
            try:
                thread_id = str(UUID(value)) if isinstance(value, str) else None
            except ValueError:
                thread_id = None
            if self.thread_id is not None and thread_id != self.thread_id:
                self.session_identity_mismatch = True
                self.retry_blocked = True
                self.retryable = False
            elif thread_id is not None:
                self.thread_id = thread_id
        elif kind == "turn.completed":
            # A later successful native turn proves that earlier errors recovered.
            self.failure = None
            self.retryable = False
            self.active_tools.clear()
        elif kind in {"error", "turn.failed"}:
            detail = event.get("error", event.get("message"))
            # Both top-level and nested codes participate, so an auth rejection
            # cannot be hidden behind an otherwise retryable capacity message.
            pieces = [event.get("code"), event.get("message")]
            if isinstance(detail, dict):
                pieces.extend(detail.get(key) for key in ("code", "type", "message"))
            else:
                pieces.append(detail)
            cause = _safe_failure(" ".join(value for value in pieces if isinstance(value, str)))
            transient = cause in _TRANSIENT_FAILURES
            if ((cause is not None and not transient)
                    or (cause is None and (kind == "error" or self.failure is None))):
                # A policy/auth/invalid-request rejection is sticky for this
                # attempt. A later transport error must never make it retryable.
                self.retry_blocked = True
            # The terminal cause takes precedence over earlier reconnect or
            # service errors. A cause-less turn.failed may refer to the last
            # native error; keep that label, explicitly identified as preceding.
            if kind == "turn.failed" and cause is None and self.failure is not None:
                if self.failure.startswith("native error:"):
                    self.failure = f"native turn.failed; preceding {self.failure}"
                return
            self.failure = f"native {kind}: {cause or 'unclassified error (see execution logs)'}"
            self.retryable = transient and not self.retry_blocked


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _terminate_process_group(
    process: subprocess.Popen[bytes],
    *,
    grace_seconds: float = 2.0,
) -> bool:
    """Stop the previous native attempt before another can receive credentials."""

    process_group_id = process.pid
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return True
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return True
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return True
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass

    deadline = time.monotonic() + min(grace_seconds, 0.5)
    while time.monotonic() < deadline:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.01)
    return False


def _enable_subreaper() -> None:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("same-session continuation requires Linux descendant cleanup")
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("same-session continuation requires pidfd process identity support")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "could not enable native descendant cleanup")


def _direct_children() -> set[int]:
    children = set()
    for task in Path(f"/proc/{os.getpid()}/task").iterdir():
        try:
            children.update(int(value) for value in (task / "children").read_text().split())
        except FileNotFoundError:
            pass  # A supervisor thread can exit during the snapshot.
    return children


def _cleanup_adopted_children(process: subprocess.Popen[bytes]) -> bool:
    """Reap this supervisor's descendants, including tools that called setsid."""
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        children = _direct_children()
        if not children:
            return True
        for pid in children:
            try:
                # Pin the identity, then recheck ownership: the numeric PID
                # could have been reused since the children snapshot.
                descriptor = os.pidfd_open(pid)
            except ProcessLookupError:
                continue
            try:
                fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
                if len(fields) < 2 or int(fields[1]) != os.getpid():
                    continue
                signal.pidfd_send_signal(descriptor, signal.SIGKILL)
            except (ProcessLookupError, FileNotFoundError):
                pass
            finally:
                os.close(descriptor)
            if pid == process.pid:
                process.poll()  # Popen owns the native root's wait status.
            else:
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass
        time.sleep(0.01)
    return not _direct_children()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Codex after removing its on-disk auth before model tools can execute."
    )
    parser.add_argument("--auth", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--audit-copy", type=Path)
    parser.add_argument("--shutdown-receipt", type=Path, help="Managed allocation cleanup receipt (no credentials).")
    parser.add_argument("--result", type=Path, help="Private, credential-free exit summary.")
    parser.add_argument("--deadline-monotonic", type=float)
    parser.add_argument("--expected-thread-id")
    parser.add_argument("--response-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--require-descendant-cleanup", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("a Codex command is required after --")

    auth_path = args.auth.resolve()
    audit_path = args.audit.resolve()
    audit_copy_path = (
        None if args.audit_copy is None else args.audit_copy.resolve()
    )
    auth_stat = auth_path.stat()
    if not stat.S_ISREG(auth_stat.st_mode):
        raise SystemExit(f"auth path must be a regular file: {auth_path}")
    if audit_path.parent != auth_path.parent:
        raise SystemExit("--audit must be in the same control directory as --auth")
    if audit_copy_path is not None:
        if (
            audit_copy_path in {auth_path, audit_path}
            or not audit_copy_path.parent.is_dir()
            or audit_copy_path.parent.is_symlink()
            or (
                audit_copy_path.exists()
                and (
                    not audit_copy_path.is_file()
                    or audit_copy_path.is_symlink()
                )
            )
        ):
            raise SystemExit("--audit-copy must be a safe regular destination")
    result_path = args.result
    if result_path is not None:
        if result_path.exists() or result_path.is_symlink():
            raise SystemExit("--result must be a new file")
        if result_path.parent.resolve() != auth_path.parent.parent:
            raise SystemExit("--result must be in the private runtime directory")
    deadline = args.deadline_monotonic
    if deadline is not None and (not isfinite(deadline) or deadline <= time.monotonic()):
        auth_path.unlink(missing_ok=True)
        raise SystemExit("the original Agent deadline has expired or is invalid")
    response_timeout = args.response_timeout_seconds
    if not isfinite(response_timeout) or response_timeout <= 0:
        auth_path.unlink(missing_ok=True)
        raise SystemExit("native response timeout must be finite and positive")
    expected_thread_id = args.expected_thread_id
    if expected_thread_id is not None:
        expected_thread_id = str(UUID(expected_thread_id))
    if args.require_descendant_cleanup:
        try:
            _enable_subreaper()
        except BaseException:
            auth_path.unlink(missing_ok=True)
            raise
    started_ns = time.time_ns()
    process = subprocess.Popen(
        command,
        stdin=sys.stdin.buffer,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    shutdown_path = args.shutdown_receipt
    def write_shutdown(cleaned):
        if shutdown_path is not None:
            _write_json_atomic(shutdown_path, {
                "execution_id": os.environ.get("PDE_AGENT_EXECUTION_ID"),
                "request_path": os.environ.get("PDE_AGENT_REQUEST"),
                "native_pid": process.pid, "process_group_cleaned": cleaned,
            })
    write_shutdown(False)
    termination_lock = threading.RLock()
    def terminate_native() -> bool:
        with termination_lock:
            group_cleaned = _terminate_process_group(process)
            if args.require_descendant_cleanup:
                descendants_cleaned = _cleanup_adopted_children(process)
                group_cleaned = descendants_cleaned and _terminate_process_group(process)
            write_shutdown(group_cleaned)
            return group_cleaned

    def terminate_from_signal(signum: int, _frame: object) -> None:
        for handled_signal in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            signal.signal(handled_signal, signal.SIG_IGN)
        auth_path.unlink(missing_ok=True)
        terminate_native()
        raise SystemExit(128 + signum)

    for handled_signal in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(handled_signal, terminate_from_signal)

    events = _NativeEvents()
    stderr_prefix = bytearray()

    def forward_stderr() -> None:
        assert process.stderr is not None
        while chunk := process.stderr.read1(_EVENT_MAX_BYTES):
            stderr_prefix.extend(chunk[:max(0, _EVENT_MAX_BYTES - len(stderr_prefix))])
            sys.stderr.buffer.write(chunk)
            sys.stderr.buffer.flush()

    cleanup = {"confirmed": False, "deadline_expired": False, "response_stalled": False}

    def watch_native_exit() -> None:
        # Also reap an orphaned tool group when the native CLI exits while a
        # descendant still holds stdout/stderr open. Reading the pipe alone
        # would otherwise wait until the entire Agent deadline.
        try:
            while process.poll() is None:
                now = time.monotonic()
                if deadline is not None and now >= deadline:
                    cleanup["deadline_expired"] = True
                    break
                if events.response_stalled(response_timeout, now):
                    cleanup["response_stalled"] = True
                    break
                try:
                    process.wait(timeout=min(0.25, response_timeout))
                except subprocess.TimeoutExpired:
                    pass
        finally:
            cleanup["confirmed"] = terminate_native()

    stderr_thread = threading.Thread(target=forward_stderr, daemon=True)
    exit_thread = threading.Thread(target=watch_native_exit, daemon=True)
    try:
        stderr_thread.start()
        exit_thread.start()
    except BaseException:
        auth_path.unlink(missing_ok=True)
        terminate_native()
        raise

    def finish(returncode: int, *, before_first_event: bool = False) -> int:
        exit_thread.join(timeout=5.0)
        stderr_thread.join(timeout=2.0)
        events.finish()
        failure = events.failure
        if cleanup["response_stalled"] and not events.retry_blocked:
            failure = "native response stalled"
            events.retryable = True
        if cleanup["deadline_expired"]:
            failure = "original Agent deadline reached"
        if expected_thread_id is not None and events.thread_id != expected_thread_id:
            failure = "native resume session identity mismatch"
            events.retry_blocked = True
        if before_first_event:
            cause = _safe_failure(stderr_prefix.decode("utf-8", errors="replace"))
            failure = "Codex exited before its first event"
            if cleanup["response_stalled"]:
                failure += ": native response stalled without a confirmed session"
            if cause:
                failure += f": {cause}"
        if result_path is not None:
            _write_json_atomic(result_path, {
                "failure": failure,
                "thread_id": events.thread_id,
                "retryable": bool(events.retryable and not events.retry_blocked
                                  and not cleanup["deadline_expired"] and not before_first_event),
                "process_group_cleaned": cleanup["confirmed"],
            })
        return returncode

    assert process.stdout is not None
    first_event = process.stdout.readline(_EVENT_MAX_BYTES)
    if not first_event:
        returncode = process.wait()
        auth_path.unlink(missing_ok=True)
        return finish(returncode if returncode != 0 else 97, before_first_event=True)

    stopped = False
    try:
        os.killpg(process.pid, signal.SIGSTOP)
        stopped = True
    except ProcessLookupError:
        pass

    auth_path.unlink()
    payload = {
        "auth_removed_before_first_event_forwarded": not auth_path.exists(),
        "control_directory_made_read_only": True,
        "first_event_observed": True,
        "first_event_removed_auth_at_unix_ns": time.time_ns(),
        "started_at_unix_ns": started_ns,
    }
    _write_json_atomic(audit_path, payload)
    if audit_copy_path is not None:
        _write_json_atomic(audit_copy_path, payload)
    os.chmod(auth_path.parent, 0o500)

    events.feed(first_event)
    if expected_thread_id is not None and events.thread_id != expected_thread_id:
        terminate_native()
        return finish(process.wait())

    if stopped:
        try:
            os.killpg(process.pid, signal.SIGCONT)
        except ProcessLookupError:
            pass

    sys.stdout.buffer.write(first_event)
    sys.stdout.buffer.flush()
    while chunk := process.stdout.read1(_EVENT_MAX_BYTES):
        events.feed(chunk)
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
    return finish(process.wait())


if __name__ == "__main__":
    raise SystemExit(main())
