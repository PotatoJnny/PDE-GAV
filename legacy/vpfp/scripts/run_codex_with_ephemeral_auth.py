#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path


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
) -> None:
    """Terminate the child-owned group, escalating after a short grace."""

    process_group_id = process.pid
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Codex after removing its on-disk auth before model tools can execute."
    )
    parser.add_argument("--auth", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--audit-copy", type=Path)
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
    started_ns = time.time_ns()
    process = subprocess.Popen(
        command,
        stdin=sys.stdin.buffer,
        stdout=subprocess.PIPE,
        stderr=None,
        start_new_session=True,
    )

    def terminate_from_signal(signum: int, _frame: object) -> None:
        for handled_signal in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            signal.signal(handled_signal, signal.SIG_IGN)
        auth_path.unlink(missing_ok=True)
        _terminate_process_group(process)
        raise SystemExit(128 + signum)

    for handled_signal in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(handled_signal, terminate_from_signal)

    assert process.stdout is not None
    first_event = process.stdout.readline()
    if not first_event:
        returncode = process.wait()
        auth_path.unlink(missing_ok=True)
        return returncode if returncode != 0 else 97

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

    if stopped:
        try:
            os.killpg(process.pid, signal.SIGCONT)
        except ProcessLookupError:
            pass

    sys.stdout.buffer.write(first_event)
    sys.stdout.buffer.flush()
    while chunk := process.stdout.read1(64 * 1024):
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
    return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
