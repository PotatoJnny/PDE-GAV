#!/usr/bin/env python3
"""Check the real canonical shell policy and loopback without invoking a model.

This is a launch prerequisite, not a scientific rehearsal. It never reads
authentication, changes host policy, installs packages, or disables a sandbox.
Every diagnostic directory and exact failure is retained for inspection.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from threading import Thread
import time
import tomllib


def check_environment(codex_binary: Path, config: Path, output: Path) -> dict:
    codex_binary, config, output = (path.expanduser().resolve() for path in (codex_binary, config, output))
    if not codex_binary.is_file() or not os.access(codex_binary, os.X_OK):
        raise ValueError("Codex executable is unavailable")
    settings = tomllib.loads(config.read_text())
    if (settings.get("sandbox_mode") != "workspace-write"
            or settings.get("approval_policy") != "never"
            or settings.get("sandbox_workspace_write", {}).get("network_access") is not True):
        raise ValueError("preflight requires the unchanged canonical workspace-write/loopback policy")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    workspace = output / "workspace"
    control = output / "codex-control"
    agent_home = output / "agent-home"
    for path in (workspace, control, agent_home, workspace / "tmp"):
        path.mkdir(mode=0o700)
    shutil.copyfile(config, output / "canonical-source.config.toml")
    # Codex 0.149's standalone sandbox command requires a named permission
    # profile. Map the checked legacy worker policy to its documented built-in
    # equivalent; do not change the actual Agent worker configuration.
    (control / "config.toml").write_text(
        'approval_policy = "never"\n'
        'default_permissions = "canonical-preflight"\n'
        '[permissions.canonical-preflight]\n'
        'extends = ":workspace"\n'
        '[permissions.canonical-preflight.network]\n'
        'enabled = true\n'
    )
    report = {
        "schema": "canonical_codex_environment_preflight_v1",
        "hostname": platform.node(), "started_at_unix": time.time(),
        "codex_binary": str(codex_binary), "codex_config_sha256": sha256(config.read_bytes()).hexdigest(),
        "policy": "workspace-write", "network_access": True,
        "approval_policy": "never", "model_invoked": False,
        "authentication_read": False, "host_policy_modified": False,
        "namespace_limits": {}, "status": "failed",
    }
    for name in ("max_net_namespaces", "max_user_namespaces"):
        path = Path("/proc/sys/user") / name
        report["namespace_limits"][name] = path.read_text().strip() if path.is_file() else None
    environment = {key: value for key, value in os.environ.items() if key in (
        "PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "LIBRARY_PATH", "VIRTUAL_ENV",
    )}
    environment.update({
        "CODEX_HOME": str(control), "HOME": str(agent_home),
        "TMPDIR": str(workspace / "tmp"), "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    server = None
    thread = None
    thread_started = False
    try:
        version = subprocess.run([str(codex_binary), "--version"], env=environment,
                                 capture_output=True, text=True, timeout=15, check=True)
        report["codex_version"] = version.stdout.strip()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"canonical-loopback-ok")

            def log_message(self, *_args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        thread.start()
        thread_started = True
        outside = output / "outside-workspace-canary"
        program = "\n".join((
            "from pathlib import Path",
            "from urllib.request import urlopen",
            "import errno, json",
            "Path('inside-workspace-canary').write_text('probe-only')",
            "outside_denied = False",
            "try:",
            f"    Path({str(outside)!r}).write_text('probe-only')",
            "except OSError as error:",
            "    outside_denied = error.errno in (errno.EACCES, errno.EPERM, errno.EROFS)",
            "    if not outside_denied: raise",
            f"with urlopen('http://127.0.0.1:{server.server_port}/', timeout=5) as response:",
            "    loopback = response.read() == b'canonical-loopback-ok'",
            "print(json.dumps({'inside_write': True, 'outside_write_denied': outside_denied, 'loopback': loopback}))",
            "raise SystemExit(0 if outside_denied and loopback else 3)",
        ))
        (workspace / "probe.py").write_text(program + "\n")
        command = [str(codex_binary), "sandbox", "--permission-profile", "canonical-preflight",
                   "-C", str(workspace), "--", sys.executable, str(workspace / "probe.py")]
        report["command"] = command
        completed = subprocess.run(command, cwd=workspace, env=environment,
                                   capture_output=True, text=True, timeout=30, check=False)
        (output / "sandbox.stdout.log").write_text(completed.stdout)
        (output / "sandbox.stderr.log").write_text(completed.stderr)
        report["returncode"] = completed.returncode
        if completed.returncode:
            raise RuntimeError(f"canonical sandbox prerequisite failed (exit {completed.returncode}); see sandbox.stderr.log")
        checks = json.loads(completed.stdout.strip().splitlines()[-1])
        if checks != {"inside_write": True, "outside_write_denied": True, "loopback": True}:
            raise RuntimeError("canonical sandbox did not establish all required boundaries")
        report.update(status="passed", checks=checks)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, subprocess.TimeoutExpired):
            for name, data in (("stdout", error.stdout), ("stderr", error.stderr)):
                (output / f"sandbox.{name}.log").write_text(
                    data.decode(errors="replace") if isinstance(data, bytes) else data or "")
    finally:
        if server is not None:
            if thread_started:
                server.shutdown()
            server.server_close()
        if thread_started:
            thread.join(timeout=2)
        report["finished_at_unix"] = time.time()
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex-binary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = check_environment(args.codex_binary, args.config, args.output)
    print(json.dumps({"status": report["status"], "hostname": report["hostname"],
                      "report": str(args.output.resolve() / "report.json"), "error": report.get("error")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
