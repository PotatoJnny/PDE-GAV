from __future__ import annotations

import os
import signal
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from agent_verifier.codex_exec.results import CodexInvocationRecord


DEFAULT_CODEX_CMD_TEMPLATE = "codex exec {model_args} --skip-git-repo-check --cd {workspace} {prompt}"


@dataclass
class CodexCliClient:
    command_template: str | None = None
    timeout_seconds: int | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    allow_network: bool | None = None
    read_isolated: bool | None = None
    mock: bool = False
    use_environment_overrides: bool = True

    def __post_init__(self) -> None:
        command_from_environment = os.environ.get("AVF_CODEX_CMD_TEMPLATE") if self.use_environment_overrides else None
        timeout_from_environment = os.environ.get("AVF_CODEX_TIMEOUT_SECONDS") if self.use_environment_overrides else None
        model_from_environment = os.environ.get("AVF_CODEX_MODEL") if self.use_environment_overrides else None
        reasoning_from_environment = (
            os.environ.get("AVF_CODEX_REASONING_EFFORT") if self.use_environment_overrides else None
        )
        network_from_environment = os.environ.get("AVF_CODEX_ALLOW_NETWORK") if self.use_environment_overrides else None
        isolation_from_environment = (
            os.environ.get("AVF_CODEX_READ_ISOLATED")
            if self.use_environment_overrides
            else None
        )
        self.command_template = command_from_environment or self.command_template or DEFAULT_CODEX_CMD_TEMPLATE
        self.timeout_seconds = int(
            timeout_from_environment
            or self.timeout_seconds
            or 600
        )
        self.model = model_from_environment or self.model or ""
        self.reasoning_effort = reasoning_from_environment or self.reasoning_effort or ""
        if network_from_environment is not None:
            self.allow_network = network_from_environment.lower() in {"1", "true", "yes"}
        elif self.allow_network is None:
            self.allow_network = False
        if isolation_from_environment is not None:
            self.read_isolated = isolation_from_environment.lower() in {
                "1",
                "true",
                "yes",
            }
        elif self.read_isolated is None:
            self.read_isolated = False

    def invoke(
        self,
        *,
        round_index: int,
        workspace: str | Path,
        prompt_text: str,
        prompt_path: str | Path,
        logs_dir: str | Path,
        expected_submission: str | Path,
    ) -> CodexInvocationRecord:
        workspace = _absolute_no_resolve(workspace)
        prompt_path = _absolute_no_resolve(prompt_path)
        logs_dir = _absolute_no_resolve(logs_dir)
        expected_submission = _absolute_no_resolve(expected_submission)
        logs_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = logs_dir / f"codex_round_{round_index:03d}.stdout.log"
        stderr_path = logs_dir / f"codex_round_{round_index:03d}.stderr.log"

        codex_path = shutil.which("codex")
        codex_version = _codex_version(codex_path)
        start = time.perf_counter()

        if self.mock or self.command_template == "mock":
            _write_mock_submission(expected_submission, round_index)
            stdout_path.write_text("mock Codex client wrote submission.py\n")
            stderr_path.write_text("")
            return CodexInvocationRecord(
                round_index=round_index,
                actual_codex_invoked=False,
                codex_binary_path=codex_path,
                codex_version=codex_version,
                command_template=str(self.command_template),
                returncode=0,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                wall_time_seconds=time.perf_counter() - start,
                prompt_path=str(prompt_path),
                workspace_path=str(workspace),
                generated_submission_path=str(expected_submission),
                unavailable_reason="mock client used; actual Codex was not invoked",
                submission_exists=expected_submission.exists(),
                model=self.model or None,
                reasoning_effort=self.reasoning_effort or None,
                environment_overrides_used=bool(self.use_environment_overrides),
                network_allowed=bool(self.allow_network),
            )

        if codex_path is None:
            stdout_path.write_text("")
            stderr_path.write_text("codex binary not found on PATH\n")
            return CodexInvocationRecord(
                round_index=round_index,
                actual_codex_invoked=False,
                codex_binary_path=None,
                codex_version=None,
                command_template=str(self.command_template),
                returncode=None,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                wall_time_seconds=time.perf_counter() - start,
                prompt_path=str(prompt_path),
                workspace_path=str(workspace),
                generated_submission_path=None,
                unavailable_reason="codex binary not found on PATH",
                model=self.model or None,
                reasoning_effort=self.reasoning_effort or None,
                environment_overrides_used=bool(self.use_environment_overrides),
                network_allowed=bool(self.allow_network),
            )

        command_workspace = Path("/tmp/workspace") if self.read_isolated else workspace
        command = self._format_command(
            command_workspace,
            prompt_text,
            command_workspace / prompt_path.name,
            command_workspace / expected_submission.name,
        )
        env = os.environ.copy()
        env["AVF_CODEX_ALLOW_NETWORK"] = "1" if self.allow_network else "0"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        isolated_home = None
        try:
            if self.read_isolated:
                isolated_home = tempfile.TemporaryDirectory(
                    prefix="avf_codex_home_"
                )
                isolated_home_path = Path(isolated_home.name)
                _seed_isolated_codex_home(isolated_home_path)
                env = _read_isolated_environment(env)
                broker_socket_value = env.get("AVF_TRIPLANE_EVAL_SOCKET")
                if not broker_socket_value:
                    raise RuntimeError(
                        "read-isolated Codex requires the evaluator broker socket"
                    )
                broker_socket = Path(broker_socket_value)
                if not broker_socket.is_socket():
                    raise RuntimeError(
                        "read-isolated Codex evaluator broker socket is unavailable"
                    )
                env["AVF_TRIPLANE_EVAL_SOCKET"] = "/tmp/evaluator.sock"
                command = _wrap_read_isolated_command(
                    command,
                    workspace=workspace,
                    isolated_codex_home=isolated_home_path,
                    broker_socket=broker_socket,
                    environment=env,
                )
            else:
                repo_src = _absolute_no_resolve(Path.cwd() / "src")
                env["PYTHONPATH"] = (
                    f"{repo_src}:{env.get('PYTHONPATH', '')}"
                )
            with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                proc = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=str(workspace),
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    start_new_session=True,
                )
                try:
                    returncode = proc.wait(timeout=self.timeout_seconds)
                except subprocess.TimeoutExpired:
                    _terminate_process_group(proc)
                    raise
            return CodexInvocationRecord(
                round_index=round_index,
                actual_codex_invoked=True,
                codex_binary_path=codex_path,
                codex_version=codex_version,
                command_template=str(self.command_template),
                returncode=returncode,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                wall_time_seconds=time.perf_counter() - start,
                prompt_path=str(prompt_path),
                workspace_path=str(workspace),
                generated_submission_path=str(expected_submission) if expected_submission.exists() else None,
                unavailable_reason=None if expected_submission.exists() else "Codex returned without submission.py",
                submission_exists=expected_submission.exists(),
                model=self.model or None,
                reasoning_effort=self.reasoning_effort or None,
                environment_overrides_used=bool(self.use_environment_overrides),
                network_allowed=bool(self.allow_network),
            )
        except subprocess.TimeoutExpired:
            exists = expected_submission.exists()
            reason = f"Codex timed out after {self.timeout_seconds}s"
            if exists:
                reason += " after writing submission.py; generated file was kept for evaluation"
            return CodexInvocationRecord(
                round_index=round_index,
                actual_codex_invoked=True,
                codex_binary_path=codex_path,
                codex_version=codex_version,
                command_template=str(self.command_template),
                returncode=None,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                wall_time_seconds=time.perf_counter() - start,
                prompt_path=str(prompt_path),
                workspace_path=str(workspace),
                generated_submission_path=str(expected_submission) if exists else None,
                unavailable_reason=reason,
                timed_out=True,
                submission_exists=exists,
                model=self.model or None,
                reasoning_effort=self.reasoning_effort or None,
                environment_overrides_used=bool(self.use_environment_overrides),
                network_allowed=bool(self.allow_network),
            )
        finally:
            if isolated_home is not None:
                isolated_home.cleanup()

    def _format_command(
        self,
        workspace: Path,
        prompt_text: str,
        prompt_path: Path,
        expected_submission: Path,
    ) -> str:
        template = str(self.command_template)
        model_args = []
        if self.model:
            model_args.extend(("-m", shlex.quote(self.model)))
        if self.reasoning_effort:
            reasoning_config = shlex.quote(f'model_reasoning_effort="{self.reasoning_effort}"')
            model_args.extend(("-c", reasoning_config))
        network_config = shlex.quote(
            f"sandbox_workspace_write.network_access={'true' if self.allow_network else 'false'}"
        )
        model_args.extend(("-c", network_config))
        if self.read_isolated:
            model_args.extend(
                (
                    "--ignore-user-config",
                    "--ephemeral",
                    "--dangerously-bypass-approvals-and-sandbox",
                )
            )
        values = {
            "workspace": shlex.quote(str(workspace)),
            "prompt": shlex.quote(prompt_text),
            "prompt_file": shlex.quote(str(prompt_path)),
            "prompt_path": shlex.quote(str(prompt_path)),
            "submission": shlex.quote(str(expected_submission)),
            "output_file": shlex.quote(str(workspace / "codex_last_message.txt")),
            "model_args": " ".join(model_args),
            "model": shlex.quote(self.model) if self.model else "",
            "reasoning_effort": shlex.quote(self.reasoning_effort) if self.reasoning_effort else "",
        }
        return template.format(**values)


def _seed_isolated_codex_home(destination: Path) -> None:
    source_home = Path(
        os.environ.get("CODEX_HOME") or (Path.home() / ".codex")
    )
    source_auth = source_home / "auth.json"
    if not source_auth.is_file():
        raise RuntimeError(
            f"read-isolated Codex authentication is unavailable: {source_auth}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    target_auth = destination / "auth.json"
    shutil.copy2(source_auth, target_auth)
    target_auth.chmod(0o600)


def _read_isolated_environment(environment: dict[str, str]) -> dict[str, str]:
    isolated = dict(environment)
    broker_names = {
        "AVF_TRIPLANE_EVAL_SOCKET",
        "AVF_TRIPLANE_EVAL_TOKEN",
    }
    private_names = {
        "AUTOENCODER_CHECKPOINT",
        "D0_CASES_ROOT",
        "D0_RUN_ROOT",
        "DIFFUSION_CHECKPOINT",
        "FINAL_GENERATED_ROOT",
        "FINAL_HOLDOUT_ROOT",
        "FINAL_POLICY_CHECKPOINT",
        "INITIAL_PREVIOUS_SUBMISSION",
        "POLICY_CHECKPOINT",
        "POLICY_OUT",
        "PREVIOUS_SUBMISSION",
        "REWARD_TABLE",
        "RUN_ROOT",
        "SPLIT_MANIFEST",
        "SSH_AUTH_SOCK",
    }
    for key in list(isolated):
        if (
            key.startswith("SLURM_")
            or key.startswith("SSH_")
            or key.startswith("XDG_")
            or (key.startswith("AVF_") and key not in broker_names)
            or key in private_names
        ):
            isolated.pop(key, None)

    home = Path.home()
    node_bin = home / ".nvm" / "versions" / "node" / "v22.22.2" / "bin"
    for key in (
        "PATH",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CPATH",
    ):
        value = _filter_isolated_path_list(isolated.get(key, ""), home=home)
        if key == "PATH":
            value = ":".join(
                item
                for item in (str(node_bin), value, "/usr/local/bin", "/usr/bin", "/bin")
                if item
            )
        if value:
            isolated[key] = value
        else:
            isolated.pop(key, None)
    isolated.update(
        {
            "HOME": str(home),
            "CODEX_HOME": str(home / ".codex"),
            "TMPDIR": "/tmp",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return isolated


def _filter_isolated_path_list(value: str, *, home: Path) -> str:
    blocked = ("/project", "/scratch", str(home))
    return ":".join(
        item
        for item in value.split(":")
        if item and not item.startswith(blocked)
    )


def _wrap_read_isolated_command(
    command: str,
    *,
    workspace: Path,
    isolated_codex_home: Path,
    broker_socket: Path,
    environment: dict[str, str],
) -> str:
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise RuntimeError("read-isolated Codex requires bubblewrap")
    home = Path.home()
    nvm_home = home / ".nvm"
    if not nvm_home.is_dir():
        raise RuntimeError(f"read-isolated Codex runtime is unavailable: {nvm_home}")

    args = [
        bwrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--ro-bind",
        "/",
        "/",
    ]
    for path in (Path("/project"), Path("/scratch"), Path("/gpfs")):
        if path.exists():
            args.extend(("--tmpfs", str(path)))
    args.extend(
        (
            "--tmpfs",
            "/home",
            "--dir",
            str(home),
            "--dir",
            str(nvm_home),
            "--ro-bind",
            str(nvm_home),
            str(nvm_home),
            "--dir",
            str(home / ".codex"),
            "--bind",
            str(isolated_codex_home),
            str(home / ".codex"),
            "--tmpfs",
            "/run",
        )
    )
    resolv_conf = Path("/run/systemd/resolve/resolv.conf")
    if resolv_conf.is_file():
        args.extend(
            (
                "--dir",
                "/run/systemd",
                "--dir",
                "/run/systemd/resolve",
                "--ro-bind",
                str(resolv_conf),
                str(resolv_conf),
            )
        )
    args.extend(
        (
            "--tmpfs",
            "/tmp",
            "--dev-bind",
            "/dev",
            "/dev",
            "--proc",
            "/proc",
            "--dir",
            "/tmp/workspace",
            "--bind",
            str(workspace),
            "/tmp/workspace",
            "--bind",
            str(broker_socket),
            "/tmp/evaluator.sock",
            "--chdir",
            "/tmp/workspace",
            "--setenv",
            "HOME",
            str(home),
            "--setenv",
            "CODEX_HOME",
            str(home / ".codex"),
            "--setenv",
            "TMPDIR",
            "/tmp",
            "--setenv",
            "PATH",
            environment["PATH"],
            "/bin/bash",
            "-lc",
            command,
        )
    )
    return shlex.join(args)


def _terminate_process_group(proc: subprocess.Popen[str], *, grace_seconds: float = 2.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait()


def _codex_version(codex_path: str | None) -> str | None:
    if codex_path is None:
        return None
    try:
        return subprocess.check_output([codex_path, "--version"], text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc!r}"


def _write_mock_submission(path: Path, round_index: int) -> None:
    path.write_text(
        "from agent_verifier.agent.baselines import CoarseSpectralNSAgent, LinearStokesAgent\n\n"
        "def solve(case):\n"
        "    if %d == 0:\n"
        "        return LinearStokesAgent(substeps=2).solve(case)\n"
        "    return CoarseSpectralNSAgent(substeps=2, coarse_shape=8).solve(case)\n" % round_index
    )


def _absolute_no_resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path.cwd() / p
