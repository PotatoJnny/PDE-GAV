from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import tomllib
import unittest

from pde_agent_verifier_solver.engine.reference_agent_runtime import (
    ReferenceAgentLaunchError,
    ReferenceAgentLaunchRequest,
    SubprocessReferenceAgentLauncher,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY_ROOT / "scripts" / "run_reference_macro_codex_agent.sh"
CONFIG = REPOSITORY_ROOT / "scripts" / "reference_macro_codex.config.toml"
AUTH_HELPER = REPOSITORY_ROOT / "scripts" / "run_codex_with_ephemeral_auth.py"
LEGACY_AUTH_HELPER = (
    REPOSITORY_ROOT
    / "legacy"
    / "vpfp"
    / "scripts"
    / "run_codex_with_ephemeral_auth.py"
)


class ReferenceMacroCodexLauncherTests(unittest.TestCase):
    def test_policy_is_strict_reference_only_and_generic(self) -> None:
        policy = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(policy["approval_policy"], "never")
        self.assertEqual(policy["sandbox_mode"], "workspace-write")
        self.assertEqual(policy["web_search"], "live")
        self.assertFalse(policy["allow_login_shell"])
        self.assertFalse(policy["features"]["guardian_approval"])
        self.assertTrue(policy["sandbox_workspace_write"]["network_access"])

        included = set(policy["shell_environment_policy"]["include_only"])
        self.assertIn("PYTHONPATH", included)
        self.assertIn("CUDA_VISIBLE_DEVICES", included)
        self.assertIn("CUDA_DEVICE_ORDER", included)
        self.assertTrue(
            {
                "PDE_MACRO_WORKSPACE",
                "PDE_MACRO_CONTEXT",
                "PDE_MACRO_PROMPT",
                "PDE_MACRO_SUBMISSION",
            }.issubset(included)
        )
        self.assertNotIn("CODEX_HOME", included)

        self.assertEqual(AUTH_HELPER.read_bytes(), LEGACY_AUTH_HELPER.read_bytes())
        generic_text = (
            LAUNCHER.read_text(encoding="utf-8")
            + CONFIG.read_text(encoding="utf-8")
            + AUTH_HELPER.read_text(encoding="utf-8")
        ).lower()
        self.assertNotIn("vpfp", generic_text)

    def test_fake_codex_exercises_auth_lifecycle_and_public_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace, context, prompt, submission = _workspace(root)
            slurm_tmp = root / "slurm"
            slurm_tmp.mkdir()
            auth_source = root / "controller-auth.json"
            auth_secret = b'{"token":"controller-only-secret"}\n'
            auth_source.write_bytes(auth_secret)
            fake_codex = _fake_codex(root)

            environment = _base_environment(
                workspace=workspace,
                context=context,
                prompt=prompt,
                submission=submission,
                slurm_tmp=slurm_tmp,
            )
            environment["PYTHONPATH"] = "/explicit/public-sdk"
            environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            environment["SHOULD_NOT_LEAK"] = "controller-private-marker"
            completed = subprocess.run(
                _launcher_command(fake_codex, auth_source),
                cwd=workspace,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn('"type":"thread.started"', completed.stdout)
            self.assertTrue(submission.is_file())
            self.assertEqual(auth_source.read_bytes(), auth_secret)
            self.assertEqual(tuple(slurm_tmp.iterdir()), ())

            capture = json.loads(
                (workspace / "fake-codex-capture.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(capture["workspace"], str(workspace.resolve()))
            self.assertEqual(capture["context"], str(context.resolve()))
            self.assertEqual(capture["prompt"], str(prompt.resolve()))
            self.assertEqual(capture["submission"], str(submission.resolve()))
            self.assertEqual(capture["prompt_stdin"], "Improve the macro ROM.\n")
            self.assertTrue(
                capture["pythonpath"].startswith(
                    str(REPOSITORY_ROOT / "src") + os.pathsep
                )
            )
            self.assertTrue(capture["pythonpath"].endswith("/explicit/public-sdk"))
            self.assertIn("127.0.0.1", capture["no_proxy"])
            self.assertEqual(capture["cuda_visible_devices"], "7")
            self.assertEqual(capture["cuda_device_order"], "PCI_BUS_ID")
            self.assertFalse(capture["unrelated_environment_present"])
            self.assertFalse(Path(capture["codex_home"]).exists())

            arguments = capture["arguments"]
            for required in (
                "--ephemeral",
                "--strict-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--disable",
                "apps",
                "plugins",
                "browser_use",
                "browser_use_external",
                "browser_use_full_cdp_access",
                "computer_use",
                "image_generation",
                "in_app_browser",
            ):
                self.assertIn(required, arguments)
            self.assertEqual(
                arguments[arguments.index("--model") + 1],
                "gpt-test",
            )
            self.assertIn('model_reasoning_effort="high"', arguments)

            audit = json.loads(
                (workspace / "codex-auth-lifecycle.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(audit["auth_removed_before_first_event_forwarded"])
            self.assertTrue((workspace / "codex-last-message.txt").is_file())
            for path in workspace.rglob("*"):
                if path.is_file():
                    self.assertNotIn(auth_secret, path.read_bytes())

    def test_hard_timeout_reaps_sessioned_codex_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace, context, prompt, submission = _workspace(root)
            slurm_tmp = root / "slurm"
            slurm_tmp.mkdir()
            auth_source = root / "controller-auth.json"
            auth_source.write_text("{}\n", encoding="utf-8")
            hanging_codex = _hanging_codex(root)
            environment = _base_environment(
                workspace=workspace,
                context=context,
                prompt=prompt,
                submission=submission,
                slurm_tmp=slurm_tmp,
            )
            for reserved in (
                "PDE_MACRO_WORKSPACE",
                "PDE_MACRO_CONTEXT",
                "PDE_MACRO_PROMPT",
                "PDE_MACRO_SUBMISSION",
            ):
                environment.pop(reserved)
            launcher = SubprocessReferenceAgentLauncher(
                tuple(_launcher_command(hanging_codex, auth_source)),
                environment=environment,
                timeout_seconds=2.0,
            )

            with self.assertRaisesRegex(
                ReferenceAgentLaunchError,
                "exceeded its time budget",
            ):
                launcher.launch(
                    ReferenceAgentLaunchRequest(
                        workspace=workspace,
                        context_path=context,
                        prompt_path=prompt,
                        submission_path=submission,
                    )
                )

            pid_path = workspace / "hanging-codex.pid"
            self.assertTrue(pid_path.is_file())
            codex_pid = int(pid_path.read_text(encoding="utf-8"))
            deadline = time.monotonic() + 2.0
            while _pid_exists(codex_pid) and time.monotonic() < deadline:
                time.sleep(0.02)
            survived_timeout = _pid_exists(codex_pid)
            if survived_timeout:
                try:
                    os.killpg(codex_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.assertFalse(
                survived_timeout,
                "sessioned Codex descendant survived the Agent hard timeout",
            )
            self.assertEqual(auth_source.read_text(encoding="utf-8"), "{}\n")

    def test_rejects_submission_outside_workspace_before_codex(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace, context, prompt, _submission = _workspace(root)
            slurm_tmp = root / "slurm"
            slurm_tmp.mkdir()
            auth_source = root / "controller-auth.json"
            auth_source.write_text("{}\n", encoding="utf-8")
            outside = root / "outside-submission.json"
            environment = _base_environment(
                workspace=workspace,
                context=context,
                prompt=prompt,
                submission=outside,
                slurm_tmp=slurm_tmp,
            )

            completed = subprocess.run(
                _launcher_command(Path("/usr/bin/true"), auth_source),
                cwd=workspace,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("must stay inside", completed.stderr)
            self.assertFalse(outside.exists())
            self.assertEqual(tuple(slurm_tmp.iterdir()), ())

    def test_requires_slurm_job_and_active_python_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace, context, prompt, submission = _workspace(root)
            slurm_tmp = root / "slurm"
            slurm_tmp.mkdir()
            auth_source = root / "controller-auth.json"
            auth_source.write_text("{}\n", encoding="utf-8")
            environment = _base_environment(
                workspace=workspace,
                context=context,
                prompt=prompt,
                submission=submission,
                slurm_tmp=slurm_tmp,
            )
            del environment["SLURM_JOB_ID"]

            completed = subprocess.run(
                _launcher_command(Path("/usr/bin/true"), auth_source),
                cwd=workspace,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("SLURM_JOB_ID", completed.stderr)

    def test_rejects_missing_or_non_singleton_cuda_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace, context, prompt, submission = _workspace(root)
            slurm_tmp = root / "slurm"
            slurm_tmp.mkdir()
            auth_source = root / "controller-auth.json"
            auth_source.write_text("{}\n", encoding="utf-8")
            base = _base_environment(
                workspace=workspace,
                context=context,
                prompt=prompt,
                submission=submission,
                slurm_tmp=slurm_tmp,
            )

            for value in (None, "", "-1", "0,1", "0 1", "all"):
                with self.subTest(cuda_visible_devices=value):
                    environment = dict(base)
                    if value is None:
                        environment.pop("CUDA_VISIBLE_DEVICES")
                    else:
                        environment["CUDA_VISIBLE_DEVICES"] = value
                    completed = subprocess.run(
                        _launcher_command(Path("/usr/bin/true"), auth_source),
                        cwd=workspace,
                        env=environment,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    self.assertEqual(completed.returncode, 2)
                    self.assertIn("CUDA_VISIBLE_DEVICES", completed.stderr)

    def test_rejects_invalid_cuda_device_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace, context, prompt, submission = _workspace(root)
            slurm_tmp = root / "slurm"
            slurm_tmp.mkdir()
            auth_source = root / "controller-auth.json"
            auth_source.write_text("{}\n", encoding="utf-8")
            environment = _base_environment(
                workspace=workspace,
                context=context,
                prompt=prompt,
                submission=submission,
                slurm_tmp=slurm_tmp,
            )
            environment["CUDA_DEVICE_ORDER"] = "UNSAFE_ORDER"

            completed = subprocess.run(
                _launcher_command(Path("/usr/bin/true"), auth_source),
                cwd=workspace,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("CUDA_DEVICE_ORDER", completed.stderr)


def _workspace(root: Path) -> tuple[Path, Path, Path, Path]:
    workspace = root / "agent-workspace"
    workspace.mkdir()
    context = workspace / "context.json"
    context.write_text('{"schema":"test"}\n', encoding="utf-8")
    prompt = workspace / "prompt.md"
    prompt.write_text("Improve the macro ROM.\n", encoding="utf-8")
    submission = workspace / "submission.json"
    return workspace, context, prompt, submission


def _base_environment(
    *,
    workspace: Path,
    context: Path,
    prompt: Path,
    submission: Path,
    slurm_tmp: Path,
) -> dict[str, str]:
    python = Path(sys.executable).resolve()
    python_prefix = python.parent.parent
    return {
        "PATH": os.pathsep.join((str(python.parent), "/usr/bin", "/bin")),
        "LANG": "C.UTF-8",
        "USER": "launcher-test",
        "VIRTUAL_ENV": str(python_prefix),
        "SLURM_JOB_ID": "12345",
        "SLURM_TMPDIR": str(slurm_tmp),
        "CUDA_VISIBLE_DEVICES": "7",
        "PDE_MACRO_WORKSPACE": str(workspace),
        "PDE_MACRO_CONTEXT": str(context),
        "PDE_MACRO_PROMPT": str(prompt),
        "PDE_MACRO_SUBMISSION": str(submission),
    }


def _launcher_command(codex_binary: Path, auth_source: Path) -> list[str]:
    return [
        "/bin/bash",
        str(LAUNCHER),
        "--model",
        "gpt-test",
        "--reasoning-effort",
        "high",
        "--codex-binary",
        str(codex_binary),
        "--auth-file",
        str(auth_source),
    ]


def _hanging_codex(root: Path) -> Path:
    fake = root / "hanging-codex"
    fake.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python
            import os
            from pathlib import Path
            import time

            workspace = Path(os.environ["PDE_MACRO_WORKSPACE"])
            (workspace / "hanging-codex.pid").write_text(
                str(os.getpid()), encoding="utf-8"
            )
            print('{"type":"thread.started"}', flush=True)
            while True:
                time.sleep(1.0)
            """
        ),
        encoding="utf-8",
    )
    fake.chmod(0o700)
    return fake


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _fake_codex(root: Path) -> Path:
    fake = root / "fake-codex"
    fake.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python
            import json
            import os
            from pathlib import Path
            import sys
            import time

            print('{"type":"thread.started"}', flush=True)
            auth = Path(os.environ["CODEX_HOME"]) / "auth.json"
            for _ in range(500):
                if not auth.exists():
                    break
                time.sleep(0.01)
            if auth.exists():
                raise SystemExit(19)

            arguments = sys.argv[1:]
            output = Path(
                arguments[arguments.index("--output-last-message") + 1]
            )
            output.write_text("fake final message\\n", encoding="utf-8")
            submission = Path(os.environ["PDE_MACRO_SUBMISSION"])
            submission.write_text(
                '{"schema":"reference-macro-agent-submission-v1"}\\n',
                encoding="utf-8",
            )
            workspace = Path(os.environ["PDE_MACRO_WORKSPACE"])
            capture = {
                "arguments": arguments,
                "codex_home": os.environ["CODEX_HOME"],
                "context": os.environ["PDE_MACRO_CONTEXT"],
                "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "no_proxy": os.environ["NO_PROXY"],
                "prompt": os.environ["PDE_MACRO_PROMPT"],
                "prompt_stdin": sys.stdin.read(),
                "pythonpath": os.environ["PYTHONPATH"],
                "submission": os.environ["PDE_MACRO_SUBMISSION"],
                "unrelated_environment_present": (
                    "SHOULD_NOT_LEAK" in os.environ
                ),
                "workspace": str(workspace),
            }
            (workspace / "fake-codex-capture.json").write_text(
                json.dumps(capture, sort_keys=True) + "\\n",
                encoding="utf-8",
            )
            """
        ),
        encoding="utf-8",
    )
    fake.chmod(0o700)
    return fake


if __name__ == "__main__":
    unittest.main()
