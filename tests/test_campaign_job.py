from __future__ import annotations

import ast
from io import BytesIO
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
from threading import Event
from types import SimpleNamespace

import pytest

from pde_agent_verifier_solver.engine import campaign_job
from pde_agent_verifier_solver.engine.campaign_job import (
    CAMPAIGN_JOB_CONTROLLER,
    CAMPAIGN_JOB_STATE_SCHEMA,
    CampaignJobActiveError,
    CampaignJobAdapter,
    CampaignJobError,
    CampaignJobExecutionError,
    CampaignJobResources,
    CampaignJobSpec,
    CampaignJobSubmissionError,
    CampaignJobTimeout,
)


class _FakeProcess:
    def __init__(self, return_code: int = 0, output: bytes = b"") -> None:
        self.pid = 424242
        self.returncode: int | None = None
        self._return_code = return_code
        self.stdout = BytesIO(output)

    def wait(self, timeout: float | int | None = None) -> int:
        del timeout
        self.returncode = self._return_code
        return self._return_code

    def poll(self) -> int | None:
        return self.returncode


def _spec(
    tmp_path: Path,
    *,
    backend: str = "local",
    run_name: str = "campaign-run",
    argv: tuple[str, ...] | None = None,
    environment: dict[str, str] | None = None,
    resources: CampaignJobResources | None = None,
) -> CampaignJobSpec:
    return CampaignJobSpec(
        run_root=tmp_path / run_name,
        workdir=tmp_path,
        argv=argv or (sys.executable, "-c", "print('campaign')"),
        backend=backend,
        environment=environment or {},
        resources=resources or CampaignJobResources(),
    )


def _install_fake_popen(
    monkeypatch: pytest.MonkeyPatch,
    return_codes: list[int],
    outputs: list[bytes] | None = None,
) -> list[tuple[tuple[str, ...], dict[str, object]]]:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
    pending_outputs = list(outputs or [b""] * len(return_codes))

    def fake_popen(argv: tuple[str, ...], **kwargs: object) -> _FakeProcess:
        calls.append((tuple(argv), kwargs))
        return _FakeProcess(return_codes.pop(0), pending_outputs.pop(0))

    monkeypatch.setattr(campaign_job.subprocess, "Popen", fake_popen)
    return calls


def test_local_success_gates_argv_until_process_group_is_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "do-not-persist-this-value"
    calls = _install_fake_popen(
        monkeypatch,
        [0],
        [f"before-{secret}-after\n".encode("utf-8")],
    )
    spec = _spec(tmp_path, environment={"CAMPAIGN_API_KEY": secret})
    assert secret not in repr(spec)

    state = CampaignJobAdapter().launch(spec)

    assert state["schema"] == CAMPAIGN_JOB_STATE_SCHEMA
    assert state["controller"] == CAMPAIGN_JOB_CONTROLLER
    assert state["status"] == "completed"
    assert state["attempt"] == 1
    assert state["process_group_id"] == 424242
    assert state["recovery_scope"] == "whole_campaign_job_relaunch"
    assert state["mid_round_recovery"] is False
    assert "stages" not in state
    assert state["environment_keys"] == ["CAMPAIGN_API_KEY"]
    assert calls[0][0][:3] == (
        sys.executable,
        "-c",
        campaign_job._GATED_EXEC_CODE,
    )
    gate_fd = int(calls[0][0][3])
    assert calls[0][1]["pass_fds"] == (gate_fd,)
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["start_new_session"] is True
    assert calls[0][1]["cwd"] == tmp_path
    assert calls[0][1]["env"]["CAMPAIGN_API_KEY"] == secret

    state_text = (spec.run_root / "campaign-job.json").read_text(encoding="utf-8")
    assert secret not in state_text
    assert stat.S_IMODE((spec.run_root / "campaign-job.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((spec.run_root / ".campaign-job.lock").stat().st_mode) == 0o600
    assert Path(str(state["log"])).is_file()
    log_text = Path(str(state["log"])).read_text(encoding="utf-8")
    assert secret not in log_text
    assert log_text == "before--after\n"


def test_local_real_process_redacts_declared_environment_from_log(
    tmp_path: Path,
) -> None:
    secret = "real-child-secret"
    spec = _spec(
        tmp_path,
        argv=(
            sys.executable,
            "-c",
            "import os; print('prefix-' + os.environ['PRIVATE_VALUE'] + '-suffix')",
        ),
        environment={"PRIVATE_VALUE": secret},
        resources=CampaignJobResources(wall_seconds=10),
    )

    state = CampaignJobAdapter().launch(spec)

    assert state["status"] == "completed"
    log_text = Path(str(state["log"])).read_text(encoding="utf-8")
    assert log_text == "prefix--suffix\n"
    assert secret not in log_text


def test_local_process_does_not_inherit_undeclared_ambient_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "inherited-top-secret-9631"
    monkeypatch.setenv("INHERITED_ONLY_SECRET", secret)
    spec = _spec(
        tmp_path,
        argv=(
            sys.executable,
            "-c",
            "import os; print(os.environ.get('INHERITED_ONLY_SECRET', 'absent'))",
        ),
        resources=CampaignJobResources(wall_seconds=10),
    )

    state = CampaignJobAdapter().launch(spec)

    assert state["status"] == "completed"
    log_text = Path(str(state["log"])).read_text(encoding="utf-8")
    assert log_text == "absent\n"
    assert secret not in log_text


def test_local_failure_resumes_same_job_as_next_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_fake_popen(monkeypatch, [7, 0])
    spec = _spec(tmp_path)
    adapter = CampaignJobAdapter()

    with pytest.raises(CampaignJobExecutionError, match="exited with code 7"):
        adapter.launch(spec)
    failed = adapter.read(spec.run_root)
    assert failed["status"] == "failed"
    assert failed["return_code"] == 7
    assert failed["error_type"] == "NonzeroExit"

    completed = adapter.resume(spec)
    assert completed["status"] == "completed"
    assert completed["attempt"] == 2
    assert len(completed["attempt_history"]) == 2
    assert completed["attempt_history"][0]["status"] == "failed"
    assert completed["attempt_history"][1]["status"] == "completed"
    assert completed["attempt_history"][0]["log"] != completed["log"]
    assert len(calls) == 2

    changed = _spec(tmp_path, argv=(sys.executable, "-c", "print('changed')"))
    # Completion is idempotent only for the exact persisted command.
    with pytest.raises(CampaignJobError, match="persisted argv"):
        adapter.resume(changed)


def test_completed_resume_is_idempotent_without_reexecution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_popen(monkeypatch, [0])
    spec = _spec(tmp_path)
    adapter = CampaignJobAdapter()
    original = adapter.launch(spec)

    def forbidden_popen(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("completed campaign must not execute again")

    monkeypatch.setattr(campaign_job.subprocess, "Popen", forbidden_popen)
    resumed = adapter.resume(spec)
    assert resumed == original
    assert resumed["attempt"] == 1


def test_local_timeout_terminates_process_group_and_is_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class TimeoutThenKilled(_FakeProcess):
        def __init__(self) -> None:
            super().__init__(-9)
            self.wait_count = 0

        def wait(self, timeout: float | int | None = None) -> int:
            del timeout
            self.wait_count += 1
            if self.wait_count <= 2:
                raise subprocess.TimeoutExpired(("campaign",), 1)
            self.returncode = -9
            return -9

        def poll(self) -> int | None:
            return None if self.returncode is None else self.returncode

    timed_out = TimeoutThenKilled()
    successful = _FakeProcess(0)
    processes = [timed_out, successful]
    signals: list[tuple[int, signal.Signals]] = []
    group_alive = True

    def fake_popen(*args: object, **kwargs: object) -> _FakeProcess:
        del args, kwargs
        return processes.pop(0)

    monkeypatch.setattr(campaign_job.subprocess, "Popen", fake_popen)
    def fake_killpg(pid: int, signum: signal.Signals | int) -> None:
        nonlocal group_alive
        if signum == 0:
            if not group_alive:
                raise ProcessLookupError
            return
        signals.append((pid, signal.Signals(signum)))
        if signum == signal.SIGKILL:
            group_alive = False

    monkeypatch.setattr(campaign_job.os, "killpg", fake_killpg)
    monkeypatch.setattr(campaign_job, "_TERMINATION_GRACE_SECONDS", 0.001)
    spec = _spec(tmp_path, resources=CampaignJobResources(wall_seconds=1))
    adapter = CampaignJobAdapter()

    with pytest.raises(CampaignJobTimeout, match="exceeded its wall time"):
        adapter.launch(spec)
    interrupted = adapter.read(spec.run_root)
    assert interrupted["status"] == "interrupted"
    assert interrupted["error_type"] == "WallClockTimeout"
    assert signals == [(timed_out.pid, signal.SIGTERM), (timed_out.pid, signal.SIGKILL)]

    completed = adapter.resume(spec)
    assert completed["status"] == "completed"
    assert completed["attempt"] == 2


def test_normal_leader_exit_terminates_a_surviving_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BackgroundPipe:
        def __init__(self) -> None:
            self.released = Event()

        def read(self, size: int) -> bytes:
            del size
            assert self.released.wait(timeout=1)
            return b""

        def close(self) -> None:
            self.released.set()

    process = _FakeProcess(0)
    background_pipe = BackgroundPipe()
    process.stdout = background_pipe
    group_alive = True
    signals: list[signal.Signals] = []

    monkeypatch.setattr(
        campaign_job.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    def fake_killpg(pid: int, signum: signal.Signals | int) -> None:
        nonlocal group_alive
        assert pid == process.pid
        if signum == 0:
            if not group_alive:
                raise ProcessLookupError
            return
        signals.append(signal.Signals(signum))
        if signum == signal.SIGTERM:
            group_alive = False
            background_pipe.released.set()

    monkeypatch.setattr(campaign_job.os, "killpg", fake_killpg)

    state = CampaignJobAdapter().launch(_spec(tmp_path))

    assert state["status"] == "completed"
    assert signals == [signal.SIGTERM]


def test_slurm_submits_one_script_without_command_or_environment_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "batch-secret-never-persist"
    inherited_secret = "undeclared-submit-secret"
    monkeypatch.setenv("INHERITED_ONLY_SECRET", inherited_secret)
    resources = CampaignJobResources(
        cpus=8,
        memory="32G",
        wall_seconds=125,
        partition="gpu",
        gres="gpu:1",
        account="research",
        qos="normal",
    )
    spec = _spec(
        tmp_path,
        backend="slurm",
        argv=(sys.executable, "-c", "print('unique-composition-command')"),
        environment={"CAMPAIGN_SECRET": secret},
        resources=resources,
    )
    submissions: list[tuple[tuple[str, ...], dict[str, object]]] = []

    monkeypatch.setattr(campaign_job.shutil, "which", lambda name: "/usr/bin/sbatch")

    def fake_run(argv: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        submissions.append((tuple(argv), kwargs))
        return SimpleNamespace(returncode=0, stdout="12345;cluster\n", stderr="")

    monkeypatch.setattr(campaign_job.subprocess, "run", fake_run)
    state = CampaignJobAdapter().launch(spec)

    assert state["status"] == "submitted"
    assert state["job_id"] == "12345"
    assert state["attempt"] == 1
    assert "stages" not in state
    assert len(submissions) == 1
    command, options = submissions[0]
    assert command[0] == "/usr/bin/sbatch"
    assert "--parsable" in command
    export_option = next(item for item in command if item.startswith("--export="))
    assert export_option != "--export=ALL"
    assert "CAMPAIGN_SECRET" in export_option.split("=", 1)[1].split(",")
    assert "INHERITED_ONLY_SECRET" not in export_option
    assert "--cpus-per-task=8" in command
    assert "--mem=32G" in command
    assert "--time=00:02:05" in command
    assert "--partition=gpu" in command
    assert "--gres=gpu:1" in command
    assert options["shell"] is False
    assert options["env"]["CAMPAIGN_SECRET"] == secret
    assert "INHERITED_ONLY_SECRET" not in options["env"]

    scripts = list((spec.run_root / "scripts").glob("*.sh"))
    assert len(scripts) == 1
    script = scripts[0].read_text(encoding="utf-8")
    assert "pde_agent_verifier_solver.engine.campaign_job" in script
    assert "_execute-slurm" in script
    assert "unique-composition-command" not in script
    assert secret not in script
    assert stat.S_IMODE(scripts[0].stat().st_mode) == 0o700
    state_text = (spec.run_root / "campaign-job.json").read_text(encoding="utf-8")
    assert secret not in state_text

    with pytest.raises(CampaignJobActiveError, match="is active"):
        CampaignJobAdapter().resume(spec)
    assert len(submissions) == 1


def test_failed_slurm_submission_resumes_as_a_new_single_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "submission-error-secret"
    spec = _spec(
        tmp_path,
        backend="slurm",
        environment={"CAMPAIGN_SECRET": secret},
    )
    results = [
        SimpleNamespace(returncode=1, stdout="", stderr=secret),
        SimpleNamespace(returncode=0, stdout="7777\n", stderr=""),
    ]
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(campaign_job.shutil, "which", lambda name: "/usr/bin/sbatch")

    def fake_run(argv: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        del kwargs
        calls.append(tuple(argv))
        return results.pop(0)

    monkeypatch.setattr(campaign_job.subprocess, "run", fake_run)
    adapter = CampaignJobAdapter()
    with pytest.raises(CampaignJobSubmissionError) as error:
        adapter.launch(spec)
    assert secret not in str(error.value)
    assert adapter.read(spec.run_root)["status"] == "failed"

    state = adapter.resume(spec)
    assert state["status"] == "submitted"
    assert state["attempt"] == 2
    assert state["job_id"] == "7777"
    assert [item["status"] for item in state["attempt_history"]] == [
        "failed",
        "submitted",
    ]
    assert len(calls) == 2
    assert len(list((spec.run_root / "scripts").glob("*.sh"))) == 2
    assert secret not in (spec.run_root / "campaign-job.json").read_text(
        encoding="utf-8"
    )


def test_external_inactivity_reconciliation_is_guarded_and_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path, backend="slurm")
    results = [
        SimpleNamespace(returncode=0, stdout="4444\n", stderr=""),
        SimpleNamespace(returncode=0, stdout="5555\n", stderr=""),
    ]
    monkeypatch.setattr(campaign_job.shutil, "which", lambda name: "/usr/bin/sbatch")
    monkeypatch.setattr(
        campaign_job.subprocess,
        "run",
        lambda *args, **kwargs: results.pop(0),
    )
    adapter = CampaignJobAdapter()
    active = adapter.launch(spec)
    assert active["status"] == "submitted"

    with pytest.raises(ValueError, match="inactivity confirmation"):
        adapter.reconcile_incomplete(
            spec.run_root,
            expected_attempt=1,
            expected_job_id="4444",
            externally_confirmed_inactive=False,
        )
    with pytest.raises(CampaignJobError, match="job id changed"):
        adapter.reconcile_incomplete(
            spec.run_root,
            expected_attempt=1,
            expected_job_id="9999",
            externally_confirmed_inactive=True,
        )
    with pytest.raises(CampaignJobError, match="attempt changed"):
        adapter.reconcile_incomplete(
            spec.run_root,
            expected_attempt=2,
            expected_job_id="4444",
            externally_confirmed_inactive=True,
        )
    assert adapter.read(spec.run_root)["status"] == "submitted"

    incomplete = adapter.reconcile_incomplete(
        spec.run_root,
        expected_attempt=1,
        expected_job_id="4444",
        externally_confirmed_inactive=True,
    )
    assert incomplete["status"] == "incomplete"
    assert incomplete["error_type"] == "ExternallyConfirmedInactive"
    assert incomplete["finished_at"] is not None
    assert (
        adapter.reconcile_incomplete(
            spec.run_root,
            expected_attempt=1,
            expected_job_id="4444",
            externally_confirmed_inactive=True,
        )
        == incomplete
    )

    resumed = adapter.resume(spec)
    assert resumed["status"] == "submitted"
    assert resumed["attempt"] == 2
    assert resumed["job_id"] == "5555"
    assert resumed["attempt_history"][0]["status"] == "incomplete"


def test_unrecorded_slurm_submission_can_be_reconciled_and_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path, backend="slurm")
    spec.run_root.mkdir(mode=0o700)
    with campaign_job._locked_run(spec.run_root, initialize=True):
        (spec.run_root / "logs").mkdir(mode=0o700)
        (spec.run_root / "scripts").mkdir(mode=0o700)
        campaign_job._atomic_json_write(
            spec.run_root / "campaign-job.json",
            campaign_job._new_state(spec),
        )

    adapter = CampaignJobAdapter()
    incomplete = adapter.reconcile_incomplete(
        spec.run_root,
        expected_attempt=1,
        expected_job_id="7777",
        externally_confirmed_inactive=True,
    )
    assert incomplete["status"] == "incomplete"
    assert incomplete["job_id"] is None

    monkeypatch.setattr(campaign_job.shutil, "which", lambda name: "/usr/bin/sbatch")
    monkeypatch.setattr(
        campaign_job.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="8888\n", stderr=""
        ),
    )
    resumed = adapter.resume(spec)
    assert resumed["status"] == "submitted"
    assert resumed["attempt"] == 2
    assert resumed["job_id"] == "8888"


def test_local_reconciliation_checks_persisted_process_group_itself(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    spec.run_root.mkdir(mode=0o700)
    with campaign_job._locked_run(spec.run_root, initialize=True):
        (spec.run_root / "logs").mkdir(mode=0o700)
        campaign_job._atomic_json_write(
            spec.run_root / "campaign-job.json",
            campaign_job._new_state(spec),
        )

    process = subprocess.Popen(
        (sys.executable, "-c", "import time; time.sleep(60)"),
        start_new_session=True,
    )
    adapter = CampaignJobAdapter()
    try:
        campaign_job._record_local_process_group(spec.run_root, 1, process.pid)
        running = adapter.read(spec.run_root)
        assert running["process_group_id"] == process.pid
        with pytest.raises(CampaignJobActiveError, match="still active"):
            adapter.reconcile_incomplete(
                spec.run_root,
                expected_attempt=1,
                expected_job_id=None,
                expected_process_group_id=process.pid,
                externally_confirmed_inactive=True,
            )
    finally:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)

    with pytest.raises(CampaignJobError, match="process group changed"):
        adapter.reconcile_incomplete(
            spec.run_root,
            expected_attempt=1,
            expected_job_id=None,
            expected_process_group_id=process.pid + 1,
            externally_confirmed_inactive=True,
        )
    incomplete = adapter.reconcile_incomplete(
        spec.run_root,
        expected_attempt=1,
        expected_job_id=None,
        expected_process_group_id=process.pid,
        externally_confirmed_inactive=True,
    )
    assert incomplete["status"] == "incomplete"
    assert incomplete["process_group_id"] == process.pid


def test_local_null_process_group_can_be_reconciled_and_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path)
    spec.run_root.mkdir(mode=0o700)
    with campaign_job._locked_run(spec.run_root, initialize=True):
        (spec.run_root / "logs").mkdir(mode=0o700)
        campaign_job._atomic_json_write(
            spec.run_root / "campaign-job.json",
            campaign_job._new_state(spec),
        )

    adapter = CampaignJobAdapter()
    incomplete = adapter.reconcile_incomplete(
        spec.run_root,
        expected_attempt=1,
        expected_job_id=None,
        expected_process_group_id=None,
        externally_confirmed_inactive=True,
    )
    assert incomplete["status"] == "incomplete"
    assert incomplete["process_group_id"] is None

    calls = _install_fake_popen(monkeypatch, [0])
    resumed = adapter.resume(spec)
    assert resumed["status"] == "completed"
    assert resumed["attempt"] == 2
    assert resumed["process_group_id"] == 424242
    assert len(calls) == 1


def test_slurm_worker_requires_declared_ambient_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "ambient-only-secret"
    spec = _spec(
        tmp_path,
        backend="slurm",
        environment={"REQUIRED_BATCH_SECRET": secret},
    )
    monkeypatch.setattr(campaign_job.shutil, "which", lambda name: "/usr/bin/sbatch")
    monkeypatch.setattr(
        campaign_job.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="9876\n", stderr=""
        ),
    )
    CampaignJobAdapter().launch(spec)
    monkeypatch.delenv("REQUIRED_BATCH_SECRET", raising=False)

    def forbidden_popen(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("worker cannot execute without every declared env key")

    monkeypatch.setattr(campaign_job.subprocess, "Popen", forbidden_popen)
    assert campaign_job._execute_slurm_attempt(spec.run_root, 1) == 78
    state = CampaignJobAdapter().read(spec.run_root)
    assert state["status"] == "failed"
    assert state["error_type"] == "MissingDeclaredEnvironment"
    state_text = (spec.run_root / "campaign-job.json").read_text(encoding="utf-8")
    assert secret not in state_text


def test_slurm_worker_runs_direct_command_from_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "ambient-worker-value"
    spec = _spec(
        tmp_path,
        backend="slurm",
        environment={"REQUIRED_BATCH_SECRET": secret},
    )
    monkeypatch.setattr(campaign_job.shutil, "which", lambda name: "/usr/bin/sbatch")
    monkeypatch.setattr(
        campaign_job.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="2468\n", stderr=""
        ),
    )
    CampaignJobAdapter().launch(spec)
    monkeypatch.setenv("REQUIRED_BATCH_SECRET", secret)
    monkeypatch.setenv("SLURM_JOB_ID", "2468")
    calls = _install_fake_popen(
        monkeypatch,
        [0],
        [f"worker-{secret}-output\n".encode("utf-8")],
    )

    assert campaign_job._execute_slurm_attempt(spec.run_root, 1) == 0
    state = CampaignJobAdapter().read(spec.run_root)
    assert state["status"] == "completed"
    assert state["job_id"] == "2468"
    assert calls[0][0] == spec.argv
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["start_new_session"] is True
    assert calls[0][1]["env"]["REQUIRED_BATCH_SECRET"] == secret
    log_text = Path(str(state["log"])).read_text(encoding="utf-8")
    assert secret not in log_text
    assert log_text == "worker--output\n"


@pytest.mark.parametrize(
    "argv",
    [
        ("program", "--token", "value"),
        ("program", "--api-key=value"),
        ("program", "--hf-token=TOP-SECRET"),
        ("program", "--openai_api_key", "TOP-SECRET"),
        ("program", "https://user:password@example.test/path"),
    ],
)
def test_credential_bearing_argv_is_rejected(tmp_path: Path, argv: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="credential"):
        _spec(tmp_path, argv=argv)


def test_ordinary_token_substring_is_not_rejected(tmp_path: Path) -> None:
    spec = _spec(tmp_path, argv=("program", "assets/tokenizer.json"))
    assert spec.argv[-1] == "assets/tokenizer.json"


def test_paths_environment_and_resources_are_strict(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        CampaignJobSpec(
            run_root=Path("relative"),
            workdir=tmp_path,
            argv=("program",),
        )
    with pytest.raises(ValueError, match="invalid name"):
        _spec(tmp_path, environment={"BAD-NAME": "value"})
    with pytest.raises(ValueError, match="invalid value"):
        _spec(tmp_path, environment={"NAME": "line\nbreak"})
    with pytest.raises(TypeError, match="cpus"):
        CampaignJobResources(cpus=True)
    with pytest.raises(ValueError, match="wall_seconds"):
        CampaignJobResources(wall_seconds=0)
    with pytest.raises(ValueError, match="partition"):
        CampaignJobResources(partition="gpu queue")


def test_module_has_no_science_lifecycle_or_problem_specific_imports() -> None:
    path = Path(campaign_job.__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = ("staged", "macro", "plugins", ".ns", ".vpfp", ".gkw")
    assert not any(
        marker in module.casefold() for module in imported for marker in forbidden
    )
    assert "MacroClosedLoopRunner" not in source
    assert "RunStage" not in source
