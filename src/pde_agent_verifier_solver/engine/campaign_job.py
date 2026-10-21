"""One-job execution adapter for a canonical campaign composition command.

The adapter schedules exactly one operating-system job for one campaign.  The
command supplied by the caller remains responsible for constructing and
running ``CampaignController``.  Retry is therefore a whole-job relaunch; this
module neither interprets controller state nor promises mid-round recovery.

Values supplied through ``CampaignJobSpec.environment`` are deliberately
absent from every persistent artifact and redacted from captured output.  The
argv is persisted for exact whole-job resume, so callers must never place a
secret in argv; semantic option checks are defense in depth, not a universal
secret detector.  For a batch job, only declared environment names are stored
and their values are obtained from the job's ambient environment.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from threading import Thread
import time
from types import MappingProxyType
from typing import BinaryIO, Callable, Iterator, Mapping, Sequence


CAMPAIGN_JOB_STATE_SCHEMA = "canonical-campaign-job-state-v1"
CAMPAIGN_JOB_CONTROLLER = "CampaignController"
CAMPAIGN_JOB_STATE_NAME = "campaign-job.json"

_LOCK_NAME = ".campaign-job.lock"
_ACTIVE_STATUSES = frozenset({"submitting", "submitted", "running"})
_RESUMABLE_STATUSES = frozenset({"failed", "interrupted", "incomplete"})
_ALL_STATUSES = _ACTIVE_STATUSES | _RESUMABLE_STATUSES | {"completed"}
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_RESOURCE_ATOM = re.compile(r"[A-Za-z0-9_.:+/@-]*\Z")
_MEMORY = re.compile(r"[1-9][0-9]*(?:[KMGTPE]i?B?|[kmgtpe]i?[bB]?)?\Z")
_JOB_ID = re.compile(r"[0-9]+\Z")
_URL_USERINFO = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://[^/@\s:]+:[^/@\s]+@")
_CREDENTIAL_OPTIONS = frozenset(
    {
        "--token",
        "--access-token",
        "--refresh-token",
        "--password",
        "--passwd",
        "--secret",
        "--client-secret",
        "--api-key",
        "--apikey",
        "--auth-file",
        "--credential",
        "--credentials",
        "--private-key",
        "--key-file",
    }
)
_CREDENTIAL_WORDS = frozenset(
    {"password", "passwd", "secret", "credential", "credentials", "authorization"}
)
_TOKEN_QUALIFIERS = frozenset(
    {"access", "refresh", "hf", "huggingface", "github", "gitlab", "api", "auth", "bearer", "oauth"}
)
_SAFE_AMBIENT_ENVIRONMENT = frozenset(
    {
        "PATH",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CPATH",
        "C_INCLUDE_PATH",
        "CPLUS_INCLUDE_PATH",
        "PKG_CONFIG_PATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TMPDIR",
        "TEMP",
        "TMP",
        "HOME",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
    }
)
_SAFE_AMBIENT_PREFIXES = ("LC_", "SLURM_")
_MAX_ARGUMENTS = 256
_MAX_ARGUMENT_SIZE = 4096
_MAX_COMMAND_SIZE = 65536
_MAX_ENVIRONMENT_ITEMS = 128
_MAX_ENVIRONMENT_VALUE_SIZE = 32768
_MAX_WALL_SECONDS = 31 * 24 * 60 * 60
_TERMINATION_GRACE_SECONDS = 5.0
_GATED_EXEC_CODE = """\
import json
import os
import sys

with os.fdopen(int(sys.argv[1]), "rb", closefd=True) as stream:
    payload = stream.read()
if not payload:
    raise SystemExit(78)
argv = json.loads(payload.decode("utf-8"))
if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
    raise SystemExit(78)
os.execvpe(argv[0], argv, dict(os.environ))
"""


class CampaignJobError(RuntimeError):
    """Base error for a campaign job operation."""


class CampaignJobActiveError(CampaignJobError):
    """A relaunch was refused because the persisted job is still active."""


class CampaignJobExecutionError(CampaignJobError):
    """A local or batch-worker composition command failed."""


class CampaignJobTimeout(CampaignJobExecutionError):
    """The whole campaign job exceeded its configured wall time."""


class CampaignJobSubmissionError(CampaignJobError):
    """A batch job could not be submitted safely."""


class _CommandWallTimeout(Exception):
    def __init__(
        self, return_code: int | None, *, cleanup_confirmed: bool
    ) -> None:
        super().__init__("command wall time exhausted")
        self.return_code = return_code
        self.cleanup_confirmed = cleanup_confirmed


class _CommandCleanupUnconfirmed(Exception):
    def __init__(self, return_code: int | None) -> None:
        super().__init__("command process-group cleanup was not confirmed")
        self.return_code = return_code


@dataclass(frozen=True)
class CampaignJobResources:
    """Portable resources for the single campaign job."""

    cpus: int = 4
    memory: str = "16G"
    wall_seconds: int = 3600
    partition: str = ""
    gres: str = ""
    account: str = ""
    qos: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.cpus, bool) or not isinstance(self.cpus, int):
            raise TypeError("cpus must be an integer")
        if not 1 <= self.cpus <= 1024:
            raise ValueError("cpus must be between 1 and 1024")
        if not isinstance(self.memory, str) or not _MEMORY.fullmatch(self.memory):
            raise ValueError("memory must be a positive Slurm memory value")
        if isinstance(self.wall_seconds, bool) or not isinstance(self.wall_seconds, int):
            raise TypeError("wall_seconds must be an integer")
        if not 1 <= self.wall_seconds <= _MAX_WALL_SECONDS:
            raise ValueError(
                f"wall_seconds must be between 1 and {_MAX_WALL_SECONDS}"
            )
        for name in ("partition", "gres", "account", "qos"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _RESOURCE_ATOM.fullmatch(value):
                raise ValueError(f"{name} must be a single safe resource value")

    def as_state(self) -> dict[str, object]:
        return {
            "cpus": self.cpus,
            "memory": self.memory,
            "wall_seconds": self.wall_seconds,
            "partition": self.partition,
            "gres": self.gres,
            "account": self.account,
            "qos": self.qos,
        }


@dataclass(frozen=True)
class CampaignJobSpec:
    """Validated request to launch one campaign composition command."""

    run_root: Path
    workdir: Path
    argv: tuple[str, ...]
    backend: str = "local"
    environment: Mapping[str, str] = field(default_factory=dict, repr=False)
    resources: CampaignJobResources = field(default_factory=CampaignJobResources)

    def __post_init__(self) -> None:
        run_root = _canonical_path(self.run_root, name="run_root", must_exist=False)
        workdir = _canonical_path(self.workdir, name="workdir", must_exist=True)
        if not workdir.is_dir():
            raise ValueError("workdir must be a directory")
        if run_root == Path(run_root.anchor):
            raise ValueError("run_root cannot be a filesystem root")

        argv = _validate_argv(self.argv)
        if self.backend not in {"local", "slurm"}:
            raise ValueError("backend must be 'local' or 'slurm'")
        if not isinstance(self.resources, CampaignJobResources):
            raise TypeError("resources must be CampaignJobResources")
        environment = _validate_environment(self.environment)

        object.__setattr__(self, "run_root", run_root)
        object.__setattr__(self, "workdir", workdir)
        object.__setattr__(self, "argv", argv)
        object.__setattr__(self, "environment", MappingProxyType(environment))


class CampaignJobAdapter:
    """Launch and whole-job-resume one canonical campaign command."""

    def launch(self, spec: CampaignJobSpec) -> Mapping[str, object]:
        if not isinstance(spec, CampaignJobSpec):
            raise TypeError("spec must be CampaignJobSpec")
        try:
            spec.run_root.mkdir(mode=0o700, parents=False, exist_ok=False)
        except FileExistsError as exc:
            raise CampaignJobError("campaign run_root already exists") from exc
        with _locked_run(spec.run_root, initialize=True):
            (spec.run_root / "logs").mkdir(mode=0o700)
            if spec.backend == "slurm":
                (spec.run_root / "scripts").mkdir(mode=0o700)
            state = _new_state(spec)
            _atomic_json_write(_state_path(spec.run_root), state)

        if spec.backend == "local":
            return self._run_local(spec)
        return self._submit_slurm(spec)

    def resume(self, spec: CampaignJobSpec) -> Mapping[str, object]:
        if not isinstance(spec, CampaignJobSpec):
            raise TypeError("spec must be CampaignJobSpec")
        if not spec.run_root.is_dir():
            raise FileNotFoundError("campaign run_root does not exist")

        with _locked_run(spec.run_root):
            state = _read_state_unlocked(spec.run_root)
            _require_matching_spec(state, spec)
            status = str(state["status"])
            if status == "completed":
                return _copy_json(state)
            if status in _ACTIVE_STATUSES:
                raise CampaignJobActiveError(
                    f"campaign job attempt {state['attempt']} is active ({status})"
                )
            if status not in _RESUMABLE_STATUSES:
                raise CampaignJobError(f"campaign job status cannot resume: {status}")
            state = _append_attempt(state, spec)
            _atomic_json_write(_state_path(spec.run_root), state)

        if spec.backend == "local":
            return self._run_local(spec)
        return self._submit_slurm(spec)

    def read(self, run_root: Path) -> Mapping[str, object]:
        root = _canonical_path(run_root, name="run_root", must_exist=True)
        if not root.is_dir():
            raise ValueError("run_root must be a directory")
        with _locked_run(root):
            return _copy_json(_read_state_unlocked(root))

    def reconcile_incomplete(
        self,
        run_root: Path,
        *,
        expected_attempt: int,
        expected_job_id: str | None,
        expected_process_group_id: int | None = None,
        externally_confirmed_inactive: bool,
    ) -> Mapping[str, object]:
        """Mark an abandoned active attempt incomplete after an external check.

        The caller must establish that the scheduler job is inactive.  Local
        process groups are additionally checked by this adapter.  Persisted
        identifiers are compared while holding the run lock, so stale
        observations cannot rewrite a newer attempt.  This method never
        performs or implies an automatic active-job takeover.

        A coordinator may die after ``sbatch`` accepts a job but before its id
        is recorded.  In that ``submitting``/null-id state, the caller must use
        the attempt's scheduler job name and script path to establish that no
        matching job is active; ``expected_job_id`` may be the externally
        recovered id or ``None`` when submission itself was not accepted.
        """

        if externally_confirmed_inactive is not True:
            raise ValueError("external inactivity confirmation is required")
        if (
            isinstance(expected_attempt, bool)
            or not isinstance(expected_attempt, int)
            or expected_attempt < 1
        ):
            raise ValueError("expected_attempt must be a positive integer")
        if expected_job_id is not None and (
            not isinstance(expected_job_id, str)
            or not _JOB_ID.fullmatch(expected_job_id)
        ):
            raise ValueError("expected_job_id must be a numeric scheduler id or None")
        if expected_process_group_id is not None and (
            isinstance(expected_process_group_id, bool)
            or not isinstance(expected_process_group_id, int)
            or expected_process_group_id < 1
        ):
            raise ValueError(
                "expected_process_group_id must be a positive integer or None"
            )
        root = _canonical_path(run_root, name="run_root", must_exist=True)
        if not root.is_dir():
            raise ValueError("run_root must be a directory")
        with _locked_run(root):
            state = _read_state_unlocked(root)
            if state["attempt"] != expected_attempt:
                raise CampaignJobError("campaign job attempt changed after confirmation")
            if state["backend"] == "local":
                if expected_job_id is not None:
                    raise ValueError("a local campaign job has no scheduler job id")
                process_group_id = state["process_group_id"]
                if process_group_id is None:
                    if expected_process_group_id is not None:
                        raise CampaignJobError(
                            "campaign process group changed after confirmation"
                        )
                    # The gated child cannot execute the campaign argv until
                    # its process group has been durably recorded.  A dead
                    # coordinator plus a null id therefore means no science
                    # workload escaped and the attempt is safe to reconcile.
                elif process_group_id != expected_process_group_id:
                    raise CampaignJobError(
                        "campaign process group changed after confirmation"
                    )
                elif _process_group_alive(int(process_group_id)):
                    raise CampaignJobActiveError(
                        "local campaign process group is still active"
                    )
            else:
                if expected_process_group_id is not None:
                    raise ValueError("a Slurm campaign job has no local process group")
                if (
                    state["job_id"] is not None
                    and state["job_id"] != expected_job_id
                ):
                    raise CampaignJobError(
                        "campaign job id changed after confirmation"
                    )
            if state["status"] == "incomplete":
                return _copy_json(state)
            if state["status"] not in _ACTIVE_STATUSES:
                raise CampaignJobError(
                    "only an externally inactive active-state attempt can be reconciled"
                )
            state = _update_attempt_unlocked(
                state,
                status="incomplete",
                return_code=state["return_code"],
                error_type="ExternallyConfirmedInactive",
                finished=True,
            )
            _validate_state(state, root)
            _atomic_json_write(_state_path(root), state)
            return _copy_json(state)

    def _run_local(self, spec: CampaignJobSpec) -> Mapping[str, object]:
        with _locked_run(spec.run_root):
            state = _read_state_unlocked(spec.run_root)
            attempt = int(state["attempt"])
            if state["backend"] != "local" or state["status"] != "running":
                raise CampaignJobError("local campaign job is not ready to run")
            log_path = _artifact_path(spec.run_root, str(state["log"]), "logs")

        environment = _job_environment(os.environ, spec.environment)
        try:
            return_code = _run_bounded_command(
                spec.argv,
                cwd=spec.workdir,
                environment=environment,
                redacted_values=tuple(spec.environment.values()),
                log_path=log_path,
                wall_seconds=spec.resources.wall_seconds,
                on_started=lambda process_group_id: _record_local_process_group(
                    spec.run_root,
                    attempt,
                    process_group_id,
                ),
            )
        except _CommandWallTimeout as exc:
            if exc.cleanup_confirmed:
                _finish_attempt(
                    spec.run_root,
                    attempt,
                    status="interrupted",
                    return_code=exc.return_code,
                    error_type="WallClockTimeout",
                )
            else:
                _record_active_error(
                    spec.run_root,
                    attempt,
                    return_code=exc.return_code,
                    error_type="WallClockTimeoutCleanupUnconfirmed",
                )
            raise CampaignJobTimeout(
                f"campaign job attempt {attempt} exceeded its wall time"
            ) from None
        except _CommandCleanupUnconfirmed as exc:
            _record_active_error(
                spec.run_root,
                attempt,
                return_code=exc.return_code,
                error_type="ProcessGroupCleanupUnconfirmed",
            )
            raise CampaignJobExecutionError(
                f"campaign job attempt {attempt} process cleanup is unconfirmed"
            ) from None
        except (KeyboardInterrupt, SystemExit):
            _finish_attempt(
                spec.run_root,
                attempt,
                status="interrupted",
                return_code=None,
                error_type="HostInterrupted",
            )
            raise
        except Exception as exc:
            _finish_attempt(
                spec.run_root,
                attempt,
                status="failed",
                return_code=None,
                error_type=type(exc).__name__,
            )
            raise CampaignJobExecutionError(
                f"campaign job attempt {attempt} could not execute"
            ) from None

        if return_code != 0:
            _finish_attempt(
                spec.run_root,
                attempt,
                status="failed",
                return_code=return_code,
                error_type="NonzeroExit",
            )
            raise CampaignJobExecutionError(
                f"campaign job attempt {attempt} exited with code {return_code}"
            )
        return _finish_attempt(
            spec.run_root,
            attempt,
            status="completed",
            return_code=0,
            error_type=None,
        )

    def _submit_slurm(self, spec: CampaignJobSpec) -> Mapping[str, object]:
        with _locked_run(spec.run_root):
            state = _read_state_unlocked(spec.run_root)
            attempt = int(state["attempt"])
            if state["backend"] != "slurm" or state["status"] != "submitting":
                raise CampaignJobError("Slurm campaign job is not ready to submit")
            script_path = _artifact_path(
                spec.run_root, str(state["script"]), "scripts"
            )
            _atomic_text_write(
                script_path,
                _render_slurm_script(spec.run_root, attempt),
                mode=0o700,
            )

        sbatch = shutil.which("sbatch")
        if sbatch is None:
            _finish_attempt(
                spec.run_root,
                attempt,
                status="failed",
                return_code=None,
                error_type="SlurmUnavailable",
            )
            raise CampaignJobSubmissionError("Slurm submission command is unavailable")

        command = _sbatch_command(sbatch, spec, script_path, attempt)
        environment = _job_environment(os.environ, spec.environment)
        try:
            result = subprocess.run(
                command,
                cwd=spec.workdir,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                shell=False,
            )
        except Exception:
            _record_active_error(
                spec.run_root,
                attempt,
                return_code=None,
                error_type="SubmissionOutcomeUnconfirmed",
            )
            raise CampaignJobSubmissionError(
                f"campaign job attempt {attempt} could not be submitted"
            ) from None

        if result.returncode != 0:
            _finish_attempt(
                spec.run_root,
                attempt,
                status="failed",
                return_code=result.returncode,
                error_type="SubmissionNonzeroExit",
            )
            raise CampaignJobSubmissionError(
                f"campaign job attempt {attempt} submission failed"
            )
        job_id = result.stdout.strip().split(";", 1)[0]
        if not _JOB_ID.fullmatch(job_id):
            _record_active_error(
                spec.run_root,
                attempt,
                return_code=result.returncode,
                error_type="InvalidSubmissionResponse",
            )
            raise CampaignJobSubmissionError(
                f"campaign job attempt {attempt} returned an invalid job id"
            )
        return _record_submission(spec.run_root, attempt, job_id)


def _canonical_path(value: Path, *, name: str, must_exist: bool) -> Path:
    if isinstance(value, (str, bytes)) or not isinstance(value, os.PathLike):
        raise TypeError(f"{name} must be a Path")
    raw = Path(value)
    if not raw.is_absolute():
        raise ValueError(f"{name} must be absolute")
    if any(part in {"", ".", ".."} for part in raw.parts[1:]):
        raise ValueError(f"{name} must be canonical")
    try:
        resolved = raw.resolve(strict=must_exist)
    except OSError as exc:
        raise ValueError(f"{name} is not a valid path") from exc
    if resolved != raw:
        raise ValueError(f"{name} must not traverse aliases or symbolic links")
    return resolved


def _validate_argv(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("argv must be a sequence of strings")
    argv = tuple(value)
    if not argv or len(argv) > _MAX_ARGUMENTS:
        raise ValueError("argv must contain between 1 and 256 arguments")
    total = 0
    for item in argv:
        if not isinstance(item, str):
            raise TypeError("argv must contain strings")
        if not item or len(item) > _MAX_ARGUMENT_SIZE:
            raise ValueError("argv contains an empty or oversized argument")
        if any(character in item for character in ("\x00", "\r", "\n")):
            raise ValueError("argv cannot contain control separators")
        total += len(item)
        if _looks_like_credential_option(item):
            raise ValueError("credential values must be supplied through environment")
        if _URL_USERINFO.match(item):
            raise ValueError("credential-bearing URLs are not allowed in argv")
    if total > _MAX_COMMAND_SIZE:
        raise ValueError("argv is too large")
    return argv


def _looks_like_credential_option(argument: str) -> bool:
    if not argument.startswith("-"):
        return False
    option = argument.split("=", 1)[0].casefold().replace("_", "-")
    if option in _CREDENTIAL_OPTIONS:
        return True
    words = tuple(word for word in option.lstrip("-").split("-") if word)
    compact = "".join(words)
    if any(word in _CREDENTIAL_WORDS for word in words):
        return True
    if compact in {"apikey", "privatekey", "keyfile", "authfile"}:
        return True
    pairs = set(zip(words, words[1:]))
    if pairs & {("api", "key"), ("private", "key"), ("key", "file"), ("auth", "file")}:
        return True
    if "token" in words:
        position = words.index("token")
        if len(words) == 1 or any(
            neighbor in _TOKEN_QUALIFIERS
            for neighbor in words[max(0, position - 1) : position + 2]
            if neighbor != "token"
        ):
            return True
    return compact.endswith("token") and compact[:-5] in _TOKEN_QUALIFIERS


def _job_environment(
    ambient: Mapping[str, str],
    declared: Mapping[str, str],
) -> dict[str, str]:
    """Build the explicit child environment without ambient credentials."""

    result = {
        name: value
        for name, value in ambient.items()
        if name in _SAFE_AMBIENT_ENVIRONMENT
        or any(name.startswith(prefix) for prefix in _SAFE_AMBIENT_PREFIXES)
    }
    result.update(declared)
    return result


def _validate_environment(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("environment must be a mapping")
    if len(value) > _MAX_ENVIRONMENT_ITEMS:
        raise ValueError("environment contains too many entries")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not _ENVIRONMENT_NAME.fullmatch(key):
            raise ValueError("environment contains an invalid name")
        if not isinstance(item, str):
            raise TypeError("environment values must be strings")
        if len(item) > _MAX_ENVIRONMENT_VALUE_SIZE or any(
            character in item for character in ("\x00", "\r", "\n")
        ):
            raise ValueError("environment contains an invalid value")
        result[key] = item
    return result


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_path(run_root: Path) -> Path:
    return run_root / CAMPAIGN_JOB_STATE_NAME


def _attempt_record(spec: CampaignJobSpec, attempt: int, now: str) -> dict[str, object]:
    log_path = spec.run_root / "logs" / f"attempt-{attempt:04d}.log"
    script_path = (
        spec.run_root / "scripts" / f"attempt-{attempt:04d}.sh"
        if spec.backend == "slurm"
        else None
    )
    return {
        "attempt": attempt,
        "status": "running" if spec.backend == "local" else "submitting",
        "log": str(log_path),
        "script": str(script_path) if script_path is not None else None,
        "job_id": None,
        "process_group_id": None,
        "return_code": None,
        "error_type": None,
        "created_at": now,
        "started_at": now if spec.backend == "local" else None,
        "finished_at": None,
        "updated_at": now,
    }


def _new_state(spec: CampaignJobSpec) -> dict[str, object]:
    now = _now()
    record = _attempt_record(spec, 1, now)
    state: dict[str, object] = {
        "schema": CAMPAIGN_JOB_STATE_SCHEMA,
        "controller": CAMPAIGN_JOB_CONTROLLER,
        "backend": spec.backend,
        "argv": list(spec.argv),
        "workdir": str(spec.workdir),
        "environment_keys": sorted(spec.environment),
        "resources": spec.resources.as_state(),
        "attempt": 1,
        "status": record["status"],
        "log": record["log"],
        "script": record["script"],
        "job_id": None,
        "process_group_id": None,
        "return_code": None,
        "error_type": None,
        "created_at": now,
        "started_at": record["started_at"],
        "finished_at": None,
        "updated_at": now,
        "recovery_scope": "whole_campaign_job_relaunch",
        "mid_round_recovery": False,
        "attempt_history": [record],
    }
    _validate_state(state, spec.run_root)
    return state


def _append_attempt(
    state: Mapping[str, object], spec: CampaignJobSpec
) -> dict[str, object]:
    result = _copy_json(state)
    attempt = int(result["attempt"]) + 1
    now = _now()
    record = _attempt_record(spec, attempt, now)
    history = list(result["attempt_history"])
    history.append(record)
    result["attempt_history"] = history
    for key in (
        "attempt",
        "status",
        "log",
        "script",
        "job_id",
        "process_group_id",
        "return_code",
        "error_type",
        "started_at",
        "finished_at",
        "updated_at",
    ):
        result[key] = record[key]
    _validate_state(result, spec.run_root)
    return result


def _finish_attempt(
    run_root: Path,
    attempt: int,
    *,
    status: str,
    return_code: int | None,
    error_type: str | None,
) -> dict[str, object]:
    if status not in {"completed", "failed", "interrupted"}:
        raise ValueError("invalid terminal status")
    with _locked_run(run_root):
        state = _read_state_unlocked(run_root)
        if int(state["attempt"]) != attempt:
            raise CampaignJobError("campaign job attempt changed unexpectedly")
        if state["status"] not in _ACTIVE_STATUSES:
            raise CampaignJobError("campaign job is no longer active")
        now = _now()
        record = dict(state["attempt_history"][-1])
        record.update(
            {
                "status": status,
                "return_code": return_code,
                "error_type": error_type,
                "finished_at": now,
                "updated_at": now,
            }
        )
        history = list(state["attempt_history"])
        history[-1] = record
        state["attempt_history"] = history
        for key in (
            "status",
            "return_code",
            "error_type",
            "finished_at",
            "updated_at",
        ):
            state[key] = record[key]
        _validate_state(state, run_root)
        _atomic_json_write(_state_path(run_root), state)
        return _copy_json(state)


def _record_active_error(
    run_root: Path,
    attempt: int,
    *,
    return_code: int | None,
    error_type: str,
) -> dict[str, object]:
    with _locked_run(run_root):
        state = _read_state_unlocked(run_root)
        if int(state["attempt"]) != attempt:
            raise CampaignJobError("campaign job attempt changed unexpectedly")
        if state["status"] not in _ACTIVE_STATUSES:
            raise CampaignJobError("campaign job is no longer active")
        state = _update_attempt_unlocked(
            state,
            status=str(state["status"]),
            return_code=return_code,
            error_type=error_type,
            finished=False,
        )
        _validate_state(state, run_root)
        _atomic_json_write(_state_path(run_root), state)
        return _copy_json(state)


def _record_local_process_group(
    run_root: Path,
    attempt: int,
    process_group_id: int,
) -> dict[str, object]:
    if (
        isinstance(process_group_id, bool)
        or not isinstance(process_group_id, int)
        or process_group_id < 1
    ):
        raise ValueError("local process_group_id must be a positive integer")
    with _locked_run(run_root):
        state = _read_state_unlocked(run_root)
        if (
            state["backend"] != "local"
            or int(state["attempt"]) != attempt
            or state["status"] != "running"
        ):
            raise CampaignJobError("local campaign attempt changed during startup")
        prior = state["process_group_id"]
        if prior is not None and prior != process_group_id:
            raise CampaignJobError("local campaign process group changed during startup")
        now = _now()
        record = dict(state["attempt_history"][-1])
        record["process_group_id"] = process_group_id
        record["updated_at"] = now
        history = list(state["attempt_history"])
        history[-1] = record
        state["attempt_history"] = history
        state["process_group_id"] = process_group_id
        state["updated_at"] = now
        _validate_state(state, run_root)
        _atomic_json_write(_state_path(run_root), state)
        return _copy_json(state)


def _record_submission(run_root: Path, attempt: int, job_id: str) -> dict[str, object]:
    with _locked_run(run_root):
        state = _read_state_unlocked(run_root)
        if int(state["attempt"]) != attempt:
            raise CampaignJobError("campaign job attempt changed during submission")
        prior_job_id = state["job_id"]
        if prior_job_id is not None and prior_job_id != job_id:
            raise CampaignJobError("scheduler job id changed during submission")
        now = _now()
        record = dict(state["attempt_history"][-1])
        record["job_id"] = job_id
        state["job_id"] = job_id
        if state["status"] == "submitting":
            record["status"] = "submitted"
            record["updated_at"] = now
            state["status"] = "submitted"
            state["updated_at"] = now
        history = list(state["attempt_history"])
        history[-1] = record
        state["attempt_history"] = history
        _validate_state(state, run_root)
        _atomic_json_write(_state_path(run_root), state)
        return _copy_json(state)


def _require_matching_spec(state: Mapping[str, object], spec: CampaignJobSpec) -> None:
    expected = {
        "backend": spec.backend,
        "argv": list(spec.argv),
        "workdir": str(spec.workdir),
        "environment_keys": sorted(spec.environment),
        "resources": spec.resources.as_state(),
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise CampaignJobError(f"resume spec does not match persisted {key}")


def _validate_state(state: Mapping[str, object], run_root: Path) -> None:
    required = {
        "schema",
        "controller",
        "backend",
        "argv",
        "workdir",
        "environment_keys",
        "resources",
        "attempt",
        "status",
        "log",
        "script",
        "job_id",
        "process_group_id",
        "return_code",
        "error_type",
        "created_at",
        "started_at",
        "finished_at",
        "updated_at",
        "recovery_scope",
        "mid_round_recovery",
        "attempt_history",
    }
    if set(state) != required:
        raise CampaignJobError("campaign job state has an invalid shape")
    if state["schema"] != CAMPAIGN_JOB_STATE_SCHEMA:
        raise CampaignJobError("campaign job state schema is unsupported")
    if state["controller"] != CAMPAIGN_JOB_CONTROLLER:
        raise CampaignJobError("campaign job state controller is invalid")
    if state["backend"] not in {"local", "slurm"}:
        raise CampaignJobError("campaign job state backend is invalid")
    if state["recovery_scope"] != "whole_campaign_job_relaunch":
        raise CampaignJobError("campaign job recovery scope is invalid")
    if state["mid_round_recovery"] is not False:
        raise CampaignJobError("campaign job cannot claim mid-round recovery")
    argv = state["argv"]
    try:
        validated_argv = _validate_argv(argv)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise CampaignJobError("campaign job state argv is invalid") from exc
    if list(validated_argv) != argv:
        raise CampaignJobError("campaign job state argv is invalid")
    workdir_value = state["workdir"]
    if not isinstance(workdir_value, str):
        raise CampaignJobError("campaign job state workdir is invalid")
    try:
        _canonical_path(Path(workdir_value), name="workdir", must_exist=True)
    except (TypeError, ValueError) as exc:
        raise CampaignJobError("campaign job state workdir is invalid") from exc
    environment_keys = state["environment_keys"]
    if (
        not isinstance(environment_keys, list)
        or environment_keys != sorted(environment_keys)
        or len(environment_keys) != len(set(environment_keys))
        or any(
            not isinstance(key, str) or not _ENVIRONMENT_NAME.fullmatch(key)
            for key in environment_keys
        )
    ):
        raise CampaignJobError("campaign job environment keys are invalid")
    if not _valid_resource_state(state["resources"]):
        raise CampaignJobError("campaign job resources are invalid")
    attempt = state["attempt"]
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise CampaignJobError("campaign job attempt is invalid")
    if state["status"] not in _ALL_STATUSES:
        raise CampaignJobError("campaign job status is invalid")
    history = state["attempt_history"]
    if not isinstance(history, list) or len(history) != attempt:
        raise CampaignJobError("campaign job attempt history is invalid")
    for index, item in enumerate(history, start=1):
        if not isinstance(item, Mapping) or item.get("attempt") != index:
            raise CampaignJobError("campaign job attempt history is invalid")
        _validate_attempt_record(item, run_root, str(state["backend"]))
        if index < attempt and item["status"] not in _RESUMABLE_STATUSES:
            raise CampaignJobError("campaign job prior attempt is not resumable")
    if state["created_at"] != history[0]["created_at"]:
        raise CampaignJobError("campaign job creation timestamp is inconsistent")
    current = history[-1]
    for key in (
        "attempt",
        "status",
        "log",
        "script",
        "job_id",
        "process_group_id",
        "return_code",
        "error_type",
        "started_at",
        "finished_at",
        "updated_at",
    ):
        if state[key] != current[key]:
            raise CampaignJobError("campaign job current attempt is inconsistent")


def _validate_attempt_record(
    record: Mapping[str, object], run_root: Path, backend: str
) -> None:
    expected = {
        "attempt",
        "status",
        "log",
        "script",
        "job_id",
        "process_group_id",
        "return_code",
        "error_type",
        "created_at",
        "started_at",
        "finished_at",
        "updated_at",
    }
    if set(record) != expected or record["status"] not in _ALL_STATUSES:
        raise CampaignJobError("campaign job attempt record is invalid")
    attempt = record["attempt"]
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise CampaignJobError("campaign job attempt record is invalid")
    log_path = _artifact_path(run_root, record["log"], "logs")
    if log_path.name != f"attempt-{attempt:04d}.log":
        raise CampaignJobError("campaign job log path is invalid")
    if backend == "local":
        if record["script"] is not None or record["job_id"] is not None:
            raise CampaignJobError("local campaign job record is invalid")
        process_group_id = record["process_group_id"]
        if process_group_id is not None and (
            isinstance(process_group_id, bool)
            or not isinstance(process_group_id, int)
            or process_group_id < 1
        ):
            raise CampaignJobError("local campaign process group is invalid")
    else:
        if record["process_group_id"] is not None:
            raise CampaignJobError("batch campaign job cannot have a local process group")
        script_path = _artifact_path(run_root, record["script"], "scripts")
        if script_path.name != f"attempt-{attempt:04d}.sh":
            raise CampaignJobError("campaign job script path is invalid")
        job_id = record["job_id"]
        if job_id is not None and (
            not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id)
        ):
            raise CampaignJobError("batch job id is invalid")
    return_code = record["return_code"]
    if return_code is not None and (
        isinstance(return_code, bool) or not isinstance(return_code, int)
    ):
        raise CampaignJobError("campaign job return code is invalid")
    error_type = record["error_type"]
    if error_type is not None and (
        not isinstance(error_type, str)
        or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,127}", error_type)
    ):
        raise CampaignJobError("campaign job error type is invalid")
    for key in ("created_at", "updated_at"):
        if not isinstance(record[key], str) or not record[key]:
            raise CampaignJobError("campaign job timestamp is invalid")
    for key in ("started_at", "finished_at"):
        if record[key] is not None and (
            not isinstance(record[key], str) or not record[key]
        ):
            raise CampaignJobError("campaign job timestamp is invalid")


def _valid_resource_state(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    try:
        resources = CampaignJobResources(**dict(value))
    except (TypeError, ValueError):
        return False
    return resources.as_state() == dict(value)


def _artifact_path(run_root: Path, value: object, directory: str) -> Path:
    if not isinstance(value, str):
        raise CampaignJobError("campaign job artifact path is invalid")
    path = Path(value)
    if not path.is_absolute() or path.parent != run_root / directory:
        raise CampaignJobError("campaign job artifact path escapes run_root")
    if path.name in {"", ".", ".."} or path.resolve(strict=False) != path:
        raise CampaignJobError("campaign job artifact path is not canonical")
    return path


@contextmanager
def _locked_run(run_root: Path, *, initialize: bool = False) -> Iterator[None]:
    lock_path = run_root / _LOCK_NAME
    flags = os.O_RDWR
    if initialize:
        flags |= os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileNotFoundError as exc:
        raise CampaignJobError("campaign job lock is unavailable") from exc
    except FileExistsError as exc:
        raise CampaignJobError("campaign job lock already exists") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CampaignJobError("campaign job lock is not a regular file")
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _read_state_unlocked(run_root: Path) -> dict[str, object]:
    path = _state_path(run_root)
    try:
        with path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignJobError("campaign job state is unavailable or invalid") from exc
    if not isinstance(state, dict):
        raise CampaignJobError("campaign job state must be an object")
    _validate_state(state, run_root)
    return state


def _atomic_json_write(path: Path, payload: Mapping[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_text_write(path: Path, content: str, *, mode: int) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_json(value: Mapping[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(value))


def _run_bounded_command(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    redacted_values: tuple[str, ...],
    log_path: Path,
    wall_seconds: int,
    on_started: Callable[[int], None] | None = None,
) -> int:
    process: subprocess.Popen[bytes] | None = None
    pump: Thread | None = None
    pump_errors: list[BaseException] = []
    pending_error: BaseException | None = None
    return_code: int | None = None
    gate_read = -1
    gate_write = -1
    with log_path.open("ab") as log:
        try:
            process_argv = argv
            process_options: dict[str, object] = {}
            if on_started is not None:
                gate_read, gate_write = os.pipe()
                process_argv = (
                    sys.executable,
                    "-c",
                    _GATED_EXEC_CODE,
                    str(gate_read),
                )
                process_options["pass_fds"] = (gate_read,)
            process = subprocess.Popen(
                process_argv,
                cwd=cwd,
                env=dict(environment),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=False,
                start_new_session=True,
                **process_options,
            )
            if on_started is not None:
                on_started(process.pid)
                payload = json.dumps(list(argv), separators=(",", ":")).encode(
                    "utf-8"
                )
                offset = 0
                while offset < len(payload):
                    offset += os.write(gate_write, payload[offset:])
                os.close(gate_write)
                gate_write = -1
                os.close(gate_read)
                gate_read = -1
            if process.stdout is None:
                raise CampaignJobExecutionError(
                    "campaign job output stream was not created"
                )
            pump = Thread(
                target=_pump_redacted_output,
                args=(process.stdout, log, redacted_values, pump_errors),
                name="campaign-job-log-pump",
                daemon=True,
            )
            pump.start()
            return_code = process.wait(timeout=wall_seconds)
        except subprocess.TimeoutExpired:
            cleanup_confirmed = (
                _terminate_process_group(process) if process is not None else True
            )
            pending_error = _CommandWallTimeout(
                process.returncode if process is not None else None,
                cleanup_confirmed=cleanup_confirmed,
            )
        except BaseException as exc:
            cleanup_confirmed = (
                _terminate_process_group(process) if process is not None else True
            )
            pending_error = (
                exc
                if cleanup_confirmed
                else _CommandCleanupUnconfirmed(
                    process.returncode if process is not None else None
                )
            )
        else:
            if process is not None and not _terminate_process_group(process):
                pending_error = _CommandCleanupUnconfirmed(return_code)
        finally:
            for descriptor in (gate_write, gate_read):
                if descriptor >= 0:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
            if process is not None and process.stdout is not None:
                pump_finished = _finish_output_pump(process.stdout, pump)
                if not pump_finished:
                    cleanup_confirmed = _terminate_process_group(process)
                    if isinstance(pending_error, _CommandWallTimeout):
                        pending_error.cleanup_confirmed = (
                            pending_error.cleanup_confirmed and cleanup_confirmed
                        )
                    elif pending_error is None:
                        pending_error = _CommandCleanupUnconfirmed(return_code)
        if pending_error is not None:
            raise pending_error
        if pump_errors:
            raise CampaignJobExecutionError("campaign job output could not be recorded")
        if return_code is None:
            raise CampaignJobExecutionError("campaign job return code is unavailable")
        return return_code


def _pump_redacted_output(
    stream: BinaryIO,
    log: BinaryIO,
    values: tuple[str, ...],
    errors: list[BaseException],
) -> None:
    patterns = tuple(
        sorted(
            {value.encode("utf-8") for value in values if value},
            key=len,
            reverse=True,
        )
    )
    try:
        if not patterns:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    return
                log.write(chunk)
                log.flush()
        pending = b""
        maximum = max(len(pattern) for pattern in patterns)
        while True:
            chunk = stream.read(65536)
            if not chunk:
                _write_redacted_prefix(log, pending, patterns, final=True, maximum=maximum)
                log.flush()
                return
            pending += chunk
            pending = _write_redacted_prefix(
                log, pending, patterns, final=False, maximum=maximum
            )
            log.flush()
    except BaseException as exc:
        errors.append(exc)


def _write_redacted_prefix(
    log: BinaryIO,
    pending: bytes,
    patterns: tuple[bytes, ...],
    *,
    final: bool,
    maximum: int,
) -> bytes:
    safe_start_limit = len(pending) if final else max(0, len(pending) - maximum + 1)
    index = 0
    output = bytearray()
    while index < safe_start_limit:
        match = next(
            (pattern for pattern in patterns if pending.startswith(pattern, index)),
            None,
        )
        if match is not None:
            index += len(match)
        else:
            output.append(pending[index])
            index += 1
    if final:
        while index < len(pending):
            match = next(
                (pattern for pattern in patterns if pending.startswith(pattern, index)),
                None,
            )
            if match is not None:
                index += len(match)
            else:
                output.append(pending[index])
                index += 1
    log.write(output)
    return pending[index:]


def _finish_output_pump(stream: BinaryIO, pump: Thread | None) -> bool:
    if pump is None:
        try:
            stream.close()
        except OSError:
            return False
        return True
    pump.join(timeout=_TERMINATION_GRACE_SECONDS)
    if pump.is_alive():
        try:
            stream.close()
        except OSError:
            return False
        pump.join(timeout=_TERMINATION_GRACE_SECONDS)
    else:
        try:
            stream.close()
        except OSError:
            return False
    return not pump.is_alive()


def _terminate_process_group(process: subprocess.Popen[bytes]) -> bool:
    process_group = process.pid
    if not _process_group_alive(process_group):
        return True
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    if _wait_for_process_group_exit(process_group, process):
        return True
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return _wait_for_process_group_exit(process_group, process)


def _process_group_alive(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(
    process_group: int, process: subprocess.Popen[bytes]
) -> bool:
    deadline = time.monotonic() + _TERMINATION_GRACE_SECONDS
    while _process_group_alive(process_group):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        if process.poll() is None:
            try:
                process.wait(timeout=min(0.05, remaining))
            except subprocess.TimeoutExpired:
                pass
        else:
            time.sleep(min(0.05, remaining))
    return True


def _slurm_time(wall_seconds: int) -> str:
    hours, remainder = divmod(wall_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _job_name(run_root: Path, attempt: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", run_root.name).strip("-")
    if not stem:
        stem = "campaign"
    return f"{stem[:80]}-a{attempt}"


def _render_slurm_script(run_root: Path, attempt: int) -> str:
    command = [
        sys.executable,
        "-m",
        "pde_agent_verifier_solver.engine.campaign_job",
        "_execute-slurm",
        str(run_root),
        str(attempt),
    ]
    return "#!/usr/bin/env bash\nset -euo pipefail\nexec " + shlex.join(command) + "\n"


def _sbatch_command(
    executable: str,
    spec: CampaignJobSpec,
    script_path: Path,
    attempt: int,
) -> tuple[str, ...]:
    state = _read_state_for_submission(spec.run_root, attempt)
    log_path = str(state["log"])
    exported_environment = _job_environment(os.environ, spec.environment)
    export_value = ",".join(sorted(exported_environment)) or "NONE"
    command = [
        executable,
        "--parsable",
        f"--export={export_value}",
        f"--job-name={_job_name(spec.run_root, attempt)}",
        f"--cpus-per-task={spec.resources.cpus}",
        f"--mem={spec.resources.memory}",
        f"--time={_slurm_time(spec.resources.wall_seconds)}",
        f"--output={log_path}",
        f"--error={log_path}",
    ]
    for flag, value in (
        ("--partition", spec.resources.partition),
        ("--gres", spec.resources.gres),
        ("--account", spec.resources.account),
        ("--qos", spec.resources.qos),
    ):
        if value:
            command.append(f"{flag}={value}")
    command.append(str(script_path))
    return tuple(command)


def _read_state_for_submission(run_root: Path, attempt: int) -> dict[str, object]:
    with _locked_run(run_root):
        state = _read_state_unlocked(run_root)
        if int(state["attempt"]) != attempt or state["status"] != "submitting":
            raise CampaignJobError("campaign job is no longer ready for submission")
        return state


def _execute_slurm_attempt(run_root: Path, attempt: int) -> int:
    root = _canonical_path(run_root, name="run_root", must_exist=True)
    with _locked_run(root):
        state = _read_state_unlocked(root)
        if state["backend"] != "slurm" or int(state["attempt"]) != attempt:
            raise CampaignJobError("batch worker does not match campaign attempt")
        if state["status"] not in {"submitting", "submitted"}:
            raise CampaignJobActiveError("batch worker attempt is not launchable")
        missing = [key for key in state["environment_keys"] if key not in os.environ]
        if missing:
            # The names are safe to persist; values are never read from state.
            state = _update_attempt_unlocked(
                state,
                status="failed",
                return_code=None,
                error_type="MissingDeclaredEnvironment",
                finished=True,
            )
            _atomic_json_write(_state_path(root), state)
            return 78
        scheduler_job_id = os.environ.get("SLURM_JOB_ID")
        if scheduler_job_id is not None and not _JOB_ID.fullmatch(scheduler_job_id):
            state = _update_attempt_unlocked(
                state,
                status="failed",
                return_code=None,
                error_type="InvalidSchedulerJobId",
                finished=True,
            )
            _atomic_json_write(_state_path(root), state)
            return 78
        if state["job_id"] is not None and (
            scheduler_job_id is not None and state["job_id"] != scheduler_job_id
        ):
            state = _update_attempt_unlocked(
                state,
                status="failed",
                return_code=None,
                error_type="SchedulerJobIdMismatch",
                finished=True,
            )
            _atomic_json_write(_state_path(root), state)
            return 78
        state = _update_attempt_unlocked(
            state,
            status="running",
            return_code=None,
            error_type=None,
            finished=False,
            job_id=scheduler_job_id or state["job_id"],
            start=True,
        )
        _atomic_json_write(_state_path(root), state)
        argv = tuple(state["argv"])
        workdir = Path(str(state["workdir"]))
        wall_seconds = int(state["resources"]["wall_seconds"])
        log_path = _artifact_path(root, state["log"], "logs")
        declared_values = tuple(os.environ[key] for key in state["environment_keys"])

    try:
        declared_environment = {
            key: os.environ[key] for key in state["environment_keys"]
        }
        return_code = _run_bounded_command(
            argv,
            cwd=workdir,
            environment=_job_environment(os.environ, declared_environment),
            redacted_values=declared_values,
            log_path=log_path,
            wall_seconds=wall_seconds,
        )
    except _CommandWallTimeout as exc:
        if exc.cleanup_confirmed:
            _finish_attempt(
                root,
                attempt,
                status="interrupted",
                return_code=exc.return_code,
                error_type="WallClockTimeout",
            )
        else:
            _record_active_error(
                root,
                attempt,
                return_code=exc.return_code,
                error_type="WallClockTimeoutCleanupUnconfirmed",
            )
        return 124
    except _CommandCleanupUnconfirmed as exc:
        _record_active_error(
            root,
            attempt,
            return_code=exc.return_code,
            error_type="ProcessGroupCleanupUnconfirmed",
        )
        return 70
    except (KeyboardInterrupt, SystemExit):
        _finish_attempt(
            root,
            attempt,
            status="interrupted",
            return_code=None,
            error_type="WorkerInterrupted",
        )
        raise
    except Exception as exc:
        _finish_attempt(
            root,
            attempt,
            status="failed",
            return_code=None,
            error_type=type(exc).__name__,
        )
        return 70

    status = "completed" if return_code == 0 else "failed"
    _finish_attempt(
        root,
        attempt,
        status=status,
        return_code=return_code,
        error_type=None if return_code == 0 else "NonzeroExit",
    )
    return return_code


def _update_attempt_unlocked(
    state: dict[str, object],
    *,
    status: str,
    return_code: int | None,
    error_type: str | None,
    finished: bool,
    job_id: object = None,
    start: bool = False,
) -> dict[str, object]:
    now = _now()
    record = dict(state["attempt_history"][-1])
    record.update(
        {
            "status": status,
            "return_code": return_code,
            "error_type": error_type,
            "updated_at": now,
        }
    )
    if start:
        record["started_at"] = now
    if finished:
        record["finished_at"] = now
    if job_id is not None:
        record["job_id"] = job_id
    history = list(state["attempt_history"])
    history[-1] = record
    state["attempt_history"] = history
    for key in (
        "status",
        "return_code",
        "error_type",
        "updated_at",
        "started_at",
        "finished_at",
        "job_id",
        "process_group_id",
    ):
        state[key] = record[key]
    return state


def _worker_main(argv: Sequence[str]) -> int:
    if len(argv) != 3 or argv[0] != "_execute-slurm":
        raise CampaignJobError("invalid campaign job worker invocation")
    try:
        attempt = int(argv[2])
    except ValueError:
        raise CampaignJobError("invalid campaign job worker attempt") from None
    if attempt < 1:
        raise CampaignJobError("invalid campaign job worker attempt")
    return _execute_slurm_attempt(Path(argv[1]), attempt)


if __name__ == "__main__":
    try:
        raise SystemExit(_worker_main(sys.argv[1:]))
    except CampaignJobError:
        # Do not echo state, argv, scheduler output, or ambient values.
        raise SystemExit(70) from None
