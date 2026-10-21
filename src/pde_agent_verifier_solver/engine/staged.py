"""Persistent scheduling for explicit legacy/control-arm campaign replay.

Canonical product runs use :class:`CampaignController`; callers must opt in at
each execution boundary before these historical local or Slurm stages run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from typing import Mapping


@dataclass(frozen=True)
class StageResources:
    cpus: int = 4
    memory: str = "16G"
    time_limit: str = "01:00:00"
    partition: str = ""
    gres: str = ""


@dataclass(frozen=True)
class StageCommand:
    argv: tuple[str, ...]
    label: str = ""
    environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.argv:
            raise ValueError("a stage command cannot be empty")


@dataclass(frozen=True)
class RunStage:
    stage_id: str
    title: str
    commands: tuple[StageCommand, ...]
    depends_on: tuple[str, ...] = ()
    resources: StageResources = field(default_factory=StageResources)
    expected_outputs: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if not self.stage_id or not self.stage_id.replace("_", "").isalnum():
            raise ValueError(f"invalid stage id: {self.stage_id!r}")
        if not self.commands:
            raise ValueError(f"stage {self.stage_id!r} has no commands")


@dataclass(frozen=True)
class StagedRunPlan:
    name: str
    problem: str
    run_root: Path
    workdir: Path
    product_root: Path
    implementation_root: Path
    stages: tuple[RunStage, ...]
    task: str = ""
    modules: tuple[str, ...] = ()
    account: str = ""
    qos: str = ""
    result_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("run name cannot be empty")
        if not self.stages:
            raise ValueError("a staged run needs at least one stage")
        ids = [stage.stage_id for stage in self.stages]
        if len(ids) != len(set(ids)):
            raise ValueError("stage ids must be unique")
        known: set[str] = set()
        for stage in self.stages:
            missing = set(stage.depends_on) - known
            if missing:
                raise ValueError(
                    f"stage {stage.stage_id!r} depends on stages that do not precede it: "
                    f"{sorted(missing)}"
                )
            known.add(stage.stage_id)

    def summary(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "problem": self.problem,
            "task": self.task or None,
            "run_root": str(self.run_root),
            "stages": [
                {
                    "id": stage.stage_id,
                    "title": stage.title,
                    "depends_on": list(stage.depends_on),
                    "commands": [list(command.argv) for command in stage.commands],
                    "expected_outputs": [str(path) for path in stage.expected_outputs],
                    "resources": {
                        "cpus": stage.resources.cpus,
                        "memory": stage.resources.memory,
                        "time_limit": stage.resources.time_limit,
                        "partition": stage.resources.partition,
                        "gres": stage.resources.gres,
                    },
                }
                for stage in self.stages
            ],
            "result_path": str(self.result_path) if self.result_path else None,
        }


class LocalStageRunner:
    """Execute an explicitly requested legacy/control-arm staged campaign."""

    def run(
        self,
        plan: StagedRunPlan,
        *,
        legacy_staged_intent: bool = False,
    ) -> Mapping[str, object]:
        _require_legacy_staged_intent(legacy_staged_intent)
        _prepare_run_root(plan, backend="local")
        environment = _stage_environment(plan)
        for stage in plan.stages:
            record_stage_event(plan.run_root, stage.stage_id, "running")
            log_path = _log_path(plan.run_root, stage.stage_id)
            try:
                with log_path.open("a", encoding="utf-8") as log:
                    for command in stage.commands:
                        command_environment = dict(environment)
                        command_environment.update(command.environment)
                        process = subprocess.Popen(
                            command.argv,
                            cwd=plan.workdir,
                            env=command_environment,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                        )
                        assert process.stdout is not None
                        with process.stdout:
                            for line in process.stdout:
                                log.write(line)
                                log.flush()
                                print(line, end="", flush=True)
                        return_code = process.wait()
                        if return_code:
                            raise subprocess.CalledProcessError(return_code, command.argv)
                _require_outputs(stage)
            except Exception as exc:
                record_stage_event(
                    plan.run_root,
                    stage.stage_id,
                    "failed",
                    message=f"{type(exc).__name__}: {exc}",
                )
                raise
            record_stage_event(plan.run_root, stage.stage_id, "completed")
        return read_staged_run(plan.run_root)


class SlurmStageLauncher:
    """Submit an explicitly requested legacy/control-arm staged campaign."""

    def launch(
        self,
        plan: StagedRunPlan,
        *,
        legacy_staged_intent: bool = False,
    ) -> Mapping[str, object]:
        _require_legacy_staged_intent(legacy_staged_intent)
        if shutil.which("sbatch") is None:
            raise RuntimeError("Slurm backend requested, but sbatch is not available")
        _prepare_run_root(plan, backend="slurm")
        job_ids: dict[str, str] = {}
        scripts_root = plan.run_root / "stages"
        for stage in plan.stages:
            script_path = scripts_root / f"{stage.stage_id}.sh"
            script_path.write_text(_render_slurm_script(plan, stage), encoding="utf-8")
            script_path.chmod(0o750)
            dependencies = [job_ids[item] for item in stage.depends_on]
            job_ids[stage.stage_id] = _submit_slurm_stage(
                plan, stage, script_path, dependencies
            )
        _atomic_json_write(
            plan.run_root / "jobs.json",
            {"submitted_at": _now(), "jobs": job_ids},
        )
        return {"run_root": str(plan.run_root), "jobs": job_ids}

    def resume(
        self,
        plan: StagedRunPlan,
        *,
        legacy_staged_intent: bool = False,
    ) -> Mapping[str, object]:
        """Continue the first incomplete stage of an existing Slurm campaign.

        Completion is decided from the scientific outputs declared by each
        stage. This supports a changed stage layout (for example, splitting one
        long five-round job into five one-round jobs) without rerunning finished
        projections, Generator training, or completed rounds.
        """

        _require_legacy_staged_intent(legacy_staged_intent)
        if shutil.which("sbatch") is None:
            raise RuntimeError("Slurm backend requested, but sbatch is not available")
        root = plan.run_root.resolve()
        run_path = root / "run.json"
        if not run_path.is_file():
            raise FileNotFoundError(f"campaign does not exist: {root}")
        existing = _read_json(run_path)
        if str(existing.get("backend")) != "slurm":
            raise ValueError("only Slurm campaigns can be resumed")
        if (
            str(existing.get("name")) != plan.name
            or str(existing.get("problem")) != plan.problem
        ):
            raise ValueError("the existing run does not match this campaign config")

        prior_jobs = _read_json(root / "jobs.json", default={}).get("jobs", {})
        if not isinstance(prior_jobs, Mapping):
            prior_jobs = {}
        active_jobs = _active_slurm_job_ids(
            tuple(str(job_id) for job_id in prior_jobs.values() if job_id)
        )
        if active_jobs:
            joined = ", ".join(sorted(active_jobs))
            raise RuntimeError(
                "cannot resume while this campaign still has active Slurm jobs: "
                f"{joined}"
            )

        for directory in (root / "stages", root / "runtime", root / "logs"):
            directory.mkdir(parents=True, exist_ok=True)
        payload = dict(plan.summary())
        payload.update(
            {
                "backend": "slurm",
                "created_at": str(existing.get("created_at") or _now()),
            }
        )
        _atomic_json_write(run_path, payload)

        completed_stages: set[str] = set()
        skipped_stages: list[str] = []
        job_ids: dict[str, str] = {}
        scripts_root = root / "stages"
        for stage in plan.stages:
            dependencies_ready = all(
                dependency in completed_stages for dependency in stage.depends_on
            )
            if dependencies_ready and _stage_outputs_exist(stage):
                record_stage_event(root, stage.stage_id, "completed")
                completed_stages.add(stage.stage_id)
                skipped_stages.append(stage.stage_id)
                continue

            unresolved = [
                dependency
                for dependency in stage.depends_on
                if dependency not in completed_stages and dependency not in job_ids
            ]
            if unresolved:
                raise RuntimeError(
                    f"stage {stage.stage_id} has unresolved dependencies: {unresolved}"
                )
            script_path = scripts_root / f"{stage.stage_id}.sh"
            script_path.write_text(_render_slurm_script(plan, stage), encoding="utf-8")
            script_path.chmod(0o750)
            dependencies = [
                job_ids[dependency]
                for dependency in stage.depends_on
                if dependency in job_ids
            ]
            job_ids[stage.stage_id] = _submit_slurm_stage(
                plan, stage, script_path, dependencies
            )

        _atomic_json_write(
            root / "jobs.json",
            {"submitted_at": _now(), "jobs": job_ids},
        )
        return {
            "run_root": str(root),
            "jobs": job_ids,
            "reused_stages": skipped_stages,
        }


def _require_legacy_staged_intent(value: bool) -> None:
    if value is not True:
        raise PermissionError(
            "staged execution is disabled for product runs; use "
            "CampaignController. Pass legacy_staged_intent=True only for an "
            "explicit history/control-arm reproduction."
        )


def record_stage_event(
    run_root: Path | str,
    stage_id: str,
    status: str,
    *,
    message: str = "",
    exit_code: int | None = None,
) -> None:
    if status not in {"submitted", "running", "completed", "failed"}:
        raise ValueError(f"unknown stage status: {status}")
    root = Path(run_root).resolve()
    path = root / "runtime" / f"{stage_id}.json"
    prior = _read_json(path, default={})
    payload = dict(prior)
    payload.update({"stage_id": stage_id, "status": status, "updated_at": _now()})
    if status == "running" and "started_at" not in payload:
        payload["started_at"] = payload["updated_at"]
    if status in {"completed", "failed"}:
        payload["finished_at"] = payload["updated_at"]
    if message:
        payload["message"] = message
    if exit_code is not None:
        payload["exit_code"] = int(exit_code)
    _atomic_json_write(path, payload)


def read_staged_run(run_root: Path | str, *, log_lines: int = 30) -> Mapping[str, object]:
    root = Path(run_root).resolve()
    run = _read_json(root / "run.json")
    jobs = _read_json(root / "jobs.json", default={}).get("jobs", {})
    detail_log = _latest_detail_log(root)
    stages = []
    statuses = []
    for row in run["stages"]:
        stage_id = str(row["id"])
        runtime = _read_json(root / "runtime" / f"{stage_id}.json", default={})
        status = str(runtime.get("status", "submitted"))
        statuses.append(status)
        stages.append(
            {
                **row,
                **runtime,
                "status": status,
                "job_id": jobs.get(stage_id),
                "log_tail": _tail(_log_path(root, stage_id), log_lines),
                "error_tail": _tail(_error_log_path(root, stage_id), log_lines),
                "detail_log": (
                    str(detail_log.relative_to(root))
                    if status == "running" and detail_log is not None
                    else None
                ),
                "detail_tail": (
                    _tail(detail_log, log_lines)
                    if status == "running" and detail_log is not None
                    else []
                ),
            }
        )
    if any(status == "failed" for status in statuses):
        status = "failed"
    elif statuses and all(status == "completed" for status in statuses):
        status = "completed"
    elif any(status == "running" for status in statuses):
        status = "running"
    else:
        status = "submitted"
    result_path = run.get("result_path")
    result = _read_json(Path(result_path), default=None) if result_path else None
    return {
        "name": run["name"],
        "problem": run["problem"],
        "task": run.get("task"),
        "run_root": str(root),
        "backend": run["backend"],
        "status": status,
        "stages": stages,
        "result": result,
    }


def _prepare_run_root(plan: StagedRunPlan, *, backend: str) -> None:
    root = plan.run_root.resolve()
    if root.exists():
        raise FileExistsError(f"run output already exists: {root}")
    (root / "stages").mkdir(parents=True)
    (root / "runtime").mkdir()
    (root / "logs").mkdir()
    payload = dict(plan.summary())
    payload.update({"backend": backend, "created_at": _now()})
    _atomic_json_write(root / "run.json", payload)


def _render_slurm_script(plan: StagedRunPlan, stage: RunStage) -> str:
    product_pythonpath = os.pathsep.join(
        (str(plan.product_root / "src"), str(plan.implementation_root / "src"))
    )
    event_base = [
        "python", "-m", "pde_agent_verifier_solver", "_stage-event",
        "--run-root", str(plan.run_root), "--stage", stage.stage_id,
    ]
    lines = [
        "#!/usr/bin/env bash", "source /etc/profile", "set -euo pipefail",
        f"export PYTHONPATH={shlex.quote(product_pythonpath)}:\"${{PYTHONPATH:-}}\"",
        "finish_stage() {", "  rc=$?",
        "  trap - EXIT",
        "  if [[ ${rc} -eq 0 ]]; then stage_status=completed; else stage_status=failed; fi",
        f"  {shlex.join(event_base)} --status \"${{stage_status}}\" --exit-code \"${{rc}}\"",
        "  exit \"${rc}\"", "}", "trap finish_stage EXIT",
    ]
    if plan.modules:
        lines.extend(("module -q purge", f"module load {shlex.join(plan.modules)}"))
    lines.extend(
        (
            f"export PYTHONPYCACHEPREFIX=\"${{SLURM_TMPDIR:-{str(plan.run_root / 'tmp')}}}/pycache\"",
            f"export OMP_NUM_THREADS={stage.resources.cpus}",
            f"export MKL_NUM_THREADS={stage.resources.cpus}",
            f"mkdir -p {shlex.quote(str(plan.run_root / 'tmp'))}",
            f"{shlex.join(event_base)} --status running",
            f"cd {shlex.quote(str(plan.workdir))}",
        )
    )
    for command in stage.commands:
        prefix = " ".join(
            f"{name}={shlex.quote(value)}" for name, value in command.environment.items()
        )
        lines.append(f"{prefix} {shlex.join(command.argv)}".strip())
    for output in stage.expected_outputs:
        quoted = shlex.quote(str(output))
        lines.append(
            f"[[ -e {quoted} ]] || {{ echo 'missing expected output: {quoted}' >&2; exit 3; }}"
        )
    return "\n".join(lines) + "\n"


def _submit_slurm_stage(
    plan: StagedRunPlan,
    stage: RunStage,
    script_path: Path,
    dependencies: list[str],
) -> str:
    command = ["sbatch", "--parsable"]
    if plan.account:
        command.extend(("--account", plan.account))
    if plan.qos:
        command.extend(("--qos", plan.qos))
    resources = stage.resources
    if resources.partition:
        command.extend(("--partition", resources.partition))
    if resources.gres:
        command.extend(("--gres", resources.gres))
    command.extend(
        (
            "--cpus-per-task", str(resources.cpus),
            "--mem", resources.memory,
            "--time", resources.time_limit,
            "--job-name", _job_name(plan, stage),
            "--output", str(plan.run_root / "logs" / f"{stage.stage_id}_%j.out"),
            "--error", str(plan.run_root / "logs" / f"{stage.stage_id}_%j.err"),
        )
    )
    if dependencies:
        command.extend(("--dependency", f"afterok:{':'.join(dependencies)}"))
    command.append(str(script_path))
    completed = subprocess.run(
        command,
        cwd=plan.product_root,
        check=True,
        capture_output=True,
        text=True,
    )
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id:
        raise RuntimeError(f"sbatch returned no job id for {stage.stage_id}")
    return job_id


def _active_slurm_job_ids(job_ids: tuple[str, ...]) -> set[str]:
    if not job_ids:
        return set()
    completed = subprocess.run(
        ["squeue", "--noheader", "--jobs", ",".join(job_ids), "--format=%A"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {line.strip() for line in completed.stdout.splitlines() if line.strip()}


def _stage_outputs_exist(stage: RunStage) -> bool:
    return bool(stage.expected_outputs) and all(
        output.exists() for output in stage.expected_outputs
    )


def _stage_environment(plan: StagedRunPlan) -> dict[str, str]:
    environment = dict(os.environ)
    additions = os.pathsep.join(
        (str(plan.product_root / "src"), str(plan.implementation_root / "src"))
    )
    current = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = additions + (os.pathsep + current if current else "")
    return environment


def _require_outputs(stage: RunStage) -> None:
    missing = [str(path) for path in stage.expected_outputs if not path.exists()]
    if missing:
        raise RuntimeError(
            f"stage {stage.stage_id} completed without expected outputs: {missing}"
        )


def _job_name(plan: StagedRunPlan, stage: RunStage) -> str:
    stem = "".join(character if character.isalnum() else "_" for character in plan.name)
    return f"pde_{stem[:18]}_{stage.stage_id[:16]}"


def _log_path(run_root: Path, stage_id: str) -> Path:
    direct = run_root / "logs" / f"{stage_id}.log"
    if direct.exists():
        return direct
    candidates = sorted((run_root / "logs").glob(f"{stage_id}_*.out"))
    return candidates[-1] if candidates else direct


def _latest_detail_log(run_root: Path) -> Path | None:
    patterns = (
        "campaign/*/*.log",
        "campaign/*/rounds/*/logs/*.log",
        "campaign/*/rounds/*/*/logs/*.log",
    )
    candidates = [
        path
        for pattern in patterns
        for path in run_root.glob(pattern)
        if path.is_file()
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _error_log_path(run_root: Path, stage_id: str) -> Path:
    direct = run_root / "logs" / f"{stage_id}.err"
    if direct.exists():
        return direct
    candidates = sorted((run_root / "logs").glob(f"{stage_id}_*.err"))
    return candidates[-1] if candidates else direct


def _tail(path: Path, line_count: int) -> list[str]:
    if not path.is_file() or line_count <= 0:
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.readlines()[-line_count:]


def _read_json(path: Path, *, default=...):
    if not path.is_file():
        if default is ...:
            raise FileNotFoundError(path)
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
