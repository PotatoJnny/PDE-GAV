#!/usr/bin/env python
"""Run one authorized Max Agent with VPFP online generator and evaluator services."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_verifier.storage_lifecycle import (
    SlurmEphemeralScope,
    sha256_file,
)
from agent_verifier.vpfp.development_evaluation import (
    FINALIST_PURPOSE,
    VPFP_DEVELOPMENT_CASCADE,
    VPFP_DEVELOPMENT_REQUEST_SCHEMA,
)
from agent_verifier.vpfp.external_launch_v2 import (
    load_and_validate_vpfp_robust_external_launch_packet,
)
from agent_verifier.vpfp.oracle_training import (
    inspect_solver_bundle,
    stage_solver_bundle,
)


REPORT_SCHEMA = "vpfp_long_time_robust_agent_worker_v2"
PREPARATION_SCHEMA = "vpfp_long_time_robust_two_max_preparation_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--agent-index", type=int)
    parser.add_argument("--agent-timeout-seconds", type=int, default=28800)
    parser.add_argument("--finalist-timeout-seconds", type=int, default=7200)
    parser.add_argument("--mock-agent", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_runtime(args)
    campaign_root = Path(args.campaign_root).resolve()
    preparation = _read_json(campaign_root / "campaign_preparation.json")
    if (
        preparation.get("schema") != PREPARATION_SCHEMA
        or preparation.get("passed") is not True
        or preparation.get("user_authorized_external_dispatch") is not True
    ):
        raise SystemExit("VPFP campaign preparation is not launchable")
    index = _agent_index(args)
    row = dict(preparation["agents"][index])
    workspace = Path(row["workspace"]).resolve()
    private = Path(row["controller_private_root"]).resolve()
    packet = load_and_validate_vpfp_robust_external_launch_packet(
        row["launch_packet"],
        workspace=workspace,
        require_workspace_unchanged=True,
    )
    if packet["agent_id"] != row["agent_id"]:
        raise SystemExit("VPFP Agent launch identity changed")
    run_root = private / "agent_run"
    if run_root.exists():
        raise FileExistsError(run_root)
    run_root.mkdir()
    service_root = run_root / "services"
    service_root.mkdir()
    ephemeral_scope = SlurmEphemeralScope.create(
        f"vpfp-robust-{row['agent_id']}"
    )
    (ephemeral_scope.root / "private").mkdir()
    online_public_root = _stage_online_numerical_views(
        ephemeral_scope=ephemeral_scope,
        workspace=workspace,
    )
    runtime_config_root = ephemeral_scope.root / "configs"
    runtime_config_root.mkdir()
    online_private_root = ephemeral_scope.root / "private" / "online_runtime"
    evaluator_private_root = ephemeral_scope.root / "private" / "evaluator_runtime"
    online_config_path = _write_runtime_service_config(
        Path(row["online_generator_config"]).resolve(),
        runtime_config_root / "online.json",
        private_root=online_private_root,
    )
    evaluator_config_path = _write_runtime_service_config(
        Path(row["development_evaluator_config"]).resolve(),
        runtime_config_root / "evaluator.json",
        private_root=evaluator_private_root,
    )
    socket_root = ephemeral_scope.root / "sockets"
    socket_root.mkdir(mode=0o700)
    socket_path = socket_root / "generator.sock"
    evaluator_stop = ephemeral_scope.root / "evaluator.stop"

    online_process = None
    evaluator_process = None
    handles: list[Any] = []
    failure = None
    finalist_response = None
    invocation = None
    final_bundle = None
    started = time.time()
    try:
        online_process, online_handles = _start_online_service(
            config_path=online_config_path,
            socket_path=socket_path,
            output_root=service_root / "online",
        )
        handles.extend(online_handles)
        evaluator_process, evaluator_handles = _start_evaluator_service(
            config_path=evaluator_config_path,
            stop_file=evaluator_stop,
            output_root=service_root / "evaluator",
        )
        handles.extend(evaluator_handles)
        evaluator_config = _read_json(evaluator_config_path)
        evaluator_status = (
            Path(evaluator_config["private_root"]).resolve()
            / "service_status.json"
        )
        _wait_until(
            lambda: evaluator_status.is_file()
            and _read_json(evaluator_status).get("state") == "ready",
            timeout_seconds=600.0,
            description="VPFP evaluator readiness",
            processes=(online_process, evaluator_process),
        )

        invocation = _invoke_agent(
            workspace=workspace,
            run_root=run_root,
            packet=packet,
            socket_path=socket_path,
            timeout_seconds=int(args.agent_timeout_seconds),
            mock_agent=bool(args.mock_agent),
        )
        if not args.mock_agent:
            if invocation.get("actual_codex_invoked") is not True:
                raise RuntimeError("external VPFP Agent was not invoked")
            returncode = int(invocation["returncode"])
            if returncode not in {0, 124, 137}:
                raise RuntimeError(
                    "external VPFP Agent runner failed with status "
                    f"{returncode}"
                )
        candidate = workspace / "submission" / "submission.py"
        bundle = inspect_solver_bundle(candidate)
        final_bundle = stage_solver_bundle(
            candidate,
            private / "frozen_final_bundle",
        )
        finalist_request_id = _submit_finalist_request(
            workspace=workspace,
            candidate_entrypoint="submission/submission.py",
            agent_id=str(row["agent_id"]),
        )
        finalist_response_path = (
            workspace
            / "evaluation_exchange"
            / "responses"
            / f"{finalist_request_id}.json"
        )
        _wait_until(
            lambda: finalist_response_path.is_file(),
            timeout_seconds=float(args.finalist_timeout_seconds),
            description="VPFP finalist compatibility response",
            processes=(online_process, evaluator_process),
        )
        finalist_response = _read_json(finalist_response_path)
        compatibility = dict(
            finalist_response.get("result", {}).get("finalist_compatibility", {})
        )
        if (
            finalist_response.get("status") != "completed"
            or compatibility.get("passed") is not True
            or compatibility.get("candidate_sha256")
            != bundle["aggregate_sha256"]
        ):
            raise RuntimeError("VPFP final bundle failed compatibility evaluation")
    except BaseException as exc:
        failure = {
            "type": type(exc).__name__,
            "message": str(exc)[:4000],
        }
    finally:
        evaluator_stop.write_text("stop\n", encoding="utf-8")
        service_shutdown = {
            "evaluator": _stop_process(
                evaluator_process,
                graceful_timeout=7200.0,
            ),
            "online": _stop_process(
                online_process,
                graceful_timeout=300.0,
            ),
        }
        for handle in handles:
            handle.close()
        storage_lifecycle = _finalize_storage_lifecycle(
            ephemeral_scope=ephemeral_scope,
            online_private_root=online_private_root,
            evaluator_private_root=evaluator_private_root,
            online_public_root=online_public_root,
            workspace=workspace,
            service_root=service_root,
        )
        if storage_lifecycle["passed"] is not True and failure is None:
            failure = {
                "type": "StorageLifecycleError",
                "message": "; ".join(storage_lifecycle["failures"])[:4000],
            }

    report = {
        "schema": REPORT_SCHEMA,
        "passed": bool(
            failure is None
            and final_bundle is not None
            and finalist_response is not None
            and finalist_response.get("status") == "completed"
        ),
        "agent_index": index,
        "agent_id": row["agent_id"],
        "model": packet["model"],
        "reasoning_effort": packet["reasoning_effort"],
        "actual_external_agent_invoked": not bool(args.mock_agent),
        "user_authorized_external_dispatch": (
            packet["user_authorized_external_dispatch"]
        ),
        "dangerous_sandbox_bypass": packet["dangerous_sandbox_bypass"],
        "network_allowed": packet["network_allowed"],
        "workspace": str(workspace),
        "controller_private_root": str(private),
        "invocation": invocation,
        "final_bundle": None if final_bundle is None else str(final_bundle),
        "final_bundle_sha256": (
            None
            if final_bundle is None
            else inspect_solver_bundle(final_bundle)["aggregate_sha256"]
        ),
        "finalist_response": finalist_response,
        "service_shutdown": service_shutdown,
        "storage_lifecycle": storage_lifecycle,
        "failure": failure,
        "wall_time_seconds": float(time.time() - started),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    _write_json(run_root / "worker_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["passed"]:
        raise SystemExit("VPFP robust Agent worker did not pass")


def _write_runtime_service_config(
    source: Path,
    destination: Path,
    *,
    private_root: Path,
) -> Path:
    payload = _read_json(source)
    payload["private_root"] = str(private_root.resolve())
    payload["storage_lifecycle"] = {
        "schema": "agent_verifier_ephemeral_service_runtime_v1",
        "source_config_sha256": sha256_file(source),
        "dense_runtime_is_ephemeral": True,
        "runtime_root": str(private_root.resolve()),
    }
    _write_json(destination, payload)
    return destination.resolve()


def _finalize_storage_lifecycle(
    *,
    ephemeral_scope: SlurmEphemeralScope,
    online_private_root: Path,
    evaluator_private_root: Path,
    online_public_root: Path,
    workspace: Path,
    service_root: Path,
) -> dict[str, Any]:
    failures: list[str] = []
    archived: dict[str, Any] = {}
    for name, source in (
        ("online", online_private_root),
        ("evaluator", evaluator_private_root),
    ):
        try:
            archived[name] = _archive_service_runtime(
                source,
                service_root / name / "runtime_summary",
                service=name,
            )
        except BaseException as exc:
            failures.append(f"{name}_archive:{type(exc).__name__}:{exc}")
            archived[name] = {
                "archived": False,
                "failure": f"{type(exc).__name__}: {exc}",
            }

    try:
        public_cleanup = _cleanup_online_numerical_views(
            workspace,
            service_root / "online" / "public_summary",
            expected_ephemeral_root=online_public_root,
        )
    except BaseException as exc:
        failures.append(
            f"online_public_cleanup:{type(exc).__name__}:{exc}"
        )
        public_cleanup = {
            "removed": False,
            "failure": f"{type(exc).__name__}: {exc}",
        }

    try:
        ephemeral_cleanup = ephemeral_scope.cleanup()
    except BaseException as exc:
        failures.append(
            f"ephemeral_scope_cleanup:{type(exc).__name__}:{exc}"
        )
        ephemeral_cleanup = {
            "removed": False,
            "failure": f"{type(exc).__name__}: {exc}",
            "root": str(ephemeral_scope.root),
        }
    return {
        "schema": "agent_verifier_worker_storage_lifecycle_v1",
        "passed": not failures,
        "dense_runtime_location": "SLURM_TMPDIR",
        "public_numerical_view_location": "SLURM_TMPDIR",
        "controller_dense_arrays_persisted": False,
        "compact_runtime_archives": archived,
        "online_public_numerical_views": public_cleanup,
        "ephemeral_scope_cleanup": ephemeral_cleanup,
        "failures": failures,
    }


def _stage_online_numerical_views(
    *,
    ephemeral_scope: SlurmEphemeralScope,
    workspace: Path,
) -> Path:
    """Expose one node-local public view directory through the stable CLI path."""

    alias = workspace / "online_generator_outputs"
    if alias.exists() or alias.is_symlink():
        raise FileExistsError(alias)
    physical_root = ephemeral_scope.require_contains(
        ephemeral_scope.root / "public" / "online_generator_outputs"
    )
    physical_root.mkdir(parents=True, exist_ok=False)
    alias.symlink_to(physical_root, target_is_directory=True)
    if alias.resolve(strict=True) != physical_root.resolve(strict=True):
        raise RuntimeError("online numerical-view alias target changed")
    return physical_root


def _archive_service_runtime(
    source_root: Path,
    destination_root: Path,
    *,
    service: str,
) -> dict[str, Any]:
    if service == "online":
        candidates = [
            source_root / "service_report.json",
            source_root / "request_log.jsonl",
            source_root / "case_descriptor_manifest.json",
            source_root / "case_descriptors.npz",
        ]
    elif service == "evaluator":
        candidates = [
            source_root / "service_report.json",
            source_root / "service_status.json",
            source_root / "runtime_manifest.json",
        ]
        candidates.extend(
            source_root.glob("candidates/*/*/report.json")
            if source_root.is_dir()
            else ()
        )
    else:
        raise ValueError(f"unknown VPFP service archive kind: {service}")

    destination_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in sorted(set(candidates)):
        if not source.exists():
            continue
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"runtime summary artifact is not a regular file: {source}")
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        rows.append(
            {
                "path": str(relative),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
    report = {
        "schema": "agent_verifier_compact_service_archive_v1",
        "service": service,
        "archived": True,
        "source_runtime_existed": source_root.is_dir(),
        "file_count": len(rows),
        "bytes": sum(int(row["bytes"]) for row in rows),
        "files": rows,
        "dense_grid_arrays_archived": False,
    }
    _write_json(destination_root / "archive_manifest.json", report)
    return report


def _cleanup_online_numerical_views(
    workspace: Path,
    archive_root: Path,
    *,
    expected_ephemeral_root: Path | None = None,
) -> dict[str, Any]:
    output_root = workspace / "online_generator_outputs"
    if not output_root.exists() and not output_root.is_symlink():
        return {
            "removed": False,
            "reason": "not_created",
            "path": str(output_root),
        }
    is_ephemeral_alias = output_root.is_symlink()
    if is_ephemeral_alias:
        if expected_ephemeral_root is None:
            raise ValueError("ephemeral online output alias lacks an expected target")
        physical_root = output_root.resolve(strict=True)
        if (
            physical_root != expected_ephemeral_root.resolve(strict=True)
            or not physical_root.is_dir()
        ):
            raise ValueError("online output alias escaped its ephemeral target")
    else:
        physical_root = output_root
        if (
            output_root.resolve().parent != workspace.resolve()
            or not output_root.is_dir()
        ):
            raise ValueError("refusing to clean an unsafe online-generator output root")

    archive_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ("README.md", "manifest.json"):
        source = physical_root / name
        if not source.exists():
            continue
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"online summary is not a regular file: {source}")
        destination = archive_root / name
        shutil.copy2(source, destination)
        rows.append(
            {
                "path": name,
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
    _write_json(
        archive_root / "archive_manifest.json",
        {
            "schema": "agent_verifier_online_public_summary_archive_v1",
            "files": rows,
            "dense_grid_arrays_archived": False,
        },
    )
    if is_ephemeral_alias:
        output_root.unlink()
    else:
        shutil.rmtree(output_root)
    return {
        "removed": True,
        "reason": "reproducible_job_scoped_numerical_views",
        "path": str(output_root),
        "physical_path": str(physical_root),
        "ephemeral_alias": is_ephemeral_alias,
        "compact_file_count_archived": len(rows),
    }


def _start_online_service(
    *,
    config_path: Path,
    socket_path: Path,
    output_root: Path,
) -> tuple[subprocess.Popen, list[Any]]:
    output_root.mkdir(parents=True)
    stdout = (output_root / "stdout.log").open("w")
    stderr = (output_root / "stderr.log").open("w")
    process = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_vpfp_answer_first_online_generator_service.py"),
            "--config",
            str(config_path),
            "--socket",
            str(socket_path),
            "--device",
            "cuda",
        ],
        cwd=ROOT,
        env=os.environ.copy(),
        stdout=stdout,
        stderr=stderr,
        text=True,
    )
    _wait_until(
        lambda: socket_path.is_socket(),
        timeout_seconds=600.0,
        description="VPFP online generator socket",
        processes=(process,),
    )
    return process, [stdout, stderr]


def _start_evaluator_service(
    *,
    config_path: Path,
    stop_file: Path,
    output_root: Path,
) -> tuple[subprocess.Popen, list[Any]]:
    output_root.mkdir(parents=True)
    stdout = (output_root / "stdout.log").open("w")
    stderr = (output_root / "stderr.log").open("w")
    process = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_vpfp_development_evaluator_service_v2.py"),
            "--config",
            str(config_path),
            "--stop-file",
            str(stop_file),
        ],
        cwd=ROOT,
        env=os.environ.copy(),
        stdout=stdout,
        stderr=stderr,
        text=True,
    )
    return process, [stdout, stderr]


def _invoke_agent(
    *,
    workspace: Path,
    run_root: Path,
    packet: dict[str, Any],
    socket_path: Path,
    timeout_seconds: int,
    mock_agent: bool,
) -> dict[str, Any]:
    output_file = workspace / "controller_last_message.txt"
    if mock_agent:
        online_probe = _run_mock_online_probe(
            workspace=workspace, socket_path=socket_path
        )
        output_file.write_text(
            "Mock Agent preserved the staged incumbent bundle.\n",
            encoding="utf-8",
        )
        return {
            "actual_codex_invoked": False,
            "returncode": 0,
            "timed_out": False,
            "wall_time_seconds": 0.0,
            "output_file": str(output_file),
            "online_probe": online_probe,
        }
    runner_source = ROOT / "scripts" / "run_vpfp_instruction_only_solver.sh"
    runner = run_root / "run_vpfp_instruction_only_solver.snapshot.sh"
    shutil.copy2(runner_source, runner)
    runner.chmod(0o700)
    timeout_binary = shutil.which("timeout")
    if timeout_binary is None:
        raise RuntimeError("GNU timeout is required for VPFP Agent development")
    command = [
        str(Path(timeout_binary).resolve()),
        "--signal=TERM",
        "--kill-after=120s",
        f"{int(timeout_seconds)}s",
        str(runner),
        "--workspace",
        str(workspace),
        "--output-file",
        str(output_file),
        "--submission",
        str(workspace / "submission" / "submission.py"),
        "--model",
        str(packet["model"]),
        "--reasoning-effort",
        str(packet["reasoning_effort"]),
    ]
    environment = os.environ.copy()
    environment["VPFP_REPO_SCRIPT_DIR"] = str((ROOT / "scripts").resolve())
    environment["VPFP_ONLINE_GENERATOR_SOCKET"] = str(socket_path)
    environment["VPFP_CODEX_RUNTIME_PARENT"] = str(run_root / "codex_runtime")
    stdout_path = run_root / "codex.stdout.jsonl"
    stderr_path = run_root / "codex.stderr.log"
    started = time.time()
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=float(timeout_seconds) + 300.0)
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            returncode = process.wait(timeout=180.0)
    return {
        "actual_codex_invoked": True,
        "returncode": int(returncode),
        "timed_out": int(returncode) in {124, 137},
        "wall_time_seconds": float(time.time() - started),
        "output_file": str(output_file),
        "output_file_exists": output_file.is_file(),
        "submission_exists": (
            workspace / "submission" / "submission.py"
        ).is_file(),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def _run_mock_online_probe(
    *, workspace: Path, socket_path: Path
) -> dict[str, Any]:
    import numpy as np

    from agent_verifier.vpfp.public_online_generator_cli import (
        ONLINE_GENERATOR_REQUEST_SCHEMA,
        send_request,
    )

    points_path = workspace / "work" / "mock_online_macro_points.npy"
    np.save(points_path, np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64))
    previous_socket = os.environ.get("VPFP_ONLINE_GENERATOR_SOCKET")
    os.environ["VPFP_ONLINE_GENERATOR_SOCKET"] = str(socket_path)
    try:
        sample = send_request(
            {
                "schema": ONLINE_GENERATOR_REQUEST_SCHEMA,
                "operation": "sample",
                "count": 1,
                "mode": "base",
                "answers": "none",
                "condition_pool": "train",
            }
        )
        cases = list(sample.get("cases") or [])
        if sample.get("ok") is not True or len(cases) != 1:
            raise RuntimeError("VPFP mock online sample probe failed")
        case_id = str(cases[0]["case_id"])
        query = send_request(
            {
                "schema": ONLINE_GENERATOR_REQUEST_SCHEMA,
                "operation": "query",
                "case_id": case_id,
                "field": "macro_truth",
                "points_file": str(points_path),
                "out": None,
            }
        )
        status = send_request(
            {
                "schema": ONLINE_GENERATOR_REQUEST_SCHEMA,
                "operation": "status",
            }
        )
    finally:
        if previous_socket is None:
            os.environ.pop("VPFP_ONLINE_GENERATOR_SOCKET", None)
        else:
            os.environ["VPFP_ONLINE_GENERATOR_SOCKET"] = previous_socket

    budgets = dict(sample.get("budgets") or {})
    cached = dict(budgets.get("prefetch_cached_answers") or {})
    if not (
        query.get("ok") is True
        and query.get("point_count") == 1
        and status.get("ok") is True
        and budgets.get("candidate_answers_used") == 1
        and budgets.get("candidate_answers_generated") == 16
        and cached.get("train") == 15
    ):
        raise RuntimeError("VPFP mock online batching/query contract failed")
    return {
        "sample_case_id": case_id,
        "sample_mode": "base",
        "macro_query_point_count": int(query["point_count"]),
        "macro_query_output": query["output"],
        "candidate_answers_used": int(budgets["candidate_answers_used"]),
        "candidate_answers_generated": int(
            budgets["candidate_answers_generated"]
        ),
        "prefetch_train_cached": int(cached["train"]),
        "primary_truth": status.get("primary_truth"),
    }


def _submit_finalist_request(
    *,
    workspace: Path,
    candidate_entrypoint: str,
    agent_id: str,
) -> str:
    request_id = uuid.uuid4().hex
    payload = {
        "schema": VPFP_DEVELOPMENT_REQUEST_SCHEMA,
        "request_id": request_id,
        "agent_id": agent_id,
        "candidate_entrypoint": candidate_entrypoint,
        "requested_evidence_case_count": 16,
        "purpose": FINALIST_PURPOSE,
        "cascade": VPFP_DEVELOPMENT_CASCADE,
    }
    requests = workspace / "evaluation_exchange" / "requests"
    temporary = requests / f".{request_id}.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(requests / f"{request_id}.json")
    return request_id


def _wait_until(
    predicate,
    *,
    timeout_seconds: float,
    description: str,
    processes: tuple[subprocess.Popen | None, ...],
) -> None:
    deadline = time.time() + float(timeout_seconds)
    while time.time() < deadline:
        if predicate():
            return
        for process in processes:
            if process is not None and process.poll() is not None:
                raise RuntimeError(
                    f"{description} failed because service exited with "
                    f"status {process.returncode}"
                )
        time.sleep(0.5)
    raise TimeoutError(f"timed out waiting for {description}")


def _stop_process(
    process: subprocess.Popen | None,
    *,
    graceful_timeout: float,
) -> dict[str, Any]:
    if process is None:
        return {"started": False, "returncode": None, "forced": False}
    forced = False
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=float(graceful_timeout))
        except subprocess.TimeoutExpired:
            forced = True
            process.kill()
            process.wait(timeout=120.0)
    return {
        "started": True,
        "returncode": process.returncode,
        "forced": forced,
    }


def _terminate_process_group(process: subprocess.Popen) -> None:
    for signal_number, timeout_seconds in (
        (signal.SIGTERM, 120.0),
        (signal.SIGKILL, 60.0),
    ):
        try:
            os.killpg(process.pid, signal_number)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout_seconds)
            return
        except subprocess.TimeoutExpired:
            continue


def _agent_index(args: argparse.Namespace) -> int:
    value = (
        args.agent_index
        if args.agent_index is not None
        else int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
    )
    if int(value) not in (0, 1):
        raise SystemExit("VPFP Agent index must be 0 or 1")
    return int(value)


def _validate_runtime(args: argparse.Namespace) -> None:
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_TMPDIR"):
        raise SystemExit("VPFP Agent workers require Slurm compute nodes")
    if not os.environ.get("VIRTUAL_ENV"):
        raise SystemExit("VPFP Agent workers require the cluster venv")
    minimum_agent_seconds = 60 if args.mock_agent else 3600
    if (
        args.agent_timeout_seconds < minimum_agent_seconds
        or args.finalist_timeout_seconds < 300
    ):
        raise SystemExit("VPFP Agent worker timeouts are too small")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
