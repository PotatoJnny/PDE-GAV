from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import multiprocessing
import os
import select
import secrets
import socket
import socketserver
import tempfile
import threading
import time
import uuid
from collections.abc import Collection
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from agent_verifier.neural_generator.triplane_mms import sample_continuous_case
from agent_verifier.neural_generator.triplane_verifier import (
    CONTINUOUS_MMS_TRUTH_CONTRACT,
    require_continuous_mms_case_contract,
)


def evaluate_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a public-only triplane solver against private references.")
    parser.add_argument("--solver", default="submission.py")
    parser.add_argument("--cases", default="cases")
    parser.add_argument("--private-manifest", default=None)
    parser.add_argument("--nx", type=int, default=0, help="Optional output x-resolution. 0 uses each case's native grid.")
    parser.add_argument("--ny", type=int, default=0, help="Optional output y-resolution. 0 uses each case's native grid.")
    parser.add_argument("--nt", type=int, default=0, help="Optional output time-resolution. 0 uses each case's native time grid.")
    parser.add_argument("--failure-reward", type=float, default=1.0e6)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of cases to evaluate in parallel. One keeps serial behavior.",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    hidden_manifest = os.environ.get("AVF_TRIPLANE_EVAL_MANIFEST")
    manifest = args.private_manifest or hidden_manifest
    if not manifest:
        raise SystemExit("missing private evaluator manifest; set AVF_TRIPLANE_EVAL_MANIFEST")

    public_cases_root = Path(args.cases)
    selected_case_ids = None
    if args.private_manifest:
        explicit_payload = _load_private_manifest(Path(args.private_manifest))
        if _is_public_selection_manifest(explicit_payload):
            if not hidden_manifest:
                raise SystemExit(
                    "a public selection manifest requires AVF_TRIPLANE_EVAL_MANIFEST"
                )
            manifest = hidden_manifest
            selected_case_ids = [
                str(row["case_id"]) for row in explicit_payload["cases"]
            ]
    if selected_case_ids is None:
        selected_case_ids = _selection_from_public_root(
            public_cases_root,
            _load_private_manifest(Path(manifest)),
        )
        if (public_cases_root / "public_question.json").is_file():
            public_cases_root = public_cases_root.parent

    payload = evaluate_solver_file(
        solver_path=Path(args.solver),
        public_cases_root=public_cases_root,
        private_manifest_path=Path(manifest),
        nx=int(args.nx),
        ny=int(args.ny),
        nt=int(args.nt),
        failure_reward=float(args.failure_reward),
        selected_case_ids=selected_case_ids,
        max_workers=int(args.workers),
    )
    public_payload = _public_eval_payload(payload)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(public_payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(public_payload, indent=2, sort_keys=True))


EVALUATOR_BROKER_SOCKET_ENV = "AVF_TRIPLANE_EVAL_SOCKET"
EVALUATOR_BROKER_TOKEN_ENV = "AVF_TRIPLANE_EVAL_TOKEN"
_BROKER_MAX_REQUEST_BYTES = 1024 * 1024
_BROKER_ENV_PREFIXES = ("FLOW_", "SOLVER_", "AVF_SOLVER_")
_BROKER_ENV_NAMES = {
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
}


class _EvaluatorClientCancelled(RuntimeError):
    """Stop an evaluator worker after its client disconnects."""


def _socket_peer_closed(connection: socket.socket) -> bool:
    try:
        readable, _, _ = select.select([connection], [], [], 0.0)
        if not readable:
            return False
        return connection.recv(
            1,
            socket.MSG_PEEK | socket.MSG_DONTWAIT,
        ) == b""
    except (BlockingIOError, InterruptedError):
        return False
    except OSError:
        return True


def _path_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _evaluate_workspace_broker_request(
    workspace_value: str,
    private_manifest_value: str,
    request: dict[str, Any],
    cache_root_value: str | None = None,
) -> dict[str, Any]:
    workspace = Path(workspace_value).resolve(strict=True)
    cancel_path_value = request.pop("_broker_cancel_path", None)
    cancel_path = Path(cancel_path_value) if cancel_path_value else None

    def resolve_workspace_path(value: Any, *, kind: str) -> Path:
        path = Path(str(value))
        if not path.is_absolute():
            path = workspace / path
        path = path.resolve(strict=True)
        if not _path_inside(path, workspace):
            raise PermissionError(f"{kind} must be inside the solver workspace")
        return path

    solver_path = resolve_workspace_path(
        request.get("solver", "submission.py"),
        kind="solver",
    )
    if not solver_path.is_file():
        raise ValueError("solver must be a regular file")
    requested_cases_root = resolve_workspace_path(
        request.get("cases", "cases"),
        kind="cases",
    )
    if not requested_cases_root.is_dir():
        raise ValueError("cases must be a directory")

    manifest_path = Path(private_manifest_value).resolve(strict=True)
    private_manifest = _load_private_manifest(manifest_path)
    selected_case_ids = _selection_from_public_root(
        requested_cases_root,
        private_manifest,
    )
    if selected_case_ids is None:
        available = {str(row["case_id"]) for row in private_manifest["cases"]}
        visible_names = sorted(
            entry.name
            for entry in requested_cases_root.iterdir()
            if entry.name in available
        )
        if visible_names and len(visible_names) < len(available):
            selected_case_ids = visible_names

    public_cases_root = Path(private_manifest["public_cases_root"]).resolve(
        strict=True
    )
    if not public_cases_root.is_dir() or not _path_inside(
        public_cases_root, workspace
    ):
        raise PermissionError(
            "private manifest public cases root must be inside the solver workspace"
        )

    density: dict[str, int] = {}
    for name in ("nx", "ny", "nt"):
        value = int(request.get(name, 0))
        if value < 0 or value > 1024:
            raise ValueError(f"{name} must be between 0 and 1024")
        density[name] = value
    failure_reward = float(request.get("failure_reward", 1.0e6))
    if not np.isfinite(failure_reward) or failure_reward <= 0.0:
        raise ValueError("failure_reward must be finite and positive")

    requested_env = request.get("environment") or {}
    if not isinstance(requested_env, dict):
        raise ValueError("environment must be an object")
    allowed_env = {
        str(key): str(value)
        for key, value in requested_env.items()
        if str(key) in _BROKER_ENV_NAMES
        or str(key).startswith(_BROKER_ENV_PREFIXES)
    }
    cache_path: Path | None = None
    if cache_root_value is not None:
        cache_root = Path(cache_root_value)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_key = _broker_evaluation_cache_key(
            solver_path=solver_path,
            private_manifest_path=manifest_path,
            selected_case_ids=selected_case_ids,
            density=density,
            failure_reward=failure_reward,
            environment=allowed_env,
        )
        cache_path = cache_root / f"{cache_key}.json"
        cached = _read_json_object(cache_path)
        if cached is not None:
            cached["cache_hit"] = True
            return cached
    old_env = {key: os.environ.get(key) for key in allowed_env}
    try:
        os.environ.update(allowed_env)
        payload = evaluate_solver_file(
            solver_path=solver_path,
            public_cases_root=public_cases_root,
            private_manifest_path=manifest_path,
            nx=density["nx"],
            ny=density["ny"],
            nt=density["nt"],
            failure_reward=failure_reward,
            selected_case_ids=selected_case_ids,
            cancel_path=cancel_path,
        )
    finally:
        for key, old_value in old_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
    public_payload = _public_eval_payload(payload)
    public_payload["solver"] = str(solver_path.relative_to(workspace))
    public_payload["cases_root"] = str(
        requested_cases_root.relative_to(workspace)
    )
    public_payload["cache_hit"] = False
    if cache_path is not None:
        _write_json_atomically(cache_path, public_payload)
    return public_payload


def _broker_evaluation_cache_key(
    *,
    solver_path: Path,
    private_manifest_path: Path,
    selected_case_ids: Collection[str] | None,
    density: dict[str, int],
    failure_reward: float,
    environment: dict[str, str],
) -> str:
    payload = {
        "solver": _file_sha256(solver_path),
        "private_manifest": _file_sha256(private_manifest_path),
        "selected_case_ids": (
            [str(case_id) for case_id in selected_case_ids]
            if selected_case_ids is not None
            else None
        ),
        "density": {name: int(density[name]) for name in ("nx", "ny", "nt")},
        "failure_reward": float(failure_reward),
        "environment": dict(sorted(environment.items())),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class _ThreadingUnixStreamServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


class _WorkspaceEvaluatorRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        response: dict[str, Any] | None = None
        audit: dict[str, Any] = {"request_id": request_id}
        cancel_path: Path | None = None
        try:
            raw = self.rfile.readline(_BROKER_MAX_REQUEST_BYTES + 1)
            if not raw or len(raw) > _BROKER_MAX_REQUEST_BYTES:
                raise ValueError("invalid evaluator request size")
            request = json.loads(raw.decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("evaluator request must be an object")
            server = self.server
            token = str(request.pop("token", ""))
            if not secrets.compare_digest(token, server.broker_token):
                raise PermissionError("invalid evaluator broker token")
            request.pop("_broker_cancel_path", None)
            cancel_path = (
                Path(tempfile.gettempdir())
                / f"avf_eval_cancel_{os.getpid()}_{request_id}"
            )
            cancel_path.unlink(missing_ok=True)
            request["_broker_cancel_path"] = str(cancel_path)
            future = server.executor.submit(
                _evaluate_workspace_broker_request,
                str(server.workspace),
                str(server.private_manifest),
                request,
                str(server.cache_root),
            )
            client_cancelled = False
            while True:
                try:
                    payload = future.result(timeout=0.1)
                    break
                except concurrent.futures.TimeoutError:
                    if _socket_peer_closed(self.request):
                        cancel_path.touch(exist_ok=True)
                        client_cancelled = True
                        break
            if client_cancelled:
                try:
                    future.result()
                except _EvaluatorClientCancelled:
                    pass
                audit.update(
                    {
                        "status": "client_cancelled",
                        "solver": str(request.get("solver", "submission.py")),
                        "cases": str(request.get("cases", "cases")),
                    }
                )
            else:
                result_path = server.result_root / f"{request_id}.json"
                _write_json_atomically(result_path, payload)
                summary = payload.get("summary") or {}
                audit.update(
                    {
                        "status": "ok",
                        "solver": str(request.get("solver", "submission.py")),
                        "cases": str(request.get("cases", "cases")),
                        "query_density": {
                            name: int(request.get(name, 0))
                            for name in ("nx", "ny", "nt")
                        },
                        "n": int(summary.get("n", 0)),
                        "mean_relative_l2": summary.get("mean_relative_l2"),
                        "cache_hit": bool(payload.get("cache_hit", False)),
                        "result": str(result_path),
                    }
                )
                response = {"ok": True, "payload": payload}
        except Exception as exc:  # noqa: BLE001
            audit.update({"status": "error", "error": repr(exc)})
            response = {
                "ok": False,
                "error": f"{type(exc).__name__}: evaluator request failed",
            }
        finally:
            if cancel_path is not None:
                cancel_path.unlink(missing_ok=True)
            audit["wall_time_seconds"] = float(time.perf_counter() - started)
            self.server.broker._record(audit)
        if response is None:
            return
        try:
            self.wfile.write(
                (json.dumps(response, allow_nan=True) + "\n").encode("utf-8")
            )
        except (BrokenPipeError, ConnectionResetError):
            pass


class WorkspaceEvaluatorBroker:
    """Keep hidden references in the parent and expose only public error payloads."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        private_manifest: str | Path,
        audit_path: str | Path,
        max_workers: int | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve(strict=True)
        self.private_manifest = Path(private_manifest).resolve(strict=True)
        self.audit_path = Path(audit_path)
        self.result_root = self.audit_path.parent / "evaluator_requests"
        self.cache_root = self.audit_path.parent / "evaluator_cache"
        self.max_workers = max_workers or min(16, max(1, os.cpu_count() or 1))
        self.socket_path = (
            Path("/tmp")
            / f"avf_eval_{os.getpid()}_{uuid.uuid4().hex}.sock"
        )
        self.token = secrets.token_urlsafe(32)
        self._audit_lock = threading.Lock()
        self._executor: concurrent.futures.ProcessPoolExecutor | None = None
        self._server: _ThreadingUnixStreamServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def environment(self) -> dict[str, str]:
        return {
            EVALUATOR_BROKER_SOCKET_ENV: str(self.socket_path),
            EVALUATOR_BROKER_TOKEN_ENV: self.token,
        }

    def __enter__(self) -> "WorkspaceEvaluatorBroker":
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.result_root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        context = multiprocessing.get_context("spawn")
        self._executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=int(self.max_workers),
            mp_context=context,
        )
        self._server = _ThreadingUnixStreamServer(
            str(self.socket_path),
            _WorkspaceEvaluatorRequestHandler,
        )
        self._server.workspace = self.workspace
        self._server.private_manifest = self.private_manifest
        self._server.broker_token = self.token
        self._server.result_root = self.result_root
        self._server.cache_root = self.cache_root
        self._server.executor = self._executor
        self._server.broker = self
        os.chmod(self.socket_path, 0o600)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="workspace-evaluator-broker",
            daemon=True,
        )
        self._thread.start()
        return self

    def _record(self, payload: dict[str, Any]) -> None:
        with self._audit_lock:
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self.socket_path.unlink(missing_ok=True)

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def evaluate_solver_file(
    *,
    solver_path: str | Path,
    public_cases_root: str | Path,
    private_manifest_path: str | Path,
    nx: int = 0,
    ny: int = 0,
    nt: int = 0,
    failure_reward: float = 1.0e6,
    checkpoint_path: str | Path | None = None,
    checkpoint_interval: int = 64,
    selected_case_ids: Collection[str] | None = None,
    solver_input_mode: str = "query",
    cancel_path: str | Path | None = None,
    max_workers: int = 1,
) -> dict[str, Any]:
    solver_path = Path(solver_path)
    public_root = Path(public_cases_root)
    manifest_path = Path(private_manifest_path)
    manifest = _load_private_manifest(manifest_path)
    if selected_case_ids is not None:
        manifest = _select_private_manifest_rows(
            manifest,
            selected_case_ids=selected_case_ids,
        )
    reward_spec = _validated_reward_spec(manifest.get("reward_spec"))
    truth_contract = manifest.get("truth_contract")
    if truth_contract not in (None, CONTINUOUS_MMS_TRUTH_CONTRACT):
        raise ValueError(f"unsupported private truth contract: {truth_contract}")
    require_continuous_mms = truth_contract == CONTINUOUS_MMS_TRUTH_CONTRACT
    if solver_input_mode not in {"query", "native"}:
        raise ValueError(
            "solver_input_mode must be 'query' or 'native', got "
            f"{solver_input_mode!r}"
        )
    worker_count = int(max_workers)
    if worker_count <= 0:
        raise ValueError("max_workers must be positive")
    expected_case_ids = [str(row["case_id"]) for row in manifest["cases"]]
    if len(expected_case_ids) != len(set(expected_case_ids)):
        raise ValueError("private evaluator manifest contains duplicate case IDs")

    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    signature = None
    resumed_rows: list[dict[str, Any]] = []
    if checkpoint is not None:
        signature = _evaluation_checkpoint_signature(
            solver_path=solver_path,
            public_cases_root=public_root,
            private_manifest_path=manifest_path,
            nx=int(nx),
            ny=int(ny),
            nt=int(nt),
            failure_reward=float(failure_reward),
            solver_input_mode=solver_input_mode,
        )
        resumed_rows = _load_evaluation_checkpoint(
            checkpoint,
            signature=signature,
            expected_case_ids=set(expected_case_ids),
        )
    rows_by_case_id = {str(row["case_id"]): row for row in resumed_rows}
    cancellation_marker = Path(cancel_path) if cancel_path is not None else None

    def raise_if_cancelled() -> None:
        if cancellation_marker is not None and cancellation_marker.exists():
            raise _EvaluatorClientCancelled("evaluator client disconnected")

    pending_rows = [
        row
        for row in manifest["cases"]
        if str(row["case_id"]) not in rows_by_case_id
    ]

    def record_case_result(case_result: dict[str, Any]) -> None:
        case_id = str(case_result["case_id"])
        rows_by_case_id[case_id] = case_result
        if checkpoint is not None and (
            len(rows_by_case_id) % max(1, int(checkpoint_interval)) == 0
            or len(rows_by_case_id) == len(expected_case_ids)
        ):
            assert signature is not None
            partial_rows = [
                rows_by_case_id[item]
                for item in expected_case_ids
                if item in rows_by_case_id
            ]
            _write_evaluation_checkpoint(
                checkpoint,
                signature=signature,
                rows=partial_rows,
            )

    raise_if_cancelled()
    if worker_count == 1:
        module, load_status, load_error = _load_and_prepare_submission(
            solver_path,
            public_root,
        )
        for row in pending_rows:
            raise_if_cancelled()
            record_case_result(
                _evaluate_manifest_case(
                    module=module,
                    load_status=load_status,
                    load_error=load_error,
                    row=row,
                    public_root=public_root,
                    nx=int(nx),
                    ny=int(ny),
                    nt=int(nt),
                    failure_reward=float(failure_reward),
                    require_continuous_mms=require_continuous_mms,
                    reward_spec=reward_spec,
                    solver_input_mode=solver_input_mode,
                )
            )
    elif pending_rows:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_initialize_evaluation_worker,
            initargs=(str(solver_path), str(public_root)),
        ) as executor:
            futures = [
                executor.submit(
                    _evaluate_manifest_case_in_worker,
                    row,
                    str(public_root),
                    int(nx),
                    int(ny),
                    int(nt),
                    float(failure_reward),
                    bool(require_continuous_mms),
                    reward_spec,
                    solver_input_mode,
                    str(cancellation_marker) if cancellation_marker is not None else None,
                )
                for row in pending_rows
            ]
            for future in concurrent.futures.as_completed(futures):
                raise_if_cancelled()
                record_case_result(future.result())

    raise_if_cancelled()
    case_rows = [rows_by_case_id[case_id] for case_id in expected_case_ids]
    payload = {
        "solver": str(solver_path),
        "cases_root": str(public_root),
        "private_manifest": str(manifest_path),
        "query_density": {"nx": int(nx), "ny": int(ny), "nt": int(nt)},
        "solver_input_mode": solver_input_mode,
        "truth_contract": truth_contract,
        "reward_spec": reward_spec,
        "worker_count": worker_count,
        "summary": summarize_metric_rows(case_rows),
        "results": case_rows,
    }
    if checkpoint is not None:
        assert signature is not None
        _write_evaluation_checkpoint(
            checkpoint,
            signature=signature,
            rows=case_rows,
            complete=True,
        )
    return payload


_EVALUATION_WORKER_MODULE: Any | None = None
_EVALUATION_WORKER_LOAD_STATUS = "submission_import_failed"
_EVALUATION_WORKER_LOAD_ERROR: str | None = None
_EVALUATION_THREAD_LIMITER: Any | None = None


def _initialize_evaluation_worker(solver_path: str, public_cases_root: str) -> None:
    """Load one solver per worker so its reusable operators stay in memory."""

    global _EVALUATION_WORKER_MODULE
    global _EVALUATION_WORKER_LOAD_STATUS
    global _EVALUATION_WORKER_LOAD_ERROR
    global _EVALUATION_THREAD_LIMITER

    # A process evaluates one case at a time. Limiting native math libraries to
    # one thread prevents four workers from silently becoming dozens of threads.
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    try:
        from threadpoolctl import threadpool_limits

        _EVALUATION_THREAD_LIMITER = threadpool_limits(limits=1)
    except ImportError:
        _EVALUATION_THREAD_LIMITER = None

    (
        _EVALUATION_WORKER_MODULE,
        _EVALUATION_WORKER_LOAD_STATUS,
        _EVALUATION_WORKER_LOAD_ERROR,
    ) = _load_and_prepare_submission(
        Path(solver_path),
        Path(public_cases_root),
    )


def _load_and_prepare_submission(
    solver_path: Path,
    public_cases_root: Path,
) -> tuple[Any | None, str, str | None]:
    try:
        module = _load_submission_module(solver_path)
    except Exception as exc:  # noqa: BLE001
        return None, "submission_import_failed", repr(exc)

    prepare = getattr(module, "prepare_solver", None)
    if prepare is not None:
        try:
            prepare(str(public_cases_root))
        except Exception as exc:  # noqa: BLE001
            return None, "submission_prepare_failed", repr(exc)
    return module, "ok", None


def _evaluate_manifest_case_in_worker(
    row: dict[str, Any],
    public_root: str,
    nx: int,
    ny: int,
    nt: int,
    failure_reward: float,
    require_continuous_mms: bool,
    reward_spec: dict[str, Any],
    solver_input_mode: str,
    cancel_path: str | None,
) -> dict[str, Any]:
    if cancel_path is not None and Path(cancel_path).exists():
        raise _EvaluatorClientCancelled("evaluator client disconnected")
    return _evaluate_manifest_case(
        module=_EVALUATION_WORKER_MODULE,
        load_status=_EVALUATION_WORKER_LOAD_STATUS,
        load_error=_EVALUATION_WORKER_LOAD_ERROR,
        row=row,
        public_root=Path(public_root),
        nx=nx,
        ny=ny,
        nt=nt,
        failure_reward=failure_reward,
        require_continuous_mms=require_continuous_mms,
        reward_spec=reward_spec,
        solver_input_mode=solver_input_mode,
    )


def _evaluate_manifest_case(
    *,
    module: Any | None,
    load_status: str,
    load_error: str | None,
    row: dict[str, Any],
    public_root: Path,
    nx: int,
    ny: int,
    nt: int,
    failure_reward: float,
    require_continuous_mms: bool,
    reward_spec: dict[str, Any],
    solver_input_mode: str,
) -> dict[str, Any]:
    case_id = str(row["case_id"])
    if module is None:
        return {
            "case_id": case_id,
            "status": load_status,
            "exception": load_error,
            "relative_l2": float(failure_reward),
            "difficulty_reward": float(failure_reward),
            "reward_metric": str(reward_spec["metric"]),
            "velocity_rmse": float(failure_reward),
            "truth_velocity_rms": 0.0,
            "train_scaled_velocity_rmse": float(failure_reward),
            "wall_time_seconds": 0.0,
        }
    return _evaluate_one_case(
        module,
        public_case_dir=public_root / case_id,
        private_case_dir=Path(row["private_case_dir"]),
        nx=int(nx),
        ny=int(ny),
        nt=int(nt),
        failure_reward=float(failure_reward),
        require_continuous_mms=bool(require_continuous_mms),
        reward_spec=reward_spec,
        solver_input_mode=solver_input_mode,
    )


def _evaluation_checkpoint_signature(
    *,
    solver_path: Path,
    public_cases_root: Path,
    private_manifest_path: Path,
    nx: int,
    ny: int,
    nt: int,
    failure_reward: float,
    solver_input_mode: str,
) -> dict[str, Any]:
    return {
        "schema": "triplane_solver_eval_checkpoint_v1",
        "solver_sha256": _file_sha256(solver_path),
        "manifest_sha256": _file_sha256(private_manifest_path),
        "public_cases_root": str(public_cases_root.resolve()),
        "query_density": {"nx": nx, "ny": ny, "nt": nt},
        "solver_input_mode": solver_input_mode,
        "failure_reward": failure_reward,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_evaluation_checkpoint(
    path: Path,
    *,
    signature: dict[str, Any],
    expected_case_ids: set[str],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if payload.get("signature") != signature:
        return []
    rows = payload.get("results")
    if not isinstance(rows, list):
        return []
    valid_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        case_id = str(row.get("case_id"))
        if case_id not in expected_case_ids or case_id in seen:
            return []
        valid_rows.append(row)
        seen.add(case_id)
    return valid_rows


def _write_evaluation_checkpoint(
    path: Path,
    *,
    signature: dict[str, Any],
    rows: list[dict[str, Any]],
    complete: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "signature": signature,
        "complete": bool(complete),
        "completed_case_count": len(rows),
        "summary": summarize_metric_rows(rows),
        "results": rows,
    }
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)

def summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray(
        [
            float(row["relative_l2"])
            for row in rows
            if isinstance(row.get("relative_l2"), (int, float)) and np.isfinite(float(row["relative_l2"]))
        ],
        dtype=np.float64,
    )
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    summary: dict[str, Any] = {
        "n": len(rows),
        "status_counts": status_counts,
        "ok": int(status_counts.get("ok", 0)),
    }
    if values.size:
        summary.update(
            {
                "mean_relative_l2": float(np.mean(values)),
                "median_relative_l2": float(np.median(values)),
                "q10_relative_l2": float(np.quantile(values, 0.10)),
                "q90_relative_l2": float(np.quantile(values, 0.90)),
                "min_relative_l2": float(np.min(values)),
                "max_relative_l2": float(np.max(values)),
            }
        )
    else:
        summary.update(
            {
                "mean_relative_l2": None,
                "median_relative_l2": None,
                "q10_relative_l2": None,
                "q90_relative_l2": None,
                "min_relative_l2": None,
                "max_relative_l2": None,
            }
        )
    for metric in (
        "difficulty_reward",
        "velocity_rmse",
        "truth_velocity_rms",
        "train_scaled_velocity_rmse",
    ):
        metric_values = np.asarray(
            [
                float(row[metric])
                for row in rows
                if isinstance(row.get(metric), (int, float))
                and np.isfinite(float(row[metric]))
            ],
            dtype=np.float64,
        )
        if metric_values.size:
            summary[f"mean_{metric}"] = float(np.mean(metric_values))
            summary[f"median_{metric}"] = float(np.median(metric_values))
    return summary


def write_private_manifest(
    path: str | Path,
    *,
    public_cases_root: str | Path,
    private_case_dirs: list[str | Path],
    reward_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    public_root = Path(public_cases_root)
    cases = []
    for private_case_dir in private_case_dirs:
        private_path = Path(private_case_dir)
        cases.append(
            {
                "case_id": private_path.name,
                "public_case_dir": str(public_root / private_path.name),
                "private_case_dir": str(private_path),
            }
        )
    payload = {
        "public_cases_root": str(public_root),
        "cases": cases,
        "note": "Private evaluator manifest. Do not place this file in a solver agent workspace.",
    }
    if reward_spec is not None:
        payload["reward_spec"] = _validated_reward_spec(reward_spec)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def reward_spec_from_autoencoder_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("loading the frozen reward normalizer requires PyTorch") from exc
    from agent_verifier.neural_generator.triplane_autoencoder import FieldNormalizer

    checkpoint = Path(path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    normalizer_payload = payload.get("field_normalizer") or (
        payload.get("train_config") or {}
    ).get("field_normalizer")
    normalizer = FieldNormalizer.from_dict(normalizer_payload)
    if normalizer is None:
        raise ValueError(f"AE checkpoint has no frozen field normalizer: {checkpoint}")
    normalizer_abi = str((normalizer_payload or {}).get("abi") or "")
    if normalizer_abi != "field_unit_gaussian_v1":
        raise ValueError(f"unsupported field normalizer ABI: {normalizer_abi!r}")
    if normalizer.fitted_split != "train":
        raise ValueError(
            f"reward normalizer must be fitted on train, got {normalizer.fitted_split!r}"
        )
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return _validated_reward_spec(
        {
            "metric": "train_scaled_velocity_rmse",
            "velocity_std": [float(value) for value in normalizer.velocity_std],
            "normalizer_abi": normalizer_abi,
            "fitted_split": normalizer.fitted_split,
            "fit_case_count": int(normalizer.fit_case_count),
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": digest.hexdigest(),
        }
    )


def write_workspace_evaluator_tool(
    path: str | Path,
    *,
    repo_src: str | Path | None = None,
    broker_only: bool = False,
) -> None:
    if broker_only:
        Path(path).write_text(
            "from __future__ import annotations\n\n"
            "import argparse\n"
            "import json\n"
            "import os\n"
            "import socket\n"
            "from pathlib import Path\n\n"
            f"SOCKET_ENV = {EVALUATOR_BROKER_SOCKET_ENV!r}\n"
            f"TOKEN_ENV = {EVALUATOR_BROKER_TOKEN_ENV!r}\n"
            f"ENV_PREFIXES = {_BROKER_ENV_PREFIXES!r}\n"
            f"ENV_NAMES = {_BROKER_ENV_NAMES!r}\n\n"
            "def main() -> None:\n"
            "    parser = argparse.ArgumentParser(description='Evaluate a solver and return public error metrics.')\n"
            "    parser.add_argument('--solver', default='submission.py')\n"
            "    parser.add_argument('--cases', default='cases')\n"
            "    parser.add_argument('--nx', type=int, default=0)\n"
            "    parser.add_argument('--ny', type=int, default=0)\n"
            "    parser.add_argument('--nt', type=int, default=0)\n"
            "    parser.add_argument('--failure-reward', type=float, default=1.0e6)\n"
            "    parser.add_argument('--out', default=None)\n"
            "    args = parser.parse_args()\n"
            "    socket_path = os.environ.get(SOCKET_ENV)\n"
            "    token = os.environ.get(TOKEN_ENV)\n"
            "    if not socket_path or not token:\n"
            "        raise SystemExit('the evaluator broker is not available')\n"
            "    environment = {\n"
            "        key: value\n"
            "        for key, value in os.environ.items()\n"
            "        if key in ENV_NAMES or key.startswith(ENV_PREFIXES)\n"
            "    }\n"
            "    request = {\n"
            "        'token': token,\n"
            "        'solver': args.solver,\n"
            "        'cases': args.cases,\n"
            "        'nx': args.nx,\n"
            "        'ny': args.ny,\n"
            "        'nt': args.nt,\n"
            "        'failure_reward': args.failure_reward,\n"
            "        'environment': environment,\n"
            "    }\n"
            "    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:\n"
            "        client.connect(socket_path)\n"
            "        client.sendall((json.dumps(request) + '\\n').encode('utf-8'))\n"
            "        response = json.loads(client.makefile('r', encoding='utf-8').readline())\n"
            "    if not response.get('ok'):\n"
            "        raise SystemExit(str(response.get('error') or 'evaluation failed'))\n"
            "    payload = response['payload']\n"
            "    rendered = json.dumps(payload, indent=2, sort_keys=True) + '\\n'\n"
            "    if args.out:\n"
            "        out = Path(args.out)\n"
            "        out.parent.mkdir(parents=True, exist_ok=True)\n"
            "        out.write_text(rendered)\n"
            "    print(rendered, end='')\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        return

    if repo_src is None:
        repo_src_expr = "Path(__file__).resolve().parents[4] / 'src'"
    else:
        repo_src_expr = f"Path({str(Path(repo_src).resolve())!r})"
    Path(path).write_text(
        "from __future__ import annotations\n\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        f"REPO_SRC = {repo_src_expr}\n"
        "if REPO_SRC.exists() and str(REPO_SRC) not in sys.path:\n"
        "    sys.path.insert(0, str(REPO_SRC))\n\n"
        "from agent_verifier.neural_generator.triplane_agent_solver import evaluate_cli\n\n"
        "if __name__ == '__main__':\n"
        "    evaluate_cli()\n"
    )


def _evaluate_one_case(
    module,
    *,
    public_case_dir: Path,
    private_case_dir: Path,
    nx: int,
    ny: int,
    nt: int,
    failure_reward: float,
    require_continuous_mms: bool = False,
    reward_spec: dict[str, Any] | None = None,
    solver_input_mode: str = "query",
) -> dict[str, Any]:
    resolved_reward_spec = _validated_reward_spec(reward_spec)
    if require_continuous_mms:
        require_continuous_mms_case_contract(private_case_dir)
    start = time.perf_counter()
    try:
        question = _load_public_question(public_case_dir)
        public_arrays = question["_arrays"].get("public_arrays", {})
        old_t = np.asarray(public_arrays.get("t", []), dtype=np.float64)
        old_x = np.asarray(public_arrays.get("x", []), dtype=np.float64)
        old_y = np.asarray(public_arrays.get("y", []), dtype=np.float64)
        velocity = None
        if (private_case_dir / "hidden_continuous_case.pt").exists():
            query_grid = _query_grid_from_public(public_arrays, nx=nx, ny=ny, nt=nt)
            fields = _continuous_materialized_view(
                private_case_dir=private_case_dir,
                public_arrays=public_arrays,
                query_grid=query_grid,
            )
            target = np.asarray(fields.velocity, dtype=np.float64)
            fluid = _resample_mask_to_grid(
                np.asarray(public_arrays["fluid_mask"], dtype=bool),
                old_x=old_x,
                old_y=old_y,
                new_x=query_grid["x"],
                new_y=query_grid["y"],
            )
            if solver_input_mode == "native":
                native_grid = {"t": old_t, "x": old_x, "y": old_y}
                output = _call_solver(
                    module, public_case_dir, question, native_grid
                )
                native_velocity = _extract_velocity(
                    output,
                    target_shape=(
                        len(old_t),
                        2,
                        len(old_x),
                        len(old_y),
                    ),
                )
                velocity = _resample_velocity_to_grid(
                    native_velocity,
                    old_t=old_t,
                    old_x=old_x,
                    old_y=old_y,
                    new_t=query_grid["t"],
                    new_x=query_grid["x"],
                    new_y=query_grid["y"],
                )
            else:
                with tempfile.TemporaryDirectory(
                    prefix="avf_continuous_public_"
                ) as tmp:
                    query_case_dir = Path(tmp) / public_case_dir.name
                    query_question = _materialize_continuous_public_view(
                        query_case_dir,
                        question,
                        public_arrays,
                        fields,
                        query_grid,
                    )
                    output = _call_solver(
                        module, query_case_dir, query_question, query_grid
                    )
        else:
            with np.load(private_case_dir / "hidden_truth_cache.npz") as hidden:
                truth = np.asarray(hidden["u_true"], dtype=np.float64)
            query_grid = _query_grid(public_arrays, truth, nx=nx, ny=ny, nt=nt)
            target = _resample_velocity_to_grid(
                truth,
                old_t=old_t,
                old_x=old_x,
                old_y=old_y,
                new_t=query_grid["t"],
                new_x=query_grid["x"],
                new_y=query_grid["y"],
            )
            fluid = _resample_mask_to_grid(
                np.asarray(public_arrays.get("fluid_mask", np.ones(truth.shape[2:], dtype=bool)), dtype=bool),
                old_x=old_x,
                old_y=old_y,
                new_x=query_grid["x"],
                new_y=query_grid["y"],
            )
            output = _call_solver(module, public_case_dir, question, query_grid)
        if velocity is None:
            velocity = _extract_velocity(output, target_shape=target.shape)
        metrics = _velocity_error_metrics(
            velocity,
            target,
            fluid,
            velocity_std=resolved_reward_spec.get("velocity_std"),
        )
        rel = float(metrics["relative_l2"])
        difficulty = float(metrics[resolved_reward_spec["metric"]])
        status = "ok" if np.isfinite(difficulty) else "bad_prediction"
        exception = None
        metadata = output.get("metadata", {}) if isinstance(output, dict) and isinstance(output.get("metadata"), dict) else {}
    except Exception as exc:  # noqa: BLE001
        velocity = None
        rel = float(failure_reward)
        difficulty = float(failure_reward)
        metrics = {
            "relative_l2": float(failure_reward),
            "velocity_rmse": float(failure_reward),
            "truth_velocity_rms": 0.0,
            "train_scaled_velocity_rmse": float(failure_reward),
        }
        status = "exception"
        exception = repr(exc)
        metadata = {}
        target = np.zeros((0, 2, 0, 0), dtype=np.float64)
        query_grid = {"x": np.asarray([]), "y": np.asarray([]), "t": np.asarray([])}
    wall = time.perf_counter() - start
    row = {
        "case_id": public_case_dir.name,
        "status": status,
        "relative_l2": float(rel if np.isfinite(rel) else failure_reward),
        "difficulty_reward": float(
            difficulty if np.isfinite(difficulty) else failure_reward
        ),
        "reward_metric": str(resolved_reward_spec["metric"]),
        "velocity_rmse": float(metrics["velocity_rmse"]),
        "truth_velocity_rms": float(metrics["truth_velocity_rms"]),
        "train_scaled_velocity_rmse": float(
            metrics["train_scaled_velocity_rmse"]
        ),
        "wall_time_seconds": float(wall),
        "target_shape": list(target.shape),
        "prediction_shape": list(np.asarray(velocity).shape) if velocity is not None else None,
        "query_density": {
            "nx": int(len(query_grid["x"])),
            "ny": int(len(query_grid["y"])),
            "nt": int(len(query_grid["t"])),
        },
        "solver_input_mode": solver_input_mode,
        "metadata": _jsonable(metadata),
    }
    if exception is not None:
        row["exception"] = exception
    return row


def _load_private_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"private manifest missing cases list: {path}")
    return payload


def _is_public_selection_manifest(payload: dict[str, Any]) -> bool:
    cases = payload.get("cases", [])
    return bool(cases) and all(
        row.get("case_id") and not row.get("private_case_dir")
        for row in cases
    )


def _selection_from_public_root(
    public_root: Path,
    private_manifest: dict[str, Any],
) -> list[str] | None:
    available = {str(row["case_id"]) for row in private_manifest["cases"]}
    if (public_root / "public_question.json").is_file():
        return [public_root.name]

    local_manifest = public_root / "case_manifest.json"
    if local_manifest.is_file():
        payload = _load_private_manifest(local_manifest)
        return [str(row["case_id"]) for row in payload["cases"]]

    visible = (
        sorted(
            path.name
            for path in public_root.iterdir()
            if path.is_dir()
            and path.name in available
            and (path / "public_question.json").is_file()
        )
        if public_root.is_dir()
        else []
    )
    if visible and len(visible) < len(available):
        return visible
    return None


def _select_private_manifest_rows(
    manifest: dict[str, Any],
    *,
    selected_case_ids: Collection[str],
) -> dict[str, Any]:
    requested = [str(case_id) for case_id in selected_case_ids]
    if not requested:
        raise ValueError("public evaluator selection contains no case IDs")
    if len(requested) != len(set(requested)):
        raise ValueError("public evaluator selection contains duplicate case IDs")
    rows_by_id = {str(row["case_id"]): row for row in manifest["cases"]}
    missing = [case_id for case_id in requested if case_id not in rows_by_id]
    if missing:
        raise ValueError(
            f"public evaluator selection contains unknown case IDs: {missing[:8]}"
        )
    return {
        **manifest,
        "cases": [rows_by_id[case_id] for case_id in requested],
    }


def _validated_reward_spec(value: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(value or {})
    metric = str(payload.get("metric") or "relative_l2")
    if metric not in {"relative_l2", "train_scaled_velocity_rmse"}:
        raise ValueError(f"unsupported triplane solver reward metric: {metric}")
    result: dict[str, Any] = {"metric": metric}
    velocity_std = payload.get("velocity_std")
    if velocity_std is not None:
        values = tuple(float(item) for item in velocity_std)
        if len(values) != 2 or not all(
            np.isfinite(item) and item > 0.0 for item in values
        ):
            raise ValueError(
                "reward velocity_std must contain two finite positive values"
            )
        result["velocity_std"] = [float(item) for item in values]
    if metric == "train_scaled_velocity_rmse" and "velocity_std" not in result:
        raise ValueError(
            "train-scaled velocity reward requires frozen velocity_std"
        )
    for key in (
        "normalizer_abi",
        "fitted_split",
        "source_checkpoint",
        "source_checkpoint_sha256",
    ):
        if payload.get(key) is not None:
            result[key] = str(payload[key])
    if payload.get("fit_case_count") is not None:
        result["fit_case_count"] = int(payload["fit_case_count"])
    return result


def _load_submission_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"triplane_solver_{abs(hash(path))}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "solve_case") and not hasattr(module, "solve"):
        raise AttributeError(f"{path} must define solve_case(case_dir, query_grid=None) or solve(question)")
    return module


def _load_public_question(case_dir: Path) -> dict[str, Any]:
    q = json.loads((case_dir / "public_question.json").read_text())
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for npz in sorted(case_dir.glob("public_*.npz")):
        with np.load(npz) as data:
            arrays[npz.stem] = {key: np.asarray(data[key]) for key in data.files}
    q["_arrays"] = arrays
    q["_path"] = str(case_dir)
    return q


def _continuous_materialized_view(
    *,
    private_case_dir: Path,
    public_arrays: dict[str, np.ndarray],
    query_grid: dict[str, np.ndarray],
):
    truth_cache_path = private_case_dir / "hidden_truth_cache.npz"
    if _grid_matches_public(public_arrays, query_grid) and truth_cache_path.exists():
        with np.load(truth_cache_path) as hidden:
            velocity = np.asarray(hidden["u_true"], dtype=np.float64)
        return SimpleNamespace(
            velocity=velocity,
            forcing=np.asarray(public_arrays["forcing"], dtype=np.float64),
        )

    cache_path = _continuous_view_cache_path(private_case_dir, query_grid)
    if cache_path.exists():
        try:
            with np.load(cache_path) as cached:
                return SimpleNamespace(
                    velocity=np.asarray(cached["velocity"], dtype=np.float64),
                    forcing=np.asarray(cached["forcing"], dtype=np.float64),
                )
        except (OSError, ValueError, KeyError):
            pass

    fields = sample_continuous_case(
        private_case_dir,
        x=query_grid["x"],
        y=query_grid["y"],
        t=query_grid["t"],
        device=os.environ.get("AVF_TRIPLANE_MMS_DEVICE", "auto"),
    )
    _write_continuous_view_cache(cache_path, fields)
    return fields


def _grid_matches_public(public_arrays: dict[str, np.ndarray], query_grid: dict[str, np.ndarray]) -> bool:
    return all(
        np.array_equal(
            np.asarray(public_arrays[name], dtype=np.float64),
            np.asarray(query_grid[name], dtype=np.float64),
        )
        for name in ("x", "y", "t")
    )


def _continuous_view_cache_path(private_case_dir: Path, query_grid: dict[str, np.ndarray]) -> Path:
    state_path = private_case_dir / "hidden_continuous_case.pt"
    state_stat = state_path.stat()
    digest = hashlib.sha256()
    digest.update(f"{state_stat.st_size}:{state_stat.st_mtime_ns}".encode("ascii"))
    for name in ("x", "y", "t"):
        values = np.ascontiguousarray(query_grid[name], dtype=np.float64)
        digest.update(name.encode("ascii"))
        digest.update(values.shape[0].to_bytes(8, byteorder="little", signed=False))
        digest.update(values.tobytes())
    return private_case_dir / ".continuous_query_cache" / f"{digest.hexdigest()}.npz"


def _write_continuous_view_cache(path: Path, fields) -> None:
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        np.savez_compressed(
            temporary_path,
            velocity=np.asarray(fields.velocity, dtype=np.float64),
            forcing=np.asarray(fields.forcing, dtype=np.float64),
        )
        os.replace(temporary_path, path)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _materialize_continuous_public_view(
    out_case_dir: Path,
    question: dict[str, Any],
    public_arrays: dict[str, np.ndarray],
    fields,
    query_grid: dict[str, np.ndarray],
) -> dict[str, Any]:
    out_case_dir.mkdir(parents=True, exist_ok=True)
    old_x = np.asarray(public_arrays["x"], dtype=np.float64)
    old_y = np.asarray(public_arrays["y"], dtype=np.float64)
    arrays: dict[str, np.ndarray] = {}
    mask_names = {"fluid_mask", "obstacle_mask", "outer_boundary_mask"}
    for name in ("fluid_mask", "obstacle_mask", "outer_boundary_mask", "obstacle_signed_distance", "brinkman_indicator"):
        value = np.asarray(public_arrays[name])
        if name == "outer_boundary_mask":
            outer = np.zeros(
                (len(query_grid["x"]), len(query_grid["y"])),
                dtype=bool,
            )
            outer[0, :] = True
            outer[-1, :] = True
            outer[:, 0] = True
            outer[:, -1] = True
            arrays[name] = outer
        elif name in mask_names:
            arrays[name] = _resample_mask_to_grid(
                value,
                old_x=old_x,
                old_y=old_y,
                new_x=query_grid["x"],
                new_y=query_grid["y"],
            )
        else:
            arrays[name] = _resample_scalar_to_grid(
                value,
                old_x=old_x,
                old_y=old_y,
                new_x=query_grid["x"],
                new_y=query_grid["y"],
            ).astype(value.dtype, copy=False)
    arrays.update(
        {
            "x": np.asarray(query_grid["x"], dtype=np.float64),
            "y": np.asarray(query_grid["y"], dtype=np.float64),
            "t": np.asarray(query_grid["t"], dtype=np.float64),
            "initial_velocity": np.asarray(fields.velocity[0], dtype=np.float32),
            "forcing": np.asarray(fields.forcing, dtype=np.float32),
        }
    )
    np.savez_compressed(out_case_dir / "public_arrays.npz", **arrays)
    public_question = {key: value for key, value in question.items() if not key.startswith("_")}
    public_question["target_shape"] = list(fields.velocity.shape)
    public_question["time"] = {
        "t0": float(query_grid["t"][0]),
        "t1": float(query_grid["t"][-1]),
        "nt": int(len(query_grid["t"])),
        "values": [float(value) for value in query_grid["t"]],
    }
    (out_case_dir / "public_question.json").write_text(json.dumps(public_question, indent=2, sort_keys=True) + "\n")
    return _load_public_question(out_case_dir)


def _call_solver(module, public_case_dir: Path, question: dict[str, Any], query_grid: dict[str, np.ndarray]):
    if hasattr(module, "solve_case"):
        try:
            return module.solve_case(str(public_case_dir), query_grid=query_grid)
        except TypeError:
            return module.solve_case(str(public_case_dir))
    return module.solve(question)


def _extract_velocity(output, *, target_shape: tuple[int, int, int, int]) -> np.ndarray:
    value = output
    if isinstance(output, dict):
        for key in ("velocity", "prediction", "u", "y_pred", "answer"):
            if key in output:
                value = output[key]
                break
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == target_shape:
        return arr
    if arr.ndim == 4 and arr.shape == (target_shape[0], target_shape[2], target_shape[3], target_shape[1]):
        return np.moveaxis(arr, -1, 1)
    if arr.shape == target_shape[1:]:
        return np.repeat(arr[None, ...], target_shape[0], axis=0)
    raise ValueError(f"velocity shape {arr.shape} does not match target {target_shape}")


def _query_grid_from_public(public_arrays: dict[str, np.ndarray], *, nx: int, ny: int, nt: int) -> dict[str, np.ndarray]:
    missing = sorted({"x", "y", "t"} - set(public_arrays))
    if missing:
        raise ValueError(f"continuous public case is missing coordinate axes: {missing}")
    return {
        "t": _requested_axis(np.asarray(public_arrays["t"], dtype=np.float64), nt),
        "x": _requested_axis(np.asarray(public_arrays["x"], dtype=np.float64), nx),
        "y": _requested_axis(np.asarray(public_arrays["y"], dtype=np.float64), ny),
    }


def _query_grid(public_arrays: dict[str, np.ndarray], truth: np.ndarray, *, nx: int, ny: int, nt: int) -> dict[str, np.ndarray]:
    old_t = np.asarray(public_arrays.get("t", np.arange(truth.shape[0])), dtype=np.float64)
    old_x = np.asarray(public_arrays.get("x", np.arange(truth.shape[2])), dtype=np.float64)
    old_y = np.asarray(public_arrays.get("y", np.arange(truth.shape[3])), dtype=np.float64)
    return {
        "t": _requested_axis(old_t, nt),
        "x": _requested_axis(old_x, nx),
        "y": _requested_axis(old_y, ny),
    }


def _requested_axis(old_axis: np.ndarray, requested: int) -> np.ndarray:
    old = np.asarray(old_axis, dtype=np.float64).reshape(-1)
    if requested <= 0 or requested == old.size:
        return old.copy()
    if old.size == 0:
        return np.arange(int(requested), dtype=np.float64)
    if old.size == 1:
        return np.full(int(requested), float(old[0]), dtype=np.float64)
    return np.linspace(float(old[0]), float(old[-1]), int(requested), dtype=np.float64)


def _resample_velocity_to_grid(
    velocity: np.ndarray,
    *,
    old_t: np.ndarray,
    old_x: np.ndarray,
    old_y: np.ndarray,
    new_t: np.ndarray,
    new_x: np.ndarray,
    new_y: np.ndarray,
) -> np.ndarray:
    out = np.asarray(velocity, dtype=np.float64)
    out = _interp_axis(out, old_t, new_t, axis=0)
    out = _interp_axis(out, old_x, new_x, axis=2)
    out = _interp_axis(out, old_y, new_y, axis=3)
    return out


def _interp_axis(arr: np.ndarray, old_axis: np.ndarray, new_axis: np.ndarray, *, axis: int) -> np.ndarray:
    old = np.asarray(old_axis, dtype=np.float64).reshape(-1)
    new = np.asarray(new_axis, dtype=np.float64).reshape(-1)
    if old.size == new.size and np.allclose(old, new, rtol=1.0e-12, atol=1.0e-12):
        return np.asarray(arr, dtype=np.float64)
    moved = np.moveaxis(np.asarray(arr, dtype=np.float64), axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    interp = np.empty((new.size, flat.shape[1]), dtype=np.float64)
    for idx in range(flat.shape[1]):
        interp[:, idx] = np.interp(new, old, flat[:, idx])
    out = interp.reshape((new.size, *moved.shape[1:]))
    return np.moveaxis(out, 0, axis)


def _resample_mask_to_grid(
    mask: np.ndarray,
    *,
    old_x: np.ndarray,
    old_y: np.ndarray,
    new_x: np.ndarray,
    new_y: np.ndarray,
) -> np.ndarray:
    old_x = np.asarray(old_x, dtype=np.float64).reshape(-1)
    old_y = np.asarray(old_y, dtype=np.float64).reshape(-1)
    new_x = np.asarray(new_x, dtype=np.float64).reshape(-1)
    new_y = np.asarray(new_y, dtype=np.float64).reshape(-1)
    ix = np.asarray([int(np.argmin(np.abs(old_x - value))) for value in new_x], dtype=np.int64)
    iy = np.asarray([int(np.argmin(np.abs(old_y - value))) for value in new_y], dtype=np.int64)
    return np.asarray(mask, dtype=bool)[np.ix_(ix, iy)]


def _resample_scalar_to_grid(
    field: np.ndarray,
    *,
    old_x: np.ndarray,
    old_y: np.ndarray,
    new_x: np.ndarray,
    new_y: np.ndarray,
) -> np.ndarray:
    out = _interp_axis(np.asarray(field, dtype=np.float64), old_x, new_x, axis=0)
    return _interp_axis(out, old_y, new_y, axis=1)


def _relative_l2_fluid(prediction: np.ndarray, truth: np.ndarray, fluid_mask: np.ndarray) -> float:
    return float(
        _velocity_error_metrics(prediction, truth, fluid_mask)["relative_l2"]
    )


def _velocity_error_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    fluid_mask: np.ndarray,
    *,
    velocity_std: list[float] | tuple[float, float] | None = None,
) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if pred.shape != target.shape or not np.all(np.isfinite(pred)):
        return {
            "relative_l2": float("inf"),
            "velocity_rmse": float("inf"),
            "truth_velocity_rms": float("nan"),
            "train_scaled_velocity_rmse": float("inf"),
        }
    fluid = np.asarray(fluid_mask, dtype=bool)
    if pred.ndim != 4 or pred.shape[1] != 2 or fluid.shape != pred.shape[2:] or not np.any(fluid):
        raise ValueError("velocity metrics require [time,2,x,y] fields and a non-empty fluid mask")
    diff = pred[:, :, fluid] - target[:, :, fluid]
    target_fluid = target[:, :, fluid]
    component_mse = np.mean(np.square(diff), axis=(0, 2))
    relative_l2 = float(np.linalg.norm(diff) / (np.linalg.norm(target_fluid) + 1.0e-12))
    velocity_rmse = float(np.sqrt(np.mean(np.square(diff))))
    truth_velocity_rms = float(np.sqrt(np.mean(np.square(target_fluid))))
    train_scaled = float("nan")
    if velocity_std is not None:
        scales = np.asarray(velocity_std, dtype=np.float64)
        if scales.shape != (2,) or not np.all(np.isfinite(scales)) or not np.all(scales > 0.0):
            raise ValueError("velocity_std must contain two finite positive train-split scales")
        train_scaled = float(np.sqrt(np.mean(component_mse / np.square(scales))))
    return {
        "relative_l2": relative_l2,
        "velocity_rmse": velocity_rmse,
        "truth_velocity_rms": truth_velocity_rms,
        "train_scaled_velocity_rmse": train_scaled,
    }


def _public_eval_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary_keys = (
        "n",
        "ok",
        "status_counts",
        "mean_relative_l2",
        "median_relative_l2",
        "min_relative_l2",
        "max_relative_l2",
        "q10_relative_l2",
        "q90_relative_l2",
        "mean_difficulty_reward",
        "median_difficulty_reward",
    )
    result_keys = (
        "case_id",
        "status",
        "relative_l2",
        "difficulty_reward",
        "reward_metric",
        "wall_time_seconds",
        "query_density",
        "exception",
    )
    summary = payload.get("summary") or {}
    return {
        "solver": payload.get("solver"),
        "cases_root": payload.get("cases_root"),
        "query_density": payload.get("query_density"),
        "summary": {key: summary[key] for key in summary_keys if key in summary},
        "results": [
            {key: row[key] for key in result_keys if key in row}
            for row in (payload.get("results") or [])
        ],
    }


def _jsonable(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


if __name__ == "__main__":
    evaluate_cli()
