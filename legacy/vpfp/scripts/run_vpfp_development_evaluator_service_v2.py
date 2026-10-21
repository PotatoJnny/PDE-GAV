#!/usr/bin/env python
"""Serve controller-private fresh paired VPFP development evaluations."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_verifier.storage_lifecycle import is_within_active_slurm_tmp
from agent_verifier.vpfp.development_evaluation import (
    VPFPDevelopmentEvaluationExchangeController,
)
from agent_verifier.vpfp.development_runtime_v2 import (
    VPFPDevelopmentEvaluatorRuntime,
    VPFPDevelopmentRuntimeConfig,
)
from agent_verifier.vpfp.runtime_paths import is_persistent_run_path


CONFIG_SCHEMA = "vpfp_development_evaluator_service_config_v2"
REPORT_SCHEMA = "vpfp_development_evaluator_service_report_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stop-file", required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--maximum-consecutive-failures", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_TMPDIR"):
        raise SystemExit("VPFP evaluator service requires a Slurm compute node")
    if not os.environ.get("VIRTUAL_ENV"):
        raise SystemExit("VPFP evaluator service requires the cluster venv")
    if args.poll_seconds <= 0.0 or args.maximum_consecutive_failures < 1:
        raise SystemExit("invalid VPFP evaluator service retry schedule")

    config_path = Path(args.config).resolve()
    config = _read_json(config_path)
    if config.get("schema") != CONFIG_SCHEMA:
        raise SystemExit("unexpected VPFP evaluator service config schema")
    workspace = Path(config["workspace"]).resolve()
    private_root = Path(config["private_root"]).resolve()
    stop_file = Path(args.stop_file).resolve()
    if not (
        is_persistent_run_path(workspace)
        and (
            is_persistent_run_path(private_root)
            or is_within_active_slurm_tmp(private_root)
        )
        and (
            is_persistent_run_path(stop_file)
            or is_within_active_slurm_tmp(stop_file)
        )
    ):
        raise SystemExit(
            "VPFP evaluator workspace needs configured run storage and runtime "
            "state needs run storage or active SLURM_TMPDIR"
        )
    if private_root.exists():
        raise FileExistsError(private_root)

    runtime = VPFPDevelopmentEvaluatorRuntime(
        workspace=workspace,
        private_root=private_root,
        base_decoder_checkpoint=config["base_decoder_checkpoint"],
        expanded_decoder_checkpoint=config["expanded_decoder_checkpoint"],
        comparison_solver=config["comparison_solver"],
        checkpoint_index=int(config["checkpoint_index"]),
        config=VPFPDevelopmentRuntimeConfig(**dict(config["runtime_config"])),
        device=str(config.get("device", "cuda")),
    )
    exchange = VPFPDevelopmentEvaluationExchangeController(workspace)
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.time()
    processed_total = 0
    consecutive_failures = 0
    failure_rows: list[dict[str, Any]] = []
    status_path = private_root / "service_status.json"
    _write_json(
        status_path,
        _status_payload(
            state="ready",
            workspace=workspace,
            processed_total=0,
            pending_count=_pending_count(workspace),
            failure_count=0,
        ),
    )

    fatal: BaseException | None = None
    try:
        while True:
            stop_requested = stop_requested or stop_file.is_file()
            pending = _pending_count(workspace)
            if stop_requested and pending == 0:
                break
            try:
                processed = exchange.process_once(runtime.handler)
            except BaseException as exc:
                consecutive_failures += 1
                failure_rows.append(
                    {
                        "time_unix": time.time(),
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc)[:1000],
                    }
                )
                _write_json(
                    status_path,
                    _status_payload(
                        state="retrying_after_failure",
                        workspace=workspace,
                        processed_total=processed_total,
                        pending_count=_pending_count(workspace),
                        failure_count=len(failure_rows),
                    ),
                )
                if consecutive_failures >= int(args.maximum_consecutive_failures):
                    raise
                time.sleep(float(args.poll_seconds))
                continue
            consecutive_failures = 0
            processed_total += int(processed)
            _write_json(
                status_path,
                _status_payload(
                    state="draining" if stop_requested else "ready",
                    workspace=workspace,
                    processed_total=processed_total,
                    pending_count=_pending_count(workspace),
                    failure_count=len(failure_rows),
                ),
            )
            if processed == 0:
                time.sleep(float(args.poll_seconds))
    except BaseException as exc:
        fatal = exc

    report = {
        "schema": REPORT_SCHEMA,
        "completed": fatal is None,
        "workspace": str(workspace),
        "private_root": str(private_root),
        "processed_request_count": int(processed_total),
        "pending_request_count": _pending_count(workspace),
        "failure_count": len(failure_rows),
        "failures": failure_rows,
        "fatal_exception": (
            None
            if fatal is None
            else {
                "type": type(fatal).__name__,
                "message": str(fatal)[:2000],
            }
        ),
        "accounting": exchange.accounting(),
        "runtime_manifest": runtime.runtime_manifest,
        "stop_requested": bool(stop_requested),
        "wall_time_seconds": float(time.time() - started),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _write_json(private_root / "service_report.json", report)
    _write_json(
        status_path,
        _status_payload(
            state="completed" if fatal is None else "failed",
            workspace=workspace,
            processed_total=processed_total,
            pending_count=_pending_count(workspace),
            failure_count=len(failure_rows),
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if fatal is not None:
        raise SystemExit(
            f"VPFP evaluator service failed: {type(fatal).__name__}: {fatal}"
        )


def _pending_count(workspace: Path) -> int:
    requests = workspace / "evaluation_exchange" / "requests"
    responses = workspace / "evaluation_exchange" / "responses"
    return sum(
        1
        for path in requests.glob("*.json")
        if path.is_file() and not (responses / path.name).is_file()
    )


def _status_payload(
    *,
    state: str,
    workspace: Path,
    processed_total: int,
    pending_count: int,
    failure_count: int,
) -> dict[str, Any]:
    return {
        "schema": "vpfp_development_evaluator_service_status_v2",
        "state": state,
        "workspace": str(workspace),
        "processed_request_count": int(processed_total),
        "pending_request_count": int(pending_count),
        "failure_count": int(failure_count),
        "updated_at_unix": time.time(),
    }


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
