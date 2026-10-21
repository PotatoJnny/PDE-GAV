#!/usr/bin/env python
"""Run one VPFP compute-budget solver call in a fresh Python process."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from agent_verifier.vpfp.compute_budget import (
    SolverComputeBudget,
    run_budgeted_submission,
)
from agent_verifier.vpfp.public_vpfp_runtime import (
    load_public_case,
    load_submission,
)


REPORT_SCHEMA = "vpfp_budgeted_solver_subprocess_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--query-times", required=True)
    parser.add_argument("--budget", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = Path(args.report).resolve()
    output_path = Path(args.output).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {
        "schema": REPORT_SCHEMA,
        "status": "exception",
        "accepted": False,
        "wall_time_seconds": None,
        "worker_wall_time_seconds": None,
        "exception_type": None,
        "exception_message": None,
        "pid": os.getpid(),
    }
    try:
        budget = SolverComputeBudget.from_dict(
            json.loads(Path(args.budget).read_text(encoding="utf-8"))
        )
        query_times = np.load(Path(args.query_times), allow_pickle=False)
        public_case = load_public_case(Path(args.case))
        module = load_submission(Path(args.solver))
        prediction, wall_seconds = run_budgeted_submission(
            module,
            public_case,
            query_times,
            budget,
        )
        np.savez_compressed(output_path, **prediction)
        report.update(
            {
                "status": "ok",
                "accepted": True,
                "wall_time_seconds": float(wall_seconds),
            }
        )
    except BaseException as exc:  # noqa: BLE001 - failure is an audited result.
        report.update(
            {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc)[:1000],
            }
        )
    report["worker_wall_time_seconds"] = float(time.perf_counter() - started)
    temporary = report_path.with_suffix(report_path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)


if __name__ == "__main__":
    main()
