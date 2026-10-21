#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from public_vpfp_runtime import evaluate_public_output, load_public_case, load_submission, run_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a VPFP submission using public equations and arrays only.")
    parser.add_argument("--submission", default="submission.py")
    parser.add_argument("--cases", default="cases")
    parser.add_argument("--out", default="public_eval.json")
    parser.add_argument("--fail-on-invalid", action="store_true")
    args = parser.parse_args()

    submission = load_submission(args.submission)
    rows = []
    for case_dir in sorted(path for path in Path(args.cases).iterdir() if path.is_dir()):
        try:
            case = load_public_case(case_dir)
            output = run_submission(submission, case_dir)
            rows.append(evaluate_public_output(case, output))
        except Exception as exc:  # noqa: BLE001 - surfaced as public solver diagnostics.
            rows.append(
                {
                    "case_id": case_dir.name,
                    "status": "exception",
                    "finite": False,
                    "exception": repr(exc),
                }
            )
    valid = [row for row in rows if row.get("status") == "ok"]
    summary = {
        "case_count": len(rows),
        "valid_case_count": len(valid),
        "all_valid": len(valid) == len(rows) and bool(rows),
        "mean_public_objective": _mean(valid, "public_objective_smaller_is_better"),
        "max_vpfp_residual_rel": _max(valid, "vpfp_residual_rel"),
        "max_g_normalization_error": _max(valid, "g_normalization_max_abs_error"),
    }
    payload = {"summary": summary, "cases": rows}
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.fail_on_invalid and not summary["all_valid"]:
        raise SystemExit("one or more public cases failed")


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else None


def _max(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
    return float(np.max(values)) if values else None


if __name__ == "__main__":
    main()
