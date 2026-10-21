#!/usr/bin/env python
"""Evaluate scalable VPFP submissions on released online resolution views."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from public_vpfp_runtime import (
        load_public_case,
        load_submission,
        run_submission,
    )
except ModuleNotFoundError:  # Package import used by repository tests.
    from agent_verifier.vpfp.public_vpfp_runtime import (
        load_public_case,
        load_submission,
        run_submission,
    )


FIELDS = ("rho", "momentum", "kinetic_energy")
ALIASES = {
    "rho": ("rho", "rho_pred"),
    "momentum": ("momentum", "j"),
    "kinetic_energy": ("kinetic_energy", "K"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", default="submission.py")
    parser.add_argument("--root", default="online_generator_outputs")
    parser.add_argument("--out", default="online_resolution_feedback.json")
    parser.add_argument("--fail-on-invalid", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = evaluate_resolution_views(
        submission=args.submission,
        root=args.root,
    )
    write_resolution_report(report, args.out)
    print_resolution_summary(report)
    if args.fail_on_invalid and not report["all_valid"]:
        raise SystemExit("one or more online resolution views failed")


def evaluate_resolution_views(
    *,
    submission: str | Path,
    root: str | Path = "online_generator_outputs",
    case_ids: set[str] | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest = _read_json(root / "manifest.json")
    records = _select_resolution_records(manifest, case_ids=case_ids)
    module = load_submission(submission)
    rows = []
    for record in records:
        case_dir = root / str(record["case_dir"])
        started = time.perf_counter()
        try:
            case = load_public_case(case_dir)
            output = run_submission(module, case_dir)
            prediction = _coerce_prediction(case, output)
            with np.load(case_dir / "answer.npz", allow_pickle=False) as payload:
                truth = {
                    name: np.asarray(payload[name], dtype=np.float64)
                    for name in FIELDS
                }
            metrics = {
                f"{name}_rel_l2": _relative_l2(prediction[name], truth[name])
                for name in FIELDS
            }
            score = float(np.mean(list(metrics.values())))
            rows.append(
                {
                    "case_id": str(record["case_id"]),
                    "status": "ok",
                    "root_case_id": str(record["root_case_id"]),
                    "study_id": str(record.get("study_id") or ""),
                    "dimension": str(record.get("dimension") or ""),
                    "level": int(record.get("level", 0)),
                    "characteristic_spacing": float(
                        record.get("characteristic_spacing", 1.0)
                    ),
                    "resolution": dict(record.get("resolution") or {}),
                    "score_smaller_is_better": score,
                    "wall_time_seconds": float(time.perf_counter() - started),
                    **metrics,
                }
            )
        except Exception as exc:  # noqa: BLE001 - development feedback retains failures.
            rows.append(
                {
                    "case_id": str(record.get("case_id") or ""),
                    "status": "exception",
                    "root_case_id": str(record.get("root_case_id") or ""),
                    "study_id": str(record.get("study_id") or ""),
                    "dimension": str(record.get("dimension") or ""),
                    "level": int(record.get("level", 0)),
                    "characteristic_spacing": float(
                        record.get("characteristic_spacing", 1.0)
                    ),
                    "resolution": dict(record.get("resolution") or {}),
                    "score_smaller_is_better": float("inf"),
                    "wall_time_seconds": float(time.perf_counter() - started),
                    "exception": repr(exc),
                }
            )
    groups = _convergence_groups(rows)
    group_summary = _convergence_group_summary(groups)
    valid = [row for row in rows if row["status"] == "ok"]
    report = {
        "schema": "vpfp_online_resolution_development_feedback_v2",
        "truth_semantics": (
            "independently joint-refined equation-first numerical macro answers"
        ),
        "selection_or_final_data": False,
        "case_count": len(rows),
        "valid_case_count": len(valid),
        "all_valid": bool(rows) and len(valid) == len(rows),
        "mean_score": (
            float(np.mean([row["score_smaller_is_better"] for row in valid]))
            if valid
            else None
        ),
        **group_summary,
        "groups": groups,
        "cases": rows,
    }
    return report


def write_resolution_report(
    report: dict[str, Any], out: str | Path
) -> None:
    Path(out).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def print_resolution_summary(report: dict[str, Any]) -> None:
    print(json.dumps({key: report[key] for key in (
        "schema",
        "case_count",
        "valid_case_count",
        "all_valid",
        "mean_score",
        "group_count",
        "pairwise_evaluable_group_count",
        "pairwise_monotone_group_count",
        "convergence_group_count",
        "monotone_group_count",
    )}, indent=2, sort_keys=True))


def _select_resolution_records(
    manifest: dict[str, Any], *, case_ids: set[str] | None
) -> list[dict[str, Any]]:
    selected = [
        dict(record)
        for record in manifest.get("cases", [])
        if bool(record.get("resolution_view"))
        and bool(record.get("numerical_answer_released"))
        and (
            case_ids is None
            or str(record.get("case_id") or "") in case_ids
        )
    ]
    if case_ids is not None:
        observed = {str(record.get("case_id") or "") for record in selected}
        missing = sorted(case_ids - observed)
        if missing:
            raise ValueError(
                "requested resolution views are absent or have no released answer: "
                + ", ".join(missing)
            )
    return selected


def _coerce_prediction(
    case: dict[str, Any],
    output: Any,
) -> dict[str, np.ndarray]:
    if not isinstance(output, dict):
        raise TypeError("solver output must be a mapping")
    result = {}
    expected_shapes = dict(case["question"]["target_shapes"])
    for name in FIELDS:
        value = next(
            (output[alias] for alias in ALIASES[name] if output.get(alias) is not None),
            None,
        )
        if value is None:
            raise ValueError(f"solver output is missing {name}")
        array = np.asarray(value, dtype=np.float64)
        expected = tuple(int(item) for item in expected_shapes[name])
        if array.shape != expected:
            raise ValueError(f"{name} shape {array.shape} != {expected}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")
        result[name] = array
    return result


def _relative_l2(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(prediction) - np.asarray(truth))
        / (np.linalg.norm(np.asarray(truth)) + 1.0e-12)
    )


def _convergence_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["status"] != "ok":
            continue
        key = (
            str(row["root_case_id"]),
            str(row["study_id"]),
        )
        grouped[key].append(row)
    result = []
    for (root_case_id, study_id), values in sorted(grouped.items()):
        nonbase_dimensions = sorted(
            {
                str(row["dimension"])
                for row in values
                if str(row["dimension"]).lower() not in {"", "base"}
            }
        )
        if len(nonbase_dimensions) == 1:
            dimension = nonbase_dimensions[0]
        elif nonbase_dimensions:
            dimension = "mixed"
        else:
            dimension = "base"
        ordered = sorted(
            values,
            key=lambda row: float(row["characteristic_spacing"]),
            reverse=True,
        )
        spacing = [float(row["characteristic_spacing"]) for row in ordered]
        errors = [float(row["score_smaller_is_better"]) for row in ordered]
        orders = [
            _observed_order(errors[index], errors[index + 1], spacing[index], spacing[index + 1])
            for index in range(len(errors) - 1)
        ]
        pairwise_evaluable = len(ordered) >= 2
        convergence_evaluable = len(ordered) >= 3
        monotone = (
            all(
                fine <= coarse * (1.0 + 1.0e-8)
                for coarse, fine in zip(errors, errors[1:], strict=False)
            )
            if pairwise_evaluable
            else None
        )
        result.append(
            {
                "root_case_id": root_case_id,
                "study_id": study_id,
                "dimension": dimension,
                "level_count": len(ordered),
                "case_ids": [str(row["case_id"]) for row in ordered],
                "characteristic_spacings": spacing,
                "aggregate_errors": errors,
                "pairwise_observed_orders": orders,
                "pairwise_evaluable": pairwise_evaluable,
                "convergence_evaluable": convergence_evaluable,
                "error_monotone_nonincreasing": monotone,
            }
        )
    return result


def _convergence_group_summary(
    groups: list[dict[str, Any]],
) -> dict[str, int]:
    pairwise = [group for group in groups if bool(group["pairwise_evaluable"])]
    convergence = [
        group for group in groups if bool(group["convergence_evaluable"])
    ]
    return {
        "group_count": len(groups),
        "pairwise_evaluable_group_count": len(pairwise),
        "pairwise_monotone_group_count": sum(
            group["error_monotone_nonincreasing"] is True
            for group in pairwise
        ),
        "convergence_group_count": len(convergence),
        "monotone_group_count": sum(
            group["error_monotone_nonincreasing"] is True
            for group in convergence
        ),
    }


def _observed_order(
    coarse_error: float,
    fine_error: float,
    coarse_spacing: float,
    fine_spacing: float,
) -> float | None:
    if min(coarse_error, fine_error, coarse_spacing, fine_spacing) <= 0.0:
        return None
    denominator = math.log(coarse_spacing / fine_spacing)
    if abs(denominator) <= 1.0e-14:
        return None
    return float(math.log(coarse_error / fine_error) / denominator)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
