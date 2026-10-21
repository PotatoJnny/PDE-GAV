#!/usr/bin/env python
"""Evaluate a VPFP macro solver as its actual compute budget increases."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

try:
    from online_generator import ONLINE_GENERATOR_REQUEST_SCHEMA, send_request
    from public_vpfp_runtime import load_public_case, load_submission
except ModuleNotFoundError:  # Package imports used by repository tests.
    from agent_verifier.vpfp.public_online_generator_cli import (
        ONLINE_GENERATOR_REQUEST_SCHEMA,
        send_request,
    )
    from agent_verifier.vpfp.public_vpfp_runtime import (
        load_public_case,
        load_submission,
    )

try:
    from compute_budget import (
        BUDGET_RESPONSE_REPORT_SCHEMA,
        MACRO_FIELDS,
        coerce_budget_output,
        interpolate_macro_times,
        macro_error_metrics,
        make_budget_schedule,
        run_budgeted_submission,
    )
except ModuleNotFoundError:  # Package imports used by repository tests.
    from agent_verifier.vpfp.compute_budget import (
        BUDGET_RESPONSE_REPORT_SCHEMA,
        MACRO_FIELDS,
        coerce_budget_output,
        interpolate_macro_times,
        macro_error_metrics,
        make_budget_schedule,
        run_budgeted_submission,
    )


STUDY_ID = "fresh_compute_budget_feedback"
RequestSender = Callable[[dict[str, Any]], dict[str, Any]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", default="submission.py")
    parser.add_argument("--champion", default="champion_solver/submission.py")
    parser.add_argument("--roots", type=int, default=4)
    parser.add_argument("--dense-nt", type=int, default=25)
    parser.add_argument("--query-time-count", type=int, default=9)
    parser.add_argument(
        "--budget-seconds",
        type=float,
        nargs="+",
        default=[15.0, 60.0, 240.0],
    )
    parser.add_argument(
        "--condition-pool", choices=("train", "validation"), default="validation"
    )
    parser.add_argument("--minimum-endpoint-relative-gain", type=float, default=0.0)
    parser.add_argument("--maximum-high-budget-champion-ratio", type=float, default=1.02)
    parser.add_argument("--root", default="online_generator_outputs")
    parser.add_argument("--out", default="compute_budget_feedback.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_budget_feedback(
        submission=args.submission,
        champion=args.champion,
        root=args.root,
        out=args.out,
        root_count=int(args.roots),
        dense_nt=int(args.dense_nt),
        query_time_count=int(args.query_time_count),
        budget_seconds=[float(value) for value in args.budget_seconds],
        condition_pool=str(args.condition_pool),
        minimum_endpoint_relative_gain=float(
            args.minimum_endpoint_relative_gain
        ),
        maximum_high_budget_champion_ratio=float(
            args.maximum_high_budget_champion_ratio
        ),
    )
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "passed": report["summary"]["passed"],
                "root_count": report["summary"]["root_count"],
                "budget_seconds": report["summary"]["budget_seconds"],
                "mean_aggregate_errors": report["summary"][
                    "mean_aggregate_errors"
                ],
                "endpoint_relative_gain": report["summary"][
                    "endpoint_relative_gain"
                ],
                "high_budget_champion_ratio": report["summary"][
                    "high_budget_champion_ratio"
                ],
                "report": str(Path(args.out)),
            },
            indent=2,
            sort_keys=True,
        )
    )


def run_budget_feedback(
    *,
    submission: str | Path,
    champion: str | Path | None,
    root: str | Path,
    out: str | Path,
    root_count: int,
    dense_nt: int,
    query_time_count: int,
    budget_seconds: list[float],
    condition_pool: str = "validation",
    minimum_endpoint_relative_gain: float = 0.0,
    maximum_high_budget_champion_ratio: float = 1.02,
    request_sender: RequestSender = send_request,
) -> dict[str, Any]:
    status = _checked_request(
        {"schema": ONLINE_GENERATOR_REQUEST_SCHEMA, "operation": "status"},
        request_sender=request_sender,
    )
    budgets = dict(status.get("budgets") or {})
    max_batch = int(budgets["max_batch_per_call"])
    if root_count < 1 or root_count > max_batch:
        raise ValueError(f"roots must lie in [1, {max_batch}]")
    allocation = dict(budgets.get("declared_allocation") or {})
    schedule = make_budget_schedule(
        budget_seconds,
        cpus=int(allocation.get("cpus") or os.environ.get("SLURM_CPUS_PER_TASK") or 1),
        gpus=int(allocation.get("gpus") or (1 if os.environ.get("CUDA_VISIBLE_DEVICES") else 0)),
        memory_bytes=int(allocation.get("memory_bytes") or 1),
    )
    sampled = _checked_request(
        {
            "schema": ONLINE_GENERATOR_REQUEST_SCHEMA,
            "operation": "sample",
            "count": int(root_count),
            "mode": "base",
            "answers": "none",
            "condition_pool": str(condition_pool),
        },
        request_sender=request_sender,
    )
    roots = [dict(row) for row in sampled.get("cases", [])]
    if len(roots) != root_count:
        raise RuntimeError(f"online service returned {len(roots)} roots")
    root_path = Path(root).resolve()
    workspace = root_path.parent
    direct_continuous_truth = (
        str(status.get("primary_truth") or "") == "continuous hidden answer"
    )
    materialization_requests = []
    query_times_by_root: dict[str, np.ndarray] = {}
    for root_index, row in enumerate(roots):
        resolution = {
            name: int(value)
            for name, value in dict(row["resolution"]).items()
        }
        resolution["nt"] = int(dense_nt)
        request = {
            "case_id": str(row["case_id"]),
            "answers": "grid",
            "resolution": resolution,
            "study_id": STUDY_ID,
            "level": 0,
        }
        if direct_continuous_truth:
            if not row.get("case_dir"):
                raise RuntimeError(
                    "answer-first sample omitted its public numerical view"
                )
            root_public_case = load_public_case(
                workspace / str(row["case_dir"])
            )
            query_times = _query_times(
                root_public_case, query_time_count, root_index
            )
            query_times_by_root[str(row["root_case_id"])] = query_times
            request["truth_times"] = query_times.tolist()
        materialization_requests.append(request)
    materialized = _checked_request(
        {
            "schema": ONLINE_GENERATOR_REQUEST_SCHEMA,
            "operation": "materialize_batch",
            "requests": materialization_requests,
        },
        request_sender=request_sender,
    )
    views = [dict(row) for row in materialized.get("cases", [])]
    if len(views) != root_count:
        raise RuntimeError(f"online service materialized {len(views)} views")

    candidate_module = load_submission(submission)
    champion_path = None if champion is None else Path(champion).resolve()
    champion_module = (
        load_submission(champion_path)
        if champion_path is not None and champion_path.is_file()
        else None
    )
    rows = []
    champion_rows = []
    query_records = []
    for root_index, view in enumerate(views):
        case_dir = workspace / str(view["case_dir"])
        public_case = load_public_case(case_dir)
        root_case_id = str(view["root_case_id"])
        query_times = (
            query_times_by_root[root_case_id]
            if direct_continuous_truth
            else _query_times(public_case, query_time_count, root_index)
        )
        exact_relative = view.get("exact_macro_answer")
        if direct_continuous_truth and not exact_relative:
            raise RuntimeError(
                "answer-first service omitted direct continuous macro truth"
            )
        truth_mode = (
            "direct_continuous_macro"
            if exact_relative
            else "legacy_time_interpolation"
        )
        truth_path = (
            None
            if not exact_relative
            else workspace / str(exact_relative)
        )
        if exact_relative:
            returned_times = np.asarray(view.get("truth_times"), dtype=np.float64)
            if returned_times.shape != query_times.shape or not np.allclose(
                returned_times, query_times, rtol=0.0, atol=1.0e-13
            ):
                raise RuntimeError("answer-first truth times changed in transit")
        truth = _load_interpolated_truth(
            case_dir,
            public_case,
            query_times,
            truth_path=truth_path,
            truth_mode=truth_mode,
        )
        query_records.append(
            {
                "case_id": str(view["case_id"]),
                "root_case_id": root_case_id,
                "query_times": query_times.tolist(),
                "truth_mode": truth_mode,
                "truth_path": (
                    None if truth_path is None else str(truth_path.resolve())
                ),
            }
        )
        for budget in schedule:
            row = {
                "case_id": str(view["case_id"]),
                "root_case_id": str(view["root_case_id"]),
                "budget": budget.to_dict(),
            }
            try:
                prediction, elapsed = run_budgeted_submission(
                    candidate_module, public_case, query_times, budget
                )
                row.update(
                    {
                        "status": "ok",
                        "wall_time_seconds": elapsed,
                        **{
                            f"{name}_rel_l2": value
                            for name, value in macro_error_metrics(
                                prediction, truth
                            ).items()
                        },
                    }
                )
            except Exception as exc:  # noqa: BLE001 - feedback retains failures.
                row.update(
                    {
                        "status": "exception",
                        "wall_time_seconds": None,
                        "exception": repr(exc),
                    }
                )
            rows.append(row)
        if champion_module is not None:
            champion_rows.append(
                _evaluate_champion(
                    champion_module,
                    public_case=public_case,
                    case_dir=case_dir,
                    query_times=query_times,
                    truth=truth,
                    high_budget=schedule[-1],
                )
            )
    summary = summarize_budget_feedback(
        rows,
        champion_rows=champion_rows,
        root_count=root_count,
        schedule=[row.to_dict() for row in schedule],
        minimum_endpoint_relative_gain=minimum_endpoint_relative_gain,
        maximum_high_budget_champion_ratio=maximum_high_budget_champion_ratio,
    )
    report = {
        "schema": BUDGET_RESPONSE_REPORT_SCHEMA,
        "selection_or_final_data": False,
        "primary_truth": status.get("primary_truth"),
        "truth_semantics": (
            "direct full-velocity-space moments of the same continuous hidden "
            "answer at requested times; no equation re-solve or time interpolation"
            if direct_continuous_truth
            else (
                "dense numerical macro answer evaluated at arbitrary requested "
                "times by cubic time interpolation"
            )
        ),
        "candidate_interpolation_owned_by_submission": True,
        "internal_discretization_owned_by_submission": True,
        "pairwise_monotonicity_is_a_hard_gate": False,
        "condition_pool": str(condition_pool),
        "query_records": query_records,
        "summary": summary,
        "cases": rows,
        "champion_cases": champion_rows,
        "budget_usage": dict(materialized.get("budgets") or {}),
    }
    output = Path(out)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def summarize_budget_feedback(
    rows: list[dict[str, Any]],
    *,
    champion_rows: list[dict[str, Any]],
    root_count: int,
    schedule: list[dict[str, Any]],
    minimum_endpoint_relative_gain: float,
    maximum_high_budget_champion_ratio: float,
) -> dict[str, Any]:
    tiers = [int(row["tier"]) for row in schedule]
    tier_summaries = []
    for tier in tiers:
        selected = [
            row
            for row in rows
            if int(dict(row["budget"])["tier"]) == tier
            and row.get("status") == "ok"
        ]
        complete = len(selected) == root_count
        aggregates = [float(row["aggregate_rel_l2"]) for row in selected]
        tier_summaries.append(
            {
                "tier": tier,
                "wall_time_seconds": float(schedule[tier]["wall_time_seconds"]),
                "valid_root_count": len(selected),
                "complete": complete,
                "aggregate_error_mean": float(np.mean(aggregates)) if complete else None,
                "aggregate_error_median": float(np.median(aggregates)) if complete else None,
                "aggregate_error_q90": float(np.quantile(aggregates, 0.9)) if complete else None,
                "mean_field_errors": {
                    field: (
                        float(
                            np.mean(
                                [float(row[f"{field}_rel_l2"]) for row in selected]
                            )
                        )
                        if complete
                        else None
                    )
                    for field in MACRO_FIELDS
                },
                "mean_actual_wall_time_seconds": (
                    float(np.mean([float(row["wall_time_seconds"]) for row in selected]))
                    if complete
                    else None
                ),
            }
        )
    complete = all(bool(row["complete"]) for row in tier_summaries)
    mean_errors = [row["aggregate_error_mean"] for row in tier_summaries]
    low = None if not complete else float(mean_errors[0])
    high = None if not complete else float(mean_errors[-1])
    endpoint_gain = (
        None if low is None or high is None else float((low - high) / max(low, 1.0e-30))
    )
    endpoint_passed = bool(
        endpoint_gain is not None
        and endpoint_gain > float(minimum_endpoint_relative_gain)
    )
    valid_champion = [
        row for row in champion_rows if row.get("status") == "ok"
    ]
    champion_mean = (
        float(np.mean([float(row["aggregate_rel_l2"]) for row in valid_champion]))
        if len(valid_champion) == root_count
        else None
    )
    champion_ratio = (
        None
        if high is None or champion_mean is None
        else float(high / max(champion_mean, 1.0e-30))
    )
    absolute_passed = bool(
        champion_ratio is not None
        and champion_ratio <= float(maximum_high_budget_champion_ratio)
    )
    root_endpoint_gains = []
    if complete:
        root_ids = sorted({str(row["root_case_id"]) for row in rows})
        for root_case_id in root_ids:
            by_tier = {
                int(dict(row["budget"])["tier"]): float(row["aggregate_rel_l2"])
                for row in rows
                if row.get("status") == "ok"
                and str(row["root_case_id"]) == root_case_id
            }
            root_endpoint_gains.append(
                {
                    "root_case_id": root_case_id,
                    "relative_gain": float(
                        (by_tier[tiers[0]] - by_tier[tiers[-1]])
                        / max(by_tier[tiers[0]], 1.0e-30)
                    ),
                }
            )
    response_slope = None
    if complete and all(float(value) > 0.0 for value in mean_errors):
        response_slope = float(
            np.polyfit(
                np.log([float(row["wall_time_seconds"]) for row in schedule]),
                np.log([float(value) for value in mean_errors]),
                1,
            )[0]
        )
    return {
        "root_count": int(root_count),
        "budget_seconds": [float(row["wall_time_seconds"]) for row in schedule],
        "all_outputs_valid": bool(complete),
        "tier_summaries": tier_summaries,
        "mean_aggregate_errors": mean_errors,
        "endpoint_relative_gain": endpoint_gain,
        "minimum_endpoint_relative_gain": float(minimum_endpoint_relative_gain),
        "endpoint_gain_passed": endpoint_passed,
        "root_endpoint_gains_diagnostic": root_endpoint_gains,
        "log_error_vs_log_budget_slope_diagnostic": response_slope,
        "champion_high_budget_mean_aggregate_error": champion_mean,
        "high_budget_champion_ratio": champion_ratio,
        "maximum_high_budget_champion_ratio": float(
            maximum_high_budget_champion_ratio
        ),
        "absolute_high_budget_passed": absolute_passed,
        "pairwise_budget_monotonicity_required": False,
        "passed": bool(complete and endpoint_passed and absolute_passed),
    }


def _query_times(public_case: Mapping[str, Any], count: int, root_index: int) -> np.ndarray:
    if count < 3:
        raise ValueError("query-time count must be at least three")
    t0 = float(public_case["time"].t0)
    t1 = float(public_case["time"].t1)
    unit = np.linspace(0.0, 1.0, count, dtype=np.float64)
    interior = unit[1:-1]
    phase = 0.37 * float(root_index + 1)
    interior += (
        0.21
        * np.sin(np.arange(1, count - 1, dtype=np.float64) * 1.618 + phase)
        / float(count - 1)
    )
    unit[1:-1] = np.sort(np.clip(interior, 1.0e-6, 1.0 - 1.0e-6))
    return t0 + (t1 - t0) * unit


def _load_interpolated_truth(
    case_dir: Path,
    public_case: Mapping[str, Any],
    query_times: np.ndarray,
    *,
    truth_path: str | Path | None = None,
    truth_mode: str = "legacy_time_interpolation",
) -> dict[str, np.ndarray]:
    if truth_mode == "direct_continuous_macro":
        if truth_path is None:
            raise ValueError("direct continuous truth requires an exact truth path")
        with np.load(Path(truth_path).resolve(), allow_pickle=False) as payload:
            fields = {
                field: np.asarray(payload[field], dtype=np.float64)
                for field in MACRO_FIELDS
            }
        for field, values in fields.items():
            if int(values.shape[0]) != int(query_times.size):
                raise ValueError(
                    f"exact truth {field} has the wrong time dimension"
                )
        return fields
    if truth_mode != "legacy_time_interpolation":
        raise ValueError(f"unknown truth mode: {truth_mode}")
    with np.load(case_dir / "answer.npz", allow_pickle=False) as payload:
        fields = {
            field: np.asarray(payload[field], dtype=np.float64)
            for field in MACRO_FIELDS
        }
    return interpolate_macro_times(
        np.asarray(public_case["arrays"]["t"], dtype=np.float64),
        fields,
        query_times,
    )


def _evaluate_champion(
    module: Any,
    *,
    public_case: Mapping[str, Any],
    case_dir: Path,
    query_times: np.ndarray,
    truth: Mapping[str, Any],
    high_budget: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": str(public_case["case_id"]),
        "root_case_id": str(public_case["case_id"]),
    }
    started = time.perf_counter()
    try:
        if callable(getattr(module, "solve_with_budget", None)):
            prediction, elapsed = run_budgeted_submission(
                module, public_case, query_times, high_budget
            )
        else:
            if callable(getattr(module, "solve", None)):
                output = module.solve(public_case)
            elif callable(getattr(module, "solve_at_resolution", None)):
                output = module.solve_at_resolution(
                    public_case,
                    {
                        "nx": int(public_case["grid"].nx),
                        "ny": int(public_case["grid"].ny),
                        "nvx": int(public_case["grid"].nvx),
                        "nvy": int(public_case["grid"].nvy),
                        "nt": int(public_case["time"].nt),
                    },
                )
            elif callable(getattr(module, "solve_case", None)):
                output = module.solve_case(str(case_dir))
            else:
                raise ValueError("champion has no supported solver entrypoint")
            dense_times = np.asarray(public_case["arrays"]["t"], dtype=np.float64)
            dense = coerce_budget_output(output, public_case, dense_times)
            prediction = interpolate_macro_times(dense_times, dense, query_times)
            elapsed = float(time.perf_counter() - started)
        row.update(
            {
                "status": "ok",
                "wall_time_seconds": elapsed,
                **{
                    f"{name}_rel_l2": value
                    for name, value in macro_error_metrics(prediction, truth).items()
                },
            }
        )
    except Exception as exc:  # noqa: BLE001 - feedback retains failures.
        row.update(
            {
                "status": "exception",
                "wall_time_seconds": float(time.perf_counter() - started),
                "exception": repr(exc),
            }
        )
    return row


def _checked_request(
    request: dict[str, Any], *, request_sender: RequestSender
) -> dict[str, Any]:
    response = request_sender(request)
    if not response.get("ok"):
        raise RuntimeError(str(response.get("error") or "online request failed"))
    return response


if __name__ == "__main__":
    main()
