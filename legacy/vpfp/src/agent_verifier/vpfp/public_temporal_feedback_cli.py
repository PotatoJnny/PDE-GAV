#!/usr/bin/env python
"""Generate fresh VPFP temporal ladders and evaluate one solver on them."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from online_generator import (
        ONLINE_GENERATOR_REQUEST_SCHEMA,
        send_request,
    )
    from online_resolution_eval import (
        FIELDS,
        evaluate_resolution_views,
    )
except ModuleNotFoundError:  # Package imports used by repository tests.
    from agent_verifier.vpfp.public_online_generator_cli import (
        ONLINE_GENERATOR_REQUEST_SCHEMA,
        send_request,
    )
    from agent_verifier.vpfp.public_resolution_eval_cli import (
        FIELDS,
        evaluate_resolution_views,
    )


TEMPORAL_FEEDBACK_SCHEMA = "vpfp_fresh_temporal_feedback_v1"
STUDY_ID = "fresh_temporal_feedback"
RequestSender = Callable[[dict[str, Any]], dict[str, Any]]
Evaluator = Callable[..., dict[str, Any]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", default="submission.py")
    parser.add_argument("--roots", type=int, default=4)
    parser.add_argument(
        "--nt-levels",
        type=int,
        nargs="+",
        default=None,
        help=(
            "At least three increasing output time-grid sizes. The default is "
            "the sampled base nt followed by two endpoint-nested refinements."
        ),
    )
    parser.add_argument(
        "--condition-pool",
        choices=("train", "validation"),
        default="validation",
    )
    parser.add_argument("--root", default="online_generator_outputs")
    parser.add_argument("--out", default="temporal_feedback.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_temporal_feedback(
        submission=args.submission,
        root=args.root,
        out=args.out,
        root_count=int(args.roots),
        nt_levels=args.nt_levels,
        condition_pool=str(args.condition_pool),
    )
    summary = report["summary"]
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "passed": summary["passed"],
                "root_count": summary["root_count"],
                "passed_root_count": summary["passed_root_count"],
                "root_monotone_fraction": summary["root_monotone_fraction"],
                "nt_levels": summary["nt_levels"],
                "mean_aggregate_errors": summary["mean_aggregate_errors"],
                "mean_field_errors": summary["mean_field_errors"],
                "worst_root_finest_over_coarsest_ratio": summary[
                    "worst_root_finest_over_coarsest_ratio"
                ],
                "report": str(Path(args.out)),
            },
            indent=2,
            sort_keys=True,
        )
    )


def run_temporal_feedback(
    *,
    submission: str | Path,
    root: str | Path,
    out: str | Path,
    root_count: int,
    nt_levels: list[int] | None = None,
    condition_pool: str = "validation",
    request_sender: RequestSender = send_request,
    evaluator: Evaluator = evaluate_resolution_views,
) -> dict[str, Any]:
    status = _checked_request(
        {
            "schema": ONLINE_GENERATOR_REQUEST_SCHEMA,
            "operation": "status",
        },
        request_sender=request_sender,
    )
    max_batch = int(status["budgets"]["max_batch_per_call"])
    if root_count < 1 or root_count > max_batch:
        raise ValueError(
            f"roots must be between 1 and the service batch limit {max_batch}"
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
        raise RuntimeError(
            f"online service returned {len(roots)} roots, expected {root_count}"
        )
    base_nts = {int(dict(row["resolution"])["nt"]) for row in roots}
    if len(base_nts) != 1:
        raise RuntimeError("fresh roots do not share one base time resolution")
    base_nt = next(iter(base_nts))
    levels = _validate_nt_levels(
        nt_levels
        if nt_levels is not None
        else [base_nt, 2 * (base_nt - 1) + 1, 4 * (base_nt - 1) + 1]
    )

    materialized_case_ids = []
    last_response = sampled
    for root_row in roots:
        root_case_id = str(root_row["case_id"])
        base_resolution = {
            name: int(value)
            for name, value in dict(root_row["resolution"]).items()
        }
        for level, nt in enumerate(levels):
            resolution = {**base_resolution, "nt": int(nt)}
            last_response = _checked_request(
                {
                    "schema": ONLINE_GENERATOR_REQUEST_SCHEMA,
                    "operation": "materialize",
                    "case_id": root_case_id,
                    "answers": "grid",
                    "resolution": resolution,
                    "study_id": STUDY_ID,
                    "level": int(level),
                },
                request_sender=request_sender,
            )
            materialized = dict(last_response["case"])
            if str(materialized.get("root_case_id")) != root_case_id:
                raise RuntimeError("materialized view changed the root equation")
            materialized_case_ids.append(str(materialized["case_id"]))

    evaluation = evaluator(
        submission=submission,
        root=root,
        case_ids=set(materialized_case_ids),
    )
    root_case_ids = [str(row["case_id"]) for row in roots]
    summary = summarize_temporal_feedback(
        evaluation,
        expected_root_case_ids=root_case_ids,
        nt_levels=levels,
    )
    report = {
        "schema": TEMPORAL_FEEDBACK_SCHEMA,
        "selection_or_final_data": False,
        "condition_pool": str(condition_pool),
        "fresh_root_case_ids": root_case_ids,
        "materialized_case_ids": materialized_case_ids,
        "summary": summary,
        "evaluation": evaluation,
        "budget_usage": dict(last_response.get("budgets") or {}),
    }
    output = Path(out)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def summarize_temporal_feedback(
    evaluation: dict[str, Any],
    *,
    expected_root_case_ids: list[str],
    nt_levels: list[int],
) -> dict[str, Any]:
    levels = _validate_nt_levels(nt_levels)
    rows_by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluation.get("cases", []):
        root_case_id = str(row.get("root_case_id") or "")
        if root_case_id in expected_root_case_ids:
            rows_by_root[root_case_id].append(dict(row))

    root_results = []
    for root_case_id in expected_root_case_ids:
        rows = sorted(
            rows_by_root.get(root_case_id, []),
            key=lambda row: int(dict(row.get("resolution") or {}).get("nt", -1)),
        )
        observed_nt = [
            int(dict(row.get("resolution") or {}).get("nt", -1))
            for row in rows
        ]
        complete = observed_nt == levels
        valid = complete and all(row.get("status") == "ok" for row in rows)
        field_errors = {
            field: [
                float(row[f"{field}_rel_l2"])
                for row in rows
                if row.get("status") == "ok"
            ]
            for field in FIELDS
        }
        aggregate_errors = [
            float(row["score_smaller_is_better"])
            for row in rows
            if row.get("status") == "ok"
        ]
        aggregate_monotone = (
            _nonincreasing(aggregate_errors) if valid else False
        )
        fields_finest_not_worse = (
            all(
                values[-1] <= values[0] * (1.0 + 1.0e-8)
                for values in field_errors.values()
            )
            if valid
            else False
        )
        passed = bool(
            valid and aggregate_monotone and fields_finest_not_worse
        )
        root_results.append(
            {
                "root_case_id": root_case_id,
                "complete": bool(complete),
                "all_outputs_valid": bool(valid),
                "observed_nt_levels": observed_nt,
                "aggregate_errors": aggregate_errors,
                "field_errors": field_errors,
                "aggregate_error_monotone_nonincreasing": bool(
                    aggregate_monotone
                ),
                "all_fields_finest_not_worse_than_coarsest": bool(
                    fields_finest_not_worse
                ),
                "passed": passed,
            }
        )

    level_means = []
    for nt in levels:
        matching = [
            row
            for rows in rows_by_root.values()
            for row in rows
            if row.get("status") == "ok"
            and int(dict(row.get("resolution") or {}).get("nt", -1)) == nt
        ]
        complete = len(matching) == len(expected_root_case_ids)
        level_means.append(
            {
                "nt": int(nt),
                "root_count": len(matching),
                "aggregate_error": (
                    float(
                        np.mean(
                            [
                                float(row["score_smaller_is_better"])
                                for row in matching
                            ]
                        )
                    )
                    if complete
                    else None
                ),
                "field_errors": {
                    field: (
                        float(
                            np.mean(
                                [
                                    float(row[f"{field}_rel_l2"])
                                    for row in matching
                                ]
                            )
                        )
                        if complete
                        else None
                    )
                    for field in FIELDS
                },
            }
        )

    mean_aggregate_errors = [row["aggregate_error"] for row in level_means]
    mean_field_errors = {
        field: [row["field_errors"][field] for row in level_means]
        for field in FIELDS
    }
    complete_means = all(value is not None for value in mean_aggregate_errors)
    mean_aggregate_monotone = bool(
        complete_means
        and _nonincreasing(
            [float(value) for value in mean_aggregate_errors if value is not None]
        )
    )
    mean_fields_finest_not_worse = bool(
        complete_means
        and all(
            values[-1] is not None
            and values[0] is not None
            and float(values[-1]) <= float(values[0]) * (1.0 + 1.0e-8)
            for values in mean_field_errors.values()
        )
    )
    passed_root_count = sum(bool(row["passed"]) for row in root_results)
    ratios = [
        float(row["aggregate_errors"][-1])
        / max(float(row["aggregate_errors"][0]), 1.0e-30)
        for row in root_results
        if row["all_outputs_valid"]
    ]
    passed = bool(
        evaluation.get("all_valid")
        and passed_root_count == len(expected_root_case_ids)
        and mean_aggregate_monotone
        and mean_fields_finest_not_worse
    )
    return {
        "passed": passed,
        "pass_semantics": (
            "every fresh root has all requested levels, finite outputs, "
            "nonincreasing aggregate exact-reference error, and no macro field "
            "whose finest error exceeds its coarsest error"
        ),
        "root_count": len(expected_root_case_ids),
        "passed_root_count": int(passed_root_count),
        "root_monotone_fraction": (
            float(passed_root_count / len(expected_root_case_ids))
            if expected_root_case_ids
            else 0.0
        ),
        "nt_levels": levels,
        "mean_aggregate_errors": mean_aggregate_errors,
        "mean_field_errors": mean_field_errors,
        "mean_aggregate_error_monotone_nonincreasing": (
            mean_aggregate_monotone
        ),
        "all_mean_fields_finest_not_worse_than_coarsest": (
            mean_fields_finest_not_worse
        ),
        "worst_root_finest_over_coarsest_ratio": (
            max(ratios) if ratios else None
        ),
        "levels": level_means,
        "roots": root_results,
    }


def _validate_nt_levels(values: list[int]) -> list[int]:
    levels = [int(value) for value in values]
    if len(levels) < 3:
        raise ValueError("a temporal feedback ladder needs at least three levels")
    if any(value < 2 for value in levels):
        raise ValueError("all temporal grid sizes must be at least two")
    if any(fine <= coarse for coarse, fine in zip(levels, levels[1:])):
        raise ValueError("temporal grid sizes must be strictly increasing")
    return levels


def _nonincreasing(values: list[float]) -> bool:
    return all(
        float(fine) <= float(coarse) * (1.0 + 1.0e-8)
        for coarse, fine in zip(values, values[1:])
    )


def _checked_request(
    request: dict[str, Any], *, request_sender: RequestSender
) -> dict[str, Any]:
    response = request_sender(request)
    if not bool(response.get("ok")):
        raise RuntimeError(
            str(response.get("error") or "online generator request failed")
        )
    return response


if __name__ == "__main__":
    main()
