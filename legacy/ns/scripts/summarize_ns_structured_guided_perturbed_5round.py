#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


GENERATOR_SCHEMA = "ns2d_structured_guided_perturbed_round_v1"
FILTER_SCHEMA = "ns2d_structured_guidance_perturbation_filter_v1"
REPORT_SCHEMA = "ns2d_structured_guided_perturbed_adversarial_summary_v1"
FAMILIES = ("pca_tangent", "spectral_complement", "pca_normal")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a structured NS perturbation campaign."
    )
    parser.add_argument("--loop-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-rounds", type=int, default=5)
    parser.add_argument("--expected-cases-per-round", type=int, default=1024)
    parser.add_argument("--expected-original-per-round", type=int, default=0)
    parser.add_argument("--expected-unperturbed-per-round", type=int, default=832)
    parser.add_argument("--expected-perturbations-per-round", type=int, default=192)
    parser.add_argument("--expected-replay-perturbations", type=int, default=96)
    parser.add_argument("--expected-pca-tangent", type=int, default=48)
    parser.add_argument("--expected-spectral-complement", type=int, default=96)
    parser.add_argument("--expected-pca-normal", type=int, default=48)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(args)
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "structured_guided_perturbed_5round_report.json"
    md_path = output / "structured_guided_perturbed_5round_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path.write_text(_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["engineering_complete"]:
        raise SystemExit("structured perturbation campaign checks failed")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    loop_root = Path(args.loop_root).resolve()
    output = Path(args.out).resolve()
    base_report_path = output / "structured_guided_5round_report.json"
    if not base_report_path.is_file():
        raise FileNotFoundError(
            f"run the base structured campaign summarizer first: {base_report_path}"
        )
    base_report = _read_json(base_report_path)
    loop_summary = _read_json(loop_root / "final_summary.json")
    expected_rounds = int(args.expected_rounds)
    expected_cases = int(args.expected_cases_per_round)
    expected_original = int(args.expected_original_per_round)
    expected_unperturbed = int(args.expected_unperturbed_per_round)
    expected_perturbations = int(args.expected_perturbations_per_round)
    expected_replay = int(args.expected_replay_perturbations)
    expected_families = {
        "pca_tangent": int(args.expected_pca_tangent),
        "spectral_complement": int(args.expected_spectral_complement),
        "pca_normal": int(args.expected_pca_normal),
    }
    if (
        expected_original + expected_unperturbed + expected_perturbations
        != expected_cases
    ):
        raise ValueError("expected mixed quotas do not equal the round budget")
    if min(expected_original, expected_unperturbed, expected_perturbations) < 0:
        raise ValueError("expected mixed quotas cannot be negative")
    if sum(expected_families.values()) != expected_perturbations:
        raise ValueError("expected family quotas do not equal perturbation quota")
    expected_continuous = expected_unperturbed + expected_perturbations

    rounds = []
    for round_index in range(expected_rounds):
        round_dir = loop_root / "rounds" / f"round_{round_index:03d}"
        receipt = _read_json(round_dir / "structured_generator_round.json")
        filter_report = _read_json(
            round_dir / "guidance_filter" / "reward_filter_report.json"
        )
        guidance_train = _read_json(round_dir / "guidance_update" / "train_report.json")
        evaluation = _read_json(round_dir / "generated_solver_metrics.json")
        manifest = _read_json(
            round_dir / "private_eval" / "generated_private_manifest.json"
        )
        group_summaries, manifest_matches_receipt = _group_eval(
            evaluation=evaluation,
            manifest=manifest,
            receipt=receipt,
        )
        composition = receipt.get("composition") or {}
        expected_source_counts = (
            {"current": expected_perturbations}
            if round_index == 0
            else {
                "current": expected_perturbations - expected_replay,
                "replay": expected_replay,
            }
        )
        expected_cumulative_input = expected_cases * (round_index + 1)
        expected_cumulative_excluded = (
            expected_original + expected_perturbations
        ) * (round_index + 1)
        expected_cumulative_retained = expected_unperturbed * (round_index + 1)
        checks = {
            "generator_schema_valid": receipt.get("generator_schema")
            == GENERATOR_SCHEMA,
            "continuous_contract_passed": receipt.get("contract_passed") is True,
            "linked_case_count_matches": int(receipt.get("linked_cases", -1))
            == expected_continuous,
            "fixed_total_budget": composition.get("fixed_total_budget") is True,
            "unperturbed_count_matches": int(
                composition.get("unperturbed_case_count", -1)
            )
            == expected_unperturbed,
            "perturbation_count_matches": int(
                composition.get("perturbation_case_count", -1)
            )
            == expected_perturbations,
            "family_counts_match": _normalized_counts(
                composition.get("perturbation_family_counts")
            )
            == expected_families,
            "source_counts_match": _normalized_counts(
                composition.get("perturbation_source_counts")
            )
            == _normalized_counts(expected_source_counts),
            "original_count_matches": int(
                group_summaries.get("original", {}).get("n", -1)
            )
            == expected_original,
            "manifest_matches_receipt": manifest_matches_receipt,
            "evaluation_complete": int((evaluation.get("summary") or {}).get("n", -1))
            == expected_cases
            and int((evaluation.get("summary") or {}).get("ok", -1))
            == expected_cases,
            "filter_schema_valid": filter_report.get("schema") == FILTER_SCHEMA,
            "perturbations_excluded_from_guidance": filter_report.get(
                "perturbations_excluded_from_generator_guidance"
            )
            is True,
            "cumulative_filter_input_count_matches": int(
                filter_report.get("input_rows", -1)
            )
            == expected_cumulative_input,
            "cumulative_filter_excluded_count_matches": int(
                filter_report.get("excluded_rows", -1)
            )
            == expected_cumulative_excluded,
            "cumulative_original_excluded_count_matches": int(
                filter_report.get("original_dataset_rows_excluded", -1)
            )
            == expected_original * (round_index + 1),
            "cumulative_filter_retained_count_matches": int(
                filter_report.get("retained_rows", -1)
            )
            == expected_cumulative_retained,
            "guidance_training_count_matches": int(
                guidance_train.get("case_count", -1)
            )
            == expected_cumulative_retained,
        }
        rounds.append(
            {
                "round": round_index,
                "sampling_policy": receipt.get("sampling_policy"),
                "composition": composition,
                "guidance_filter": filter_report,
                "current_solver_on_mixed_round": evaluation.get("summary"),
                "current_solver_by_case_group": group_summaries,
                "checks": checks,
                "engineering_complete": all(checks.values()),
            }
        )

    global_checks = {
        "loop_completed_expected_rounds": int(loop_summary.get("rounds_completed", -1))
        == expected_rounds,
        "base_guided_report_complete": base_report.get("engineering_complete") is True,
        "all_mixed_round_checks_pass": all(
            row["engineering_complete"] for row in rounds
        ),
        "all_reported_values_finite": _all_finite(rounds),
    }
    return {
        "schema": REPORT_SCHEMA,
        "loop_root": str(loop_root),
        "policy": {
            "round_budget": expected_cases,
            "original_per_round": expected_original,
            "unperturbed_per_round": expected_unperturbed,
            "perturbations_per_round": expected_perturbations,
            "perturbation_fraction": expected_perturbations / expected_cases,
            "family_counts_per_round": expected_families,
            "replay_perturbations_after_round_zero": expected_replay,
            "perturbations_train_solver": True,
            "perturbations_train_generator_guidance": False,
            "base_generator_and_pca_frozen": True,
        },
        "interpretation_note": (
            f"Generator audits cover the {expected_unperturbed}+"
            f"{expected_perturbations} continuous stream; solver metrics cover "
            f"the final {expected_original}+{expected_unperturbed}+"
            f"{expected_perturbations} mixed stream."
        ),
        "engineering_checks": global_checks,
        "engineering_complete": all(global_checks.values()),
        "rounds": rounds,
    }


def _group_eval(
    *,
    evaluation: dict[str, Any],
    manifest: dict[str, Any],
    receipt: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    manifest_rows = manifest.get("cases", [])
    alias_to_case = {
        str(row["case_id"]): Path(str(row["private_case_dir"])).name
        for row in manifest_rows
    }
    role_by_case = {str(row["case_id"]): row for row in receipt.get("cases", [])}
    manifest_case_ids = set(alias_to_case.values())
    receipt_case_ids = set(role_by_case)
    grouped: dict[str, list[dict[str, Any]]] = {
        "original": [],
        "unperturbed": [],
    }
    for family in FAMILIES:
        grouped[family] = []
    unmapped = False
    evaluated_case_ids: list[str] = []
    for result in evaluation.get("results", []):
        case_id = alias_to_case.get(str(result.get("case_id")))
        if case_id is None:
            unmapped = True
            continue
        evaluated_case_ids.append(case_id)
        role = role_by_case.get(case_id)
        if role is None:
            group = "original"
        else:
            family = role.get("perturbation_family")
            group = str(family) if family else "unperturbed"
        if group not in grouped:
            grouped[group] = []
        grouped[group].append(result)
    summaries = {name: _error_summary(rows) for name, rows in grouped.items()}
    return summaries, (
        not unmapped
        and receipt_case_ids.issubset(manifest_case_ids)
        and len(alias_to_case) == len(manifest_rows)
        and len(manifest_case_ids) == len(manifest_rows)
        and len(evaluated_case_ids) == len(manifest_rows)
        and set(evaluated_case_ids) == manifest_case_ids
    )


def _error_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray(
        [
            float(row["relative_l2"])
            for row in rows
            if row.get("status") == "ok"
            and isinstance(row.get("relative_l2"), (int, float))
        ],
        dtype=np.float64,
    )
    return {
        "n": len(rows),
        "ok": int(values.size),
        "mean_relative_l2": float(np.mean(values)) if values.size else None,
        "median_relative_l2": float(np.median(values)) if values.size else None,
        "q10_relative_l2": float(np.quantile(values, 0.10)) if values.size else None,
        "q90_relative_l2": float(np.quantile(values, 0.90)) if values.size else None,
    }


def _normalized_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): int(count) for key, count in value.items() if int(count) != 0}


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _markdown(report: dict[str, Any]) -> str:
    policy = report["policy"]
    family_counts = policy["family_counts_per_round"]
    lines = [
        "# Structured guided + perturbation adversarial report",
        "",
        f"- Engineering complete: `{report['engineering_complete']}`",
        f"- Loop root: `{report['loop_root']}`",
        "- Fixed round budget: "
        f"`{policy['original_per_round']} original + "
        f"{policy['unperturbed_per_round']} unperturbed + "
        f"{policy['perturbations_per_round']} perturbations = "
        f"{policy['round_budget']}`",
        "- Perturbations: "
        f"`{family_counts['pca_tangent']} PCA tangent + "
        f"{family_counts['spectral_complement']} spectral complement + "
        f"{family_counts['pca_normal']} PCA normal`",
        "- Perturbations train the solver but are excluded from generator guidance.",
        "",
        "| round | original mean | unperturbed mean | tangent mean | spectral mean | normal mean | checks |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report["rounds"]:
        groups = row["current_solver_by_case_group"]
        lines.append(
            "| {round} | {o} | {u} | {t} | {s} | {n} | {ok} |".format(
                round=row["round"],
                o=_fmt(groups["original"]["mean_relative_l2"]),
                u=_fmt(groups["unperturbed"]["mean_relative_l2"]),
                t=_fmt(groups["pca_tangent"]["mean_relative_l2"]),
                s=_fmt(groups["spectral_complement"]["mean_relative_l2"]),
                n=_fmt(groups["pca_normal"]["mean_relative_l2"]),
                ok="PASS" if row["engineering_complete"] else "FAIL",
            )
        )
    lines.extend(["", report["interpretation_note"], ""])
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6g}"


if __name__ == "__main__":
    main(sys.argv[1:])
