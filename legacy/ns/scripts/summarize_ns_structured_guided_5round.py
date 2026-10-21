#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


ROUND_PATTERN = re.compile(
    r"(?:structured_guided|caltech_guided|ns_public_(?:base|guided))_r(\d{3})_"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a structured solver-guided NS campaign."
    )
    parser.add_argument("--loop-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-rounds", type=int, default=5)
    parser.add_argument("--expected-cases-per-round", type=int, default=1024)
    parser.add_argument("--expected-original-per-round", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    loop_root = Path(args.loop_root).resolve()
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    final_path = loop_root / "final_summary.json"
    if not final_path.is_file():
        raise SystemExit(f"missing adversarial-loop final summary: {final_path}")
    loop_summary = json.loads(final_path.read_text())
    expected_rounds = int(args.expected_rounds)
    expected_cases = int(args.expected_cases_per_round)
    expected_original = int(args.expected_original_per_round)
    expected_continuous = expected_cases - expected_original
    if expected_original < 0 or expected_continuous <= 0:
        raise ValueError("expected original/public case counts are invalid")
    rounds = []
    provenance_checks = []
    replay_matrix: dict[str, dict[str, Any]] = {}

    for round_index in range(expected_rounds):
        round_dir = loop_root / "rounds" / f"round_{round_index:03d}"
        summary = _read_json(round_dir / "round_summary.json")
        sample, materialization, component_audits, continuous_case_count = (
            _round_generator_audits(round_dir)
        )
        guidance_train = _read_json(
            round_dir / "guidance_update" / "train_report.json"
        )
        submission = round_dir / "submission.py"
        previous_submission = (
            loop_root / "rounds" / f"round_{round_index - 1:03d}" / "submission.py"
            if round_index > 0
            else None
        )
        guidance = sample.get("guidance")
        provenance_ok = guidance is None if round_index == 0 else bool(
            guidance
            and previous_submission is not None
            and str(guidance.get("target_solver_sha256"))
            == _sha256(previous_submission)
        )
        provenance_checks.append(provenance_ok)
        generated_distribution = sample["distribution"]
        nearest = generated_distribution["nearest_train_distance"]["generated"]
        physical = _generated_physical_statistics(materialization)
        reward_validation = guidance_train["validation"]
        rounds.append(
            {
                "round": round_index,
                "sampling_policy": sample["sampling_policy"],
                "generated_cases": int(summary["generated_case_count"]),
                "original_cases": int(
                    (summary.get("case_mix") or {}).get("original", 0)
                ),
                "continuous_case_count": int(continuous_case_count),
                "continuous_contract_passed": int(continuous_case_count),
                "generator_component_audits": component_audits,
                "guidance_targets_previous_solver": provenance_ok,
                "guidance": guidance,
                "generator_distribution": {
                    "prdc": generated_distribution[
                        "generated_vs_validation_prdc"
                    ],
                    "rbf_mmd": generated_distribution[
                        "generated_vs_validation_rbf_mmd"
                    ],
                    "sliced_wasserstein": generated_distribution[
                        "generated_vs_validation_sliced_wasserstein"
                    ],
                    "nearest_train": nearest,
                    "score_statistics": generated_distribution[
                        "score_statistics"
                    ]["generated"],
                    "effective_rank": generated_distribution[
                        "effective_rank"
                    ]["generated"],
                },
                "continuous_physical_statistics": physical,
                "materialization_audit": {
                    "schema": materialization.get("schema"),
                    "family_counts": materialization.get("family_counts"),
                    "pressure_claim": materialization.get("pressure_claim"),
                    "pressure_coefficients_nonzero": materialization.get(
                        "pressure_coefficients_nonzero"
                    ),
                    "probe_residual_abs_max": materialization.get(
                        "probe_residual_abs_max"
                    ),
                },
                "previous_solver_on_current_generated": summary.get(
                    "previous_solver_generated_summary"
                ),
                "current_solver_on_current_generated": summary.get(
                    "generated_summary"
                ),
                "solver_improvement_on_current_generated": summary.get(
                    "solver_improvement_on_generated"
                ),
                "solver_improvement_on_cumulative_replay": summary.get(
                    "solver_improvement_on_cumulative_generated"
                ),
                "generator_difficulty_change_under_previous_solver": summary.get(
                    "generator_difficulty_change_under_previous_solver"
                ),
                "hidden_holdout": summary.get("hidden_holdout_summary"),
                "reward_model": {
                    "case_count": guidance_train["case_count"],
                    "train_count": guidance_train["train_count"],
                    "validation_count": guidance_train["validation_count"],
                    "clean_state_pearson": reward_validation[
                        "clean_state_pearson"
                    ],
                    "clean_state_spearman": reward_validation[
                        "clean_state_spearman"
                    ],
                    "clean_state_top_quartile_reward_lift": reward_validation[
                        "clean_state_top_quartile_reward_lift"
                    ],
                },
                "submission": {
                    "path": str(submission),
                    "sha256": _sha256(submission),
                    "line_count": len(submission.read_text().splitlines()),
                },
            }
        )
        replay_matrix[f"solver_r{round_index:03d}"] = _replay_row(round_dir)

    baseline_physical = rounds[0]["continuous_physical_statistics"]
    for row in rounds:
        row["physical_median_ratio_to_round_000"] = _physical_median_ratios(
            row["continuous_physical_statistics"], baseline_physical
        )

    checks = {
        "all_rounds_completed": (
            int(loop_summary.get("rounds_completed", -1)) == expected_rounds
        ),
        "all_round_case_counts_match": all(
            row["generated_cases"] == expected_cases for row in rounds
        ),
        "all_original_case_counts_match": all(
            row["original_cases"] == expected_original for row in rounds
        ),
        "all_continuous_contracts_pass": all(
            row["continuous_contract_passed"] == expected_continuous
            for row in rounds
        ),
        "guidance_provenance_chain_valid": all(provenance_checks),
        "all_policy_updates_present": all(
            (
                loop_root
                / "rounds"
                / f"round_{index:03d}"
                / "guidance_update"
                / "reward_guidance.pt"
            ).is_file()
            for index in range(expected_rounds)
        ),
        "all_hidden_holdout_evaluations_present": all(
            row["hidden_holdout"] is not None for row in rounds
        ),
        "no_exact_train_copies_reported": all(
            row["generator_distribution"]["nearest_train"][
                "exact_or_near_copy_fraction"
            ]
            == 0.0
            for row in rounds
        ),
        "all_reported_values_finite": _all_finite(rounds),
    }
    report = {
        "schema": "ns2d_structured_guided_adversarial_summary_v1",
        "loop_root": str(loop_root),
        "expected_rounds": expected_rounds,
        "expected_cases_per_round": expected_cases,
        "engineering_checks": checks,
        "engineering_complete": all(checks.values()),
        "scientific_interpretation_requires_solver_source_review": True,
        "rounds": rounds,
        "solver_replay_matrix": replay_matrix,
        "generator_difficulty_change_under_previous_solver": [
            row["generator_difficulty_change_under_previous_solver"]
            for row in rounds
        ],
        "solver_improvement_on_current_generated": [
            row["solver_improvement_on_current_generated"] for row in rounds
        ],
        "hidden_holdout_mean_relative_l2": [
            (row["hidden_holdout"] or {}).get("mean_relative_l2")
            for row in rounds
        ],
    }
    report_path = output / "structured_guided_5round_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "structured_guided_5round_report.md").write_text(
        _markdown(report)
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["engineering_complete"]:
        raise SystemExit("structured guided campaign checks failed")


def _round_generator_audits(
    round_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], int]:
    """Read either the historical single stream or the explicit base/guided mix."""

    legacy_sample = round_dir / "structured_generator_sample" / "sample_audit.json"
    legacy_materialization = (
        round_dir
        / "structured_generator_materialization"
        / "answer_first_report.json"
    )
    if legacy_sample.is_file() and legacy_materialization.is_file():
        sample = _read_json(legacy_sample)
        materialization = _read_json(legacy_materialization)
        passed = int(materialization["continuous_contract_passed"])
        return (
            sample,
            materialization,
            [
                {
                    "kind": "single_stream",
                    "cases": int(materialization["generated_cases"]),
                }
            ],
            passed,
        )

    receipt = _read_json(round_dir / "structured_generator_round.json")
    base_receipt = _read_json(round_dir / "base_generator_round.json")
    component_rows = base_receipt.get("components")
    if not isinstance(component_rows, list) or not component_rows:
        raise ValueError(f"round has no Generator component reports: {round_dir}")
    audits: list[dict[str, Any]] = []
    loaded: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for component in component_rows:
        if not isinstance(component, dict):
            raise ValueError(f"invalid Generator component report: {round_dir}")
        sample_path = _report_path(component.get("sample_audit"), round_dir)
        materialization_path = _report_path(
            component.get("materialization_audit"), round_dir
        )
        sample = _read_json(sample_path)
        materialization = _read_json(materialization_path)
        kind = str(component.get("kind") or "generated")
        audits.append(
            {
                "kind": kind,
                "cases": int(component.get("cases", 0)),
                "sampling_policy": sample.get("sampling_policy"),
                "continuous_contract_passed": int(
                    materialization.get("continuous_contract_passed", 0)
                ),
            }
        )
        loaded.append((component, sample, materialization))

    guided = next(
        (row for row in reversed(loaded) if str(row[0].get("kind")) == "guided"),
        loaded[-1],
    )
    if receipt.get("contract_passed") is not True:
        raise ValueError(f"round continuous contract did not pass: {round_dir}")
    continuous_case_count = int(receipt.get("linked_cases", -1))
    if continuous_case_count <= 0:
        raise ValueError(f"round has no continuous generated cases: {round_dir}")
    return guided[1], guided[2], audits, continuous_case_count


def _report_path(value: Any, round_dir: Path) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else (round_dir / path).resolve()


def _replay_row(round_dir: Path) -> dict[str, Any]:
    evaluation = _read_json(round_dir / "cumulative_generated_solver_metrics.json")
    manifest_path = (
        round_dir / "private_eval" / "solver_feedback_private_manifest.json"
    )
    if not manifest_path.is_file():
        manifest_path = round_dir / "private_eval" / "generated_private_manifest.json"
    manifest = _read_json(manifest_path)
    target_round_by_id = {}
    for row in manifest["cases"]:
        match = ROUND_PATTERN.search(Path(str(row["private_case_dir"])).name)
        target_round_by_id[str(row["case_id"])] = (
            int(match.group(1)) if match is not None else None
        )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in evaluation["results"]:
        target_round = target_round_by_id[str(row["case_id"])]
        label = (
            f"generated_r{target_round:03d}"
            if target_round is not None
            else "original_d0"
        )
        grouped.setdefault(label, []).append(row)
    return {
        label: _error_summary(rows) for label, rows in sorted(grouped.items())
    }


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
        "q10_relative_l2": float(np.quantile(values, 0.1)) if values.size else None,
        "q90_relative_l2": float(np.quantile(values, 0.9)) if values.size else None,
    }


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if not isinstance(numerator, (int, float)) or not isinstance(
        denominator, (int, float)
    ):
        return None
    if abs(float(denominator)) <= 1.0e-12:
        return None
    return float(numerator) / float(denominator)


def _generated_physical_statistics(
    materialization: dict[str, Any],
) -> dict[str, Any] | None:
    """Return legacy physical statistics when the materializer reports them.

    Caltech measurement-anchored materialization has its own audit schema: it
    reports family coverage, the pressure-equivalence claim, and exact MMS
    residuals, but does not duplicate the base generator observation-space
    physical-distribution audit in every adversarial round.
    """
    physical_distribution = materialization.get("physical_distribution")
    if not isinstance(physical_distribution, dict):
        return None
    groups = physical_distribution.get("groups")
    if not isinstance(groups, dict):
        return None
    generated = groups.get("generated")
    return generated if isinstance(generated, dict) else None


def _physical_median_ratios(
    physical: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
) -> dict[str, float | None] | None:
    if not isinstance(physical, dict) or not isinstance(baseline, dict):
        return None
    return {
        metric: _safe_ratio(
            values.get("median") if isinstance(values, dict) else None,
            baseline.get(metric, {}).get("median")
            if isinstance(baseline.get(metric), dict)
            else None,
        )
        for metric, values in physical.items()
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Structured solver-guided adversarial report",
        "",
        f"- engineering complete: `{report['engineering_complete']}`",
        f"- rounds: `{report['expected_rounds']}`",
        f"- cases per round: `{report['expected_cases_per_round']}`",
        "",
        "| round | previous solver error | current solver error | solver improvement | generator change | hidden holdout | reward Pearson |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rounds"]:
        previous = row["previous_solver_on_current_generated"] or {}
        current = row["current_solver_on_current_generated"] or {}
        holdout = row["hidden_holdout"] or {}
        lines.append(
            "| {round} | {previous} | {current} | {improvement} | {change} | {holdout} | {pearson} |".format(
                round=row["round"],
                previous=_fmt(previous.get("mean_relative_l2")),
                current=_fmt(current.get("mean_relative_l2")),
                improvement=_fmt(row["solver_improvement_on_current_generated"]),
                change=_fmt(row["generator_difficulty_change_under_previous_solver"]),
                holdout=_fmt(holdout.get("mean_relative_l2")),
                pearson=_fmt(row["reward_model"]["clean_state_pearson"]),
            )
        )
    lines.extend(
        [
            "",
            "Solver source still requires a direct method review before scientific completion.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "NA" if not isinstance(value, (int, float)) else f"{float(value):.6g}"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main(sys.argv[1:])
