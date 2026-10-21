"""Choose the NS representation and Base Generator from validation results.

This module is intentionally small and readable.  The long-running scripts
produce scientific measurements; these functions make the one decision needed
by the next stage and write it to a stable JSON file.

The final hidden cases are never used here.  PCA uses the validation
reconstruction report, and Flow checkpoints use a separate validation sample
cohort.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_FLOW_WEIGHTS = {
    "validation_loss": 2.0,
    "latent_mmd": 1.0,
    "latent_sliced_wasserstein": 1.0,
    "latent_coverage": 1.0,
    "physical_mmd": 1.0,
    "physical_sliced_wasserstein": 1.0,
    "physical_coverage": 1.0,
}

_LOWER_IS_BETTER = {
    "validation_loss",
    "latent_mmd",
    "latent_sliced_wasserstein",
    "physical_mmd",
    "physical_sliced_wasserstein",
}


def select_pca_rank(
    report: Mapping[str, Any],
    *,
    metric: str = "manifold_velocity_relative_l2",
    statistic: str = "median",
    relative_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Select the smallest rank whose physical error is close to the best.

    A rank is eligible when its selected validation error is no more than
    ``relative_tolerance`` above the best candidate.  Choosing the smallest
    eligible rank keeps the latent model compact without accepting a material
    loss in physical reconstruction quality.
    """

    if relative_tolerance < 0.0:
        raise ValueError("PCA relative_tolerance cannot be negative")
    candidates: list[dict[str, Any]] = []
    for raw in _rows(report, "ranks"):
        rank = int(raw["rank"])
        physical = _mapping(raw.get("physical_reconstruction"))
        metric_values = _mapping(physical.get(metric))
        value = _finite_number(metric_values.get(statistic), f"rank {rank} {metric}.{statistic}")
        candidates.append(
            {
                "rank": rank,
                "physical_error": value,
                "explained_variance_fraction": _optional_finite_number(
                    raw.get("explained_variance_fraction")
                ),
            }
        )
    if not candidates:
        raise ValueError("PCA audit contains no rank candidates")

    best_error = min(row["physical_error"] for row in candidates)
    threshold = best_error + relative_tolerance * max(abs(best_error), 1.0e-12)
    eligible = [row for row in candidates if row["physical_error"] <= threshold]
    selected = min(eligible, key=lambda row: row["rank"])
    selected_rank = int(selected["rank"])
    return {
        "schema": "ns_pca_selection_v1",
        "selected_rank": selected_rank,
        "rule": "smallest_rank_within_relative_tolerance_of_best_physical_error",
        "metric": metric,
        "statistic": statistic,
        "relative_tolerance": float(relative_tolerance),
        "best_physical_error": float(best_error),
        "eligibility_threshold": float(threshold),
        "candidates": [
            {**row, "eligible": int(row["rank"]) in {int(item["rank"]) for item in eligible}}
            for row in sorted(candidates, key=lambda item: int(item["rank"]))
        ],
    }


def select_flow_checkpoint(
    *,
    validation_losses: Mapping[int, float],
    candidate_reports: Mapping[int, tuple[Mapping[str, Any], Mapping[str, Any]]],
    minimum_contract_pass_fraction: float = 1.0,
    weights: Mapping[str, float] | None = None,
    selection_rule: str = "weighted_rank",
    empirical_baseline_trials: Iterable[Mapping[str, Any]] | None = None,
    baseline_lower_quantile: float = 0.05,
    baseline_upper_quantile: float = 0.95,
) -> dict[str, Any]:
    """Select a Flow checkpoint after physical gates and metric rank aggregation.

    Validation loss has extra weight by default.  Distribution measurements
    are compared by rank rather than by adding unlike numerical units.  Every
    metric contribution remains visible in the returned candidate table.
    """

    if not 0.0 <= minimum_contract_pass_fraction <= 1.0:
        raise ValueError("minimum contract pass fraction must lie in [0, 1]")
    if selection_rule not in {"weighted_rank", "empirical_d0_gate"}:
        raise ValueError(
            "Flow selection_rule must be weighted_rank or empirical_d0_gate"
        )
    resolved_weights = dict(DEFAULT_FLOW_WEIGHTS)
    if weights is not None:
        unknown = set(weights) - set(DEFAULT_FLOW_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown Flow selection weights: {sorted(unknown)}")
        resolved_weights.update({name: float(value) for name, value in weights.items()})
    if any(not math.isfinite(value) or value < 0.0 for value in resolved_weights.values()):
        raise ValueError("Flow selection weights must be finite and non-negative")
    if not any(value > 0.0 for value in resolved_weights.values()):
        raise ValueError("at least one Flow selection weight must be positive")

    candidates: list[dict[str, Any]] = []
    for step in sorted(candidate_reports):
        if step not in validation_losses:
            raise ValueError(f"missing validation loss for Flow checkpoint step {step}")
        sample, materialization = candidate_reports[step]
        latent = _mapping(sample.get("distribution"))
        latent_prdc = _mapping(latent.get("generated_vs_validation_prdc"))
        physical = _mapping(materialization.get("physical_distribution"))
        joint = _mapping(physical.get("joint_log_metric_distribution"))
        physical_prdc = _mapping(joint.get("generated_vs_reference_pca_prdc"))
        generated_cases = int(materialization.get("generated_cases", 0))
        passed_cases = int(materialization.get("continuous_contract_passed", -1))
        if generated_cases <= 0:
            raise ValueError(f"Flow checkpoint step {step} has no materialized cases")
        pass_fraction = passed_cases / generated_cases
        metrics = {
            "validation_loss": _finite_number(validation_losses[step], "validation_loss"),
            "latent_mmd": _finite_number(
                latent.get("generated_vs_validation_rbf_mmd"), "latent_mmd"
            ),
            "latent_sliced_wasserstein": _finite_number(
                latent.get("generated_vs_validation_sliced_wasserstein"),
                "latent_sliced_wasserstein",
            ),
            "latent_coverage": _finite_number(
                latent_prdc.get("coverage"), "latent_coverage"
            ),
            "physical_mmd": _finite_number(
                joint.get("generated_vs_reference_pca_rbf_mmd"), "physical_mmd"
            ),
            "physical_sliced_wasserstein": _finite_number(
                joint.get("generated_vs_reference_pca_sliced_wasserstein"),
                "physical_sliced_wasserstein",
            ),
            "physical_coverage": _finite_number(
                physical_prdc.get("coverage"), "physical_coverage"
            ),
        }
        candidates.append(
            {
                "step": int(step),
                "contract_pass_fraction": float(pass_fraction),
                "eligible": pass_fraction >= minimum_contract_pass_fraction,
                "metrics": metrics,
            }
        )

    if selection_rule == "empirical_d0_gate":
        if empirical_baseline_trials is None:
            raise ValueError(
                "empirical_d0_gate requires repeated D0-vs-D0 baseline trials"
            )
        return _select_empirical_flow_checkpoint(
            candidates,
            empirical_baseline_trials,
            minimum_contract_pass_fraction=minimum_contract_pass_fraction,
            lower_quantile=baseline_lower_quantile,
            upper_quantile=baseline_upper_quantile,
        )

    eligible = [row for row in candidates if bool(row["eligible"])]
    if not eligible:
        raise ValueError("no Flow checkpoint passed the physical contract gate")
    metric_ranks: dict[str, dict[int, int]] = {}
    for metric in DEFAULT_FLOW_WEIGHTS:
        reverse = metric not in _LOWER_IS_BETTER
        ordered = sorted(
            eligible,
            key=lambda row: float(_mapping(row["metrics"])[metric]),
            reverse=reverse,
        )
        metric_ranks[metric] = _competition_ranks(ordered, metric)

    for row in candidates:
        step = int(row["step"])
        if not row["eligible"]:
            row["metric_ranks"] = None
            row["rank_score"] = None
            continue
        ranks = {metric: metric_ranks[metric][step] for metric in DEFAULT_FLOW_WEIGHTS}
        row["metric_ranks"] = ranks
        row["rank_score"] = float(
            sum(resolved_weights[metric] * ranks[metric] for metric in DEFAULT_FLOW_WEIGHTS)
        )

    selected = min(
        eligible,
        key=lambda row: (
            float(row["rank_score"]),
            float(_mapping(row["metrics"])["validation_loss"]),
            -int(row["step"]),
        ),
    )
    return {
        "schema": "ns_flow_selection_v1",
        "selected_step": int(selected["step"]),
        "rule": "physical_contract_gate_then_weighted_metric_rank",
        "minimum_contract_pass_fraction": float(minimum_contract_pass_fraction),
        "weights": resolved_weights,
        "candidates": candidates,
    }



_EMPIRICAL_DISTRIBUTION_METRICS = (
    "latent_mmd",
    "latent_sliced_wasserstein",
    "latent_coverage",
    "physical_mmd",
    "physical_sliced_wasserstein",
    "physical_coverage",
)


def empirical_baseline_trials_from_reports(
    candidate_reports: Mapping[
        int, tuple[Mapping[str, Any], Mapping[str, Any]]
    ],
) -> tuple[dict[str, float], ...]:
    """Read repeated D0-vs-D0 trials emitted by the NS audit scripts.

    The first checkpoint is sufficient because every candidate audit uses the
    same held-out D0 cohort and seed. The function still requires both latent
    and physical trials so one gate cannot silently stand in for the other.
    """

    if not candidate_reports:
        raise ValueError("cannot read an empirical baseline without candidates")
    step = min(candidate_reports)
    sample, materialization = candidate_reports[step]
    latent_trials = _mapping(sample.get("distribution")).get(
        "empirical_d0_trials"
    )
    joint = _mapping(
        _mapping(materialization.get("physical_distribution")).get(
            "joint_log_metric_distribution"
        )
    )
    physical_trials = joint.get("empirical_d0_trials")
    if (
        not isinstance(latent_trials, list)
        or not isinstance(physical_trials, list)
        or not latent_trials
        or len(latent_trials) != len(physical_trials)
    ):
        raise ValueError(
            "empirical selection needs equal nonempty latent and physical "
            "D0-vs-D0 trial lists"
        )
    rows = []
    for latent, physical in zip(latent_trials, physical_trials, strict=True):
        if not isinstance(latent, Mapping) or not isinstance(physical, Mapping):
            raise ValueError("empirical D0 trial rows must be objects")
        rows.append(
            {
                "latent_mmd": _finite_number(latent.get("mmd"), "latent baseline MMD"),
                "latent_sliced_wasserstein": _finite_number(
                    latent.get("sliced_wasserstein"),
                    "latent baseline sliced Wasserstein",
                ),
                "latent_coverage": _finite_number(
                    latent.get("coverage"), "latent baseline coverage"
                ),
                "physical_mmd": _finite_number(
                    physical.get("mmd"), "physical baseline MMD"
                ),
                "physical_sliced_wasserstein": _finite_number(
                    physical.get("sliced_wasserstein"),
                    "physical baseline sliced Wasserstein",
                ),
                "physical_coverage": _finite_number(
                    physical.get("coverage"), "physical baseline coverage"
                ),
            }
        )
    return tuple(rows)


def _select_empirical_flow_checkpoint(
    candidates: list[dict[str, Any]],
    baseline_trials: Iterable[Mapping[str, Any]],
    *,
    minimum_contract_pass_fraction: float,
    lower_quantile: float,
    upper_quantile: float,
) -> dict[str, Any]:
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("empirical baseline quantiles must satisfy 0 <= lower < upper <= 1")
    rows = tuple(baseline_trials)
    if len(rows) < 2:
        raise ValueError("empirical selection needs at least two D0-vs-D0 trials")

    thresholds: dict[str, dict[str, float | str]] = {}
    for metric in _EMPIRICAL_DISTRIBUTION_METRICS:
        values = [
            _finite_number(row.get(metric), f"empirical baseline {metric}")
            for row in rows
        ]
        if metric in _LOWER_IS_BETTER:
            thresholds[metric] = {
                "direction": "maximum",
                "value": _quantile(values, upper_quantile),
            }
        else:
            thresholds[metric] = {
                "direction": "minimum",
                "value": _quantile(values, lower_quantile),
            }

    for candidate in candidates:
        physical_passed = bool(candidate["eligible"])
        candidate["physical_contract_eligible"] = physical_passed
        metrics = _mapping(candidate["metrics"])
        gates: dict[str, bool] = {}
        for metric, threshold in thresholds.items():
            value = float(metrics[metric])
            boundary = float(threshold["value"])
            gates[metric] = (
                value <= boundary
                if threshold["direction"] == "maximum"
                else value >= boundary
            )
        candidate["empirical_gates"] = gates
        candidate["eligible"] = physical_passed and all(gates.values())
        candidate["rank_score"] = None
        candidate["metric_ranks"] = None

    eligible = [row for row in candidates if bool(row["eligible"])]
    if not eligible:
        raise ValueError(
            "no Flow checkpoint passed the physical and empirical D0 gates"
        )
    selected = min(
        eligible,
        key=lambda row: (
            float(_mapping(row["metrics"])["validation_loss"]),
            -int(row["step"]),
        ),
    )
    return {
        "schema": "ns_flow_selection_v2",
        "selected_step": int(selected["step"]),
        "rule": "empirical_d0_gate_then_lowest_validation_loss",
        "minimum_contract_pass_fraction": float(
            minimum_contract_pass_fraction
        ),
        "baseline_trial_count": len(rows),
        "baseline_lower_quantile": float(lower_quantile),
        "baseline_upper_quantile": float(upper_quantile),
        "empirical_thresholds": thresholds,
        "candidates": candidates,
    }


def _quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a quantile from an empty sequence")
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

def read_validation_losses(metrics_path: Path | str) -> dict[int, float]:
    """Read the latest validation loss recorded at every training step."""

    path = Path(metrics_path)
    losses: dict[int, float] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid Flow metrics JSON at {path}:{line_number}") from exc
        if row.get("event") == "validation":
            losses[int(row["step"])] = _finite_number(row.get("loss"), "validation loss")
    if not losses:
        raise ValueError(f"Flow metrics contain no validation rows: {path}")
    return losses


def write_selection(path: Path | str, payload: Mapping[str, Any]) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return output


def _competition_ranks(
    ordered: Iterable[Mapping[str, Any]], metric: str
) -> dict[int, int]:
    ranks: dict[int, int] = {}
    previous: float | None = None
    previous_rank = 0
    for position, row in enumerate(ordered, 1):
        value = float(_mapping(row["metrics"])[metric])
        rank = previous_rank if previous is not None and value == previous else position
        ranks[int(row["step"])] = rank
        previous = value
        previous_rank = rank
    return ranks


def _rows(report: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = report.get(key)
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise ValueError(f"report field {key!r} must be a list of objects")
    return list(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is missing or is not numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{label} is not finite")
    return resolved


def _optional_finite_number(value: Any) -> float | None:
    try:
        return _finite_number(value, "optional value")
    except ValueError:
        return None


def _load_object(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _parse_weight(value: str) -> tuple[str, float]:
    name, separator, raw = value.partition("=")
    if not separator or name not in DEFAULT_FLOW_WEIGHTS:
        raise argparse.ArgumentTypeError(
            "weight must be one of "
            + ", ".join(DEFAULT_FLOW_WEIGHTS)
            + " followed by =NUMBER"
        )
    return name, float(raw)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select validated NS model settings")
    commands = parser.add_subparsers(dest="command", required=True)
    pca = commands.add_parser("pca", help="Choose the PCA rank")
    pca.add_argument("--audit", required=True)
    pca.add_argument("--out", required=True)
    pca.add_argument("--metric", default="manifold_velocity_relative_l2")
    pca.add_argument("--statistic", default="median")
    pca.add_argument("--relative-tolerance", type=float, default=0.05)

    flow = commands.add_parser("flow", help="Choose the Base Generator checkpoint")
    flow.add_argument("--flow-root", required=True)
    flow.add_argument("--candidate-root", required=True)
    flow.add_argument("--steps", required=True, help="Colon-separated checkpoint steps")
    flow.add_argument("--out", required=True)
    flow.add_argument("--selected-checkpoint", required=True)
    flow.add_argument("--minimum-contract-pass-fraction", type=float, default=1.0)
    flow.add_argument(
        "--selection-rule",
        choices=("weighted_rank", "empirical_d0_gate"),
        default="weighted_rank",
    )
    flow.add_argument("--baseline-lower-quantile", type=float, default=0.05)
    flow.add_argument("--baseline-upper-quantile", type=float, default=0.95)
    flow.add_argument("--weight", action="append", type=_parse_weight, default=[])
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "pca":
        selection = select_pca_rank(
            _load_object(args.audit),
            metric=str(args.metric),
            statistic=str(args.statistic),
            relative_tolerance=float(args.relative_tolerance),
        )
        write_selection(args.out, selection)
        print(json.dumps(selection, indent=2, sort_keys=True), flush=True)
        return

    steps = [int(value) for value in str(args.steps).split(":") if value]
    candidate_root = Path(args.candidate_root).resolve()
    reports = {
        step: (
            _load_object(candidate_root / f"step_{step:07d}" / "sample" / "sample_audit.json"),
            _load_object(
                candidate_root
                / f"step_{step:07d}"
                / "materialized"
                / "answer_first_report.json"
            ),
        )
        for step in steps
    }
    selection = select_flow_checkpoint(
        validation_losses=read_validation_losses(Path(args.flow_root) / "metrics.jsonl"),
        candidate_reports=reports,
        minimum_contract_pass_fraction=float(args.minimum_contract_pass_fraction),
        weights=dict(args.weight) if args.weight else None,
        selection_rule=str(args.selection_rule),
        empirical_baseline_trials=(
            empirical_baseline_trials_from_reports(reports)
            if str(args.selection_rule) == "empirical_d0_gate"
            else None
        ),
        baseline_lower_quantile=float(args.baseline_lower_quantile),
        baseline_upper_quantile=float(args.baseline_upper_quantile),
    )
    selected_step = int(selection["selected_step"])
    source = Path(args.flow_root).resolve() / "checkpoints" / f"flow_step_{selected_step:07d}.pt"
    if not source.is_file():
        raise FileNotFoundError(source)
    link = Path(args.selected_checkpoint).resolve()
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != source.resolve():
            raise FileExistsError(
                f"selected checkpoint already points somewhere else: {link}"
            )
    else:
        link.symlink_to(os.path.relpath(source, link.parent))
    selection["selected_checkpoint"] = str(link)
    selection["selected_checkpoint_source"] = str(source)
    write_selection(args.out, selection)
    print(json.dumps(selection, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
