#!/usr/bin/env python3
"""Summarize existing NS Generator audits; never train, sample, or reevaluate fields."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics


EXPECTED = {"seed": 20260912, "samples": 256, "integration_steps": 256,
            "flow_state": "raw", "sampling_policy": "rectified_flow"}
MMD_NOTE = ("The original rbf_mmd field is a zero-clipped MMD-squared U-statistic-form estimate, "
            "with an RBF bandwidth selected separately for each comparison from the median "
            "positive pooled pairwise squared distance; it is not the square root of MMD-squared.")
BASELINE_NOTE = (
    "Each repeat randomly partitions the 193 validation windows into two disjoint "
    "96-window groups, using the existing sampler and 32 repeats. This is a "
    "validation-half empirical D0 baseline, not a training bootstrap. Adjacent "
    "windows can share source frames, and repeats reuse windows; its variation "
    "is descriptive and is not an independent-experiment confidence interval."
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != payload:
            raise FileExistsError(f"Refusing to change existing artifact: {path}")
        return
    with path.open("x") as stream:
        stream.write(payload)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(not isinstance(value, bool), "Boolean metric")
    value = float(value)
    require(math.isfinite(value), "Nonfinite metric")
    return value


def describe(values):
    values = [number(value) for value in values]
    return {"count": len(values), "mean": statistics.fmean(values),
            "median": statistics.median(values), "minimum": min(values),
            "maximum": max(values), "population_std": statistics.pstdev(values)}


def source_contract(path):
    tree = ast.parse(path.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    samples = [node for node in calls if isinstance(node.func, ast.Attribute)
               and isinstance(node.func.value, ast.Name)
               and node.func.value.id == "model" and node.func.attr == "sample"]
    require(samples and all(any(item.arg == "method" and isinstance(item.value, ast.Constant)
                                and item.value.value == "heun" for item in node.keywords)
                            for node in samples), "Frozen sampler does not explicitly use Heun")
    baseline = [node for node in calls if isinstance(node.func, ast.Name)
                and node.func.id == "_empirical_two_sample_trials"]
    require(len(baseline) == 1 and isinstance(baseline[0].args[0], ast.Name)
            and baseline[0].args[0].id == "validation_metric", "Baseline source changed")
    return {"path": str(path), "sha256": digest(path), "model_sample_method": "heun",
            "empirical_baseline_argument": "validation_metric"}


def summarize_one(spec):
    directory, prepared = Path(spec["evaluation"]), Path(spec["prepared"])
    paths = {"quality": directory / "quality_report.json",
             "audit": directory / "latent_audit/sample_audit.json",
             "command": directory / "sampling_command.json",
             "contract": prepared / "continuous_contract.json",
             "train_manifest": prepared / "coefficients/train/coefficient_manifest.json",
             "validation_manifest": prepared / "coefficients/validation/coefficient_manifest.json"}
    quality, audit, command = (read_json(paths[key]) for key in ("quality", "audit", "command"))
    require(quality["schema"] == "ns_reconstructed_generator_quality_v1", "Unknown quality schema")
    require(audit["schema"] == "ns2d_structured_coefficient_flow_sample_audit_v1", "Unknown audit schema")
    for key, value in EXPECTED.items():
        require(audit[key] == value, f"{spec['id']}: mismatched {key}")
    require(audit["guidance"] is None, "Expected unguided Base draws")
    require(audit["flow_step"] == spec["step"], "Wrong checkpoint step")
    require(audit["pca_rank"] == spec["rank"], "Unexpected PCA rank")
    require(quality["generator_sample_count"] == 256 and quality["validation_case_count"] == 193,
            "Unexpected physical evaluation counts")
    require(quality["generated_values_finite"] is True, "Nonfinite generated velocity")
    require(quality["input_mode"] == "measured" and quality["pressure_model"] == "zero", "Input convention changed")
    require(Path(quality["prepared_directory"]).resolve() == prepared, "Wrong prepared dataset")
    for key in ("flow_checkpoint", "flow_checkpoint_sha256", "flow_state"):
        require(quality[key] == audit[key], f"Quality/audit {key} mismatch")
    argv = command["command"]
    options = dict(zip(argv[2::2], argv[3::2]))
    for option, value in {"--samples": "256", "--integration-steps": "256", "--seed": "20260912",
                          "--state": "raw", "--baseline-repeats": "32", "--metric-rank": str(spec["rank"]),
                          "--flow-checkpoint": audit["flow_checkpoint"], "--sample-batch": "128",
                          "--train-manifest": str(paths["train_manifest"]),
                          "--validation-manifest": str(paths["validation_manifest"])}.items():
        require(options.get(option) == value, f"Sampling command mismatch: {option}")
    implementation = source_contract(Path(argv[1]))
    distribution = audit["distribution"]
    require(distribution == quality["latent_distribution"], "Quality/audit distribution mismatch")
    require(distribution["metric_rank"] == spec["rank"] and distribution["comparison_case_count"] == 96,
            "Metric comparison changed")
    trials = distribution["empirical_d0_trials"]
    require(len(trials) == distribution["empirical_baseline_repeats"] == 32, "Wrong baseline repeats")
    require([row["repeat"] for row in trials] == list(range(32)), "Missing/reordered baseline repeat")
    baseline = {key: describe(row[key] for row in trials)
                for key in ("sliced_wasserstein", "mmd", "precision", "recall", "density", "coverage")}
    sw = number(distribution["generated_vs_validation_sliced_wasserstein"])
    sw_base = baseline["sliced_wasserstein"]["mean"]
    require(sw_base > 0, "Nonpositive SW baseline")
    prdc = distribution["generated_vs_validation_prdc"]
    relative = {
        "sliced_wasserstein": {"generated_vs_validation": sw, "baseline_mean": sw_base,
                               "ratio_to_baseline_mean": sw / sw_base, "gap_from_baseline_mean": sw - sw_base},
        "rbf_mmd": {"interpretation": MMD_NOTE, "generated_vs_validation": distribution["generated_vs_validation_rbf_mmd"],
                    "baseline_mean": baseline["mmd"]["mean"],
                    "gap_from_baseline_mean": distribution["generated_vs_validation_rbf_mmd"] - baseline["mmd"]["mean"]},
        "prdc": {key: {"generated_vs_validation": number(prdc[key]), "baseline_mean": baseline[key]["mean"],
                        "gap_from_baseline_mean": number(prdc[key]) - baseline[key]["mean"]}
                 for key in ("precision", "recall", "density", "coverage")},
    }
    velocity = {}
    for key in ("x", "y"):
        values = quality["velocity_rms"][key]
        mean_gen, mean_ref = (number(values[role]["mean"]) for role in ("generated", "validation"))
        velocity[key] = {**values, "generated_to_validation_mean_ratio": mean_gen / mean_ref,
                         "mean_relative_difference": mean_gen / mean_ref - 1,
                         "mean_difference": mean_gen - mean_ref}
    return {**spec, "sampling": {**EXPECTED, "method": "heun", "guidance": None, "sample_batch": 128},
            "checkpoint_sha256": audit["flow_checkpoint_sha256"],
            "sample_file_sha256_recorded_in_audit": audit["sample_file_sha256"],
            "source_files": {key: {"path": str(path), "sha256": digest(path)} for key, path in paths.items()},
            "sampler_implementation": implementation, "continuous_contract": read_json(paths["contract"]),
            "baseline": {"kind": "validation_half_empirical_d0", "note": BASELINE_NOTE,
                         "summary_of_32_repeats": baseline, "trials": trials},
            "relative_to_own_baseline": relative, "velocity_rms": velocity,
            "velocity_reference": "Each run's held-out continuous fields evaluated on native measured support; RMS per component over time and sites, then summarized over windows.",
            "paired_representation_errors": {key: value for key, value in quality.items()
                                              if key.endswith("validation_relative_l2")},
            "effective_rank": distribution["effective_rank"],
            "nearest_train_distance": distribution["nearest_train_distance"],
            "score_statistics": distribution["score_statistics"],
            "original_latent_distribution": distribution}


def markdown(report):
    rows = report["checkpoints"]
    lines = ["# Cylinder Generator distribution comparison", "", report["scope"], "", BASELINE_NOTE, "",
             "All four existing reports pass the fixed sampling checks: seed 20260912, 256 raw Base draws, 256 Heun steps, no guidance. The distribution comparison uses 96 generated versus 96 validation windows; velocity summaries use all 256 versus 193 windows.", "",
             "| Checkpoint | Role | Rank | SW / own baseline mean | SW gap in own coordinates | Precision / baseline | Recall / baseline |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        metrics = row["relative_to_own_baseline"]; sw = metrics["sliced_wasserstein"]
        p, r = (metrics["prdc"][key] for key in ("precision", "recall"))
        lines.append(f"| {row['id']} | {row['role']} | {row['rank']} | {sw['ratio_to_baseline_mean']:.4f} | {sw['gap_from_baseline_mean']:.6g} | {p['generated_vs_validation']:.4f} / {p['baseline_mean']:.4f} | {r['generated_vs_validation']:.4f} / {r['baseline_mean']:.4f} |")
    lines += ["", "SW ratios are descriptive calibration against each run's own finite-sample baseline. Raw SW/gaps and nearest-neighbor distances are not directly comparable across different ranks or PCA coordinate systems. Baseline-normalized ratios also do not establish a universal rank-invariant quality score.", "",
              "| Checkpoint | u RMS generated / validation | u relative mean difference | v RMS generated / validation | v relative mean difference |",
              "|---|---:|---:|---:|---:|"]
    for row in rows:
        x, y = row["velocity_rms"]["x"], row["velocity_rms"]["y"]
        lines.append(f"| {row['id']} | {x['generated']['mean']:.8g} / {x['validation']['mean']:.8g} | {100*x['mean_relative_difference']:+.3f}% | {y['generated']['mean']:.8g} / {y['validation']['mean']:.8g} | {100*y['mean_relative_difference']:+.3f}% |")
    lines += ["", "These RMS values summarize unpaired Generator draws against each run's held-out continuous representation. They are distribution statistics, not pointwise reconstruction errors, and the reference itself changes with the representation. Pressure is manufactured zero.", "",
              "| Checkpoint | Effective rank train / validation / generated | Nearest-train mean validation / generated | Coordinate std mean train / validation / generated |",
              "|---|---:|---:|---:|"]
    for row in rows:
        rank, nearest, scores = row["effective_rank"], row["nearest_train_distance"], row["score_statistics"]
        roles = ("train", "validation", "generated")
        rank_text = " / ".join(f"{rank[key]:.4g}" for key in roles)
        std_text = " / ".join(f"{scores[key]['coordinate_std_mean']:.4g}" for key in roles)
        lines.append(f"| {row['id']} | {rank_text} | {nearest['validation']['mean']:.4g} / {nearest['generated']['mean']:.4g} | {std_text} |")
    lines += ["", "A mismatch with held-out D0 is an observed distribution mismatch. These reports lack a matched train-versus-validation distribution null, so they cannot assign the mismatch entirely to Flow; training/validation shift and model fit may both contribute. No checkpoint is selected by this summary. The new final 5000-step result is the primary comparison matching the old budget; 1000 and 3000 steps remain predeclared diagnostics.", "",
              "Full baseline repeat distributions, PRDC density/coverage, the zero-clipped adaptive-bandwidth RBF-MMD-squared estimate, latent statistics, paired representation-stage errors, configurations, and source hashes are retained in the JSON. This summary reads existing JSON reports and source code only; it launches no model, sampling, fitting, or solver evaluation.", ""]
    lines.extend(f"- [{row['id']} quality report]({row['source_files']['quality']['path']})" for row in rows)
    return "\n".join(lines) + "\n"


def freeze(args):
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    old, new, diagnostics = (Path(getattr(args, key)).resolve() for key in ("old_prepared", "new_prepared", "diagnostics"))
    script = Path(__file__).resolve()
    target = out / "distribution_summary_snapshot" / script.name
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        require(digest(target) == digest(script), "Frozen summary script already differs")
    else:
        shutil.copyfile(script, target)
    specs = [{"id": "v2_5000", "role": "old primary", "step": 5000, "rank": 64,
              "prepared": str(old), "evaluation": str(old / "evaluation_flow_step_0005000_raw")}]
    for step in (1000, 3000, 5000):
        path = new / "evaluation_flow_step_0005000_raw" if step == 5000 else diagnostics / f"step_{step}"
        specs.append({"id": f"v3_{step}", "role": "new primary" if step == 5000 else "diagnostic only",
                      "step": step, "rank": 256, "prepared": str(new), "evaluation": str(path)})
    plan = {"schema": "ns_existing_generator_distribution_summary_plan_v1",
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "out": str(out),
            "script": str(target), "script_sha256": digest(target), "checkpoints": specs,
            "fixed_sampling": EXPECTED, "baseline_note": BASELINE_NOTE,
            "primary_rule": "Compare old 5000 to new 5000; 1000/3000 diagnostic only, no checkpoint selection.",
            "operations": "Read existing reports/source and arithmetic only; no sampling, training, or field reevaluation."}
    write_once(out / "generator_distribution_summary_plan.json", plan)
    print(json.dumps({"plan": str(out / "generator_distribution_summary_plan.json"), "script": str(target)}))


def summarize(args):
    plan = read_json(args.plan)
    require(digest(Path(__file__).resolve()) == plan["script_sha256"], "Summary script is not the frozen implementation")
    rows = [summarize_one(spec) for spec in plan["checkpoints"]]
    require(all(row["baseline"]["trials"] == rows[1]["baseline"]["trials"] for row in rows[2:]),
            "New checkpoints do not share an identical validation-half baseline")
    report = {"schema": "ns_existing_generator_distribution_summary_v1",
              "scope": "Primary comparison: old rank64, 5000 steps versus new rank256, 5000 steps. New 1000/3000 checkpoints are predeclared diagnostics only.",
              "plan": {"path": str(Path(args.plan).resolve()), "sha256": digest(args.plan)},
              "checks": {"fixed_sampling_passed": True, "new_baselines_identical": True,
                         "quality_audit_binding_passed": True, "no_new_sampling_or_training": True},
              "interpretation_limits": [BASELINE_NOTE,
                  "Never compare raw latent SW across ranks/PCA representations as the same metric space.",
                  "Unpaired generated draws have distribution statistics, not paired reconstruction errors.",
                  "No matched train-versus-validation distribution null is available; do not attribute all mismatch to Flow.",
                  "Representation-stage relative L2 metrics are paired fits; Generator draws are not.",
                  "Ratios to own baselines remain descriptive and are not universal rank-invariant scores."],
              "checkpoints": rows}
    out = Path(plan["out"])
    write_once(out / "generator_distribution_summary.json", report)
    rendered = markdown(report)
    md = out / "generator_distribution_summary.md"
    if md.exists():
        require(md.read_text() == rendered, "Existing Markdown differs")
    else:
        with md.open("x") as stream:
            stream.write(rendered)
    print(json.dumps({"summary": str(out / "generator_distribution_summary.json"), "markdown": str(md),
                      "checkpoints": [row["id"] for row in rows]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    freeze_parser = commands.add_parser("freeze")
    for name in ("old-prepared", "new-prepared", "diagnostics", "out"):
        freeze_parser.add_argument(f"--{name}", required=True)
    run_parser = commands.add_parser("summarize")
    run_parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    (freeze if args.mode == "freeze" else summarize)(args)


if __name__ == "__main__":
    main()
