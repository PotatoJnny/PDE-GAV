#!/usr/bin/env python
"""Audit a measured- or reconstructed-data Generator with the original numerical Flow sampler.

Generator samples are unpaired draws, so no per-sample error against an unrelated
experimental trajectory is reported. Native-point plots and distribution metrics
separate reconstruction, continuous projection, PCA, and generation quality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from prepare_ns_reconstructed_generator import (
    ROOT, file_sha256, relative_errors, summary, velocity_from_coefficients, write_json,
)
from agent_verifier.neural_generator.structured_ns_distribution import StructuredCoefficientPCA


def plots(out, *, x, y, times, source_velocity, projected, compressed, generated, input_mode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fields = [source_velocity[0], projected[0], compressed[0], generated[0]]
    titles = ["Held-out measurements" if input_mode == "measured" else "Held-out reconstruction",
              "Continuous projection", "PCA reconstruction", "Unpaired Generator sample"]
    frame = len(times)//2
    speed = [np.linalg.norm(field[frame], axis=0) for field in fields]
    maximum = max(float(value.max()) for value in speed)
    figure, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    for axis, values, title in zip(axes, speed, titles):
        artist = axis.scatter(x, y, c=values, s=7, marker="s", vmin=0, vmax=maximum, cmap="viridis")
        axis.set(title=title, xlabel="x", ylabel="y", aspect="equal")
    figure.colorbar(artist, ax=axes, label="Speed in declared velocity units", shrink=.7)
    figure.suptitle(f"Native measured support, window-local t={times[frame]:.5g}; Generator draw is unpaired")
    figure.savefig(out / "velocity_fields.png", dpi=160)
    plt.close(figure)
    reference_rms = np.sqrt(np.mean(projected**2, axis=(1, 3)))
    generated_rms = np.sqrt(np.mean(generated**2, axis=(1, 3)))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for index, axis in enumerate(axes):
        bins = np.histogram_bin_edges(np.r_[reference_rms[:, index], generated_rms[:, index]], bins=20)
        axis.hist(reference_rms[:, index], bins=bins, density=True, alpha=.55, label="Held-out continuous fields")
        axis.hist(generated_rms[:, index], bins=bins, density=True, alpha=.55, label="Generator")
        axis.set(xlabel=f"Window RMS velocity {'x' if index == 0 else 'y'}", ylabel="Density")
        axis.legend()
    figure.savefig(out / "velocity_distributions.png", dpi=160)
    plt.close(figure)


def evaluate(args):
    prepared = Path(args.prepared).expanduser().resolve(strict=True)
    preparation_config = json.loads((prepared / "preparation_config.json").read_text())
    input_mode = preparation_config.get("input_mode", "reconstructed")
    if input_mode not in ("measured", "reconstructed"):
        raise ValueError("prepared input_mode must be measured or reconstructed")
    if getattr(args, "input_mode", None) not in (None, input_mode):
        raise ValueError("evaluation input_mode differs from the frozen preparation")
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    else:
        checkpoint = Path(json.loads((prepared / "base_generator/train_report.json").read_text())["final_checkpoint"])
    out = Path(args.out).expanduser().resolve() if args.out else prepared / f"evaluation_{checkpoint.stem}_{args.state}"
    if out.exists():
        raise FileExistsError(f"refusing to reuse Generator evaluation directory: {out}")
    out.mkdir(parents=True)
    contract = json.loads((prepared / "continuous_contract.json").read_text())
    selection = json.loads((prepared / "pca_selection.json").read_text())
    validation_manifest = prepared / "coefficients/validation/coefficient_manifest.json"
    validation_count = json.loads(validation_manifest.read_text())["encoded_cases"]
    command = [sys.executable, str(ROOT / "legacy/ns/scripts/sample_ns_structured_coefficient_flow.py"),
        "--flow-checkpoint", str(checkpoint), "--pca-checkpoint", str(prepared / "pca_checkpoint.pt"),
        "--train-manifest", str(prepared / "coefficients/train/coefficient_manifest.json"),
        "--validation-manifest", str(validation_manifest), "--out", str(out / "latent_audit"),
        "--samples", str(args.samples), "--sample-batch", str(min(args.samples, 128)),
        "--integration-steps", str(args.integration_steps), "--train-audit-cases", "1000000",
        "--validation-audit-cases", str(validation_count), "--metric-rank", str(selection["selected_rank"]),
        "--baseline-repeats", str(args.baseline_repeats), "--state", args.state,
        "--seed", str(args.seed), "--device", args.device]
    write_json(out / "sampling_command.json", {"command": command})
    with (out / "sampling.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    audit = json.loads((out / "latent_audit/sample_audit.json").read_text())
    with np.load(out / "latent_audit/structured_flow_samples.npz") as data:
        samples = data["coefficients"].copy()
    with np.load(prepared / "validation_reference.npz") as data:
        x, y, times = (data[key].copy() for key in ("x", "y", "time"))
        reference_coefficients = data["coefficients"].copy()
        source_velocity = data["measured" if input_mode == "measured" else "reconstructed"].copy()
        if input_mode == "reconstructed":
            validation_coefficients = data["reconstruction_validation_coefficients"].copy()
            truth, masks = data["reconstruction_validation_truth"].copy(), data["reconstruction_validation_mask"].copy()
    kwargs = {"x": x, "y": y, "times": times, "geometry": contract["geometry"],
              "projection_config": contract["projection_config"], "device": args.device}
    projected = velocity_from_coefficients(reference_coefficients, **kwargs)
    generated = velocity_from_coefficients(samples, **kwargs)
    pca = StructuredCoefficientPCA.load(prepared / "pca_checkpoint.pt").with_rank(selection["selected_rank"])
    compressed = velocity_from_coefficients(pca.reconstruct(reference_coefficients).cpu().numpy(), **kwargs)
    projection_key = ("continuous_projection_to_measured_validation_relative_l2" if input_mode == "measured"
                      else "continuous_projection_to_reconstructed_validation_relative_l2")
    stage_metrics = {projection_key: summary(relative_errors(projected, source_velocity)),
                     "pca_to_continuous_validation_relative_l2": summary(relative_errors(compressed, projected))}
    stage_labels = [("Continuous fit against measured windows" if input_mode == "measured"
                     else "Continuous fit against reconstructed windows", projection_key),
                    ("PCA against continuous fields", "pca_to_continuous_validation_relative_l2")]
    if input_mode == "reconstructed":
        validation_velocity = velocity_from_coefficients(validation_coefficients, **kwargs)
        withheld_errors = []
        for fitted, reference, mask in zip(validation_velocity, truth, masks):
            visible = np.broadcast_to(mask[:, None, :], fitted.shape)
            withheld_errors.append(float(np.linalg.norm((fitted-reference)[visible]) /
                                        max(np.linalg.norm(reference[visible]), 1e-12)))
        stage_metrics["continuous_reconstruction_withheld_measured_relative_l2"] = summary(withheld_errors)
        stage_labels.append(("Reconstruction + continuous fit against withheld measurements",
                             "continuous_reconstruction_withheld_measured_relative_l2"))
    rms = lambda values: np.sqrt(np.mean(values**2, axis=(1, 3)))
    reference_rms, generated_rms = rms(projected), rms(generated)
    report = {"schema": "ns_reconstructed_generator_quality_v1", "prepared_directory": str(prepared),
        "flow_checkpoint": str(checkpoint), "flow_checkpoint_sha256": file_sha256(checkpoint),
        "flow_state": args.state, "generator_sample_count": len(samples), "validation_case_count": len(projected),
        "input_mode": input_mode, **stage_metrics,
        "latent_distribution": audit["distribution"],
        "velocity_rms": {component: {"validation": summary(reference_rms[:, i]), "generated": summary(generated_rms[:, i])}
                         for i, component in enumerate(("x", "y"))},
        "generated_values_finite": bool(np.isfinite(generated).all()),
        "pressure_model": "zero", "manufactured_viscosity": preparation_config.get("viscosity"),
        "viscosity_used_for_training_or_metrics": False,
        "evaluation_scope": "Generator quality only; final_test/verifier/development excluded from fitting and selection; no NS solver experiment",
        "plots": ["velocity_fields.png", "velocity_distributions.png"],
        "independence_note": "Adjacent windows overlap within their declared split; window count is not an independent sample count."}
    plots(out, x=x, y=y, times=times, source_velocity=source_velocity, projected=projected,
          compressed=compressed, generated=generated, input_mode=input_mode)
    write_json(out / "quality_report.json", report)
    input_description = ("Full measured fields enter the numerical projector directly; no observation reconstruction is used."
                         if input_mode == "measured" else "Sparse-observation reconstruction precedes them.")
    (out / "quality_report.md").write_text(
        f"# {input_mode.capitalize()}-data numerical NS Generator\n\n"
        f"Checkpoint: `{checkpoint}` ({args.state}); {len(samples)} generated windows.\n\n"
        "The continuous representation, PCA, Flow model/trainer and Heun sampler are the numerical NS implementations. "
        f"{input_description} Pressure is manufactured zero; viscosity is not required for this report.\n\n"
        "| Validation stage | Median relative L2 | Mean relative L2 |\n|---|---:|---:|\n" +
        "\n".join(f"| {title} | {report[key]['median']:.6g} | {report[key]['mean']:.6g} |" for title, key in stage_labels) +
        "\n\nGenerated trajectories are unpaired draws. Compare their distribution with the held-out continuous fields in "
        "`quality_report.json` and `latent_audit/sample_audit.md`; the fields figure is not a paired prediction-error plot. "
        "Overlapping windows are not independent repetitions. No solver results or untouched final-test scores are claimed.\n\n"
        "![Velocity fields](velocity_fields.png)\n\n![Velocity distributions](velocity_distributions.png)\n")
    print(json.dumps({"quality_report": str(out / "quality_report.json"),
                      "plots": [str(out / item) for item in report["plots"]]}, indent=2), flush=True)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--prepared", required=True)
    result.add_argument("--input-mode", choices=("measured", "reconstructed"),
                        help="Optional assertion; the mode is otherwise read from the frozen preparation.")
    result.add_argument("--checkpoint")
    result.add_argument("--out")
    result.add_argument("--samples", type=int, default=256)
    result.add_argument("--integration-steps", type=int, default=256)
    result.add_argument("--baseline-repeats", type=int, default=32)
    result.add_argument("--state", choices=("raw", "ema"), default="raw")
    result.add_argument("--device", default="cuda")
    result.add_argument("--seed", type=int, default=20260912)
    return result


if __name__ == "__main__":
    evaluate(parser().parse_args())
