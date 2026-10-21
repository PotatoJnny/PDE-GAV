"""Bounded, train-fitted VPFP noise/decoder diagnostics; never an admission test.

Reuse real numerical trajectories and frozen generators without any PDE solve.
Conditional-mean reconstructions and clean-plus-noise sensitivity probes are
explicitly not free generated samples or distribution-validation evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
loader = importlib.util.spec_from_file_location("vpfp_geometry_shared", ROOT / "scripts/train_vpfp_structured_diffusion.py")
shared = importlib.util.module_from_spec(loader)
loader.loader.exec_module(shared)

from pde_agent_verifier_solver.plugins.vpfp.canonical_binding import VPFPOnlineModelSpec, build_vpfp_online_binding
from pde_agent_verifier_solver.plugins.vpfp.diffusion_data import read_manifest, load_split
from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import load_vpfp_diffusion_sampler
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import _value_split
from pde_agent_verifier_solver.plugins.vpfp.noise_geometry import VPFPTemporalCovarianceTransform
from pde_agent_verifier_solver.plugins.vpfp.derivative_features import (
    VPFPDerivativeFeaturePlan, extract_vpfp_derivative_features, _probe_points,
)


def jet_summary(matrix, descriptors, plan):
    """Per-trajectory RMS first, then ensemble summaries; not a domain norm."""
    names = [entry["name"] for entry in descriptors]
    output = {}
    for quantity in ("f", "rho", "momentum_x", "momentum_y", "kinetic_energy"):
        count = plan.phase_probe_count if quantity == "f" else plan.macro_probe_count
        for component in ("value", "dt", "dx", "dy"):
            for window, mask in (
                ("all", np.ones(len(plan.time_fractions), dtype=bool)),
                ("early", np.asarray(plan.time_fractions) <= .25),
                ("late", np.asarray(plan.time_fractions) >= .5),
            ):
                columns = [names.index(f"{quantity}.probe{t * count + p}.{component}")
                           for t in np.flatnonzero(mask) for p in range(count)]
                rms = np.sqrt(np.mean(np.asarray(matrix[:, columns], dtype=np.float64)**2, axis=1))
                output[f"{quantity}.{component}.{window}"] = {
                    "mean_rms": float(rms.mean()), "median_rms": float(np.median(rms)),
                    "p90_rms": float(np.quantile(rms, .9)), "maximum_rms": float(rms.max()),
                }
    return output


def raw_amplitude_jets(decoder, points, latent):
    """Replay the pinned direct decoder BEFORE a/||a||, without altering it."""
    cfg = decoder.config
    if not cfg.structured_direct_coefficients or float(cfg.structured_residual_scale) != 0.:
        raise ValueError("raw amplitude replay requires the direct zero-residual decoder")
    parameter = next(decoder.parameters())
    query = torch.as_tensor(points, device=parameter.device, dtype=parameter.dtype)
    raw = torch.as_tensor(latent, device=parameter.device, dtype=parameter.dtype).reshape(1, -1)
    rows = (raw * cfg.latent_input_scale).expand(len(query), -1)

    def function(q):
        tau = decoder._coordinate_features(q)[2]
        return decoder._structured_latent_features(q, rows, tau)[2]

    direction = torch.zeros_like(query)
    direction[:, 0] = 1.
    with torch.enable_grad():
        amplitude, dt = torch.autograd.functional.jvp(function, query, direction, create_graph=False, strict=True)
    norm = torch.linalg.vector_norm(amplitude, dim=-1)
    if not torch.isfinite(amplitude).all() or not torch.isfinite(dt).all() or torch.any(norm <= torch.finfo(norm.dtype).eps):
        raise ValueError("raw amplitude replay encountered a nonfinite or decoder fallback point")
    unit = amplitude / norm[:, None]
    tangent = dt - unit * (unit * dt).sum(-1, keepdim=True)
    angular_dt = torch.linalg.vector_norm(tangent, dim=-1) / norm
    with torch.no_grad():
        # This checks that the diagnostic uses the very same pre-normalized
        # amplitude as the production fields(), not an approximate re-encoder.
        condition = [np.mean(cfg.collision_nu_range), np.mean(cfg.temperature_range)]
        actual = decoder.fields(query, raw, equation_condition=condition)["hermite_coefficients"]
        if not torch.allclose(unit, actual, rtol=2.e-5, atol=2.e-6):
            raise ValueError("raw amplitude diagnostic differs from the production decoder")
    return norm.detach().cpu().numpy(), angular_dt.detach().cpu().numpy()


def amplitude_summary(decoder, raw, plan, extra_points=None):
    macro, phase = _probe_points(decoder.config, plan, np.float32)
    points = np.concatenate((macro, phase), axis=0)
    if extra_points is not None:
        extra_points = np.asarray(extra_points, dtype=np.float32)
        if extra_points.ndim != 2 or extra_points.shape[1] != 5 or not np.isfinite(extra_points).all():
            raise ValueError("extra amplitude points must be finite [N,5]")
        points = np.concatenate((points, extra_points), axis=0)
    rows = []
    for latent in raw:
        norm, angular = raw_amplitude_jets(decoder, points, latent)
        rows.append({"minimum_norm": float(norm.min()), "norm_q01": float(np.quantile(norm, .01)),
            "median_norm": float(np.median(norm)), "fraction_norm_lt_01": float(np.mean(norm < .1)),
            "angular_dt_rms": float(np.sqrt(np.mean(angular.astype(np.float64)**2))),
            "angular_dt_maximum": float(angular.max()),
            "minimum_norm_probe": int(norm.argmin()), "angular_dt_maximum_probe": int(angular.argmax()),
            "norm_by_probe": norm.tolist(), "angular_dt_by_probe": angular.tolist()})
    return {"probe_order": "macro_then_phase_then_extra_source_points; derivative arrays time_major",
            "points": points.tolist(),
            "per_case": rows, "minimum_norm": min(row["minimum_norm"] for row in rows),
            "mean_angular_dt_rms": float(np.mean([row["angular_dt_rms"] for row in rows]))}


def extract_diagnostic(decoder, raw, conditions, path, *, origin, extra_points=None):
    shared.emit("physical_diagnostic", name=path.stem, count=len(raw))
    plan = VPFPDerivativeFeaturePlan()
    matrix, schema, receipt = extract_vpfp_derivative_features(decoder, raw, conditions, plan=plan)
    np.save(path.with_suffix(".npy"), matrix, allow_pickle=False)
    report = {"origin": origin, "scientific_distribution_validation": False,
        "plan": asdict(plan), "summary": jet_summary(matrix, schema, plan),
        "amplitudes": amplitude_summary(decoder, raw, plan, extra_points),
        "answer_sha256": [row["input_answer_sha256"] for row in receipt["per_case"]],
        "feature_file_sha256": shared.file_sha(path.with_suffix(".npy"))}
    shared.write_json(path, report)
    return report


def reduced_error(prediction, target, config):
    rho, kinetic = _value_split(prediction - target, config)
    result = {"rho_mse": float(rho.square().mean()), "kinetic_mse": float(kinetic.square().mean())}
    for label, value in (("rho", rho), ("kinetic", kinetic)):
        result[f"{label}_first_difference_mse"] = float(torch.diff(value, dim=-2).square().mean())
    return result


def run(arguments):
    started = time.monotonic()
    manifest = read_manifest(arguments.cache / "corpus", verify_hashes=True)
    spec = VPFPOnlineModelSpec.from_mapping(manifest["model_spec"])
    checkpoint = arguments.pilot / "seed-0/value_base/checkpoints/best.pt"
    digest = shared.file_sha(checkpoint)
    sampler = load_vpfp_diffusion_sampler(checkpoint, digest, device=arguments.device)
    training = load_split(arguments.cache / "corpus", "train", verify_hashes=False)
    validation = load_split(arguments.cache / "corpus", "validation", verify_hashes=False)
    shared.assert_checkpoint_corpus(sampler, manifest, training.sample_ids)
    arguments.output.mkdir(parents=True, exist_ok=False)
    plan = {k: str(v) if isinstance(v, Path) else v for k, v in vars(arguments).items()}
    plan.update(schema="vpfp_noise_geometry_diagnostic_v1", source_revision=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), checkpoint_sha256=digest,
        corpus_manifest_sha256=manifest["manifest_sha256"], new_pde_solves=0,
        no_model_training=True, scientific_distribution_validation=False,
        fit_split="train_only", diagnostic_split="already_opened_validation",
        deliberately_selected_generated_outliers="rows 12,16,50 retained in addition to first 16; mechanism only",
        sigmas=[.007521206, .03, .106365918, .3, 1.50424124, 5., 21.2731836],
        noise_sensitivity_sigmas=[.01, .03], covariance_relative_floor=.05)
    physical_receipt = arguments.pilot / "seed-0/full_base/generated-physical.json"
    plan["extra_source_probe_receipt_sha256"] = shared.file_sha(physical_receipt)
    source_points = json.loads(physical_receipt.read_text())["physical"]["phase_points"]
    shared.write_json(arguments.output / "frozen-plan.json", plan)
    normalizer, cfg = sampler.normalizer, sampler.model.config
    shared.emit("fit_train_temporal_covariance", regime=arguments.regime, count=len(training))
    normalized = normalizer.encode(torch.from_numpy(np.array(training.latents, copy=True)).to(arguments.device))
    transform = VPFPTemporalCovarianceTransform.fit(cfg, normalized,
        sample_ids=training.sample_ids.tolist(), center_kinetic_dc=True, relative_floor=.05).to(arguments.device)
    shared.write_json(arguments.output / "train-covariance.json", transform.to_payload())
    shared.write_json(arguments.output / "train-geometry-report.json", transform.report())
    del normalized
    selected = np.arange(min(arguments.answers, len(validation)))
    raw = np.array(validation.latents[selected], copy=True)
    conditions = np.array(validation.conditions[selected], copy=True)
    clean = normalizer.encode(torch.from_numpy(raw).to(arguments.device))
    random = torch.Generator(device=arguments.device).manual_seed(arguments.seed)
    noise = torch.randn(clean.shape, device=arguments.device, generator=random)
    binding = build_vpfp_online_binding(spec, output_root=arguments.output / "runtime", device=arguments.device, latent_sampler=sampler)
    decoder = binding.plugin.generator.backend.runtime.base_model
    summaries = {}
    summaries["clean_validation"] = extract_diagnostic(decoder, raw, conditions,
        arguments.output / "clean-validation.json", origin="held_out_numerical_trajectory_not_generated",
        extra_points=source_points)
    denoise = []
    for sigma in plan["sigmas"]:
        sigma_tensor = clean.new_full((len(clean),), sigma)
        with torch.no_grad():
            noisy = clean + sigma * noise
            prediction = sampler.model.denoise(noisy, sigma_tensor, conditions)
            gaussian = transform.posterior_mean(noisy, sigma_tensor)
        row = {"sigma": sigma, "learned": reduced_error(prediction, clean, cfg),
               "train_temporal_gaussian": reduced_error(gaussian, clean, cfg),
               "identity": reduced_error(noisy, clean, cfg)}
        denoise.append(row)
        shared.emit("denoising_diagnostic", regime=arguments.regime, **row)
        if sigma == plan["sigmas"][1]:
            for label, values in (("learned_denoised", prediction), ("gaussian_denoised", gaussian)):
                decoded = normalizer.decode(values).detach().cpu().numpy()
                summaries[label] = extract_diagnostic(decoder, decoded, conditions,
                    arguments.output / f"{label}.json", origin="paired_conditional_mean_reconstruction_not_free_generation")
    shared.write_json(arguments.output / "denoising-by-sigma.json", {"rows": denoise,
        "not_a_distribution_test": True, "paired_noise_identifiers": True, "validation_ids": validation.sample_ids[selected].tolist()})
    # Compare temporal coloring at equal finite-batch, per-field coefficient
    # noise energy: any derivative benefit cannot be attributed to smaller RMS.
    colored = transform.decode(noise) - transform.decode(torch.zeros_like(noise))
    white_groups, color_groups = _value_split(noise, cfg), _value_split(colored, cfg)
    balanced = []
    for white, color in zip(white_groups, color_groups):
        axes = (0, 1, 2) if white.ndim == 3 else (0, 2, 3)
        factor = (white.square().mean(axes) / color.square().mean(axes)).sqrt()
        balanced.append(color * (factor if white.ndim == 3 else factor[None, :, None, None]))
    colored = torch.cat([value.flatten(1) for value in balanced], dim=1)
    shared.write_json(arguments.output / "noise-energy-match.json", {
        "matching": "finite_batch_per_field_group_in_existing_value_normalized_coordinates",
        "white": reduced_error(noise, torch.zeros_like(noise), cfg),
        "colored": reduced_error(colored, torch.zeros_like(colored), cfg),
        "not_a_free_generated_sample": True})
    for sigma in plan["noise_sensitivity_sigmas"]:
        for label, perturbation in (("white", noise), ("temporal_colored", colored)):
            decoded = normalizer.decode(clean + sigma * perturbation).detach().cpu().numpy()
            key = f"{label}_noise_{sigma:g}"
            summaries[key] = extract_diagnostic(decoder, decoded, conditions,
                arguments.output / f"{key}.json", origin="oracle_clean_plus_matched_energy_noise_not_free_generation")
    for arm in ("full_base", "full_time", "value_base", "value_time"):
        directory = arguments.pilot / "seed-0" / arm
        generated = np.load(directory / "generated-coefficients.npy", mmap_mode="r", allow_pickle=False)
        generated_conditions = np.load(directory / "generated-conditions.npy", allow_pickle=False)
        rows = sorted(set(range(min(arguments.answers, len(generated)))) | {i for i in (12,16,50) if i < len(generated)})
        key = f"existing_{arm}"
        summaries[key] = extract_diagnostic(decoder, np.array(generated[rows], copy=True), generated_conditions[rows],
            arguments.output / f"{key}.json", origin="existing_free_generation_deliberately_includes_known_outliers",
            extra_points=source_points)
        summaries[key]["selected_original_rows"] = rows
        summaries[key]["source_coefficient_sha256"] = shared.file_sha(directory / "generated-coefficients.npy")
    read_manifest(arguments.cache / "corpus", verify_hashes=True)
    return {"execution_passed": True, "regime": arguments.regime, "wall_seconds": time.monotonic() - started,
        "summaries": summaries, "denoising": denoise, "training_performed": False,
        "new_pde_solves": 0, "scientific_distribution_validation": False, "admitted": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "pilot", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--regime", choices=("weak", "strong"), required=True)
    parser.add_argument("--seed", type=int, default=2026090507)
    parser.add_argument("--answers", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("real-data noise geometry diagnostics require a Slurm compute allocation")
    if arguments.answers < 1 or arguments.answers > 32 or arguments.seed < 0:
        parser.error("bounded diagnostics require 1..32 answers and nonnegative seed")
    if arguments.output.exists() or arguments.output.is_symlink() or any(
        arguments.output.resolve().is_relative_to(path.resolve()) for path in (arguments.cache, arguments.pilot)):
        parser.error("choose a new output outside the immutable source runs")
    torch.set_num_threads(min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    try:
        report = run(arguments)
    except Exception as error:
        report = {"execution_passed": False, "error": repr(error), "traceback": traceback.format_exc(),
                  "partial_artifacts_preserved": True, "admitted": False}
    if arguments.output.is_dir():
        shared.write_json(arguments.output / "result.json", report)
    shared.emit("noise_geometry_complete", execution_passed=report["execution_passed"], error=report.get("error"))
    return 0 if report["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
