"""Bounded held-out diagnostics for a real learned VPFP answer checkpoint.

Finite positive manufactured answers establish structural validity only.
Distribution diagnostics against the held-out numerical corpus are reported
separately; neither constitutes high-fidelity or production admission.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pde_agent_verifier_solver.plugins.vpfp.canonical_binding import build_vpfp_online_binding
from pde_agent_verifier_solver.plugins.vpfp.learned_latent import (
    VPFPConditionalLatentDistribution, VPFPLearnedBatchGenerator, _file_sha256,
)
from pde_agent_verifier_solver.sdk import CaseVisibility


def distribution_summary(reference, generated):
    reference, generated = np.asarray(reference, dtype=float), np.asarray(generated, dtype=float)
    scale = np.maximum(reference.std(axis=0), 1.e-8)
    probabilities = np.linspace(.05, .95, 19)
    # Marginal Wasserstein diagnostic; it does not certify the joint law.
    w1 = np.mean(np.abs(np.quantile(reference, probabilities, axis=0)
                        - np.quantile(generated, probabilities, axis=0)), axis=0)
    return {
        "reference_count": len(reference), "generated_count": len(generated),
        "reference_mean": reference.mean(axis=0).tolist(),
        "generated_mean": generated.mean(axis=0).tolist(),
        "reference_std": reference.std(axis=0).tolist(),
        "generated_std": generated.std(axis=0).tolist(),
        "marginal_wasserstein_1_approx": w1.tolist(),
        "standardized_mean_gap_rms": float(np.sqrt(np.mean(
            ((reference.mean(axis=0) - generated.mean(axis=0)) / scale) ** 2))),
    }


def qualify(arguments):
    model = VPFPConditionalLatentDistribution.load(
        arguments.checkpoint, expected_sha256=arguments.checkpoint_sha256)
    with np.load(arguments.corpus, allow_pickle=False) as corpus:
        ids = np.asarray(corpus["sample_ids"])
        wanted = set(model.metadata["validation_sample_ids"])
        if wanted & set(model.metadata["train_sample_ids"]):
            raise ValueError("training and qualification sample identities overlap")
        selected = np.asarray([i for i, value in enumerate(ids) if int(value) in wanted])
        if len(selected) != len(wanted):
            raise ValueError("held-out sample identities are missing from corpus")
        reference_latents = np.asarray(corpus["latents"][selected], dtype=np.float32)
        reference_conditions = np.asarray(corpus["equation_conditions"][selected], dtype=np.float32)
    sampled_seeds = tuple(range(arguments.seed_start, arguments.seed_start + arguments.samples))
    if set(sampled_seeds) & (wanted | set(model.metadata["train_sample_ids"])):
        raise ValueError("qualification generated seeds overlap the offline corpus")
    sampler = VPFPLearnedBatchGenerator(model, checkpoint_sha256=arguments.checkpoint_sha256)
    generated_latents, generated_conditions = model.sample(sampled_seeds)
    payload = json.loads(arguments.config.read_text())
    binding = build_vpfp_online_binding(payload.get("vpfp", payload),
        output_root=arguments.report.parent / "qualification-runtime", device=arguments.device,
        latent_sampler=sampler)
    runtime = binding.plugin.generator.backend.runtime
    online_module = importlib.import_module("agent_verifier.vpfp.online_training")
    original_generate = online_module.VPFPOnlineBatchGenerator.generate
    online_calls = []

    def forbidden_online_solve(*args, **kwargs):
        online_calls.append(True)
        raise RuntimeError("learned sampling attempted an online coarse PDE solve")

    online_module.VPFPOnlineBatchGenerator.generate = forbidden_online_solve
    random = np.random.default_rng(arguments.seed_start + 1000000)
    times = random.uniform(*runtime.expanded_model.config.time_domain, size=8)
    domain = np.asarray(runtime.grid.x_domain)
    coordinates = random.uniform(size=(8, 2)) * domain
    macro_points = np.column_stack((times, coordinates))
    phase_points = np.column_stack((macro_points, random.normal(size=(8, 2))))

    def macro_vector(latent, condition):
        embedded = runtime.api.embed_vpfp_structured_latent(
            runtime.base_model.config, runtime.expanded_config, latent)
        values = runtime._evaluate_macro_latent(embedded, condition, macro_points)
        return np.concatenate((values["rho"], values["momentum"][:, 0],
                               values["momentum"][:, 1], values["kinetic_energy"]))

    source_rms, density_min, conditional_min, generated_macros = [], [], [], []
    try:
        reference_macros = np.stack([macro_vector(latent, condition)
            for latent, condition in zip(reference_latents, reference_conditions)])
        for seed in sampled_seeds:
            case_id = f"qualification-learned-{seed}"
            rows = runtime.generate_base(seeds=(seed,), case_ids=(case_id,),
                visibility=CaseVisibility.HIDDEN, round_index=0, cohort="qualification")
            if rows[0].transaction is not None:
                rows[0].transaction.commit()
            try:
                source = runtime.evaluate_source(case_id, phase_points)
                resident = runtime._resident_for_id(case_id)
                rho = runtime._evaluate_field_chunks(resident, phase_points, "rho")
                conditional = runtime._evaluate_field_chunks(resident, phase_points, "g")
                values = runtime.evaluate_macros(case_id, macro_points)
                generated_macros.append(np.concatenate((values["rho"], values["momentum"][:, 0],
                    values["momentum"][:, 1], values["kinetic_energy"])))
                if not all(np.isfinite(value).all() for value in (source, rho, conditional)):
                    raise ValueError("learned continuous answer/source contains nonfinite values")
                if np.min(rho) <= 0 or np.min(conditional) < 0:
                    raise ValueError("learned answer failed sampled positivity")
                source_rms.append(float(np.sqrt(np.mean(source * source))))
                density_min.append(float(np.min(rho)))
                conditional_min.append(float(np.min(conditional)))
            finally:
                runtime.release_cases((case_id,))
    finally:
        online_module.VPFPOnlineBatchGenerator.generate = original_generate
    standardized_reference = (reference_latents - model.latent_center) / model.latent_scale
    standardized_generated = (generated_latents - model.latent_center) / model.latent_scale
    return {
        "schema": "vpfp_learned_answer_held_out_diagnostics_v1",
        "execution_passed": True,
        "scientific_qualification": "insufficient_evidence",
        "qualification_reason": "small numerical corpus; no calibrated distribution admission or high-fidelity validation",
        "checkpoint_sha256": arguments.checkpoint_sha256,
        "corpus_sha256": _file_sha256(arguments.corpus),
        "training_data_origin": model.metadata["provenance"]["data_origin"],
        "experimental_data_validation": False,
        "held_out_sample_ids": sorted(wanted), "generated_seeds": list(sampled_seeds),
        "online_pde_calls_during_learned_sampling": len(online_calls),
        "source_rms_per_case": source_rms, "sampled_density_min_per_case": density_min,
        "sampled_conditional_density_min_per_case": conditional_min,
        "exact_source_derivation": "pinned analytic VPFP residual of sampled continuous answer",
        "mms_structure_does_not_certify_distribution_fidelity": True,
        "coefficient_diagnostics": {
            "standardized_mean_gap_rms": float(np.sqrt(np.mean((
                standardized_reference.mean(axis=0) - standardized_generated.mean(axis=0)) ** 2))),
            "reference_standardized_variance_mean": float(np.var(standardized_reference, axis=0).mean()),
            "generated_standardized_variance_mean": float(np.var(standardized_generated, axis=0).mean()),
        },
        "macro_fields": ["rho", "momentum_x", "momentum_y", "kinetic_energy"],
        "macro_points_txy": macro_points.tolist(),
        "macro_distribution": distribution_summary(reference_macros, np.stack(generated_macros)),
        "condition_distribution": distribution_summary(reference_conditions, generated_conditions),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "corpus", "checkpoint", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=9900000)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("real VPFP qualification requires a Slurm compute allocation")
    if arguments.samples < 2 or arguments.seed_start < 0:
        parser.error("need >=2 generated answers and nonnegative seeds")
    if arguments.report.exists():
        parser.error("report exists; choose a fresh artifact path")
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = qualify(arguments)
    except Exception as error:
        report = {"schema": "vpfp_learned_answer_held_out_diagnostics_v1",
                  "execution_passed": False, "scientific_qualification": "failed",
                  "error": repr(error), "traceback": traceback.format_exc()}
    with arguments.report.open("x") as target:
        json.dump(report, target, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({"report": str(arguments.report), "execution_passed": report["execution_passed"]}))
    return 0 if report["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
