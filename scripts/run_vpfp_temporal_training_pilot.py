"""Small VPFP temporal-geometry x noise-coverage training experiment.

All arms use the same value-state network and base EDM regression, without an
old scalar-normalizer Sobolev operator. Opened references are development data;
mechanism evidence and physical effects inform a subsequent larger experiment,
not automatic model admission or a claim of distribution equivalence.
"""
from __future__ import annotations

import argparse
from copy import copy
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
loader = importlib.util.spec_from_file_location("vpfp_temporal_pilot_reuse", ROOT / "scripts/run_vpfp_value_control_pilot.py")
previous = importlib.util.module_from_spec(loader)
loader.loader.exec_module(previous)
shared = previous.shared

from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import VPFPValueEDMConfig, VPFPValueControlEDM, VPFPValueControlNormalizer
from pde_agent_verifier_solver.plugins.vpfp.noise_training import VPFPLowNoiseValueControlEDM
from pde_agent_verifier_solver.plugins.vpfp.temporal_covariance_diffusion import (
    VPFPTemporalCovarianceNormalizer, VPFPTemporalCovarianceEDM, VPFPLowNoiseTemporalCovarianceEDM,
)

ARMS = ("value_base", "value_low", "cov_base", "cov_low")


def factories(arguments, spec, arm):
    if arm not in ARMS:
        raise ValueError("unknown temporal training arm")
    cfg = VPFPValueEDMConfig.from_full_config(shared.model_config(arguments, spec))
    model_type = {"value_base": VPFPValueControlEDM, "value_low": VPFPLowNoiseValueControlEDM,
                  "cov_base": VPFPTemporalCovarianceEDM, "cov_low": VPFPLowNoiseTemporalCovarianceEDM}[arm]
    normalizer_type = VPFPTemporalCovarianceNormalizer if arm.startswith("cov_") else VPFPValueControlNormalizer

    def make_model(config):
        # Same-dimensional inputs and identical parameterized network in all
        # four arms. Coordinate units and sigma draws differ as declared.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(arguments.seed + 100)
            return model_type(config, low_min=.01, low_max=.3) if arm.endswith("_low") else model_type(config)

    def fit(config, full_raw, *, sample_ids):
        if arm.startswith("cov_"):
            return normalizer_type.fit(config, full_raw, sample_ids=sample_ids,
                relative_floor=arguments.covariance_floor, center_kinetic_dc=True, rho_policy="identity")
        return normalizer_type.fit(config, full_raw, sample_ids=sample_ids)
    return cfg, make_model, fit


def preflight(arguments, spec, corpus):
    training = previous.load_split(corpus, "train", verify_hashes=False)
    raw = torch.from_numpy(np.array(training.latents[:8], copy=True))
    conditions = np.array(training.conditions[:arguments.batch_size], copy=True)
    reports = {}
    for arm in ARMS:
        cfg, make, fit = factories(arguments, spec, arm)
        normalizer = fit(cfg, raw, sample_ids=training.sample_ids[:len(raw)].tolist()).to(arguments.device)
        model = make(cfg).to(arguments.device)
        normalized = normalizer.encode(raw[:arguments.batch_size].to(arguments.device))
        reconstructed = normalizer.decode(normalized)
        if not torch.allclose(reconstructed, raw[:arguments.batch_size].to(reconstructed), rtol=2.e-4, atol=2.e-5):
            raise ValueError("temporal preconditioner did not reconstruct the full reference coefficients")
        loss, _ = model.loss(normalized, conditions)
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("temporal pilot preflight loss/gradient failed")
        sampled = normalizer.decode(model.sample(conditions[:1], [arguments.seed + 9], steps=2))
        if sampled.shape != (1, cfg.full_dimension) or not torch.isfinite(sampled).all():
            raise ValueError("temporal pilot preflight did not return a complete finite decoder vector")
        reports[arm] = {"execution_passed": True, "full_dimension": cfg.full_dimension,
            "random_dimension": cfg.latent_dimension, "parameter_count": sum(p.numel() for p in model.parameters()),
            "full_coefficient_roundtrip_max_abs": float((reconstructed - raw[:len(reconstructed)].to(reconstructed)).abs().max()),
            "scientific_validation": False}
    return reports


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "cache", "output", "diagnosis"):
        result.add_argument(f"--{name}", type=Path, required=True)
    result.add_argument("--regime", choices=("weak", "strong"), required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--replicas", type=int, default=2)
    result.add_argument("--train-seconds", type=int, default=360)
    result.add_argument("--steps", type=int, default=100000)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--validate-every", type=int, default=1000)
    result.add_argument("--learning-rate", type=float, default=2.e-4)
    result.add_argument("--width", type=int, default=256)
    result.add_argument("--depth", type=int, default=4)
    result.add_argument("--covariance-floor", type=float, default=.05)
    result.add_argument("--generated-answers", type=int, default=64)
    result.add_argument("--sample-steps", type=int, default=64)
    result.add_argument("--permutations", type=int, default=99)
    result.add_argument("--bootstrap-repeats", type=int, default=49)
    result.add_argument("--constraint-tolerance", type=float, default=1.e-5)
    result.add_argument("--device", default="cuda")
    return result


def main(argv=None):
    cli = parser()
    arguments = cli.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        cli.error("real temporal-geometry training requires a Slurm compute allocation")
    if arguments.output.exists() or arguments.output.is_symlink() or any(
            arguments.output.resolve().is_relative_to(path.resolve()) for path in (arguments.cache, arguments.diagnosis)):
        cli.error("choose a fresh output outside the immutable source experiments")
    if any(getattr(arguments, name) < 1 for name in ("steps", "train_seconds", "batch_size", "validate_every")):
        cli.error("training budgets and batch sizes must be positive")
    if arguments.batch_size > 8 or arguments.replicas < 2 or arguments.generated_answers < 32 or arguments.generated_answers % 8 or arguments.sample_steps < 2:
        cli.error("pilot needs batch<=8, >=2 seeds, balanced >=32 generated cases and >=2 sampling steps")
    if arguments.seed < 0 or not 0 < arguments.covariance_floor <= 1 or any(
        not np.isfinite(getattr(arguments, key)) or getattr(arguments, key) <= 0 for key in ("learning_rate", "covariance_floor", "constraint_tolerance")):
        cli.error("invalid numeric training settings")
    previous.DiffusionAuditConfig(permutations=arguments.permutations, bootstrap_repeats=arguments.bootstrap_repeats).validate()
    payload = json.loads(arguments.config.read_text())
    spec = previous.VPFPOnlineModelSpec.from_mapping(payload.get("vpfp", payload))
    spec.preflight()
    corpus = arguments.cache / "corpus"
    manifest = previous.read_manifest(corpus, verify_hashes=True)
    if spec != previous.VPFPOnlineModelSpec.from_mapping(manifest["model_spec"]) or arguments.regime not in spec.regime_id:
        raise ValueError("model spec/regime differs from the complete frozen source corpus")
    frozen_diagnosis = {str(arguments.diagnosis / name): shared.file_sha(arguments.diagnosis / name)
                        for name in ("result.json", "frozen-plan.json")}
    diagnostic = json.loads((arguments.diagnosis / "result.json").read_text())
    diagnostic_plan = json.loads((arguments.diagnosis / "frozen-plan.json").read_text())
    if (diagnostic.get("execution_passed") is not True or diagnostic.get("regime") != arguments.regime
            or diagnostic_plan.get("corpus_manifest_sha256") != manifest["manifest_sha256"]):
        raise ValueError("a completed mechanism diagnostic on the same corpus is required")
    previous.assert_hashes(frozen_diagnosis)
    frozen_cache = previous.cache_inventory(arguments.cache)
    selected_rows = previous.fixed_reference_rows(corpus, spec, arguments.generated_answers)
    arguments.source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    arguments.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    plan = {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()}
    plan.update(schema="vpfp_temporal_geometry_training_pilot_v1", model_spec=asdict(spec), arms=ARMS,
        cached_feature_sha256=frozen_cache, source_corpus_manifest_sha256=manifest["manifest_sha256"],
        source_file_sha256=shared.source_fingerprints(), script_sha256=shared.file_sha(__file__),
        diagnostic_sha256=frozen_diagnosis[str(arguments.diagnosis / "result.json")],
        diagnostic_file_sha256=frozen_diagnosis, selected_reference_rows=selected_rows,
        objective="base_EDM_only_in_each_declared_noise_coordinate_system",
        normalization="paired_old_value_RMS_or_kinetic_only_train_DC_centered_full_rank_temporal_covariance",
        rho_policy="identity_in_parent_value_normalized_coordinates",
        noise="original_or_half_original_half_log_uniform_0.01_to_0.3",
        initialization="identical_network_parameters_for_all_four_arms_within_seed",
        cross_normalizer_validation_loss_comparable=False, arm_order="forward_then_reverse",
        evidence_role="development_screen_on_previously_opened_reference", new_pde_solves=0,
        scientific_admission=False, auto_large_experiment=False,
        scale_up_policy="manual_evidence_based_promising_configuration_not_full_equivalence_required",
        original_20_percent_scorecard="retained_as_descriptive_historical_heuristic_not_new_scale_up_gate")
    shared.write_json(arguments.output / "frozen-plan.json", plan)
    plan_sha = shared.file_sha(arguments.output / "frozen-plan.json")
    started, stage = time.monotonic(), "source_constraint_check"
    try:
        constraint = previous.audit_value_control_corpus(corpus, batch_size=8, tolerance=arguments.constraint_tolerance)
        shared.write_json(arguments.output / "value-slope-corpus-audit.json", constraint)
        previous.require_value_control_audit(constraint)
        if constraint["source_manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("corpus changed after plan freeze")
        stage = "full_layout_preflight"
        shared.write_json(arguments.output / "allocation-preflight.json", preflight(arguments, spec, corpus))
        checkpoints, frozen, training = {}, {}, {}
        for replica in range(arguments.replicas):
            run_args = copy(arguments)
            run_args.seed = arguments.seed + replica * 10000
            replica_root = arguments.output / f"seed-{replica}"
            replica_root.mkdir()
            for arm in ARMS if replica % 2 == 0 else tuple(reversed(ARMS)):
                key = f"seed-{replica}/{arm}"
                stage = f"train_{key}"
                shared.write_json(arguments.output / "stage.json", {"stage": stage, "status": "running"})
                config, make_model, fit = factories(run_args, spec, arm)
                run_args.ablation_metadata = {"plan_sha256": plan_sha, "arm": arm, "replica": replica,
                    "cross_normalizer_validation_loss_comparable": False, "evidence_role": plan["evidence_role"]}
                arm_root = replica_root / arm
                arm_root.mkdir()
                checkpoint, report = shared.train(run_args, spec, corpus, arm_root,
                    model_config_factory=lambda *a, cfg=config: cfg, model_factory=make_model,
                    normalizer_factory=fit, training_metric_factory=None, validation_metric_factory=None)
                checkpoints[key], training[key], frozen[str(checkpoint)] = checkpoint, report, report["checkpoint_sha256"]
            if len({training[f"seed-{replica}/{arm}"]["initial_model_sha256"] for arm in ARMS}) != 1:
                raise ValueError("four-arm initial network parameter identities differ")
            for prefix in ("value", "cov"):
                if training[f"seed-{replica}/{prefix}_base"]["normalizer_sha256"] != training[f"seed-{replica}/{prefix}_low"]["normalizer_sha256"]:
                    raise ValueError("paired noise coverage arms used different normalization")
        previous.assert_hashes(frozen)
        shared.write_json(arguments.output / "frozen-checkpoints.json", {"all_arms_selected_before_any_feature_evaluation": True, "sha256": frozen})
        previous.assert_hashes(frozen_cache)
        cached, schemas = previous.load_cached_features(arguments.cache, manifest, selected_rows, constraint)
        first = checkpoints["seed-0/value_base"]
        sampler = previous.load_vpfp_diffusion_sampler(first, frozen[str(first)], device=arguments.device, steps=arguments.sample_steps)
        binding = previous.build_vpfp_online_binding(spec, output_root=arguments.output / "audit-runtime", device=arguments.device, latent_sampler=sampler)
        decoder = binding.plugin.generator.backend.runtime.base_model
        summaries = {}
        for replica in range(arguments.replicas):
            run_args = copy(arguments)
            run_args.seed = arguments.seed + replica * 10000
            matrices = {role: {kind: cached[kind][role].features for kind in cached} for role in ("reference_a", "reference_b")}
            reports = {}
            for arm in ARMS:
                key = f"seed-{replica}/{arm}"
                stage = f"frozen_development_audit_{key}"
                shared.write_json(arguments.output / "stage.json", {"stage": stage, "status": "running"})
                matrices[arm], reports[arm] = previous.sample_and_audit(run_args, spec, checkpoints[key], frozen[str(checkpoints[key])],
                    arguments.output / key, decoder, cached, schemas)
                summaries[key] = {kind: {"status": report["status"], "sw_effect": report["comparisons"]["generated"]["statistics"]["sliced_wasserstein_max"]["effect"]}
                                  for kind, report in reports[arm].items()}
            card = previous.build_vpfp_direction_scorecard(regime=arguments.regime, seed=run_args.seed,
                objective="base_EDM_noise_geometry_and_coverage", baseline_arm="value_base", samples=matrices,
                physical_schema=schemas["physical"], derivative_schema=schemas["derivative"], joint_reports=reports,
                physical_time_fractions=previous.VPFPFeaturePlan().time_fractions,
                derivative_time_fractions=previous.VPFPDerivativeFeaturePlan().time_fractions,
                macro_probe_count=previous.VPFPDerivativeFeaturePlan().macro_probe_count,
                phase_probe_count=previous.VPFPDerivativeFeaturePlan().phase_probe_count)
            shared.write_json(arguments.output / f"seed-{replica}/physical-effect-scorecard.json", card)
        previous.assert_hashes(frozen)
        previous.assert_hashes(frozen_cache)
        previous.assert_hashes(frozen_diagnosis)
        if previous.read_manifest(corpus, verify_hashes=True)["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("source corpus manifest changed during the pilot")
        report = {"execution_passed": True, "training": training, "audits": summaries,
            "wall_seconds": time.monotonic() - started, "admitted": False,
            "large_experiment_launched": False, "canonical_campaign_smoke": "separate_required_step",
            "scale_up_decision": "pending_physical_effect_and_mechanism_review_not_automatic_scorecard_boolean",
            "evidence_role": plan["evidence_role"]}
    except Exception as error:
        report = {"execution_passed": False, "failed_stage": stage, "error": repr(error),
            "traceback": traceback.format_exc(), "partial_artifacts_preserved": True, "admitted": False}
    shared.write_json(arguments.output / "result.json", report)
    shared.write_json(arguments.output / "stage.json", {"stage": stage, "status": "complete" if report["execution_passed"] else "failed"})
    shared.emit("temporal_pilot_complete", execution_passed=report["execution_passed"], error=report.get("error"))
    return 0 if report["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
