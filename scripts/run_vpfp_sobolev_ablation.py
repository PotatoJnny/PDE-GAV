"""Matched-budget VPFP EDM/Sobolev experiment with fresh, sealed evaluation.

Only a fixed positive quadratic residual metric changes between arms. The
nonlinear physical derivative distribution is independently evaluated on real
reverse-process samples, never imposed on one-step posterior-mean outputs.
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
loader = importlib.util.spec_from_file_location("vpfp_edm_shared_training", ROOT / "scripts/train_vpfp_structured_diffusion.py")
shared = importlib.util.module_from_spec(loader)
loader.loader.exec_module(shared)

from pde_agent_verifier_solver.plugins.vpfp.canonical_binding import VPFPOnlineModelSpec, build_vpfp_online_binding
from pde_agent_verifier_solver.plugins.vpfp.diffusion_data import read_manifest, load_split, collect_fresh_evaluation_corpus
from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import (
    VPFPCoefficientNormalizer, VPFPStructuredEDM, load_vpfp_diffusion_sampler,
)
from pde_agent_verifier_solver.plugins.vpfp.sobolev_matching import VPFPSobolevMetric
from pde_agent_verifier_solver.plugins.vpfp.derivative_features import (
    VPFPDerivativeFeaturePlan, extract_vpfp_derivative_features,
)
from pde_agent_verifier_solver.plugins.vpfp.diffusion_audit import (
    AuditFeature, AuditSample, DiffusionAuditConfig, audit_diffusion_distribution, audit_selfcheck,
)


def check_source(spec, manifest):
    if spec != VPFPOnlineModelSpec.from_mapping(manifest["model_spec"]):
        raise ValueError("ablation requires the complete frozen source model specification")
    if spec.base_decoder_sha256 != manifest["base_decoder_sha256"]:
        raise ValueError("ablation source corpus has a different decoder")
    for name in ("collision_nu_range", "temperature_range", "time_domain"):
        if tuple(manifest[name]) != tuple(getattr(spec, name)):
            raise ValueError(f"ablation source corpus has different {name}")
    counts = manifest["split_counts"]
    if len({counts[name] for name in ("validation", "calibration", "reference_a", "reference_b")}) != 1:
        raise ValueError("this paired pilot requires equal validation/calibration/reference counts")
    if counts["reference_a"] < 32 or counts["reference_a"] % 8:
        raise ValueError("paired audit counts must be multiples of eight and >=32")


def objective_factory(arguments):
    return lambda config, normalizer: VPFPSobolevMetric(config, normalizer,
        spatial_weight=arguments.spatial_weight, temporal_weight=arguments.temporal_weight,
        spatial_lengths=arguments.spatial_lengths, latent_input_scale=arguments.latent_input_scale,
        structured_local_scale=arguments.structured_local_scale).to(arguments.device)


def assert_frozen_checkpoints(checkpoints, frozen):
    pass


def metric_preflight(arguments, spec):
    """Exercise full-size gradients on training-only rows before fresh solves."""
    split = load_split(arguments.source_corpus, "train", verify_hashes=False)
    count = min(len(split), max(8, arguments.batch_size))
    raw = torch.from_numpy(np.array(split.latents[:count], copy=True))
    config = shared.model_config(arguments, spec)
    normalizer = VPFPCoefficientNormalizer.fit(config, raw, sample_ids=split.sample_ids[:count].tolist()).to(arguments.device)
    metric = objective_factory(arguments)(config, normalizer)
    model = VPFPStructuredEDM(config).to(arguments.device)
    raw = raw[:arguments.batch_size].to(arguments.device)
    conditions = np.array(split.conditions[:len(raw)], copy=True)
    started = time.monotonic()
    loss, _ = model.loss(normalizer.encode(raw), conditions, residual_metric=metric)
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError("Sobolev full-layout gradient preflight failed")
    if str(arguments.device).startswith("cuda"):
        torch.cuda.synchronize()
    return {"execution_passed": True, "data_role": "training_rows_only_mechanics_preflight",
        "normalizer_for_preflight_only": True, "scientific_qualification": False,
        "loss": float(loss.detach()), "wall_seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if str(arguments.device).startswith("cuda") else None,
        "metric": metric.report()}


def write_derivative_sample(directory, role, decoder, latents, conditions, ids, plan):
    matrix, descriptors, physical = extract_vpfp_derivative_features(decoder, latents, conditions, plan=plan)
    hashes = tuple(row["input_answer_sha256"] for row in physical["per_case"])
    evaluated_conditions = np.asarray([row["input_equation_condition"] for row in physical["per_case"]], dtype=np.float32)
    np.save(directory / f"{role}.npy", matrix, allow_pickle=False)
    shared.write_json(directory / f"{role}.json", {"root_ids": ids, "answer_sha256": hashes,
        "conditions": evaluated_conditions.tolist(), "physical": physical,
        "feature_sha256": shared.file_sha(directory / f"{role}.npy")})
    return AuditSample(matrix, evaluated_conditions, ids, answer_sha256=hashes), descriptors


def derivative_audits(arguments, spec, corpus, checkpoints, output):
    """Shared reference features; each arm retains its own generated trajectories."""
    directory = output / "derivative-features"
    directory.mkdir()
    plan = VPFPDerivativeFeaturePlan()
    sampler = load_vpfp_diffusion_sampler(checkpoints["baseline"], shared.file_sha(checkpoints["baseline"]),
                                         device=arguments.device, steps=arguments.sample_steps)
    manifest = read_manifest(corpus, verify_hashes=True)
    shared.assert_checkpoint_corpus(sampler, manifest, load_split(corpus, "train", verify_hashes=False).sample_ids)
    binding = build_vpfp_online_binding(spec, output_root=output / "derivative-runtime", device=arguments.device, latent_sampler=sampler)
    decoder = binding.plugin.generator.backend.runtime.base_model
    samples, descriptions = {}, None
    for role in ("train", "validation", "calibration", "reference_a", "reference_b"):
        split = load_split(corpus, role, verify_hashes=False)
        ids = tuple(f"equation:{int(seed)}" for seed in split.sample_ids)
        shared.emit("physical_derivatives", split=role, count=len(split))
        sample, descriptor = write_derivative_sample(directory, role, decoder, split.latents, split.conditions, ids, plan)
        if descriptions is not None and descriptions != descriptor:
            raise RuntimeError("physical derivative schema changed between reference splits")
        samples[role], descriptions = sample, descriptor
    features = tuple(AuditFeature(**entry) for entry in descriptions)
    shared.write_json(directory / "schema.json", {"features": descriptions, "plan": asdict(plan)})
    config = DiffusionAuditConfig(permutations=arguments.permutations, bootstrap_repeats=arguments.bootstrap_repeats,
                                  seed=arguments.seed + 1400)
    nu_edges, temperature_edges = np.linspace(*spec.collision_nu_range, 5), np.linspace(*spec.temperature_range, 3)
    shared.write_json(output / "derivative-audit-selfcheck.json", audit_selfcheck(config=config, features=features,
        calibration_bank=samples["calibration"], nu_edges=nu_edges, temperature_edges=temperature_edges,
        trials=arguments.selfcheck_trials, cases_per_stratum=arguments.heldout_answers // 8))
    results = {}
    for arm, checkpoint in checkpoints.items():
        arm_root = output / arm
        provenance = json.loads((arm_root / "generated-provenance.json").read_text())
        raw = np.load(arm_root / "generated-coefficients.npy", mmap_mode="r", allow_pickle=False)
        conditions = np.load(arm_root / "generated-conditions.npy", allow_pickle=False)
        digest = shared.file_sha(checkpoint)
        if provenance["checkpoint_sha256"] != digest:
            raise ValueError("generated derivative sample belongs to another checkpoint")
        ids = tuple(f"generated:{digest}:{int(seed)}" for seed in provenance["noise_seeds"])
        sample, descriptor = write_derivative_sample(directory, f"generated-{arm}", decoder, raw, conditions, ids, plan)
        if descriptor != descriptions:
            raise RuntimeError("generated and reference physical derivative schemas differ")
        # Same input snapshot identities as the original physical-feature audit.
        old_hashes = json.loads((arm_root / "physical-features/generated.json").read_text())["answer_sha256"]
        if tuple(old_hashes) != tuple(sample.answer_sha256):
            raise ValueError("derivatives and original physical features used different answers")
        report = audit_diffusion_distribution({**samples, "generated": sample}, features=features,
            nu_edges=nu_edges, temperature_edges=temperature_edges, config=config)
        report.update(checkpoint_sha256=digest, corpus_manifest_sha256=manifest["manifest_sha256"],
            feature_plan=asdict(plan), used_for_checkpoint_selection=False,
            physical_derivative_distribution_in_training_loss=False,
            training_objective="fixed_linear_sobolev_residual_metric" if arm == "sobolev" else "original_edm",
            independent_spatial_quadrature_convergence=False, admitted=False)
        shared.write_json(arm_root / "derivative-distribution-audit.json", report)
        results[arm] = {"status": report["status"], "joint_pvalue": report["comparisons"]["generated"]["joint_pvalue"],
                       "reference_null_joint_pvalue": report["comparisons"]["reference_null"]["joint_pvalue"], "admitted": False}
    read_manifest(corpus, verify_hashes=True)
    return results


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "source-corpus", "output"):
        result.add_argument(f"--{name}", type=Path, required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--train-seconds", type=int, default=1200)
    result.add_argument("--steps", type=int, default=100000)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--collect-batch-size", type=int, default=8)
    result.add_argument("--validate-every", type=int, default=1000)
    result.add_argument("--learning-rate", type=float, default=2.e-4)
    result.add_argument("--width", type=int, default=256)
    result.add_argument("--depth", type=int, default=4)
    result.add_argument("--spatial-weight", type=float, default=.25)
    result.add_argument("--temporal-weight", type=float, default=.25)
    result.add_argument("--sample-steps", type=int, default=64)
    result.add_argument("--permutations", type=int, default=99)
    result.add_argument("--bootstrap-repeats", type=int, default=99)
    result.add_argument("--selfcheck-trials", type=int, default=20)
    result.add_argument("--device", default="cuda")
    return result


def main(argv=None):
    cli = parser()
    arguments = cli.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        cli.error("real VPFP Sobolev ablation requires a Slurm compute allocation")
    if arguments.output.exists() or arguments.output.is_symlink() or arguments.output.resolve().is_relative_to(arguments.source_corpus.resolve()):
        cli.error("choose a fresh output outside the frozen source corpus")
    for name in ("train_seconds", "steps", "batch_size", "collect_batch_size", "validate_every", "selfcheck_trials"):
        if getattr(arguments, name) < 1:
            cli.error(f"{name} must be positive")
    if arguments.seed < 0 or arguments.sample_steps < 2:
        cli.error("invalid seed or sampling step count")
    for name in ("spatial_weight", "temporal_weight", "learning_rate"):
        value = getattr(arguments, name)
        if not np.isfinite(value) or value < 0 or (name == "learning_rate" and value == 0):
            cli.error(f"invalid {name}")
    if arguments.spatial_weight + arguments.temporal_weight == 0:
        cli.error("the Sobolev comparison arm must have a nonzero declared derivative weight")
    DiffusionAuditConfig(permutations=arguments.permutations, bootstrap_repeats=arguments.bootstrap_repeats).validate()
    payload = json.loads(arguments.config.read_text())
    spec = VPFPOnlineModelSpec.from_mapping(payload.get("vpfp", payload))
    spec.preflight()
    source = read_manifest(arguments.source_corpus, verify_hashes=True)
    check_source(spec, source)
    decoder_config = source["provenance"]["decoder_config"]
    arguments.spatial_lengths = tuple(decoder_config["equation"]["x_domain"])
    arguments.latent_input_scale = float(decoder_config["latent_input_scale"])
    arguments.structured_local_scale = float(decoder_config["structured_local_scale"])
    config = shared.model_config(arguments, spec)
    arguments.heldout_answers = source["split_counts"]["reference_a"]
    arguments.source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    arguments.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    plan = {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()}
    plan.update(schema="vpfp_score_compatible_sobolev_ablation_v1", model_spec=asdict(spec), model_config=config.to_dict(),
        source_corpus_manifest_sha256=source["manifest_sha256"], source_file_sha256=shared.source_fingerprints(),
        ablation_script_sha256=shared.file_sha(__file__), derivative_feature_plan=asdict(VPFPDerivativeFeaturePlan()),
        comparison="same_initialization_training_data_noise_streams_wall_time_budget_common_validation_metric",
        update_counts_may_differ=True, arm_order=["baseline", "sobolev"],
        physical_derivative_distribution_loss=False, nonlinear_derivatives_used_for_training=False,
        full_reverse_process_backpropagation=False, new_reference_required=True,
        generated_conditions_and_noise_matched_across_arms=True, admitted=False)
    shared.write_json(arguments.output / "frozen-plan.json", plan)
    arguments.ablation_metadata = {"plan_sha256": shared.file_sha(arguments.output / "frozen-plan.json"),
                                  "comparison": plan["comparison"]}
    stage, started = "metric_preflight", time.monotonic()
    try:
        shared.write_json(arguments.output / "metric-preflight.json", metric_preflight(arguments, spec))
        stage = "fresh_evaluation_collection"
        shared.write_json(arguments.output / "stage.json", {"stage": stage, "status": "running"})
        corpus = arguments.output / "corpus"
        collect_fresh_evaluation_corpus(arguments.source_corpus, corpus, arguments.seed,
            batch_size=arguments.collect_batch_size, device=arguments.device)
        fresh = read_manifest(corpus, verify_hashes=True)
        if fresh["source_corpus"]["manifest_sha256"] != source["manifest_sha256"]:
            raise ValueError("source corpus changed between plan freeze and fresh collection")
        checkpoints, training = {}, {}
        factory = objective_factory(arguments)
        for arm in ("baseline", "sobolev"):
            stage = f"train_{arm}"
            shared.write_json(arguments.output / "stage.json", {"stage": stage, "status": "running"})
            arm_root = arguments.output / arm
            arm_root.mkdir()
            arm_arguments = copy(arguments)
            arm_arguments.ablation_metadata = {**arguments.ablation_metadata, "arm": arm}
            checkpoints[arm], training[arm] = shared.train(arm_arguments, spec, corpus, arm_root,
                training_metric_factory=factory if arm == "sobolev" else None, validation_metric_factory=factory)
        for key in ("initial_model_sha256", "normalizer_sha256", "common_validation_metric"):
            if training["baseline"][key] != training["sobolev"][key]:
                raise ValueError(f"paired training did not share {key}")
        frozen = {arm: {"path": str(path), "sha256": training[arm]["checkpoint_sha256"]} for arm, path in checkpoints.items()}
        assert_frozen_checkpoints(checkpoints, frozen)
        shared.write_json(arguments.output / "frozen-checkpoints.json", {
            "both_arms_frozen_before_any_reference_features": True,
            "checkpoints": frozen})
        audits = {}
        for arm, checkpoint in checkpoints.items():
            stage = f"frozen_distribution_audit_{arm}"
            shared.write_json(arguments.output / "stage.json", {"stage": stage, "status": "running"})
            assert_frozen_checkpoints(checkpoints, frozen)
            audits[arm] = shared.frozen_audit(arguments, spec, corpus, checkpoint, arguments.output / arm,
                expected_checkpoint_sha256=frozen[arm]["sha256"])
        stage = "frozen_physical_derivative_audits"
        shared.write_json(arguments.output / "stage.json", {"stage": stage, "status": "running"})
        derivatives = derivative_audits(arguments, spec, corpus, checkpoints, arguments.output)
        assert_frozen_checkpoints(checkpoints, frozen)
        report = {"execution_passed": True, "training": training, "distribution_audits": audits,
            "derivative_distribution_audits": derivatives, "admitted": False,
            "wall_seconds": time.monotonic() - started, "canonical_campaign_smoke": "separate_required_step"}
    except Exception as error:
        report = {"execution_passed": False, "failed_stage": stage, "error": repr(error),
                  "traceback": traceback.format_exc(), "partial_artifacts_preserved": True, "admitted": False}
    shared.write_json(arguments.output / "result.json", report)
    shared.write_json(arguments.output / "stage.json", {"stage": stage, "status": "complete" if report["execution_passed"] else "failed"})
    shared.emit("finished", **report)
    return 0 if report["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
