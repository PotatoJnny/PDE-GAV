"""First strong-collision VPFP main-training stage on a new 2048-answer corpus.

Reuse existing EDM training, native sampler assets and physical audits. Two
seeds compare value_base/cov_base from scratch; all five numerical-answer
splits exclude every source/historical equation ID. Models and time snapshots
freeze before reference features open. Assets remain unqualified candidates
for the existing Campaign flow, not automatically admitted production models.
"""
from __future__ import annotations

import argparse
from copy import copy
from dataclasses import asdict
import importlib.util
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
loader = importlib.util.spec_from_file_location("vpfp_strong_main_scaleup_reuse", ROOT / "scripts/run_vpfp_scaleup_experiment.py")
scaleup = importlib.util.module_from_spec(loader)
loader.loader.exec_module(scaleup)
shared, previous, temporal = scaleup.shared, scaleup.previous, scaleup.temporal
from pde_agent_verifier_solver.plugins.vpfp.diffusion_data import collect_independent_corpus

ARMS = ("value_base", "cov_base")
COUNTS = {"train": 2048, "validation": 128, "calibration": 128, "reference_a": 128, "reference_b": 128}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "source-corpus", "output"):
        result.add_argument(f"--{name}", type=Path, required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--regime", choices=("strong",), default="strong")
    for name, default in (("train-answers", 2048), ("heldout-answers", 128), ("replicas", 2),
            ("train-seconds", 7200), ("steps", 1000000), ("batch-size", 4), ("collect-batch-size", 8),
            ("validate-every", 1000), ("width", 256), ("depth", 4), ("generated-answers", 128),
            ("sample-steps", 64), ("permutations", 99), ("bootstrap-repeats", 49)):
        result.add_argument(f"--{name}", type=int, default=default)
    for name, default in (("learning-rate", 2.e-4), ("covariance-floor", .05), ("constraint-tolerance", 1.e-5)):
        result.add_argument(f"--{name}", type=float, default=default)
    result.add_argument("--checkpoint-seconds", nargs="+", type=float, default=(1800., 3600.))
    result.add_argument("--device", default="cuda")
    return result


def train_paired(arguments, spec, corpus, plan_sha, stage, guard):
    """Same factories and initialization pairing as the completed scale-up."""
    training, models = {}, {}
    for replica in range(2):
        run_args = copy(arguments)
        run_args.seed = arguments.seed + replica * 10000
        replica_root = arguments.output / f"seed-{replica}"
        replica_root.mkdir()
        for arm in ARMS if replica == 0 else tuple(reversed(ARMS)):
            key = f"seed-{replica}/{arm}"
            stage("train_" + key)
            guard()
            config, make_model, fit = temporal.factories(run_args, spec, arm)
            run_args.ablation_metadata = {"plan_sha256": plan_sha, "arm": arm, "replica": replica,
                "stage": "strong_main_training_stage_1", "evidence_role": scaleup.EVIDENCE_ROLE,
                "cross_normalizer_validation_loss_comparable": False}
            directory = replica_root / arm
            directory.mkdir()
            checkpoint, report = shared.train(run_args, spec, corpus, directory,
                model_config_factory=lambda *a, cfg=config: cfg, model_factory=make_model,
                normalizer_factory=fit, training_metric_factory=None, validation_metric_factory=None)
            training[key] = report
            models.update(scaleup.register_models(key, replica, arm, directory, checkpoint, report, arguments.checkpoint_seconds))
            shared.write_json(arguments.output / "training-progress.json", {"training": training, "models": models,
                "status": "not_all_models_frozen", "physical_features_opened": False})
        if len({training[f"seed-{replica}/{arm}"]["initial_model_sha256"] for arm in ARMS}) != 1:
            raise ValueError("paired initial network parameter identities differ")
    for arm in ARMS:
        if training[f"seed-0/{arm}"]["normalizer_sha256"] != training[f"seed-1/{arm}"]["normalizer_sha256"]:
            raise ValueError("same-representation training-only normalizers differ between seeds")
    frozen = {entry["checkpoint"]: entry["sha256"] for entry in models.values()}
    guard()
    previous.assert_hashes(frozen)
    shared.write_json(arguments.output / "frozen-checkpoints.json", {
        "all_arms_and_snapshots_frozen_before_any_feature_evaluation": True, "sha256": frozen,
        "models": models, "plan_sha256": plan_sha, "time_snapshots_used_for_checkpoint_selection": False})
    shared.write_json(arguments.output / "training-progress.json", {"training": training, "models": models,
        "status": "all_models_frozen", "physical_features_opened": False})
    return training, models, frozen


def evaluate_frozen(arguments, spec, corpus, manifest, models, stage, guard):
    """Reuse the tested full-decoder-D sampler and both existing feature audits."""
    stage("frozen_reference_features")
    if previous.read_manifest(corpus, verify_hashes=True)["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("new corpus changed after all-model freeze")
    first = models["seed-0/value_base/best"]
    sampler = previous.load_vpfp_diffusion_sampler(first["checkpoint"], first["sha256"],
        device=arguments.device, steps=arguments.sample_steps)
    shared.assert_checkpoint_corpus(sampler, manifest, previous.load_split(corpus, "train", verify_hashes=False).sample_ids)
    binding = previous.build_vpfp_online_binding(spec, output_root=arguments.output / "audit-runtime",
        device=arguments.device, latent_sampler=sampler)
    decoder = binding.plugin.generator.backend.runtime.base_model
    samples, schemas, feature_hashes = scaleup.write_reference_features(arguments, corpus, decoder, manifest)
    summaries, best_matrices, best_reports = {}, {}, {}
    for key, entry in models.items():
        stage("frozen_audit_" + key)
        guard()
        previous.assert_hashes(feature_hashes)
        run_args = copy(arguments)
        run_args.seed = arguments.seed + entry["replica"] * 10000
        run_args.cache = arguments.output  # Reused audit binds the NEW corpus and complete decoder ABI.
        directory = arguments.output / f"seed-{entry['replica']}" / entry["arm"] / "evaluation" / key.rsplit("/", 1)[1]
        directory.mkdir(parents=True)
        matrices, reports = scaleup.audit_frozen_model(run_args, spec, entry, directory, decoder, samples, schemas)
        summaries[key] = {"checkpoint": entry, "evaluation_directory": str(directory), "audit": {
            kind: {"status": report["status"], "sw_effect": report["comparisons"]["generated"]["statistics"]["sliced_wasserstein_max"]["effect"]}
            for kind, report in reports.items()}}
        shared.write_json(arguments.output / "audit-progress.json", {"completed_audits": summaries,
            "evidence_role": scaleup.EVIDENCE_ROLE, "admitted": False})
        if entry["checkpoint_role"] == "training_validation_selected_best":
            best_matrices.setdefault(entry["replica"], {})[entry["arm"]] = matrices
            best_reports.setdefault(entry["replica"], {})[entry["arm"]] = reports
    stage("paired_best_physical_effect_scorecards")
    for replica in range(2):
        matrices = {role: {kind: samples[kind][role].features for kind in samples} for role in ("reference_a", "reference_b")}
        matrices.update(best_matrices[replica])
        card = previous.build_vpfp_direction_scorecard(regime="strong", seed=arguments.seed + replica * 10000,
            objective="strong_main_stage_1_base_EDM_noise_geometry", baseline_arm="value_base", samples=matrices,
            physical_schema=schemas["physical"], derivative_schema=schemas["derivative"], joint_reports=best_reports[replica],
            physical_time_fractions=previous.VPFPFeaturePlan().time_fractions,
            derivative_time_fractions=previous.VPFPDerivativeFeaturePlan().time_fractions,
            macro_probe_count=previous.VPFPDerivativeFeaturePlan().macro_probe_count,
            phase_probe_count=previous.VPFPDerivativeFeaturePlan().phase_probe_count)
        card.update(evidence_role=scaleup.EVIDENCE_ROLE, used_for_checkpoint_selection=False,
            automatic_scaleup_or_admission=False, heuristic_boolean_used_as_gate=False)
        shared.write_json(arguments.output / f"seed-{replica}/physical-effect-scorecard.json", card)
    previous.assert_hashes(feature_hashes)
    return summaries


def main(argv=None):
    cli = parser()
    arguments = cli.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        cli.error("real strong VPFP main training requires a Slurm compute allocation")
    if (arguments.output.exists() or arguments.output.is_symlink()
            or arguments.output.resolve().is_relative_to(arguments.source_corpus.resolve())):
        cli.error("choose a fresh output outside the immutable source corpus")
    if (arguments.train_answers != 2048 or arguments.heldout_answers != 128
            or arguments.generated_answers != 128 or arguments.replicas != 2):
        cli.error("stage 1 fixes train2048, validation/evaluation/generated128 and two replicas")
    if any(getattr(arguments, name) < 1 for name in ("steps", "train_seconds", "batch_size", "collect_batch_size", "validate_every", "width", "depth")):
        cli.error("training budgets, dimensions and batch sizes must be positive")
    if arguments.batch_size > 8 or arguments.collect_batch_size > 8 or 128 % arguments.collect_batch_size or arguments.sample_steps < 2:
        cli.error("batch sizes must be <=8, collection batch must divide 128, and Heun needs >=2 steps")
    if arguments.seed < 0 or arguments.covariance_floor > 1 or any(
            not np.isfinite(getattr(arguments, name)) or getattr(arguments, name) <= 0
            for name in ("learning_rate", "covariance_floor", "constraint_tolerance")):
        cli.error("invalid seed or finite positive numeric setting")
    try:
        arguments.checkpoint_seconds = shared._checkpoint_seconds(arguments)
    except ValueError as error:
        cli.error(str(error))
    if max(arguments.checkpoint_seconds) > arguments.train_seconds:
        cli.error("snapshot thresholds cannot exceed the per-arm training wall budget")
    previous.DiffusionAuditConfig(permutations=arguments.permutations, bootstrap_repeats=arguments.bootstrap_repeats).validate()
    config = scaleup.read_json(arguments.config)
    spec = previous.VPFPOnlineModelSpec.from_mapping(config.get("vpfp", config))
    spec.preflight()
    source = previous.read_manifest(arguments.source_corpus, verify_hashes=True)
    if spec != previous.VPFPOnlineModelSpec.from_mapping(source["model_spec"]) or "strong" not in spec.regime_id:
        raise ValueError("main stage requires the exact frozen strong-collision source specification")
    frozen_inputs = {str(path): shared.file_sha(path) for path in (
        arguments.config, arguments.source_corpus / "manifest.json", arguments.source_corpus / source["seed_plan"]["path"])}
    fingerprints = shared.source_fingerprints()
    for filename in ("run_vpfp_scaleup_experiment.py", "run_vpfp_temporal_training_pilot.py",
                     "run_vpfp_value_control_pilot.py", Path(__file__).name):
        fingerprints[f"scripts/{filename}"] = shared.file_sha(ROOT / "scripts" / filename)
    frozen_code = {str(ROOT / name): digest for name, digest in fingerprints.items()}
    arguments.source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    arguments.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    plan = {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()}
    plan.update(schema="vpfp_strong_main_training_stage_1_v1", model_spec=asdict(spec), arms=ARMS,
        split_counts=COUNTS, source_corpus_manifest_sha256=source["manifest_sha256"],
        frozen_input_sha256=frozen_inputs, source_file_sha256=fingerprints, collection_seed=arguments.seed + 5000000,
        numerical_answers=2560, reused_answer_splits=[], history_policy="exclude_source_all_splits_plus_embedded_cumulative_history",
        training_start="from_scratch_complete_model_no_resume", normalization_fit="each_arm_only_new_2048_training_answers",
        initialization="identical_within_seed_different_between_seeds", arm_order="forward_then_reverse",
        best_selection="training_validation_only_within_arm", snapshot_selection="fixed_elapsed_time_never_best_selection",
        objective="base_EDM_in_declared_full_rank_noise_coordinates_no_explicit_physical_derivative_distribution_loss",
        macro_loss_fraction_meaning="rho_coefficient_MSE_share_not_jx_jy_or_kinetic_energy_supervision",
        covariance_floor=arguments.covariance_floor, covariance_rho_policy="identity", covariance_center="training_kinetic_DC_mean",
        nominal_total_training_seconds=4 * arguments.train_seconds,
        planned_generated_answers=4 * (1 + len(arguments.checkpoint_seconds)) * arguments.generated_answers,
        generated_seed_pairing="same_conditions_and_noise_within_replica_across_models_and_timepoints",
        previous_audits_read_for_model_selection=False, evidence_role=scaleup.EVIDENCE_ROLE,
        cross_normalizer_validation_loss_comparable=False, blind_test=False,
        multiple_models_and_timepoints_adjusted=False, admitted=False,
        canonical_campaign_smoke="separate_required_step_using_existing_native_asset_and_Campaign_flow",
        automatic_followup_job_submitted=False)
    shared.write_json(arguments.output / "frozen-plan.json", plan)
    plan_sha = shared.file_sha(arguments.output / "frozen-plan.json")
    frozen_inputs[str(arguments.output / "frozen-plan.json")] = plan_sha
    started, current_stage = time.monotonic(), "source_training_rows_preflight"
    training, models, summaries, frozen_models = {}, {}, {}, {}

    def stage(name):
        nonlocal current_stage
        current_stage = name
        shared.write_json(arguments.output / "stage.json", {"stage": name, "status": "running"})

    def guard():
        previous.assert_hashes({**frozen_inputs, **frozen_code, **frozen_models})

    try:
        stage(current_stage)
        guard()
        shared.write_json(arguments.output / "allocation-preflight.json", temporal.preflight(arguments, spec, arguments.source_corpus))
        stage("collect_new_independent_five_splits")
        guard()
        corpus = arguments.output / "corpus"
        collect_independent_corpus(arguments.source_corpus, corpus, COUNTS, plan["collection_seed"],
            batch_size=arguments.collect_batch_size, device=arguments.device)
        manifest = previous.read_manifest(corpus, verify_hashes=True)
        if (manifest["split_counts"] != COUNTS or manifest["model_spec"] != source["model_spec"]
                or manifest["source_corpus"]["manifest_sha256"] != source["manifest_sha256"]
                or manifest["source_corpus"]["reused_splits"] != []
                or manifest["source_corpus"]["regenerated_splits"] != list(previous.ROLES)):
            raise ValueError("new corpus does not bind five independent splits to the frozen source history")
        frozen_inputs[str(corpus / "manifest.json")] = shared.file_sha(corpus / "manifest.json")
        shared.write_json(arguments.output / "new-corpus-receipt.json", {"plan_sha256": plan_sha,
            "corpus_manifest_sha256": manifest["manifest_sha256"], "source_corpus": manifest["source_corpus"],
            "split_counts": COUNTS, "new_pde_solves": 2560, "physical_features_opened": False})
        stage("new_corpus_encoding_check")
        # Source-encoding validity, not physical feature comparison or model
        # selection. Fail before eight hours of training on malformed targets.
        constraint = previous.audit_value_control_corpus(corpus, batch_size=8, tolerance=arguments.constraint_tolerance)
        shared.write_json(arguments.output / "value-slope-corpus-audit.json", constraint)
        previous.require_value_control_audit(constraint)
        if constraint["source_manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("new corpus changed before training")
        training, models, frozen_models = train_paired(arguments, spec, corpus, plan_sha, stage, guard)
        guard()
        summaries = evaluate_frozen(arguments, spec, corpus, manifest, models, stage, guard)
        stage("final_immutability_checks")
        guard()
        if (previous.read_manifest(corpus, verify_hashes=True)["manifest_sha256"] != manifest["manifest_sha256"]
                or previous.read_manifest(arguments.source_corpus, verify_hashes=True)["manifest_sha256"] != source["manifest_sha256"]):
            raise ValueError("source or new main-training corpus changed during the experiment")
        report = {"execution_passed": True, "regime": "strong", "training": training, "audits": summaries,
            "new_pde_solves": 2560, "generated_answers": len(models) * arguments.generated_answers,
            "all_learning_curve_thresholds_reached": all(not value["unreached_checkpoint_seconds"] for value in training.values()),
            "wall_seconds": time.monotonic() - started, "evidence_role": scaleup.EVIDENCE_ROLE,
            "admitted": False, "automatic_followup_job_submitted": False,
            "canonical_campaign_smoke": plan["canonical_campaign_smoke"]}
    except Exception as error:
        report = {"execution_passed": False, "regime": "strong", "failed_stage": current_stage,
            "error": repr(error), "traceback": traceback.format_exc(), "partial_artifacts_preserved": True,
            "completed_training_receipt": (str(arguments.output / "training-progress.json")
                if (arguments.output / "training-progress.json").exists() else None),
            "completed_audit_receipt": (str(arguments.output / "audit-progress.json")
                if (arguments.output / "audit-progress.json").exists() else None),
            "admitted": False, "automatic_followup_job_submitted": False}
    shared.write_json(arguments.output / "result.json", report)
    shared.write_json(arguments.output / "stage.json", {"stage": current_stage, "status": "complete" if report["execution_passed"] else "failed"})
    shared.emit("strong_main_stage_complete", execution_passed=report["execution_passed"], error=report.get("error"))
    return 0 if report["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
