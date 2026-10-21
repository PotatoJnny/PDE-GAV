"""Paired VPFP scale-up of a manually frozen pilot candidate, without admission.

Two training seeds compare value_base with one declared recipe. Training
and validation bytes are reused; 384 newly screened evaluation answers are
collected. Every selected model and elapsed-time EMA snapshot freezes before
physical/derivative features or generated trajectories are evaluated. The same
fresh references are reused across models and predeclared learning-curve times:
this is not a blind, multiplicity-adjusted confirmation or an admission gate.
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
loader = importlib.util.spec_from_file_location("vpfp_scaleup_temporal_reuse", ROOT / "scripts/run_vpfp_temporal_training_pilot.py")
temporal = importlib.util.module_from_spec(loader)
loader.loader.exec_module(temporal)
previous, shared = temporal.previous, temporal.shared

from pde_agent_verifier_solver.plugins.vpfp.diffusion_data import collect_fresh_evaluation_corpus

CANDIDATES = ("value_low", "cov_base", "cov_low")
EVIDENCE_ROLE = "fresh_reference_paired_comparison_and_predeclared_learning_curves_not_blind_confirmation"
COUNTS = {"train": 512, "validation": 128, "calibration": 128, "reference_a": 128, "reference_b": 128}
SELECTION_FIELDS = {"schema", "regime", "candidate", "pilot_result_sha256", "pilot_plan_sha256", "evidence", "scientific_admission"}


def read_json(path):
    value = json.loads(Path(path).read_text())
    json.dumps(value, allow_nan=False)
    return value


def freeze_inputs(arguments, spec, manifest):
    """Check the manual choice before creating output or collecting answers."""
    paths = (arguments.config, arguments.selection, arguments.pilot / "result.json",
             arguments.pilot / "frozen-plan.json", arguments.cache / "corpus/manifest.json")
    frozen = {str(path): shared.file_sha(path) for path in paths}
    selection = read_json(arguments.selection)
    pilot = read_json(arguments.pilot / "result.json")
    pilot_plan = read_json(arguments.pilot / "frozen-plan.json")
    if (not isinstance(selection, dict) or set(selection) != SELECTION_FIELDS
            or selection.get("schema") != "vpfp_scaleup_candidate_selection_v1"
            or selection.get("regime") != arguments.regime or selection.get("candidate") != arguments.candidate
            or selection.get("scientific_admission") is not False
            or not isinstance(selection.get("evidence"), (str, dict)) or not selection["evidence"]
            or selection.get("pilot_result_sha256") != frozen[str(arguments.pilot / "result.json")]
            or selection.get("pilot_plan_sha256") != frozen[str(arguments.pilot / "frozen-plan.json")]):
        raise ValueError("manual candidate selection must match its strict schema, regime, candidate and pilot SHAs")
    expected_training = {f"seed-{replica}/{arm}" for replica in range(2) for arm in temporal.ARMS}
    if (pilot.get("execution_passed") is not True or pilot.get("admitted") is not False
            or not expected_training.issubset(pilot.get("training", {}))
            or pilot_plan.get("schema") != "vpfp_temporal_geometry_training_pilot_v1"
            or pilot_plan.get("regime") != arguments.regime
            or pilot_plan.get("source_corpus_manifest_sha256") != manifest["manifest_sha256"]):
        raise ValueError("a completed two-seed temporal pilot on this source corpus is required")
    if (spec != previous.VPFPOnlineModelSpec.from_mapping(manifest["model_spec"])
            or arguments.regime not in spec.regime_id or manifest["split_counts"] != COUNTS):
        raise ValueError("scale-up must preserve the source specification and train512/validation128/evaluation128 counts")
    previous.assert_hashes(frozen)
    return frozen, selection


def write_reference_features(arguments, corpus, decoder, manifest):
    """Extract each frozen numerical answer once per existing feature plan."""
    samples, schemas, frozen = {}, {}, {}
    for kind, extractor, plan in (
            ("physical", previous.extract_vpfp_features, previous.VPFPFeaturePlan()),
            ("derivative", previous.extract_vpfp_derivative_features, previous.VPFPDerivativeFeaturePlan())):
        directory = arguments.output / "reference-features" / kind
        directory.mkdir(parents=True)
        samples[kind], schema = {}, None
        for role in previous.ROLES:
            split = previous.load_split(corpus, role, verify_hashes=False)
            ids = tuple(f"equation:{int(seed)}" for seed in split.sample_ids)
            shared.emit("scaleup_reference_features", kind=kind, split=role, count=len(split))
            matrix, descriptors, physical = extractor(decoder, split.latents, split.conditions, plan=plan)
            descriptors = tuple(descriptors)
            if schema is not None and schema != descriptors:
                raise ValueError("reference feature schema changed across corpus splits")
            schema = descriptors
            hashes = tuple(row["input_answer_sha256"] for row in physical["per_case"])
            evaluated = np.asarray([row["input_equation_condition"] for row in physical["per_case"]], dtype=np.float32)
            if not np.array_equal(evaluated, split.conditions):
                raise ValueError("reference feature extraction changed equation conditions")
            if kind == "derivative" and hashes != samples["physical"][role].answer_sha256:
                raise ValueError("physical/derivative reference extraction used different answer snapshots")
            samples[kind][role] = previous.AuditSample(matrix, evaluated, ids, answer_sha256=hashes)
            np.save(directory / f"{role}.npy", matrix, allow_pickle=False)
            shared.write_json(directory / f"{role}.json", {"root_ids": ids, "answer_sha256": hashes,
                "conditions": evaluated.tolist(), "physical": physical,
                "feature_sha256": shared.file_sha(directory / f"{role}.npy"),
                "corpus_manifest_sha256": manifest["manifest_sha256"], "evidence_role": EVIDENCE_ROLE})
            for suffix in ("npy", "json"):
                path = directory / f"{role}.{suffix}"
                frozen[str(path)] = shared.file_sha(path)
        schemas[kind] = schema
        path = directory / "schema.json"
        shared.write_json(path, {"features": schema, "plan": asdict(plan), "evidence_role": EVIDENCE_ROLE})
        frozen[str(path)] = shared.file_sha(path)
    if previous.read_manifest(corpus, verify_hashes=True)["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("fresh corpus changed during reference feature extraction")
    shared.write_json(arguments.output / "frozen-reference-features.json", {
        "sha256": frozen, "corpus_manifest_sha256": manifest["manifest_sha256"],
        "opened_only_after_all_model_hashes_frozen": True, "evidence_role": EVIDENCE_ROLE})
    return samples, schemas, frozen


def audit_frozen_model(arguments, spec, entry, directory, decoder, samples, schemas):
    """Reuse native full-D generation; relabel even partial audit artifacts."""
    metadata = {"evidence_role": EVIDENCE_ROLE, "independent_confirmation": False, "blind_test": False,
        "admitted": False, "used_for_checkpoint_selection": False,
        "checkpoint_role": entry["checkpoint_role"], "requested_seconds": entry.get("requested_seconds"),
        "evaluation_reuse": "same_fresh_references_across_arms_replicas_and_prespecified_timepoints",
        "multiple_models_and_timepoints_adjusted": False,
        "learning_curve_used_for_checkpoint_selection": False}
    try:
        matrices, reports = previous.sample_and_audit(arguments, spec, Path(entry["checkpoint"]),
            entry["sha256"], directory, decoder, samples, schemas)
    finally:
        # The reused helper labels its original historical-cache use. Correct
        # that label on any completed subreport even if the second audit fails.
        for kind in ("physical", "derivative"):
            path = directory / f"{kind}-distribution-audit.json"
            if path.exists():
                report = read_json(path)
                report.update(metadata)
                shared.write_json(path, report)
    for report in reports.values():
        report.update(metadata)
    return matrices, reports


def register_models(key, replica, arm, arm_root, checkpoint, report, thresholds):
    """Freeze only validation-selected best and declared elapsed-time receipts."""
    def entry(path, digest, **metadata):
        path = Path(path)
        if not path.resolve().is_relative_to((arm_root / "checkpoints").resolve()):
            raise ValueError("training checkpoint escaped its arm output")
        return {"replica": replica, "arm": arm, "checkpoint": str(path), "sha256": digest, **metadata}

    models = {key + "/best": entry(checkpoint, report["checkpoint_sha256"],
        checkpoint_role="training_validation_selected_best", step=report["best_step"],
        validation_score=report["validation_selection_score"])}
    receipts = report.get("checkpoint_snapshots", [])
    if len(receipts) > len(thresholds) or report.get("unreached_checkpoint_seconds") != list(thresholds[len(receipts):]):
        raise ValueError("training snapshot receipts do not account for the declared schedule")
    for index, receipt in enumerate(receipts):
        if (receipt["requested_seconds"] != thresholds[index] or isinstance(receipt["step"], bool)
                or not isinstance(receipt["step"], int) or receipt["step"] < 1
                or not np.isfinite(receipt["actual_elapsed_seconds"])
                or receipt["actual_elapsed_seconds"] < thresholds[index]
                or not np.isfinite(receipt["validation_score"])):
            raise ValueError("invalid elapsed-time training snapshot receipt")
        models[key + f"/snapshot-{index:02d}-{thresholds[index]:g}s"] = entry(receipt["checkpoint"], receipt["sha256"],
            checkpoint_role="predeclared_developmental_learning_curve", step=receipt["step"],
            requested_seconds=receipt["requested_seconds"], actual_elapsed_seconds=receipt["actual_elapsed_seconds"],
            validation_score=receipt["validation_score"])
    return models


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "cache", "pilot", "selection", "output"):
        result.add_argument(f"--{name}", type=Path, required=True)
    result.add_argument("--regime", choices=("weak", "strong"), required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--candidate", choices=CANDIDATES, required=True)
    result.add_argument("--replicas", type=int, default=2)
    result.add_argument("--train-seconds", type=int, default=1800)
    result.add_argument("--checkpoint-seconds", type=float, nargs="+", default=(360., 900.))
    result.add_argument("--steps", type=int, default=100000)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--collect-batch-size", type=int, default=8)
    result.add_argument("--validate-every", type=int, default=1000)
    result.add_argument("--learning-rate", type=float, default=2.e-4)
    result.add_argument("--width", type=int, default=256)
    result.add_argument("--depth", type=int, default=4)
    result.add_argument("--covariance-floor", type=float, default=.05)
    result.add_argument("--generated-answers", type=int, default=128)
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
        cli.error("real VPFP scale-up requires a Slurm compute allocation")
    if arguments.output.exists() or arguments.output.is_symlink() or any(
            arguments.output.resolve().is_relative_to(path.resolve()) for path in (arguments.cache, arguments.pilot)):
        cli.error("choose a fresh output outside the immutable source experiments")
    if any(getattr(arguments, name) < 1 for name in ("steps", "train_seconds", "batch_size", "collect_batch_size", "validate_every", "width", "depth")):
        cli.error("training budgets, dimensions and batch sizes must be positive")
    if arguments.replicas != 2 or arguments.generated_answers != 128 or arguments.batch_size > 8 or arguments.sample_steps < 2:
        cli.error("this fixed paired experiment requires two replicas, 128 generated/reference answers, batch<=8 and >=2 Heun steps")
    if arguments.collect_batch_size > 8 or 128 % arguments.collect_batch_size:
        cli.error("collection batch size must be <=8 and divide each 128-answer evaluation split")
    if arguments.seed < 0 or arguments.covariance_floor > 1 or any(
            not np.isfinite(getattr(arguments, name)) or getattr(arguments, name) <= 0
            for name in ("learning_rate", "covariance_floor", "constraint_tolerance")):
        cli.error("invalid seed or finite positive numeric setting")
    try:
        arguments.checkpoint_seconds = shared._checkpoint_seconds(arguments)
    except ValueError as error:
        cli.error(str(error))
    if max(arguments.checkpoint_seconds) > arguments.train_seconds:
        cli.error("checkpoint thresholds must not exceed the per-arm training wall budget")
    previous.DiffusionAuditConfig(permutations=arguments.permutations, bootstrap_repeats=arguments.bootstrap_repeats).validate()
    config = read_json(arguments.config)
    spec = previous.VPFPOnlineModelSpec.from_mapping(config.get("vpfp", config))
    spec.preflight()
    source_corpus = arguments.cache / "corpus"
    source = previous.read_manifest(source_corpus, verify_hashes=True)
    frozen_inputs, selection = freeze_inputs(arguments, spec, source)
    arguments.source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    fingerprints = shared.source_fingerprints()
    for filename in ("run_vpfp_temporal_training_pilot.py", "run_vpfp_value_control_pilot.py", Path(__file__).name):
        fingerprints[f"scripts/{filename}"] = shared.file_sha(ROOT / "scripts" / filename)
    frozen_code = {str(ROOT / name): digest for name, digest in fingerprints.items()}
    arms = ("value_base", arguments.candidate)
    arguments.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    plan = {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()}
    plan.update(schema="vpfp_paired_scaleup_experiment_v1", model_spec=asdict(spec), arms=arms,
        source_corpus_manifest_sha256=source["manifest_sha256"], frozen_input_sha256=frozen_inputs,
        source_file_sha256=fingerprints, manual_selection=selection, split_counts=COUNTS,
        collection_seed=arguments.seed + 5000000, newly_solved_evaluation_answers=384,
        training_bytes="independent_byte_identical_copy_of_source_train_and_validation",
        evaluation_seeds="exclude_all_source_equation_seeds_and_embedded_cumulative_history",
        training_start="from_scratch_no_pilot_optimizer_or_EMA_resume",
        best_selection="training_validation_only_within_arm",
        snapshot_selection="fixed_elapsed_time_only_never_best_selection",
        objective="base_EDM_in_declared_full_rank_noise_coordinates",
        rho_policy="identity_in_parent_value_normalized_coordinates",
        noise="original_or_half_original_half_log_uniform_0.01_to_0.3",
        normalization="paired_value_RMS_or_kinetic_only_train_DC_centered_temporal_covariance",
        initialization="identical_network_parameters_within_each_replica", arm_order="forward_then_reverse",
        generated_condition_design="independent_uniform_within_same_4_by_2_strata_not_reference_conditions",
        generated_seed_pairing="same_seed_plus_200_conditions_and_seed_plus_300000000_noise_within_replica_across_models",
        cross_normalizer_validation_loss_comparable=False, evidence_role=EVIDENCE_ROLE,
        evaluation_reuse="same_fresh_references_for_all_models_and_timepoints",
        blind_test=False, multiple_models_and_timepoints_adjusted=False,
        planned_best_models=4, planned_time_snapshots=4 * len(arguments.checkpoint_seconds),
        planned_generated_answers=4 * (1 + len(arguments.checkpoint_seconds)) * arguments.generated_answers,
        nominal_total_training_seconds=4 * arguments.train_seconds,
        budget_excludes="collection_preflight_sampling_feature_extraction_and_audits; training_includes_validation_and_checkpoint_IO",
        scientific_admission=False, automatic_submission=False)
    shared.write_json(arguments.output / "frozen-plan.json", plan)
    plan_sha = shared.file_sha(arguments.output / "frozen-plan.json")
    frozen_inputs[str(arguments.output / "frozen-plan.json")] = plan_sha
    started, stage = time.monotonic(), "source_constraint_check"
    training, models, summaries = {}, {}, {}

    def set_stage(value):
        nonlocal stage
        stage = value
        shared.write_json(arguments.output / "stage.json", {"stage": stage, "status": "running"})

    try:
        set_stage(stage)
        previous.assert_hashes({**frozen_inputs, **frozen_code})
        constraint = previous.audit_value_control_corpus(source_corpus, batch_size=8, tolerance=arguments.constraint_tolerance)
        shared.write_json(arguments.output / "source-value-slope-corpus-audit.json", constraint)
        previous.require_value_control_audit(constraint)
        if constraint["source_manifest_sha256"] != source["manifest_sha256"]:
            raise ValueError("source corpus changed after plan freeze")
        set_stage("full_layout_preflight")
        shared.write_json(arguments.output / "allocation-preflight.json", temporal.preflight(arguments, spec, source_corpus))
        set_stage("collect_fresh_evaluation")
        previous.assert_hashes({**frozen_inputs, **frozen_code})
        corpus = arguments.output / "corpus"
        collect_fresh_evaluation_corpus(source_corpus, corpus, plan["collection_seed"],
            batch_size=arguments.collect_batch_size, device=arguments.device)
        manifest = previous.read_manifest(corpus, verify_hashes=True)
        if (manifest["split_counts"] != COUNTS or manifest["source_corpus"]["manifest_sha256"] != source["manifest_sha256"]
                or manifest["model_spec"] != source["model_spec"]):
            raise ValueError("fresh corpus does not preserve the frozen source counts and identity")
        frozen_inputs[str(corpus / "manifest.json")] = shared.file_sha(corpus / "manifest.json")
        shared.write_json(arguments.output / "fresh-corpus-receipt.json", {
            "plan_sha256": plan_sha, "corpus_manifest_sha256": manifest["manifest_sha256"],
            "source_corpus_manifest_sha256": source["manifest_sha256"], "source_corpus": manifest["source_corpus"],
            "new_pde_solves": 384, "physical_features_opened": False, "generated_samples_drawn": False})
        for replica in range(2):
            run_args = copy(arguments)
            run_args.seed = arguments.seed + replica * 10000
            replica_root = arguments.output / f"seed-{replica}"
            replica_root.mkdir()
            for arm in arms if replica == 0 else tuple(reversed(arms)):
                key = f"seed-{replica}/{arm}"
                set_stage("train_" + key)
                previous.assert_hashes({**frozen_inputs, **frozen_code})
                cfg, make_model, fit = temporal.factories(run_args, spec, arm)
                run_args.ablation_metadata = {"plan_sha256": plan_sha, "arm": arm, "replica": replica,
                    "evidence_role": EVIDENCE_ROLE, "manual_selection_sha256": frozen_inputs[str(arguments.selection)],
                    "cross_normalizer_validation_loss_comparable": False}
                arm_root = replica_root / arm
                arm_root.mkdir()
                checkpoint, report = shared.train(run_args, spec, corpus, arm_root,
                    model_config_factory=lambda *a, cfg=cfg: cfg, model_factory=make_model,
                    normalizer_factory=fit, training_metric_factory=None, validation_metric_factory=None)
                training[key] = report
                models.update(register_models(key, replica, arm, arm_root, checkpoint, report, arguments.checkpoint_seconds))
            paired = [training[f"seed-{replica}/{arm}"] for arm in arms]
            if paired[0]["initial_model_sha256"] != paired[1]["initial_model_sha256"]:
                raise ValueError("paired initial network parameter identities differ")
            if arguments.candidate == "value_low" and paired[0]["normalizer_sha256"] != paired[1]["normalizer_sha256"]:
                raise ValueError("paired value noise-coverage arms used different normalizers")
        frozen_models = {entry["checkpoint"]: entry["sha256"] for entry in models.values()}
        previous.assert_hashes({**frozen_inputs, **frozen_code, **frozen_models})
        shared.write_json(arguments.output / "frozen-checkpoints.json", {
            "all_arms_and_snapshots_frozen_before_any_feature_evaluation": True,
            "sha256": frozen_models, "models": models, "plan_sha256": plan_sha,
            "time_snapshots_used_for_checkpoint_selection": False})
        set_stage("frozen_reference_feature_extraction")
        first = models["seed-0/value_base/best"]
        sampler = previous.load_vpfp_diffusion_sampler(first["checkpoint"], first["sha256"],
            device=arguments.device, steps=arguments.sample_steps)
        shared.assert_checkpoint_corpus(sampler, manifest, previous.load_split(corpus, "train", verify_hashes=False).sample_ids)
        binding = previous.build_vpfp_online_binding(spec, output_root=arguments.output / "audit-runtime",
            device=arguments.device, latent_sampler=sampler)
        decoder = binding.plugin.generator.backend.runtime.base_model
        samples, schemas, frozen_features = write_reference_features(arguments, corpus, decoder, manifest)
        best_matrices, best_reports = {}, {}
        for key, entry in models.items():
            set_stage("frozen_audit_" + key)
            previous.assert_hashes({**frozen_inputs, **frozen_code, **frozen_models, **frozen_features})
            run_args = copy(arguments)
            run_args.seed = arguments.seed + entry["replica"] * 10000
            run_args.cache = arguments.output  # Native sampler binds to the NEW manifest and FULL decoder D.
            directory = arguments.output / f"seed-{entry['replica']}" / entry["arm"] / "evaluation" / key.rsplit("/", 1)[1]
            directory.mkdir(parents=True)
            matrices, reports = audit_frozen_model(run_args, spec, entry, directory, decoder, samples, schemas)
            summaries[key] = {"checkpoint": entry, "evaluation_directory": str(directory),
                "generated_answers": arguments.generated_answers, "audit": {
                    kind: {"status": report["status"], "sw_effect": report["comparisons"]["generated"]["statistics"]["sliced_wasserstein_max"]["effect"]}
                    for kind, report in reports.items()}}
            if entry["checkpoint_role"] == "training_validation_selected_best":
                best_matrices.setdefault(entry["replica"], {})[entry["arm"]] = matrices
                best_reports.setdefault(entry["replica"], {})[entry["arm"]] = reports
        set_stage("paired_best_physical_effect_scorecards")
        for replica in range(2):
            matrices = {role: {kind: samples[kind][role].features for kind in samples} for role in ("reference_a", "reference_b")}
            matrices.update(best_matrices[replica])
            card = previous.build_vpfp_direction_scorecard(regime=arguments.regime, seed=arguments.seed + replica * 10000,
                objective="prespecified_scaleup_base_EDM_noise_geometry_and_coverage", baseline_arm="value_base", samples=matrices,
                physical_schema=schemas["physical"], derivative_schema=schemas["derivative"], joint_reports=best_reports[replica],
                physical_time_fractions=previous.VPFPFeaturePlan().time_fractions,
                derivative_time_fractions=previous.VPFPDerivativeFeaturePlan().time_fractions,
                macro_probe_count=previous.VPFPDerivativeFeaturePlan().macro_probe_count,
                phase_probe_count=previous.VPFPDerivativeFeaturePlan().phase_probe_count)
            card.update(evidence_role=EVIDENCE_ROLE, used_for_checkpoint_selection=False,
                automatic_scaleup_or_admission=False, heuristic_boolean_used_as_gate=False)
            shared.write_json(arguments.output / f"seed-{replica}/physical-effect-scorecard.json", card)
        set_stage("final_immutability_checks")
        previous.assert_hashes({**frozen_inputs, **frozen_code, **frozen_models, **frozen_features})
        if (previous.read_manifest(corpus, verify_hashes=True)["manifest_sha256"] != manifest["manifest_sha256"]
                or previous.read_manifest(source_corpus, verify_hashes=True)["manifest_sha256"] != source["manifest_sha256"]):
            raise ValueError("source or fresh corpus changed during the frozen experiment")
        report = {"execution_passed": True, "regime": arguments.regime, "candidate": arguments.candidate,
            "training": training, "audits": summaries, "wall_seconds": time.monotonic() - started,
            "new_pde_solves": 384, "generated_answers": len(models) * arguments.generated_answers,
            "all_learning_curve_thresholds_reached": all(not value["unreached_checkpoint_seconds"] for value in training.values()),
            "evidence_role": EVIDENCE_ROLE, "blind_test": False, "admitted": False,
            "canonical_campaign_smoke": "separate_required_step", "automatic_followup_job_submitted": False}
    except Exception as error:
        report = {"execution_passed": False, "regime": arguments.regime, "candidate": arguments.candidate,
            "failed_stage": stage, "error": repr(error), "traceback": traceback.format_exc(),
            "training": training, "completed_audits": summaries, "partial_artifacts_preserved": True,
            "evidence_role": EVIDENCE_ROLE, "admitted": False, "automatic_followup_job_submitted": False}
    shared.write_json(arguments.output / "result.json", report)
    shared.write_json(arguments.output / "stage.json", {"stage": stage, "status": "complete" if report["execution_passed"] else "failed"})
    shared.emit("scaleup_experiment_complete", execution_passed=report["execution_passed"], error=report.get("error"))
    return 0 if report["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
