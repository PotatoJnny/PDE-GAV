"""Source-aware strong development: exact continuation and one rho control.

This driver never equates matching the old reference source law with being in
the physical regime. It preserves the original distribution reports as
development diagnostics; source-scale qualification is a separate report.
"""
from __future__ import annotations

import argparse
from copy import copy, deepcopy
from dataclasses import replace
import gc
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
loader = importlib.util.spec_from_file_location("strong_coherent_pilot_reuse", ROOT / "scripts/run_vpfp_strong_initial_pilot.py")
old = importlib.util.module_from_spec(loader)
loader.loader.exec_module(old)
shared, previous = old.shared, old.previous
from pde_agent_verifier_solver.plugins.vpfp.temporal_covariance_diffusion import VPFPTemporalCovarianceNormalizer

RHO_ARM = "initial_conditioned_rho_cov"
ARMS = (*old.ARMS, RHO_ARM)
CONTINUATION_STEPS = (30000, 40000)
NEW_STEPS = 20000
EVIDENCE_ROLE = "development_source_aware_strong_regime_not_old_source_law_admission"
LEGACY_REVISION = "d4d50a145c910f42de75ed875daea96f9d26fa92"
LEGACY_TRAINER_SHA = "e934c62e543d283097767dfc5b3c1f0a01006b7bba66ea8606d764e8ba6de1af"


def factories(arguments, spec, manifest, arm, initialization):
    if arm not in ARMS:
        raise ValueError("unknown strong coherent arm")
    config, make, fit = old.factories(arguments, spec, manifest,
        "initial_conditioned" if arm == RHO_ARM else arm, initialization)
    if arm == RHO_ARM:
        prior = old.KnownInitialPriorSpec.from_manifest(manifest)
        def fit(config, raw, *, sample_ids):
            return old.VPFPInitialConditionedNormalizer.fit(config, raw, sample_ids=sample_ids,
                initial_prior=prior, rho_policy="covariance", center_kinetic_dc=True,
                covariance_strength=1., relative_floor=.05)
    return config, make, fit


def preflight(arguments, spec, manifest, *, arms=(RHO_ARM,)):
    """Eight TRAIN rows only; normalizer/AdamW/EMA/native sampling, no training run."""
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError("preflight needs a distinct nonempty subset of declared arms")
    split = previous.load_split(arguments.source / "corpus", "train", verify_hashes=False)
    raw = torch.from_numpy(np.array(split.latents[:8], copy=True))
    conditions = np.array(split.conditions[:4], copy=True)
    validation = replace(split, coefficients=raw[:4].numpy(), conditions=conditions,
        equation_seeds=split.equation_seeds[:4], case_ids=split.case_ids[:4], stratum_ids=split.stratum_ids[:4])
    reports = {}
    for arm in arms:
        if torch.device(arguments.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(arguments.device)
        initialization = {}
        config, make, fit = factories(arguments, spec, manifest, arm, initialization)
        normalizer = fit(config, raw, sample_ids=list(map(int, split.sample_ids[:8]))).to(arguments.device)
        model = make(config).to(arguments.device)
        ema = deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=0.)
        encoded = normalizer.encode(raw[:4].to(arguments.device))
        restored = normalizer.decode(encoded)
        numerical = old.audit_normalizer_roundtrip(normalizer, raw[:4].to(restored), restored)
        shared.write_json(arguments.output / f"preflight-{arm}-normalization.json", {"arm": arm, **numerical})
        if not numerical["passed"]:
            raise ValueError(f"{arm} original-tolerance numerical mechanics gate failed: {numerical['status']}")
        random = torch.Generator(device=arguments.device).manual_seed(arguments.seed + 901)
        loss, _ = model.loss(encoded, conditions, generator=random)
        if not torch.isfinite(loss):
            raise ValueError("strong preflight loss is nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for target, source in zip(ema.parameters(), model.parameters()):
                target.lerp_(source, .5)
        report = shared.validation_report(ema, normalizer, validation, batch_size=4, seed=arguments.seed+103)
        if not np.isfinite(report["selection_score"]):
            raise ValueError("strong preflight EMA validation is nonfinite")
        if arm != "cov_base":
            sampler = old.VPFPInitialConditionedBatchSampler(ema, normalizer,
                base_decoder_sha256=spec.base_decoder_sha256, checkpoint_sha256="0"*64, steps=2)
            sampled = sampler.sample_conditions(conditions[:1], [arguments.seed + 90])
            del sampler
        else:
            sampled = normalizer.decode(ema.sample(conditions[:1], [arguments.seed + 90], steps=2))
        if sampled.shape != (1, config.full_dimension) or not torch.isfinite(sampled).all():
            raise ValueError("strong preflight native full-D sampler failed")
        reports[arm] = {"execution_passed": True, **initialization, "normalization_gate": numerical,
            "adamw_and_ema_update_passed": True, "ema_validation": report,
            "full_dimension": config.full_dimension, "random_dimension": config.latent_dimension,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(arguments.device) if sampled.is_cuda else None,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(arguments.device) if sampled.is_cuda else None,
            "training_updates_committed": 0, "new_pde_solves": 0, "reference_features_opened": False,
            "scientific_validation": False, "data_role": "first_eight_training_rows_mechanics_only"}
        del model, ema, optimizer, normalizer, encoded, restored, sampled, loss, source, target
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    shared.write_json(arguments.output / "allocation-preflight.json", reports)
    return reports


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "source", "pilot", "output"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    cli.add_argument("--preflight-only", action="store_true")
    cli.add_argument("--device", choices=("cuda",), default="cuda")
    cli.set_defaults(seed=1620260906, regime="strong", train_seconds=2400., batch_size=4,
        width=256, depth=4, learning_rate=2.e-4, validate_every=1000, covariance_floor=.05,
        checkpoint_seconds=(), generated_answers=128, sample_steps=64, permutations=99, bootstrap_repeats=49,
        maximum_wall_seconds=8*3600., budget_receipt_margin_seconds=120.)
    return cli


def source_inventory(arguments, manifest):
    """Bind old last states now, without opening any generated/reference features."""
    pilot = arguments.pilot
    plan, result, frozen = (old.read_json(pilot / name) for name in
        ("frozen-plan.json", "result.json", "frozen-checkpoints.json"))
    expected = {"seed": arguments.seed, "steps": 20000, "batch_size": 4, "width": 256,
        "depth": 4, "learning_rate": 2.e-4, "validate_every": 1000, "regime": "strong",
        "source_revision": LEGACY_REVISION, "source_corpus_manifest_sha256": manifest["manifest_sha256"]}
    if (result.get("execution_passed") is not True or any(plan.get(k) != v for k, v in expected.items())
            or Path(plan["source"]).resolve() != arguments.source.resolve()
            or frozen.get("all_models_completed_updates") != 20000):
        raise ValueError("source must be the completed frozen 20k strong initial pilot and its original corpus")
    source_scripts = [value for key, value in plan["frozen_input_sha256"].items()
                      if key.endswith("/scripts/train_vpfp_structured_diffusion.py")]
    if source_scripts != [LEGACY_TRAINER_SHA]:
        raise ValueError("legacy trainer bytes differ from the reviewed explicit-RNG graph")
    hashes = {str(pilot / name): shared.file_sha(pilot / name) for name in
              ("frozen-plan.json", "result.json", "frozen-checkpoints.json")}
    entries = {}
    for replica in range(2):
        for arm in old.ARMS:
            key = f"seed-{replica}/{arm}"
            directory = pilot / key
            summary = old.read_json(directory / "training-summary.json")
            last, best = directory / "checkpoints/last.pt", directory / "checkpoints/best.pt"
            if (summary["final_step"] != 20000 or summary["stop_reason"] != "step_limit"
                    or summary["training_presentations"] != 80000
                    or frozen["models"][key]["sha256"] != shared.file_sha(best)
                    or summary["checkpoint_sha256"] != frozen["models"][key]["sha256"]):
                raise ValueError(f"source model completion/best binding differs: {key}")
            for path in (last, best, directory / "training-summary.json"):
                if path.is_symlink() or not path.is_file():
                    raise ValueError("source assets must be regular files")
                hashes[str(path)] = shared.file_sha(path)
            entries[key] = {"last": str(last), "last_sha256": hashes[str(last)],
                "best": str(best), "best_sha256": hashes[str(best)], "best_step": summary["best_step"],
                "normalizer_sha256": summary["normalizer_sha256"],
                "initial_model_sha256": summary["initial_model_sha256"]}
    return entries, hashes, plan["generation_design"]


def train_models(arguments, spec, manifest, inputs, plan_sha, stage, guard):
    training, models, initializations = {}, {}, {}
    for replica in range(2):
        for arm in ARMS if replica == 0 else ARMS[::-1]:
            key = f"seed-{replica}/{arm}"
            stage("training/" + key)
            guard()
            args = copy(arguments)
            args.seed = arguments.seed + 10000*replica
            args.steps = NEW_STEPS if arm == RHO_ARM else CONTINUATION_STEPS[-1]
            args.checkpoint_steps = (NEW_STEPS,) if arm == RHO_ARM else CONTINUATION_STEPS
            args.ablation_metadata = {"arm": arm, "replica": replica, "plan_sha256": plan_sha,
                "evidence_role": EVIDENCE_ROLE, "cross_normalizer_validation_loss_comparable": False,
                "fixed_snapshots_not_used_for_selection": True,
                "rho_policy": "covariance" if arm == RHO_ARM else "identity",
                "conditional_rho_only_change": arm == RHO_ARM}
            if arm != RHO_ARM:
                source = inputs[key]
                args.resume_checkpoint, args.expected_resume_sha256 = Path(source["last"]), source["last_sha256"]
                args.resume_best_checkpoint, args.expected_resume_best_sha256 = Path(source["best"]), source["best_sha256"]
                args.legacy_resume_contract = {"source_revision": LEGACY_REVISION, "source_script_sha256": LEGACY_TRAINER_SHA,
                    "batch_size": 4, "validate_every": 1000}
            initialization = {}
            config, make, fit = factories(args, spec, manifest, arm, initialization)
            load = (VPFPTemporalCovarianceNormalizer if arm == "cov_base" else old.VPFPInitialConditionedNormalizer).from_payload
            directory = arguments.output / key
            directory.mkdir(parents=True)
            if arm != RHO_ARM:
                # last holds the actual step-20k EMA, unlike seed-0 cov best
                # (19k). Export a native safe view, not a new optimizer state.
                payload = shared._read_training_asset(args.resume_checkpoint, args.expected_resume_sha256)
                if payload["training"]["step"] != 20000 or "continuation_state" not in payload:
                    raise ValueError("learning-curve starting EMA is not the frozen 20000-update last state")
                del payload["continuation_state"]
                payload["training"].update(checkpoint_role="fixed_update_continuation_start_EMA",
                    parent_last_sha256=args.expected_resume_sha256,
                    snapshot_validation_used_for_checkpoint_selection=False)
                path = directory / "initial-step-020000.pt"
                shared.save_checkpoint(path, payload)
                models[f"{key}/step-020000"] = {"replica": replica, "arm": arm, "step": 20000,
                    "checkpoint": str(path), "sha256": shared.file_sha(path),
                    "parent_last_sha256": args.expected_resume_sha256,
                    "checkpoint_role": "exact_20000_EMA_from_frozen_last_not_historical_best"}
                del payload
                shared.write_json(arguments.output / "training-progress.json", {"training": training, "models": models,
                    "physical_features_opened": False, "status": "starting_EMA_frozen_before_continuation"})
            _, report = shared.train(args, spec, arguments.source / "corpus", directory,
                model_config_factory=lambda *a, cfg=config: cfg, model_factory=make,
                normalizer_factory=fit, normalizer_loader=load)
            training[key], initializations[key] = report, initialization
            if arm != RHO_ARM and report["normalizer_sha256"] != inputs[key]["normalizer_sha256"]:
                raise ValueError("continued normalization changed")
            for receipt in report.get("step_snapshots", []):
                state = f"{key}/step-{receipt['step']:06d}"
                models[state] = {"replica": replica, "arm": arm, "step": receipt["step"],
                    "checkpoint": receipt["checkpoint"], "sha256": receipt["sha256"],
                    "checkpoint_role": "fixed_update_snapshot_not_validation_selected", **initialization}
            shared.write_json(arguments.output / "training-progress.json", {"training": training, "models": models,
                "physical_features_opened": False, "status": "not_all_models_frozen"})
            if report["final_step"] != args.steps or report["stop_reason"] != "step_limit" or report.get("unreached_checkpoint_steps"):
                raise RuntimeError(f"{key} did not complete its fixed update budget; retain partial experiment")
        if initializations[f"seed-{replica}/{RHO_ARM}"]["common_trunk_sha256"] != initializations[f"seed-{replica}/initial_conditioned"]["common_trunk_sha256"]:
            raise ValueError("rho geometry control did not retain the same initial trunk")
    frozen = {entry["checkpoint"]: entry["sha256"] for entry in models.values()}
    # Best/last assets remain durable even though only predeclared steps enter
    # the generation comparison. Bind all of them before reference features.
    for key, report in training.items():
        frozen[report["checkpoint"]] = report["checkpoint_sha256"]
        path = arguments.output / key / "checkpoints/last.pt"
        frozen[str(path)] = shared.file_sha(path)
    previous.assert_hashes(frozen)
    shared.write_json(arguments.output / "frozen-checkpoints.json", {"models": models, "sha256": frozen,
        "all_models_frozen_before_feature_reading": True, "generation_selection": "fixed_updates_only",
        "continuation_steps": CONTINUATION_STEPS, "new_rho_covariance_steps": NEW_STEPS,
        "within_model_best_retained_but_not_used_to_select_learning_curve": True})
    return training, models, frozen


def evaluate_models(arguments, spec, manifest, models, design, constraint, stage, guard):
    samples, schemas = old.load_references(arguments, manifest, constraint)
    first = next(iter(models.values()))
    sampler = previous.load_vpfp_diffusion_sampler(first["checkpoint"], first["sha256"], device=arguments.device, steps=64)
    binding = previous.build_vpfp_online_binding(spec, output_root=arguments.output / "audit-runtime",
        device=arguments.device, latent_sampler=sampler)
    decoder = binding.plugin.generator.backend.runtime.base_model
    summaries = {}
    for replica in range(2):
        matrices = {role: {kind: samples[kind][role].features for kind in samples} for role in ("reference_a", "reference_b")}
        reports = {}
        args = copy(arguments)
        args.seed += 10000*replica
        for key, entry in models.items():
            if entry["replica"] != replica:
                continue
            stage("development_audit/" + key)
            guard()
            directory = arguments.output / key / "evaluation"
            directory.mkdir(parents=True)
            # The reused sampler chooses by native class, not arm label. Give
            # its original conditioning name solely for truthful prior flags.
            native_entry = dict(entry, arm="cov_base" if entry["arm"] == "cov_base" else "initial_conditioned")
            name = f"{entry['arm']}__{entry['step']}"
            matrices[name], reports[name] = old.sample_and_audit(args, spec, manifest, native_entry, directory,
                decoder, samples, schemas, design[f"seed-{replica}"])
            for kind, report in reports[name].items():
                report.update(evidence_role=EVIDENCE_ROLE, experiment_arm=entry["arm"], global_update=entry["step"],
                    fixed_update_learning_curve=True, old_source_law_match_required_for_regime=False,
                    scale_based_regime_qualification="separate_pending_audit_not_established_here")
                shared.write_json(directory / f"{kind}-distribution-audit.json", report)
            provenance = old.read_json(directory / "generated-provenance.json")
            provenance.update(experiment_arm=entry["arm"], global_update=entry["step"], evidence_role=EVIDENCE_ROLE,
                source_at_product_boundary="continuous_answer_induced_standard_VPFP_residual",
                source_shape_must_match_old_parameter_family=False,
                inference_design="reused_frozen_conditions_and_seeds_for_development_learning_curve_not_new_blind_sample")
            shared.write_json(directory / "generated-provenance.json", provenance)
            summaries[key] = {kind: {"status": report["status"],
                "sw_effect": report["comparisons"]["generated"]["statistics"]["sliced_wasserstein_max"]["effect"]}
                for kind, report in reports[name].items()}
            shared.write_json(arguments.output / "audit-progress.json", summaries)
        # Every card compares equal GLOBAL update counts. Learning-curve
        # trends across these three cards are not an unequal-budget arm test.
        index = {}
        for step in (20000, *CONTINUATION_STEPS):
            names = {name for name in reports if name.endswith(f"__{step}")}
            selected = {name: value for name, value in matrices.items() if name in names or name in ("reference_a", "reference_b")}
            card = previous.build_vpfp_direction_scorecard(regime="strong", seed=args.seed,
                objective="fixed_update_source_aware_continuation_and_rho_geometry_development",
                baseline_arm=f"cov_base__{step}", samples=selected, physical_schema=schemas["physical"],
                derivative_schema=schemas["derivative"], joint_reports={name: reports[name] for name in names},
                physical_time_fractions=previous.VPFPFeaturePlan().time_fractions,
                derivative_time_fractions=previous.VPFPDerivativeFeaturePlan().time_fractions,
                macro_probe_count=previous.VPFPDerivativeFeaturePlan().macro_probe_count,
                phase_probe_count=previous.VPFPDerivativeFeaturePlan().phase_probe_count)
            card.update(evidence_role=EVIDENCE_ROLE, used_for_checkpoint_selection=False, heuristic_boolean_used_as_gate=False,
                automatic_scaleup_or_admission=False, global_update=step, equal_global_update_counts=True,
                equal_parameter_causal_comparison=False, old_source_law_match_required_for_regime=False)
            path = arguments.output / f"seed-{replica}/physical-effect-scorecard-step-{step:06d}.json"
            shared.write_json(path, card)
            index[str(step)] = {"path": str(path), "sha256": shared.file_sha(path)}
        shared.write_json(arguments.output / f"seed-{replica}/physical-effect-scorecards.json", index)
    return summaries


def main(argv=None):
    cli = parser()
    arguments = cli.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID", "").isdigit() or not os.environ.get("SLURM_TMPDIR"):
        cli.error("strong coherent experiment requires a Slurm compute allocation and local scratch")
    if not torch.cuda.is_available():
        cli.error("a usable CUDA device is required")
    if arguments.output.is_symlink():
        cli.error("output must not be a symlink")
    for name in ("source", "pilot", "config", "output"):
        setattr(arguments, name, getattr(arguments, name).resolve())
    output = arguments.output
    regular_location = (output.parent == old.SCRATCH.resolve()
        and output.name == f"vpfp-strong-regime-coherent-{os.environ['SLURM_JOB_ID']}")
    preflight_location = arguments.preflight_only and output.is_relative_to(old.SCRATCH.resolve())
    if (output.exists() or not (regular_location or preflight_location)
            or output.is_relative_to(arguments.source) or output.is_relative_to(arguments.pilot)):
        cli.error("use a fresh exact job-specific strong-regime-coherent scratch output")
    arguments.source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    output.mkdir(parents=True)
    current_stage, started = "source_preflight", time.monotonic()
    result = {"execution_passed": False, "training": {}, "audits": {}, "regime": "strong",
        "evidence_role": EVIDENCE_ROLE, "admitted": False, "new_pde_solves": 0,
        "scale_based_regime_qualification": "separate_pending_audit", "automatic_followup_submission": False}
    frozen_models = {}
    def stage(name):
        nonlocal current_stage
        current_stage = name
        shared.write_json(output / "stage.json", {"stage": name, "status": "running"})
    try:
        spec = previous.VPFPOnlineModelSpec.from_mapping(old.read_json(arguments.config)["vpfp"])
        spec.preflight()
        corpus = arguments.source / "corpus"
        manifest = previous.read_manifest(corpus, verify_hashes=True)
        if manifest["split_counts"] != old.COUNTS or spec != previous.VPFPOnlineModelSpec.from_mapping(manifest["model_spec"]):
            raise ValueError("strong coherent run requires original 2048/128x4 native corpus")
        old.check_completed_source(arguments.source)
        baseline = old.check_baseline_recipe(arguments.source, manifest)
        inputs, frozen, design = source_inventory(arguments, manifest)
        frozen.update(old.reference_inventory(arguments.source, manifest))
        paths = [arguments.config, corpus / "manifest.json", corpus / manifest["seed_plan"]["path"], Path(__file__),
            ROOT / "scripts/run_vpfp_strong_initial_pilot.py", ROOT / "scripts/run_vpfp_temporal_training_pilot.py",
            ROOT / "scripts/run_vpfp_value_control_pilot.py"]
        frozen.update({str(path): shared.file_sha(path) for path in paths})
        frozen.update({str(ROOT / key): value for key, value in shared.source_fingerprints().items()})
        plan = {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()}
        plan.update(schema="vpfp_source_aware_strong_continuation_and_rho_control_v1", inputs=inputs,
            source_corpus_manifest_sha256=manifest["manifest_sha256"], frozen_input_sha256=frozen,
            baseline_recipe=baseline, continuation_steps=CONTINUATION_STEPS, new_rho_steps=NEW_STEPS,
            generation_design=design, generation_design_reused_from_opened_pilot=True,
            network_loss_sampler_unchanged=True, conditional_rho_covariance_only_new_training_change=True,
            new_rho_arm_same_model_initialization_as_original_conditional=True,
            old_source_law_match_is_development_diagnostic_not_regime_admission=True,
            hard_equation_consistency_and_scale_based_regime_audit_are_separate=True,
            fixed_update_snapshots_not_validation_selected=True, cross_normalizer_validation_loss_comparable=False,
            old_cov_seed0_best_step=inputs["seed-0/cov_base"]["best_step"],
            old_pilot_generation_is_validation_best_not_all_exact_20000=True,
            new_learning_curve_20000_is_exact_last_EMA_native_export=True,
            maximum_training_gpu_seconds=6*arguments.train_seconds, user_strong_gpu_hours_cap=8,
            no_new_pde_solves=True, blind_test=False, admitted=False)
        shared.write_json(output / "frozen-plan.json", plan)
        plan_sha = shared.file_sha(output / "frozen-plan.json")
        frozen[str(output / "frozen-plan.json")] = plan_sha
        def guard():
            if time.monotonic()-started >= arguments.maximum_wall_seconds-arguments.budget_receipt_margin_seconds:
                raise TimeoutError("strong eight-GPU-hour wall allocation exhausted; preserve partial artifacts")
            previous.assert_hashes({**frozen, **frozen_models})
        torch.set_num_threads(min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        stage("train_only_new_rho_preflight")
        result["preflight"] = preflight(arguments, spec, manifest)
        if not arguments.preflight_only:
            stage("source_encoding_constraint")
            constraint = previous.audit_value_control_corpus(corpus, batch_size=8, tolerance=1.e-5)
            shared.write_json(output / "value-slope-corpus-audit.json", constraint)
            previous.require_value_control_audit(constraint)
            training, models, frozen_models = train_models(arguments, spec, manifest, inputs, plan_sha, stage, guard)
            result["training"] = training
            result["training_wall_seconds"] = sum(report["wall_seconds"] for report in training.values())
            guard()
            stage("frozen_reference_feature_loading")
            audit_started = time.monotonic()
            result["audits"] = evaluate_models(arguments, spec, manifest, models, design, constraint, stage, guard)
            result["audit_wall_seconds"] = time.monotonic()-audit_started
        guard()
        if previous.read_manifest(corpus, verify_hashes=True)["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("original corpus changed during experiment")
        result.update(execution_passed=True, preflight_only=arguments.preflight_only, wall_seconds=time.monotonic()-started,
            allocated_gpu_wall_hours=(time.monotonic()-started)/3600., maximum_gpu_hours=8,
            hard_wall_enforcement="Slurm_single_GPU_8h_limit_with_driver_stage_boundary_guard",
            generated_cases_per_state=128, requested_generation_count=(0 if arguments.preflight_only else 14*128))
        shared.write_json(output / "stage.json", {"stage": "completed", "status": "completed"})
        shared.write_json(output / "result.json", result)
        return result
    except Exception as error:
        progress = output / "training-progress.json"
        if progress.exists():
            result["partial_training"] = old.read_json(progress)
        result.update(failed_stage=current_stage, error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc(), wall_seconds=time.monotonic()-started)
        shared.write_json(output / "result.json", result)
        shared.write_json(output / "stage.json", {"stage": current_stage, "status": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
