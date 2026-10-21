"""Weak VPFP: fixed train centering, paired covariance-strength debugging.

Three centered arms differ ONLY by Sigma_alpha=(1-alpha)*I+alpha*C_regularized
in their common parent value coordinates. An uncentered value_base is an anchor,
not a centering-controlled causal comparison. Previously opened scale-up answers
are development data. Eight 1800-second models freeze before reference features
open; elapsed-time snapshots are retained but not physically evaluated here.
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
loader = importlib.util.spec_from_file_location("vpfp_weak_debug_scaleup_reuse", ROOT / "scripts/run_vpfp_scaleup_experiment.py")
scaleup = importlib.util.module_from_spec(loader)
loader.loader.exec_module(scaleup)
previous, shared = scaleup.previous, scaleup.shared

from pde_agent_verifier_solver.plugins.vpfp.temporal_covariance_diffusion import (
    VPFPTemporalCovarianceNormalizer, VPFPTemporalCovarianceEDM, MIXTURE_DEFINITION, _json_sha,
)

ARMS = ("value_base", "cov_centered_iso", "cov_shrink_half", "cov_base")
STRENGTHS = {"cov_centered_iso": 0., "cov_shrink_half": .5, "cov_base": 1.}
EVIDENCE_ROLE = "weak_covariance_strength_debug_on_previously_opened_scaleup_development_references"


def factories(arguments, spec, arm):
    if arm not in ARMS:
        raise ValueError("unknown weak covariance debugging arm")
    cfg = previous.VPFPValueEDMConfig.from_full_config(shared.model_config(arguments, spec))

    def make(config):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(arguments.seed + 100)
            kind = previous.VPFPValueControlEDM if arm == "value_base" else VPFPTemporalCovarianceEDM
            return kind(config)

    def fit(config, full_raw, *, sample_ids):
        if arm == "value_base":
            return previous.VPFPValueControlNormalizer.fit(config, full_raw, sample_ids=sample_ids)
        return VPFPTemporalCovarianceNormalizer.fit(config, full_raw, sample_ids=sample_ids,
            relative_floor=arguments.covariance_floor, center_kinetic_dc=True, rho_policy="identity",
            covariance_strength=STRENGTHS[arm])
    return cfg, make, fit


def preflight(arguments, spec, corpus):
    split = previous.load_split(corpus, "train", verify_hashes=False)
    raw = torch.from_numpy(np.array(split.latents[:8], copy=True))
    conditions = np.array(split.conditions[:arguments.batch_size], copy=True)
    reports = {}
    for arm in ARMS:
        cfg, make, fit = factories(arguments, spec, arm)
        normalizer = fit(cfg, raw, sample_ids=split.sample_ids[:len(raw)].tolist()).to(arguments.device)
        model = make(cfg).to(arguments.device)
        batch = raw[:arguments.batch_size].to(arguments.device)
        white = normalizer.encode(batch)
        if not torch.allclose(normalizer.decode(white), batch, rtol=2.e-4, atol=2.e-5):
            raise ValueError("weak debug affine transform did not reconstruct the full decoder ABI")
        loss, _ = model.loss(white, conditions)
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("weak debug preflight loss/gradient failed")
        sample = normalizer.decode(model.sample(conditions[:1], [arguments.seed + 9], steps=2))
        if sample.shape != (1, cfg.full_dimension) or not torch.isfinite(sample).all():
            raise ValueError("weak debug sampler did not return a finite full decoder vector")
        reports[arm] = {"execution_passed": True, "scientific_validation": False,
            "training_rows_only": True, "full_dimension": cfg.full_dimension,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "normalization": {"class": type(normalizer).__name__, "payload_sha256": _json_sha(normalizer.to_payload()),
                "fit_metadata": normalizer.fit_metadata}}
    return reports


def freeze_source(arguments, spec, manifest):
    """Hash-only reference inventory; do not open feature arrays before training."""
    source = arguments.source
    paths = [arguments.config, source / "corpus/manifest.json", source / "result.json",
        source / "frozen-plan.json", source / "frozen-reference-features.json", source / "fresh-corpus-receipt.json"]
    frozen = {str(path): shared.file_sha(path) for path in paths}
    result = scaleup.read_json(source / "result.json")
    plan = scaleup.read_json(source / "frozen-plan.json")
    receipt = scaleup.read_json(source / "fresh-corpus-receipt.json")
    registry = scaleup.read_json(source / "frozen-reference-features.json")
    if (result.get("execution_passed") is not True or result.get("regime") != "weak"
            or result.get("admitted") is not False
            or plan.get("schema") != "vpfp_paired_scaleup_experiment_v1" or plan.get("regime") != "weak"
            or receipt.get("corpus_manifest_sha256") != manifest["manifest_sha256"]
            or receipt.get("plan_sha256") != frozen[str(source / "frozen-plan.json")]
            or registry.get("corpus_manifest_sha256") != manifest["manifest_sha256"]
            or registry.get("opened_only_after_all_model_hashes_frozen") is not True):
        raise ValueError("a completed weak scale-up with bound corpus and feature receipts is required")
    if (spec != previous.VPFPOnlineModelSpec.from_mapping(manifest["model_spec"])
            or "weak" not in spec.regime_id or manifest["split_counts"] != scaleup.COUNTS):
        raise ValueError("debugging must preserve the weak source specification and 512/128/128/128/128 splits")
    expected = set()
    for kind in ("physical", "derivative"):
        directory = source / "reference-features" / kind
        expected.add(str(directory / "schema.json"))
        expected.update(str(directory / f"{role}.{suffix}") for role in previous.ROLES for suffix in ("npy", "json"))
    if not isinstance(registry.get("sha256"), dict) or set(registry["sha256"]) != expected:
        raise ValueError("source feature registry must bind exactly this source's complete feature bank")
    frozen.update(registry["sha256"])
    previous.assert_hashes(frozen)
    return frozen


def load_reference_features(arguments, manifest, constraint):
    samples, schemas = {}, {}
    for kind, plan in (("physical", previous.VPFPFeaturePlan()), ("derivative", previous.VPFPDerivativeFeaturePlan())):
        directory = arguments.source / "reference-features" / kind
        schema = scaleup.read_json(directory / "schema.json")
        if json.dumps(schema["plan"], sort_keys=True) != json.dumps(asdict(plan), sort_keys=True):
            raise ValueError("saved feature plan differs from this frozen debugging experiment")
        samples[kind], schemas[kind] = {}, tuple(schema["features"])
        for role in previous.ROLES:
            path = directory / f"{role}.npy"
            info = scaleup.read_json(directory / f"{role}.json")
            split = previous.load_split(arguments.source / "corpus", role, verify_hashes=False)
            ids = tuple(f"equation:{int(seed)}" for seed in split.sample_ids)
            conditions = np.asarray(info["conditions"], dtype=np.float32)
            if (info["corpus_manifest_sha256"] != manifest["manifest_sha256"]
                    or info["feature_sha256"] != shared.file_sha(path)
                    or info["answer_sha256"] != constraint["splits"][role]["input_answer_sha256"]
                    or tuple(info["root_ids"]) != ids or not np.array_equal(conditions, split.conditions)):
                raise ValueError("saved feature identity/hash differs from the frozen weak corpus")
            matrix = np.load(path, allow_pickle=False)
            if (matrix.shape != (manifest["split_counts"][role], len(schemas[kind]))
                    or not np.isfinite(matrix).all()):
                raise ValueError("invalid finite saved weak feature matrix")
            samples[kind][role] = previous.AuditSample(matrix, conditions, ids, tuple(info["answer_sha256"]))
    for role in previous.ROLES:
        if samples["physical"][role].answer_sha256 != samples["derivative"][role].answer_sha256:
            raise ValueError("physical and derivative references used different answer snapshots")
    return samples, schemas


def native_geometry_binding(arguments, manifest, checkpoint, digest, arm):
    """Native reload, without sampling, binds the centered causal comparison."""
    sampler = previous.load_vpfp_diffusion_sampler(checkpoint, digest, device="cpu", steps=2)
    ids = previous.load_split(arguments.source / "corpus", "train", verify_hashes=False).sample_ids
    shared.assert_checkpoint_corpus(sampler, manifest, ids)
    normalizer = sampler.normalizer
    if arm == "value_base":
        if type(normalizer) is not previous.VPFPValueControlNormalizer:
            raise ValueError("uncentered anchor must retain its original value normalizer")
        return {"value_normalizer_sha256": _json_sha(normalizer.to_payload()),
            "centered": False, "native_sampler": sampler.identity}
    if (not isinstance(normalizer, VPFPTemporalCovarianceNormalizer)
            or normalizer.covariance_strength != STRENGTHS[arm]
            or normalizer.rho_policy != "identity"
            or normalizer.temporal.fit_metadata["center_kinetic_dc"] is not True):
        raise ValueError("native checkpoint does not represent its declared centered covariance arm")
    return {"value_normalizer_sha256": normalizer.fit_metadata["value_normalizer_sha256"],
        "temporal_transform_sha256": normalizer.fit_metadata["temporal_transform_sha256"],
        "centered": True, "covariance_strength": normalizer.covariance_strength,
        "native_sampler": sampler.identity}


def momentum_diagnostics(matrices, schema, conditions, spec):
    """Separate coherent spatial mean from fluctuation power; no new decoder calls."""
    names = [row["name"] for row in schema]
    report = {"evidence_role": EVIDENCE_ROLE, "admitted": False,
        "time_fraction": 1., "spatial_mean_rms_definition": "sqrt(mean_cases(momentum.time7.mean**2)); NOT all-time RMS",
        "spectral_definition": "saved 17x17 grid, high band kx^2+ky^2>16; not a convergence certificate",
        "descriptive_only": True, "new_scientific_pass_gate": False,
        "roles": {}}
    arrays = {key: value["physical"] for key, value in matrices.items()}
    arrays["reference_pooled"] = np.concatenate([arrays["reference_a"], arrays["reference_b"]])
    conditions = dict(conditions)
    conditions["reference_pooled"] = np.concatenate([conditions["reference_a"], conditions["reference_b"]])
    edges = (np.linspace(*spec.collision_nu_range, 5), np.linspace(*spec.temperature_range, 3))
    report["condition_strata"] = {"nu_edges": edges[0].tolist(), "temperature_edges": edges[1].tolist(),
        "count": 8, "threshold_scope": "pooled_reference_max_over_all_strata"}

    def summarize(matrix):
        fields = {}
        if len(matrix) == 0:
            return fields
        for field in ("momentum_x", "momentum_y", "kinetic_energy"):
            def column(statistic):
                return matrix[:, names.index(f"{field}.time7.{statistic}")]
            high = column("spectrum_high")
            reference_high = arrays["reference_pooled"][:, names.index(f"{field}.time7.spectrum_high")]
            fields[field] = {
                "spatial_mean_case_rms": float(np.sqrt(np.mean(column("mean")**2))),
                "mean_spatial_std": float(column("std").mean()),
                "mean_spectrum_low": float(column("spectrum_low").mean()),
                "mean_spectrum_middle": float(column("spectrum_middle").mean()),
                "mean_spectrum_high": float(high.mean()), "median_spectrum_high": float(np.median(high)),
                "p95_spectrum_high": float(np.quantile(high, .95)),
                "pooled_reference_max_high": float(reference_high.max()),
                "count_above_pooled_reference_max_high": int(np.sum(high > reference_high.max())),
                "fraction_above_pooled_reference_max_high": float(np.mean(high > reference_high.max()))}
        return fields

    for role, matrix in arrays.items():
        if len(conditions[role]) != len(matrix):
            raise ValueError("momentum descriptive conditions and saved feature rows disagree")
        bins = previous._strata(np.asarray(conditions[role]), edges)
        report["roles"][role] = {"count": len(matrix), "fields": summarize(matrix), "strata": [
            {"nu_bin": group // 2, "temperature_bin": group % 2, "count": int(np.sum(bins == group)),
             "fields": summarize(matrix[bins == group])} for group in range(8)]}
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "source", "output"):
        result.add_argument(f"--{name}", type=Path, required=True)
    result.add_argument("--regime", choices=("weak",), required=True)
    result.add_argument("--seed", type=int, required=True)
    for name, default in (("replicas", 2), ("train-seconds", 1800), ("steps", 100000), ("batch-size", 4),
            ("validate-every", 1000), ("width", 256), ("depth", 4), ("generated-answers", 128),
            ("sample-steps", 64), ("permutations", 99), ("bootstrap-repeats", 49)):
        result.add_argument(f"--{name}", type=int, default=default)
    result.add_argument("--checkpoint-seconds", type=float, nargs="+", default=(360., 900.))
    result.add_argument("--learning-rate", type=float, default=2.e-4)
    result.add_argument("--covariance-floor", type=float, default=.05)
    result.add_argument("--constraint-tolerance", type=float, default=1.e-5)
    result.add_argument("--device", default="cuda")
    return result


def main(argv=None):
    cli = parser()
    arguments = cli.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        cli.error("real weak covariance debugging requires a Slurm compute allocation")
    if (arguments.output.exists() or arguments.output.is_symlink()
            or arguments.output.resolve().is_relative_to(arguments.source.resolve())):
        cli.error("choose a fresh output outside the immutable source experiment")
    if (arguments.replicas != 2 or arguments.train_seconds != 1800 or arguments.generated_answers != 128
            or not 1 <= arguments.batch_size <= 8 or arguments.sample_steps < 2
            or any(getattr(arguments, key) < 1 for key in ("steps", "validate_every", "width", "depth"))):
        cli.error("fixed debug needs 2 replicas, 1800 seconds, 128 generated answers, batch<=8 and positive budgets")
    if arguments.seed < 0 or arguments.covariance_floor > 1 or any(
            not np.isfinite(getattr(arguments, key)) or getattr(arguments, key) <= 0
            for key in ("learning_rate", "covariance_floor", "constraint_tolerance")):
        cli.error("invalid finite positive debugging setting")
    try:
        arguments.checkpoint_seconds = shared._checkpoint_seconds(arguments)
    except ValueError as error:
        cli.error(str(error))
    if arguments.checkpoint_seconds != (360., 900.):
        cli.error("debug snapshots are fixed at 360 and 900 seconds and are not physically evaluated")
    previous.DiffusionAuditConfig(permutations=arguments.permutations, bootstrap_repeats=arguments.bootstrap_repeats).validate()
    for name in ("config", "source", "output"):
        setattr(arguments, name, getattr(arguments, name).resolve())
    config = scaleup.read_json(arguments.config)
    spec = previous.VPFPOnlineModelSpec.from_mapping(config.get("vpfp", config))
    spec.preflight()
    corpus = arguments.source / "corpus"
    manifest = previous.read_manifest(corpus, verify_hashes=True)
    frozen = freeze_source(arguments, spec, manifest)
    arguments.source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    fingerprints = shared.source_fingerprints()
    for filename in ("run_vpfp_scaleup_experiment.py", "run_vpfp_temporal_training_pilot.py", "run_vpfp_value_control_pilot.py", Path(__file__).name):
        fingerprints[f"scripts/{filename}"] = shared.file_sha(ROOT / "scripts" / filename)
    frozen_code = {str(ROOT / path): digest for path, digest in fingerprints.items()}
    arguments.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    plan = {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()}
    plan.update(schema="vpfp_weak_centered_covariance_strength_debug_v1", model_spec=asdict(spec),
        arms=ARMS, covariance_strengths=STRENGTHS, covariance_definition=MIXTURE_DEFINITION,
        centering="same_train_only_kinetic_DC_mean_for_all_three_centered_arms",
        uncentered_anchor="value_base_not_a_centering_controlled_comparison", causal_baseline="cov_centered_iso",
        rho_policy="unchanged_parent_value_coordinates", objective="original_EDM_only_no_added_physical_loss",
        source_corpus_manifest_sha256=manifest["manifest_sha256"], split_counts=manifest["split_counts"],
        frozen_source_sha256=frozen, source_file_sha256=fingerprints,
        initialization="same_network_within_seed", arm_order="forward_then_reverse",
        cross_normalizer_validation_loss_comparable=False, evidence_role=EVIDENCE_ROLE,
        new_pde_solves=0, planned_generated_answers=1024, nominal_total_training_seconds=14400,
        snapshot_evaluation="retained_and_hashed_only_no_generated_samples_or_physical_audits",
        all_best_selected_before_reference_features_open=True, scientific_admission=False, blind_test=False)
    shared.write_json(arguments.output / "frozen-plan.json", plan)
    plan_sha = shared.file_sha(arguments.output / "frozen-plan.json")
    frozen[str(arguments.output / "frozen-plan.json")] = plan_sha
    started, stage = time.monotonic(), "source_constraint_check"
    training, models, bindings, summaries = {}, {}, {}, {}

    def set_stage(value):
        nonlocal stage
        stage = value
        shared.write_json(arguments.output / "stage.json", {"stage": stage, "status": "running"})

    try:
        set_stage(stage)
        previous.assert_hashes({**frozen, **frozen_code})
        constraint = previous.audit_value_control_corpus(corpus, batch_size=8, tolerance=arguments.constraint_tolerance)
        shared.write_json(arguments.output / "value-slope-corpus-audit.json", constraint)
        previous.require_value_control_audit(constraint)
        if constraint["source_manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("source corpus changed after plan freeze")
        set_stage("full_layout_preflight")
        shared.write_json(arguments.output / "allocation-preflight.json", preflight(arguments, spec, corpus))
        for replica in range(2):
            run_args = copy(arguments)
            run_args.seed += replica * 10000
            for arm in ARMS if replica == 0 else tuple(reversed(ARMS)):
                key = f"seed-{replica}/{arm}"
                set_stage("train_" + key)
                previous.assert_hashes({**frozen, **frozen_code})
                cfg, make, fit = factories(run_args, spec, arm)
                run_args.ablation_metadata = {"plan_sha256": plan_sha, "arm": arm, "replica": replica,
                    "covariance_strength": STRENGTHS.get(arm), "evidence_role": EVIDENCE_ROLE,
                    "uncentered_anchor": arm == "value_base", "cross_normalizer_validation_loss_comparable": False}
                directory = arguments.output / key
                directory.mkdir(parents=True)
                checkpoint, report = shared.train(run_args, spec, corpus, directory,
                    model_config_factory=lambda *a, cfg=cfg: cfg, model_factory=make, normalizer_factory=fit,
                    training_metric_factory=None, validation_metric_factory=None)
                training[key] = report
                models.update(scaleup.register_models(key, replica, arm, directory, checkpoint, report, arguments.checkpoint_seconds))
                bindings[key] = native_geometry_binding(arguments, manifest, checkpoint, report["checkpoint_sha256"], arm)
            paired = [training[f"seed-{replica}/{arm}"] for arm in ARMS]
            if len({entry["initial_model_sha256"] for entry in paired}) != 1:
                raise ValueError("four-arm initial network parameter identities differ")
            bound = [bindings[f"seed-{replica}/{arm}"] for arm in ARMS]
            if len({entry["value_normalizer_sha256"] for entry in bound}) != 1:
                raise ValueError("four arms used different parent value units")
            if len({entry["temporal_transform_sha256"] for entry in bound[1:]}) != 1:
                raise ValueError("centered arms used different training means or covariance estimates")
        frozen_models = {entry["checkpoint"]: entry["sha256"] for entry in models.values()}
        previous.assert_hashes({**frozen, **frozen_code, **frozen_models})
        shared.write_json(arguments.output / "frozen-checkpoints.json", {
            "all_arms_and_snapshots_frozen_before_any_feature_evaluation": True, "sha256": frozen_models,
            "models": models, "native_geometry_bindings": bindings, "snapshots_physically_evaluated": False})
        set_stage("open_frozen_development_reference_features")
        cached, schemas = load_reference_features(arguments, manifest, constraint)
        first = models["seed-0/value_base/best"]
        sampler = previous.load_vpfp_diffusion_sampler(first["checkpoint"], first["sha256"], device=arguments.device, steps=arguments.sample_steps)
        binding = previous.build_vpfp_online_binding(spec, output_root=arguments.output / "audit-runtime", device=arguments.device, latent_sampler=sampler)
        decoder = binding.plugin.generator.backend.runtime.base_model
        for replica in range(2):
            run_args = copy(arguments)
            run_args.seed += replica * 10000
            run_args.cache = arguments.source
            matrices = {role: {kind: cached[kind][role].features for kind in cached} for role in ("reference_a", "reference_b")}
            conditions = {role: cached["physical"][role].conditions for role in ("reference_a", "reference_b")}
            reports = {}
            for arm in ARMS:
                key = f"seed-{replica}/{arm}"
                set_stage("frozen_development_audit_" + key)
                previous.assert_hashes({**frozen, **frozen_code, **frozen_models})
                entry = models[key + "/best"]
                directory = arguments.output / key / "evaluation/best"
                directory.mkdir(parents=True)
                matrices[arm], reports[arm] = previous.sample_and_audit(run_args, spec, Path(entry["checkpoint"]),
                    entry["sha256"], directory, decoder, cached, schemas)
                conditions[arm] = np.load(directory / "generated-conditions.npy", allow_pickle=False)
                summaries[key] = {kind: {"status": report["status"],
                    "sw_effect": report["comparisons"]["generated"]["statistics"]["sliced_wasserstein_max"]["effect"]}
                    for kind, report in reports[arm].items()}
            shared.write_json(arguments.output / f"seed-{replica}/spatial-momentum-diagnostic.json",
                momentum_diagnostics(matrices, schemas["physical"], conditions, spec))
            causal = {key: value for key, value in matrices.items() if key != "value_base"}
            card = previous.build_vpfp_direction_scorecard(regime="weak", seed=run_args.seed,
                objective="fixed_center_covariance_strength_original_EDM", baseline_arm="cov_centered_iso", samples=causal,
                physical_schema=schemas["physical"], derivative_schema=schemas["derivative"],
                joint_reports={arm: reports[arm] for arm in STRENGTHS},
                physical_time_fractions=previous.VPFPFeaturePlan().time_fractions,
                derivative_time_fractions=previous.VPFPDerivativeFeaturePlan().time_fractions,
                macro_probe_count=previous.VPFPDerivativeFeaturePlan().macro_probe_count,
                phase_probe_count=previous.VPFPDerivativeFeaturePlan().phase_probe_count)
            card.update(evidence_role=EVIDENCE_ROLE, uncentered_anchor_excluded_from_causal_comparison="value_base",
                heuristic_boolean_used_as_gate=False, automatic_scaleup_or_admission=False)
            shared.write_json(arguments.output / f"seed-{replica}/physical-effect-scorecard.json", card)
        previous.assert_hashes({**frozen, **frozen_code, **frozen_models})
        if previous.read_manifest(corpus, verify_hashes=True)["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("source corpus changed during debugging")
        result = {"execution_passed": True, "regime": "weak", "training": training, "audits": summaries,
            "native_geometry_bindings": bindings, "wall_seconds": time.monotonic() - started,
            "new_pde_solves": 0, "generated_answers": 1024, "snapshots_physically_evaluated": False,
            "all_learning_curve_thresholds_reached": all(not row["unreached_checkpoint_seconds"] for row in training.values()),
            "evidence_role": EVIDENCE_ROLE, "blind_test": False, "admitted": False,
            "canonical_campaign_smoke": "separate_required_step", "automatic_followup_job_submitted": False}
    except Exception as error:
        result = {"execution_passed": False, "regime": "weak", "failed_stage": stage, "error": repr(error),
            "traceback": traceback.format_exc(), "training": training, "completed_audits": summaries,
            "partial_artifacts_preserved": True, "evidence_role": EVIDENCE_ROLE, "admitted": False,
            "automatic_followup_job_submitted": False}
    shared.write_json(arguments.output / "result.json", result)
    shared.write_json(arguments.output / "stage.json", {"stage": stage, "status": "complete" if result["execution_passed"] else "failed"})
    shared.emit("weak_regime_debug_complete", execution_passed=result["execution_passed"], error=result.get("error"))
    return 0 if result["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
