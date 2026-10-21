"""Paired weak VPFP architecture pilot, at fixed half covariance and original EDM.

Research only: same native trainer/sampler/decoder/SDK lifecycle. All six best
checkpoints freeze before opened development references or denoising diagnostics.
"""
from __future__ import annotations

from copy import copy
from dataclasses import asdict
import importlib.util
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.util.spec_from_file_location(
    "vpfp_structure_weak_reuse", ROOT / "scripts/run_vpfp_weak_regime_debug.py")
weak = importlib.util.module_from_spec(loader)
loader.loader.exec_module(weak)
previous, shared, scaleup = weak.previous, weak.shared, weak.scaleup

from pde_agent_verifier_solver.plugins.vpfp.structure_prior_diffusion import VPFPWeakStructurePriorEDM
from pde_agent_verifier_solver.plugins.vpfp.temporal_covariance_diffusion import (
    VPFPTemporalCovarianceNormalizer, VPFPTemporalCovarianceEDM, _json_sha,
)

ARMS = ("cov_shrink_half", "branch_identity_control", "lowmode_coherent")
EVIDENCE_ROLE = "weak_architecture_pilot_on_previously_opened_development_references"


def factories(arguments, spec, arm):
    if arm not in ARMS:
        raise ValueError("unknown weak structure arm")
    cfg = previous.VPFPValueEDMConfig.from_full_config(shared.model_config(arguments, spec))

    def make(config):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(arguments.seed + 100)
            return (VPFPTemporalCovarianceEDM(config) if arm == ARMS[0]
                    else VPFPWeakStructurePriorEDM(config, branch_kind=arm))

    def fit(config, full_raw, *, sample_ids):
        return VPFPTemporalCovarianceNormalizer.fit(config, full_raw, sample_ids=sample_ids,
            relative_floor=arguments.covariance_floor, center_kinetic_dc=True,
            rho_policy="identity", covariance_strength=.5)
    return cfg, make, fit


def require_matched_parameters(models):
    """Compare learned parameters, not the deliberately different fixed buffers."""
    base = dict(models[ARMS[0]].named_parameters())
    control = dict(models[ARMS[1]].named_parameters())
    prior = dict(models[ARMS[2]].named_parameters())
    if set(control) != set(prior) or any(not torch.equal(control[k], prior[k]) for k in control):
        raise ValueError("prior/control initial learned parameters differ")
    if not set(base).issubset(control) or any(not torch.equal(base[k], control[k]) for k in base):
        raise ValueError("auxiliary branch changed original trunk initialization")
    if len(control) <= len(base):
        raise ValueError("expected an auxiliary architecture branch")
    return {"same_trunk_initial_parameters": True, "same_auxiliary_initial_parameters": True,
            "fixed_projector_intentionally_differs": True,
            "parameter_counts": {arm: sum(p.numel() for p in model.parameters())
                                 for arm, model in models.items()}}


def preflight(arguments, spec, corpus):
    split = previous.load_split(corpus, "train", verify_hashes=False)
    raw = torch.from_numpy(np.array(split.latents[:8], copy=True))
    reports = {}
    for replica in range(arguments.replicas):
        run_args = copy(arguments)
        run_args.seed += 10000 * replica
        models = {arm: factories(run_args, spec, arm)[1](
            factories(run_args, spec, arm)[0]) for arm in ARMS}
        paired = require_matched_parameters(models)
        cfg, _, fit = factories(run_args, spec, ARMS[0])
        normalizer = fit(cfg, raw, sample_ids=split.sample_ids[:len(raw)].tolist()).to(arguments.device)
        batch = raw[:arguments.batch_size].to(arguments.device)
        white = normalizer.encode(batch)
        if not torch.allclose(normalizer.decode(white), batch, rtol=2.e-4, atol=2.e-5):
            raise ValueError("half normalizer failed full ABI reconstruction")
        conditions = np.array(split.conditions[:len(batch)], copy=True)
        arm_reports = {}
        for arm, model in models.items():
            model.to(arguments.device)
            loss, _ = model.loss(white, conditions)
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                              for p in model.parameters()):
                raise FloatingPointError("architecture preflight loss/gradient failed")
            model.eval()
            sample = normalizer.decode(model.sample(conditions[:1], [run_args.seed + 9], steps=2))
            if sample.shape != (1, cfg.full_dimension) or not torch.isfinite(sample).all():
                raise ValueError("architecture did not return finite full decoder vector")
            arm_reports[arm] = (model.architecture_report() if hasattr(model, "architecture_report")
                                else {"architecture": "unchanged_full_support_trunk"})
            model.to("cpu")
        reports[f"seed-{replica}"] = {**paired, "arms": arm_reports, "execution_passed": True,
            "training_rows_only": True, "scientific_validation": False}
    return reports


def native_binding(arguments, manifest, entry, arm):
    sampler = previous.load_vpfp_diffusion_sampler(entry["checkpoint"], entry["sha256"], device="cpu", steps=2)
    ids = previous.load_split(arguments.source / "corpus", "train", verify_hashes=False).sample_ids
    shared.assert_checkpoint_corpus(sampler, manifest, ids)
    norm = sampler.normalizer
    if (not isinstance(norm, VPFPTemporalCovarianceNormalizer)
            or norm.covariance_strength != .5 or norm.rho_policy != "identity"
            or norm.temporal.fit_metadata["center_kinetic_dc"] is not True):
        raise ValueError("architecture pilot changed fixed centered half geometry")
    model = sampler.model
    if arm == ARMS[0]:
        if type(model) is not VPFPTemporalCovarianceEDM:
            raise ValueError("baseline architecture mismatch")
    elif not isinstance(model, VPFPWeakStructurePriorEDM) or model.branch_kind != arm:
        raise ValueError("native architecture mismatch")
    return {"normalizer_sha256": _json_sha(norm.to_payload()), "native_sampler": sampler.identity,
            "architecture": model.architecture_report() if hasattr(model, "architecture_report") else ARMS[0]}


def load_diagnostic_module():
    loader = importlib.util.spec_from_file_location("vpfp_structure_diagnostic",
        ROOT / "scripts/diagnose_vpfp_weak_structure_prior.py")
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def main(argv=None):
    cli = weak.parser()
    cli.description = __doc__
    args = cli.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        cli.error("real architecture pilot requires a Slurm compute allocation")
    for name in ("config", "source", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists() or args.output.is_symlink() or args.output.is_relative_to(args.source):
        cli.error("choose a fresh output outside immutable source")
    if (args.replicas != 2 or args.train_seconds != 1800 or args.generated_answers != 128
            or args.width != 256 or args.depth != 4 or args.seed != 1320260905
            or args.steps != 100000 or args.validate_every != 1000 or args.batch_size != 4
            or args.sample_steps != 64 or args.covariance_floor != .05 or args.learning_rate != 2.e-4
            or args.device != "cuda"):
        cli.error("this frozen pilot fixes two seeds, 3 arms, 1800 s, half geometry and standard EDM settings")
    args.checkpoint_seconds = shared._checkpoint_seconds(args)
    if args.checkpoint_seconds != (360., 900.):
        cli.error("retain fixed 360/900 second snapshots; no snapshot physical selection")
    if not np.isfinite(args.constraint_tolerance) or args.constraint_tolerance <= 0:
        cli.error("constraint tolerance must be positive finite")
    previous.DiffusionAuditConfig(permutations=args.permutations, bootstrap_repeats=args.bootstrap_repeats).validate()
    spec_config = scaleup.read_json(args.config)
    spec = previous.VPFPOnlineModelSpec.from_mapping(spec_config.get("vpfp", spec_config))
    spec.preflight()
    corpus = args.source / "corpus"
    manifest = previous.read_manifest(corpus, verify_hashes=True)
    frozen = weak.freeze_source(args, spec, manifest)
    args.source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    fingerprints = shared.source_fingerprints()
    for name in (Path(__file__).name, "run_vpfp_weak_regime_debug.py", "run_vpfp_scaleup_experiment.py",
                 "run_vpfp_temporal_training_pilot.py", "run_vpfp_value_control_pilot.py",
                 "diagnose_vpfp_weak_structure_prior.py"):
        fingerprints[f"scripts/{name}"] = shared.file_sha(ROOT / "scripts" / name)
    frozen_code = {str(ROOT / name): digest for name, digest in fingerprints.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    plan = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    plan.update(schema="vpfp_weak_structure_prior_pilot_v1", model_spec=asdict(spec), arms=ARMS,
        source_corpus_manifest_sha256=manifest["manifest_sha256"], frozen_source_sha256=frozen,
        source_file_sha256=fingerprints, split_counts=manifest["split_counts"],
        geometry="same_train_centered_half_covariance_all_arms", objective="original_EDM_only",
        projector="axis_box_abs_kx_le_2_abs_ky_le_2_in_auxiliary_branch_only",
        full_trunk_support_preserved=True, hermite_channels="all_120_jointly_learned_amplitudes_not_f_moments",
        matched_control="identical_auxiliary_parameters_operations_with_identity_projector",
        branch_width=64, branch_depth=4, arm_order="forward_then_reverse",
        time_budget_comparison="paired_auxiliary_arms_match_parameters_and_operations; record_actual_updates",
        best_selection="fixed_same_coordinate_validation_only_before_any_physical_features",
        low_sigma_diagnostic="8_validation_cases_stratified; .03/.1/.3; after_all_checkpoint_hashes_frozen",
        snapshot_evaluation="hash_and_retain_only_not_physically_evaluated",
        evidence_role=EVIDENCE_ROLE, blind_test=False, scientific_admission=False,
        planned_generated_answers=768, nominal_total_training_seconds=10800, new_pde_solves=0)
    shared.write_json(args.output / "frozen-plan.json", plan)
    frozen[str(args.output / "frozen-plan.json")] = plan_sha = shared.file_sha(args.output / "frozen-plan.json")
    started, stage = time.monotonic(), "source_constraint_check"
    training, models, bindings, summaries = {}, {}, {}, {}

    def set_stage(value):
        nonlocal stage
        stage = value
        shared.write_json(args.output / "stage.json", {"stage": stage, "status": "running"})

    try:
        set_stage(stage)
        previous.assert_hashes({**frozen, **frozen_code})
        constraint = previous.audit_value_control_corpus(corpus, batch_size=8, tolerance=args.constraint_tolerance)
        previous.require_value_control_audit(constraint)
        shared.write_json(args.output / "value-slope-corpus-audit.json", constraint)
        if constraint["source_manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("source corpus changed")
        set_stage("paired_architecture_preflight")
        shared.write_json(args.output / "allocation-preflight.json", preflight(args, spec, corpus))
        for replica in range(2):
            run_args = copy(args)
            run_args.seed += 10000 * replica
            for arm in ARMS if replica == 0 else tuple(reversed(ARMS)):
                key = f"seed-{replica}/{arm}"
                set_stage("train_" + key)
                previous.assert_hashes({**frozen, **frozen_code})
                cfg, make, fit = factories(run_args, spec, arm)
                run_args.ablation_metadata = {"plan_sha256": plan_sha, "arm": arm, "replica": replica,
                    "covariance_strength": .5, "evidence_role": EVIDENCE_ROLE,
                    "same_coordinate_validation_loss_comparable": True}
                directory = args.output / key
                directory.mkdir(parents=True)
                checkpoint, report = shared.train(run_args, spec, corpus, directory,
                    model_config_factory=lambda *a, cfg=cfg: cfg, model_factory=make, normalizer_factory=fit)
                training[key] = report
                models.update(scaleup.register_models(key, replica, arm, directory, checkpoint, report, args.checkpoint_seconds))
                bindings[key] = native_binding(args, manifest, models[key + "/best"], arm)
        if len({entry["normalizer_sha256"] for entry in bindings.values()}) != 1:
            raise ValueError("architecture experiment normalizers are not identical")
        frozen_models = {entry["checkpoint"]: entry["sha256"] for entry in models.values()}
        previous.assert_hashes({**frozen, **frozen_code, **frozen_models})
        shared.write_json(args.output / "frozen-checkpoints.json", {
            "all_arms_and_snapshots_frozen_before_any_feature_evaluation": True, "sha256": frozen_models,
            "models": models, "native_geometry_bindings": bindings, "snapshots_physically_evaluated": False})
        set_stage("open_frozen_development_reference_features")
        cached, schemas = weak.load_reference_features(args, manifest, constraint)
        first = models[f"seed-0/{ARMS[0]}/best"]
        sampler = previous.load_vpfp_diffusion_sampler(first["checkpoint"], first["sha256"], device=args.device, steps=args.sample_steps)
        binding = previous.build_vpfp_online_binding(spec, output_root=args.output / "audit-runtime",
            device=args.device, latent_sampler=sampler)
        decoder = binding.plugin.generator.backend.runtime.base_model
        diagnostic = load_diagnostic_module()
        for replica in range(2):
            run_args = copy(args)
            run_args.seed += 10000 * replica
            run_args.cache = args.source
            matrices = {role: {kind: cached[kind][role].features for kind in cached}
                        for role in ("reference_a", "reference_b")}
            conditions = {role: cached["physical"][role].conditions for role in matrices}
            for arm in ARMS:
                key = f"seed-{replica}/{arm}"
                set_stage("frozen_development_audit_" + key)
                previous.assert_hashes({**frozen, **frozen_code, **frozen_models})
                entry = models[key + "/best"]
                directory = args.output / key / "evaluation/best"
                directory.mkdir(parents=True)
                matrices[arm], reports = previous.sample_and_audit(run_args, spec, Path(entry["checkpoint"]),
                    entry["sha256"], directory, decoder, cached, schemas)
                conditions[arm] = np.load(directory / "generated-conditions.npy", allow_pickle=False)
                summaries[key] = {kind: {"status": report["status"],
                    "statistics": report["comparisons"]["generated"]["statistics"]} for kind, report in reports.items()}
            momentum = weak.momentum_diagnostics(matrices, schemas["physical"], conditions, spec)
            momentum["evidence_role"] = EVIDENCE_ROLE
            shared.write_json(args.output / f"seed-{replica}/spatial-momentum-diagnostic.json", momentum)
            set_stage(f"post_freeze_low_sigma_diagnostic_seed-{replica}")
            denoising_report = diagnostic.run_structure_denoising_diagnostic(
                {arm: models[f"seed-{replica}/{arm}/best"] for arm in ARMS}, corpus,
                args.output / f"seed-{replica}/low-sigma-denoising-diagnostic",
                seed=run_args.seed, device=args.device, sample_count=8, batch_size=2,
                expected_corpus_sha256=manifest["manifest_sha256"])
            if denoising_report.get("execution_passed") is not True:
                raise RuntimeError("post-freeze denoising diagnostic failed: " + str(denoising_report.get("error")))
        previous.assert_hashes({**frozen, **frozen_code, **frozen_models})
        if previous.read_manifest(corpus, verify_hashes=True)["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("source corpus changed during experiment")
        result = {"execution_passed": True, "regime": "weak", "training": training, "audits": summaries,
            "native_geometry_bindings": bindings, "wall_seconds": time.monotonic() - started,
            "new_pde_solves": 0, "generated_answers": 768, "snapshots_physically_evaluated": False,
            "all_learning_curve_thresholds_reached": all(not r["unreached_checkpoint_seconds"] for r in training.values()),
            "evidence_role": EVIDENCE_ROLE, "blind_test": False, "admitted": False,
            "canonical_campaign_smoke": "separate_required_step", "automatic_followup_job_submitted": False}
    except Exception as error:
        result = {"execution_passed": False, "regime": "weak", "failed_stage": stage, "error": repr(error),
            "traceback": traceback.format_exc(), "training": training, "completed_audits": summaries,
            "partial_artifacts_preserved": True, "evidence_role": EVIDENCE_ROLE, "admitted": False,
            "automatic_followup_job_submitted": False}
    shared.write_json(args.output / "result.json", result)
    shared.write_json(args.output / "stage.json", {"stage": stage, "status": "complete" if result["execution_passed"] else "failed"})
    shared.emit("weak_structure_prior_complete", execution_passed=result["execution_passed"], error=result.get("error"))
    return 0 if result["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
