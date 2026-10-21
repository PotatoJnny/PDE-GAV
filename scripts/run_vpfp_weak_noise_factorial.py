"""Weak VPFP noise energy x temporal shape, with explicitly reused frozen anchors.

Only tau*I and C/tau are newly trained (two seeds, 1800 seconds each).
I and C are hash-bound best checkpoints and generated features from the already
opened weak covariance debug. The eight best models freeze before feature
arrays are opened here. This is development evidence, not independent testing,
not four newly rerun arms, and not a scientific admission decision.
"""
from __future__ import annotations

import argparse
import ast
from copy import copy, deepcopy
from dataclasses import asdict
from hashlib import sha256
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
loader = importlib.util.spec_from_file_location("vpfp_factorial_debug_reuse", ROOT / "scripts/run_vpfp_weak_regime_debug.py")
debug = importlib.util.module_from_spec(loader)
loader.loader.exec_module(debug)
previous, shared, scaleup = debug.previous, debug.shared, debug.scaleup
from pde_agent_verifier_solver.plugins.vpfp.temporal_covariance_diffusion import (
    VPFPTemporalCovarianceNormalizer, VPFPTemporalCovarianceEDM, FACTORIAL_RECIPES, TAU_DEFINITION, _json_sha,
)

NEW_ARMS = ("cov_energy_iso", "cov_unit_shape")
ANCHOR_ARMS = ("cov_centered_iso", "cov_base")
ARMS = ("cov_centered_iso", "cov_energy_iso", "cov_unit_shape", "cov_base")
RECIPES = {"cov_energy_iso": "trace_isotropic", "cov_unit_shape": "unit_trace_shape"}
FACTORIAL_CELLS = {
    "cov_centered_iso": {"energy": "1", "shape": "I", "covariance": "I", "reused_anchor": True},
    "cov_energy_iso": {"energy": "tau_g", "shape": "I", "covariance": "tau_g*I", "reused_anchor": False},
    "cov_unit_shape": {"energy": "1", "shape": "C_g/tau_g", "covariance": "C_g/tau_g", "reused_anchor": False},
    "cov_base": {"energy": "tau_g", "shape": "C_g/tau_g", "covariance": "C_g", "reused_anchor": True},
}
EVIDENCE_ROLE = "development_noise_energy_shape_factorial_with_previously_opened_frozen_anchors"
PAIRED_SETTINGS = ("seed", "replicas", "train_seconds", "steps", "batch_size", "validate_every", "width", "depth",
    "generated_answers", "sample_steps", "permutations", "bootstrap_repeats", "learning_rate",
    "covariance_floor", "constraint_tolerance", "checkpoint_seconds")
TRAINING_BITS = ("seed", "parameters", "precision", "optimizer", "learning_rate", "gradient_clip_norm", "ema",
    "training_noise_distribution", "residual_metric", "common_validation_metric", "train_case_ids", "validation_case_ids",
    "initial_model_sha256", "unique_training_answers", "source_script_sha256")
VPFP_PREFIX = "src/pde_agent_verifier_solver/plugins/vpfp/"


def factories(arguments, spec, arm):
    if arm not in NEW_ARMS:
        raise ValueError("only the two new factorial arms may be trained")
    cfg = previous.VPFPValueEDMConfig.from_full_config(shared.model_config(arguments, spec))
    def make(config):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(arguments.seed + 100)
            return VPFPTemporalCovarianceEDM(config)
    def fit(config, raw, *, sample_ids):
        return VPFPTemporalCovarianceNormalizer.fit(config, raw, sample_ids=sample_ids,
            relative_floor=arguments.covariance_floor, center_kinetic_dc=True, rho_policy="identity",
            covariance_recipe=RECIPES[arm])
    return cfg, make, fit


def preflight(arguments, spec, corpus):
    split = previous.load_split(corpus, "train", verify_hashes=False)
    raw = torch.from_numpy(np.array(split.latents[:8], copy=True))
    conditions = np.array(split.conditions[:arguments.batch_size], copy=True)
    reports = {}
    for arm in NEW_ARMS:
        cfg, make, fit = factories(arguments, spec, arm)
        norm = fit(cfg, raw, sample_ids=split.sample_ids[:len(raw)].tolist()).to(arguments.device)
        model = make(cfg).to(arguments.device)
        batch = raw[:arguments.batch_size].to(arguments.device)
        white = norm.encode(batch)
        if not torch.allclose(norm.decode(white), batch, rtol=2.e-4, atol=2.e-5):
            raise ValueError("factorial preflight full decoder reconstruction failed")
        loss, _ = model.loss(white, conditions)
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("factorial preflight loss/gradient failed")
        generated = norm.decode(model.sample(conditions[:1], [arguments.seed + 9], steps=2))
        if generated.shape != (1, cfg.full_dimension) or not torch.isfinite(generated).all():
            raise ValueError("factorial preflight native sample failed")
        reports[arm] = {"execution_passed": True, "scientific_validation": False, "training_rows_only": True,
            "parameter_count": sum(p.numel() for p in model.parameters()), "normalization": norm.report()}
    return reports


def _legacy_ast(source, *, strip_structure_dispatch=False):
    tree = ast.parse(source)
    if strip_structure_dispatch:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "load_vpfp_diffusion_sampler":
                # The only permitted native-loader extension dispatches the two
                # separate structural schemas, neither used by this experiment.
                node.body = [statement for statement in node.body if not (
                    isinstance(statement, ast.If)
                    and any(isinstance(child, ast.ImportFrom) and child.module == "structure_prior_diffusion"
                        for child in ast.walk(statement)))]
    return ast.dump(tree, include_attributes=False)


def compare_code(arguments, anchor_plan, current):
    """Require identical physical sources and old network/training/sampling bits."""
    old = anchor_plan["source_file_sha256"]
    changed = {key: {"anchor_sha256": digest, "current_sha256": current.get(key)}
        for key, digest in old.items() if current.get(key) != digest}
    allowed = {VPFP_PREFIX + "temporal_covariance_diffusion.py", VPFP_PREFIX + "structured_diffusion.py"}
    if set(changed) - allowed:
        raise ValueError("anchor code changed outside explicit geometry/dispatch extensions: " + str(sorted(set(changed)-allowed)))
    for path in changed:
        old_text = subprocess.check_output(["git", "show", f"{anchor_plan['source_revision']}:{path}"], cwd=ROOT, text=True)
        if sha256(old_text.encode()).hexdigest() != old[path]:
            raise ValueError("anchor git revision does not reproduce its recorded source bytes")
        new_text = (ROOT / path).read_text()
        if path.endswith("structured_diffusion.py"):
            if _legacy_ast(old_text) != _legacy_ast(new_text, strip_structure_dispatch=True):
                raise ValueError("old native model/trainer/sampler semantics changed beyond added structural dispatch")
        else:
            def model_definitions(source):
                return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(source).body
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in (
                        "VPFPTemporalCovarianceEDM", "VPFPLowNoiseTemporalCovarianceEDM", "_sampler_from_temporal_payload",
                        "_check_transform_parent", "_json_sha")}
            if model_definitions(old_text) != model_definitions(new_text):
                raise ValueError("factorial changed legacy temporal model/loss/loader or parent provenance checks")
    return {"physical_training_value_decoder_and_feature_sources_unchanged": True,
        "permitted_geometry_and_unused_dispatch_extensions": changed,
        "new_source_files": sorted(set(current)-set(old)),
        "legacy_model_loss_and_sampler_code_checked": True,
        "anchor_rerun": False}


def freeze_anchors(arguments, manifest, fingerprints):
    anchor = arguments.anchors
    paths = [anchor / name for name in ("frozen-plan.json", "frozen-checkpoints.json", "result.json")]
    frozen = {str(path): shared.file_sha(path) for path in paths}
    plan, registry, result = map(scaleup.read_json, paths)
    if (plan.get("schema") != "vpfp_weak_centered_covariance_strength_debug_v1"
            or plan.get("regime") != "weak" or result.get("execution_passed") is not True
            or result.get("admitted") is not False or plan.get("blind_test") is not False
            or Path(plan["source"]).resolve() != arguments.source
            or plan.get("source_corpus_manifest_sha256") != manifest["manifest_sha256"]
            or registry.get("all_arms_and_snapshots_frozen_before_any_feature_evaluation") is not True):
        raise ValueError("completed frozen weak debug anchors on this exact development corpus required")
    for key in PAIRED_SETTINGS:
        if json.dumps(plan.get(key), sort_keys=True) != json.dumps(getattr(arguments, key), sort_keys=True):
            raise ValueError(f"anchor paired setting differs: {key}")
    if shared.file_sha(arguments.config) != plan["frozen_source_sha256"].get(plan["config"]):
        raise ValueError("anchor model configuration bytes differ")
    # Bind the reference bank to the OLD audit too, not merely to a freshly
    # consistent registry at the same pathname. No arrays are opened here.
    previous.assert_hashes(plan["frozen_source_sha256"])
    frozen.update(plan["frozen_source_sha256"])
    code = compare_code(arguments, plan, fingerprints)
    models, bindings, training = {}, {}, {}
    for replica in range(2):
        for arm in ANCHOR_ARMS:
            key = f"seed-{replica}/{arm}"
            entry = deepcopy(registry["models"][key + "/best"])
            expected_path = anchor / key / "checkpoints/best.pt"
            if (not expected_path.is_file() or expected_path.is_symlink()
                    or Path(entry["checkpoint"]) != expected_path or entry["arm"] != arm or entry["replica"] != replica
                    or entry["checkpoint_role"] != "training_validation_selected_best"
                    or registry["sha256"].get(str(expected_path)) != entry["sha256"]):
                raise ValueError("anchor best checkpoint registry identity mismatch")
            frozen[str(expected_path)] = entry["sha256"]
            summary_path = anchor / key / "training-summary.json"
            frozen[str(summary_path)] = shared.file_sha(summary_path)
            summary = scaleup.read_json(summary_path)
            bound = registry["native_geometry_bindings"][key]
            native, bits = bound["native_sampler"], bound["native_sampler"]["training"]
            if (summary["checkpoint_sha256"] != entry["sha256"] or native["checkpoint_sha256"] != entry["sha256"]
                    or native["corpus_manifest_sha256"] != manifest["manifest_sha256"]
                    or native["base_decoder_sha256"] != manifest["base_decoder_sha256"]
                    or bits["ablation"]["plan_sha256"] != frozen[str(anchor / "frozen-plan.json")]
                    or bits["seed"] != arguments.seed + replica*10000
                    or bits["precision"] != "float32_ieee_no_amp_no_tf32" or bits["residual_metric"] is not None
                    or bits["common_validation_metric"] is not None or bits["unique_training_answers"] != 512
                    or summary["wall_budget_seconds"] != 1800 or summary["audit_used_for_selection"] is not False
                    or summary["initial_model_sha256"] != bits["initial_model_sha256"]
                    or bound["centered"] is not True or bound["covariance_strength"] != debug.STRENGTHS[arm]):
                raise ValueError("anchor native training recipe or corpus identity mismatch")
            entry.update(reused_anchor=True, newly_trained=False, anchor_training_summary_sha256=frozen[str(summary_path)])
            models[key + "/best"], bindings[key], training[key] = entry, bound, summary
            directory = anchor / key / "evaluation/best"
            for filename in ("generated-provenance.json", "generated-conditions.npy",
                    "generated-physical.npy", "generated-physical.json", "physical-distribution-audit.json",
                    "generated-derivative.npy", "generated-derivative.json", "derivative-distribution-audit.json"):
                path = directory / filename
                frozen[str(path)] = shared.file_sha(path)
        pair = [bindings[f"seed-{replica}/{arm}"] for arm in ANCHOR_ARMS]
        if any(pair[0][field] != pair[1][field] for field in ("value_normalizer_sha256", "temporal_transform_sha256")):
            raise ValueError("anchor parent units or fixed center/covariance differ")
        if any(pair[0]["native_sampler"]["training"][field] != pair[1]["native_sampler"]["training"][field] for field in TRAINING_BITS):
            raise ValueError("anchor training bits or initialization differ within seed")
    previous.assert_hashes(frozen)
    return frozen, models, bindings, training, code


def native_geometry_binding(arguments, manifest, checkpoint, digest, arm):
    sampler = previous.load_vpfp_diffusion_sampler(checkpoint, digest, device="cpu", steps=2)
    ids = previous.load_split(arguments.source / "corpus", "train", verify_hashes=False).sample_ids
    shared.assert_checkpoint_corpus(sampler, manifest, ids)
    norm = sampler.normalizer
    if (type(norm) is not VPFPTemporalCovarianceNormalizer or norm.covariance_recipe != RECIPES[arm]
            or norm.covariance_strength != 1 or norm.rho_policy != "identity"
            or norm.temporal.fit_metadata["center_kinetic_dc"] is not True):
        raise ValueError("new checkpoint factorial geometry identity mismatch")
    return {"value_normalizer_sha256": norm.fit_metadata["value_normalizer_sha256"],
        "temporal_transform_sha256": norm.fit_metadata["temporal_transform_sha256"], "centered": True,
        "covariance_recipe": norm.covariance_recipe, "kinetic_tau": norm.fit_metadata["kinetic_tau"],
        "native_sampler": sampler.identity}


def check_paired_model(anchor, current):
    for field in ("value_normalizer_sha256", "temporal_transform_sha256"):
        if anchor[field] != current[field]:
            raise ValueError("factorial parent/center/covariance changed: " + field)
    old, new = anchor["native_sampler"], current["native_sampler"]
    if _json_sha(old["model_config"]) != _json_sha(new["model_config"]):
        raise ValueError("factorial model configuration differs from frozen anchors")
    for field in TRAINING_BITS:
        if old["training"][field] != new["training"][field]:
            raise ValueError("factorial training bits differ from frozen anchors: " + field)


def load_anchor_features(arguments, manifest, entry, schemas, frozen):
    """Load saved small arrays only after all eight best checkpoint hashes freeze.

The old generated coefficient file is intentionally NOT redecoded or rehashed:
its original published SHA and per-answer extraction hashes remain provenance,
while the exact saved arrays/reports/provenance are bound by this run's freeze.
"""
    previous.assert_hashes(frozen)
    key = f"seed-{entry['replica']}/{entry['arm']}"
    directory = arguments.anchors / key / "evaluation/best"
    seed = arguments.seed + entry["replica"]*10000
    provenance = scaleup.read_json(directory / "generated-provenance.json")
    native = provenance["sampler"]
    if (provenance["checkpoint_sha256"] != entry["sha256"] or native["checkpoint_sha256"] != entry["sha256"]
            or native["corpus_manifest_sha256"] != manifest["manifest_sha256"]
            or native["base_decoder_sha256"] != manifest["base_decoder_sha256"]
            or provenance["condition_seed"] != seed+200 or provenance["pde_solves"] != 0
            or provenance["noise_seeds"] != list(range(seed+300000000, seed+300000128))
            or native["sampling"]["steps"] != arguments.sample_steps or native["sampling"]["churn"] != 0):
        raise ValueError("reused generated samples differ from declared checkpoint/noise/condition provenance")
    conditions = np.load(directory / "generated-conditions.npy", allow_pickle=False)
    cfg = previous.VPFPValueEDMConfig.from_dict(native["model_config"])
    expected_conditions = shared.balanced_conditions(cfg, 128, seed+200)
    if not np.array_equal(conditions, expected_conditions) or not np.isfinite(conditions).all():
        raise ValueError("reused generated conditions differ from paired seed stream")
    matrices, reports, answers = {}, {}, {}
    for kind, plan in (("physical", previous.VPFPFeaturePlan()), ("derivative", previous.VPFPDerivativeFeaturePlan())):
        path = directory / f"generated-{kind}.npy"
        info = scaleup.read_json(directory / f"generated-{kind}.json")
        feature = info["physical"]
        answer_ids = tuple(info["answer_sha256"])
        if (info["feature_sha256"] != frozen[str(path)]
                or json.dumps(feature["plan"], sort_keys=True) != json.dumps(asdict(plan), sort_keys=True)
                or info["root_ids"] != [f"generated:{entry['sha256']}:{value}" for value in provenance["noise_seeds"]]
                or len(answer_ids) != 128 or len(set(answer_ids)) != 128
                or answer_ids != tuple(row["input_answer_sha256"] for row in feature["per_case"])
                or not np.array_equal(conditions, np.asarray([row["input_equation_condition"] for row in feature["per_case"]], dtype=np.float32))):
            raise ValueError("reused generated feature identity or physical extraction plan mismatch")
        matrix = np.load(path, allow_pickle=False)
        if matrix.shape != (128, len(schemas[kind])) or not np.isfinite(matrix).all():
            raise ValueError("invalid reused generated feature dimensions or finite values")
        report = scaleup.read_json(directory / f"{kind}-distribution-audit.json")
        if (report["checkpoint_sha256"] != entry["sha256"]
                or report["corpus_manifest_sha256"] != manifest["manifest_sha256"]
                or report["used_for_checkpoint_selection"] is not False or report["independent_confirmation"] is not False
                or report["admitted"] is not False):
            raise ValueError("reused audit checkpoint/corpus or development evidence identity mismatch")
        matrices[kind], reports[kind], answers[kind] = matrix, report, answer_ids
    if answers["physical"] != answers["derivative"]:
        raise ValueError("reused physical and derivative features are not from identical answer snapshots")
    return matrices, reports, conditions


def parser():
    result = debug.parser()
    result.description = __doc__
    result.add_argument("--anchors", type=Path, required=True)
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        cli.error("real factorial training requires a Slurm compute allocation")
    if (args.output.exists() or args.output.is_symlink() or any(args.output.resolve().is_relative_to(path.resolve())
            for path in (args.source, args.anchors))):
        cli.error("choose a fresh output outside immutable source and anchor experiments")
    expected = {"seed": 1320260905, "replicas": 2, "train_seconds": 1800, "steps": 100000, "batch_size": 4,
        "validate_every": 1000, "width": 256, "depth": 4, "generated_answers": 128, "sample_steps": 64,
        "permutations": 99, "bootstrap_repeats": 49, "learning_rate": 2.e-4, "covariance_floor": .05,
        "constraint_tolerance": 1.e-5}
    if any(getattr(args, key) != value for key, value in expected.items()):
        cli.error("factorial settings are fixed to the published two-seed 1800-second anchor recipe")
    args.checkpoint_seconds = shared._checkpoint_seconds(args)
    if args.checkpoint_seconds != (360., 900.):
        cli.error("factorial snapshots are fixed at 360/900 seconds and retained without physical evaluation")
    for key in ("config", "source", "anchors", "output"):
        setattr(args, key, getattr(args, key).resolve())
    args.source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config = scaleup.read_json(args.config)
    spec = previous.VPFPOnlineModelSpec.from_mapping(config.get("vpfp", config))
    spec.preflight()
    corpus = args.source / "corpus"
    manifest = previous.read_manifest(corpus, verify_hashes=True)
    fingerprints = shared.source_fingerprints()
    for name in ("run_vpfp_scaleup_experiment.py", "run_vpfp_temporal_training_pilot.py", "run_vpfp_value_control_pilot.py",
            "run_vpfp_weak_regime_debug.py", Path(__file__).name):
        fingerprints["scripts/"+name] = shared.file_sha(ROOT / "scripts" / name)
    frozen_code = {str(ROOT / path): digest for path, digest in fingerprints.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    started, stage = time.monotonic(), "freeze_source_and_anchors"
    training, models, bindings, summaries = {}, {}, {}, {}
    def set_stage(value):
        nonlocal stage
        stage = value
        shared.write_json(args.output / "stage.json", {"stage": stage, "status": "running"})
    try:
        set_stage(stage)
        frozen = debug.freeze_source(args, spec, manifest)
        anchor_files, models, bindings, anchor_training, code = freeze_anchors(args, manifest, fingerprints)
        frozen.update(anchor_files)
        plan = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        plan.update(schema="vpfp_weak_noise_energy_shape_factorial_v1", model_spec=asdict(spec),
            arms=ARMS, newly_trained_arms=NEW_ARMS, reused_anchor_arms=ANCHOR_ARMS, factorial_cells=FACTORIAL_CELLS,
            covariance_recipes=FACTORIAL_RECIPES, tau_definition=TAU_DEFINITION,
            centering="identical_train_only_kinetic_DC_mean_in_all_four_cells", rho_policy="unchanged_parent_value_coordinates",
            objective="original_EDM_only_no_added_physical_loss", source_corpus_manifest_sha256=manifest["manifest_sha256"],
            split_counts=manifest["split_counts"], source_file_sha256=fingerprints, frozen_source_sha256=frozen,
            source_compatibility=code, paired_settings=list(PAIRED_SETTINGS), initialization="matches_hash_recorded_in_same_seed_anchors",
            arm_order="new_arms_forward_then_reverse", cross_normalizer_validation_loss_comparable=False,
            planned_new_training_models=4, planned_reused_anchor_models=4, planned_generated_answers=512,
            planned_reused_generated_answers=512, planned_combined_answers=1024, nominal_total_training_seconds=7200,
            snapshot_evaluation="new_snapshots_retained_and_hashed_only", all_eight_best_frozen_before_features_open=True,
            anchor_checkpoints_and_feature_bytes_reused_not_rerun=True, evidence_role=EVIDENCE_ROLE,
            new_pde_solves=0, scientific_admission=False, blind_test=False, independent_confirmation=False)
        shared.write_json(args.output / "frozen-plan.json", plan)
        plan_sha = shared.file_sha(args.output / "frozen-plan.json")
        frozen[str(args.output / "frozen-plan.json")] = plan_sha
        previous.assert_hashes({**frozen, **frozen_code})
        set_stage("source_constraint_check")
        constraint = previous.audit_value_control_corpus(corpus, batch_size=8, tolerance=args.constraint_tolerance)
        shared.write_json(args.output / "value-slope-corpus-audit.json", constraint)
        previous.require_value_control_audit(constraint)
        if constraint["source_manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("source corpus changed after factorial plan freeze")
        set_stage("full_layout_preflight")
        shared.write_json(args.output / "allocation-preflight.json", preflight(args, spec, corpus))
        for replica in range(2):
            run = copy(args)
            run.seed += replica*10000
            anchor = bindings[f"seed-{replica}/cov_centered_iso"]
            for arm in NEW_ARMS if replica == 0 else tuple(reversed(NEW_ARMS)):
                key = f"seed-{replica}/{arm}"
                set_stage("train_" + key)
                previous.assert_hashes({**frozen, **frozen_code})
                cfg, make, fit = factories(run, spec, arm)
                # Check network initialization before spending any training budget.
                initial = make(cfg)
                if (_json_sha(cfg.to_dict()) != _json_sha(anchor["native_sampler"]["model_config"])
                        or shared.model_state_sha(initial) != anchor["native_sampler"]["training"]["initial_model_sha256"]):
                    raise ValueError("new factorial model/init does not match same-seed frozen anchors")
                del initial
                run.ablation_metadata = {"plan_sha256": plan_sha, "arm": arm, "replica": replica,
                    "covariance_recipe": RECIPES[arm], "factorial_cell": FACTORIAL_CELLS[arm],
                    "evidence_role": EVIDENCE_ROLE, "anchor_rerun": False,
                    "cross_normalizer_validation_loss_comparable": False}
                directory = args.output / key
                directory.mkdir(parents=True)
                checkpoint, report = shared.train(run, spec, corpus, directory,
                    model_config_factory=lambda *a, cfg=cfg: cfg, model_factory=make, normalizer_factory=fit,
                    training_metric_factory=None, validation_metric_factory=None)
                training[key] = report
                registered = scaleup.register_models(key, replica, arm, directory, checkpoint, report, args.checkpoint_seconds)
                for entry in registered.values():
                    entry.update(reused_anchor=False, newly_trained=True)
                models.update(registered)
                bindings[key] = native_geometry_binding(args, manifest, checkpoint, report["checkpoint_sha256"], arm)
                check_paired_model(anchor, bindings[key])
        frozen_models = {entry["checkpoint"]: entry["sha256"] for entry in models.values()}
        previous.assert_hashes({**frozen, **frozen_code, **frozen_models})
        shared.write_json(args.output / "frozen-checkpoints.json", {
            "all_eight_best_frozen_before_any_feature_evaluation": True, "newly_trained_best_count": 4,
            "reused_anchor_best_count": 4, "sha256": frozen_models, "models": models,
            "native_geometry_bindings": bindings, "snapshots_physically_evaluated": False,
            "frozen_anchor_artifacts_sha256": anchor_files, "evidence_role": EVIDENCE_ROLE})
        set_stage("open_frozen_development_reference_features")
        cached, schemas = debug.load_reference_features(args, manifest, constraint)
        first = models["seed-0/cov_energy_iso/best"]
        sampler = previous.load_vpfp_diffusion_sampler(first["checkpoint"], first["sha256"], device=args.device, steps=args.sample_steps)
        binding = previous.build_vpfp_online_binding(spec, output_root=args.output / "audit-runtime", device=args.device, latent_sampler=sampler)
        decoder = binding.plugin.generator.backend.runtime.base_model
        for replica in range(2):
            run = copy(args)
            run.seed += replica*10000
            run.cache = args.source
            matrices = {role: {kind: cached[kind][role].features for kind in cached} for role in ("reference_a", "reference_b")}
            conditions = {role: cached["physical"][role].conditions for role in ("reference_a", "reference_b")}
            reports = {}
            for arm in ARMS:
                key = f"seed-{replica}/{arm}"
                entry = models[key + "/best"]
                previous.assert_hashes({**frozen, **frozen_code, **frozen_models})
                if arm in ANCHOR_ARMS:
                    set_stage("reuse_frozen_anchor_features_" + key)
                    matrices[arm], reports[arm], conditions[arm] = load_anchor_features(args, manifest, entry, schemas, anchor_files)
                else:
                    set_stage("frozen_development_audit_" + key)
                    directory = args.output / key / "evaluation/best"
                    directory.mkdir(parents=True)
                    matrices[arm], reports[arm] = previous.sample_and_audit(run, spec, Path(entry["checkpoint"]),
                        entry["sha256"], directory, decoder, cached, schemas)
                    conditions[arm] = np.load(directory / "generated-conditions.npy", allow_pickle=False)
                summaries[key] = {"reused_anchor": arm in ANCHOR_ARMS, **{kind: {"status": report["status"],
                    "sw_effect": report["comparisons"]["generated"]["statistics"]["sliced_wasserstein_max"]["effect"]}
                    for kind, report in reports[arm].items()}}
            for kind in schemas:
                nulls = [_json_sha(reports[arm][kind]["comparisons"]["reference_null"]) for arm in ARMS]
                if len(set(nulls)) != 1:
                    raise ValueError("reused and new audits used different reference nulls or audit seed/settings")
            diagnostic = debug.momentum_diagnostics(matrices, schemas["physical"], conditions, spec)
            diagnostic.update(evidence_role=EVIDENCE_ROLE, factorial_cells=FACTORIAL_CELLS,
                anchor_checkpoints_and_features_reused_not_rerun=True)
            shared.write_json(args.output / f"seed-{replica}/spatial-momentum-diagnostic.json", diagnostic)
            card = previous.build_vpfp_direction_scorecard(regime="weak", seed=run.seed,
                objective="fixed_center_noise_energy_shape_original_EDM", baseline_arm="cov_centered_iso", samples=matrices,
                physical_schema=schemas["physical"], derivative_schema=schemas["derivative"], joint_reports=reports,
                physical_time_fractions=previous.VPFPFeaturePlan().time_fractions,
                derivative_time_fractions=previous.VPFPDerivativeFeaturePlan().time_fractions,
                macro_probe_count=previous.VPFPDerivativeFeaturePlan().macro_probe_count,
                phase_probe_count=previous.VPFPDerivativeFeaturePlan().phase_probe_count)
            card.update(evidence_role=EVIDENCE_ROLE, factorial_cells=FACTORIAL_CELLS,
                energy_comparisons=(("cov_centered_iso", "cov_energy_iso"), ("cov_unit_shape", "cov_base")),
                shape_comparisons=(("cov_centered_iso", "cov_unit_shape"), ("cov_energy_iso", "cov_base")),
                heuristic_boolean_used_as_gate=False, automatic_scaleup_or_admission=False,
                anchor_checkpoints_and_features_reused_not_rerun=True, independent_confirmation=False)
            shared.write_json(args.output / f"seed-{replica}/physical-effect-scorecard.json", card)
        previous.assert_hashes({**frozen, **frozen_code, **frozen_models})
        if previous.read_manifest(corpus, verify_hashes=True)["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("source corpus changed during factorial experiment")
        result = {"execution_passed": True, "regime": "weak", "training": training,
            "reused_anchor_training": anchor_training, "audits": summaries, "native_geometry_bindings": bindings,
            "source_compatibility": code, "wall_seconds": time.monotonic()-started, "new_pde_solves": 0,
            "generated_answers": 512, "reused_generated_answers": 512, "combined_answers": 1024,
            "snapshots_physically_evaluated": False, "evidence_role": EVIDENCE_ROLE, "blind_test": False,
            "independent_confirmation": False, "admitted": False, "canonical_campaign_smoke": "separate_required_step",
            "automatic_followup_job_submitted": False}
    except Exception as error:
        result = {"execution_passed": False, "regime": "weak", "failed_stage": stage, "error": repr(error),
            "traceback": traceback.format_exc(), "training": training, "completed_audits": summaries,
            "partial_artifacts_preserved": True, "evidence_role": EVIDENCE_ROLE, "admitted": False,
            "automatic_followup_job_submitted": False}
    shared.write_json(args.output / "result.json", result)
    shared.write_json(args.output / "stage.json", {"stage": stage, "status": "complete" if result["execution_passed"] else "failed"})
    shared.emit("weak_noise_factorial_complete", execution_passed=result["execution_passed"], error=result.get("error"))
    return 0 if result["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
