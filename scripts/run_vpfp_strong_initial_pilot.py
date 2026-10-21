"""Equal-update strong-collision known-initial/future pilot on opened data.

Two seeds compare the original full-covariance value EDM with known analytic
initial-prior projection plus genuinely conditional future EDM. This changes
the model package and adds 31,232 parameters; it is NOT an equal-parameter
causal isolation. Cached references are development data, not a fresh test.
"""
from __future__ import annotations

import argparse
from copy import copy, deepcopy
from dataclasses import asdict, replace
import gc
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
loader = importlib.util.spec_from_file_location("strong_initial_temporal_reuse", ROOT / "scripts/run_vpfp_temporal_training_pilot.py")
temporal = importlib.util.module_from_spec(loader)
loader.loader.exec_module(temporal)
previous, shared = temporal.previous, temporal.shared
from pde_agent_verifier_solver.plugins.vpfp.initial_conditioned_diffusion import (
    VPFPInitialConditionedEDMConfig, VPFPInitialConditionedEDM,
    VPFPInitialConditionedNormalizer, VPFPInitialConditionedBatchSampler)
from pde_agent_verifier_solver.plugins.vpfp.known_initial_prior import KnownInitialPriorSpec
from pde_agent_verifier_solver.plugins.vpfp.numerical_roundtrip import audit_normalizer_roundtrip
from pde_agent_verifier_solver.plugins.vpfp.diffusion_data import _source_exclusion_seeds

ARMS = ("cov_base", "initial_conditioned")
COUNTS = dict(zip(previous.ROLES, (2048, 128, 128, 128, 128)))
EVIDENCE_ROLE = "development_comparison_on_previously_opened_strong_main_references"
UPDATE_COUNT = 20000
SCRATCH = Path("${PDE_GAV_RUN_ROOT}")


def read_json(path):
    payload = json.loads(Path(path).read_text())
    json.dumps(payload, allow_nan=False)
    return payload


def common_trunk_sha(model):
    digest = sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith("initial_input."):
            continue
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def factories(arguments, spec, manifest, arm, initialization):
    if arm not in ARMS:
        raise ValueError("unknown strong initial-prior pilot arm")
    if arm == "cov_base":
        config, base_make, fit = temporal.factories(arguments, spec, "cov_base")
    else:
        config = VPFPInitialConditionedEDMConfig(shared.model_config(arguments, spec))
        initial_prior = KnownInitialPriorSpec.from_manifest(manifest)
        def base_make(config):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(arguments.seed + 100)
                return VPFPInitialConditionedEDM(config)
        def fit(config, raw, *, sample_ids):
            return VPFPInitialConditionedNormalizer.fit(config, raw, sample_ids=sample_ids,
                initial_prior=initial_prior, rho_policy="identity", center_kinetic_dc=True,
                covariance_strength=1., relative_floor=.05)
    def make(config):
        model = base_make(config)
        initialization.update(common_trunk_sha256=common_trunk_sha(model),
            parameter_count=sum(parameter.numel() for parameter in model.parameters()))
        return model
    return config, make, fit


def reference_inventory(source, manifest):
    path = source / "frozen-reference-features.json"
    receipt = read_json(path)
    if receipt["corpus_manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("cached references belong to another strong corpus")
    expected_paths = {str(source / "reference-features" / kind / name) for kind in ("physical", "derivative")
        for name in ["schema.json", *(f"{role}.{suffix}" for role in previous.ROLES for suffix in ("npy", "json"))]}
    if set(receipt["sha256"]) != expected_paths:
        raise ValueError("strong cached-reference receipt has unexpected paths")
    previous.assert_hashes(receipt["sha256"])
    return {str(path): shared.file_sha(path), **receipt["sha256"]}


def check_baseline_recipe(source, manifest):
    plan = read_json(source / "frozen-plan.json")
    expected = {"covariance_floor": .05, "covariance_rho_policy": "identity", "covariance_center": "training_kinetic_DC_mean",
        "width": 256, "depth": 4, "batch_size": 4, "learning_rate": 2.e-4}
    if any(plan.get(key) != value for key, value in expected.items()) or plan.get("model_spec") != manifest["model_spec"]:
        raise ValueError("source cov baseline is not the expected centered full covariance recipe")
    return {**expected, "covariance_strength": 1., "covariance_recipe": "mixture",
            "source_plan_sha256": shared.file_sha(source / "frozen-plan.json"),
            "interpretation": "retrain_original_recipe_from_scratch_not_resume_old_weights"}


def check_completed_source(source):
    path = source / "result.json"
    result = read_json(path)
    if result.get("execution_passed") is not True or result.get("regime") != "strong":
        raise ValueError("source result must be a completed strong-collision experiment")
    return {"execution_passed": True, "regime": "strong", "result_sha256": shared.file_sha(path)}


def generation_design(arguments, manifest):
    old_plan = read_json(arguments.source / "corpus" / manifest["seed_plan"]["path"])
    excluded = set(_source_exclusion_seeds(old_plan))
    excluded.update(old_plan.get("excluded_pilot_equation_seeds", []))
    designs, seen_prior = {}, set()
    for replica in range(2):
        seed = arguments.seed + replica * 10000
        conditions = shared.balanced_conditions(shared.model_config(arguments,
            previous.VPFPOnlineModelSpec.from_mapping(manifest["model_spec"])), 128, seed + 200)
        seeds = list(range(seed + 300000000, seed + 300000128))
        prior_seeds = [int.from_bytes(sha256(f"vpfp-initial-conditioned-prior:{value}".encode()).digest()[:8], "little") % 2**63 for value in seeds]
        if excluded.intersection(prior_seeds) or seen_prior.intersection(prior_seeds) or len(set(prior_seeds)) != len(prior_seeds):
            raise ValueError("fresh known-initial seed overlaps source history or another generated case")
        seen_prior.update(prior_seeds)
        designs[f"seed-{replica}"] = {"conditions": conditions.tolist(), "sampling_seeds": seeds,
            "known_initial_seeds": prior_seeds, "initial_seeds_exclude_source_and_embedded_history": True}
    return designs


def preflight(arguments, spec, manifest, *, arms=None):
    arms = ARMS if arms is None else tuple(arms)
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError("preflight arms must be a nonempty distinct subset of the existing arms")
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
        numerical = audit_normalizer_roundtrip(normalizer, raw[:4].to(restored), restored)
        shared.write_json(arguments.output / f"preflight-{arm}-normalization.json", {"arm": arm, **numerical})
        if not numerical["passed"]:
            raise ValueError(f"strong pilot {arm} numerical normalization gate {numerical['status']}; see per-arm preflight report")
        loss, _ = model.loss(encoded, conditions)
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("strong pilot full-layout forward/backward failed")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for target, source in zip(ema.parameters(), model.parameters()):
                target.lerp_(source, .5)  # Exact shared-trainer step-one warmup.
        validation_report = shared.validation_report(ema, normalizer, validation,
            batch_size=4, seed=arguments.seed + 103)
        if not np.isfinite(validation_report["selection_score"]):
            raise ValueError("strong pilot EMA validation failed")
        if arm == "initial_conditioned":
            sampler = VPFPInitialConditionedBatchSampler(ema, normalizer,
                base_decoder_sha256=spec.base_decoder_sha256, checkpoint_sha256="0"*64, steps=2)
            sampled = sampler.sample_conditions(conditions[:1], [arguments.seed + 90])
            del sampler
        else:
            sampled = normalizer.decode(ema.sample(conditions[:1], [arguments.seed + 90], steps=2))
        if sampled.shape != (1, config.full_dimension) or not torch.isfinite(sampled).all():
            raise ValueError("strong pilot native full-D generation preflight failed")
        reports[arm] = {"execution_passed": True, **initialization, "full_dimension": config.full_dimension,
            "random_dimension": config.latent_dimension, "new_pde_solves": 0, "normalization_gate": numerical,
            "adamw_and_ema_update_passed": True, "ema_validation": validation_report,
            "peak_cuda_allocated_bytes": (torch.cuda.max_memory_allocated(arguments.device)
                if torch.device(arguments.device).type == "cuda" else None),
            "peak_cuda_reserved_bytes": (torch.cuda.max_memory_reserved(arguments.device)
                if torch.device(arguments.device).type == "cuda" else None),
            "scientific_validation": False, "data": "first_eight_training_cases_only_mechanics_not_training"}
        del model, ema, optimizer, normalizer, encoded, restored, loss, sampled, source, target
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if all(arm in reports for arm in ARMS) and reports[ARMS[0]]["common_trunk_sha256"] != reports[ARMS[1]]["common_trunk_sha256"]:
        raise ValueError("preflight common trunk initializations differ")
    return reports


def load_references(arguments, manifest, constraint):
    samples, schemas = {}, {}
    for kind in ("physical", "derivative"):
        directory = arguments.source / "reference-features" / kind
        schema = read_json(directory / "schema.json")
        expected_plan = asdict(previous.VPFPFeaturePlan() if kind == "physical" else previous.VPFPDerivativeFeaturePlan())
        if json.dumps(schema["plan"], sort_keys=True) != json.dumps(expected_plan, sort_keys=True):
            raise ValueError("cached feature plan differs from the current frozen plan")
        samples[kind], schemas[kind] = {}, tuple(schema["features"])
        for role in previous.ROLES:
            info = read_json(directory / f"{role}.json")
            split = previous.load_split(arguments.source / "corpus", role, verify_hashes=False)
            ids = tuple(f"equation:{int(seed)}" for seed in split.sample_ids)
            if (info["corpus_manifest_sha256"] != manifest["manifest_sha256"]
                    or info["answer_sha256"] != constraint["splits"][role]["input_answer_sha256"]
                    or info["feature_sha256"] != shared.file_sha(directory / f"{role}.npy")
                    or tuple(info["root_ids"]) != ids
                    or not np.array_equal(np.asarray(info["conditions"], dtype=np.float32), split.conditions)):
                raise ValueError("cached physical feature identities differ from scanned source answers")
            matrix = np.load(directory / f"{role}.npy", allow_pickle=False)
            if matrix.shape != (len(split), len(schemas[kind])) or not np.isfinite(matrix).all():
                raise ValueError("cached physical features have invalid finite shape")
            samples[kind][role] = previous.AuditSample(matrix, np.array(split.conditions, copy=True), ids, tuple(info["answer_sha256"]))
    return samples, schemas


def sample_and_audit(arguments, spec, manifest, entry, directory, decoder, samples, schemas, design):
    sampler = previous.load_vpfp_diffusion_sampler(entry["checkpoint"], entry["sha256"], device=arguments.device, steps=64)
    shared.assert_checkpoint_corpus(sampler, manifest,
        previous.load_split(arguments.source / "corpus", "train", verify_hashes=False).sample_ids)
    conditions, seeds = np.asarray(design["conditions"], dtype=np.float32), tuple(design["sampling_seeds"])
    raw = np.lib.format.open_memmap(directory / "generated-coefficients.npy", mode="w+", dtype="float32",
        shape=(128, sampler.identity["latent_dimension"]))
    for index, (condition, seed) in enumerate(zip(conditions, seeds)):
        if isinstance(sampler, VPFPInitialConditionedBatchSampler):
            value = sampler.sample_conditions(condition[None], [seed])
        else:
            value = sampler.normalizer.decode(sampler.model.sample(condition[None], [seed], steps=64))
        raw[index] = value[0].detach().cpu().numpy()
        if (index+1) % 16 == 0:
            raw.flush()
            shared.emit("strong_initial_generated", arm=entry["arm"], completed=index+1, total=128)
    raw.flush()
    np.save(directory / "generated-conditions.npy", conditions, allow_pickle=False)
    coefficient_sha = shared.file_sha(directory / "generated-coefficients.npy")
    ids = tuple(f"generated:{entry['sha256']}:{seed}" for seed in seeds)
    matrices, reports, snapshots = {}, {}, {}
    for kind, extractor, plan in (("physical", previous.extract_vpfp_features, previous.VPFPFeaturePlan()),
            ("derivative", previous.extract_vpfp_derivative_features, previous.VPFPDerivativeFeaturePlan())):
        matrix, descriptors, physical = extractor(decoder, raw, conditions, plan=plan)
        if tuple(descriptors) != schemas[kind]:
            raise ValueError("generated feature schema differs from cached references")
        hashes = tuple(row["input_answer_sha256"] for row in physical["per_case"])
        theta = np.asarray([row["input_equation_condition"] for row in physical["per_case"]], dtype=np.float32)
        if not np.array_equal(theta, conditions):
            raise ValueError("physical extraction changed generated conditions")
        snapshots[kind], matrices[kind] = hashes, matrix
        np.save(directory / f"generated-{kind}.npy", matrix, allow_pickle=False)
        shared.write_json(directory / f"generated-{kind}.json", {"physical": physical, "root_ids": ids,
            "answer_sha256": hashes, "feature_sha256": shared.file_sha(directory / f"generated-{kind}.npy")})
        generated = previous.AuditSample(matrix, theta, ids, hashes)
        report = previous.audit_diffusion_distribution({**samples[kind], "generated": generated},
            features=tuple(previous.AuditFeature(**item) for item in schemas[kind]),
            nu_edges=np.linspace(*spec.collision_nu_range, 5), temperature_edges=np.linspace(*spec.temperature_range, 3),
            config=previous.DiffusionAuditConfig(permutations=99, bootstrap_repeats=49,
                seed=arguments.seed + (1400 if kind == "derivative" else 400)))
        report.update(checkpoint_sha256=entry["sha256"], corpus_manifest_sha256=manifest["manifest_sha256"],
            evidence_role=EVIDENCE_ROLE, used_for_checkpoint_selection=False, independent_confirmation=False,
            blind_test=False, admitted=False, unobserved_source_pairwise_matching=False)
        reports[kind] = report
        shared.write_json(directory / f"{kind}-distribution-audit.json", report)
    if (snapshots["physical"] != snapshots["derivative"]
            or shared.file_sha(directory / "generated-coefficients.npy") != coefficient_sha
            or shared.file_sha(entry["checkpoint"]) != entry["sha256"]):
        raise ValueError("frozen generated answer or model changed during audits")
    shared.write_json(directory / "generated-provenance.json", {"checkpoint_sha256": entry["sha256"],
        "coefficient_sha256": coefficient_sha, "sampler": sampler.identity, "design": design,
        "known_initial_seed_applied": entry["arm"] == "initial_conditioned", "new_pde_solves": 0,
        "evidence_role": EVIDENCE_ROLE, "source_law": "marginalized_unobserved_random_source_not_pairwise_target"})
    return matrices, reports


def train_models(arguments, spec, manifest, plan_sha, stage, guard):
    training, models, initializations = {}, {}, {}
    for replica in range(2):
        run_args = copy(arguments)
        run_args.seed = arguments.seed + replica * 10000
        for arm in ARMS if replica == 0 else ARMS[::-1]:
            key = f"seed-{replica}/{arm}"
            stage("train_" + key)
            guard()
            initialization = {}
            config, make, fit = factories(run_args, spec, manifest, arm, initialization)
            run_args.ablation_metadata = {"plan_sha256": plan_sha, "arm": arm, "replica": replica,
                "equal_update_budget": UPDATE_COUNT, "cross_normalizer_validation_loss_comparable": False,
                "equal_parameter_causal_comparison": False, "evidence_role": EVIDENCE_ROLE}
            directory = arguments.output / key
            directory.mkdir(parents=True)
            checkpoint, report = shared.train(run_args, spec, arguments.source / "corpus", directory,
                model_config_factory=lambda *args, cfg=config: cfg, model_factory=make, normalizer_factory=fit)
            training[key], initializations[key] = report, initialization
            models[key] = {"replica": replica, "arm": arm, "checkpoint": str(checkpoint),
                "sha256": report["checkpoint_sha256"], "best_step": report["best_step"], **initialization}
            shared.write_json(arguments.output / "training-progress.json", {"training": training, "models": models,
                "physical_features_opened": False, "status": "not_all_models_frozen"})
            if report["final_step"] != UPDATE_COUNT or report["stop_reason"] != "step_limit":
                raise RuntimeError(f"{key} did not complete {UPDATE_COUNT} updates; paired comparison rejected")
        if initializations[f"seed-{replica}/{ARMS[0]}"]["common_trunk_sha256"] != initializations[f"seed-{replica}/{ARMS[1]}"]["common_trunk_sha256"]:
            raise ValueError("paired common trunk initializations differ")
    for arm in ARMS:
        if training[f"seed-0/{arm}"]["normalizer_sha256"] != training[f"seed-1/{arm}"]["normalizer_sha256"]:
            raise ValueError("same recipe fitted different normalization between replicas")
    frozen = {entry["checkpoint"]: entry["sha256"] for entry in models.values()}
    previous.assert_hashes(frozen)
    shared.write_json(arguments.output / "frozen-checkpoints.json", {"models": models, "sha256": frozen,
        "all_models_frozen_before_feature_reading": True, "all_models_completed_updates": UPDATE_COUNT,
        "best_selection": "validation_only_within_each_arm_not_across_different_coordinates"})
    return training, models, frozen


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "source", "output"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    cli.add_argument("--seed", type=int, default=1620260906)
    cli.add_argument("--device", choices=("cuda",), default="cuda")
    cli.set_defaults(regime="strong", steps=UPDATE_COUNT, train_seconds=2400, batch_size=4,
        width=256, depth=4, learning_rate=2.e-4, validate_every=1000, covariance_floor=.05,
        checkpoint_seconds=(), generated_answers=128, sample_steps=64, permutations=99, bootstrap_repeats=49)
    return cli


def main(argv=None):
    cli = parser()
    arguments = cli.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID", "").isdigit() or not os.environ.get("SLURM_TMPDIR"):
        cli.error("strong initial pilot requires a Slurm compute allocation and local scratch")
    output = arguments.output.resolve()
    if (arguments.seed < 0 or arguments.seed + 300030128 >= 2**63
            or arguments.output.exists() or arguments.output.is_symlink() or output.parent != SCRATCH.resolve()
            or output.name != f"vpfp-strong-initial-{os.environ['SLURM_JOB_ID']}" or output.is_relative_to(arguments.source.resolve())):
        cli.error("choose valid seed and fresh exact job-specific strong-initial scratch output")
    if not torch.cuda.is_available():
        cli.error("a usable CUDA device is required")
    arguments.output, arguments.source, arguments.config = output, arguments.source.resolve(), arguments.config.resolve()
    arguments.source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    output.mkdir(parents=True)
    current_stage, started = "source_preflight", time.monotonic()
    result = {"execution_passed": False, "training": {}, "audits": {}, "evidence_role": EVIDENCE_ROLE,
        "new_pde_solves": 0, "admitted": False, "automatic_large_training": False}
    frozen_models = {}
    def stage(name):
        nonlocal current_stage
        current_stage = name
        shared.write_json(output / "stage.json", {"stage": name, "status": "running"})
    try:
        spec = previous.VPFPOnlineModelSpec.from_mapping(read_json(arguments.config)["vpfp"])
        spec.preflight()
        corpus = arguments.source / "corpus"
        manifest = previous.read_manifest(corpus, verify_hashes=True)
        if (manifest["split_counts"] != COUNTS or spec != previous.VPFPOnlineModelSpec.from_mapping(manifest["model_spec"])
                or "strong" not in spec.regime_id):
            raise ValueError("pilot requires the exact strong-main 2048/128x4 source corpus")
        baseline = check_baseline_recipe(arguments.source, manifest)
        source_result = check_completed_source(arguments.source)
        frozen = reference_inventory(arguments.source, manifest)
        paths = [arguments.config, corpus / "manifest.json", corpus / manifest["seed_plan"]["path"],
                 arguments.source / "frozen-plan.json", arguments.source / "result.json", Path(__file__), ROOT / "scripts/run_vpfp_strong_initial.slurm",
                 ROOT / "scripts/run_vpfp_temporal_training_pilot.py", ROOT / "scripts/run_vpfp_value_control_pilot.py"]
        frozen.update({str(path): shared.file_sha(path) for path in paths})
        frozen.update({str(ROOT / name): digest for name, digest in shared.source_fingerprints().items()})
        design = generation_design(arguments, manifest)
        plan = {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()}
        plan.update(schema="vpfp_strong_known_initial_equal_update_pilot_v1", arms=ARMS, replicas=2,
            source_corpus_manifest_sha256=manifest["manifest_sha256"], frozen_input_sha256=frozen,
            baseline_recipe=baseline, completed_source=source_result, generation_design=design, new_pde_solves=0,
            same_common_trunk_initialization=True, extra_initial_conditioning_parameters=31232,
            equal_parameter_causal_comparison=False, compared_package="known_prior_full_spatial_conditioning_and_future_Schur_coordinates",
            shared_coordinate_noise=False,
            noise_comparison="baseline_33_time_values_vs_conditional_32_future_values_with_domain_hashed_inference_future_seed;_same_sigma_law_not_same_noise_realization",
            initial_prior="original_native_projection_of_known_analytic_Gaussian_packet_law",
            initial_prior_lossless_analytic_fidelity=False, cross_normalizer_validation_loss_comparable=False,
            best_selection="validation_only_within_each_arm", best_step_need_not_equal_between_arms=True,
            paired_run_requires_all_arms_complete_20000_updates=True, source_is_network_condition=False,
            source_pairwise_error_target=False, evidence_role=EVIDENCE_ROLE, blind_test=False,
            scientific_distribution_qualification=False, automatic_followup_submission=False)
        shared.write_json(output / "frozen-plan.json", plan)
        plan_sha = shared.file_sha(output / "frozen-plan.json")
        frozen[str(output / "frozen-plan.json")] = plan_sha
        def guard():
            previous.assert_hashes({**frozen, **frozen_models})
        torch.set_num_threads(min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        stage("source_encoding_constraint")
        constraint = previous.audit_value_control_corpus(corpus, batch_size=8, tolerance=1.e-5)
        shared.write_json(output / "value-slope-corpus-audit.json", constraint)
        previous.require_value_control_audit(constraint)
        if constraint["source_manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("source encoding scan belongs to another corpus")
        stage("full_layout_preflight")
        shared.write_json(output / "allocation-preflight.json", preflight(arguments, spec, manifest))
        training, models, frozen_models = train_models(arguments, spec, manifest, plan_sha, stage, guard)
        result["training"] = training
        guard()
        stage("frozen_cached_reference_loading")
        samples, schemas = load_references(arguments, manifest, constraint)
        first = models["seed-0/cov_base"]
        sampler = previous.load_vpfp_diffusion_sampler(first["checkpoint"], first["sha256"], device="cuda", steps=64)
        binding = previous.build_vpfp_online_binding(spec, output_root=output / "audit-runtime", device="cuda", latent_sampler=sampler)
        decoder = binding.plugin.generator.backend.runtime.base_model
        for replica in range(2):
            matrices = {role: {kind: samples[kind][role].features for kind in samples} for role in ("reference_a", "reference_b")}
            reports = {}
            run_args = copy(arguments)
            run_args.seed = arguments.seed + replica * 10000
            for arm in ARMS:
                key = f"seed-{replica}/{arm}"
                stage("development_audit_" + key)
                guard()
                directory = output / key / "evaluation"
                directory.mkdir()
                matrices[arm], reports[arm] = sample_and_audit(run_args, spec, manifest, models[key], directory,
                    decoder, samples, schemas, design[f"seed-{replica}"])
                result["audits"][key] = {kind: {"status": report["status"],
                    "sw_effect": report["comparisons"]["generated"]["statistics"]["sliced_wasserstein_max"]["effect"]}
                    for kind, report in reports[arm].items()}
                shared.write_json(output / "audit-progress.json", result["audits"])
            card = previous.build_vpfp_direction_scorecard(regime="strong", seed=run_args.seed,
                objective="known_initial_prior_and_true_conditional_future_EDM_package", baseline_arm="cov_base", samples=matrices,
                physical_schema=schemas["physical"], derivative_schema=schemas["derivative"], joint_reports=reports,
                physical_time_fractions=previous.VPFPFeaturePlan().time_fractions,
                derivative_time_fractions=previous.VPFPDerivativeFeaturePlan().time_fractions,
                macro_probe_count=previous.VPFPDerivativeFeaturePlan().macro_probe_count,
                phase_probe_count=previous.VPFPDerivativeFeaturePlan().phase_probe_count)
            card.update(evidence_role=EVIDENCE_ROLE, used_for_checkpoint_selection=False, automatic_scaleup_or_admission=False,
                heuristic_boolean_used_as_gate=False, equal_parameter_causal_comparison=False)
            shared.write_json(output / f"seed-{replica}/physical-effect-scorecard.json", card)
        guard()
        if previous.read_manifest(corpus, verify_hashes=True)["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("source corpus changed during the completed pilot")
        result.update(execution_passed=True, all_models_completed_updates=UPDATE_COUNT, plan_sha256=plan_sha)
    except Exception as error:
        result["failure"] = {"stage": current_stage, "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        # Shared trainer already preserves partial model/history artifacts.
        if (output / "training-progress.json").is_file():
            result["training"] = read_json(output / "training-progress.json")["training"]
        shared.write_json(output / "result.json", result)
        shared.write_json(output / "stage.json", {"stage": current_stage, "status": "failed", "failure": result["failure"]})
        raise
    result["wall_seconds"] = time.monotonic() - started
    shared.write_json(output / "result.json", result)
    shared.write_json(output / "stage.json", {"stage": "complete", "status": "completed"})
    return result


if __name__ == "__main__":
    main()
