"""Small VPFP representation x time-Sobolev development screen.

The already-opened reference corpus is explicitly DEVELOPMENT evidence here.
No online PDE collection, automatic large training, or scientific admission.
Two training seeds per regime; all selected checkpoints freeze before samples
or feature comparisons are opened. A successful screen requests fresh blind
confirmation; it does not itself authorize scientific claims.
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
loader = importlib.util.spec_from_file_location("vpfp_value_pilot_shared", ROOT / "scripts/train_vpfp_structured_diffusion.py")
shared = importlib.util.module_from_spec(loader)
loader.loader.exec_module(shared)

from pde_agent_verifier_solver.plugins.vpfp.canonical_binding import VPFPOnlineModelSpec, build_vpfp_online_binding
from pde_agent_verifier_solver.plugins.vpfp.diffusion_data import read_manifest, load_split
from pde_agent_verifier_solver.plugins.vpfp.diffusion_features import VPFPFeaturePlan, extract_vpfp_features
from pde_agent_verifier_solver.plugins.vpfp.derivative_features import VPFPDerivativeFeaturePlan, extract_vpfp_derivative_features
from pde_agent_verifier_solver.plugins.vpfp.structured_diffusion import (
    VPFPStructuredEDM, VPFPCoefficientNormalizer, load_vpfp_diffusion_sampler,
)
from pde_agent_verifier_solver.plugins.vpfp.value_control_diffusion import (
    VPFPValueEDMConfig, VPFPValueControlEDM, VPFPValueControlNormalizer, VPFPValueSobolevMetric,
)
from pde_agent_verifier_solver.plugins.vpfp.sobolev_matching import VPFPSobolevMetric
from pde_agent_verifier_solver.plugins.vpfp.value_control_audit import audit_value_control_corpus, require_value_control_audit
from pde_agent_verifier_solver.plugins.vpfp.diffusion_audit import (
    AuditFeature, AuditSample, DiffusionAuditConfig, audit_diffusion_distribution, _strata,
)
from pde_agent_verifier_solver.plugins.vpfp.direction_scorecard import (
    build_vpfp_direction_scorecard, aggregate_vpfp_direction_scorecards,
)

ROLES = ("train", "validation", "calibration", "reference_a", "reference_b")
ARMS = ("full_base", "full_time", "value_base", "value_time")


def cache_directories(cache):
    return {"physical": cache / "baseline/physical-features", "derivative": cache / "derivative-features"}


def cache_inventory(cache):
    paths = []
    for directory in cache_directories(cache).values():
        paths.append(directory / "schema.json")
        paths.extend(directory / f"{role}.{suffix}" for role in ROLES for suffix in ("npy", "json"))
    paths.extend(cache / "baseline" / name for name in ("distribution-audit.json", "derivative-distribution-audit.json"))
    return {str(path): shared.file_sha(path) for path in paths}


def assert_hashes(frozen):
    pass


def fixed_reference_rows(corpus, spec, count):
    """Select balanced rows by metadata only, before opening any features."""
    edges = (np.linspace(*spec.collision_nu_range, 5), np.linspace(*spec.temperature_range, 3))
    selected = {}
    for role in ("reference_a", "reference_b"):
        split = load_split(corpus, role, verify_hashes=False)
        bins = _strata(split.conditions, edges)
        rows = [np.flatnonzero(bins == group)[:count // 8] for group in range(8)]
        if any(len(row) != count // 8 for row in rows):
            raise ValueError("cached references do not have enough cases in every joint stratum")
        selected[role] = np.concatenate(rows).tolist()
    return selected


def load_cached_features(cache, manifest, rows, constraint):
    """Load immutable historical features as labeled development diagnostics."""
    all_samples, schemas = {}, {}
    for kind, directory in cache_directories(cache).items():
        report_name = "distribution-audit.json" if kind == "physical" else "derivative-distribution-audit.json"
        old = json.loads((cache / "baseline" / report_name).read_text())
        if old["corpus_manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("cached features belong to a different source corpus")
        schema = json.loads((directory / "schema.json").read_text())
        expected_plan = asdict(VPFPFeaturePlan() if kind == "physical" else VPFPDerivativeFeaturePlan())
        if json.dumps(schema["plan"], sort_keys=True) != json.dumps(expected_plan, sort_keys=True):
            raise ValueError("cached feature plan differs from this frozen pilot")
        schemas[kind], all_samples[kind] = tuple(schema["features"]), {}
        for role in ROLES:
            matrix = np.load(directory / f"{role}.npy", allow_pickle=False)
            info = json.loads((directory / f"{role}.json").read_text())
            if info["answer_sha256"] != constraint["splits"][role]["input_answer_sha256"]:
                raise ValueError("cached answer snapshots differ from the newly scanned corpus bytes")
            key = "features_sha256" if kind == "physical" else "feature_sha256"
            split = load_split(cache / "corpus", role, verify_hashes=False)
            ids = tuple(f"equation:{int(seed)}" for seed in split.sample_ids)
            if tuple(info["root_ids"]) != ids or not np.array_equal(np.asarray(info["conditions"], dtype=np.float32), split.conditions):
                raise ValueError("cached feature identities differ from the corpus split")
            indices = np.asarray(rows.get(role, list(range(len(matrix)))), dtype=np.int64)
            all_samples[kind][role] = AuditSample(matrix[indices], np.asarray(info["conditions"], dtype=np.float32)[indices],
                tuple(ids[index] for index in indices), tuple(info["answer_sha256"][index] for index in indices))
    for role in ROLES:
        if all_samples["physical"][role].answer_sha256 != all_samples["derivative"][role].answer_sha256:
            raise ValueError("cached physical and derivative features used different answer snapshots")
    return all_samples, schemas


def factories(arguments, spec, representation, decoder_config):
    full_config = shared.model_config(arguments, spec)
    config = full_config if representation == "full" else VPFPValueEDMConfig.from_full_config(full_config)
    model_type = VPFPStructuredEDM if representation == "full" else VPFPValueControlEDM
    normalizer_type = VPFPCoefficientNormalizer if representation == "full" else VPFPValueControlNormalizer
    metric_type = VPFPSobolevMetric if representation == "full" else VPFPValueSobolevMetric

    def model_factory(cfg):
        # CPU-only construction: share every shape-compatible trainable tensor
        # across representations. Different input/output dimensions are real
        # design changes and cannot have identical initialization identities.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(arguments.seed + 100)
            template = VPFPStructuredEDM(full_config)
        if representation == "full":
            return template
        model = model_type(cfg)
        template_parameters = dict(template.named_parameters())
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in template_parameters and parameter.shape == template_parameters[name].shape:
                    parameter.copy_(template_parameters[name])
        return model

    def metric_factory(cfg, normalizer):
        return metric_type(cfg, normalizer, spatial_weight=0., temporal_weight=arguments.temporal_weight,
            spatial_lengths=decoder_config["equation"]["x_domain"],
            latent_input_scale=decoder_config["latent_input_scale"],
            structured_local_scale=decoder_config["structured_local_scale"]).to(arguments.device)
    return config, model_factory, normalizer_type.fit, metric_factory


def full_layout_preflight(arguments, spec, corpus, decoder_config):
    rows = load_split(corpus, "train", verify_hashes=False)
    raw = torch.from_numpy(np.array(rows.latents[:max(8, arguments.batch_size)], copy=True))
    output = {}
    for representation in ("full", "value"):
        cfg, make_model, fit, metric_factory = factories(arguments, spec, representation, decoder_config)
        normalizer = fit(cfg, raw, sample_ids=rows.sample_ids[:len(raw)].tolist()).to(arguments.device)
        model = make_model(cfg).to(arguments.device)
        metric = metric_factory(cfg, normalizer)
        loss, _ = model.loss(normalizer.encode(raw[:arguments.batch_size].to(arguments.device)),
            np.array(rows.conditions[:arguments.batch_size], copy=True), residual_metric=metric)
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("direction pilot preflight gradient failed")
        sampled = normalizer.decode(model.sample(np.array(rows.conditions[:1], copy=True), [arguments.seed + 9], steps=2))
        if sampled.shape != (1, spec_dimension(corpus)) or not torch.isfinite(sampled).all():
            raise ValueError("pilot preflight did not return a finite complete decoder vector")
        output[representation] = {"execution_passed": True, "random_dimension": cfg.latent_dimension,
            "full_dimension": sampled.shape[1], "parameters": sum(p.numel() for p in model.parameters()),
            "metric": metric.report(), "scientific_qualification": False, "training_rows_only": True}
    return output


def spec_dimension(corpus):
    return read_manifest(corpus, verify_hashes=False)["latent_dimension"]


def sample_and_audit(arguments, spec, checkpoint, expected_sha, directory, decoder, cached, schemas):
    sampler = load_vpfp_diffusion_sampler(checkpoint, expected_sha, device=arguments.device, steps=arguments.sample_steps)
    manifest = read_manifest(arguments.cache / "corpus", verify_hashes=False)
    shared.assert_checkpoint_corpus(sampler, manifest, load_split(arguments.cache / "corpus", "train", verify_hashes=False).sample_ids)
    conditions = shared.balanced_conditions(sampler.model.config, arguments.generated_answers, arguments.seed + 200)
    seeds = tuple(range(arguments.seed + 300000000, arguments.seed + 300000000 + arguments.generated_answers))
    raw = np.lib.format.open_memmap(directory / "generated-coefficients.npy", mode="w+", dtype="float32",
        shape=(len(seeds), sampler.identity["latent_dimension"]))
    for index, (condition, seed) in enumerate(zip(conditions, seeds)):
        raw[index] = sampler.normalizer.decode(sampler.model.sample(condition[None], [seed], steps=arguments.sample_steps))[0].cpu().numpy()
        if (index + 1) % 16 == 0:
            raw.flush()
            shared.emit("pilot_sample", arm=directory.name, count=index + 1, total=len(seeds), pde_solves=0)
    raw.flush()
    np.save(directory / "generated-conditions.npy", conditions, allow_pickle=False)
    coefficient_sha = shared.file_sha(directory / "generated-coefficients.npy")
    ids = tuple(f"generated:{expected_sha}:{seed}" for seed in seeds)
    reports, matrices, snapshots = {}, {}, {}
    for kind, extractor, plan in (("physical", extract_vpfp_features, VPFPFeaturePlan()),
                                  ("derivative", extract_vpfp_derivative_features, VPFPDerivativeFeaturePlan())):
        shared.emit("pilot_features", arm=directory.name, kind=kind, count=len(raw))
        matrix, descriptor, physical = extractor(decoder, raw, conditions, plan=plan)
        if tuple(descriptor) != schemas[kind]:
            raise ValueError("pilot generated feature schema differs from cached references")
        hashes = tuple(row["input_answer_sha256"] for row in physical["per_case"])
        evaluated = np.asarray([row["input_equation_condition"] for row in physical["per_case"]], dtype=np.float32)
        sample = AuditSample(matrix, evaluated, ids, hashes)
        matrices[kind], snapshots[kind] = matrix, hashes
        np.save(directory / f"generated-{kind}.npy", matrix, allow_pickle=False)
        shared.write_json(directory / f"generated-{kind}.json", {"physical": physical, "root_ids": ids,
            "answer_sha256": hashes, "feature_sha256": shared.file_sha(directory / f"generated-{kind}.npy")})
        report = audit_diffusion_distribution({**cached[kind], "generated": sample},
            features=tuple(AuditFeature(**entry) for entry in schemas[kind]),
            nu_edges=np.linspace(*spec.collision_nu_range, 5), temperature_edges=np.linspace(*spec.temperature_range, 3),
            config=DiffusionAuditConfig(permutations=arguments.permutations, bootstrap_repeats=arguments.bootstrap_repeats,
                                        seed=arguments.seed + (1400 if kind == "derivative" else 400)))
        report.update(checkpoint_sha256=expected_sha, corpus_manifest_sha256=manifest["manifest_sha256"],
            evidence_role="development_screen_on_previously_opened_reference", used_for_checkpoint_selection=False,
            independent_confirmation=False, admitted=False)
        shared.write_json(directory / f"{kind}-distribution-audit.json", report)
        reports[kind] = report
    if snapshots["physical"] != snapshots["derivative"] or shared.file_sha(directory / "generated-coefficients.npy") != coefficient_sha:
        raise ValueError("physical and derivative extraction did not preserve identical generated answers")
    shared.write_json(directory / "generated-provenance.json", {"checkpoint_sha256": expected_sha,
        "coefficient_sha256": coefficient_sha, "noise_seeds": seeds, "condition_seed": arguments.seed + 200,
        "pde_solves": 0, "sampler": sampler.identity})
    return matrices, reports


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "cache", "output"):
        result.add_argument(f"--{name}", type=Path, required=True)
    result.add_argument("--regime", choices=("weak", "strong"), required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--replicas", type=int, default=2)
    result.add_argument("--train-seconds", type=int, default=240)
    result.add_argument("--steps", type=int, default=100000)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--validate-every", type=int, default=1000)
    result.add_argument("--learning-rate", type=float, default=2.e-4)
    result.add_argument("--width", type=int, default=256)
    result.add_argument("--depth", type=int, default=4)
    result.add_argument("--temporal-weight", type=float, default=.25)
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
        cli.error("real VPFP direction pilot requires a Slurm compute allocation")
    if arguments.output.exists() or arguments.output.is_symlink() or arguments.output.resolve().is_relative_to(arguments.cache.resolve()):
        cli.error("choose a fresh output outside the immutable historical run")
    for name in ("replicas", "train_seconds", "steps", "batch_size", "validate_every"):
        if getattr(arguments, name) < 1:
            cli.error(f"{name} must be positive")
    if arguments.replicas < 2 or arguments.generated_answers < 32 or arguments.generated_answers % 8 or arguments.sample_steps < 2:
        cli.error("pilot needs >=2 training seeds, balanced >=32 generated cases, >=2 sample steps")
    if arguments.seed < 0 or any(not np.isfinite(getattr(arguments, key)) or getattr(arguments, key) <= 0
                                 for key in ("learning_rate", "temporal_weight", "constraint_tolerance")):
        cli.error("invalid seed, objective, or numeric tolerance")
    DiffusionAuditConfig(permutations=arguments.permutations, bootstrap_repeats=arguments.bootstrap_repeats).validate()
    payload = json.loads(arguments.config.read_text())
    spec = VPFPOnlineModelSpec.from_mapping(payload.get("vpfp", payload))
    spec.preflight()
    corpus = arguments.cache / "corpus"
    manifest = read_manifest(corpus, verify_hashes=True)
    if spec != VPFPOnlineModelSpec.from_mapping(manifest["model_spec"]):
        raise ValueError("pilot config must match the complete cached corpus model specification")
    if arguments.regime not in spec.regime_id:
        raise ValueError("pilot regime label does not match its model specification")
    config = shared.model_config(arguments, spec)
    decoder_config = manifest["provenance"]["decoder_config"]
    frozen_cache = cache_inventory(arguments.cache)
    selected_rows = fixed_reference_rows(corpus, spec, arguments.generated_answers)
    arguments.source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    arguments.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    plan = {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()}
    plan.update(schema="vpfp_value_control_direction_pilot_v1", full_model_config=config.to_dict(),
        source_corpus_manifest_sha256=manifest["manifest_sha256"], cached_feature_sha256=frozen_cache,
        selected_reference_rows=selected_rows, source_file_sha256=shared.source_fingerprints(),
        script_sha256=shared.file_sha(__file__), model_spec=asdict(spec), arms=list(ARMS),
        evidence_role="development_screen_on_previously_opened_reference", fresh_confirmation_required=True,
        automatic_large_experiment=False, automatic_admission=False,
        initialization="same_seed_within_representation_shared_shape_compatible_trunk_across_representations",
        cross_representation_validation_loss_comparable=False,
        noise_streams="same_seed_identifiers_and_conditions_but_different_dimensional_Gaussian_states",
        arm_order="forward_on_even_replica_reverse_on_odd_replica", new_pde_solves=0)
    shared.write_json(arguments.output / "frozen-plan.json", plan)
    plan_sha = shared.file_sha(arguments.output / "frozen-plan.json")
    stage, started = "full_corpus_constraint_audit", time.monotonic()
    try:
        shared.write_json(arguments.output / "stage.json", {"stage": stage, "status": "running"})
        constraint = audit_value_control_corpus(corpus, batch_size=8, tolerance=arguments.constraint_tolerance)
        shared.write_json(arguments.output / "value-slope-corpus-audit.json", constraint)
        require_value_control_audit(constraint)
        if constraint["source_manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("corpus changed after the pilot plan freeze")
        stage = "full_layout_preflight"
        shared.write_json(arguments.output / "stage.json", {"stage": stage, "status": "running"})
        shared.write_json(arguments.output / "allocation-preflight.json", full_layout_preflight(arguments, spec, corpus, decoder_config))
        checkpoints, training, frozen = {}, {}, {}
        for replica in range(arguments.replicas):
            run_args = copy(arguments)
            run_args.seed = arguments.seed + replica * 10000
            replica_root = arguments.output / f"seed-{replica}"
            replica_root.mkdir()
            order = ARMS if replica % 2 == 0 else tuple(reversed(ARMS))
            for arm in order:
                key = f"seed-{replica}/{arm}"
                stage = f"train_{key}"
                shared.write_json(arguments.output / "stage.json", {"stage": stage, "status": "running"})
                representation, objective = arm.split("_")
                cfg, model_factory, normalizer_factory, metric_factory = factories(run_args, spec, representation, decoder_config)
                run_args.ablation_metadata = {"plan_sha256": plan_sha, "arm": arm, "replica": replica,
                    "cross_representation_validation_loss_comparable": False, "evidence_role": plan["evidence_role"]}
                arm_root = replica_root / arm
                arm_root.mkdir()
                checkpoint, report = shared.train(run_args, spec, corpus, arm_root,
                    model_config_factory=lambda *a, cfg=cfg: cfg, model_factory=model_factory,
                    normalizer_factory=normalizer_factory,
                    training_metric_factory=metric_factory if objective == "time" else None,
                    validation_metric_factory=metric_factory)
                checkpoints[key], training[key], frozen[str(checkpoint)] = checkpoint, report, report["checkpoint_sha256"]
            for representation in ("full", "value"):
                first, second = (training[f"seed-{replica}/{representation}_{objective}"] for objective in ("base", "time"))
                for field in ("initial_model_sha256", "normalizer_sha256", "common_validation_metric"):
                    if first[field] != second[field]:
                        raise ValueError(f"within-representation paired training differs in {field}")
        assert_hashes(frozen)
        shared.write_json(arguments.output / "frozen-checkpoints.json", {"all_arms_selected_before_any_feature_evaluation": True, "sha256": frozen})
        assert_hashes(frozen_cache)
        cached, schemas = load_cached_features(arguments.cache, manifest, selected_rows, constraint)
        first_checkpoint = checkpoints["seed-0/full_base"]
        sampler = load_vpfp_diffusion_sampler(first_checkpoint, frozen[str(first_checkpoint)], device=arguments.device, steps=arguments.sample_steps)
        binding = build_vpfp_online_binding(spec, output_root=arguments.output / "audit-runtime", device=arguments.device, latent_sampler=sampler)
        decoder = binding.plugin.generator.backend.runtime.base_model
        scorecards = {"base": [], "time": []}
        audit_summaries = {}
        for replica in range(arguments.replicas):
            run_args = copy(arguments)
            run_args.seed = arguments.seed + replica * 10000
            matrices = {role: {kind: cached[kind][role].features for kind in cached} for role in ("reference_a", "reference_b")}
            reports = {}
            for arm in ARMS:
                key = f"seed-{replica}/{arm}"
                stage = f"frozen_development_audit_{key}"
                shared.write_json(arguments.output / "stage.json", {"stage": stage, "status": "running"})
                assert_hashes(frozen)
                matrices[arm], reports[arm] = sample_and_audit(run_args, spec, checkpoints[key], frozen[str(checkpoints[key])],
                    arguments.output / key, decoder, cached, schemas)
                audit_summaries[key] = {kind: {"status": report["status"],
                    "sw_effect": report["comparisons"]["generated"]["statistics"]["sliced_wasserstein_max"]["effect"]}
                    for kind, report in reports[arm].items()}
            for objective in ("base", "time"):
                names = ("reference_a", "reference_b", f"full_{objective}", f"value_{objective}")
                card = build_vpfp_direction_scorecard(regime=arguments.regime, seed=run_args.seed, objective=objective,
                    baseline_arm=f"full_{objective}", samples={name: matrices[name] for name in names},
                    physical_schema=schemas["physical"], derivative_schema=schemas["derivative"],
                    joint_reports={name: reports[name] for name in names[2:]},
                    physical_time_fractions=VPFPFeaturePlan().time_fractions,
                    derivative_time_fractions=VPFPDerivativeFeaturePlan().time_fractions,
                    macro_probe_count=VPFPDerivativeFeaturePlan().macro_probe_count,
                    phase_probe_count=VPFPDerivativeFeaturePlan().phase_probe_count)
                scorecards[objective].append(card)
                shared.write_json(arguments.output / f"seed-{replica}/direction-{objective}.json", card)
        assert_hashes(frozen)
        assert_hashes(frozen_cache)
        read_manifest(corpus, verify_hashes=True)
        decisions = {objective: aggregate_vpfp_direction_scorecards(cards, minimum_seeds=2)
                     for objective, cards in scorecards.items()}
        report = {"execution_passed": True, "training": training, "audits": audit_summaries,
            "direction_decisions": decisions, "wall_seconds": time.monotonic() - started,
            "evidence_role": plan["evidence_role"], "fresh_confirmation_required": True,
            "cross_regime_scorecard_aggregation_required": True,
            "large_experiment_launched": False, "canonical_campaign_smoke": "separate_required_step", "admitted": False}
    except Exception as error:
        report = {"execution_passed": False, "failed_stage": stage, "error": repr(error),
            "traceback": traceback.format_exc(), "partial_artifacts_preserved": True, "admitted": False}
    shared.write_json(arguments.output / "result.json", report)
    shared.write_json(arguments.output / "stage.json", {"stage": stage, "status": "complete" if report["execution_passed"] else "failed"})
    shared.emit("finished", **report)
    return 0 if report["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
