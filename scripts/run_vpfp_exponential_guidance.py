"""Train one exponential-reward Guidance, then freeze a three-arm solver test.

The positive regression target is I(valid)*exp(beta*(bounded_error-fit_center)).
The sampler adds sigma^2*grad(log_h); beta is NOT applied a second time. All
numerical work requires Slurm. Development metrics never gate the actual test.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from run_vpfp_strong_solver_guidance_round import file_sha, load_round_base, load_training_rows, write
from run_vpfp_strong_guidance_effect import load_effect_config
from run_vpfp_guidance_calibration import build_validation_cache, applied_displacement

SCHEMA = "vpfp_exponential_guidance_experiment_v1"
OBJECTIVE = "exponential_reward_log_h_v1"


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    if (cfg.get("schema") != SCHEMA
            or any(type(cfg.get(k)) is not int for k in
                   ("steps", "seed", "calibration_seed", "validation_seed", "bootstrap_replicates"))
            or cfg["steps"] != 2000
            or cfg.get("seed") != 3120260908 or cfg.get("beta") != 2.
            or cfg.get("external_strength") != 1. or cfg.get("bootstrap_replicates") != 10000
            or cfg.get("calibration_seed") != 7120260908 or cfg.get("validation_seed") != 7220260908):
        raise ValueError("fixed one-seed/2000-step beta2 log-desirability control required")
    split = cfg.get("fresh_split_seeds", {})
    if (set(split) != {"fit", "validation", "selection", "confirmation", "training"}
            or any(type(v) is not int or not 0 <= v < 2**63 for v in split.values())
            or type(cfg.get("bootstrap_seed")) is not int
            or not 0 <= cfg["bootstrap_seed"] < 2**63
            or len({*split.values(), cfg["seed"], cfg["calibration_seed"], cfg["validation_seed"],
                    cfg["bootstrap_seed"]}) != 9):
        raise ValueError("independent registered random streams required")
    if not isinstance(cfg.get("prior_effects"), list) or len(cfg["prior_effects"]) != 1:
        raise ValueError("retain the previously evaluated calibrated comparison as prior evidence")
    return cfg


def tensor_sha(value):
    tensor = value.detach().cpu().contiguous()
    digest = sha256()
    digest.update(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def validation_probe(model, cache, targets, *, fit_weight_mean, check):
    import torch
    records = []
    model.eval()
    for item in cache:
        check()
        state = item["noisy"].detach().requires_grad_(True)
        prediction = model(state, item["sigma"], item["conditions"], initial_context=item["initial"])
        potential = model.score_potential(prediction)
        gradient = torch.autograd.grad(potential.sum(), state)[0]
        delta, norm32 = applied_displacement(item["denoised"], gradient, item["sigma"], strength=1.)
        norm64 = torch.linalg.vector_norm(gradient.detach().double(), dim=1)
        for j, root in enumerate(item["roots"]):
            records.append({"root_id": root, "stratum": item["strata"][j], "sigma": item["sigma_value"],
                "replicate": item["replicate"], "weight_target": float(targets[item["start"]+j]),
                "log_desirability": float(prediction["log_desirability"][j].detach()),
                "desirability": float(prediction["desirability"][j].detach()),
                "valid_probability": float(prediction["valid_probability"][j].detach()),
                "gradient_l2_fp32": float(norm32[j]), "gradient_l2_fp64": float(norm64[j]),
                "actual_delta_max": float(delta[j].abs().max()),
                "actual_delta_nonzero": bool(torch.any(delta[j] != 0)),
                "full_gradient_sha256": tensor_sha(gradient[j]),
                "full_delta_sha256": tensor_sha(delta[j])})
    if len(records) != 160:
        raise ValueError("16 development roots x 5 sigma x 2 replicates required")
    if any(not math.isfinite(r[k]) for r in records for k in (
            "weight_target", "log_desirability", "desirability", "valid_probability",
            "gradient_l2_fp32", "gradient_l2_fp64", "actual_delta_max")):
        raise FloatingPointError("nonfinite exponential validation diagnostics")
    summary = {"record_count": 160, "independent_roots": 16,
        "desirability_mse": statistics.mean((r["desirability"]-r["weight_target"])**2 for r in records),
        "fit_constant_mse": statistics.mean((fit_weight_mean-r["weight_target"])**2 for r in records),
        "fit_weight_mean": fit_weight_mean,
        "log_h_range": [min(r["log_desirability"] for r in records), max(r["log_desirability"] for r in records)],
        "sigma1_effective_roots": sum(all(r["actual_delta_nonzero"] for r in records
            if r["root_id"] == root and r["sigma"] == 1.) for root in sorted({r["root_id"] for r in records})),
        "selection_or_solver_effect_gate": False,
        "not_comparable_to_raw_reward_MSE": True, "scientific_admission": False}
    return records, summary


def build_effect_config(template, cfg, checkpoint, training_report):
    effect = deepcopy(template)
    effect.update(guidance_objective=OBJECTIVE, beta=cfg["beta"], strength=1.,
        reference_guide=deepcopy(template["guide"]),
        reference_training_report=deepcopy(template["training_report"]), reference_strength=2.,
        guide=checkpoint, training_report=training_report, fresh_split_seeds=cfg["fresh_split_seeds"],
        prior_effects=cfg["prior_effects"], bootstrap_seed=cfg["bootstrap_seed"],
        bootstrap_replicates=cfg["bootstrap_replicates"],
        selection_note={"basis": "fixed_seed_and_fixed_step_not_regression_selection",
            "training_objective": "joint_valid_exponential_reward_MSE",
            "old_guidance": "frozen_combined_direct_reward_seed3120260908_step2000",
            "new_guidance": "log_h_score_beta2_external_strength1",
            "primary_endpoint": "new_vs_base_actual_solver_error",
            "secondary_endpoint": "new_vs_old_actual_solver_error",
            "regression_metrics_are_diagnostics_only": True,
            "public_macro_contract_changed": False, "online_assets_replaced": False})
    return effect


def run(args):
    if not os.environ.get("SLURM_JOB_ID", "").isdigit() or not os.environ.get("SLURM_TMPDIR"):
        raise ValueError("Slurm compute allocation required")
    if not math.isfinite(args.wall_seconds) or args.wall_seconds <= 120:
        raise ValueError("bounded training allocation required")
    scratch = Path("${PDE_GAV_RUN_ROOT}")
    output = args.output.absolute()
    if output == scratch or not output.is_relative_to(scratch) or output.exists() or output.is_symlink():
        raise ValueError("new specific scratch output required; no implicit retry/resume")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    def verify_source():
        if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != revision
                or subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()):
            raise ValueError("source must remain frozen and clean")
    verify_source()
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {"schema": SCHEMA, "execution_passed": False, "source_revision": revision,
        "job_id": os.environ["SLURM_JOB_ID"], "completed_step": 0, "target_step": 2000,
        "base_training_updates": 0, "solver_training_updates": 0, "new_solver_calls": 0,
        "online_assets_replaced": False, "scientific_admission": False, "failures": []}
    def save():
        report["wall_seconds"] = time.monotonic()-started
        write(output / "result.json", report)
    def check():
        if time.monotonic()-started > args.wall_seconds-30:
            raise TimeoutError("training allocation exhausted; no silent reduction of updates or tests")
    save()
    try:
        import torch
        from pde_agent_verifier_solver.plugins.vpfp.exponential_solver_error_guidance import (
            make_exponential_regressor, fit_exponential_calibration, exponential_targets, exponential_payload)
        from pde_agent_verifier_solver.plugins.vpfp.solver_error_guidance import (
            SolverErrorGuidanceConfig, load_solver_error_regressor, _module_sha, _json_sha, GuidedIC40Sampler)
        from run_vpfp_guidance_training_debug import verify_training_draws
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one allocated CUDA device required")
        torch.set_num_threads(min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))))
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        cfg = load_config(args.config)
        template, _, _, old_training = load_effect_config(ROOT / cfg["reference_effect_config"])
        if old_training.get("final_step") != 2000 or old_training.get("seed") != cfg["seed"]:
            raise ValueError("reference must be same seed and step2000")
        root_cfg = deepcopy(template)
        root_cfg.update(fresh_split_seeds=cfg["fresh_split_seeds"], prior_effects=cfg["prior_effects"],
                        bootstrap_seed=cfg["bootstrap_seed"])
        root_path = output / "root-preregistration.json"
        write(root_path, root_cfg)
        _, _, roots, _ = load_effect_config(root_path)
        evidence = {k: template[k] for k in ("guide", "training_report", "training_labels", "training_plan", "prior_effect")}
        evidence["previous_calibrated_effect"] = cfg["prior_effects"][0]
        old_plan = json.loads(Path(template["training_plan"]["path"]).read_text())
        labels = json.loads(Path(template["training_labels"]["path"]).read_text())
        bindings = verify_training_draws(labels, old_plan)
        rows = {}
        for split, count in (("fit", 64), ("validation", 16)):
            values = load_training_rows(labels["labels"][split])
            if values["excluded_rows"] or len(values["root_ids"]) != count or not bool(values["valid"].all()):
                raise ValueError("all original 64 fit and 16 development real labels must be retained")
            rows[split] = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in values.items()}
        raw = torch.load(template["guide"]["path"], map_location="cpu", weights_only=True)
        gc = SolverErrorGuidanceConfig.from_dict(raw["config"])
        recipe = old_plan["config"]["training"]
        if (recipe["batch_size"] != 4 or recipe["learning_rate"] != .0002 or recipe["steps"] != 2000
                or gc.width != 32 or gc.depth != 2 or gc.error_scale != 1. or gc.validity_weight != .1
                or raw["training"]["seed"] != cfg["seed"] or raw["training"]["step"] != 2000):
            raise ValueError("frozen backbone, original training recipe and reference mismatch")
        base, portable = load_round_base(old_plan["config"], old_plan["assets"], device="cuda")
        base.model.eval().requires_grad_(False); base.normalizer.eval().requires_grad_(False)
        if (next(base.model.parameters()).dtype != torch.float32 or gc.base_config != base.model.config
                or gc.normalizer_sha256 != _json_sha(base.normalizer.to_payload())
                or gc.base_checkpoint_sha256 != old_plan["config"]["base_checkpoint_sha256"]):
            raise ValueError("exact frozen FP32 Base/config/normalizer binding required")
        base_sha, norm_sha = _module_sha(base.model), _json_sha(base.normalizer.to_payload())
        torch.manual_seed(cfg["seed"])
        model = make_exponential_regressor(gc, parameter_dtype=torch.float32, device="cuda", beta=cfg["beta"])
        calibration = fit_exponential_calibration(model, rows["fit"], seed=cfg["calibration_seed"], batch_size=4)
        targets = {split: exponential_targets(data["errors"], data["valid"], prototype=data["future_white"],
            reward_center=model.reward_center, beta=model.beta, error_scale=gc.error_scale)[0]
            for split, data in rows.items()}
        fit_mean = float(targets["fit"].double().mean())
        report.update(config=cfg, original_recipe=recipe, model_config=gc.to_dict(), calibration=calibration,
            evidence=evidence, portable_binding=portable, draw_label_semantic_bindings_verified=80,
            reference_checkpoint=template["guide"], registered_confirmation_roots=roots,
            beta=2., external_strength=1., regression_gate=False,
            parameter_count=sum(p.numel() for p in model.parameters()))
        write(output / "frozen-plan.json", {**report, "input_bindings": bindings})
        cache = build_validation_cache(base, rows["validation"], old_plan, seed=cfg["validation_seed"], check=check)
        initial_records, initial = validation_probe(model, cache, targets["validation"], fit_weight_mean=fit_mean, check=check)
        write(output / "initial-validation.json", {"summary": initial, "records": initial_records})
        report.update(stage="training", initial_validation=initial); save()
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"])
        rng = torch.Generator(device="cuda").manual_seed(cfg["seed"]+1)
        model.train()
        trace = []
        for step in range(1, 2001):
            check()
            index = torch.randint(64, (4,), device="cuda", generator=rng)
            optimizer.zero_grad(set_to_none=True)
            value, diag = model.loss(rows["fit"]["future_white"][index], rows["fit"]["conditions"][index],
                rows["fit"]["initial_context"][index], rows["fit"]["errors"][index], rows["fit"]["valid"][index],
                generator=rng)
            if not torch.isfinite(value):
                raise FloatingPointError("nonfinite exponential regression loss")
            value.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            report["completed_step"] = step
            if step <= 5 or step % 100 == 0:
                trace.append({"step": step, "loss": float(value.detach()),
                    "desirability_loss": float(diag["desirability_loss"]), "validity_loss": float(diag["validity_loss"]),
                    "gradient_norm_before_clip": float(grad_norm),
                    "predicted_h_min": float(diag["predicted_desirability"].min()),
                    "predicted_h_max": float(diag["predicted_desirability"].max())})
                write(output / "training-trace.json", trace); save()
        model.eval()
        records, validation = validation_probe(model, cache, targets["validation"], fit_weight_mean=fit_mean, check=check)
        write(output / "validation.json", {"summary": validation, "records": records})
        payload = exponential_payload(model, label_manifest_sha256=template["training_labels"]["sha256"],
            train_root_ids=rows["fit"]["root_ids"], validation_root_ids=rows["validation"]["root_ids"],
            training={"label_source": "actual_public_solver_macro_error", "step": 2000,
                "seed": cfg["seed"], "beta": 2., "base_trainable": False, "solver_trainable": False,
                "checkpoint_choice": "fixed_step_not_best", "objective": OBJECTIVE})
        checkpoint = output / "fixed-step-002000.pt"
        with checkpoint.open("xb") as stream:
            torch.save(payload, stream)
        checkpoint.chmod(0o444)
        checkpoint_binding = {"path": str(checkpoint), "sha256": file_sha(checkpoint)}
        restored = load_solver_error_regressor(checkpoint, checkpoint_binding["sha256"], device="cuda")
        replay_records, replay_summary = validation_probe(restored, cache, targets["validation"],
            fit_weight_mean=fit_mean, check=check)
        if replay_records != records or replay_summary != validation:
            raise ValueError("reload changed actual predictions, full gradient vectors or FP32 displacements")
        sampler = GuidedIC40Sampler(base, restored, strength=1.,
            max_gradient_norm=old_plan["config"]["sampling"]["max_gradient_norm"],
            sigma_min=old_plan["config"]["sampling"]["guidance_sigma_min"],
            sigma_max=old_plan["config"]["sampling"]["guidance_sigma_max"])
        sampler.assert_frozen()
        with (output / "continuation.pt").open("xb") as stream:
            torch.save({"schema": SCHEMA+"_continuation", "step": 2000, "payload": payload,
                "optimizer": optimizer.state_dict(), "training_rng": rng.get_state(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, stream)
        verify_source()
        training = {"final_step": 2000, "target_step": 2000, "seed": cfg["seed"], "beta": 2.,
            "label_manifest_sha256": template["training_labels"]["sha256"],
            "checkpoint": checkpoint_binding, "validation": validation, "sample_presentations": 8000,
            "parameter_count": report["parameter_count"], "gpu_wall_seconds": time.monotonic()-started,
            "reload_full_gradient_and_displacement_hashes_exact": True, "objective": OBJECTIVE,
            "base_trainable": False, "solver_trainable": False, "regression_gate": False}
        training_path = output / "training.json"
        write(training_path, training); training_path.chmod(0o444)
        effect = build_effect_config(template, cfg, checkpoint_binding,
            {"path": str(training_path), "sha256": file_sha(training_path)})
        effect_path = output / "effect-config.json"
        write(effect_path, effect)
        _, _, effect_roots, _ = load_effect_config(effect_path)
        if effect_roots != roots:
            raise ValueError("effect roots differ from pre-training registration")
        effect_path.chmod(0o444)
        report.update(stage="complete", execution_passed=True, checkpoint=checkpoint_binding,
            final_validation=validation, source_final_verified=True, evidence_final_verified=True,
            base_final_verified=True, reload_full_gradient_and_displacement_hashes_exact=True,
            effect_config={"path": str(effect_path), "sha256": file_sha(effect_path)},
            next_stage="direct_three_arm_solver_test_without_regression_gate")
    except Exception as error:
        report["failures"].append({"stage": report.get("stage"), "type": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc()})
    finally:
        save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vpfp_exponential_guidance_20260908.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, default=600.)
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps({k: result.get(k) for k in ("execution_passed", "stage", "completed_step", "next_stage", "failures")}))
    return 0 if result["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
