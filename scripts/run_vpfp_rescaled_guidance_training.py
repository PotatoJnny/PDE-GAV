"""Train matched standardized mean/exp VPFP guides on the original 64 fit roots.

One seed, 2000 updates per head, same initialization and optimizer random stream.
Development probes do not select checkpoints or sampler gains. No solver calls
or online-asset replacement occurs here. All numerical work requires Slurm.
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
from run_vpfp_exponential_guidance import tensor_sha

SCHEMA = "vpfp_rescaled_guidance_experiment_v1"
OBJECTIVES = {"mean": "standardized_mean_reward", "exp": "standardized_exponential_reward_log_h_v1"}
SPLIT_SEEDS = {"fit": 9320260908, "validation": 9420260908, "selection": 9520260908,
               "confirmation": 9620260908, "training": 9720260908}


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    fixed = {"steps": 2000, "seed": 3120260908, "calibration_seed": 7120260908,
             "validation_seed": 7220260908, "bootstrap_seed": 9820260908, "bootstrap_replicates": 10000}
    if (not isinstance(cfg, dict) or cfg.get("schema") != SCHEMA
            or any(type(cfg.get(key)) is not int or cfg[key] != value for key, value in fixed.items())
            or type(cfg.get("beta_std")) not in (int, float) or cfg["beta_std"] != .5
            or cfg.get("fresh_split_seeds") != SPLIT_SEEDS
            or any(type(value) is not int for value in cfg["fresh_split_seeds"].values())
            or cfg.get("reference_effect_config") != "configs/vpfp_calibrated_guidance_effect_20260908.json"):
        raise ValueError("fixed same-seed 2000-step mean/exp beta_std=0.5 experiment required")
    prior = cfg.get("prior_effects")
    if (not isinstance(prior, list) or len(prior) != 2
            or any(not isinstance(item, dict) or set(item) != {"path", "sha256"} for item in prior)
            or len({item["path"] for item in prior}) != 2):
        raise ValueError("two distinct SHA-bound historical effect reports required")
    return cfg


def validation_probe(model, cache, targets, valid, *, fit_constant, check):
    """160 fixed noisy inputs; exact whole-gradient and actual FP32 delta hashes."""
    import torch
    exp_head = model.objective == OBJECTIVES["exp"]
    records = []
    model.eval()
    for item in cache:
        check()
        state = item["noisy"].detach().requires_grad_(True)
        prediction = model(state, item["sigma"], item["conditions"], initial_context=item["initial"])
        potential = model.score_potential(prediction)
        gradient = torch.autograd.grad(potential.sum(), state)[0]
        # Unit lambda is only a diagnostic, not a selected gain for the solver test.
        delta, norm32 = applied_displacement(item["denoised"], gradient, item["sigma"], strength=1.)
        norm64 = torch.linalg.vector_norm(gradient.detach().double(), dim=1)
        predicted = prediction["desirability" if exp_head else "standardized_reward"]
        for index, root in enumerate(item["roots"]):
            offset = item["start"] + index
            record = {"root_id": root, "stratum": item["strata"][index], "sigma": item["sigma_value"],
                "replicate": item["replicate"], "valid": bool(valid[offset]),
                "target": float(targets[offset]), "prediction": float(predicted[index].detach()),
                "score_potential": float(potential[index].detach()),
                "valid_probability": float(prediction["valid_probability"][index].detach()),
                "gradient_l2_fp32": float(norm32[index]), "gradient_l2_fp64": float(norm64[index]),
                "actual_delta_max": float(delta[index].abs().max()),
                "actual_delta_nonzero": bool(torch.any(delta[index] != 0)),
                "full_gradient_sha256": tensor_sha(gradient[index]),
                "full_delta_sha256": tensor_sha(delta[index])}
            if exp_head:
                record.update(log_desirability=float(prediction["log_desirability"][index].detach()),
                              desirability=record["prediction"])
            else:
                record["standardized_reward"] = record["prediction"]
            records.append(record)
    inventory = {(record["root_id"], record["sigma"], record["replicate"]) for record in records}
    if len(records) != 160 or len(inventory) != 160 or len({record["root_id"] for record in records}) != 16:
        raise ValueError("exactly 16 development roots x 5 sigma x 2 replicates required")
    if any(not math.isfinite(record[key]) for record in records for key in
           ("target", "prediction", "score_potential", "valid_probability", "gradient_l2_fp32",
            "gradient_l2_fp64", "actual_delta_max")):
        raise FloatingPointError("nonfinite standardized guidance development diagnostics")
    eligible = [record for record in records if exp_head or record["valid"]]
    if not eligible:
        raise ValueError("development set has no defined regression targets")
    by_sigma = {}
    for sigma in sorted({record["sigma"] for record in records}):
        selected = [record for record in records if record["sigma"] == sigma]
        by_sigma[str(sigma)] = {
            "record_count": len(selected), "nonzero_actual_delta_records": sum(r["actual_delta_nonzero"] for r in selected),
            "median_gradient_l2_fp64": statistics.median(r["gradient_l2_fp64"] for r in selected),
            "median_actual_delta_max": statistics.median(r["actual_delta_max"] for r in selected)}
    summary = {"objective": model.objective, "record_count": 160, "independent_roots": 16,
        "regression_mse": statistics.mean((record["prediction"] - record["target"])**2 for record in eligible),
        "fit_constant_mse": statistics.mean((fit_constant - record["target"])**2 for record in eligible),
        "fit_constant": fit_constant, "diagnostic_external_lambda": 1.,
        "potential_range": [min(record["score_potential"] for record in records),
                            max(record["score_potential"] for record in records)],
        "prediction_range": [min(record["prediction"] for record in records),
                             max(record["prediction"] for record in records)],
        "by_sigma": by_sigma, "selection_or_solver_effect_gate": False,
        "different_objective_losses_not_directly_comparable": True, "scientific_admission": False}
    return records, summary


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
        "job_id": os.environ["SLURM_JOB_ID"], "completed_step": 0, "target_step_per_head": 2000,
        "base_training_updates": 0, "solver_training_updates": 0, "new_solver_calls": 0,
        "online_assets_replaced": False, "scientific_admission": False, "heads": {}, "failures": [],
        "gain_calibration_performed": False, "regression_gate": False}
    def save():
        report["wall_seconds"] = time.monotonic() - started
        write(output / "result.json", report)
    def check():
        if time.monotonic() - started > args.wall_seconds - 30:
            raise TimeoutError("training allocation exhausted; no silent reduction of updates or probes")
    save()
    try:
        import torch
        from pde_agent_verifier_solver.plugins.vpfp.standardized_solver_error_guidance import (
            make_standardized_regressor, fit_standardized_calibration, standardized_targets, standardized_payload)
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
        config_binding = {"path": str(args.config.resolve()), "sha256": file_sha(args.config)}
        template_path = ROOT / cfg["reference_effect_config"]
        template, _, _, old_training = load_effect_config(template_path)
        if old_training.get("final_step") != 2000 or old_training.get("seed") != cfg["seed"]:
            raise ValueError("reference must be the original same-seed step2000 guide")
        if "guidance_objective" in template or "prior_effect" not in template:
            raise ValueError("root registration must use the untouched old calibrated guide and historical prior")
        root_cfg = deepcopy(template)
        root_cfg.update(fresh_split_seeds=cfg["fresh_split_seeds"], prior_effects=cfg["prior_effects"],
            bootstrap_seed=cfg["bootstrap_seed"], bootstrap_replicates=cfg["bootstrap_replicates"])
        root_path = output / "root-preregistration.json"
        write(root_path, root_cfg)
        _, _, roots, _ = load_effect_config(root_path)
        root_path.chmod(0o444)
        root_binding = {"path": str(root_path), "sha256": file_sha(root_path)}
        roots_path = output / "preregistered-confirmation-roots.json"
        write(roots_path, roots)
        roots_path.chmod(0o444)
        evidence = {key: template[key] for key in
                    ("guide", "training_report", "training_labels", "training_plan", "prior_effect")}
        evidence.update({f"prior_effects_{index:04d}": item for index, item in enumerate(cfg["prior_effects"])})
        evidence["experiment_config"] = config_binding
        evidence["reference_effect_config"] = {"path": str(template_path), "sha256": file_sha(template_path)}
        evidence["root_preregistration"] = root_binding
        evidence["registered_roots"] = {"path": str(roots_path), "sha256": file_sha(roots_path)}
        old_plan = json.loads(Path(template["training_plan"]["path"]).read_text())
        labels = json.loads(Path(template["training_labels"]["path"]).read_text())
        bindings = verify_training_draws(labels, old_plan)
        if len(bindings) != 160:
            raise ValueError("all original 80 draw/result pairs (160 SHA bindings) required")
        rows = {}
        for split, count in (("fit", 64), ("validation", 16)):
            values = load_training_rows(labels["labels"][split])
            if values["excluded_rows"] or len(values["root_ids"]) != count or not bool(values["valid"].all()):
                raise ValueError("all original 64 fit and 16 development real valid labels must be retained")
            expected_roots = {row["slot_id"] for row in old_plan["roots"][split]}
            if set(values["root_ids"]) != expected_roots:
                raise ValueError("original training/development root identities differ")
            rows[split] = {key: value.to("cuda") if isinstance(value, torch.Tensor) else value
                           for key, value in values.items()}
        raw = torch.load(template["guide"]["path"], map_location="cpu", weights_only=True)
        gc = SolverErrorGuidanceConfig.from_dict(raw["config"])
        recipe = old_plan["config"]["training"]
        if (recipe["batch_size"] != 4 or recipe["learning_rate"] != .0002 or recipe["steps"] != 2000
                or gc.width != 32 or gc.depth != 2 or gc.error_scale != 1. or gc.validity_weight != .1
                or raw["training"]["seed"] != cfg["seed"] or raw["training"]["step"] != 2000):
            raise ValueError("frozen backbone, original recipe or reference mismatch")
        base, portable = load_round_base(old_plan["config"], old_plan["assets"], device="cuda")
        base.model.eval().requires_grad_(False)
        base.normalizer.eval().requires_grad_(False)
        if (next(base.model.parameters()).dtype != torch.float32 or gc.base_config != base.model.config
                or gc.normalizer_sha256 != _json_sha(base.normalizer.to_payload())
                or gc.base_checkpoint_sha256 != old_plan["config"]["base_checkpoint_sha256"]):
            raise ValueError("exact frozen FP32 Base/config/normalizer binding required")
        base_sha, norm_sha = _module_sha(base.model), _json_sha(base.normalizer.to_payload())
        report.update(config=cfg, original_recipe=recipe, model_config=gc.to_dict(), evidence=evidence,
            portable_binding=portable, draw_label_semantic_bindings_verified=80,
            root_preregistration=root_binding, registered_confirmation_roots=roots,
            registered_roots=evidence["registered_roots"], beta_std=cfg["beta_std"],
            original_fit_all_valid=True, original_validation_all_valid=True,
            invalid_semantics_not_empirically_tested_in_this_training_pool=True)
        frozen_plan = output / "frozen-plan.json"
        write(frozen_plan, {**report, "input_bindings": bindings})
        frozen_plan.chmod(0o444)
        report["frozen_plan"] = {"path": str(frozen_plan), "sha256": file_sha(frozen_plan)}
        cache = build_validation_cache(base, rows["validation"], old_plan,
            seed=cfg["validation_seed"], check=check)
        initialization_sha = stream_sha = fitted_statistics = None

        def verify_evidence():
            verify_source()

        for name, objective in OBJECTIVES.items():
            check()
            head_started = time.monotonic()
            directory = output / name
            directory.mkdir()
            report.update(stage="calibration", current_head=name, completed_step=0)
            save()
            # Restart initialization and optimizer/noising streams for each head.
            torch.manual_seed(cfg["seed"])
            model = make_standardized_regressor(gc, parameter_dtype=torch.float32, device="cuda",
                beta_std=cfg["beta_std"], objective=objective)
            calibration = fit_standardized_calibration(model, rows["fit"],
                seed=cfg["calibration_seed"], batch_size=4)
            initial_sha = _module_sha(model)
            statistics_keys = ("fit_reward_mean", "fit_reward_std", "log_fit_mean_exp", "effective_beta",
                               "fit_desirability_ess", "fit_label_records_sha256")
            current_statistics = {key: calibration[key] for key in statistics_keys}
            if initialization_sha is None:
                initialization_sha, fitted_statistics = initial_sha, current_statistics
            elif initial_sha != initialization_sha or current_statistics != fitted_statistics:
                raise ValueError("mean/exp initial backbone, fit scales or reward statistics differ")
            targets, masks = {}, {}
            for split, values in rows.items():
                targets[split], masks[split] = standardized_targets(values["errors"], values["valid"],
                    prototype=values["future_white"], fit_reward_mean=model.fit_reward_mean,
                    fit_reward_std=model.fit_reward_std, log_fit_mean_exp=model.log_fit_mean_exp,
                    beta_std=model.beta_std, objective=objective, error_scale=gc.error_scale)
            fit_constant = 1. if name == "exp" else 0.
            initial_records, initial = validation_probe(model, cache, targets["validation"], masks["validation"],
                fit_constant=fit_constant, check=check)
            write(directory / "initial-validation.json", {"summary": initial, "records": initial_records})
            write(directory / "calibration.json", calibration)
            report.update(stage="training")
            save()
            optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"])
            rng = torch.Generator(device="cuda").manual_seed(cfg["seed"] + 1)
            stream_digest = sha256()
            trace = []
            model.train()
            for step in range(1, 2001):
                check()
                stream_digest.update(tensor_sha(rng.get_state()).encode())
                index = torch.randint(64, (4,), device="cuda", generator=rng)
                stream_digest.update(tensor_sha(index).encode())
                optimizer.zero_grad(set_to_none=True)
                loss, diag = model.loss(rows["fit"]["future_white"][index], rows["fit"]["conditions"][index],
                    rows["fit"]["initial_context"][index], rows["fit"]["errors"][index], rows["fit"]["valid"][index],
                    generator=rng)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite standardized guidance training loss")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                report["completed_step"] = step
                if step <= 5 or step % 100 == 0:
                    predicted = diag["predicted_desirability" if name == "exp" else "predicted_standardized_reward"]
                    trace.append({"step": step, "loss": float(loss.detach()),
                        "regression_loss": float(diag["regression_loss"]),
                        "validity_loss": float(diag["validity_loss"]),
                        "gradient_norm_before_clip": float(grad_norm),
                        "prediction_min": float(predicted.min()), "prediction_max": float(predicted.max())})
                    write(directory / "training-trace.json", trace)
                    save()
            stream_digest.update(tensor_sha(rng.get_state()).encode())
            current_stream_sha = stream_digest.hexdigest()
            if stream_sha is None:
                stream_sha = current_stream_sha
            elif current_stream_sha != stream_sha:
                raise ValueError("mean/exp optimizer batch and noising random streams differ")
            model.eval()
            records, validation = validation_probe(model, cache, targets["validation"], masks["validation"],
                fit_constant=fit_constant, check=check)
            write(directory / "validation.json", {"summary": validation, "records": records})
            payload = standardized_payload(model, label_manifest_sha256=template["training_labels"]["sha256"],
                train_root_ids=rows["fit"]["root_ids"], validation_root_ids=rows["validation"]["root_ids"],
                training={"label_source": "actual_public_solver_macro_error", "step": 2000,
                    "seed": cfg["seed"], "beta_std": cfg["beta_std"], "base_trainable": False,
                    "solver_trainable": False, "checkpoint_choice": "fixed_step_not_best",
                    "objective": objective, "optimizer_random_stream_sha256": current_stream_sha})
            checkpoint = directory / "fixed-step-002000.pt"
            with checkpoint.open("xb") as stream:
                torch.save(payload, stream)
            checkpoint.chmod(0o444)
            checkpoint_binding = {"path": str(checkpoint), "sha256": file_sha(checkpoint)}
            restored = load_solver_error_regressor(checkpoint, checkpoint_binding["sha256"], device="cuda")
            replay_records, replay_summary = validation_probe(restored, cache, targets["validation"], masks["validation"],
                fit_constant=fit_constant, check=check)
            if replay_records != records or replay_summary != validation:
                raise ValueError("reload changed predictions, full gradient vectors or actual FP32 displacements")
            sampler = GuidedIC40Sampler(base, restored, strength=1.,
                max_gradient_norm=old_plan["config"]["sampling"]["max_gradient_norm"],
                sigma_min=old_plan["config"]["sampling"]["guidance_sigma_min"],
                sigma_max=old_plan["config"]["sampling"]["guidance_sigma_max"])
            sampler.assert_frozen()
            with (directory / "continuation.pt").open("xb") as stream:
                torch.save({"schema": SCHEMA + "_continuation", "objective": objective, "step": 2000,
                    "payload": payload, "optimizer": optimizer.state_dict(), "training_rng": rng.get_state(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, stream)
            verify_evidence()
            training = {"final_step": 2000, "target_step": 2000, "seed": cfg["seed"],
                "objective": objective, "beta_std": cfg["beta_std"], "effective_beta": model.beta,
                "label_manifest_sha256": template["training_labels"]["sha256"],
                "checkpoint": checkpoint_binding, "calibration": calibration, "validation": validation,
                "sample_presentations": 8000, "parameter_count": sum(p.numel() for p in model.parameters()),
                "gpu_wall_seconds": time.monotonic() - head_started,
                "initialization_state_sha256": initial_sha, "optimizer_random_stream_sha256": current_stream_sha,
                "reload_full_gradient_and_displacement_hashes_exact": True, "reload_probe_count": 160,
                "draw_label_semantic_bindings_verified": 80, "base_trainable": False, "solver_trainable": False,
                "source_final_verified": True, "evidence_final_verified": True, "base_final_verified": True,
                "regression_gate": False, "gain_calibration_performed": False,
                "root_preregistration": root_binding, "registered_confirmation_roots": roots,
                "sampler_identity_unit_lambda_diagnostic_only": sampler.identity,
                "invalid_semantics_not_empirically_tested_in_this_training_pool": True}
            training_path = directory / "training.json"
            write(training_path, training)
            training_path.chmod(0o444)
            report["heads"][name] = {"objective": objective, "checkpoint": checkpoint_binding,
                "training_report": {"path": str(training_path), "sha256": file_sha(training_path)},
                "completed_step": 2000, "calibration": calibration, "validation": validation,
                "initial_validation": initial, "reload_full_gradient_and_displacement_hashes_exact": True}
            save()
            # Model assets remain on disk; free old training objects before next head.
            del optimizer, sampler, restored, model, payload
        verify_evidence()
        for head in report["heads"].values():
            for role in ("checkpoint", "training_report"):
                item = head[role]
        report.update(stage="complete", execution_passed=True, source_final_verified=True,
            evidence_final_verified=True, base_final_verified=True,
            mean_exp_initialization_exact=True, mean_exp_optimizer_random_stream_exact=True,
            initialization_state_sha256=initialization_sha, optimizer_random_stream_sha256=stream_sha,
            reload_full_gradient_and_displacement_hashes_exact=True, reload_probe_count_per_head=160,
            next_stage="fit_only_gain_calibration_then_preregistered_32_root_solver_test")
    except Exception as error:
        report["failures"].append({"stage": report.get("stage"), "head": report.get("current_head"),
            "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
    finally:
        save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, default=1200.)
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps({key: result.get(key) for key in
                      ("execution_passed", "stage", "current_head", "completed_step", "next_stage", "failures")}))
    return 0 if result["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
