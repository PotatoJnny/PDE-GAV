"""Fit calibrated continuous-error Guidance controls, then freeze a solver test.

Research-only: the original sigmoid-MSE and labels remain unchanged. Development
screening is not a hardness result. Only the existing fresh paired solver runner
may establish that the selected frozen model makes harder valid questions.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from run_vpfp_strong_solver_guidance_round import file_sha, json_sha, load_round_base, load_training_rows, write
from run_vpfp_strong_guidance_effect import load_effect_config
from vpfp_guidance_calibration_metrics import summarize_predictions, select_variant

SCHEMA = "vpfp_guidance_calibration_experiment_v1"
VARIANTS = {"original": (False, False), "scale": (True, False),
            "initialization": (False, True), "combined": (True, True)}
SIGMAS = (.002, .05, .2, 1., 2.)
SEEDS = (3120260908, 3120260909)


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    if (cfg.get("schema") != SCHEMA or type(cfg.get("steps")) is not int or cfg.get("steps") != 2000
            or cfg.get("seeds") != list(SEEDS) or cfg.get("variants") != list(VARIANTS)
            or cfg.get("validation_sigmas") != list(SIGMAS)
            or type(cfg.get("validation_replicates")) is not int
            or cfg.get("validation_replicates") != 2 or cfg.get("strength") != 2
            or cfg.get("bootstrap_replicates") != 10000):
        raise ValueError("fixed four controls, two seeds, 2000-step comparison and probes required")
    for key in ("calibration_seed", "validation_seed", "bootstrap_seed"):
        if type(cfg.get(key)) is not int or not 0 <= cfg[key] < 2**63:
            raise ValueError("explicit independent seed required: " + key)
    if (set(cfg.get("fresh_split_seeds", {})) != {"fit", "validation", "selection", "confirmation", "training"}
            or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in cfg["fresh_split_seeds"].values())):
        raise ValueError("complete independently registered split seeds required")
    if len({*cfg["seeds"], cfg["calibration_seed"], cfg["validation_seed"], cfg["bootstrap_seed"],
            *cfg["fresh_split_seeds"].values()}) != 10:
        raise ValueError("independent training/calibration/validation/confirmation seeds required")
    return cfg


def noise_seed(seed, root_id, sigma, replicate):
    text = f"vpfp-guidance-calibration-validation:{seed}:{root_id}:{sigma:g}:{replicate}"
    return int.from_bytes(sha256(text.encode()).digest()[:8], "little") % 2**63


def training_rows_by_stratum(rows, plan, targets):
    mapping = {row["slot_id"]: row["stratum"] for row in plan["roots"]["fit"]}
    values = {s: [] for s in range(8)}
    for root, target in zip(rows["root_ids"], targets.detach().double().cpu().tolist()):
        values[mapping[root]].append(target)
    if any(len(items) != 8 for items in values.values()):
        raise ValueError("eight fit roots per stratum required")
    return {str(key): sum(items) / len(items) for key, items in values.items()}


def build_validation_cache(base, rows, plan, *, seed, check):
    import torch
    cache = []
    stratum = {row["slot_id"]: row["stratum"] for row in plan["roots"]["validation"]}
    with torch.no_grad():
        for sigma in SIGMAS:
            for replicate in range(2):
                # Production draws one root at a time. Match its Conv/GEMM
                # batch shape as well as the FP32 correction formula.
                for start in range(len(rows["root_ids"])):
                    check()
                    roots = rows["root_ids"][start:start+1]
                    clean = rows["future_white"][start:start+1]
                    noises = [torch.randn(clean[0].shape, dtype=clean.dtype, device=clean.device,
                        generator=torch.Generator(device=clean.device).manual_seed(
                            noise_seed(seed, root, sigma, replicate))) for root in roots]
                    noisy = clean + sigma * torch.stack(noises)
                    levels = clean.new_full((len(roots),), sigma)
                    context = rows["initial_context"][start:start+1]
                    theta = rows["conditions"][start:start+1]
                    denoised = base.model.denoise(noisy, levels, theta, initial_context=context)
                    if not torch.isfinite(denoised).all():
                        raise ValueError("nonfinite frozen Base validation denoiser")
                    cache.append({"roots": roots, "strata": [stratum[r] for r in roots],
                        "start": start, "sigma_value": sigma, "replicate": replicate,
                        "sigma": levels, "noisy": noisy, "initial": context,
                        "conditions": theta, "denoised": denoised})
    return cache


def applied_displacement(baseline, gradient, sigma, *, strength=2., clip=1.):
    """Exact production arithmetic, including FP32 clipping and addition."""
    import torch
    norm = torch.linalg.vector_norm(gradient, dim=1, keepdim=True)
    if not torch.isfinite(gradient).all() or not torch.isfinite(norm).all():
        raise FloatingPointError("nonfinite validation utility gradient")
    scale = torch.where(norm > clip, clip / norm.clamp_min(torch.finfo(norm.dtype).tiny), torch.ones_like(norm))
    applied = gradient.detach() * scale
    displacement = strength * sigma[:, None].square() * applied
    guided = baseline + displacement
    if not torch.isfinite(guided).all():
        raise FloatingPointError("nonfinite validation guided denoiser")
    return guided - baseline, norm[:, 0]


def evaluate_validation(model, cache, targets, *, fit_mean, fit_stratum_means, check):
    import torch
    records = []
    model.eval()
    for entry in cache:
        check()
        state = entry["noisy"].detach().requires_grad_(True)
        pred = model(state, entry["sigma"], entry["conditions"], initial_context=entry["initial"])
        gradient = torch.autograd.grad((pred["reward"] * pred["valid_probability"]).sum(), state)[0]
        delta, norm32 = applied_displacement(entry["denoised"], gradient, entry["sigma"])
        norm64 = torch.linalg.vector_norm(gradient.detach().double(), dim=1)
        for j, root in enumerate(entry["roots"]):
            records.append({"root_id": root, "stratum": entry["strata"][j],
                "sigma": entry["sigma_value"], "replicate": entry["replicate"],
                "target": float(targets[entry["start"]+j]), "reward": float(pred["reward"][j].detach()),
                "valid_probability": float(pred["valid_probability"][j].detach()),
                "displacement_nonzero": bool(torch.any(delta[j] != 0)), "finite": True,
                "changed_coordinates": int(torch.count_nonzero(delta[j])),
                "maximum_actual_displacement": float(delta[j].abs().max()),
                "gradient_l2_fp32": float(norm32[j]), "gradient_l2_fp64": float(norm64[j])})
    return records, summarize_predictions(records, fit_mean=fit_mean, fit_stratum_means=fit_stratum_means)


def prepare_effect_config(template, cfg, selected, output):
    effect = deepcopy(template)
    effect.update(guide=selected["checkpoint"], training_report=selected["training_report"],
        fresh_split_seeds=cfg["fresh_split_seeds"], strength=cfg["strength"],
        bootstrap_seed=cfg["bootstrap_seed"], bootstrap_replicates=cfg["bootstrap_replicates"],
        prior_effect=cfg["prior_effect"])
    path = output / "selected-effect-config.json"
    write(path, effect)
    # Validate all hashes and all old/new roots BEFORE making solver attempts.
    _, _, roots, _ = load_effect_config(path)
    write(output / "selected-effect-roots.json", roots)
    path.chmod(0o444)
    return {"path": str(path), "sha256": file_sha(path), "roots": len(roots)}


def run(args):
    if not os.environ.get("SLURM_JOB_ID", "").isdigit() or not os.environ.get("SLURM_TMPDIR"):
        raise ValueError("Slurm compute allocation required")
    if not math.isfinite(args.wall_seconds) or args.wall_seconds <= 120:
        raise ValueError("bounded usable allocation required")
    scratch = Path("${PDE_GAV_RUN_ROOT}")
    output = args.output.absolute()
    if output == scratch or not output.is_relative_to(scratch) or output.exists() or output.is_symlink():
        raise ValueError("new specific scratch output required; no implicit resume")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    def verify_source():
        if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != revision
                or subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()):
            raise ValueError("training source must remain clean and frozen")
    verify_source()
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {"schema": SCHEMA, "execution_passed": False, "source_revision": revision,
        "job_id": os.environ["SLURM_JOB_ID"], "online_assets_replaced": False,
        "scientific_admission": False, "base_training_updates": 0, "solver_training_updates": 0,
        "new_solver_calls": 0, "runs": [], "failures": []}
    def save():
        report["wall_seconds"] = time.monotonic() - started
        write(output / "result.json", report)
    def check():
        if time.monotonic() - started > args.wall_seconds - 30:
            raise TimeoutError("training allocation cap; preserve failures and unexecuted controls")
    save()
    try:
        import torch
        from pde_agent_verifier_solver.plugins.vpfp.calibrated_solver_error_guidance import (
            make_calibrated_regressor, fit_calibration, guidance_payload)
        from pde_agent_verifier_solver.plugins.vpfp.solver_error_guidance import (
            SolverErrorGuidanceConfig, load_solver_error_regressor, _module_sha, _json_sha)
        from run_vpfp_guidance_training_debug import verify_training_draws, target_values, fixed_probe, original_validation
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one allocated CUDA device required")
        torch.set_num_threads(min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))))
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        cfg = load_config(args.config)
        template_path = ROOT / cfg["effect_template"]
        template, _, _, _ = load_effect_config(template_path)
        # Check the intended untouched confirmation inventory before spending
        # training compute, then bind the selected checkpoint again at handoff.
        fresh_template = deepcopy(template)
        fresh_template.update(fresh_split_seeds=cfg["fresh_split_seeds"], prior_effect=cfg["prior_effect"])
        fresh_path = output / "confirmation-preregistration.json"
        write(fresh_path, fresh_template)
        _, _, fresh_roots, _ = load_effect_config(fresh_path)
        report["preregistered_confirmation_roots"] = fresh_roots
        evidence = {role: template[role] for role in ("guide", "training_plan", "training_labels", "training_report")}
        evidence["prior_effect"] = cfg["prior_effect"]
        plan = json.loads(Path(evidence["training_plan"]["path"]).read_text())
        manifest = json.loads(Path(evidence["training_labels"]["path"]).read_text())
        bindings = verify_training_draws(manifest, plan)
        raw = torch.load(evidence["guide"]["path"], map_location="cpu", weights_only=True)
        gc = SolverErrorGuidanceConfig.from_dict(raw["config"])
        recipe = plan["config"]["training"]
        if (recipe["steps"] != 2000 or recipe["batch_size"] != 4 or recipe["learning_rate"] != .0002
                or gc.width != 32 or gc.depth != 2 or gc.error_scale != 1. or gc.validity_weight != .1
                or plan["config"]["seeds"]["training"] != SEEDS[0]):
            raise ValueError("original continuous regression recipe differs")
        data = {}
        for split, count in (("fit", 64), ("validation", 16)):
            rows = load_training_rows(manifest["labels"][split])
            if rows["excluded_rows"] or len(rows["root_ids"]) != count or not bool(rows["valid"].all()):
                raise ValueError("do not discard any original real solver label")
            data[split] = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in rows.items()}
        base, portable = load_round_base(plan["config"], plan["assets"], device="cuda")
        base.model.eval().requires_grad_(False)
        base.normalizer.eval().requires_grad_(False)
        if (next(base.model.parameters()).dtype != torch.float32 or gc.base_config != base.model.config
                or gc.base_checkpoint_sha256 != plan["config"]["base_checkpoint_sha256"]
                or gc.normalizer_sha256 != _json_sha(base.normalizer.to_payload())):
            raise ValueError("exact frozen FP32 Base and normalizer binding required")
        base_sha, normalizer_sha = _module_sha(base.model), _json_sha(base.normalizer.to_payload())
        # Targets depend only on immutable config, not initialized weights.
        from types import SimpleNamespace
        targets = {s: target_values(rows, SimpleNamespace(config=gc)) for s, rows in data.items()}
        fit_mean = float(targets["fit"].double().mean())
        stratum_means = training_rows_by_stratum(data["fit"], plan, targets["fit"])
        report.update(config=cfg, original_recipe=recipe, evidence=evidence,
            draw_label_semantic_bindings_verified=80, portable_codec_binding=portable,
            fit_mean=fit_mean, fit_stratum_means=stratum_means,
            statistical_unit="16 development roots; 5 sigma x 2 repeats; not 160 independent samples",
            selection="fixed step2000, both seeds pass engineering screen; variant average reward MSE then seed MSE",
            base_sha256=base_sha, normalizer_sha256=normalizer_sha)
        write(output / "frozen-plan.json", {**report, "model_config": gc.to_dict(),
            "input_bindings": bindings, "training_root_partition": {s: data[s]["root_ids"] for s in data}})
        report["stage"] = "cache_fixed_validation_denoisers"; save()
        cache = build_validation_cache(base, data["validation"], plan, seed=cfg["validation_seed"], check=check)
        for variant, (scale_enabled, init_enabled) in VARIANTS.items():
            for seed in SEEDS:
                entry = {"variant": variant, "seed": seed, "status": "pending", "validation": None}
                report["runs"].append(entry)
        for entry in report["runs"]:
            variant, seed = entry["variant"], entry["seed"]
            report["stage"] = f"{variant}/seed-{seed}"; save()
            check()
            run_output = output / variant / f"seed-{seed}"
            run_output.mkdir(parents=True)
            run_started = time.monotonic()
            model = optimizer = loaded = None
            try:
                torch.manual_seed(seed)
                scale_enabled, init_enabled = VARIANTS[variant]
                model = make_calibrated_regressor(gc, parameter_dtype=torch.float32, device="cuda",
                    scale_enabled=scale_enabled, init_enabled=init_enabled)
                calibration = fit_calibration(model, data["fit"], seed=cfg["calibration_seed"], batch_size=4)
                entry.update(status="training", calibration=calibration,
                    parameter_count=sum(p.numel() for p in model.parameters()), completed_step=0)
                write(run_output / "initial-probe.json",
                    fixed_probe(model, data["validation"], sigma=1., seed=cfg["validation_seed"]))
                optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"])
                rng = torch.Generator(device="cuda").manual_seed(seed+1)
                trace = []
                model.train()
                for step in range(1, cfg["steps"]+1):
                    check()
                    index = torch.randint(64, (4,), generator=rng, device="cuda")
                    optimizer.zero_grad(set_to_none=True)
                    value, diag = model.loss(data["fit"]["future_white"][index], data["fit"]["conditions"][index],
                        data["fit"]["initial_context"][index], data["fit"]["errors"][index],
                        data["fit"]["valid"][index], generator=rng)
                    if not torch.isfinite(value):
                        raise FloatingPointError("nonfinite regression loss")
                    value.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    entry["completed_step"] = step
                    if step <= 5 or step % 100 == 0:
                        trace.append({"step": step, "loss": float(value.detach()),
                            "reward_loss": float(diag["reward_loss"]), "validity_loss": float(diag["validity_loss"]),
                            "gradient_norm_before_clip": float(grad_norm),
                            "reward_min": float(diag["predicted_reward"].min()),
                            "reward_max": float(diag["predicted_reward"].max())})
                        write(run_output / "trace.json", trace)
                        save()
                model.eval()
                original_val = original_validation(model, data["validation"], seed)
                records, validation = evaluate_validation(model, cache, targets["validation"],
                    fit_mean=fit_mean, fit_stratum_means=stratum_means, check=check)
                write(run_output / "validation-records.json", records)
                write(run_output / "final-probe.json",
                    fixed_probe(model, data["validation"], sigma=1., seed=cfg["validation_seed"]))
                metadata = {"label_source": "actual_public_solver_macro_error", "step": 2000,
                    "base_trainable": False, "solver_trainable": False, "variant": variant, "seed": seed,
                    "validation_loss": original_val["total_loss"], "checkpoint_choice": "fixed_step_not_best"}
                payload = guidance_payload(model, label_manifest_sha256=evidence["training_labels"]["sha256"],
                    train_root_ids=data["fit"]["root_ids"], validation_root_ids=data["validation"]["root_ids"],
                    training=metadata)
                checkpoint = run_output / "fixed-step-002000.pt"
                with checkpoint.open("xb") as stream:
                    torch.save(payload, stream)
                checkpoint.chmod(0o444)
                checkpoint_binding = {"path": str(checkpoint), "sha256": file_sha(checkpoint)}
                loaded = load_solver_error_regressor(checkpoint, checkpoint_binding["sha256"], device="cuda")
                replay_records, replay_summary = evaluate_validation(loaded, cache, targets["validation"],
                    fit_mean=fit_mean, fit_stratum_means=stratum_means, check=check)
                if replay_records != records or replay_summary != validation:
                    raise ValueError("saved model changed predictions or utility gradients on reload")
                with (run_output / "continuation.pt").open("xb") as stream:
                    torch.save({"schema": SCHEMA + "_training_state", "step": 2000, "payload": payload,
                        "optimizer": optimizer.state_dict(), "training_rng": rng.get_state(),
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, stream)
                training = {"final_step": 2000, "target_step": 2000, "variant": variant, "seed": seed,
                    "label_manifest_sha256": evidence["training_labels"]["sha256"],
                    "original_validation": original_val, "validation": validation,
                    "sample_presentations": 8000, "optimizer": "unchanged_AdamW_lr2e-4_clip1",
                    "checkpoint": checkpoint_binding, "reload_predictions_and_gradient_diagnostics_exact": True,
                    "gpu_wall_seconds": time.monotonic()-run_started}
                training_path = run_output / "training.json"
                write(training_path, training); training_path.chmod(0o444)
                entry.update(status="completed", validation=validation, checkpoint=checkpoint_binding,
                    training_report={"path": str(training_path), "sha256": file_sha(training_path)},
                    sample_presentations=8000, gpu_wall_seconds=training["gpu_wall_seconds"],
                    reload_predictions_and_gradient_diagnostics_exact=True)
            except Exception as error:
                entry.update(status="failed", failure={"type": type(error).__name__, "message": str(error),
                    "traceback": traceback.format_exc()}, gpu_wall_seconds=time.monotonic()-run_started)
                write(run_output / "failure.json", entry)
            finally:
                model = optimizer = loaded = None
                torch.cuda.empty_cache()
                save()
        verify_source()
        selection = select_variant(report["runs"])
        report.update(selection=selection, source_final_verified=True, evidence_final_verified=True,
            base_final_verified=True, normalizer_final_verified=True)
        write(output / "selection.json", selection)
        if selection["selected"] is not None:
            report["selected_effect"] = prepare_effect_config(template, cfg, selection["selected"], output)
            report["next_stage"] = "fresh_32_root_paired_solver_test"
        else:
            report["next_stage"] = "skipped_no_eligible_candidate"
        report.update(stage="complete", execution_passed=True)
    except Exception as error:
        report["failures"].append({"stage": report.get("stage"), "type": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc()})
    finally:
        save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vpfp_guidance_calibration_20260908.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, default=900.)
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps({key: result.get(key) for key in ("execution_passed", "stage", "next_stage", "failures")}))
    return 0 if result["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
