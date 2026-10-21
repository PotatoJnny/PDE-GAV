"""Diagnose continuous-reward regression without changing production assets.

Replay only the original first 500 updates on retained real solver labels.
No new solver labels, diffusion draws, loss replacement or deployment. Extra
observations use independent RNGs and do not consume the training stream.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
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
from run_vpfp_strong_solver_guidance_round import file_sha, json_sha, write, load_training_rows

LAST_SHA = "94894b507a69d081f06c4f6d20de07b11b22fc300ee638be540fef524d9d0130"
TRACE_STEPS = frozenset((1, 2, 4, 8, 16, 32, 64, 128, 200, 250, 275, 300, 325, 350, 375, 400, 450, 500))
SIGMAS = (.002, .1, 1., 2., 10., 80.)


def gradient_summary(named, gradients):
    import torch
    groups = {}
    total = 0.
    for (name, parameter), gradient in zip(named, gradients):
        group = name.split(".")[0]
        row = groups.setdefault(group, {"squared_l2": 0., "max_abs": 0., "nonzero_elements": 0, "elements": 0})
        row["elements"] += parameter.numel()
        if gradient is None:
            continue
        g = gradient.detach().double()
        if not bool(torch.isfinite(g).all()):
            raise FloatingPointError("nonfinite diagnostic gradient")
        squared = float(g.square().sum())
        total += squared
        row["squared_l2"] += squared
        row["max_abs"] = max(row["max_abs"], float(g.abs().max()))
        row["nonzero_elements"] += int(torch.count_nonzero(g))
    for row in groups.values():
        row["l2"] = math.sqrt(row.pop("squared_l2"))
    return {"l2_fp64": math.sqrt(total), "groups": groups}


def loss_gradient_comparison(model, prediction, target, valid):
    import torch
    from torch.nn import functional as F
    named = list(model.named_parameters())
    parameters = [p for _, p in named]
    reward_loss = (prediction["reward"][valid] - target[valid]).square().mean()
    validity_loss = model.config.validity_weight * F.binary_cross_entropy_with_logits(
        prediction["validity_logit"], valid.to(prediction["reward"]))
    reward = torch.autograd.grad(reward_loss, parameters, retain_graph=True, allow_unused=True)
    validity = torch.autograd.grad(validity_loss, parameters, retain_graph=True, allow_unused=True)
    dot = sum(float((a.detach().double() * b.detach().double()).sum())
              for a, b in zip(reward, validity) if a is not None and b is not None)
    r, v = gradient_summary(named, reward), gradient_summary(named, validity)
    denominator = r["l2_fp64"] * v["l2_fp64"]
    return {"reward": r, "weighted_validity": v, "cosine": dot / denominator if denominator else None,
            "weighted_validity_over_reward": v["l2_fp64"] / r["l2_fp64"] if r["l2_fp64"] else None}


def target_values(rows, model):
    from pde_agent_verifier_solver.plugins.vpfp.solver_error_guidance import bounded_error_targets
    return bounded_error_targets(rows["errors"], rows["valid"], prototype=rows["future_white"],
                                 error_scale=model.config.error_scale)[0]


def fixed_probe(model, rows, *, sigma, seed, count=4, variant="original"):
    import torch
    from vpfp_guidance_training_diagnostics import ActivationProbe, tensor_stats
    # These are explicitly diagnostic branch knockouts, not valid generated
    # questions, alternative training data, or evidence of a working fix.
    clean, context = rows["future_white"][:count], rows["initial_context"][:count]
    if variant == "zero_initial":
        context = torch.zeros_like(context)
    elif variant == "zero_future":
        clean = torch.zeros_like(clean)
    elif variant != "original":
        raise ValueError("unknown diagnostic branch knockout")
    rng = torch.Generator(device=clean.device).manual_seed(seed)
    noise = torch.randn(clean.shape, dtype=clean.dtype, device=clean.device, generator=rng)
    noisy = (clean + sigma * noise).detach().requires_grad_(True)
    level = clean.new_full((len(clean),), sigma)
    retained = {}
    def observe_head(_module, inputs, output):
        width = inputs[0].shape[1] // 2
        retained["pooled_mean"] = tensor_stats(inputs[0][:, :width])
        retained["pooled_square"] = tensor_stats(inputs[0][:, width:])
        retained["reward_logits"] = output[:, 0].detach().cpu().tolist()
        retained["validity_logits"] = output[:, 1].detach().cpu().tolist()
    hook = model.head.register_forward_hook(observe_head)
    try:
        with ActivationProbe(model) as probe:
            prediction = model(noisy, level, rows["conditions"][:count], initial_context=context)
            utility = prediction["reward"] * prediction["valid_probability"]
            reward_gradient = torch.autograd.grad(prediction["reward"].sum(), noisy, retain_graph=True)[0]
            utility_gradient = torch.autograd.grad(utility.sum(), noisy)[0]
            result = {"sigma": sigma, "variant": variant, "root_ids": rows["root_ids"][:count],
                "target_rewards": target_values(rows, model)[:count].detach().cpu().tolist(),
                "predicted_rewards": prediction["reward"].detach().cpu().tolist(),
                "valid_probabilities": prediction["valid_probability"].detach().cpu().tolist(),
                "reward_input_gradient": tensor_stats(reward_gradient),
                "utility_input_gradient": tensor_stats(utility_gradient),
                "utility_gradient_l2_fp64_by_root": torch.linalg.vector_norm(utility_gradient.detach().double(), dim=1).cpu().tolist(),
                "utility_gradient_l2_fp32_by_root": torch.linalg.vector_norm(utility_gradient.detach(), dim=1).cpu().tolist(),
                "activations": probe.snapshot(), **retained}
    finally:
        hook.remove()
    return result


def original_validation(model, rows, seed):
    import torch
    rng = torch.Generator(device=rows["future_white"].device).manual_seed(seed + 2)
    with torch.no_grad():
        value, diagnostics = model.loss(rows["future_white"], rows["conditions"], rows["initial_context"],
                                       rows["errors"], rows["valid"], generator=rng)
    return {"total_loss": float(value), "reward_loss": float(diagnostics["reward_loss"]),
        "validity_loss": float(diagnostics["validity_loss"]),
        "predicted_rewards": diagnostics["predicted_reward"].cpu().tolist(),
        "target_rewards": diagnostics["reward_target"].cpu().tolist(),
        "sigma": diagnostics["sigma"].cpu().tolist()}


def verify_training_draws(manifest, plan):
    import torch
    from vpfp_strong_guidance_rollouts import _validate_draw
    count = 0
    bindings = []
    for split in ("fit", "validation"):
        expected = {row["slot_id"]: row for row in plan["roots"][split]}
        for slot, entry in manifest["labels"][split].items():
            row = expected[slot]
            if entry["row"] != row or entry["root"] != row:
                raise ValueError("label manifest root/row mismatch")
            item = entry["private_draw"]
            draw = torch.load(item["path"], map_location="cpu", weights_only=True)
            if draw["row"] != row:
                raise ValueError("tensor draw row differs from label")
            _validate_draw(draw, row)
            count += 1
            bindings.extend((item, {"path": entry["result_path"], "sha256": entry["result_sha256"]}))
    if count != 80:
        raise ValueError("original 64 fit and 16 validation roots required")
    return bindings


def run(args):
    if not os.environ.get("SLURM_JOB_ID", "").isdigit() or not os.environ.get("SLURM_TMPDIR"):
        raise ValueError("Slurm compute allocation required")
    output = args.output.absolute()
    scratch = Path("${PDE_GAV_RUN_ROOT}")
    if (output == scratch or not output.is_relative_to(scratch) or output.exists() or output.is_symlink()
            or not math.isfinite(args.wall_seconds) or args.wall_seconds < 180):
        raise ValueError("fresh specific scratch output and bounded allocation required")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    def verify_source():
        if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != revision
                or subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()):
            raise ValueError("diagnostic source changed")
    verify_source()
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {"schema": "vpfp_continuous_guidance_training_debug_v1", "execution_passed": False,
        "source_revision": revision, "job_id": os.environ["SLURM_JOB_ID"],
        "production_training_updates": 0, "diagnostic_replay_updates": 0,
        "new_diffusion_draws": 0, "new_solver_calls": 0, "production_assets_changed": False,
        "regression_target_changed": False, "failures": [], "trace": []}
    def save():
        report["wall_seconds"] = time.monotonic() - started
        write(output / "result.json", report)
    def check():
        if time.monotonic() - started > args.wall_seconds - 45:
            raise TimeoutError("diagnostic allocation ending; no automatic retry or resume")
    save()
    try:
        import torch
        from pde_agent_verifier_solver.plugins.vpfp.solver_error_guidance import (
            SolverErrorGuidanceConfig, make_solver_error_regressor, regressor_from_payload, _module_sha)
        from vpfp_guidance_training_diagnostics import ActivationProbe, tensor_stats
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one allocated GPU required")
        torch.set_num_threads(min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))))
        # Keep original torch process defaults; record rather than silently
        # replace numerical settings with the later effect-test recipe.
        report["environment"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32, "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
        effect = json.loads(args.config.read_text())
        evidence = {role: effect[role] for role in ("guide", "training_plan", "training_labels", "training_report")}
        evidence["last"] = {"path": str(Path(evidence["guide"]["path"]).with_name("last.pt")), "sha256": LAST_SHA}
        plan = json.loads(Path(evidence["training_plan"]["path"]).read_text())
        manifest = json.loads(Path(evidence["training_labels"]["path"]).read_text())
        original_training = json.loads(Path(evidence["training_report"]["path"]).read_text())
        if original_training["final_step"] != 2000 or original_training["label_manifest_sha256"] != evidence["training_labels"]["sha256"]:
            raise ValueError("wrong completed original training")
        report["evidence"] = evidence
        report["old_source_revision"] = plan["source_revision"]
        module = "src/pde_agent_verifier_solver/plugins/vpfp/solver_error_guidance.py"
        old_code = subprocess.check_output(["git", "show", plan["source_revision"] + ":" + module], cwd=ROOT)
        if old_code != (ROOT / module).read_bytes():
            raise ValueError("production Guidance source differs from old training")
        report["production_guidance_source_identical_to_original"] = True
        bindings = verify_training_draws(manifest, plan)
        report["draw_label_semantic_bindings_verified"] = 80
        report["stage"] = "load_retained_labels"; save()
        data = {split: load_training_rows(manifest["labels"][split]) for split in ("fit", "validation")}
        for split, rows in data.items():
            if rows["excluded_rows"] or len(rows["root_ids"]) != {"fit": 64, "validation": 16}[split]:
                raise ValueError("diagnosis must not drop original valid labels")
            data[split] = {key: value.to("cuda") if isinstance(value, torch.Tensor) else value
                           for key, value in rows.items()}
        raw = torch.load(evidence["guide"]["path"], map_location="cpu", weights_only=True)
        gc = SolverErrorGuidanceConfig.from_dict(raw["config"])
        recipe, seed = plan["config"]["training"], plan["config"]["seeds"]["training"]
        if gc.validity_weight != recipe["validity_weight"] or gc.error_scale != recipe["error_scale"]:
            raise ValueError("original loss config mismatch")
        report.update(recipe=recipe, training_seed=seed,
            scope="diagnostic first-500-update replay; no usable new guidance model or effectiveness claim",
            model_config=raw["config"])
        best = regressor_from_payload(raw, device="cuda")
        targets = {split: target_values(rows, best) for split, rows in data.items()}
        fit_mean = float(targets["fit"].double().mean())
        report["target_baselines"] = {"fit_mean": fit_mean,
            "zero_validation_mse": float(targets["validation"].double().square().mean()),
            "fit_mean_validation_mse": float((targets["validation"].double() - fit_mean).square().mean()),
            "fit": tensor_stats(targets["fit"]), "validation": tensor_stats(targets["validation"])}
        report["inputs"] = {split: {key: tensor_stats(rows[key]) for key in
            ("initial_context", "future_white", "conditions", "errors")} for split, rows in data.items()}
        report["stage"] = "frozen_checkpoint_probes"; save()
        for name in ("guide", "last"):
            check()
            payload = raw if name == "guide" else torch.load(evidence[name]["path"], map_location="cpu", weights_only=True)
            if payload["label_manifest_sha256"] != evidence["training_labels"]["sha256"] or payload["config"] != raw["config"]:
                raise ValueError("frozen guide metadata differs")
            model = best if name == "guide" else regressor_from_payload(payload, device="cuda")
            checkpoint = {"step": payload["training"]["step"],
                "original_validation": original_validation(model, data["validation"], seed),
                "probes": [fixed_probe(model, data["validation"], sigma=s, seed=seed+1234) for s in SIGMAS]}
            write(output / (name + "-diagnostics.json"), checkpoint)
            report[name + "_original_validation"] = checkpoint["original_validation"]
            if name != "guide":
                del model
            save()
        # Exact initialization and training RNG order from train_surrogate.
        torch.manual_seed(seed)
        model = make_solver_error_regressor(gc, parameter_dtype=next(best.parameters()).dtype, device="cuda")
        initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"])
        training_rng = torch.Generator(device="cuda").manual_seed(seed + 1)
        original_rng_state = training_rng.get_state().clone()
        initial = {"state_sha256": _module_sha(model),
            "original_validation": original_validation(model, data["validation"], seed),
            "probes": [fixed_probe(model, data["validation"], sigma=s, seed=seed+1234) for s in SIGMAS],
            "branch_knockouts": [fixed_probe(model, data["validation"], sigma=1., seed=seed+1234, variant=variant)
                                for variant in ("zero_initial", "zero_future")]}
        # Inference-only initialization counterfactual: change just the reward
        # output row/bias. No optimizer steps, no claimed trained fix.
        counterfactual = deepcopy(model)
        with torch.no_grad():
            counterfactual.head[-1].weight[0].zero_()
            counterfactual.head[-1].bias[0].fill_(math.log(fit_mean / (1 - fit_mean)))
        initial["calibrated_reward_row_no_training"] = fixed_probe(
            counterfactual, data["validation"], sigma=1., seed=seed+1234)
        del counterfactual
        if not torch.equal(training_rng.get_state(), original_rng_state):
            raise ValueError("diagnostics consumed training RNG")
        write(output / "initial-diagnostics.json", initial)
        report["initial_original_validation"] = initial["original_validation"]
        report["stage"] = "original_recipe_replay"; save()
        fit = data["fit"]
        for step in range(1, 501):
            check()
            model.train()
            index = torch.randint(len(fit["root_ids"]), (recipe["batch_size"],), generator=training_rng, device="cuda")
            optimizer.zero_grad(set_to_none=True)
            detailed = step in TRACE_STEPS
            retained = {}
            def capture(_module, _inputs, prediction):
                retained["prediction"] = prediction
            handle = model.register_forward_hook(capture) if detailed else None
            try:
                with ActivationProbe(model) if detailed else nullcontext() as probe:
                    value, diagnostics = model.loss(fit["future_white"][index], fit["conditions"][index],
                        fit["initial_context"][index], fit["errors"][index], fit["valid"][index], generator=training_rng)
                    if not bool(torch.isfinite(value)):
                        raise FloatingPointError("original replay has nonfinite loss")
                    row = {"step": step, "root_ids": [fit["root_ids"][i] for i in index.cpu().tolist()],
                        "loss_before_update": float(value), "reward_loss": float(diagnostics["reward_loss"]),
                        "validity_loss": float(diagnostics["validity_loss"]),
                        "sigma": diagnostics["sigma"].cpu().tolist(),
                        "targets": diagnostics["reward_target"].cpu().tolist(),
                        "predictions": diagnostics["predicted_reward"].cpu().tolist()}
                    if detailed:
                        row["activations"] = probe.snapshot()
                        row["loss_gradients"] = loss_gradient_comparison(
                            model, retained["prediction"], diagnostics["reward_target"], fit["valid"][index])
                    value.backward()
                    if detailed:
                        row["combined_gradient_fp64"] = gradient_summary(list(model.named_parameters()), [p.grad for p in model.parameters()])
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    row["clip_input_norm_fp32"] = float(norm)
                    row["clip_coefficient"] = min(1., 1. / (float(norm) + 1e-6))
                    optimizer.step()
            finally:
                if handle is not None:
                    handle.remove()
                retained.clear()
            report["diagnostic_replay_updates"] = step
            if step in TRACE_STEPS:
                model.eval()
                rng_before_probe = training_rng.get_state().clone()
                row["validation_after_update"] = original_validation(model, data["validation"], seed)
                write(output / "probes" / f"step-{step:04d}.json", {
                    "step": step, "probe": fixed_probe(model, data["validation"], sigma=1., seed=seed+1234)})
                if not torch.equal(training_rng.get_state(), rng_before_probe):
                    raise ValueError("probe consumed original training RNG")
            report["trace"].append(row)
            if step % 10 == 0 or detailed:
                report["stage"] = f"original_recipe_replay_{step}"; save()
        report["replay_step500_validation"] = original_validation(model, data["validation"], seed)
        # Comparing saved weights does not presume cross-GPU bitwise training
        # reproduction; explicitly report differences and observed loss traces.
        differences = [(model.state_dict()[name].detach().cpu().double() - value.double()).abs().max().item()
                       for name, value in raw["model"].items()]
        report["replay_vs_old_best_step500"] = {"bitwise_equal": all(value == 0 for value in differences),
                                              "maximum_state_abs_difference": max(differences),
                                              "old_recorded_validation_loss": raw["training"]["validation_loss"]}
        report["original_state_sha256"] = initial["state_sha256"]
        report["replay_state_sha256"] = _module_sha(model)
        verify_source()
        report.update(stage="complete", execution_passed=True, source_final_verified=True,
                      evidence_final_verified=True, training_rng_not_consumed_by_probes=True)
    except Exception as error:
        report["failures"].append({"stage": report.get("stage"), "type": type(error).__name__,
                                  "message": str(error), "traceback": traceback.format_exc()})
    finally:
        save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vpfp_strong_guidance_effect_demand_20260908.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, default=1500.)
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps({k: result.get(k) for k in ("execution_passed", "stage", "diagnostic_replay_updates", "failures")}))
    return 0 if result["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
