"""Evaluate a frozen Guidance against Base using the accelerated public Oracle.

No optimizer, training, strength selection, or online deployment is performed.
All numerical work runs inside one Slurm GPU allocation. Every original root
and generation attempt is retained; solver failures are not difficulty gains.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from run_vpfp_strong_solver_guidance_round import file_sha, json_sha, load_round_base, validate_config, write
from vpfp_strong_guidance_rollouts import freeze_round_assets, registered_roots

SCHEMA = "vpfp_frozen_guidance_effect_v1"
EXPONENTIAL_OBJECTIVE = "exponential_reward_log_h_v1"


def _evidence_roles(cfg):
    return ("guide", "training_plan", "training_labels", "training_report") + (
        ("prior_effect",) if "prior_effect" in cfg else ()) + (
        ("reference_guide", "reference_training_report") if "reference_guide" in cfg else ())


def _evidence_items(cfg):
    items = [(role, cfg[role]) for role in _evidence_roles(cfg)]
    prior = cfg.get("prior_effects", [])
    if not isinstance(prior, list):
        raise ValueError("prior_effects must be a list of SHA-bound reports")
    return items + [(f"prior_effects_{index:04d}", item) for index, item in enumerate(prior)]


def _training_report(item, label_sha, *, role="Guidance"):
    report = json.loads(Path(item["path"]).read_text())
    if (type(report.get("final_step")) is not int or report["final_step"] < 1
            or report["final_step"] != report.get("target_step")):
        raise ValueError(role + " training has not completed")
    if report.get("label_manifest_sha256") != label_sha:
        raise ValueError(role + " training labels changed")
    return report


def validate_checkpoint_training(raw, training, label_sha, old_plan, *, role="guide"):
    """Verify both models use real labels and the same frozen root partition."""
    step = raw.get("training", {}).get("step")
    if (raw.get("label_manifest_sha256") != label_sha
            or raw.get("training", {}).get("label_source") != "actual_public_solver_macro_error"
            or type(step) is not int or not 0 < step <= training["final_step"]):
        raise ValueError(role + " checkpoint does not bind completed actual-solver training")
    for key, split in (("train_root_ids", "fit"), ("validation_root_ids", "validation")):
        actual = raw.get("root_partition", {}).get(key)
        expected = {row["slot_id"] for row in old_plan["roots"][split]}
        if not isinstance(actual, list) or len(actual) != len(expected) or set(actual) != expected:
            raise ValueError(role + " actual training roots disagree with frozen plan: " + key)
    return step


def effect_arm_order(index, *, with_reference=False):
    if with_reference:
        arms = ("base", "control", "guided")
        offset = index % 3
        return arms[offset:] + arms[:offset]
    return ("base", "guided") if index % 2 == 0 else ("guided", "base")


def remap_comparison_pairs(pairs, *, baseline_arm, guided_arm):
    """Reuse the same independent roots; never turn three arms into 96 roots."""
    if baseline_arm == guided_arm or {baseline_arm, guided_arm} - {"base", "control", "guided"}:
        raise ValueError("comparison requires two distinct registered arms")
    return [{"root_seed": row["root_seed"], "stratum": row["stratum"],
             "base": deepcopy(row.get(baseline_arm)), "guided": deepcopy(row.get(guided_arm))}
            for row in pairs]


def validate_loaded_guidance(cfg, guide, reference=None):
    if cfg.get("guidance_objective") == EXPONENTIAL_OBJECTIVE:
        if (getattr(guide, "beta", None) != cfg["beta"]
                or not callable(getattr(guide, "score_potential", None))
                or getattr(guide, "expected_sampling_strength", None) != cfg["strength"]):
            raise ValueError("loaded exponential guide does not bind beta and unit score multiplier")
        if (reference is None or hasattr(reference, "beta")
                or guide.config.to_dict() != reference.config.to_dict()):
            raise ValueError("reference must use the old objective and same frozen guide/base config")


def _prior_effect_roots(item):
    """Exclude every historically registered pair, including failed arms."""
    report = json.loads(Path(item["path"]).read_text())
    if (not isinstance(report, dict) or report.get("schema") != SCHEMA
            or not isinstance(report.get("pairs"), list) or not report["pairs"]):
        raise ValueError("prior_effect must contain a nonempty historical effect pair inventory")
    roots = []
    for row in report["pairs"]:
        seed = row.get("root_seed") if isinstance(row, dict) else None
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("prior_effect pairs require explicit integer root_seed values")
        roots.append(seed)
    if len(roots) != len(set(roots)):
        raise ValueError("prior_effect pair root seeds must be unique")
    return tuple(roots)


def load_effect_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg.get("schema") != SCHEMA or cfg.get("expected_pairs") != 32:
        raise ValueError("frozen 32-pair Guidance effect configuration required")
    if (type(cfg.get("strength")) not in (int, float)
            or not math.isfinite(cfg["strength"]) or cfg["strength"] <= 0):
        raise ValueError("one fixed positive Guidance strength required")
    if cfg.get("bootstrap_replicates") != 10000 or type(cfg.get("bootstrap_seed")) is not int:
        raise ValueError("fixed bootstrap seed and 10000 replicates required")
    raw_path = Path(cfg["round_config"])
    if not raw_path.is_absolute():
        raw_path = ROOT / raw_path
    round_cfg = validate_config(json.loads(raw_path.read_text()))
    if round_cfg.get("public_access_mode") != "demand_legacy_arrays_v1":
        raise ValueError("current frozen array-only solver requires explicit demand adapter")
    exponential = cfg.get("guidance_objective") == EXPONENTIAL_OBJECTIVE
    if "guidance_objective" in cfg and not exponential:
        raise ValueError("unsupported explicit Guidance objective")
    if exponential:
        if (type(cfg.get("beta")) not in (int, float) or cfg["beta"] != 2.
                or cfg["strength"] != 1.):
            raise ValueError("exponential Guidance fixes beta=2 and strength=1; do not apply beta twice")
        if (type(cfg.get("reference_strength")) not in (int, float)
                or cfg["reference_strength"] != 2.
                or cfg["reference_strength"] != max(round_cfg["sampling"]["selection_strengths"])
                or "reference_guide" not in cfg or "reference_training_report" not in cfg):
            raise ValueError("exponential comparison requires the frozen original reference at strength=2")
    else:
        if any(key in cfg for key in ("reference_guide", "reference_training_report", "reference_strength", "beta")):
            raise ValueError("reference/beta settings require the explicit exponential objective")
        if cfg["strength"] not in round_cfg["sampling"]["selection_strengths"]:
            raise ValueError("use a previously registered strength, not a new tuning search")
        if cfg["strength"] != max(round_cfg["sampling"]["selection_strengths"]):
            raise ValueError("this comparison fixes the original maximum nonzero strength")
    for role, item in _evidence_items(cfg):
        if not isinstance(item, dict) or not {"path", "sha256"}.issubset(item):
            raise ValueError("frozen " + role + " requires path and sha256")
    old_plan = json.loads(Path(cfg["training_plan"]["path"]).read_text())
    training = _training_report(cfg["training_report"], cfg["training_labels"]["sha256"])
    if exponential:
        _training_report(cfg["reference_training_report"], cfg["training_labels"]["sha256"], role="reference Guidance")
    for key in ("base_checkpoint_sha256", "solver_sha256", "sampling", "condition_strata", "portable_codec"):
        if round_cfg[key] != old_plan["config"][key]:
            raise ValueError("evaluation changes frozen model/solver/sampling setting: " + key)
    round_cfg = deepcopy(round_cfg)
    round_cfg["seeds"] = dict(cfg["fresh_split_seeds"])
    round_cfg["experiment_id"] = "vpfp-frozen-guidance-demand-effect-20260908"
    prior_items = ([cfg["prior_effect"]] if "prior_effect" in cfg else []) + cfg.get("prior_effects", [])
    additional_roots = tuple(seed for item in prior_items for seed in _prior_effect_roots(item))
    roots = fresh_confirmation_roots(round_cfg, old_plan, additional_roots=additional_roots)
    return cfg, round_cfg, roots, training


def fresh_confirmation_roots(round_cfg, old_plan, additional_roots=()):
    registered = registered_roots(round_cfg)
    old = {row["root_seed"] for rows in old_plan["roots"].values() for row in rows}
    additional_roots = tuple(additional_roots)
    if any(type(seed) is not int or not 0 <= seed < 2**63 for seed in additional_roots):
        raise ValueError("additional prior-effect roots must be integer seeds")
    old.update(additional_roots)
    current = {row["root_seed"] for rows in registered.values() for row in rows}
    if old & current:
        raise ValueError("new effect roots overlap prior fit/validation/selection/confirmation or prior effect")
    roots = registered["confirmation"]
    if len(roots) != 32 or any(sum(row["stratum"] == s for row in roots) != 4 for s in range(8)):
        raise ValueError("confirmation must contain four new independent roots per stratum")
    return roots


def paired_draw_diagnostics(base, guided):
    import torch
    for key in ("root_seeds", "known_prior_seeds", "future_noise_seeds"):
        if base[key] != guided[key]:
            raise ValueError("pair changed shared sampling stream: " + key)
    for key in ("initial_context", "equation_conditions"):
        if not torch.equal(base[key], guided[key]):
            raise ValueError("pair changed physical condition or known initial state: " + key)
    result = {"shared_initial_context_exact": True, "shared_conditions_exact": True,
              "shared_prior_and_future_noise_seeds": True}
    for key in ("full", "future_white"):
        left, right = base[key].detach(), guided[key].detach()
        if left.shape != right.shape or not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise ValueError("paired draw is not finite and shape-compatible")
        delta = right.double() - left.double()
        result[key] = {"bitwise_equal": bool(torch.equal(left, right)),
            "max_abs_difference": float(delta.abs().max()),
            "rms_difference": float(delta.square().mean().sqrt()),
            "l2_difference": float(torch.linalg.vector_norm(delta)),
            "base_l2": float(torch.linalg.vector_norm(left.double()))}
    return result


class HeadDiagnostic:
    """Observe actual sampling logits without changing outputs or gradients."""
    def __init__(self, *, kind="reward"):
        if kind not in ("reward", "log_desirability"):
            raise ValueError("unsupported Guidance head diagnostic kind")
        self.kind = kind
        self.rows = []

    def __call__(self, _module, _inputs, output):
        import torch
        if self.kind == "log_desirability":
            value = output["log_desirability"].detach()
            desirability = output["desirability"].detach()
            valid_probability = output["valid_probability"].detach()
            if not bool(torch.isfinite(value).all() & torch.isfinite(desirability).all()
                        & torch.isfinite(valid_probability).all()):
                return
            self.rows.append({"log_desirability_min": float(value.min()),
                "log_desirability_max": float(value.max()),
                "log_desirability_mean": float(value.mean()),
                "desirability_min": float(desirability.min()),
                "desirability_max": float(desirability.max()),
                "valid_probability_min": float(valid_probability.min()),
                "valid_probability_max": float(valid_probability.max())})
            return
        value = output.detach()
        if not bool(torch.isfinite(value).all()):
            return
        reward = value[:, 0].sigmoid()
        self.rows.append({"reward_logit_min": float(value[:, 0].min()),
            "reward_logit_max": float(value[:, 0].max()),
            "validity_logit_min": float(value[:, 1].min()),
            "validity_logit_max": float(value[:, 1].max()),
            "predicted_reward_min": float(reward.min()),
            "predicted_reward_max": float(reward.max()),
            "all_reward_exact_zero": bool(torch.all(reward == 0)),
            "all_reward_exact_one": bool(torch.all(reward == 1))})

    def summary(self):
        if self.kind == "log_desirability":
            result = {"kind": self.kind, "observed_calls": len(self.rows)}
            for key in ("log_desirability_min", "desirability_min", "valid_probability_min"):
                result[key] = min((row[key] for row in self.rows), default=None)
            for key in ("log_desirability_max", "desirability_max", "valid_probability_max"):
                result[key] = max((row[key] for row in self.rows), default=None)
            means = [row["log_desirability_mean"] for row in self.rows]
            result["log_desirability_call_mean_min"] = min(means, default=None)
            result["log_desirability_call_mean_max"] = max(means, default=None)
            result["log_desirability_mean_over_calls"] = sum(means) / len(means) if means else None
            return result
        if not self.rows:
            return {"observed_calls": 0}
        result = {"observed_calls": len(self.rows),
            "all_reward_exact_zero_calls": sum(x["all_reward_exact_zero"] for x in self.rows),
            "all_reward_exact_one_calls": sum(x["all_reward_exact_one"] for x in self.rows)}
        for key in ("reward_logit_min", "validity_logit_min", "predicted_reward_min"):
            result[key] = min(row[key] for row in self.rows)
        for key in ("reward_logit_max", "validity_logit_max", "predicted_reward_max"):
            result[key] = max(row[key] for row in self.rows)
        return result


def freeze_evidence(cfg, output):
    copied = {}
    for role, item in _evidence_items(cfg):
        source = Path(item["path"])
        target = output / "frozen-assets" / (role + source.suffix)
        with source.open("rb") as reader, target.open("xb") as writer:
            shutil.copyfileobj(reader, writer, 1024 * 1024)
        target.chmod(0o444)
        copied[role] = {"path": str(target), "source_path": str(source), "sha256": item["sha256"]}
    return copied


def run(args):
    if not os.environ.get("SLURM_JOB_ID", "").isdigit() or not os.environ.get("SLURM_TMPDIR"):
        raise ValueError("Slurm compute allocation required")
    if not math.isfinite(args.wall_seconds) or args.wall_seconds <= 120:
        raise ValueError("bounded usable allocation required")
    output = args.output.absolute()
    scratch = Path("${PDE_GAV_RUN_ROOT}")
    if output == scratch or not output.is_relative_to(scratch) or output.exists() or output.is_symlink():
        raise ValueError("new specific scratch output directory required; no automatic retries/resume")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    def verify_source():
        if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != revision
                or subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()):
            raise ValueError("evaluation source must remain clean and frozen")
    verify_source()
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {"schema": SCHEMA, "execution_passed": False, "source_revision": revision,
        "job_id": os.environ["SLURM_JOB_ID"], "training_updates": 0, "online_assets_replaced": False,
        "scientific_admission": False, "scope": "frozen_guide_effect_on_current_solver_only",
        "solver_interface": "unchanged_GRID_solver_with_explicit_lazy_array_compatibility",
        "new_diffusion_draw_attempts": 0, "pairs": [], "failures": []}
    def save():
        report["wall_seconds"] = time.monotonic() - started
        write(output / "result.json", report)
    def check():
        if time.monotonic() - started >= args.wall_seconds - 45:
            raise TimeoutError("allocation limit; retain completed and pending original attempts")
    save()
    try:
        import torch
        from pde_agent_verifier_solver.plugins.vpfp.frozen_solver_reward import load_frozen_solver
        from pde_agent_verifier_solver.plugins.vpfp.solver_error_guidance import (
            draw_frozen_ic40, GuidedIC40Sampler, load_solver_error_regressor,
        )
        from vpfp_strong_guidance_rollouts import build_rollout_runtime, evaluate_draw
        from vpfp_guidance_effect_summary import summarize_pairs
        from run_plasma_canonical_smoke import _forbid_online_vpfp_generation
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one allocated CUDA device required")
        torch.set_num_threads(min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))))
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        cfg, round_cfg, roots, training = load_effect_config(args.config)
        with_reference = "reference_guide" in cfg
        arms = ("base", "control", "guided") if with_reference else ("base", "guided")
        report.update(config_sha256=json_sha(cfg), strength=cfg["strength"],
            strength_choice=("beta_is_in_training_target; unit_log_h_score_multiplier"
                             if with_reference else "fixed_original_maximum_nonzero_setting; not_old_selected_winner"),
            training_completed_step=training["final_step"], bootstrap_seed=cfg["bootstrap_seed"],
            bootstrap_replicates=cfg["bootstrap_replicates"], public_access_mode=round_cfg["public_access_mode"])
        if with_reference:
            report.update(guidance_objective=cfg["guidance_objective"], beta=cfg["beta"],
                reference_strength=cfg["reference_strength"], expected_draw_attempts=96,
                independent_root_count=32, statistical_unit="32 whole paired roots, not 96 independent draws")
        report["pairs"] = [{**deepcopy(row), **{arm: {"status": "pending", "valid": False,
            "error": None, "attempt_count": 0} for arm in arms}} for row in roots]
        save()
        assets, native = freeze_round_assets(round_cfg, output)
        evidence = freeze_evidence(cfg, output)
        plan = {"schema": SCHEMA, "config": cfg, "round_config": round_cfg, "roots": roots,
            "source_revision": revision, "assets": assets, "evidence": evidence,
            "statistical_unit": "whole_root_paired; stratified8; four_roots_per_stratum",
            "execution_order": ("cyclic_base_control_guided_offset_index_modulo_3" if with_reference
                                else "base_then_guided_on_even_index; reversed_on_odd_index"),
            "executed_split": "fresh_confirmation_only", "training_updates": 0}
        write(output / "frozen-plan.json", plan)
        report["frozen_plan_sha256"] = file_sha(output / "frozen-plan.json")
        base, portable = load_round_base(round_cfg, assets, device="cuda")
        base.model.eval().requires_grad_(False)
        base.normalizer.eval().requires_grad_(False)
        raw = torch.load(evidence["guide"]["path"], map_location="cpu", weights_only=True)
        old_plan = json.loads(Path(evidence["training_plan"]["path"]).read_text())
        report["evaluated_guidance_checkpoint_step"] = validate_checkpoint_training(
            raw, training, evidence["training_labels"]["sha256"], old_plan)
        report["actual_training_root_partition_verified"] = True
        report["portable_codec_binding"] = portable
        guide = load_solver_error_regressor(evidence["guide"]["path"], evidence["guide"]["sha256"], device="cuda")
        reference = None
        if with_reference:
            reference_training = _training_report(evidence["reference_training_report"],
                evidence["training_labels"]["sha256"], role="reference Guidance")
            reference_raw = torch.load(evidence["reference_guide"]["path"], map_location="cpu", weights_only=True)
            report["evaluated_reference_checkpoint_step"] = validate_checkpoint_training(reference_raw,
                reference_training, evidence["training_labels"]["sha256"], old_plan, role="reference guide")
            report["reference_training_completed_step"] = reference_training["final_step"]
            reference = load_solver_error_regressor(evidence["reference_guide"]["path"],
                evidence["reference_guide"]["sha256"], device="cuda")
        validate_loaded_guidance(cfg, guide, reference)
        settings = round_cfg["sampling"]
        sampler = GuidedIC40Sampler(base, guide, strength=cfg["strength"],
            max_gradient_norm=settings["max_gradient_norm"], sigma_min=settings["guidance_sigma_min"],
            sigma_max=settings["guidance_sigma_max"])
        samplers, guides = {"guided": sampler}, {"guided": guide}
        if with_reference:
            samplers["control"] = GuidedIC40Sampler(base, reference, strength=cfg["reference_strength"],
                max_gradient_norm=settings["max_gradient_norm"], sigma_min=settings["guidance_sigma_min"],
                sigma_max=settings["guidance_sigma_max"])
            guides["control"] = reference
        binding, replay = build_rollout_runtime(native, base, output / "native-runtime", "cuda")
        runtime = binding.plugin.generator.backend.runtime
        report["runtime_identity"] = binding.report
        solver = load_frozen_solver(assets["assets"]["solver"]["path"], round_cfg["solver_sha256"])
        with _forbid_online_vpfp_generation(report):
            for pair in report["pairs"]:
                row = {key: pair[key] for key in ("slot_id", "split", "index", "stratum", "root_seed", "conditions")}
                draws = {}
                order = effect_arm_order(row["index"], with_reference=with_reference)
                for arm in order:
                    check()
                    report["stage"] = row["slot_id"] + "/" + arm
                    pair[arm].update(status="running", attempt_count=1)
                    report["new_diffusion_draw_attempts"] += 1
                    save()
                    arm_started = time.monotonic()
                    log_h = with_reference and arm == "guided"
                    diagnostic = HeadDiagnostic(kind="log_desirability" if log_h else "reward")
                    observed = guides.get(arm)
                    hook = (None if observed is None else
                            (observed if log_h else observed.head).register_forward_hook(diagnostic))
                    try:
                        theta = torch.tensor([row["conditions"]], dtype=torch.float32, device="cuda")
                        draw = (draw_frozen_ic40(base, [row["root_seed"]], conditions=theta) if arm == "base"
                                else samplers[arm].draw([row["root_seed"]], conditions=theta))
                        pair[arm]["sampling_seconds"] = time.monotonic() - arm_started
                        draws[arm] = draw
                        measured = evaluate_draw(draw, row, runtime=runtime, replay=replay, solver=solver,
                            output=output / "rollouts" / row["slot_id"] / arm,
                            budget_seconds=round_cfg["solver_budget_seconds"], check=check,
                            public_access_mode=round_cfg["public_access_mode"])
                        pair[arm].update(status="completed", valid=measured["valid"],
                            error=measured["error"] if measured["valid"] else None,
                            label_kind=measured["label_kind"], label_unavailable_reason=measured["label_unavailable_reason"],
                            result_path=measured["result_path"], result_sha256=measured["result_sha256"],
                            rollout_seconds=measured["wall_seconds"], guidance=measured["guidance"],
                            oracle_usage=measured.get("public_oracle_usage"),
                            numerical_validity_only=True)
                    except TimeoutError as interruption:
                        pair[arm]["status"] = "interrupted"
                        retained = getattr(interruption, "rollout_result", None)
                        if isinstance(retained, dict):
                            pair[arm]["interrupted_rollout_receipt"] = retained
                        raise
                    except Exception as error:
                        pair[arm].update(status="failed", valid=False, error=None,
                            failure={"type": type(error).__name__, "reason": str(error)})
                    finally:
                        if hook is not None:
                            hook.remove()
                            pair[arm]["head_diagnostic"] = diagnostic.summary()
                        save()
                if "base" in draws and "guided" in draws:
                    pair["draw_comparison"] = paired_draw_diagnostics(draws["base"], draws["guided"])
                if "base" in draws and "control" in draws:
                    pair["reference_draw_comparison"] = paired_draw_diagnostics(draws["base"], draws["control"])
                if "control" in draws and "guided" in draws:
                    pair["new_vs_reference_draw_comparison"] = paired_draw_diagnostics(draws["control"], draws["guided"])
                save()
        for frozen_sampler in samplers.values():
            frozen_sampler.assert_frozen()
        if runtime._resident_cases:
            raise ValueError("evaluation left resident answers unreleased")
        report["summary"] = summarize_pairs(report["pairs"], bootstrap_seed=cfg["bootstrap_seed"],
            bootstrap_replicates=cfg["bootstrap_replicates"])
        write(output / "paired-effect-summary.json", report["summary"])
        if with_reference:
            for key, left, right, filename in (
                ("reference_summary", "base", "control", "reference-effect-summary.json"),
                ("new_vs_reference_summary", "control", "guided", "new-vs-reference-effect-summary.json"),
            ):
                report[key] = summarize_pairs(remap_comparison_pairs(report["pairs"],
                    baseline_arm=left, guided_arm=right), bootstrap_seed=cfg["bootstrap_seed"],
                    bootstrap_replicates=cfg["bootstrap_replicates"])
                report[key]["arm_mapping"] = {"base": left, "guided": right}
                report[key]["shared_with_primary_roots"] = True
                write(output / filename, report[key])
        verify_source()
        report.update(execution_passed=True, stage="complete", frozen_assets_final_verified=True,
            source_final_verified=True, all_residents_released=True)
    except Exception as error:
        report["failures"].append({"stage": report.get("stage"), "type": type(error).__name__,
            "reason": str(error), "traceback": traceback.format_exc()})
    finally:
        save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vpfp_strong_guidance_effect_demand_20260908.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, default=3000.)
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps({"execution_passed": result["execution_passed"], "output": str(args.output),
        "summary": result.get("summary")}, allow_nan=False))
    return 0 if result["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
