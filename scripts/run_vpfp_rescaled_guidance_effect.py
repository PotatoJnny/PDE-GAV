"""Frozen eight-arm VPFP solver experiment: objective versus sampling scale.

Fit trajectories calibrate fixed gains without new solver calls. All policies
are then frozen before the 32 new paired roots. Numerical/physical diagnostics
are separate from real solver difficulty; no development-loss eligibility gate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from run_vpfp_strong_solver_guidance_round import file_sha, json_sha, load_round_base, write
from run_vpfp_strong_guidance_effect import (
    load_effect_config, freeze_evidence, paired_draw_diagnostics, validate_checkpoint_training)
from vpfp_strong_guidance_rollouts import freeze_round_assets

SCHEMA = "vpfp_rescaled_guidance_effect_v1"
ARMS = ("base", "mean_natural", "mean_fixed", "mean_relative", "exp_natural",
        "exp_fixed", "exp_relative", "random_relative")
PRIMARY = (("base", "exp_relative"), ("random_relative", "exp_relative"))


def validate_rescaling_config(cfg):
    expected = {"target_ratio": .1, "gain_max": 1.e6, "max_flow_ratio": .25,
        "minimum_gradient_rms": 1.e-12, "minimum_base_drift_rms": 1.e-8,
        "random_seed": 9920260908,
        "fixed_gain_rule": "median_target_over_natural_flow_ratio_first_fit_root_per_stratum",
        "calibration_roots": 8, "arms": list(ARMS)}
    if (cfg.get("schema") != "vpfp_rescaled_guidance_experiment_v1"
            or cfg.get("beta_std") != .5 or cfg.get("steps") != 2000
            or cfg.get("rescaling") != expected
            or cfg.get("physics") != {"light_probe_every_attempt": True,
                "local_audit_first_root_per_stratum": True, "hard_admission_gate": False}):
        raise ValueError("fixed rescaling and diagnostic protocol required; no outcome-based tuning")
    return cfg


def arm_order(index):
    if type(index) is not int or index < 0:
        raise ValueError("nonnegative integer root index required")
    # Stratum is index % 8. Rotate across repeats too, to avoid fixing one
    # execution order for all four roots of a condition stratum.
    offset = (index % len(ARMS) + index // len(ARMS)) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def first_per_stratum(rows):
    selected = []
    for stratum in range(8):
        group = [row for row in rows if row["stratum"] == stratum]
        if not group:
            raise ValueError("one registered calibration/audit root per stratum required")
        selected.append(min(group, key=lambda row: row["index"]))
    if len({row["root_seed"] for row in selected}) != 8:
        raise ValueError("calibration/audit roots must be distinct")
    return selected


def fixed_gain_from_records(records, settings):
    ratios = [row["applied_flow_ratio"] for row in records
        if row.get("skip_reason") is None and row.get("raw_gradient_rms", 0.) >= settings["minimum_gradient_rms"]
        and row.get("base_drift_rms", 0.) >= settings["minimum_base_drift_rms"]
        and type(row.get("applied_flow_ratio")) in (int, float)
        and math.isfinite(row["applied_flow_ratio"]) and row["applied_flow_ratio"] > 0.]
    # A numerically empty guide still proceeds to the solver experiment. Do not
    # invent a direction, pass a regression gate, or tune on confirmation errors.
    wanted = statistics.median(settings["target_ratio"] / value for value in ratios) if ratios else 1.
    return {"fixed_gain": min(wanted, settings["gain_max"]), "uncapped_median_gain": wanted,
        "active_calibration_calls": len(ratios), "all_calibration_calls": len(records),
        "natural_ratio_min": min(ratios, default=None), "natural_ratio_max": max(ratios, default=None),
        "natural_ratio_median": statistics.median(ratios) if ratios else None,
        "gain_capped": wanted > settings["gain_max"], "empty_direction_fallback": not ratios,
        "solver_calls_for_calibration": 0, "confirmation_outcomes_used": False}


def comparison_pairs(rows, left, right):
    if left not in ARMS or right not in ARMS or left == right:
        raise ValueError("two distinct registered arms required")
    return [{"root_seed": row["root_seed"], "stratum": row["stratum"],
        "base": deepcopy(row[left]), "guided": deepcopy(row[right])} for row in rows]


def summarize_comparisons(pairs, cfg):
    from vpfp_guidance_effect_summary import summarize_pairs
    comparisons = list(PRIMARY) + [("base", arm) for arm in ARMS[1:] if arm != "exp_relative"]
    comparisons += [(name + "_natural", name + "_" + mode) for name in ("mean", "exp") for mode in ("fixed", "relative")]
    comparisons += [("mean_relative", "exp_relative"), ("mean_fixed", "mean_relative"), ("exp_fixed", "exp_relative")]
    results = {}
    for left, right in comparisons:
        summary = summarize_pairs(comparison_pairs(pairs, left, right),
            bootstrap_seed=cfg["bootstrap_seed"], bootstrap_replicates=cfg["bootstrap_replicates"])
        summary.update(arm_mapping={"base": left, "guided": right},
            preregistered_primary=(left, right) in PRIMARY,
            intervals_are_pointwise_not_simultaneous=True, automatic_candidate_admission=False)
        results[right + "_vs_" + left] = summary
    return results


def _freeze_item(item, destination):
    with Path(item["path"]).open("rb") as source, destination.open("xb") as target:
        shutil.copyfileobj(source, target, 1024 * 1024)
    destination.chmod(0o444)
    return {"path": str(destination), "source_path": item["path"], "sha256": item["sha256"]}


def run(args):
    if not os.environ.get("SLURM_JOB_ID", "").isdigit() or not os.environ.get("SLURM_TMPDIR"):
        raise ValueError("Slurm compute allocation required")
    if not math.isfinite(args.wall_seconds) or args.wall_seconds <= 120:
        raise ValueError("bounded usable evaluation allocation required")
    output = args.output.absolute()
    scratch = Path("${PDE_GAV_RUN_ROOT}")
    if output == scratch or not output.is_relative_to(scratch) or output.exists() or output.is_symlink():
        raise ValueError("new specific scratch output required; no silent retry or overwrite")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    def verify_source():
        if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != revision
                or subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()):
            raise ValueError("source must remain frozen and clean")
    verify_source()
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {"schema": SCHEMA, "execution_passed": False, "source_revision": revision,
        "job_id": os.environ["SLURM_JOB_ID"], "training_updates": 0, "online_assets_replaced": False,
        "scientific_admission": False, "exact_tilted_law_claimed": False,
        "new_diffusion_draw_attempts": 0, "calibration_draw_attempts": 0,
        "pairs": [], "failures": [], "expected_draw_attempts": 256,
        "independent_root_count": 32, "statistical_unit": "32 whole paired roots, not 256 independent samples"}
    def save():
        report["wall_seconds"] = time.monotonic() - started
        write(output / "result.json", report)
    def check():
        if time.monotonic() - started >= args.wall_seconds - 45:
            raise TimeoutError("allocation exhausted; retain all original planned attempts")
    save()
    try:
        import torch
        from pde_agent_verifier_solver.plugins.vpfp.frozen_solver_reward import load_frozen_solver
        from pde_agent_verifier_solver.plugins.vpfp.solver_error_guidance import (
            draw_frozen_ic40, GuidedIC40Sampler, load_solver_error_regressor)
        from pde_agent_verifier_solver.plugins.vpfp.rescaled_solver_error_guidance import (
            RescalingPolicy, RescaledIC40Sampler)
        from pde_agent_verifier_solver.plugins.vpfp.standardized_solver_error_guidance import OBJECTIVE, MEAN_OBJECTIVE
        from vpfp_strong_guidance_rollouts import build_rollout_runtime, evaluate_draw
        from run_plasma_canonical_smoke import _forbid_online_vpfp_generation
        from vpfp_rescaling_physics import audit_draw
        from pde_agent_verifier_solver.plugins.vpfp.perturbation_backend import _legacy_api
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one allocated CUDA device required")
        torch.set_num_threads(min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))))
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        cfg = validate_rescaling_config(json.loads(args.config.read_text()))
        trained = json.loads(args.training_result.read_text())
        if (trained.get("execution_passed") is not True or trained.get("source_revision") != revision
                or trained.get("config") != cfg or set(trained.get("heads", {})) != {"mean", "exp"}):
            raise ValueError("both completed heads from this frozen source/config required")
        root_item = trained["root_preregistration"]
        root_cfg, round_cfg, roots, _ = load_effect_config(root_item["path"])
        if roots != trained["registered_confirmation_roots"]:
            raise ValueError("confirmation roots changed after training")
        report["pairs"] = [{**deepcopy(row), **{arm: {"status": "pending", "valid": False,
            "error": None, "attempt_count": 0} for arm in ARMS}} for row in roots]
        report.update(config=cfg, public_access_mode=round_cfg["public_access_mode"],
            regression_gate=False, primary_comparisons=[list(pair) for pair in PRIMARY])
        save()
        assets, native = freeze_round_assets(round_cfg, output)
        evidence = freeze_evidence(root_cfg, output)
        for role, item in (("training_result", {"path": str(args.training_result), "sha256": file_sha(args.training_result)}),
                           ("root_preregistration", root_item)):
            evidence[role] = _freeze_item(item, output / "frozen-assets" / (role + ".json"))
        base, portable = load_round_base(round_cfg, assets, device="cuda")
        base.model.eval().requires_grad_(False)
        base.normalizer.eval().requires_grad_(False)
        old_plan = json.loads(Path(evidence["training_plan"]["path"]).read_text())
        guides = {}
        for name, objective in (("mean", MEAN_OBJECTIVE), ("exp", OBJECTIVE)):
            for role, suffix in (("checkpoint", ".pt"), ("training_report", ".json")):
                item = trained["heads"][name][role]
                evidence[name + "_" + role] = _freeze_item(item,
                    output / "frozen-assets" / (name + "_" + role + suffix))
            item = evidence[name + "_checkpoint"]
            raw = torch.load(item["path"], map_location="cpu", weights_only=True)
            training = json.loads(Path(evidence[name + "_training_report"]["path"]).read_text())
            if (training.get("final_step") != 2000 or training.get("target_step") != 2000
                    or training.get("seed") != cfg["seed"]
                    or training.get("label_manifest_sha256") != evidence["training_labels"]["sha256"]):
                raise ValueError("same-seed same-step real-label training evidence required")
            if validate_checkpoint_training(raw, training, evidence["training_labels"]["sha256"], old_plan) != 2000:
                raise ValueError("both fixed step2000 checkpoints required")
            guide = load_solver_error_regressor(item["path"], item["sha256"], device="cuda")
            if guide.objective != objective or guide.beta_std != .5:
                raise ValueError("loaded head does not bind registered standardized objective")
            guides[name] = guide
        if guides["mean"].config != guides["exp"].config:
            raise ValueError("two objective heads must share the frozen architecture")
        settings = round_cfg["sampling"]
        sampler_args = {"max_gradient_norm": settings["max_gradient_norm"],
            "sigma_min": settings["guidance_sigma_min"], "sigma_max": settings["guidance_sigma_max"]}
        policy_args = {key: cfg["rescaling"][key] for key in (
            "target_ratio", "gain_max", "max_flow_ratio", "minimum_gradient_rms",
            "minimum_base_drift_rms", "random_seed")}
        samplers = {name + "_natural": RescaledIC40Sampler(base, guide,
            RescalingPolicy(mode="natural", **policy_args), **sampler_args) for name, guide in guides.items()}
        cal_roots = first_per_stratum(old_plan["roots"]["fit"])
        local_roots = {row["root_seed"] for row in first_per_stratum(roots)}
        plan = {"schema": SCHEMA, "source_revision": revision, "config": cfg, "round_config": round_cfg,
            "roots": roots, "assets": assets, "evidence": evidence, "calibration_roots": cal_roots,
            "local_physics_root_seeds": sorted(local_roots), "execution_order": "cyclic_offset_(index_mod8+index_div8)_mod8",
            "primary_comparisons": [list(pair) for pair in PRIMARY],
            "other_comparisons": "exploratory_uncorrected_intervals_not_an_automatic_winner_gate",
            "random_control": "matches_exp_relative_local_clipped_gradient_RMS_and_scalar_gain_at_its_own_state; paths_differ",
            "physical_scale_percentage_claimed": False, "physical_admission_gate": False,
            "new_solver_calls_during_gain_calibration": 0, "online_assets_replaced": False}
        write(output / "preregistered-plan.json", plan)
        (output / "preregistered-plan.json").chmod(0o444)
        preregistered_sha = file_sha(output / "preregistered-plan.json")
        calibration = {}
        report.update(stage="fit_only_gain_calibration", portable_codec_binding=portable); save()
        # Calibration now precedes build_rollout_runtime, which historically
        # initialized the pinned legacy import path. Bootstrap that exact
        # checked API before installing its no-online-generation guard.
        _legacy_api()
        with _forbid_online_vpfp_generation(report):
            for name in guides:
                records = []
                sampler = samplers[name + "_natural"]
                for index, row in enumerate(cal_roots):
                    check()
                    theta = torch.tensor([row["conditions"]], dtype=torch.float32, device="cuda")
                    report["calibration_draw_attempts"] += 1
                    draw = sampler.draw([row["root_seed"]], conditions=theta)
                    records.extend(deepcopy(sampler.last_report["records"]))
                    if index == 0:
                        old_sampler = GuidedIC40Sampler(base, guides[name], strength=1., **sampler_args)
                        report["calibration_draw_attempts"] += 1
                        old_draw = old_sampler.draw([row["root_seed"]], conditions=theta)
                        comparison = paired_draw_diagnostics(old_draw, draw)
                        if not all(comparison[key]["bitwise_equal"] for key in ("full", "future_white")):
                            raise ValueError("natural rescaled sampler changed native old Guidance arithmetic")
                        old_sampler.assert_frozen()
                calibration[name] = {**fixed_gain_from_records(records, cfg["rescaling"]),
                    "root_seeds": [row["root_seed"] for row in cal_roots],
                    "legacy_natural_sampling_bitwise_equal": True, "records": records}
        for name, guide in guides.items():
            for mode in ("fixed", "relative"):
                samplers[name + "_" + mode] = RescaledIC40Sampler(base, guide,
                    RescalingPolicy(mode=mode, fixed_gain=calibration[name]["fixed_gain"] if mode == "fixed" else 1.,
                        **policy_args), **sampler_args)
        samplers["random_relative"] = RescaledIC40Sampler(base, guides["exp"],
            RescalingPolicy(mode="relative", random_direction=True, **policy_args), **sampler_args)
        gain_path = output / "fit-only-gain-calibration.json"
        write(gain_path, calibration); gain_path.chmod(0o444)
        plan.update(calibration={"path": str(gain_path), "sha256": file_sha(gain_path)},
            policies={name: sampler.policy.to_dict() for name, sampler in samplers.items()})
        write(output / "frozen-plan.json", plan); (output / "frozen-plan.json").chmod(0o444)
        report.update(frozen_plan_sha256=file_sha(output / "frozen-plan.json"),
            preregistered_plan_sha256=preregistered_sha,
            fixed_gain_calibration={name: {key: value for key, value in item.items() if key != "records"}
                                   for name, item in calibration.items()},
            policies=plan["policies"], stage="solver_confirmation"); save()
        binding, replay = build_rollout_runtime(native, base, output / "native-runtime", "cuda")
        runtime = binding.plugin.generator.backend.runtime
        report["runtime_identity"] = binding.report
        solver = load_frozen_solver(assets["assets"]["solver"]["path"], round_cfg["solver_sha256"])
        with _forbid_online_vpfp_generation(report):
            for pair in report["pairs"]:
                row = {key: pair[key] for key in ("slot_id", "split", "index", "stratum", "root_seed", "conditions")}
                draws = {}
                for arm in arm_order(row["index"]):
                    check()
                    report["stage"] = row["slot_id"] + "/" + arm
                    pair[arm].update(status="running", attempt_count=1)
                    report["new_diffusion_draw_attempts"] += 1
                    save()
                    arm_started = time.monotonic()
                    try:
                        theta = torch.tensor([row["conditions"]], dtype=torch.float32, device="cuda")
                        draw = (draw_frozen_ic40(base, [row["root_seed"]], conditions=theta) if arm == "base"
                            else samplers[arm].draw([row["root_seed"]], conditions=theta))
                        pair[arm]["sampling_seconds"] = time.monotonic() - arm_started
                        draws[arm] = draw
                        audit = audit_draw(draw, runtime.base_model,
                            full_local_audit=row["root_seed"] in local_roots)
                        audit_path = output / "physics" / row["slot_id"] / (arm + ".json")
                        write(audit_path, audit)
                        pair[arm]["physics"] = {"path": str(audit_path), "sha256": file_sha(audit_path),
                            "execution_passed": audit["execution_passed"],
                            "numerically_undetermined": audit["numerically_undetermined"],
                            "summary": audit["summary"], "scientific_admission": False}
                        measured = evaluate_draw(draw, row, runtime=runtime, replay=replay, solver=solver,
                            output=output / "rollouts" / row["slot_id"] / arm,
                            budget_seconds=round_cfg["solver_budget_seconds"], check=check,
                            public_access_mode=round_cfg["public_access_mode"])
                        pair[arm].update(status="completed", valid=measured["valid"],
                            error=measured["error"] if measured["valid"] else None,
                            label_kind=measured["label_kind"], label_unavailable_reason=measured["label_unavailable_reason"],
                            result_path=measured["result_path"], result_sha256=measured["result_sha256"],
                            rollout_seconds=measured["wall_seconds"],
                            guidance={key: value for key, value in (measured["guidance"] or {}).items() if key != "records"},
                            oracle_usage=measured.get("public_oracle_usage"), numerical_validity_only=True)
                    except TimeoutError as interruption:
                        pair[arm]["status"] = "interrupted"
                        retained = getattr(interruption, "rollout_result", None)
                        if isinstance(retained, dict):
                            pair[arm]["interrupted_rollout_receipt"] = retained
                        raise
                    except Exception as error:
                        pair[arm].update(status="failed", valid=False, error=None,
                            failure={"type": type(error).__name__, "reason": str(error), "traceback": traceback.format_exc()})
                    finally:
                        pair[arm]["total_seconds"] = time.monotonic() - arm_started
                        save()
                pair["draw_comparisons"] = {arm: paired_draw_diagnostics(draws["base"], draws[arm])
                    for arm in ARMS[1:] if "base" in draws and arm in draws}
                save()
        for sampler in samplers.values():
            sampler.assert_frozen()
        if runtime._resident_cases:
            raise ValueError("evaluation left resident answers unreleased")
        if (report["new_diffusion_draw_attempts"] != 256 or len(report["pairs"]) != 32
                or any(pair[arm]["attempt_count"] != 1 for pair in report["pairs"] for arm in ARMS)):
            raise ValueError("all original 32x8 attempts required; failures do not permit replacement roots")
        report["comparisons"] = summarize_comparisons(report["pairs"], cfg)
        write(output / "paired-effect-summary.json", report["comparisons"])
        for pair in report["pairs"]:
            for arm in ARMS:
                item = pair[arm]
        verify_source()
        report.update(execution_passed=True, stage="complete", source_final_verified=True,
            frozen_assets_final_verified=True, all_residents_released=True,
            all_rollout_and_physics_receipts_verified=True)
    except Exception as error:
        report["failures"].append({"stage": report.get("stage"), "type": type(error).__name__,
            "reason": str(error), "traceback": traceback.format_exc()})
    finally:
        if report["pairs"] and "comparisons" not in report and "config" in report:
            try:
                report["comparisons"] = summarize_comparisons(report["pairs"], report["config"])
                write(output / "paired-effect-summary.json", report["comparisons"])
            except Exception as error:
                report["failures"].append({"stage": "partial_summary", "type": type(error).__name__, "reason": str(error)})
        save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vpfp_rescaled_guidance_20260908.json")
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, default=9000.)
    result = run(parser.parse_args(argv))
    print(json.dumps({key: result.get(key) for key in ("execution_passed", "stage", "new_diffusion_draw_attempts", "failures")}))
    return 0 if result["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
