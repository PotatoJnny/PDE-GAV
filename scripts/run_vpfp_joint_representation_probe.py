"""Bounded pre-training VPFP representation experiment; no model admission."""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import torch

from diagnose_vpfp_weak_structure_prior import _file_sha, _write_json

ROOT = Path(__file__).resolve().parents[1]
STRONG = Path("${PDE_GAV_RUN_ROOT}/vpfp-strong-main-5244223")
WEAK = Path("${PDE_GAV_RUN_ROOT}/vpfp-scaleup-weak-5241246")
WEAK_MODELS = Path("${PDE_GAV_RUN_ROOT}/vpfp-weak-structure-prior-5248333")
MODULES = ("diagnose_vpfp_known_initial", "diagnose_vpfp_conditional_geometry",
           "diagnose_vpfp_drift_codec", "diagnose_vpfp_reverse_joint")


def _check_frozen(inputs):
    pass


def frozen_inputs():
    strong_registry = STRONG / "frozen-checkpoints.json"
    weak_registry = WEAK_MODELS / "frozen-checkpoints.json"
    strong_entry = json.loads(strong_registry.read_text())["models"]["seed-0/cov_base/best"]
    weak_entries = {f"seed-{seed}": json.loads(weak_registry.read_text())["models"][f"seed-{seed}/cov_shrink_half/best"]
                    for seed in (0, 1)}
    entries = [strong_entry, *weak_entries.values()]
    inputs = {entry["checkpoint"]: entry["sha256"] for entry in entries}
    paths = [strong_registry, weak_registry, STRONG / "corpus/manifest.json", WEAK / "corpus/manifest.json",
             ROOT / "configs/vpfp_unified_live_strong_collision.json", ROOT / "configs/vpfp_unified_live_weak_collision.json",
             Path(__file__), ROOT / "scripts/diagnose_vpfp_weak_structure_prior.py"]
    paths.extend(ROOT / "scripts" / (name + ".py") for name in MODULES)
    paths.extend((ROOT / "src/pde_agent_verifier_solver/plugins/vpfp").glob("*.py"))
    paths.extend((ROOT / "legacy/vpfp/src/agent_verifier/vpfp").glob("*.py"))
    for path in paths:
        inputs[str(path)] = _file_sha(path)
    _check_frozen(inputs)
    return inputs, strong_entry, weak_entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID", "").isdigit() or not os.environ.get("SLURM_TMPDIR"):
        parser.error("real representation probes require a Slurm compute allocation and local scratch")
    raw_output = args.output
    output = raw_output.resolve()
    run_parent = Path("${PDE_GAV_RUN_ROOT}").resolve()
    if (raw_output.exists() or raw_output.is_symlink() or output.parent != run_parent
            or output.name != f"vpfp-joint-representation-{os.environ['SLURM_JOB_ID']}"):
        parser.error("output must be the fresh exact job-specific scratch directory")
    if not torch.cuda.is_available():
        parser.error("no usable CUDA device")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all", "--",
                                "src", "scripts", "configs", "legacy/vpfp"], cwd=ROOT, text=True).strip():
        parser.error("freeze all relevant source before starting probes")
    torch.set_num_threads(min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    inputs, strong_entry, weak_entries = frozen_inputs()
    output.mkdir(parents=True, exist_ok=False)
    plan = {"schema": "vpfp_joint_representation_stage_zero_v1", "job_id": os.environ["SLURM_JOB_ID"],
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "device": torch.cuda.get_device_name(0), "frozen_inputs": inputs,
        "strong_checkpoint": strong_entry, "weak_checkpoints": weak_entries,
        "stages": list(MODULES), "strong_cases": "8 train and 8 validation; one per condition bin",
        "weak_codec_cases": "8 train and 8 validation; 4 times x 4 spatial points per case",
        "weak_sampler_conditions": "same 8 validation conditions for both existing model seeds",
        "weak_sample_steps": [64, 128, 256], "same_initial_white_noise": True, "churn": 0,
        "new_pde_solves": 0, "training_seconds": 0, "scheduled_walltime_cap_seconds": 7200,
        "evidence_role": "development_representation_and_sampling_diagnostics_not_distribution_qualification",
        "strong_prior": "reuse_exact_frozen_Gaussian_packet_parameter_law_no_learned_initial_prior",
        "future_condition": "t0_values_not_t0_slopes_or_whole_time_whitened_slice",
        "future_geometry": "training_metric_Schur_complement_not_Gaussian_target_assumption",
        "weak_translation": "retain_all_original_Hermite_channels; report_projection_loss_before_normalization",
        "weak_reverse_estimates": "intermediate_denoised_conditional_estimates_not_generated_answers",
        "reference_256_steps_is_truth": False, "native_assets_modified": False,
        "automatic_followup_training": False, "scientific_admission": False}
    _write_json(output / "frozen-plan.json", plan)
    inputs[str(output / "frozen-plan.json")] = _file_sha(output / "frozen-plan.json")
    strong_config = ROOT / "configs/vpfp_unified_live_strong_collision.json"
    weak_config = ROOT / "configs/vpfp_unified_live_weak_collision.json"
    stages = (
        ("strong_known_initial", MODULES[0], "run", (strong_config, STRONG / "corpus", output / "strong-known-initial"), {}),
        ("strong_conditional_geometry", MODULES[1], "run", (strong_entry["checkpoint"], STRONG / "corpus", output / "strong-conditional-geometry"), {}),
        ("weak_drift_codec", MODULES[2], "run", (WEAK / "corpus", weak_config, output / "weak-drift-codec"), {}),
        ("weak_reverse_joint", MODULES[3], "run_reverse_joint_diagnostic",
         (weak_entries, WEAK / "corpus", output / "weak-reverse-joint"), {"seed": 1420260906}),
    )
    started = time.monotonic()
    report = {"execution_passed": False, "training_run": False, "new_pde_solves": 0,
              "scientific_admission": False, "stages": {}}
    for name, module_name, function, positional, kwargs in stages:
        stage_start = time.monotonic()
        print(f"Starting {name}", flush=True)
        _write_json(output / "stage.json", {"stage": name, "status": "running"})
        try:
            _check_frozen(inputs)
            result = getattr(importlib.import_module(module_name), function)(*positional, device="cuda", **kwargs)
            if not isinstance(result, dict) or result.get("execution_passed") is not True:
                raise RuntimeError(f"{name} did not report execution_passed=True; inspect retained stage artifacts")
            _check_frozen(inputs)
            report["stages"][name] = {"execution_passed": True, "elapsed_seconds": time.monotonic() - stage_start}
        except Exception as error:
            report["stages"][name] = {"execution_passed": False, "elapsed_seconds": time.monotonic() - stage_start,
                "failure_type": type(error).__name__, "failure_reason": str(error), "traceback": traceback.format_exc()}
            print(f"FAILED {name}: {error}; preserving failure and checking remaining independent stages", flush=True)
        _write_json(output / "result.partial.json", report)
        gc.collect()
        torch.cuda.empty_cache()
    report["execution_passed"] = all(item["execution_passed"] for item in report["stages"].values())
    report["elapsed_seconds"] = time.monotonic() - started
    _write_json(output / "result.json", report)
    _write_json(output / "stage.json", {"stage": "complete", "status": "completed" if report["execution_passed"] else "failed"})
    return 0 if report["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
