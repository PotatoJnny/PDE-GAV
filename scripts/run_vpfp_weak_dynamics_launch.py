"""One bounded weak development allocation; no training or automatic resubmission."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_vpfp_independent_confirmation import file_sha, write_json, run_child
from run_vpfp_strong_regime_followup import accounting_row, duration_seconds, query, SACCT, SCONTROL

SCRATCH = Path("${PDE_GAV_RUN_ROOT}")
PRIOR = {"5268596": ("FAILED", "1:0", 35), "5268892": ("COMPLETED", "0:0", 11673)}
WEAK_CAP_SECONDS = 72000
LAUNCH_CAP_SECONDS = 21600


def validate_cpu_gate(path, digest, revision):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_relative_to(SCRATCH):
        raise ValueError('explicit original CPU gate receipt path and SHA required')
    rows = [line.split("\t") for line in path.read_text().splitlines()]
    if any(len(row) != 2 for row in rows) or len(dict(rows)) != len(rows):
        raise ValueError("malformed CPU mechanics receipt")
    receipt = dict(rows)
    if any(receipt.get(key) != value for key, value in {
        "source_revision": revision, "mechanics_tests": "0", "launch": "0",
        "requested_gpus": "0", "scientific_admission": "false"}.items()):
        raise ValueError("successful same-source CPU gate required; it is not physics admission")
    return {"path": str(path), "sha256": digest, "receipt": receipt}


def validate_accounting(rows, current, control):
    if set(rows) != set(PRIOR):
        raise ValueError("both original weak allocations, including failure, required")
    for job, expected in PRIOR.items():
        row = rows[job]
        if (row["job_id"] != job or row["allocated_gpus"] != 1
                or (row["state"], row["exit_code"], row["elapsed_seconds"]) != expected):
            raise ValueError("original weak allocation accounting differs from frozen terminal evidence")
    spent = sum(row["allocated_gpus"] * row["elapsed_seconds"] for row in rows.values())
    limit = duration_seconds(control.get("TimeLimit", ""))
    runtime = duration_seconds(control.get("RunTime", ""))
    if (current["job_id"] in PRIOR or control.get("JobId") != current["job_id"]
            or current["state"] != "RUNNING" or control.get("JobState") != "RUNNING"
            or current["allocated_gpus"] != 1
            or limit != current["time_limit_minutes"] * 60
            or limit <= 60 or limit > min(LAUNCH_CAP_SECONDS, WEAK_CAP_SECONDS - spent)):
        raise ValueError("one GPU, at most six hours inside remaining original weak20h required")
    return {"prior_allocations": rows, "current": current, "control": control,
        "weak_allocated_gpu_seconds_before_this_job": spent,
        "weak_stage_cap_gpu_seconds": WEAK_CAP_SECONDS,
        "launch_cap_seconds": LAUNCH_CAP_SECONDS,
        "remaining_current_seconds": limit - max(runtime, current["elapsed_seconds"]),
        "campaign_coordination": "root maintains total48 GPUh and at most2 concurrent campaign GPUs",
        "retries": "not automatic; future receipts must also debit this allocation"}


def scheduler_budget(job):
    def read(identity):
        return accounting_row(query([SACCT, "-X", "-n", "-P", "-j", identity,
            "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES%200,TimelimitRaw"]), identity)
    rows = {identity: read(identity) for identity in PRIOR}
    current = read(job)
    control = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)",
        query([SCONTROL, "--oneliner", "show", "job", job])))
    return validate_accounting(rows, current, control)


def run(args):
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit() or not os.environ.get("SLURM_TMPDIR"):
        raise ValueError("separate scheduler compute allocation required")
    output = Path(args.output).absolute()
    if (output.exists() or output.is_symlink() or output == SCRATCH
            or not output.is_relative_to(SCRATCH)):
        raise ValueError("fresh scratch output required")
    output.mkdir(parents=True)
    report = {"schema": "vpfp_weak_dynamics_launch_v1", "execution_passed": False,
        "scientific_admission": False, "new_training": False, "stages": [],
        "evidence_role": "opened_development_diagnostics_not_fresh512_confirmation"}
    started = time.monotonic()
    def save():
        report["wall_seconds"] = time.monotonic() - started
        write_json(output / "result.json", report)
    save()
    try:
        revision = query(["git", "-C", str(ROOT), "rev-parse", "--verify", "HEAD^{commit}"]).strip()
        dirty = query(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all",
            "--", "src", "scripts", "configs", "tests", "legacy/vpfp"])
        if dirty or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("freeze complete relevant source before launch")
        report["source_revision"] = revision
        report["cpu_gate"] = validate_cpu_gate(args.cpu_gate, args.cpu_gate_sha256, revision)
        budget_started = time.monotonic()
        report["scheduler_budget"] = scheduler_budget(job)
        deadline = budget_started + report["scheduler_budget"]["remaining_current_seconds"] - 30
        save()
        def child(name, command, cap=None):
            end = min(deadline, time.monotonic() + cap) if cap is not None else deadline
            row = {"name": name, **run_child(command, output / (name + ".log"), end)}
            report["stages"].append(row)
            save()
            if row["exit_code"] != 0 or row["timed_out"]:
                raise RuntimeError(name + " incomplete; no later stages launched")
        # Explicitly require CUDA; a skipped CPU-only device test is not a GPU gate.
        child("cuda_available", [sys.executable, "-c",
            "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count()==1; print(torch.cuda.get_device_name(0))"], 120)
        child("mechanics_cuda", [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "--basetemp", str(Path(os.environ["SLURM_TMPDIR"]) / "vpfp-dynamics-gpu-tests"),
            "tests/test_vpfp_moving_frame.py", "tests/test_vpfp_moving_frame_backend.py",
            "tests/test_vpfp_coherent_diffusion.py", "tests/test_vpfp_coherent_velocity_dynamics.py",
            "tests/test_vpfp_weak_dynamics_diagnostic.py"], 900)
        child("velocity_dynamics_preflight", [sys.executable,
            str(ROOT / "scripts/run_vpfp_velocity_dynamics_preflight.py"),
            "--output", str(output / "velocity-preflight"), "--device", "cpu"], 1800)
        remaining = int(deadline - time.monotonic() - 10)
        if remaining <= 30:
            raise TimeoutError("budget exhausted before fixed cross diagnosis; do not downsample")
        child("cross_diagnostic", [sys.executable,
            str(ROOT / "scripts/run_vpfp_weak_dynamics_diagnostic.py"),
            "--output", str(output / "cross-diagnostic"), "--device", "cuda",
            "--wall-seconds", str(remaining)])
        if query(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).strip() != revision:
            raise ValueError("source changed during diagnosis")
        if query(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all",
                "--", "src", "scripts", "configs", "tests", "legacy/vpfp"]):
            raise ValueError("source modified during diagnosis")
        report["execution_passed"] = True
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc()}
    finally:
        save()
    return 0 if report["execution_passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu-gate", required=True)
    parser.add_argument("--cpu-gate-sha256", required=True)
    raise SystemExit(run(parser.parse_args()))
