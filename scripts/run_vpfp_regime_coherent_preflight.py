"""First bounded compute allocation: mechanics tests and real representation probes.

This does not train models, select compression tolerance, submit child jobs,
modify old assets, or spend the subsequent stage budgets.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda:source.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def write(path,data):
    partial = path.with_suffix(path.suffix+".partial")
    partial.write_text(json.dumps(data,sort_keys=True,indent=2,allow_nan=False))
    partial.replace(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args(argv)
    output = args.output.resolve()
    if (not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_TMPDIR")
            or output.exists() or not output.is_relative_to(Path("/scratch"))):
        p.error("fresh scratch output and Slurm compute allocation required")
    output.mkdir()
    config_path = ROOT/"configs/vpfp_regime_coherent_campaign_20260907.json"
    config = json.loads(config_path.read_text())
    if sum(config["stage_gpu_hour_caps"].values()) != 48 or config["stage_gpu_hour_caps"]["preflight"] != 1:
        raise ValueError("registered allocation budget changed")
    started = time.monotonic()
    result = {"schema":"vpfp_regime_coherent_preflight_run_v1","execution_passed":False,
        "slurm_job_id":os.environ["SLURM_JOB_ID"],"allocated_gpus":1,"reserved_gpu_hours":1,
        "remaining_unreserved_gpu_hours":47,"stage":"tests","stages":{},
        "scientific_admission":False,"training_started":False,"automatic_child_submission":False,
        "campaign_config_sha256":digest(config_path),"source_revision":subprocess.check_output(
            ["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()}
    write(output/"result.json",result)

    def run(name,command,limit):
        result["stage"] = name
        write(output/"result.json",result)
        before = time.monotonic()
        remaining = int(3500-(before-started))
        if remaining <= 0:
            raise TimeoutError("preflight allocation wall budget exhausted; no next stage")
        with (output/f"{name}.log").open("w") as log:
            try:
                completed = subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    timeout=min(limit,remaining),check=False)
                receipt = {"command":command,"exit_code":completed.returncode,
                    "wall_seconds":time.monotonic()-before,"log":str(output/f"{name}.log")}
                result["stages"][name] = receipt
                if completed.returncode:
                    raise RuntimeError(f"{name} failed with exit {completed.returncode}; see retained log")
            except subprocess.TimeoutExpired:
                result["stages"][name] = {"command":command,"status":"time_budget_exhausted",
                    "wall_seconds":time.monotonic()-before,"log":str(output/f"{name}.log")}
                raise
        write(output/"result.json",result)

    try:
        tests = ["test_vpfp_coherent_diffusion.py","test_vpfp_training_resume.py",
            "test_vpfp_training_snapshots.py","test_vpfp_conditional_rho_covariance_gate.py",
            "test_vpfp_numerical_roundtrip.py","test_vpfp_moving_frame.py",
            "test_vpfp_regime_fidelity_coverage.py","test_vpfp_regime_fidelity_coverage_cli.py",
            "test_vpfp_weak_coherent_driver.py",
            "test_vpfp_initial_conditioned_diffusion.py","test_vpfp_temporal_covariance_diffusion.py",
            "test_vpfp_strong_regime_coherent.py"]
        missing = [name for name in tests if not (ROOT/"tests"/name).is_file()]
        if missing:
            raise FileNotFoundError(f"planned preflight tests missing: {missing}")
        run("mechanics_tests",[sys.executable,"-m","pytest","-q","-p","no:cacheprovider",
            "--basetemp",str(output/"pytest-temporary"),*[str(ROOT/"tests"/name) for name in tests]],900)
        run("strong_preflight",[sys.executable,str(ROOT/"scripts/run_vpfp_strong_regime_coherent.py"),
            "--preflight-only","--config",str(ROOT/"configs/vpfp_unified_live_strong_collision.json"),
            "--source",config["strong_source"],"--pilot",config["strong_parent_experiment"],
            "--output",str(output/"strong-preflight"),
            "--device","cuda"],600)
        run("moving_frame_preflight",[sys.executable,str(ROOT/"scripts/diagnose_vpfp_moving_frame.py"),
            "--config",str(ROOT/"configs/vpfp_unified_live_weak_collision.json"),
            "--source",config["weak_source"],"--output",str(output/"moving-frame-preflight"),
            "--device","cuda"],2400)
        result.update(execution_passed=True,stage="preflight_complete_requires_result_review")
    except Exception as error:
        result["failure"] = {"type":type(error).__name__,"reason":str(error),"traceback":traceback.format_exc()}
        raise
    finally:
        result["wall_seconds"] = time.monotonic()-started
        result["observed_gpu_hours_lower_bound"] = result["wall_seconds"]/3600
        result["accounting_note"] = "final actual GPU-hour charge requires Slurm allocation elapsed, not just Python wall"
        write(output/"result.json",result)


if __name__ == "__main__":
    main()
