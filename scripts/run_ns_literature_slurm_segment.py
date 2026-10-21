#!/usr/bin/env python3
"""Run one literature-baseline segment and submit only a safely paused successor.

The baseline CLI owns scientific checkpoints. The existing Slurm continuation
helper owns the bounded submission chain. Neither this script nor its wrapper
changes method budgets, Case counts, caching, or the frozen experiment config.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import ns_slurm_continuation as continuation


SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "src"))
PROFILE = "literature_3h"


def _digest(path):
    return sha256(path.read_bytes()).hexdigest()


def _resume_record(config, output):
    checkpoint = output / "baseline_checkpoints/checkpoint.json"
    return {
        "config": str(config), "config_sha256": _digest(config),
        "output": str(output), "checkpoint": str(checkpoint),
        "checkpoint_sha256": _digest(checkpoint),
    }


def _commit_record(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix=".resume-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run_segment(config, output, *, max_segments=12, baseline_main=None):
    config, output = Path(config).resolve(), Path(output).resolve()
    if not 1 <= max_segments <= 12:
        raise ValueError("this run supports between 1 and 12 allocation segments")
    expected_cluster = os.environ.get("NS_LITERATURE_EXPECTED_CLUSTER")
    if expected_cluster and os.environ.get("SLURM_CLUSTER_NAME") != expected_cluster:
        raise ValueError("this wrapper requires an allocation on the configured Slurm cluster")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu or gpu == "-1" or "," in gpu or any(c.isspace() for c in gpu):
        raise ValueError("exactly one allocated GPU is required")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 16:
        raise ValueError("the paired literature profile requires 16 CPUs per allocation")
    started = float(os.environ["NS_LITERATURE_SEGMENT_STARTED_UNIX"])
    control = output.parent / ("." + output.name + ".slurm")
    previous_root = os.environ.get("NS_CONTINUATION_RUN_ROOT")
    if previous_root is not None and Path(previous_root).resolve() != control:
        raise ValueError("continuation environment belongs to another run")
    if int(os.environ.get("NS_CONTINUATION_MAX_SEGMENTS", str(max_segments))) != max_segments:
        raise ValueError("continuation segment limit changed")
    index = int(os.environ.get("NS_CONTINUATION_SEGMENT_INDEX", "1"))
    receipt = control / "baseline_resume.json"
    with continuation.runner_lease(control, os.environ["SLURM_JOB_ID"], index,
                                   max_segments=max_segments):
        if index == 1:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError("start this wrapper only with a new baseline output")
        elif json.loads(receipt.read_text()) != _resume_record(config, output):
            raise ValueError("scientific checkpoint or config differs from the paused segment")
        if baseline_main is None:
            from pde_agent_verifier_solver.plugins.ns.baselines.cli import main as baseline_main
        # Count imports/setup against the allocation, keeping the CLI's original
        # Agent clocks and 300-second flush margin unchanged.
        remaining = int(10800 - max(0, time.time() - started))
        if remaining <= 300:
            raise RuntimeError("allocation has insufficient time to start the baseline CLI")
        command = ["--config", str(config), "--output", str(output),
                   "--gpu-token", gpu, "--segment-seconds", str(remaining),
                   "--flush-seconds", "300"]
        if index > 1:
            command.append("--resume")
        status = baseline_main(command)
        if status == 75:
            _commit_record(receipt, _resume_record(config, output))
            continuation.publish_outcome(control, status="paused", checkpoint_path=receipt)
        elif status == 0:
            result = json.loads((output / "baseline_result.json").read_text())
            if result.get("status") != "completed":
                raise RuntimeError("baseline returned 0 without a completed result")
            continuation.publish_outcome(control, status="completed")
        else:
            raise RuntimeError(f"baseline failed with exit code {status}; no successor submitted")
    if status == 75:
        state = continuation.submit_next(
            control, slurm_script=SOURCE / "scripts/slurm_ns_literature_segment.sh",
            script_args=[str(SOURCE), sys.executable, str(config), str(output), str(max_segments)],
            gpu_count=1, max_segments=max_segments, resource_profile=PROFILE,
        )
        print(json.dumps({"status": state["status"], "output": str(output),
                          "successor_job": state["submissions"][-1]["job_id"]}))
    # afterok successors require a successful parent, including safe pauses.
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-segments", type=int, default=12)
    args = parser.parse_args(argv)
    return run_segment(args.config, args.output, max_segments=args.max_segments)


if __name__ == "__main__":
    raise SystemExit(main())
