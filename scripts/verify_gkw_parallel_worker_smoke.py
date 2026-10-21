#!/usr/bin/env python3
"""Fail closed unless the minimal GKW 2+1 GPU chain smoke completed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    run_root = _arguments().run_root.expanduser().resolve()
    result_path = run_root / "result" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("schema") != "pde-reference-agent-iteration-result-v5"
        or result.get("rounds_completed") != 1
        or result.get("terminal", {}).get("evaluation_status") != "valid"
        or result.get("terminal", {}).get("root_count") != 1
        or result.get("terminal", {}).get("member_case_count") != 2
    ):
        raise RuntimeError("smoke did not finish the common terminal chain")

    rounds = result.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 1:
        raise RuntimeError("smoke round publication is incomplete")
    round_result = rounds[0]
    completion = round_result.get("search_completion", {})
    if (
        round_result.get("agent_failures") != {}
        or round_result.get("development_case_count") != 4
        or len(round_result.get("submitted_candidates", ())) != 2
        or len(completion) != 2
        or not all(row.get("sufficient") is True for row in completion.values())
    ):
        raise RuntimeError("one or more smoke Agent chains were incomplete")

    provenance = result.get("provenance", {})
    execution = provenance.get("agent_worker_execution", {})
    controller = provenance.get("controller_cuda", {})
    execution_rounds = execution.get("rounds")
    if (
        execution.get("worker_count") != 2
        or execution.get("controller", {}).get("gpu_device")
        != controller.get("visible_devices")
        or controller.get("device_count") != 1
        or controller.get("current_device") != 0
        or not isinstance(execution_rounds, list)
        or len(execution_rounds) != 1
        or execution_rounds[0].get("same_round_barrier") is not True
        or execution_rounds[0].get("overlap_verified") is not True
    ):
        raise RuntimeError("smoke controller/worker topology was not verified")
    workers = execution_rounds[0].get("workers")
    if not isinstance(workers, list) or len(workers) != 2:
        raise RuntimeError("smoke worker audit is incomplete")
    pids = {row.get("pid") for row in workers}
    gpu_tokens = {row.get("gpu_device") for row in workers}
    if (
        len(pids) != 2
        or len(gpu_tokens) != 2
        or controller.get("visible_devices") in gpu_tokens
        or any(
            row.get("status") != "complete"
            or row.get("cuda_device_count") != 1
            or row.get("cuda_current_device") != 0
            or row.get("visible_devices") != row.get("gpu_device")
            or row.get("plugin_identity_verified") is not True
            for row in workers
        )
    ):
        raise RuntimeError("smoke workers were not distinct singleton-GPU processes")

    agent_audits = tuple(
        run_root.glob(
            "_controller/agent_workers/round-0001/agent-*/"
            "agent_runtime/**/scripted-gpu-smoke-agent-audit.json"
        )
    )
    if len(agent_audits) != 2:
        raise RuntimeError("scripted Agent GPU audits are incomplete")
    scripted_tokens = set()
    for path in agent_audits:
        audit = json.loads(path.read_text(encoding="utf-8"))
        gpu = audit.get("gpu", {})
        if (
            gpu.get("torch_device_count") != 1
            or gpu.get("torch_current_device") != 0
            or audit.get("initial_release", {}).get("parent_linkage_valid") is not True
            or audit.get("screen", {}).get("completed_root_count") != 1
            or audit.get("submission_written") is not True
        ):
            raise RuntimeError("scripted Agent did not complete its real GPU/RPC chain")
        scripted_tokens.add(gpu.get("cuda_visible_devices"))
    if scripted_tokens != gpu_tokens:
        raise RuntimeError("scripted Agent GPU tokens differ from worker assignments")
    print(json.dumps({"status": "passed", "run_root": str(run_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
