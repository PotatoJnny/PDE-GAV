"""Explicit GPU-candidate opt-in; old recipes remain CPU-only.

This validates allocation metadata, not hardware availability or OS isolation.
The execution worker must independently prove that its CUDA device works.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
import math
import re


GPU_EXECUTION_PROTOCOL = "shared_gpu_v1"
_TOKEN = re.compile(r"^(?:[0-9]+|GPU-[0-9A-Fa-f-]+|MIG-[A-Za-z0-9._:/-]+)$")


def gpu_execution_settings(execution: Mapping) -> dict | None:
    """Validate the narrow additive config without claiming a CUDA probe passed."""
    if "gpu_execution" not in execution:
        return None
    value = execution["gpu_execution"]
    required = {"protocol", "candidate_devices"}
    if (not isinstance(value, Mapping) or not required <= set(value)
            or set(value) - required - {"agent_sandbox_backend", "agent_runtime_sha256"}):
        raise ValueError("gpu_execution requires protocol/candidate_devices and optional pinned Agent sandbox")
    if value["protocol"] != GPU_EXECUTION_PROTOCOL:
        raise ValueError("unsupported GPU execution protocol")
    tokens = value["candidate_devices"]
    if (not isinstance(tokens, (list, tuple)) or not tokens
            or any(not isinstance(token, str) or not _TOKEN.fullmatch(token) for token in tokens)
            or len(set(tokens)) != len(tokens)):
        raise ValueError("candidate_devices must be distinct allocated GPU tokens")
    result = {"protocol": GPU_EXECUTION_PROTOCOL, "candidate_devices": list(tokens)}
    if "agent_sandbox_backend" in value:
        if value["agent_sandbox_backend"] not in ("legacy-landlock", "bwrap-nvidia-v1"):
            raise ValueError("unsupported Agent GPU sandbox backend")
        result["agent_sandbox_backend"] = value["agent_sandbox_backend"]
    if "agent_runtime_sha256" in value:
        result["agent_runtime_sha256"] = value["agent_runtime_sha256"]  # Legacy metadata only.
    return result


def legacy_execution_settings(execution: Mapping) -> dict:
    """Validate the extension, then expose the unchanged frozen legacy fields."""
    gpu_execution_settings(execution)
    return {key: value for key, value in execution.items() if key != "gpu_execution"}


def development_solver_placement(execution: Mapping) -> str:
    placement = execution.get("development_solver_placement", "shared")
    if placement not in ("shared", "agent_gpu"):
        raise ValueError("development_solver_placement must be shared or agent_gpu")
    if placement == "agent_gpu" and gpu_execution_settings(execution) is None:
        raise ValueError("agent_gpu placement requires an explicit separate Official GPU")
    if not isinstance(execution.get("trace_calls", False), bool):
        raise TypeError("execution.trace_calls must be a boolean")
    return placement


def candidate_execution_capabilities(execution: Mapping) -> dict:
    placement = development_solver_placement(execution)
    parallel = {key: execution.get(key, 1) for key in
                ("solver_batch_size", "gpu_slots_per_device", "oracle_concurrency_per_device")}
    if "development_solver_placement" in execution:
        parallel["development_solver_placement"] = placement
    if gpu_execution_settings(execution) is None:
        device = execution.get("candidate_device", "cpu")
        if device not in ("cpu", "cpu_or_cuda"):
            raise ValueError("candidate_device must be cpu or cpu_or_cuda")
        return {"device": device, "cuda_available": device == "cpu_or_cuda", **parallel}
    return {
        "device": "cuda", "cuda_available": True,
        "protocol": GPU_EXECUTION_PROTOCOL,
        "gpu_policy": ("one_exclusive_case_per_device" if parallel["gpu_slots_per_device"] == 1
                       else "bounded_shared_device"),
        **parallel,
        "cuda_probe_required": True,
    }


def agent_gpu_worker_settings(execution: Mapping) -> dict:
    settings = gpu_execution_settings(execution)
    if settings is None or "agent_sandbox_backend" not in settings:
        return {}
    policy = {"protocol": GPU_EXECUTION_PROTOCOL,
              "sandbox_backend": settings["agent_sandbox_backend"], "required": True}
    if "agent_runtime_sha256" in settings:
        policy["runtime_sha256"] = settings["agent_runtime_sha256"]
    return {"agent_gpu_execution": policy}


def candidate_gpu_kwargs(execution: Mapping, *, reserved_devices: Sequence[str],
                         environment: Mapping[str, str] | None = None) -> dict:
    """Require dedicated Candidate tokens within this allocation, never borrow.

    SLURM_JOB_GPUS remains the allocation record after the Controller reorders
    CUDA_VISIBLE_DEVICES. Prefer that record when present. Token aliases must
    be resolved consistently by the launcher; this function never guesses.
    """
    settings = gpu_execution_settings(execution)
    if settings is None:
        return {}
    env = os.environ if environment is None else environment
    allocated = env.get("SLURM_JOB_GPUS") or env.get("CUDA_VISIBLE_DEVICES", "")
    allocated_tokens = allocated.split(",") if allocated else []
    requested = set(settings["candidate_devices"])
    if (not allocated_tokens or any(not _TOKEN.fullmatch(token) for token in allocated_tokens)
            or len(set(allocated_tokens)) != len(allocated_tokens)
            or not requested <= set(allocated_tokens)):
        raise ValueError("GPU Candidate devices must match this allocation's explicit tokens")
    if requested.intersection(reserved_devices):
        raise ValueError("GPU Candidate devices must be separate from Agent and Controller devices")
    return {"gpu_devices": tuple(settings["candidate_devices"])}


def validate_gpu_run_budget(config: Mapping) -> None:
    """Do not attach the new GPU prompt to a legacy preparation-limited Task."""
    execution = config["execution"]
    if gpu_execution_settings(execution) is None:
        return
    for name in ("candidate_case_hard_wall_seconds", "agent_hard_wall_seconds"):
        value = execution.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if "agent_active_budget_seconds" in execution:
        raise ValueError("GPU execution uses the complete Agent hard-wall budget")
    task = config.get("task", {})
    if isinstance(task, Mapping) and task.get("candidate_compute_budget") is not None:
        raise ValueError("shared_gpu_v1 cannot retain a separate preparation budget; configure only the per-Case execution budget")
