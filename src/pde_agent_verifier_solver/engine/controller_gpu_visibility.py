"""Resolve Controller evaluation devices before a new process initializes CUDA."""
from __future__ import annotations

import os
import sys
from typing import MutableMapping, Sequence


def configure_controller_evaluation_devices(
    controller_gpu: str,
    worker_gpus: Sequence[str],
    *,
    environment: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Keep the Controller on cuda:0 and expose only already-visible devices.

    Updated launchers retain their full Slurm allocation. Older launchers that
    expose only the Controller GPU keep their original single-device behavior;
    this function never expands visibility using unverified command-line IDs.
    Call before constructing a Generator or other CUDA state.
    """
    if isinstance(worker_gpus, (str, bytes)):
        raise TypeError("worker_gpus must be a sequence of GPU tokens")
    devices = (controller_gpu, *tuple(worker_gpus))
    if any(not isinstance(d, str) or not d or d.strip() != d
           or any(c.isspace() for c in d) or "," in d for d in devices):
        raise ValueError("GPU tokens must be nonempty, whitespace-free identifiers")
    if len(devices) != len(set(devices)):
        raise ValueError("Controller and Agent GPU tokens must be distinct")
    env = os.environ if environment is None else environment
    raw_visible = env.get("CUDA_VISIBLE_DEVICES", "")
    if not isinstance(raw_visible, str):
        raise TypeError("CUDA_VISIBLE_DEVICES must be a string")
    visible = tuple(d.strip() for d in raw_visible.split(","))
    if any(not d or any(c.isspace() for c in d) for d in visible):
        raise ValueError("CUDA_VISIBLE_DEVICES must name nonempty GPU tokens")
    if len(visible) != len(set(visible)):
        raise ValueError("CUDA_VISIBLE_DEVICES contains duplicate GPU tokens")
    if visible == (controller_gpu,):
        return {}  # Explicit legacy single-device configuration, no borrowing.
    if not set(devices).issubset(visible):
        raise ValueError("Controller evaluation GPUs must already be CUDA-visible")
    requested = ",".join(devices)
    if raw_visible != requested:
        torch = sys.modules.get("torch")
        cuda = getattr(torch, "cuda", None)
        initialized = getattr(cuda, "is_initialized", None)
        if callable(initialized) and initialized():
            raise RuntimeError("configure Controller GPU visibility before CUDA initialization")
        env["CUDA_VISIBLE_DEVICES"] = requested
    return {physical: f"cuda:{index}" for index, physical in enumerate(devices)}
