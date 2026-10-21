"""Explicit, bounded GPU assignment for candidate and development workers.

These helpers do not discover or grant devices. The allocation owner supplies
the exact permitted CUDA tokens. They are resource coordination, not a hostile
code security boundary; OS/cgroup isolation remains the deployment's job.
"""

from __future__ import annotations

import math
import os
import re
from threading import Condition
import time
from typing import Mapping, Sequence


def validated_gpu_devices(devices: Sequence[str]) -> tuple[str, ...]:
    if isinstance(devices, (str, bytes)):
        raise TypeError("GPU devices must be a sequence of explicit CUDA tokens")
    tokens = tuple(devices)
    if any(not isinstance(token, str) or not re.fullmatch(
            r"(?:[0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9/-]+)", token)
            for token in tokens):
        raise ValueError("GPU devices must contain explicit numeric, GPU UUID or MIG tokens")
    if len(set(tokens)) != len(tokens):
        raise ValueError("GPU devices must be distinct")
    return tokens


def validated_memory_fraction(value: float | None) -> float | None:
    if value is None:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 < value <= 1):
        raise ValueError("GPU memory fraction must be finite and in (0, 1]")
    return float(value)


class GpuDevicePool:
    """Bound active leases per assigned device; default to one slot.

    Short acquire timeouts let the owning service enforce cancellation and its
    outer deadline while waiting. A device whose process cleanup is uncertain
    is quarantined rather than handed to another candidate.
    """

    def __init__(self, devices: Sequence[str], *, slots_per_device: int = 1) -> None:
        self.devices = validated_gpu_devices(devices)
        if not self.devices:
            raise ValueError("GPU pool needs at least one assigned device")
        if isinstance(slots_per_device, bool) or not isinstance(slots_per_device, int) or slots_per_device < 1:
            raise ValueError("slots_per_device must be a positive integer")
        self.slots_per_device = slots_per_device
        self._active_counts = {device: 0 for device in self.devices}
        self._condition = Condition()
        self._leased: set[str] = set()
        self._quarantined: set[str] = set()

    def acquire(self, *, timeout: float = 0.1,
                devices: Sequence[str] | None = None) -> str | None:
        allowed = self.devices if devices is None else validated_gpu_devices(devices)
        if not allowed or not set(allowed).issubset(self.devices):
            raise ValueError("GPU lease must select devices assigned to this pool")
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("GPU lease timeout must be finite and nonnegative")
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if set(allowed).issubset(self._quarantined):
                    raise RuntimeError("All requested GPUs are quarantined after uncertain cleanup")
                for device in allowed:
                    if self._active_counts[device] < self.slots_per_device and device not in self._quarantined:
                        self._active_counts[device] += 1
                        self._leased.add(device)
                        return device
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def release(self, device: str, *, healthy: bool = True) -> None:
        if not isinstance(healthy, bool):
            raise TypeError("GPU release healthy must be a boolean")
        with self._condition:
            if device not in self._leased:
                raise ValueError("GPU device has no active lease")
            self._active_counts[device] -= 1
            if self._active_counts[device] == 0:
                self._leased.remove(device)
            if not healthy:
                self._quarantined.add(device)
            self._condition.notify_all()

    def snapshot(self) -> Mapping[str, object]:
        with self._condition:
            return {"devices": list(self.devices), "leased": sorted(self._leased),
                    "quarantined": sorted(self._quarantined), "slots_per_device": self.slots_per_device,
                    **({"active_leases_by_device": dict(self._active_counts)}
                       if self.slots_per_device > 1 else {})}


def initialize_cuda_device(assigned_gpu: str, *, memory_fraction: float | None = None) -> dict[str, object]:
    """Fail closed unless this fresh worker can run on its single assigned GPU."""
    validated_gpu_devices((assigned_gpu,))
    fraction = validated_memory_fraction(memory_fraction)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != assigned_gpu:
        raise RuntimeError("CUDA visibility differs from the assigned worker GPU")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Requested GPU is unavailable; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    if fraction is not None:
        # PyTorch allocator budget only; not a hard security/device memory cap.
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
    tensor = torch.ones((16, 16), device="cuda:0")
    result = tensor @ tensor
    torch.cuda.synchronize(0)
    if float(result.sum().item()) != 4096.0:
        raise RuntimeError("Assigned GPU readiness kernel returned an invalid value")
    del result, tensor
    properties = torch.cuda.get_device_properties(0)
    uuid = str(getattr(properties, "uuid", ""))
    if not uuid:
        raise RuntimeError("Assigned GPU identity cannot be verified: UUID unavailable")
    if assigned_gpu.startswith("GPU-"):
        normalize = lambda value: value.lower().removeprefix("gpu-").replace("-", "")
        if normalize(assigned_gpu) != normalize(uuid):
            raise RuntimeError("Runtime GPU UUID differs from its assigned token")
    torch.cuda.reset_peak_memory_stats(0)
    return {"schema": "candidate-cuda-execution-v1", "device": "cuda:0",
            "cuda_visible_devices": assigned_gpu, "uuid": uuid,
            "name": str(properties.name), "device_count": 1,
            "readiness_kernel_passed": True, "synchronized": False,
            "memory_fraction": fraction, "memory_limit_scope": "pytorch_allocator_advisory"}


def synchronize_cuda_device(metadata: dict[str, object]) -> None:
    """Synchronize work before ending the charged worker lifetime and audit it."""
    import torch

    torch.cuda.synchronize(0)
    metadata.update(synchronized=True,
                    peak_memory_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
                    peak_memory_reserved_bytes=int(torch.cuda.max_memory_reserved(0)))


def validate_cuda_metadata(value: object, assigned_gpu: str) -> dict[str, object]:
    if (not isinstance(value, Mapping)
            or value.get("schema") != "candidate-cuda-execution-v1"
            or value.get("device") != "cuda:0"
            or value.get("cuda_visible_devices") != assigned_gpu
            or value.get("device_count") != 1
            or value.get("readiness_kernel_passed") is not True
            or value.get("synchronized") is not True
            or not isinstance(value.get("uuid"), str) or not value["uuid"]
            or not isinstance(value.get("name"), str) or not value["name"]):
        raise ValueError("Candidate GPU readiness or synchronization evidence is invalid")
    for name in ("peak_memory_allocated_bytes", "peak_memory_reserved_bytes"):
        if isinstance(value.get(name), bool) or not isinstance(value.get(name), int) or value[name] < 0:
            raise ValueError("Candidate GPU memory evidence is invalid")
    return dict(value)
