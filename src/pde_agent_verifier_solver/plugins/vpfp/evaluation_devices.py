"""Per-device public VPFP Oracle replicas, alive only during hidden evaluation."""
from __future__ import annotations

from copy import copy, deepcopy
from threading import RLock
from weakref import WeakSet

from pde_agent_verifier_solver.sdk.evaluation_devices import (
    current_evaluation_device_pool, register_cuda_workspace_cleanup,
)


def public_evaluation_session(runtime, resident):
    """Unknown/CPU runtimes keep their original public session implementation."""
    from .perturbation_backend import VPFPLegacyStructuredRuntime

    pool = current_evaluation_device_pool()
    if pool is None or not pool.devices or type(runtime) is not VPFPLegacyStructuredRuntime:
        return None
    if not _supports_replication(runtime.expanded_model):
        return None
    register_cuda_workspace_cleanup(pool, runtime.api.torch)
    device = pool.device_for(resident.case.case_id)
    # The pinned context changes process-wide FP32 settings. Keep IEEE fixed
    # for the entire scope so concurrent device-local contexts restore IEEE.
    # Register before device replicas so precision is restored after they close.
    pool.get_resource(("vpfp-coordinate-precision", id(runtime)),
                      lambda: _CoordinatePrecision(runtime))
    resource = pool.get_resource(("vpfp-public-oracle", id(runtime), device),
                                 lambda: _VPFPDeviceRuntime(runtime, device))
    return resource.session(resident)


def _supports_replication(model):
    parameter = next(model.parameters())
    return bool(
        getattr(parameter.device, "type", None) == "cuda"
        and type(model).__name__ == "_VPFPContinuousHermiteField"
        and type(model).__module__.endswith(".continuous_hermite")
        and not model.training
        and not any(p.requires_grad for p in model.parameters())
    )


class _CoordinatePrecision:
    def __init__(self, runtime):
        self._context = runtime.api.ieee_float32_coordinate_matmul(runtime.expanded_model)
        self._context.__enter__()

    def close(self):
        context, self._context = self._context, None
        if context is not None:
            context.__exit__(None, None, None)


def _copy_runtime(original, device):
    """Copy existing weights/buffers without building a model or consuming RNG."""
    torch = original.api.torch
    replica = copy(original)
    replica._runtime_lock = RLock()
    replica._resident_cases = {}
    replica.device = device
    model = None
    try:
        with original._runtime_lock, torch.cuda.device(device), torch.no_grad():
            # Every nested module's torch attribute points to this same module.
            model = deepcopy(original.expanded_model, {id(torch): torch})
            model.to(device=device)
            model.eval().requires_grad_(False)
            torch.cuda.synchronize(device)
        replica.expanded_model = model
        return replica
    except BaseException:
        # A retained traceback must not retain a half-built borrowed-GPU model.
        model = None
        replica = None
        raise


class _VPFPDeviceRuntime:
    def __init__(self, original, device):
        self.device = device
        self.runtime = _copy_runtime(original, device)
        self._lock = RLock()
        self._references = {}
        self._sessions = WeakSet()
        self._closed = False
        self._initialization_failed = False

    def session(self, resident):
        from .perturbation_backend import _ResidentCase, _VPFPPublicSession, _latent_for_model

        with self._lock:
            if self._closed:
                raise RuntimeError("VPFP evaluation device resource is closed")
            runtime = self.runtime
            case_id = resident.case.case_id
            with runtime._runtime_lock:
                if case_id not in runtime._resident_cases:
                    torch = runtime.api.torch
                    try:
                        with torch.cuda.device(self.device), torch.no_grad():
                            latent = _latent_for_model(resident.latent, runtime.expanded_model, torch)
                            # Own the tensor even when the original uses this same GPU.
                            latent = latent.detach().clone()
                            torch.cuda.synchronize(self.device)
                        runtime._resident_cases[case_id] = _ResidentCase(
                            resident.case, resident.public_snapshot, latent,
                            resident.equation_condition,
                        )
                    except BaseException:
                        # A retained traceback can own a partly copied latent.
                        # Close known resources, but quarantine the borrowed GPU.
                        self._initialization_failed = True
                        raise
                self._references[case_id] = self._references.get(case_id, 0) + 1
            session = _VPFPPublicSession(runtime, resident.case)
            session._evaluation_resource = self
            # Each replica has its own runtime lock. Different GPUs can now
            # serve requests independently of the Generator operation lock.
            session.public_calls_thread_safe = True
            self._sessions.add(session)
            return session

    def release(self, case_id):
        with self._lock:
            count = self._references.get(case_id, 0)
            if count <= 1:
                self._references.pop(case_id, None)
                if self.runtime is not None:
                    with self.runtime._runtime_lock:
                        self.runtime._resident_cases.pop(case_id, None)
            else:
                self._references[case_id] = count - 1

    def close(self):
        with self._lock:
            if self._closed:
                if self._initialization_failed:
                    raise RuntimeError("VPFP GPU cleanup unconfirmed after latent copy failure")
                return
            self._closed = True
            sessions = tuple(self._sessions)
        for session in sessions:
            session.close()
        runtime = self.runtime
        torch = runtime.api.torch
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            with self._lock:
                runtime._resident_cases.clear()
                self._references.clear()
                self._sessions.clear()
                runtime.expanded_model = None
                self.runtime = None
            torch.cuda.empty_cache()
        if self._initialization_failed:
            raise RuntimeError("VPFP GPU cleanup unconfirmed after latent copy failure")
