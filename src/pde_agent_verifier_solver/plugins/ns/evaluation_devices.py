"""Scoped copies of frozen NS fields for Controller public Oracle queries."""
from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from weakref import WeakSet

from pde_agent_verifier_solver.sdk.evaluation_devices import (
    current_evaluation_device_pool, register_cuda_workspace_cleanup,
)


def public_evaluation_answer(case_id, answer):
    """Keep reference answers untouched; route only a public-query adapter."""
    from .continuous_backend import StructuredNSMMSAnswer

    pool = current_evaluation_device_pool()
    if (pool is None or not pool.devices or type(answer) is not StructuredNSMMSAnswer
            or not answer._thread_safe_field):
        return answer, None
    register_cuda_workspace_cleanup(pool, answer.raw_answer.torch)
    device = pool.device_for(case_id)
    resource = pool.get_resource(("ns-public-oracle", device),
                                 lambda: _NSDeviceAnswers(device))
    return resource.answer(case_id, answer), resource


def _copy_answer(answer, device):
    from .continuous_backend import StructuredNSMMSAnswer, _NS_CUDA_CAPTURE_LOCK

    field = answer.raw_answer
    torch = field.torch
    with _NS_CUDA_CAPTURE_LOCK.shared_query(), torch.cuda.device(device):
        with torch.inference_mode(False), torch.no_grad():
            coefficients = type(field.coefficients)(
                psi=field.coefficients.psi.to(device=device, copy=True),
                pressure=field.coefficients.pressure.to(device=device, copy=True),
            )
            copied = type(field)(
                coefficients, config=field.config, geometry_spec=field.geometry_spec,
                canonical_geometry=(None if field.canonical_geometry is None else
                                    field.canonical_geometry.to(device=device, copy=True)),
                continuous_mms_abi=field.continuous_mms_abi,
                domain_bounds=field.domain_bounds, viscosity=field.viscosity,
                t0=field.t0, t1=field.t1, device=device,
            )
        # Queries run on different science threads. Publish only completed data.
        torch.cuda.synchronize(device)
    return StructuredNSMMSAnswer(copied, public_geometry=answer.public_geometry,
                                public_metadata=answer.public_metadata)


class _NSDeviceAnswers:
    """Bound the idle cache; active sessions own at most their worker's field."""

    def __init__(self, device, *, cache_limit=64):
        self.device = device
        self.cache_limit = cache_limit
        self._answers = OrderedDict()
        self._sessions = WeakSet()
        self._lock = RLock()
        self._closed = False
        self._initialization_failed = False
        self._analytic_cuda_cache = None

    def answer(self, case_id, original):
        key = (case_id, id(original))
        with self._lock:
            if self._closed:
                raise RuntimeError("NS evaluation device resource is closed")
            if key not in self._answers:
                try:
                    self._answers[key] = _copy_answer(original, self.device)
                except BaseException:
                    # A caller may retain the failing constructor traceback and
                    # its GPU tensors. Cleanup cannot certify this card as free.
                    self._initialization_failed = True
                    raise
            result = self._answers[key]
            if result._analytic_field and result.raw_answer.coefficients.psi.device.type == "cuda":
                if self._analytic_cuda_cache is None:
                    from .analytic_cuda import NSAnalyticCudaCache

                    self._analytic_cuda_cache = NSAnalyticCudaCache(self.device)
                result._analytic_cuda_cache = self._analytic_cuda_cache
            self._answers.move_to_end(key)
            while len(self._answers) > self.cache_limit:
                self._answers.popitem(last=False)
            return result

    def attach(self, session):
        with self._lock:
            if self._closed:
                session.close()
                raise RuntimeError("NS evaluation device resource is closed")
            self._sessions.add(session)

    def close(self):
        import torch

        with self._lock:
            if self._closed:
                if self._initialization_failed:
                    raise RuntimeError("NS GPU cleanup unconfirmed after field copy failure")
                return
            self._closed = True
            sessions = tuple(self._sessions)
        # Closing each session waits for its in-flight call and drops bound
        # methods, including fields evicted from the bounded idle cache.
        for session in sessions:
            session.close()
        # Sessions drain before graph buffers are freed and borrowed GPUs returned.
        if self._analytic_cuda_cache is not None:
            try:
                self._analytic_cuda_cache.close()
            except BaseException:
                self._initialization_failed = True
                raise
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            with self._lock:
                self._answers.clear()
                self._sessions.clear()
            torch.cuda.empty_cache()
        if self._initialization_failed:
            raise RuntimeError("NS GPU cleanup unconfirmed after field copy failure")
