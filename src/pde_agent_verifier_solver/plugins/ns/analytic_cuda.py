"""Evaluation-scoped CUDA graphs for the audited frozen NS analytic field.

The owning NS evaluation device resource closes its sessions before this cache.
It is not a process-global cache and must not run during Generator work that
has not joined the NS capture guard. Callers provide already-validated physical
coordinates in x/y/t order and completed, immutable coefficients on this GPU.
All Controller admission/capture waiting uses the existing credited guards;
actual eager work, graph preparation and replay remain numerical service time.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import Condition, RLock
from contextlib import contextmanager
from time import perf_counter


@dataclass
class _GraphEntry:
    field: object
    coordinates: object
    graph: object
    output: object

    def close(self):
        try:
            if self.graph is not None:
                self.graph.reset()
        finally:
            self.graph = self.output = self.coordinates = self.field = None


def _field_signature(field):
    """Read frozen metadata without repeating polygon numerical validation."""
    geometry = field.geometry_spec
    if geometry is None:
        raise NotImplementedError("NS analytic CUDA graphs require polygon geometry")
    config = field.config
    if not config.is_legacy_basis:
        raise NotImplementedError("NS CUDA graph fastpath is only audited for cosine/Chebyshev fields")
    return (
        field.dtype, field.continuous_mms_abi, tuple(field.domain_bounds),
        float(field.t0), float(field.t1), float(field.viscosity),
        geometry.abi_version, tuple(geometry.domain_bounds), geometry.outer_boundary,
        tuple(tuple(tuple(point) for point in polygon) for polygon in geometry.obstacle_polygons),
        tuple(sorted(vars(config).items())),
        tuple(field.coefficients.psi.shape), tuple(field.coefficients.pressure.shape),
    )


class _ParallelGraphLanes:
    """Bound independent graph buffers; sessions drain before resource cleanup."""
    maximum_lanes = 8

    def __init__(self, device, max_entries):
        self.device, self.max_entries = device, max_entries
        self.condition = Condition()
        self.caches = {}
        self.busy = set()
        self.closed = False

    @contextmanager
    def lease(self, concurrency):
        from pde_agent_verifier_solver.sdk._candidate_timing import controller_queue_wait
        limit = min(concurrency, self.maximum_lanes)
        with self.condition:
            with controller_queue_wait() as check_cancelled:
                while True:
                    check_cancelled()
                    if self.closed:
                        raise RuntimeError("NS analytic CUDA cache is closed")
                    slot = next((i for i in range(limit) if i not in self.busy), None)
                    if slot is not None:
                        break
                    self.condition.wait(.05)
            if slot not in self.caches:
                self.caches[slot] = NSAnalyticCudaCache(
                    self.device, max_entries=self.max_entries, _parallel_lane=True)
            self.busy.add(slot)
            cache = self.caches[slot]
        try:
            yield cache
        finally:
            with self.condition:
                self.busy.remove(slot)
                self.condition.notify_all()

    def diagnostics(self):
        with self.condition:
            return [cache.diagnostics for cache in self.caches.values()]

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
            while self.busy:
                self.condition.wait(.05)
            caches = tuple(self.caches.values())
        error = None
        for cache in caches:
            try:
                cache.close()
            except BaseException as caught:
                error = error or caught
        if error is not None:
            raise error


class NSAnalyticCudaCache:
    """A small graph cache owned by one hidden evaluation's NS GPU resource.

    ``evaluate`` owns device admission itself; do not wrap it in an existing NS
    query context, whose shared guard cannot be upgraded to a capture writer.
    Unsupported large point counts use the same analytic eager kernel. Capture
    starts on the twelfth forcing or twentieth velocity occurrence of a key,
    amortizing preparation over frequently reused queries. Subsequent Cases
    copy both coefficient arrays and all coordinates into private static buffers.
    Parallel calls lease separate caches/streams (at most eight), each bounded
    by max_entries. They never replay into another active call's buffers.
    """

    max_points = 4096
    _capture_after_occurrences = {"forcing": 12, "velocity": 20}

    def __init__(self, device, max_entries=8, *, _parallel_lane=False):
        import torch

        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or not 1 <= max_entries <= 8:
            raise ValueError("NS analytic CUDA max_entries must be an integer from 1 to 8")
        requested = torch.device(device)
        if requested.type != "cuda":
            raise ValueError("NS analytic CUDA cache requires a CUDA device")
        index = requested.index if requested.index is not None else torch.cuda.current_device()
        self.device = torch.device("cuda", index)
        self.max_entries = max_entries
        self._torch = torch
        self._entries = OrderedDict()
        self._seen = OrderedDict()
        self._stream = None
        self._lock = RLock()
        self._closed = False
        self._failed = False
        self._capture_count = 0
        self._preparation_seconds = 0.0
        self._replays = 0
        self._eager_calls = 0
        self._evictions = 0
        self._parallel = None if _parallel_lane else _ParallelGraphLanes(self.device, max_entries)

    @property
    def diagnostics(self):
        with self._lock:
            result = {
                "device": str(self.device), "max_entries": self.max_entries,
                "max_points": self.max_points,
                "capture_after_occurrences": dict(self._capture_after_occurrences),
                "capture_count": self._capture_count,
                "preparation_seconds": self._preparation_seconds,
                "replays": self._replays, "eager_calls": self._eager_calls,
                "entries": len(self._entries), "pending_keys": len(self._seen),
                "evictions": self._evictions, "closed": self._closed, "failed": self._failed,
            }
            lanes = self._parallel.diagnostics() if self._parallel is not None else []
            if lanes:
                for key in ("capture_count", "preparation_seconds", "replays", "eager_calls",
                            "entries", "pending_keys", "evictions"):
                    result[key] += sum(lane[key] for lane in lanes)
                result["failed"] = result["failed"] or any(lane["failed"] for lane in lanes)
                result["parallel_lanes"] = len(lanes)
                result["maximum_lanes"] = self._parallel.maximum_lanes
                result["max_entries_per_lane"] = self.max_entries
            return result

    def _compute(self, field, coordinates, kind, uniform):
        from agent_verifier.neural_generator.structured_ns_analytic import (
            analytic_forcing, analytic_velocity,
        )

        function = analytic_forcing if kind == "forcing" else analytic_velocity
        values = function(field, coordinates, uniform_time=uniform)
        return self._torch.stack(values, dim=-1)

    def _eager(self, field, coordinates, kind, uniform):
        torch = self._torch
        tensor = torch.from_numpy(coordinates).to(device=self.device, dtype=field.dtype)
        result = self._compute(field, tensor, kind, uniform).detach().cpu().numpy().copy()
        self._eager_calls += 1
        return result

    def _capture(self, field, coordinates, kind, uniform):
        torch = self._torch
        started = perf_counter()
        try:
            # Construct exactly one private field per graph. The ordinary field
            # constructor validates metadata here, outside replay's hot path.
            coefficients = type(field.coefficients)(
                psi=field.coefficients.psi.detach().clone(),
                pressure=field.coefficients.pressure.detach().clone(),
            )
            static_field = type(field)(
                coefficients, config=field.config, geometry_spec=field.geometry_spec,
                continuous_mms_abi=field.continuous_mms_abi,
                domain_bounds=field.domain_bounds, viscosity=field.viscosity,
                t0=field.t0, t1=field.t1, device=self.device,
            )
            static_coordinates = torch.from_numpy(coordinates).to(device=self.device, dtype=field.dtype)
            for _ in range(3):
                warm_output = self._compute(static_field, static_coordinates, kind, uniform)
            self._stream.synchronize()
            del warm_output
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self._stream):
                output = self._compute(static_field, static_coordinates, kind, uniform)
            self._stream.synchronize()
            self._capture_count += 1
            return _GraphEntry(static_field, static_coordinates, graph, output)
        finally:
            self._preparation_seconds += perf_counter() - started

    def _replay(self, entry, field, coordinates):
        entry.field.coefficients.psi.copy_(field.coefficients.psi)
        entry.field.coefficients.pressure.copy_(field.coefficients.pressure)
        entry.coordinates.copy_(self._torch.from_numpy(coordinates))
        entry.graph.replay()
        # The CPU copy completes this stream's numerical work before releasing
        # FIFO; numpy.copy prevents retained responses from aliasing future ones.
        result = entry.output.detach().cpu().numpy().copy()
        self._replays += 1
        return result

    def evaluate(self, field, physical_coords_numpy_xyt, kind):
        import numpy as np
        from .continuous_backend import _NS_CUDA_CAPTURE_LOCK, _NS_CUDA_QUERY_ADMISSION

        if kind not in {"forcing", "velocity"}:
            raise ValueError("NS analytic CUDA kind must be forcing or velocity")
        coordinates = np.array(physical_coords_numpy_xyt, dtype=np.float64, copy=True, order="C")
        if coordinates.ndim != 2 or coordinates.shape[1] != 3 or coordinates.shape[0] < 1:
            raise ValueError("NS analytic CUDA coordinates must be nonempty [N,3] x/y/t")
        if field.coefficients.psi.device != self.device or field.coefficients.pressure.device != self.device:
            raise ValueError("NS analytic CUDA field must reside on this cache's GPU")
        from pde_agent_verifier_solver.engine.fair_execution import current_oracle_policy
        concurrency = current_oracle_policy()[1]
        if self._parallel is not None and concurrency > 1:
            with self._lock:
                if self._closed or self._failed:
                    raise RuntimeError("NS analytic CUDA cache is closed or unavailable")
            with self._parallel.lease(concurrency) as lane:
                try:
                    return lane.evaluate(field, coordinates, kind)
                except BaseException:
                    if lane._failed:
                        with self._lock:
                            self._failed = True
                    raise
        uniform = bool(np.all(coordinates[:, 2] == coordinates[0, 2]))
        key = (self.device.index, len(coordinates), kind, uniform, _field_signature(field))
        torch = self._torch
        with _NS_CUDA_QUERY_ADMISSION.for_device(self.device.index), self._lock:
            if self._closed:
                raise RuntimeError("NS analytic CUDA cache is closed")
            if self._failed:
                raise RuntimeError("NS analytic CUDA cache is unavailable after a numerical CUDA failure")
            entry = self._entries.get(key)
            occurrences = self._seen.get(key, 0) + 1
            should_capture = (entry is None and len(coordinates) <= self.max_points
                              and occurrences >= self._capture_after_occurrences[kind])
            # Never request a writer while holding a reader. All graph/eager
            # kernels and host copies remain inside the chosen guard.
            guard = _NS_CUDA_CAPTURE_LOCK if should_capture else _NS_CUDA_CAPTURE_LOCK.shared_query()
            with guard:
                try:
                    with torch.cuda.device(self.device), torch.inference_mode(False), torch.no_grad(), \
                            torch.autocast(device_type="cuda", enabled=False):
                        if self._stream is None:
                            self._stream = torch.cuda.Stream(device=self.device)
                        self._stream.wait_stream(torch.cuda.default_stream(self.device))
                        with torch.cuda.stream(self._stream):
                            if entry is not None:
                                self._entries.move_to_end(key)
                                return self._replay(entry, field, coordinates)
                            if should_capture:
                                self._seen.pop(key, None)
                                if len(self._entries) == self.max_entries:
                                    _, previous = self._entries.popitem(last=False)
                                    previous.close()
                                    self._evictions += 1
                                entry = self._capture(field, coordinates, kind, uniform)
                                self._entries[key] = entry
                                return self._replay(entry, field, coordinates)
                            if len(coordinates) <= self.max_points:
                                self._seen[key] = occurrences
                                self._seen.move_to_end(key)
                                while len(self._seen) > self.max_entries * 4:
                                    self._seen.popitem(last=False)
                            return self._eager(field, coordinates, kind, uniform)
                except BaseException:
                    # A retained exception traceback can itself own CUDA
                    # buffers. Propagate the failure and do not certify this
                    # resource's GPU as reusable merely because close ran.
                    self._failed = True
                    raise

    def close(self):
        from .continuous_backend import _NS_CUDA_CAPTURE_LOCK, _NS_CUDA_QUERY_ADMISSION

        torch = self._torch
        # Do not hold a device permit while waiting for other lanes to drain.
        parallel_error = None
        if self._parallel is not None:
            try:
                self._parallel.close()
            except BaseException as caught:
                parallel_error = caught
                self._failed = True
        with _NS_CUDA_QUERY_ADMISSION.for_device(self.device.index), self._lock:
            if not self._closed:
                error = None
                with _NS_CUDA_CAPTURE_LOCK.shared_query(), torch.cuda.device(self.device):
                    self._closed = True
                    try:
                        if self._stream is not None:
                            self._stream.synchronize()
                    except BaseException as caught:
                        self._failed = True
                        error = caught
                    for entry in self._entries.values():
                        try:
                            entry.close()
                        except BaseException as caught:
                            self._failed = True
                            error = error or caught
                    self._entries.clear()
                    self._seen.clear()
                    self._stream = None
                if error is not None:
                    raise error
            if parallel_error is not None:
                raise parallel_error
            if self._failed:
                raise RuntimeError("NS CUDA cleanup unconfirmed after analytic graph execution failure")
