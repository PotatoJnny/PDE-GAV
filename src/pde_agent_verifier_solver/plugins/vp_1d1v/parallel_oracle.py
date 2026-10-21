"""Private, phase-scoped Artemis oracle workers; one resident model per GPU.

Only already-issued hidden fields cross these trusted process pipes. No model,
truth trajectory, sidecar or Candidate-controlled object is sent. The caller
owns phase admission and returns to the ordinary backend when this pool closes.
"""

from __future__ import annotations

from dataclasses import asdict
import importlib
import multiprocessing
import os
from threading import RLock
import time
from typing import Mapping, Sequence


class ArtemisOracleWorkerError(RuntimeError):
    """A resident scientific worker failed, without silently changing lanes."""


def _plain(value):
    """Copy small CPU metadata, not the public case's large sampled arrays."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_plain(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"non-scalar Artemis oracle metadata: {type(value).__name__}")


def _case_descriptor(case):
    return dict(case_id=case.case_id, source=case.source.value,
                visibility=case.visibility.value, round_index=case.round_index,
                parent_case_id=case.parent_case_id, tags=tuple(case.tags),
                public_data={"equation": _plain(case.public_data["equation"])})


def _export_case(source, case):
    issued, use_presented = source._active_issued(case, label="parallel public")
    runtime = source.runtime
    handle = runtime._exact_query_handle(issued.manufactured)
    field = handle.field
    # These are the small original conditioning coordinates, not re-sampled
    # coordinates. Amplitudes and all learned parameters belong to the same
    # pinned model loaded in each worker.
    cpu = lambda value: value.detach().to(device="cpu").numpy().copy()
    return dict(
        case=_case_descriptor(case if use_presented else issued.case),
        condition_scores=cpu(field._condition_scores), latent=cpu(field._latent),
        curvatures=None if handle.curvatures is None else cpu(handle.curvatures),
        spatial_phase=asdict(handle.spatial_phase),
        audit=asdict(issued.manufactured.audit),
        artifact_sha256=source.artifact_sha256,
    )


class _NoTransferredTruth:
    def answer(self, query):
        raise PermissionError("parallel public workers do not carry hidden truth")


class _ResidentArtemis:
    """Bind frozen coordinates to the existing D5 classes, without manufacture."""

    def __init__(self, settings, device):
        from .product import load_artemis_backend

        self.binding = load_artemis_backend(**settings, device=device)
        self.source = self.binding.plugin.generator.backend.source
        self.runtime = self.source.runtime
        self.module = importlib.import_module(type(self.runtime).__module__)
        self.records = {}
        import torch

        current = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(current)
        self.device_metadata = dict(
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            device_count=torch.cuda.device_count(), current_device=current,
            name=properties.name, uuid=str(getattr(properties, "uuid", "")),
            cpu_threads=torch.get_num_threads(),
        )

    def register(self, descriptor):
        from pde_agent_verifier_solver.sdk import Case, CaseSource, CaseVisibility

        runtime, module = self.runtime, self.module
        if descriptor["artifact_sha256"] != self.source.artifact_sha256:
            raise ValueError("parallel worker runtime identity differs from issued case")
        row = dict(descriptor["case"])
        row["source"], row["visibility"] = CaseSource(row["source"]), CaseVisibility(row["visibility"])
        case = Case(**row)
        if case.case_id in self.records:
            raise ValueError("parallel case registered twice")
        field = module.ArtemisCompactFactorizedExactFieldV4(
            runtime.artifact.model, descriptor["condition_scores"], descriptor["latent"],
            jvp_mode=runtime.oracle_options.factorized_jvp_mode,
            trusted_source=runtime.oracle_options.exact_query_execution_mode == "trusted",
            vp_specialized=runtime.oracle_options.exact_query_kernel_mode == "vp_specialized",
        )
        import torch

        curvatures = descriptor["curvatures"]
        if curvatures is not None:
            curvatures = torch.as_tensor(curvatures, dtype=field.dtype, device=field.device).detach()
        audit_module = importlib.import_module(module.VP1DManufacturedCase.__module__)
        audit = audit_module.VP1DManufacturingAudit(**descriptor["audit"])
        handle = module._ExactQueryHandle(
            runtime_token=runtime._runtime_token, state_sha256=audit.state_sha256,
            field=field, curvatures=curvatures,
            spatial_phase=module._PeriodicSpatialPhase(**descriptor["spatial_phase"]),
        )
        manufactured = module.VP1DManufacturedCase(
            case_id=case.case_id, public_data=case.public_data,
            truth=_NoTransferredTruth(), hidden_handle=handle, audit=audit,
        )
        self.records[case.case_id] = (manufactured, case)

    def open(self, case_id, presented):
        from dataclasses import replace

        manufactured, case = self.records[case_id]
        # The parent source already validates presented causal projections.
        case = replace(case, public_data=presented["public_data"])
        return self.runtime.open_public_session(manufactured, case)

    def close(self):
        self.records.clear()


def _worker_main(connection, settings, physical_device, factory):
    # Spawn, never fork an initialized CUDA parent. The module has no torch
    # import; install the physical device mask before loading scientific code.
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_device
    resident = None
    sessions = {}
    try:
        resident = factory(settings, "cuda:0")
        connection.send({"ok": True, "ready": True, "pid": os.getpid(),
                         "device_metadata": dict(getattr(resident, "device_metadata", {}))})
        while True:
            request = connection.recv()
            kind = request["kind"]
            if kind == "shutdown":
                break
            session = None
            try:
                if kind == "register":
                    for descriptor in request["cases"]:
                        resident.register(descriptor)
                    result = None
                elif kind == "open":
                    session_id = request["session_id"]
                    if session_id in sessions:
                        raise ValueError("parallel session opened twice")
                    sessions[session_id] = resident.open(request["case_id"], request["case"])
                    session = sessions[session_id]
                    result = None
                elif kind == "close":
                    session = sessions.pop(request["session_id"])
                    result = None
                    session.close()
                elif kind == "query":
                    method = request["method"]
                    if method not in ("query_source", "query_initial_distribution"):
                        raise PermissionError("undeclared parallel public method")
                    session = sessions[request["session_id"]]
                    result = getattr(session, method)(request["points"])
                else:
                    raise ValueError("unknown parallel oracle operation")
                usage = {} if session is None else dict(session.usage_snapshot)
                connection.send({"ok": True, "result": result, "usage": usage})
            except Exception as error:
                usage = {} if session is None else dict(session.usage_snapshot)
                connection.send({"ok": False, "error_type": type(error).__name__,
                                 "error": str(error), "usage": usage})
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as error:
        try:
            connection.send({"ok": False, "error_type": type(error).__name__, "error": str(error)})
        except (OSError, EOFError):
            pass
    finally:
        for session in sessions.values():
            try:
                session.close()
            except Exception:
                pass
        if resident is not None:
            resident.close()
        connection.close()


class _Lane:
    def __init__(self, device):
        self.device = device
        self.lock = RLock()
        self.process = self.connection = None
        self.failed = False
        self.device_metadata = {}
        self.request_count = self.source_calls = self.distribution_calls = 0

    def receive(self, timeout):
        try:
            if not self.connection.poll(timeout):
                self.failed = True
                raise ArtemisOracleWorkerError(f"Artemis GPU {self.device} response timed out")
            return self.connection.recv()
        except (EOFError, OSError) as error:
            self.failed = True
            raise ArtemisOracleWorkerError(f"Artemis GPU {self.device} worker disconnected") from error

    def rpc(self, request, timeout):
        with self.lock:
            if self.failed or self.connection is None or not self.process.is_alive():
                raise ArtemisOracleWorkerError(f"Artemis GPU {self.device} worker is unavailable")
            try:
                self.connection.send(request)
            except (OSError, EOFError) as error:
                self.failed = True
                raise ArtemisOracleWorkerError(f"Artemis GPU {self.device} send failed") from error
            if request["kind"] == "query":
                self.request_count += 1
                if request["method"] == "query_source":
                    self.source_calls += 1
                else:
                    self.distribution_calls += 1
            return self.receive(timeout)


def _raise_worker_error(reply):
    if not reply.get("ok"):
        message = f"Artemis worker {reply.get('error_type', 'error')}: {reply.get('error', '')}"
        exception = {"ValueError": ValueError, "TypeError": TypeError,
                     "PermissionError": PermissionError}.get(reply.get("error_type"), ArtemisOracleWorkerError)
        raise exception(message)


class ArtemisParallelOraclePool:
    """A hidden-evaluation scope's stable case-to-GPU assignment."""

    def __init__(self, source, *, cases: Sequence, backend_settings: Mapping,
                 devices: Sequence[str], startup_timeout_seconds=180.0,
                 request_timeout_seconds=900.0, _resident_factory=_ResidentArtemis,
                 _exporter=_export_case):
        from pde_agent_verifier_solver.sdk import CaseVisibility

        self.source = source
        self.cases = tuple(cases)
        devices = tuple(str(device) for device in devices)
        if not devices or len(set(devices)) != len(devices) or any(not item or "," in item for item in devices):
            raise ValueError("parallel Artemis requires distinct physical GPU tokens")
        if not self.cases or len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("parallel Artemis requires distinct issued cases")
        if any(case.visibility is not CaseVisibility.HIDDEN for case in self.cases):
            raise PermissionError("parallel Artemis scope is hidden-evaluation only")
        if startup_timeout_seconds <= 0 or request_timeout_seconds <= 0:
            raise ValueError("parallel Artemis timeouts must be positive")
        self._settings = dict(backend_settings)
        for key in ("device", "guidance_adapter_factory", "reward_guidance_settings"):
            self._settings.pop(key, None)
        self._lanes = tuple(_Lane(device) for device in devices)
        self._assignments = {case.case_id: index % len(devices) for index, case in enumerate(self.cases)}
        self._startup_timeout = float(startup_timeout_seconds)
        self._request_timeout = float(request_timeout_seconds)
        self._resident_factory, self._exporter = _resident_factory, _exporter
        self._state_lock = RLock()
        self._started = self._closed = False
        self._cleanup_failed = False
        self._next_session_id = 0
        self._startup_wall_seconds = 0.0

    def start(self):
        with self._state_lock:
            if self._closed:
                raise RuntimeError("parallel Artemis pool is closed")
            if self._started:
                return self
            started = time.perf_counter()
            descriptors = [self._exporter(self.source, case) for case in self.cases]
            context = multiprocessing.get_context("spawn")
            try:
                for lane in self._lanes:
                    parent, child = context.Pipe(duplex=True)
                    lane.connection = parent
                    lane.process = context.Process(target=_worker_main,
                        args=(child, self._settings, lane.device, self._resident_factory),
                        name=f"artemis-oracle-gpu-{lane.device}", daemon=True)
                    try:
                        lane.process.start()
                    finally:
                        child.close()
                for index, lane in enumerate(self._lanes):
                    ready = lane.receive(self._startup_timeout)
                    _raise_worker_error(ready)
                    lane.device_metadata = dict(ready.get("device_metadata", {}))
                    assigned = [row for row in descriptors if self._assignments[row["case"]["case_id"]] == index]
                    _raise_worker_error(lane.rpc({"kind": "register", "cases": assigned}, self._startup_timeout))
                self._started = True
                return self
            except BaseException:
                self.close()
                raise
            finally:
                self._startup_wall_seconds = time.perf_counter() - started

    def operation_lock(self, case_id):
        try:
            return self._lanes[self._assignments[case_id]].lock
        except KeyError as error:
            raise PermissionError("case is outside this parallel Artemis scope") from error

    def session(self, case):
        self.operation_lock(case.case_id)
        # Retain the existing active-case and causal-projection boundary.
        issued, use_presented = self.source._active_issued(case, label="parallel public")
        with self._state_lock:
            if not self._started or self._closed:
                raise RuntimeError("parallel Artemis pool is not active")
            self._next_session_id += 1
            session_id = self._next_session_id
        lane = self._lanes[self._assignments[case.case_id]]
        reply = lane.rpc({"kind": "open", "session_id": session_id, "case_id": case.case_id,
                          "case": _case_descriptor(case if use_presented else issued.case)}, self._request_timeout)
        _raise_worker_error(reply)
        return _ParallelSession(self, lane, case.case_id, session_id, reply.get("usage", {}))

    @property
    def stats_snapshot(self):
        lanes = [dict(device=lane.device, pid=None if lane.process is None else lane.process.pid,
                      alive=lane.process is not None and lane.process.is_alive(), failed=lane.failed,
                      device_metadata=dict(lane.device_metadata),
                      request_count=lane.request_count, source_calls=lane.source_calls,
                      distribution_calls=lane.distribution_calls) for lane in self._lanes]
        return dict(started=self._started, closed=self._closed,
                    cleanup_complete=self._closed and not self._cleanup_failed and all(not lane["alive"] for lane in lanes),
                    cleanup_failed=self._cleanup_failed,
                    startup_wall_seconds=self._startup_wall_seconds, lanes=lanes,
                    case_assignments={key: self._lanes[index].device for key, index in self._assignments.items()})

    def stats(self):
        return self.stats_snapshot

    def close(self):
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        # Attempt every owned lane even when an earlier cleanup step fails.
        errors = []
        for lane in self._lanes:
            with lane.lock:
                if lane.connection is not None:
                    if not lane.failed:
                        try:
                            lane.connection.send({"kind": "shutdown"})
                        except (OSError, EOFError):
                            pass
                        except Exception as error:
                            errors.append(error)
                    try:
                        lane.connection.close()
                    except Exception as error:
                        errors.append(error)
                if lane.process is not None and lane.process.pid is not None:
                    try:
                        lane.process.join(timeout=5.0)
                    except Exception as error:
                        errors.append(error)
                    for stop in (lane.process.terminate, lane.process.kill):
                        try:
                            if lane.process.is_alive():
                                stop()
                                lane.process.join(timeout=5.0)
                        except Exception as error:
                            errors.append(error)
        for lane in self._lanes:
            try:
                if lane.process is not None and lane.process.is_alive():
                    errors.append(ArtemisOracleWorkerError("parallel Artemis worker remains alive after cleanup"))
            except Exception as error:
                errors.append(error)
        if errors:
            self._cleanup_failed = True
            raise ArtemisOracleWorkerError("parallel Artemis worker cleanup failed") from errors[0]


class _ParallelSession:
    def __init__(self, pool, lane, case_id, session_id, usage):
        self._pool, self._lane = pool, lane
        self.case_id, self._session_id = case_id, session_id
        self._usage, self._closed = dict(usage), False

    def _query(self, method, points):
        import numpy as np

        with self._lane.lock:
            if self._closed or self._pool._closed:
                raise RuntimeError("parallel Artemis public session is closed")
            reply = self._lane.rpc(dict(kind="query", session_id=self._session_id,
                                       method=method, points=np.asarray(points, dtype=np.float64)),
                                   self._pool._request_timeout)
            self._usage = dict(reply.get("usage", self._usage))
            _raise_worker_error(reply)
            return reply["result"]

    def query_source(self, points):
        return self._query("query_source", points)

    def query_initial_distribution(self, points):
        return self._query("query_initial_distribution", points)

    @property
    def usage_snapshot(self):
        with self._lane.lock:
            return dict(self._usage)

    def close(self):
        with self._lane.lock:
            if self._closed:
                return
            self._closed = True
            if not self._pool._closed and not self._lane.failed:
                reply = self._lane.rpc(dict(kind="close", session_id=self._session_id), self._pool._request_timeout)
                self._usage = dict(reply.get("usage", self._usage))
                _raise_worker_error(reply)
