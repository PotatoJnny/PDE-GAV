"""Private phase-scoped VPFP numerical replicas; no sampling or hidden truth.

By default one spawned process per allocated GPU owns the deterministic continuous
decoder and its assigned original latent/condition pairs. The explicit
``workers_per_gpu`` option adds isolated processes, never shared mutable caches.
Only source/rho0/g0
queries cross these private pipes. The Controller retains all truth, scoring,
Guidance, case/query order and promotion. Imports here do not initialize CUDA.
"""
from __future__ import annotations

from collections import OrderedDict
import importlib
import multiprocessing
import os
from threading import RLock
import time
from typing import Mapping

import numpy as np


class VPFPOracleWorkerError(RuntimeError):
    """A replica failed; never replay its request on a different GPU."""


def _validated_workers_per_gpu(value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("workers_per_gpu must be a positive integer")
    if value < 1:
        raise ValueError("workers_per_gpu must be a positive integer")
    return value


def _plain(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays are not VPFP replica data")
        return value.copy()
    if isinstance(value, np.generic):
        return value.copy()
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("VPFP replica mapping keys must be strings")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_plain(item) for item in value)
    if value is None or isinstance(value, (str, bytes, bool, int, float, complex)):
        return value
    raise TypeError(f"unsupported VPFP replica value: {type(value).__name__}")


def _case_descriptor(case):
    return dict(case_id=case.case_id, source=case.source.value,
                visibility=case.visibility.value, round_index=case.round_index,
                parent_case_id=case.parent_case_id, tags=tuple(case.tags),
                public_data=_plain(case.public_data))


def _case_from_descriptor(descriptor):
    from pde_agent_verifier_solver.sdk import Case, CaseSource, CaseVisibility
    row = dict(descriptor)
    row["source"] = CaseSource(row["source"])
    row["visibility"] = CaseVisibility(row["visibility"])
    return Case(**row)


def _host_array(value):
    value = value.detach() if hasattr(value, "detach") else value
    value = value.cpu() if hasattr(value, "cpu") else value
    value = value.numpy() if hasattr(value, "numpy") else value
    return np.asarray(value).copy()


def _export_case(runtime, case):
    resident = runtime._active_resident(case)
    # _ResidentCase does not contain a truth object. Full carrier coefficients
    # (not seeds or a resampled surrogate) and condition are the only private
    # numerical data exported. No manufactured source grid is constructed.
    return dict(case=_case_descriptor(resident.public_snapshot),
                latent=_host_array(resident.latent),
                equation_condition=_host_array(resident.equation_condition))


class _ResidentVPFP:
    def __init__(self, settings, device):
        from .perturbation_backend import VPFPLegacyStructuredRuntime, _legacy_api
        from .throughput_oracle import ThroughputResidentRuntime, oracle_execution_options
        options = oracle_execution_options(**settings.get("oracle_execution_options", {}))
        api = _legacy_api()
        if options["oracle_variant"] == "ordered_throughput_v1":
            # Replica-local CPU policy matches the measured multi-process arm;
            # the Controller and its Guidance trainer retain their own settings.
            api.torch.set_num_threads(1)
        hermite = importlib.import_module("agent_verifier.vpfp.continuous_hermite")
        config = hermite.VPFPContinuousHermiteConfig.from_dict(settings["decoder_config"])
        # This is exactly the original expanded runtime's model construction:
        # all decoder NN parameters are unused/zero, the full latent is primary.
        model = api.build_deterministic_direct_model(config, device=device)
        api.analytic_source.require_analytic_structured_source(model)
        runtime_type = (ThroughputResidentRuntime if options["oracle_variant"] == "ordered_throughput_v1"
                        else VPFPLegacyStructuredRuntime)
        runtime = object.__new__(runtime_type)
        runtime.api, runtime.expanded_model = api, model
        runtime.oracle_kernel = "structured_reuse_v1"
        runtime.oracle_execution_options = options
        runtime.materialize_cases = False
        runtime._runtime_lock = RLock()
        runtime._parallel_oracle_pool = None
        runtime._resident_cases = {}
        runtime._prepared_oracles = OrderedDict()
        runtime._prepared_oracle_stats = {"hits": 0, "misses": 0, "evictions": 0}
        self.runtime = runtime
        torch = api.torch
        self.device_metadata = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                                "device": str(device), "cpu_threads": torch.get_num_threads(),
                                "torch_num_threads": torch.get_num_threads(),
                                **options}
        if str(device).startswith("cuda"):
            current = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(current)
            self.device_metadata.update(device_count=torch.cuda.device_count(), current_device=current,
                                        name=properties.name, uuid=str(getattr(properties, "uuid", "")))

    def register(self, descriptor):
        from pde_agent_verifier_solver.sdk import CaseVisibility
        from .perturbation_backend import _ResidentCase, _snapshot_vpfp_public_case
        if set(descriptor) != {"case", "latent", "equation_condition"}:
            raise ValueError("unexpected VPFP private replica descriptor")
        case = _case_from_descriptor(descriptor["case"])
        if case.visibility is not CaseVisibility.HIDDEN:
            raise PermissionError("VPFP replica only accepts hidden cases")
        if case.case_id in self.runtime._resident_cases:
            raise ValueError("VPFP replica case registered twice")
        latent, condition = descriptor["latent"], descriptor["equation_condition"]
        if np.size(latent) != self.runtime.expanded_model.config.latent_dim or np.size(condition) != 2:
            raise ValueError("VPFP replica latent/condition shape mismatch")
        # Pipe decoding already gives private CPU ownership. Qualification is
        # the parent factory's responsibility, not repeated online validation.
        self.runtime._resident_cases[case.case_id] = _ResidentCase(
            case=case, public_snapshot=_snapshot_vpfp_public_case(case),
            latent=latent, equation_condition=condition)

    def open(self, case_id):
        resident = self.runtime._resident_for_id(case_id)
        return self.runtime.public_session_factory(resident.case)

    def stats(self):
        return dict(self.runtime.prepared_oracle_cache_info(), **self.runtime.oracle_execution_options)

    def release(self, case_ids):
        self.runtime.release_cases(case_ids)

    def close(self):
        self.runtime._prepared_oracles.clear()
        self.runtime._resident_cases.clear()


def _worker_main(connection, settings, physical_device, factory):
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_device
    if settings.get("oracle_execution_options", {}).get("oracle_variant") == "ordered_throughput_v1":
        # Set these before loading the numerical replica. Spawn may already
        # have imported NumPy; the explicit Torch intra-op setting above also
        # handles libraries initialized during child-module imports.
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "1"
    resident, sessions, session_cases = None, {}, {}
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
                    resident.register(request["case"])
                    result = None
                elif kind == "release":
                    identifiers = tuple(request["case_ids"])
                    if set(identifiers) & set(session_cases.values()):
                        raise RuntimeError("cannot release a replica Case with an open session")
                    resident.release(identifiers)
                    result = None
                elif kind == "open":
                    identifier = request["session_id"]
                    if identifier in sessions:
                        raise ValueError("VPFP replica session opened twice")
                    session = sessions[identifier] = resident.open(request["case_id"])
                    session_cases[identifier] = request["case_id"]
                    result = None
                elif kind == "close":
                    session = sessions.pop(request["session_id"])
                    session.close()
                    session_cases.pop(request["session_id"])
                    result = None
                elif kind == "query":
                    if request["method"] not in ("source", "rho0", "g0"):
                        raise PermissionError("undeclared VPFP public replica method")
                    session = sessions[request["session_id"]]
                    result = getattr(session, request["method"])(request["points"])
                elif kind == "stats":
                    result = resident.stats()
                else:
                    raise ValueError("unknown VPFP replica operation")
                usage = {} if session is None else _plain(session.usage_snapshot)
                connection.send({"ok": True, "result": result, "usage": usage})
            except Exception as error:
                usage = {} if session is None else _plain(session.usage_snapshot)
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
        try:
            if resident is not None:
                resident.close()
        finally:
            connection.close()


def _raise_worker_error(reply):
    if not reply.get("ok"):
        error = {"ValueError": ValueError, "TypeError": TypeError,
                 "PermissionError": PermissionError}.get(reply.get("error_type"), VPFPOracleWorkerError)
        raise error(f"VPFP replica {reply.get('error_type', 'error')}: {reply.get('error', '')}")


class _Lane:
    def __init__(self, device, *, lane_index, worker_index):
        self.device, self.lock = device, RLock()
        self.lane_index, self.worker_index = lane_index, worker_index
        self.process = self.connection = None
        self.failed = False
        self.device_metadata = {}
        self.requests = self.registered = 0
        self.by_method = {"source": 0, "rho0": 0, "g0": 0}

    def receive(self, timeout):
        try:
            if not self.connection.poll(timeout):
                self.failed = True
                raise VPFPOracleWorkerError(f"VPFP GPU {self.device} response timed out")
            return self.connection.recv()
        except (EOFError, OSError) as error:
            self.failed = True
            raise VPFPOracleWorkerError(f"VPFP GPU {self.device} disconnected") from error

    def rpc(self, request, timeout):
        with self.lock:
            if self.failed or self.connection is None or not self.process.is_alive():
                raise VPFPOracleWorkerError(f"VPFP GPU {self.device} is unavailable")
            try:
                self.connection.send(request)
            except (OSError, EOFError) as error:
                self.failed = True
                raise VPFPOracleWorkerError(f"VPFP GPU {self.device} send failed") from error
            if request["kind"] == "query":
                self.requests += 1
                if request["method"] in self.by_method:
                    self.by_method[request["method"]] += 1
            return self.receive(timeout)


class _StructuredRuntimeAdapter:
    """Default strong carrier; weak plugins may explicitly supply their adapter."""

    @staticmethod
    def validate(runtime):
        if getattr(runtime, "oracle_kernel", None) != "structured_reuse_v1":
            raise ValueError("VPFP GPU replicas require the fixed structured_reuse_v1 kernel")

    @staticmethod
    def settings(runtime):
        return {"decoder_config": _plain(runtime.expanded_model.config.to_dict())}

    @staticmethod
    def active(runtime, case):
        return runtime._active_resident(case)

    @staticmethod
    def descriptor_bytes(descriptor):
        return np.asarray(descriptor["latent"]).nbytes

    @staticmethod
    def metadata(workers_per_gpu):
        return dict(oracle_kernel="structured_reuse_v1", maximum_prepared_cases_per_process=16,
                    maximum_prepared_cases_per_gpu=16 * workers_per_gpu)


class VPFPParallelOraclePool:
    def __init__(self, runtime, *, cases, devices, workers_per_gpu=1, startup_timeout_seconds=180.,
                 request_timeout_seconds=900., _resident_factory=_ResidentVPFP,
                 _exporter=_export_case, _settings=None, _runtime_adapter=None,
                 _development_cases=False, oracle_variant="structured_reuse_v1",
                 source_request_block=8192, macro_point_block=512):
        from pde_agent_verifier_solver.sdk import CaseVisibility
        from .throughput_oracle import oracle_execution_options
        self.oracle_execution_options = oracle_execution_options(oracle_variant=oracle_variant,
            source_request_block=source_request_block, macro_point_block=macro_point_block)
        self.runtime, self.cases = runtime, tuple(cases)
        devices = tuple(devices)
        if not devices or len(set(devices)) != len(devices) or any(
                not isinstance(item, str) or not item or "," in item for item in devices):
            raise ValueError("VPFP replicas require distinct allocated GPU tokens")
        self.devices = devices
        self.workers_per_gpu = _validated_workers_per_gpu(workers_per_gpu)
        if not self.cases or len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("VPFP replicas require distinct issued cases")
        if type(_development_cases) is not bool:
            raise TypeError("development replica mode must be an explicit boolean")
        self._development_cases = _development_cases
        visibility = CaseVisibility.DEVELOPMENT if _development_cases else CaseVisibility.HIDDEN
        if any(case.visibility is not visibility for case in self.cases):
            raise PermissionError("VPFP GPU replicas are hidden-evaluation only")
        self._adapter = _StructuredRuntimeAdapter if _runtime_adapter is None else _runtime_adapter
        self._adapter.validate(runtime)
        if startup_timeout_seconds <= 0 or request_timeout_seconds <= 0:
            raise ValueError("VPFP replica timeouts must be positive")
        self._settings = (_plain(_settings) if _settings is not None else
                          self._adapter.settings(runtime))
        if _runtime_adapter is None:
            # Weak carriers own a different numerical execution-options schema.
            # Preserve their explicit descriptor instead of injecting strong options.
            self._settings["oracle_execution_options"] = dict(self.oracle_execution_options)
        self._factory, self._exporter = _resident_factory, _exporter
        # Keep physical devices interleaved so short case batches still use all
        # allocated GPUs. Each case has exactly one process/cache owner.
        self._lanes = tuple(
            _Lane(device, lane_index=worker * len(devices) + index, worker_index=worker)
            for worker in range(self.workers_per_gpu) for index, device in enumerate(devices)
        )
        self._assignments = {case.case_id: index % len(self._lanes) for index, case in enumerate(self.cases)}
        self._startup_timeout, self._request_timeout = float(startup_timeout_seconds), float(request_timeout_seconds)
        self._state_lock = RLock()
        self._registration_lock = RLock()
        self._started = self._closed = self._cleanup_failed = False
        self._next_session_id = 0
        self._startup_wall_seconds = 0.
        self._transferred_latent_bytes = 0

    def start(self):
        with self._state_lock:
            if self._closed:
                raise RuntimeError("VPFP replica pool is closed")
            if self._started:
                return self
            started = time.perf_counter()
            context = multiprocessing.get_context("spawn")
            try:
                for lane in self._lanes:
                    parent, child = context.Pipe(duplex=True)
                    lane.connection = parent
                    lane.process = context.Process(target=_worker_main,
                        args=(child, self._settings, lane.device, self._factory),
                        name=(f"vpfp-oracle-gpu-{lane.device}" +
                              (f"-worker-{lane.worker_index}" if self.workers_per_gpu > 1 else "")), daemon=True)
                    try:
                        lane.process.start()
                    finally:
                        child.close()
                for lane in self._lanes:
                    reply = lane.receive(self._startup_timeout)
                    _raise_worker_error(reply)
                    lane.device_metadata = dict(reply.get("device_metadata", {}))
                # Do not create a second whole-cohort copy of 8.5 MB carriers.
                # Export/transfer one original resident at a time to one lane.
                for case in self.cases:
                    descriptor = self._exporter(self.runtime, case)
                    lane = self._lanes[self._assignments[case.case_id]]
                    _raise_worker_error(lane.rpc({"kind": "register", "case": descriptor}, self._startup_timeout))
                    lane.registered += 1
                    self._transferred_latent_bytes += self._adapter.descriptor_bytes(descriptor)
                self._started = True
                return self
            except BaseException as error:
                try:
                    self.close()
                except BaseException as cleanup_error:
                    error.add_note(f"VPFP startup cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}")
                raise
            finally:
                self._startup_wall_seconds = time.perf_counter() - started

    def operation_lock(self, case_id):
        try:
            return self._lanes[self._assignments[case_id]].lock
        except KeyError as error:
            raise PermissionError("case is outside this VPFP replica scope") from error

    def register_cases(self, cases):
        """Controller-only dynamic registration, explicitly opted in by a plugin.

        The default hidden cohort remains immutable. Numerical callbacks never
        call this method: development provisioning precedes Candidate startup.
        """
        with self._registration_lock:
            return self._register_cases(cases)

    def _register_cases(self, cases):
        from pde_agent_verifier_solver.sdk import CaseVisibility
        if not self._development_cases:
            raise PermissionError("hidden replica cohorts cannot be changed")
        cases = tuple(cases)
        if len({case.case_id for case in cases}) != len(cases):
            raise ValueError("development registration repeats a Case")
        if any(case.visibility is not CaseVisibility.DEVELOPMENT for case in cases):
            raise PermissionError("development replicas accept only development Cases")
        for case in cases:
            self._adapter.active(self.runtime, case)
        for case in cases:
            with self._state_lock:
                if not self._started or self._closed:
                    raise RuntimeError("VPFP replica pool is not active")
                if case.case_id in self._assignments:
                    continue
                lane = min(self._lanes, key=lambda item: (item.registered, item.lane_index))
            descriptor = self._exporter(self.runtime, case)
            _raise_worker_error(lane.rpc({"kind": "register", "case": descriptor}, self._startup_timeout))
            with self._state_lock:
                self._assignments[case.case_id] = lane.lane_index
                self.cases += (case,)
                lane.registered += 1
                self._transferred_latent_bytes += self._adapter.descriptor_bytes(descriptor)

    def release_cases(self, case_ids):
        with self._registration_lock:
            return self._release_cases(case_ids)

    def _release_cases(self, case_ids):
        if not self._development_cases:
            raise PermissionError("hidden replica cohorts cannot be changed")
        identifiers = tuple(case_ids)
        for identifier in identifiers:
            with self._state_lock:
                index = self._assignments.get(identifier)
            if index is None:
                continue
            lane = self._lanes[index]
            _raise_worker_error(lane.rpc({"kind": "release", "case_ids": (identifier,)}, self._request_timeout))
            with self._state_lock:
                self._assignments.pop(identifier)
                self.cases = tuple(case for case in self.cases if case.case_id != identifier)
                lane.registered -= 1

    def session(self, case):
        self.operation_lock(case.case_id)
        self._adapter.active(self.runtime, case)
        with self._state_lock:
            if not self._started or self._closed:
                raise RuntimeError("VPFP replica pool is not active")
            self._next_session_id += 1
            identifier = self._next_session_id
        lane = self._lanes[self._assignments[case.case_id]]
        reply = lane.rpc({"kind": "open", "session_id": identifier, "case_id": case.case_id}, self._request_timeout)
        _raise_worker_error(reply)
        return _ParallelSession(self, lane, case.case_id, identifier, reply.get("usage", {}))

    def stats(self):
        lanes = [dict(device=lane.device, lane_index=lane.lane_index, worker_index=lane.worker_index,
                      pid=None if lane.process is None else lane.process.pid,
                      alive=lane.process is not None and lane.process.is_alive(), failed=lane.failed,
                      device_metadata=dict(lane.device_metadata), registered_cases=lane.registered,
                      request_count=lane.requests, by_method=dict(lane.by_method)) for lane in self._lanes]
        return dict(started=self._started, closed=self._closed, lanes=lanes,
                    worker_layout="fixed_case_owner_v1", gpu_device_count=len(self.devices),
                    workers_per_gpu=self.workers_per_gpu, process_count=len(self._lanes),
                    cleanup_complete=self._closed and not self._cleanup_failed and all(not lane["alive"] for lane in lanes),
                    cleanup_failed=self._cleanup_failed, startup_wall_seconds=self._startup_wall_seconds,
                    transferred_latent_bytes=self._transferred_latent_bytes,
                    **(self.oracle_execution_options if self._adapter is _StructuredRuntimeAdapter else {}),
                    **self._adapter.metadata(self.workers_per_gpu),
                    case_assignments={key: self._lanes[index].device for key, index in self._assignments.items()},
                    case_lane_assignments=dict(self._assignments))

    def close(self):
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        errors = []
        for lane in self._lanes:
            acquired = lane.lock.acquire(timeout=5.)
            try:
                if lane.connection is not None:
                    if acquired and not lane.failed:
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
                        lane.process.join(timeout=5.)
                    except Exception as error:
                        errors.append(error)
                    for stop in (lane.process.terminate, lane.process.kill):
                        try:
                            if lane.process.is_alive():
                                stop()
                                lane.process.join(timeout=5.)
                        except Exception as error:
                            errors.append(error)
            finally:
                if acquired:
                    lane.lock.release()
        for lane in self._lanes:
            try:
                if lane.process is not None and lane.process.is_alive():
                    errors.append(VPFPOracleWorkerError("VPFP replica remains alive after cleanup"))
            except Exception as error:
                errors.append(error)
        if errors:
            self._cleanup_failed = True
            raise VPFPOracleWorkerError("VPFP replica cleanup failed") from errors[0]


class _ParallelSession:
    def __init__(self, pool, lane, case_id, session_id, usage):
        self._pool, self._lane = pool, lane
        self.case_id, self._session_id = case_id, session_id
        self._usage, self._closed = dict(usage), False

    def _query(self, method, points):
        with self._lane.lock:
            if self._closed or self._pool._closed:
                raise RuntimeError("VPFP public replica session is closed")
            # Match the ordinary runtime's per-query resident-id check. This
            # cheap Controller lookup preserves release/revocation semantics;
            # it does not inspect numerical arrays or recompute validation.
            self._pool.runtime._resident_for_id(self.case_id)
            reply = self._lane.rpc({"kind": "query", "session_id": self._session_id,
                                   "method": method, "points": np.asarray(points, dtype=np.float64)},
                                  self._pool._request_timeout)
            self._usage = dict(reply.get("usage", self._usage))
            _raise_worker_error(reply)
            return reply["result"]

    def source(self, points):
        return self._query("source", points)

    def rho0(self, points):
        return self._query("rho0", points)

    def g0(self, points):
        return self._query("g0", points)

    @property
    def usage_snapshot(self):
        with self._lane.lock:
            return _plain(self._usage)

    def close(self):
        with self._lane.lock:
            if self._closed:
                return
            self._closed = True
            if not self._pool._closed and not self._lane.failed:
                reply = self._lane.rpc({"kind": "close", "session_id": self._session_id}, self._pool._request_timeout)
                self._usage = dict(reply.get("usage", self._usage))
                _raise_worker_error(reply)
