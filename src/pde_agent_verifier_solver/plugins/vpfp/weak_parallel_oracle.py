"""Phase-scoped weak moving-frame replicas of the frozen numerical decoder.

Only original m/z, nu/T and public snapshots cross the Controller-owned pipes.
No sampler, Guidance network, truth object, source grid or solver is exported.
NumPy precedes all lazy Torch imports, preserving frozen initialization order.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import os
from threading import RLock

import numpy as np

from pde_agent_verifier_solver.engine.evaluation_resources import HiddenEvaluationResourceScope
from .parallel_oracle import (VPFPParallelOraclePool, _case_descriptor,
    _case_from_descriptor, _host_array, _plain, _validated_workers_per_gpu)


SCHEMA = "vpfp_weak_moving_frame_resident_replica_v1"


def _descriptor_settings(runtime):
    """Copy only the deterministic decoder, never the learned generator."""
    from .weak_portable_base import _state_sha
    state = runtime.decoder.state_dict()
    result = dict(schema=SCHEMA, decoder_config=runtime.decoder.base_decoder.config.to_dict(),
        macro_config=runtime.decoder.macro_config.to_dict(),
        value_config=runtime.decoder.value_config.to_dict(),
        decoder_identity=runtime.decoder.identity(),
        decoder_state={key: _host_array(value) for key, value in state.items()},
        decoder_state_sha256=_state_sha(state), source_chunk=runtime.source_chunk,
        field_chunk=runtime.field_chunk, source_implementation=_plain(runtime.source_implementation))
    options = getattr(runtime, "oracle_execution_options", None)
    if options is not None:
        result["oracle_execution_options"] = _plain(options)
    return result


def _export_weak_case(runtime, case):
    resident = runtime._active(case)
    return dict(case=_case_descriptor(resident.public_snapshot),
        carrier={key: _host_array(value) for key, value in resident.carrier.items()},
        condition=_host_array(resident.condition), answer_sha256=resident.answer_sha256)


class _WeakRuntimeAdapter:
    @staticmethod
    def validate(runtime):
        from .moving_frame import MovingFrameDecoder
        from .moving_frame_backend import MovingFrameBackend
        if (not isinstance(runtime, MovingFrameBackend) or type(runtime.decoder) is not MovingFrameDecoder
                or not runtime.adaptive_source_only or runtime.batched_source
                or (runtime.source_implementation or {}).get("name") != "adaptive_exact_v1"):
            raise ValueError("weak replicas require the unchanged adaptive-only moving-frame runtime")
        if runtime.decoder.training or any(parameter.requires_grad for parameter in runtime.decoder.parameters()):
            raise ValueError("weak replica decoder must be frozen and eval")

    settings = staticmethod(_descriptor_settings)

    @staticmethod
    def active(runtime, case):
        return runtime._active(case)

    @staticmethod
    def descriptor_bytes(descriptor):
        return sum(np.asarray(value).nbytes for value in descriptor["carrier"].values())

    @staticmethod
    def metadata(workers_per_gpu):
        return dict(oracle_kernel="adaptive_exact_v1", decoder_schema=SCHEMA,
                    carrier="original_complete_m_z", generated_networks_loaded=False)


class _ResidentWeakVPFP:
    """Query-only instance of the original MovingFrameBackend, without files."""

    def __init__(self, settings, device):
        import torch
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from .moving_frame import MovingFrameDecoder, MovingFrameMacroConfig
        from .moving_frame_backend import MovingFrameBackend
        from .perturbation_backend import _legacy_api
        from .value_control_diffusion import VPFPValueEDMConfig
        from .weak_portable_base import _load_state, _state_sha
        expected = {"schema", "decoder_config", "macro_config", "value_config", "decoder_identity",
            "decoder_state", "decoder_state_sha256", "source_chunk", "field_chunk", "source_implementation"}
        from .moving_frame_oracle_options import validated_source_execution_options
        options = validated_source_execution_options(settings.get("oracle_execution_options"))
        if "oracle_execution_options" in settings:
            expected.add("oracle_execution_options")
        if set(settings) != expected or settings["schema"] != SCHEMA:
            raise ValueError("unexpected weak replica settings")
        if any(type(settings[key]) is not int or settings[key] < 1 for key in ("source_chunk", "field_chunk")):
            raise ValueError("weak replica chunk sizes must remain positive integers")
        implementation = settings["source_implementation"]
        if (implementation.get("name") != "adaptive_exact_v1"
                or implementation.get("adaptive_source_only") is not True
                or implementation.get("macro_evaluator") != "original"):
            raise ValueError("weak replica must preserve the original numerical implementation")
        if implementation.get("oracle_execution_options") != options:
            raise ValueError("weak replica Oracle options differ from the source identity")
        # Validate the transmitted bytes before constructing/loading anything.
        state = {key: torch.from_numpy(np.asarray(value).copy()) for key, value in settings["decoder_state"].items()}
        if _state_sha(state) != settings["decoder_state_sha256"]:
            raise ValueError("weak replica decoder state fingerprint differs")
        api = _legacy_api()
        hermite = importlib.import_module("agent_verifier.vpfp.continuous_hermite")
        config = hermite.VPFPContinuousHermiteConfig.from_dict(settings["decoder_config"])
        model = api.build_deterministic_direct_model(config, device=device).eval().requires_grad_(False)
        decoder = MovingFrameDecoder(model, MovingFrameMacroConfig.from_dict(settings["macro_config"]),
            value_config=VPFPValueEDMConfig.from_dict(settings["value_config"])).eval().requires_grad_(False)
        if decoder.identity() != settings["decoder_identity"]:
            raise ValueError("weak replica decoder identity differs")
        _load_state(decoder, {"state": state, "state_sha256": settings["decoder_state_sha256"]}, deterministic=True)
        runtime = object.__new__(MovingFrameBackend)
        runtime.decoder = decoder
        runtime.batched_source, runtime.adaptive_source_only = False, True
        runtime.source_chunk, runtime.field_chunk = settings["source_chunk"], settings["field_chunk"]
        runtime.oracle_execution_options = options
        runtime.source_implementation = _plain(implementation)
        runtime._runtime_lock = RLock()
        runtime._resident_cases, runtime._source_evaluators = {}, {}
        runtime.oracle_calls = {key: 0 for key in ("source", "rho0", "g0", "macros")}
        self.runtime = runtime
        self.device_metadata = dict(cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            device=str(device), cpu_threads=torch.get_num_threads(), decoder_state_sha256=settings["decoder_state_sha256"],
            generated_networks_loaded=False)
        if str(device).startswith("cuda"):
            current = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(current)
            self.device_metadata.update(device_count=torch.cuda.device_count(), current_device=current,
                name=properties.name, uuid=str(getattr(properties, "uuid", "")))

    def register(self, descriptor):
        import torch
        from pde_agent_verifier_solver.sdk import CaseVisibility
        from .moving_frame_backend import _Resident
        from .perturbation_backend import _snapshot_vpfp_public_case
        if set(descriptor) != {"case", "carrier", "condition", "answer_sha256"}:
            raise ValueError("unexpected weak replica carrier descriptor")
        case = _case_from_descriptor(descriptor["case"])
        visibility = (CaseVisibility.DEVELOPMENT if getattr(self, "_development_cases", False)
                      else CaseVisibility.HIDDEN)
        if case.visibility is not visibility:
            raise PermissionError("weak replicas only accept hidden cases")
        if case.case_id in self.runtime._resident_cases:
            raise ValueError("weak replica case registered twice")
        if set(descriptor["carrier"]) != {"m", "z"}:
            raise ValueError("weak replicas require the complete m/z carrier")
        carrier = {key: torch.from_numpy(np.asarray(value).copy()) for key, value in descriptor["carrier"].items()}
        if any(value.ndim != 2 or len(value) != 1 or not value.is_floating_point()
                or not bool(torch.isfinite(value).all()) for value in carrier.values()):
            raise ValueError("one finite complete weak carrier per case required")
        self.runtime.decoder.carrier_to_payload(carrier)
        condition = torch.from_numpy(np.asarray(descriptor["condition"]).copy())
        if condition.shape != (2,) or not condition.is_floating_point() or not bool(torch.isfinite(condition).all()):
            raise ValueError("weak replica requires one finite nu/T pair")
        self.runtime.decoder.base_decoder._equation_condition_rows(condition, count=1)
        digest = descriptor["answer_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("weak replica requires the original answer SHA256")
        self.runtime._resident_cases[case.case_id] = _Resident(case,
            _snapshot_vpfp_public_case(case), carrier, condition, digest)

    def open(self, case_id):
        return self.runtime.public_session_factory(self.runtime._resident_for_id(case_id).case)

    def stats(self):
        return dict(schema=SCHEMA, resident_case_count=len(self.runtime._resident_cases),
            prepared_case_count=len(self.runtime._source_evaluators), oracle_calls=dict(self.runtime.oracle_calls),
            source_execution=self.runtime.source_execution_stats(),
            source_implementation=_plain(self.runtime.source_implementation), generated_networks_loaded=False)

    def close(self):
        self.runtime._source_evaluators.clear()
        self.runtime._resident_cases.clear()


class _DevelopmentResidentWeakVPFP(_ResidentWeakVPFP):
    """Same decoder, but only Controller-issued development Cases are admitted."""
    _development_cases = True

    def release(self, case_ids):
        self.runtime.release_cases(case_ids)


class WeakVPFPParallelOraclePool(VPFPParallelOraclePool):
    def __init__(self, runtime, *, cases, devices, workers_per_gpu=2,
                 startup_timeout_seconds=180., request_timeout_seconds=900.,
                 _resident_factory=_ResidentWeakVPFP):
        super().__init__(runtime, cases=cases, devices=devices, workers_per_gpu=workers_per_gpu,
            startup_timeout_seconds=startup_timeout_seconds, request_timeout_seconds=request_timeout_seconds,
            _resident_factory=_resident_factory, _exporter=_export_weak_case, _runtime_adapter=_WeakRuntimeAdapter)


class WeakVPFPDevelopmentPool(VPFPParallelOraclePool):
    def __init__(self, runtime, *, cases, devices, workers_per_gpu=2,
                 startup_timeout_seconds=180., request_timeout_seconds=900.,
                 _resident_factory=_DevelopmentResidentWeakVPFP):
        devices = tuple(devices)
        if len(devices) != 1:
            raise ValueError("development replicas use only the Controller GPU")
        super().__init__(runtime, cases=cases, devices=devices, workers_per_gpu=workers_per_gpu,
            startup_timeout_seconds=startup_timeout_seconds, request_timeout_seconds=request_timeout_seconds,
            _resident_factory=_resident_factory, _exporter=_export_weak_case,
            _runtime_adapter=_WeakRuntimeAdapter, _development_cases=True)


class WeakVPFPDevelopmentOracle:
    """Round-reusable development replicas, provisioned before Candidate timing.

    This object is Controller-private. It exports only the complete frozen
    carrier of already issued development Cases, never a sampler or solution
    access method. Its numerical lanes have their own processes and locks.
    """

    def __init__(self, runtime, *, controller_gpu, workers_per_gpu=2, events=None,
                 pool_factory=WeakVPFPDevelopmentPool):
        _WeakRuntimeAdapter.validate(runtime)
        if not isinstance(controller_gpu, str) or not controller_gpu or "," in controller_gpu:
            raise ValueError("one allocated Controller GPU token is required")
        self.runtime, self.controller_gpu = runtime, controller_gpu
        self.workers_per_gpu = _validated_workers_per_gpu(workers_per_gpu)
        self.events, self.pool_factory = events, pool_factory
        self._lock, self._pool = RLock(), None
        self._active_batches, self._closed = 0, False
        self.reports = []

    def _emit(self, event_type, message, payload):
        if self.events is not None:
            self.events.emit(event_type, message, payload=payload)

    def ensure_cases(self, cases):
        from pde_agent_verifier_solver.sdk import CaseVisibility
        cases = tuple(cases)
        if not cases:
            return
        if any(case.visibility is not CaseVisibility.DEVELOPMENT for case in cases):
            raise PermissionError("development Oracle cannot serve hidden Cases")
        for case in cases:
            self.runtime._active(case)
        with self._lock:
            if self._closed:
                raise RuntimeError("development Oracle has been shut down")
            if self._pool is None:
                pool = self.pool_factory(self.runtime, cases=cases,
                    devices=(self.controller_gpu,), workers_per_gpu=self.workers_per_gpu)
                try:
                    pool.start()
                    self._emit("development_oracle_ready", "Development numerical workers are ready",
                        {"gpu_devices": [self.controller_gpu], "oracle_process_count": self.workers_per_gpu,
                         "provisioning": "before_candidate_startup", "pool": pool.stats()})
                except BaseException:
                    pool.close()
                    raise
                self._pool = pool
            else:
                self._pool.register_cases(cases)

    def operation_lock(self, case_id):
        # Evaluator constructs persistent sessions before execute_cases starts
        # Candidate processes. Provisioning must stay on that setup boundary.
        self.ensure_cases((self.runtime._resident_for_id(case_id).case,))
        with self._lock:
            return self._pool.operation_lock(case_id)

    def session(self, case):
        self.runtime._active(case)
        # The Evaluator already holds this Case's lane lock when it lazily
        # opens a session. Never take the registration/manager lock here:
        # another Controller thread may hold it while registering a later
        # Case on this lane. Phase boundaries only detach a settled pool.
        pool = self._pool
        if pool is None or case.case_id not in pool._assignments:
            # Direct Agent public-Oracle probes do not construct persistent
            # Candidate sessions. They may provision here: an unregistered
            # Case cannot already own a pool lane lock. The Candidate path
            # always provisions in operation_lock, before its charged clock.
            self.ensure_cases((case,))
            pool = self._pool
        return pool.session(case)

    @contextmanager
    def batch(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("development Oracle has been shut down")
            self._active_batches += 1
        try:
            yield
        finally:
            with self._lock:
                self._active_batches -= 1

    def release_cases(self, case_ids):
        with self._lock:
            if self._pool is not None:
                self._pool.release_cases(tuple(case_ids))

    def suspend(self):
        """Called only at settled formal-phase boundaries; later rounds restart."""
        with self._lock:
            if self._active_batches:
                raise RuntimeError("cannot suspend active development Candidate evaluations")
            self._close_pool()

    def _close_pool(self):
        pool = self._pool
        if pool is None:
            return
        try:
            pool.close()
        finally:
            report = pool.stats()
            self.reports.append(report)
            self._emit("development_oracle_released", "Development numerical workers released", report)
        if report.get("cleanup_complete") is not True:
            raise RuntimeError("development Oracle cleanup was not confirmed")
        self._pool = None

    def shutdown(self):
        with self._lock:
            self._closed = True
            self._close_pool()


def install_development_oracle(binding, candidate_execution, *, controller_gpu,
                               workers_per_gpu=2, events=None, pool_factory=WeakVPFPDevelopmentPool):
    """Attach only to this new campaign's service instance, not engine globals."""
    from pde_agent_verifier_solver.sdk import EvaluationPhase
    backend = binding.plugin.generator.backend
    if getattr(backend, "_development_oracle_pool", None) is not None:
        raise RuntimeError("development Oracle is already installed")
    manager = WeakVPFPDevelopmentOracle(backend.backend, controller_gpu=controller_gpu,
        workers_per_gpu=workers_per_gpu, events=events, pool_factory=pool_factory)
    original = candidate_execution.execute_cases

    def execute_cases(artifact, cases, plans, evaluation, **kwargs):
        cases, plans = tuple(cases), tuple(plans)
        if plans and all(plan.phase is EvaluationPhase.DEVELOPMENT for plan in plans):
            # These Cases carry opaque candidate-facing IDs. The Evaluator
            # already provisioned the real Controller Cases while constructing
            # persistent sessions; never export a candidate-facing snapshot.
            with manager.batch():
                return original(artifact, cases, plans, evaluation, **kwargs)
        return original(artifact, cases, plans, evaluation, **kwargs)

    backend._development_oracle_pool = manager
    candidate_execution.execute_cases = execute_cases
    # The shared service fixes its global fair-admission capacity at construction.
    # Installing this PDE pool must not change just the visible attribute while
    # leaving that capacity stale. The composition supplies the global limit.
    return manager


class WeakVPFPHiddenEvaluationScope(HiddenEvaluationResourceScope):
    """Temporary routing for Official/Terminal only; no allocation or sampling."""

    def __init__(self, binding, candidate_execution, *, devices, workers_per_gpu=2, events=None, pool_factory=None):
        self.backend = binding.plugin.generator.backend
        self.runtime = self.backend.backend
        _WeakRuntimeAdapter.validate(self.runtime)
        if not callable(getattr(self.backend, "public_session_operation_lock", None)):
            raise TypeError("weak evaluation requires the explicit lane-lock backend hook")
        # Keep this branch's existing Candidate wire codec. Strong-run transport
        # tuning is not implemented here; assigning unused service attributes
        # would not configure it and must not appear as a throughput claim.
        self.workers_per_gpu = _validated_workers_per_gpu(workers_per_gpu)
        factory = WeakVPFPParallelOraclePool if pool_factory is None else pool_factory

        def create_pool(*, cases, devices):
            return factory(self.runtime, cases=cases, devices=devices, workers_per_gpu=self.workers_per_gpu)

        def activate(pool):
            if self.backend._parallel_oracle_pool is not None:
                raise RuntimeError("weak numerical pool is already installed")
            self.backend._parallel_oracle_pool = pool

        super().__init__(candidate_execution, devices=devices, pool_factory=create_pool,
            activate=activate, deactivate=lambda: setattr(self.backend, "_parallel_oracle_pool", None), events=events)

    @contextmanager
    def __call__(self, *, cases, unavailable_gpu_devices=()):
        development = getattr(self.backend, "_development_oracle_pool", None)
        if development is not None:
            development.suspend()
        with super().__call__(cases=cases, unavailable_gpu_devices=unavailable_gpu_devices) as pool:
            if pool is not None:
                self.execution.oracle_host_count *= self.workers_per_gpu
            yield pool

    def _emit(self, event_type, message, payload):
        payload = dict(payload)
        if "gpu_devices" in payload:
            count = len(payload["gpu_devices"])
            payload.update(gpu_device_count=count, workers_per_gpu=self.workers_per_gpu,
                oracle_process_count=count * self.workers_per_gpu, oracle_kernel="adaptive_exact_v1",
                candidate_transport="unchanged_existing_wire_codec")
            if "oracle_workers" in payload:
                payload["oracle_workers"] = count * self.workers_per_gpu
        super()._emit(event_type, message, payload)
