"""Thin unified binding to the existing physical VPFP perturbation runtime."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import sha256
import importlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from threading import RLock
from time import perf_counter
from types import MappingProxyType, SimpleNamespace
from typing import Mapping, Protocol, Sequence

import numpy as np

from pde_agent_verifier_solver.sdk import (
    Case,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    GenerationKind,
    GenerationPurpose,
    GenerationRequest,
    RecoverableSamplingRejection,
)

from .problem import VPFPContinuousMacroTruth, load_vpfp_case_record
from .batched_source import evaluate_requested_source
from .public_problem import (
    VPFP_DEVELOPMENT_SOLUTION_SESSION_METHODS,
    VPFP_MAX_ORACLE_POINTS_PER_CALL,
    VPFP_PUBLIC_SESSION_METHODS,
)


VPFP_UNIFIED_PERTURBATION_SCHEMA = "vpfp_unified_anchored_perturbation_v1"
VPFP_UNIFIED_GENERATION_AUDIT = "unified_generation_audit.json"
_STRUCTURED_ORACLE_KERNELS = ("structured_batch_v1", "structured_reuse_v1")
_TRUSTED_ORACLE_KERNELS = ("basic_vectorized_v1",) + _STRUCTURED_ORACLE_KERNELS
_PREPARED_ORACLE_MAX_CASES = 16
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PHYSICAL_REPORT_SCHEMA = "vpfp_structured_answer_perturbation_v2"
_PHYSICAL_AUDIT_SCHEMA = "vpfp_structured_answer_perturbation_audit_v2"
_EXPECTED_LEGACY_MODULE_SHA256 = {
    "agent_verifier.vpfp.robust_perturbation": (
        "38e27d17246e582616f0d73c324ff35d8d68901f8c0dad6815333aace2ad37b6"
    ),
    "agent_verifier.vpfp.analytic_source": (
        "c44daa8d87c6a3ff019521dc650798afd88e8c43d6e255b5e487b3873d8734b0"
    ),
    "agent_verifier.vpfp.continuous_cases": (
        "3e637e4239574c2b82d1b3487cce81c5958a7b830b070fbb06628dfd8ae7ef9d"
    ),
    "agent_verifier.vpfp.continuous_macros": (
        "b4a22c6af2fe12146d41100d94c2a2a792b3bf30b542cebff7693cb9ace49614"
    ),
    "agent_verifier.vpfp.operators": (
        "df5188d237cf95eddcb22449bcee252844faf6efca736779f4165abe9f8b7b43"
    ),
}
_ANSWER_FIRST_VERIFICATION_METRICS = (
    "answer_first_initial_data_rederived_from_primary",
    "rho0_materialized_cache_exact",
    "g0_materialized_cache_exact",
)
_PHYSICAL_GATES = frozenset({
    "finite",
    "rho_positive_by_sample_and_analytic_bound",
    "macro_displacement_visible",
    "macro_displacement_bounded",
    "microscopic_displacement_visible",
    "microscopic_displacement_bounded",
    "expanded_hermite_energy_visible",
    "source_increment_visible",
    "source_increment_bounded",
    "temporal_derivative_bounded",
    "initial_macro_preserved",
    "initial_coefficients_preserved",
})
_PURPOSE_COHORTS = {
    GenerationPurpose.DEVELOPMENT: "development",
    GenerationPurpose.DEVELOPMENT_SCREEN: "development_screen",
    GenerationPurpose.SELECTION: "selection",
    GenerationPurpose.PROMOTION: "promotion",
    GenerationPurpose.TERMINAL: "terminal",
    GenerationPurpose.GUIDANCE_FEEDBACK: "guidance_feedback",
}


class VPFPGenerationTransaction(Protocol):
    """Controller-private publication transaction for one generated batch."""

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


@dataclass(frozen=True)
class VPFPRuntimeCase:
    """Physical result; handle and audit remain controller-private."""

    record: CaseRecord
    private_handle: object | None = None
    audit: Mapping[str, object] = field(default_factory=dict)
    transaction: VPFPGenerationTransaction | None = None


class VPFPPhysicalRuntime(Protocol):
    """The only equation-specific boundary used by the unified backend."""

    supports_perturbations: bool

    def generate_base(
        self,
        *,
        seeds: Sequence[int],
        case_ids: Sequence[str],
        visibility: CaseVisibility,
        round_index: int,
        cohort: str,
    ) -> tuple[VPFPRuntimeCase, ...]: ...

    def generate_perturbed(
        self,
        *,
        anchors: Sequence[Case],
        private_handles: Sequence[object],
        seeds: Sequence[int],
        case_ids: Sequence[str],
        visibility: CaseVisibility,
        round_index: int,
        cohort: str,
    ) -> tuple[VPFPRuntimeCase, ...]: ...

    public_session_methods: tuple[str, ...]
    development_solution_session_methods: tuple[str, ...]

    def public_session_factory(self, case: Case) -> object: ...

    def development_solution_session_factory(self, case: Case) -> object: ...

    def release_cases(self, case_ids: Sequence[str]) -> None: ...


@dataclass(frozen=True)
class _Anchor:
    case: Case
    private_handle: object
    provenance: tuple[str, str, str]


class VPFPAnchoredPerturbationBackend:
    """Implement clean roots and anchored children for a frozen runtime."""

    supports_perturbations = True
    supports_guidance = False

    def __init__(self, runtime: VPFPPhysicalRuntime) -> None:
        if getattr(runtime, "supports_perturbations", False) is not True:
            raise ValueError("VPFP runtime must explicitly support perturbations")
        self.runtime = runtime
        self._anchors: dict[str, _Anchor] = {}
        self._issued_case_ids: set[str] = set()
        self.public_session_methods = tuple(
            getattr(runtime, "public_session_methods", ())
        )
        self.development_solution_session_methods = tuple(
            getattr(runtime, "development_solution_session_methods", ())
        )

    def train_base(self, *args, **kwargs):
        raise RuntimeError("VPFP unified runs require an already-frozen Generator")

    def fit_guidance(self, *args, **kwargs):
        raise RuntimeError("VPFP unified runs do not update frozen guidance")

    def public_session_factory(self, case: Case) -> object:
        factory = getattr(self.runtime, "public_session_factory", None)
        if not callable(factory):
            raise RuntimeError("VPFP runtime has no public numerical Oracle")
        return factory(case)

    def development_solution_session_factory(self, case: Case) -> object:
        factory = getattr(
            self.runtime, "development_solution_session_factory", None
        )
        if not callable(factory):
            raise RuntimeError("VPFP runtime has no completed-solution Oracle")
        return factory(case)

    def release_cases(self, case_ids: Sequence[str]) -> None:
        identifiers = _validated_release_case_ids(case_ids)
        release = getattr(self.runtime, "release_cases", None)
        if callable(release):
            release(identifiers)
        elif self.public_session_methods or self.development_solution_session_methods:
            raise RuntimeError("VPFP runtime has no resident-case release hook")
        for case_id in identifiers:
            self._anchors.pop(case_id, None)
            self._issued_case_ids.discard(case_id)

    def generate(self, base_state, guidance_state, request, progress=None):
        del base_state, progress
        if guidance_state is not None:
            raise ValueError("anchored VPFP generation uses frozen guidance")
        _validate_request(request)
        provenance = _provenance(request)

        if request.kind is GenerationKind.BASE:
            seeds = _seeds(request)
            case_ids = _case_ids(request, seeds)
            if any(case_id in self._issued_case_ids for case_id in case_ids):
                raise ValueError("VPFP BASE request would reuse a case id")
            rows = tuple(self.runtime.generate_base(
                seeds=seeds,
                case_ids=case_ids,
                visibility=request.visibility,
                round_index=request.round_index,
                cohort=provenance[2],
            ))
            transaction = _runtime_transaction(rows)
            try:
                _validate_rows(rows, request, case_ids)
                pending = {}
                for row in rows:
                    if row.private_handle is None or row.audit:
                        raise ValueError(
                            "VPFP BASE result has an invalid private payload"
                        )
                    pending[row.record.case.case_id] = _Anchor(
                        row.record.case, row.private_handle, provenance
                    )
            except BaseException:
                if transaction is not None:
                    transaction.rollback()
                raise
            if transaction is not None:
                transaction.commit()
            self._issued_case_ids.update(case_ids)
            self._anchors.update(pending)
            return tuple(row.record for row in rows)

        bound = []
        for anchor in request.anchors:
            cached = self._anchors.get(anchor.case_id)
            if cached is None or cached.case is not anchor:
                raise ValueError("VPFP anchor was not generated by this backend")
            if cached.provenance != provenance:
                raise ValueError("VPFP anchor Generator provenance changed")
            bound.append(cached)
        seeds = _seeds(request)
        case_ids = _case_ids(request, seeds)
        if any(case_id in self._issued_case_ids for case_id in case_ids):
            raise ValueError("VPFP PERTURBED request would reuse a case id")
        rows = tuple(self.runtime.generate_perturbed(
            anchors=request.anchors,
            private_handles=tuple(item.private_handle for item in bound),
            seeds=seeds,
            case_ids=case_ids,
            visibility=request.visibility,
            round_index=request.round_index,
            cohort=provenance[2],
        ))
        transaction = _runtime_transaction(rows)
        try:
            _validate_rows(rows, request, case_ids)
            for row in rows:
                _require_physical_audit(row.audit)
        except BaseException:
            if transaction is not None:
                transaction.rollback()
            raise
        if transaction is not None:
            transaction.commit()
        self._issued_case_ids.update(case_ids)
        for anchor in request.anchors:
            del self._anchors[anchor.case_id]
        return tuple(row.record for row in rows)


@dataclass(frozen=True)
class _LegacyHandle:
    base_latent: object
    embedded_latent: object
    equation_condition: object


@dataclass(frozen=True)
class _ResidentCase:
    """Private continuous answer retained by the VPFP numerical runtime."""

    case: Case
    public_snapshot: Case
    latent: object
    equation_condition: np.ndarray


class _VPFPMeteredSession:
    """Per-candidate numerical view with observational usage accounting."""

    _METHOD_NAMES: tuple[str, ...] = ()

    def __init__(self, runtime, case: Case) -> None:
        self.case_id = case.case_id
        self._runtime = runtime
        self._usage = self._empty_usage()
        self._by_method = {
            method: self._empty_usage() for method in self._METHOD_NAMES
        }
        self._closed = False
        self._session_lock = RLock()

    @staticmethod
    def _empty_usage():
        return {
            "query_calls": 0,
            "evaluated_points": 0,
            "backend_wall_seconds": 0.0,
            "attempted_calls": 0,
            "requested_points": 0,
            "completed_points": 0,
            "failed_calls": 0,
            "unknown_request_size_calls": 0,
            "unknown_completed_size_calls": 0,
        }

    @staticmethod
    def _point_count(value):
        # Accounting must not replace a validator/backend exception, or make
        # an otherwise valid __array__ input fail merely because it has no len.
        try:
            return len(value)
        except Exception:
            return None

    def _invoke(self, points, evaluator, *, method):
        with self._session_lock:
            counters = (self._usage, self._by_method[method])
            requested = self._point_count(points)
            for counts in counters:
                counts["attempted_calls"] += 1
                if requested is None:
                    counts["unknown_request_size_calls"] += 1
                else:
                    counts["requested_points"] += requested
            started = perf_counter()
            try:
                if self._closed:
                    raise RuntimeError("VPFP numerical session is closed")
                result = evaluator(points)
            except BaseException:
                for counts in counters:
                    counts["failed_calls"] += 1
                raise
            finally:
                elapsed = perf_counter() - started
                for counts in counters:
                    counts["backend_wall_seconds"] += elapsed
            completed = self._point_count(
                result.get("rho") if isinstance(result, Mapping) else result
            )
            for counts in counters:
                counts["query_calls"] += 1
                if completed is None:
                    counts["unknown_completed_size_calls"] += 1
                else:
                    counts["completed_points"] += completed
                    counts["evaluated_points"] += completed
            return result

    @property
    def usage_snapshot(self) -> Mapping[str, object]:
        with self._session_lock:
            return MappingProxyType(
                {
                    **self._usage,
                    "cost_unit": "evaluated_physical_point",
                    "query_calls_semantics": "successful_returning_calls",
                    "evaluated_points_semantics": (
                        "successful_returned_points; not points entering the math kernel"
                    ),
                    "requested_points_semantics": (
                        "sum of available request outer lengths, including invalid requests; "
                        "unknown lengths recorded separately"
                    ),
                    "by_method": {
                        name: dict(counts) for name, counts in self._by_method.items()
                    },
                }
            )

    def close(self) -> None:
        with self._session_lock:
            self._closed = True


class _VPFPPublicSession(_VPFPMeteredSession):
    _METHOD_NAMES = ("source", "rho0", "g0")

    def close(self) -> None:
        with self._session_lock:
            self._closed = True
            resource = getattr(self, "_evaluation_resource", None)
            if resource is not None:
                self._runtime = None
                self._evaluation_resource = None
        if resource is not None:
            resource.release(self.case_id)

    def source(self, points):
        return self._invoke(
            points,
            lambda value: self._runtime.evaluate_source(self.case_id, value),
            method="source",
        )

    def rho0(self, points_xy):
        return self._invoke(
            points_xy,
            lambda value: self._runtime.evaluate_rho0(self.case_id, value),
            method="rho0",
        )

    def g0(self, points_xyv):
        return self._invoke(
            points_xyv,
            lambda value: self._runtime.evaluate_g0(self.case_id, value),
            method="g0",
        )


class _VPFPDevelopmentSession(_VPFPMeteredSession):
    _METHOD_NAMES = ("released_macros", "released_transport_macros")

    def released_macros(self, points_txy):
        return self._invoke(
            points_txy,
            lambda value: self._runtime.evaluate_macros(self.case_id, value),
            method="released_macros",
        )

    def released_transport_macros(self, points_txy):
        return self._invoke(
            points_txy,
            lambda value: self._runtime.evaluate_transport_macros(self.case_id, value),
            method="released_transport_macros",
        )


class VPFPLegacyStructuredRuntime:
    """Use vendored Fourier--Hermite calibration and source reconstruction.

    The supplied Generator and base model are existing qualified objects. This
    shim never trains or modifies them.
    """

    supports_perturbations = True

    def __init__(
        self,
        *,
        generator,
        base_model,
        expanded_decoder_checkpoint,
        output_root,
        materialized_grid,
        materialized_time,
        perturbation_config=None,
        device=None,
        materialization_chunk_size=262_144,
        maximum_perturbation_attempts=4,
        verification_samples=4,
        materialize_cases=True,
        oracle_kernel="reference_v1",
    ) -> None:
        if oracle_kernel not in ("reference_v1",) + _TRUSTED_ORACLE_KERNELS:
            raise ValueError("unknown VPFP oracle_kernel")
        self.oracle_kernel = oracle_kernel
        self.api = _legacy_api()
        self.generator = generator
        self.base_model = base_model
        self.perturbation_config = (
            perturbation_config or self.api.VPFPRobustPerturbationConfig()
        )
        self.expanded_config = self.api.expanded_vpfp_perturbation_config(
            base_model.config, self.perturbation_config
        )
        parameter = next(base_model.parameters())
        self.device = str(device or parameter.device)
        self.expanded_model = self.api.build_deterministic_direct_model(
            self.expanded_config, device=parameter.device
        )
        self.checkpoint = Path(expanded_decoder_checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file() or self.checkpoint.is_symlink():
            raise FileNotFoundError(self.checkpoint)
        checkpoint = self.api.torch.load(
            self.checkpoint, map_location="cpu", weights_only=True
        )
        adapter = dict(checkpoint.get("robust_answer_first_adapter") or {})
        if (
            checkpoint.get("decoder_config") != self.expanded_config.to_dict()
            or adapter.get("unused_neural_parameters_zeroed") is not True
            or adapter.get("continuous_hidden_answer_is_primary") is not True
            or adapter.get("generator_parameterization_public") is not False
        ):
            raise ValueError("VPFP expanded decoder checkpoint is not qualified")
        if oracle_kernel in _TRUSTED_ORACLE_KERNELS:
            # Structural compatibility is a load-boundary property, not work
            # to repeat at every coordinate batch in a trusted numerical call.
            self.api.analytic_source.require_analytic_structured_source(self.expanded_model)
        self.grid = materialized_grid
        self.time = materialized_time
        self.grid.validate()
        self.time.validate()
        self.output_root = Path(output_root).expanduser().resolve()
        if self.output_root == Path(self.output_root.anchor):
            raise ValueError("VPFP output_root must be a specific directory")
        if not isinstance(materialize_cases, bool):
            raise TypeError("VPFP materialize_cases must be boolean")
        self.materialize_cases = materialize_cases
        self.public_session_methods = (
            () if materialize_cases else VPFP_PUBLIC_SESSION_METHODS
        )
        self.development_solution_session_methods = (
            ()
            if materialize_cases
            else VPFP_DEVELOPMENT_SOLUTION_SESSION_METHODS
        )
        self._resident_cases: dict[str, _ResidentCase] = {}
        self._parallel_oracle_pool = None
        self._prepared_oracles = OrderedDict()
        self._prepared_oracle_stats = {"hits": 0, "misses": 0, "evictions": 0}
        self._runtime_lock = RLock()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.chunk_size = _positive(materialization_chunk_size, "chunk size")
        self.maximum_attempts = _positive(
            maximum_perturbation_attempts, "perturbation attempts"
        )
        self.verification_samples = _positive(
            verification_samples, "verification samples"
        )

    @classmethod
    def from_development_runtime(
        cls, runtime, *, output_root, materialize_cases=True
    ):
        """Reuse objects already qualified by the legacy robust runtime."""

        return cls(
            generator=runtime.generator,
            base_model=runtime.base_model,
            expanded_decoder_checkpoint=runtime.expanded_decoder_checkpoint,
            output_root=output_root,
            materialized_grid=runtime.materialized_grid,
            materialized_time=runtime.materialized_time,
            perturbation_config=runtime.perturbation_config,
            device=runtime.device,
            materialization_chunk_size=runtime.config.materialization_chunk_size,
            materialize_cases=materialize_cases,
        )

    def public_session_factory(self, case: Case) -> object:
        resident = self._active_resident(case)
        pool = getattr(self, "_parallel_oracle_pool", None)
        if pool is not None:
            return pool.session(case)
        from .evaluation_devices import public_evaluation_session

        session = public_evaluation_session(self, resident)
        return session if session is not None else _VPFPPublicSession(self, resident.case)

    def public_session_operation_lock(self, case: Case):
        """Independent locks exist only for explicitly active GPU replicas."""
        pool = getattr(self, "_parallel_oracle_pool", None)
        return None if pool is None else pool.operation_lock(case.case_id)

    def development_solution_session_factory(self, case: Case) -> object:
        if not isinstance(case, Case):
            raise TypeError("VPFP completed-solution session requires a Case")
        if case.visibility is not CaseVisibility.DEVELOPMENT:
            raise PermissionError(
                "VPFP completed-solution access is DEVELOPMENT-only"
            )
        resident = self._active_resident(case)
        return _VPFPDevelopmentSession(self, resident.case)

    def release_cases(self, case_ids: Sequence[str]) -> None:
        """Forget resident answers; disk-materialized replay remains durable."""

        if not hasattr(self, "_resident_cases"):
            # Compatibility for old test/replay runtimes built without __init__.
            return
        identifiers = _validated_release_case_ids(case_ids)
        if getattr(self, "materialize_cases", True):
            return
        with self._runtime_lock:
            for case_id in identifiers:
                self._resident_cases.pop(case_id, None)
                getattr(self, "_prepared_oracles", {}).pop(case_id, None)

    def _prepared_oracle_for(self, resident):
        """Lazily prepare only a queried immutable answer; never a source grid.

        The caller holds _runtime_lock. A bounded per-Case object owns device
        coefficients and exact-coordinate intermediate reuse; no final source
        or solver output is cached. The batch-only arm discards it after use.
        """
        from .structured_oracle import StructuredOracle
        reuse = self.oracle_kernel == "structured_reuse_v1"
        if not hasattr(self, "_prepared_oracles"):
            self._prepared_oracles = OrderedDict()
            self._prepared_oracle_stats = {"hits": 0, "misses": 0, "evictions": 0}
        cache = self._prepared_oracles
        key = resident.case.case_id
        if reuse and key in cache:
            self._prepared_oracle_stats["hits"] += 1
            cache.move_to_end(key)
            return cache[key]
        self._prepared_oracle_stats["misses"] += 1
        if reuse and len(cache) >= _PREPARED_ORACLE_MAX_CASES:
            cache.popitem(last=False)
            self._prepared_oracle_stats["evictions"] += 1
        prepared = StructuredOracle(self.expanded_model, resident.latent,
                                    resident.equation_condition)
        if reuse:
            cache[key] = prepared
        return prepared

    def prepared_oracle_cache_info(self):
        """Controller-only counters for offline profiling, not a public method."""
        with self._runtime_lock:
            cache = getattr(self, "_prepared_oracles", {})
            return {"case_count": len(cache), "max_cases": _PREPARED_ORACLE_MAX_CASES,
                    **getattr(self, "_prepared_oracle_stats", {"hits": 0, "misses": 0, "evictions": 0}),
                    "cases": {key: value.cache_info() for key, value in cache.items()}}

    def evaluate_source(self, case_id: str, points):
        """Answer only requested pairs, sharing their exact coordinate jets."""

        query = _validated_oracle_points(points, width=5, name="source")
        self._validate_query_times(query[:, 0])
        with self._runtime_lock:
            resident = self._resident_for_id(case_id)
            if getattr(self, "oracle_kernel", "reference_v1") in _STRUCTURED_ORACLE_KERNELS:
                with self.api.torch.no_grad():
                    values = self._prepared_oracle_for(resident).source(query, chunk_size=8192)
                return _trusted_oracle_array(values, (len(query),), "source")
            runtime_latent = _latent_for_model(
                resident.latent, self.expanded_model, self.api.torch
            )
            with self.api.torch.no_grad():
                basic = getattr(self, "oracle_kernel", "reference_v1") == "basic_vectorized_v1"
                if basic:
                    from .basic_oracle import evaluate_source_basic
                    operation = evaluate_source_basic
                else:
                    operation = evaluate_requested_source
                values = operation(
                    self.expanded_model,
                    query,
                    runtime_latent,
                    equation_condition=resident.equation_condition,
                    chunk_size=8192,
                )
            # One device-to-host source transfer per public request, not one
            # per historical 512-point source chunk. Only the reference
            # kernel retains its historical per-call result checks.
            convert = _trusted_oracle_array if basic else _finite_oracle_array
            return convert(values, (len(query),), "source")

    def evaluate_source_reference(self, case_id: str, points):
        """Controller-only historical 512-point path for numerical comparison.

        Not a public-session method: candidates still see only ``source``.
        Preserve the former order, kernel, and per-chunk conversion for replay.
        """

        query = _validated_oracle_points(points, width=5, name="source")
        self._validate_query_times(query[:, 0])
        rows = []
        with self._runtime_lock:
            resident = self._resident_for_id(case_id)
            runtime_latent = _latent_for_model(
                resident.latent, self.expanded_model, self.api.torch
            )
            with self.api.ieee_float32_coordinate_matmul(self.expanded_model):
                with self.api.torch.no_grad():
                    for chunk in _oracle_chunks(query, 512):
                        values = self.api.manufacture_standard_vpfp_source_analytic(
                            self.expanded_model,
                            chunk,
                            runtime_latent,
                            equation_condition=resident.equation_condition,
                        )["source"]
                        rows.append(
                            _finite_oracle_array(
                                values, (len(chunk),), "source"
                            )
                        )
        return np.concatenate(rows)

    def evaluate_rho0(self, case_id: str, points_xy):
        query = _validated_oracle_points(points_xy, width=2, name="rho0")
        t0 = float(self.expanded_model.config.time_domain[0])
        phase = np.column_stack(
            (np.full(len(query), t0), query, np.zeros((len(query), 2)))
        )
        with self._runtime_lock:
            resident = self._resident_for_id(case_id)
            return self._evaluate_field_chunks(resident, phase, "rho")

    def evaluate_g0(self, case_id: str, points_xyv):
        query = _validated_oracle_points(points_xyv, width=4, name="g0")
        t0 = float(self.expanded_model.config.time_domain[0])
        phase = np.column_stack((np.full(len(query), t0), query))
        with self._runtime_lock:
            resident = self._resident_for_id(case_id)
            return self._evaluate_field_chunks(resident, phase, "g")

    def evaluate_macros(self, case_id: str, points_txy):
        query = _validated_oracle_points(points_txy, width=3, name="macros")
        self._validate_query_times(query[:, 0])
        with self._runtime_lock:
            resident = self._resident_for_id(case_id)
            return self._evaluate_macro_latent(
                resident.latent, resident.equation_condition, query
            )

    def evaluate_transport_macros(self, case_id: str, points_txy):
        from .transport_moments import evaluate_hermite_transport_points
        query = _validated_oracle_points(points_txy, width=3, name="transport_macros")
        self._validate_query_times(query[:, 0])
        with self._runtime_lock:
            resident = self._resident_for_id(case_id)
            latent = _latent_for_model(resident.latent, self.expanded_model, self.api.torch)
            with self.api.ieee_float32_coordinate_matmul(self.expanded_model):
                values = evaluate_hermite_transport_points(
                    self.expanded_model, latent, query,
                    equation_condition=resident.equation_condition,
                    chunk_size=min(self.chunk_size, 8192))
        return _finite_macro_oracle(values, len(query), transport=True)

    def _evaluate_field_chunks(self, resident, phase_points, field_name):
        rows = []
        with self._runtime_lock:
            if getattr(self, "oracle_kernel", "reference_v1") in _STRUCTURED_ORACLE_KERNELS:
                with self.api.torch.no_grad():
                    values = self._prepared_oracle_for(resident).initial(
                        phase_points, field_name=field_name)
                return _trusted_oracle_array(values, (len(phase_points),), field_name)
            runtime_latent = _latent_for_model(
                resident.latent, self.expanded_model, self.api.torch
            )
            with self.api.ieee_float32_coordinate_matmul(self.expanded_model):
                with self.api.torch.no_grad():
                    for chunk in _oracle_chunks(phase_points, 8192):
                        basic = getattr(self, "oracle_kernel", "reference_v1") == "basic_vectorized_v1"
                        if basic:
                            from .basic_oracle import evaluate_initial_basic
                            values = evaluate_initial_basic(
                                self.expanded_model, chunk, runtime_latent,
                                equation_condition=resident.equation_condition,
                                field_name=field_name,
                            )
                        else:
                            values = self.expanded_model.fields(
                                chunk,
                                runtime_latent,
                                equation_condition=resident.equation_condition,
                            )[field_name]
                        convert = _trusted_oracle_array if basic else _finite_oracle_array
                        rows.append(
                            convert(
                                values, (len(chunk),), field_name
                            )
                        )
        return np.concatenate(rows)

    def _evaluate_macro_latent(self, latent, condition, points_txy):
        with self._runtime_lock:
            runtime_latent = _latent_for_model(
                latent, self.expanded_model, self.api.torch
            )
            with self.api.ieee_float32_coordinate_matmul(self.expanded_model):
                values = self.api.evaluate_continuous_hermite_macro_points(
                    self.expanded_model,
                    runtime_latent,
                    points_txy,
                    equation_condition=condition,
                    chunk_size=min(self.chunk_size, 8192),
                )
        return _finite_macro_oracle(values, len(points_txy))

    def _active_resident(self, case: Case) -> _ResidentCase:
        if not isinstance(case, Case):
            raise TypeError("VPFP numerical session requires a Case")
        with self._runtime_lock:
            resident = self._resident_cases.get(case.case_id)
        if resident is None:
            raise ValueError(
                "VPFP numerical session requires an active issued case"
            )
        if not _equivalent_vpfp_public_snapshot(
            resident.public_snapshot, case
        ):
            raise ValueError("VPFP numerical session public case snapshot changed")
        return resident

    def _resident_for_id(self, case_id: str) -> _ResidentCase:
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("VPFP resident case id cannot be empty")
        with self._runtime_lock:
            resident = self._resident_cases.get(case_id)
        if resident is None:
            raise ValueError("VPFP resident answer is no longer active")
        return resident

    def _register_resident(self, record: CaseRecord, latent, condition) -> None:
        case_id = record.case.case_id
        host_condition = _host_equation_condition(condition)
        if getattr(self, "oracle_kernel", "reference_v1") in _TRUSTED_ORACLE_KERNELS:
            # nu/T are immutable for the resident answer: validate once here.
            if (host_condition.shape not in ((2,), (1, 2))
                    or not np.isfinite(host_condition).all()
                    or host_condition.reshape(-1)[0] < 0
                    or host_condition.reshape(-1)[1] <= 0):
                raise ValueError("VPFP resident equation condition must have finite nu>=0, T>0")
        resident = _ResidentCase(
            case=record.case,
            public_snapshot=_snapshot_vpfp_public_case(record.case),
            latent=_host_latent(latent),
            equation_condition=host_condition,
        )
        with self._runtime_lock:
            if case_id in self._resident_cases:
                raise ValueError(
                    "VPFP runtime would replace an active resident answer"
                )
            self._resident_cases[case_id] = resident

    def _validate_query_times(self, values) -> None:
        t0, t1 = (
            float(value) for value in self.expanded_model.config.time_domain
        )
        tolerance = 1.0e-12 * max(1.0, abs(t1 - t0))
        if np.any(values < t0 - tolerance) or np.any(values > t1 + tolerance):
            raise ValueError(
                "VPFP Oracle query time lies outside the public interval"
            )

    def _preflight_resident_ids(self, case_ids: Sequence[str]) -> None:
        if not hasattr(self, "_resident_cases"):
            return
        with self._runtime_lock:
            active = tuple(
                case_id
                for case_id in case_ids
                if case_id in self._resident_cases
            )
        if active:
            raise FileExistsError(
                "VPFP case id already has an active resident answer: "
                f"{active[0]}"
            )

    def generate_base(
        self, *, seeds, case_ids, visibility, round_index, cohort
    ) -> tuple[VPFPRuntimeCase, ...]:
        _preflight_batch_targets(
            self.output_root, case_ids, companion_lengths=(len(seeds),)
        )
        self._preflight_resident_ids(case_ids)
        published_targets = []
        registered_case_ids = []
        try:
            configured_batch = getattr(
                getattr(self.generator, "config", None),
                "batch_size",
                len(seeds),
            )
            batch_size = _positive(configured_batch, "Generator batch size")
            rows = []
            for start in range(0, len(seeds), batch_size):
                stop = min(start + batch_size, len(seeds))
                rows.extend(self._generate_base_batch(
                    seeds=seeds[start:stop],
                    case_ids=case_ids[start:stop],
                    visibility=visibility,
                    round_index=round_index,
                    cohort=cohort,
                    published_targets=published_targets,
                    registered_case_ids=registered_case_ids,
                ))
            transaction = _PublishedBatchTransaction(
                self.output_root,
                published_targets,
                release_cases=self.release_cases,
                case_ids=tuple(registered_case_ids),
            )
            wrapped = tuple(
                VPFPRuntimeCase(
                    row.record, row.private_handle, row.audit, transaction
                )
                for row in rows
            )
        except BaseException:
            try:
                self.release_cases(registered_case_ids)
            finally:
                _rollback_published_targets(
                    self.output_root, published_targets
                )
            raise
        return wrapped

    def generate_perturbed(
        self,
        *,
        anchors,
        private_handles,
        seeds,
        case_ids,
        visibility,
        round_index,
        cohort,
    ) -> tuple[VPFPRuntimeCase, ...]:
        _preflight_batch_targets(
            self.output_root,
            case_ids,
            companion_lengths=(
                len(anchors), len(private_handles), len(seeds)
            ),
        )
        self._preflight_resident_ids(case_ids)
        published_targets = []
        registered_case_ids = []
        try:
            rows = self._generate_perturbed_batch(
                anchors=anchors,
                private_handles=private_handles,
                seeds=seeds,
                case_ids=case_ids,
                visibility=visibility,
                round_index=round_index,
                cohort=cohort,
                published_targets=published_targets,
                registered_case_ids=registered_case_ids,
            )
            transaction = _PublishedBatchTransaction(
                self.output_root,
                published_targets,
                release_cases=self.release_cases,
                case_ids=tuple(registered_case_ids),
            )
            wrapped = tuple(
                VPFPRuntimeCase(
                    row.record, row.private_handle, row.audit, transaction
                )
                for row in rows
            )
        except BaseException:
            try:
                self.release_cases(registered_case_ids)
            finally:
                _rollback_published_targets(
                    self.output_root, published_targets
                )
            raise
        return wrapped

    def _generate_base_batch(
        self,
        *,
        seeds,
        case_ids,
        visibility,
        round_index,
        cohort,
        published_targets,
        registered_case_ids=None,
    ) -> tuple[VPFPRuntimeCase, ...]:
        batch = self.generator.generate(tuple(seeds))
        required = ("latents", "equation_conditions", "equation_seeds")
        if not isinstance(batch, Mapping) or any(name not in batch for name in required):
            raise ValueError("VPFP legacy Generator returned an incomplete batch")
        returned_seeds = batch["equation_seeds"]
        shape = getattr(returned_seeds, "shape", (len(returned_seeds),))
        if tuple(shape) != (len(seeds),):
            raise ValueError("VPFP legacy Generator returned malformed equation seeds")
        if tuple(int(value) for value in batch["equation_seeds"]) != tuple(seeds):
            raise ValueError("VPFP legacy Generator changed requested equation seeds")
        if len(batch["latents"]) != len(seeds) or len(batch["equation_conditions"]) != len(seeds):
            raise ValueError("VPFP legacy Generator returned the wrong batch size")

        rows = []
        for index, (seed, case_id) in enumerate(zip(seeds, case_ids)):
            latent = batch["latents"][index]
            condition = _host_equation_condition(
                batch["equation_conditions"][index]
            )
            embedded = self.api.embed_vpfp_structured_latent(
                self.base_model.config, self.expanded_config, latent
            )
            record = self._materialize(
                case_id, self.expanded_model, embedded, condition,
                visibility, round_index, cohort, None, "base",
                {"equation_seed": int(seed)},
                published_targets=published_targets,
                registered_case_ids=registered_case_ids,
            )
            rows.append(
                VPFPRuntimeCase(
                    record,
                    _LegacyHandle(
                        _host_latent(latent),
                        _host_latent(embedded),
                        condition,
                    ),
                )
            )
        return tuple(rows)

    def _generate_perturbed_batch(
        self,
        *,
        anchors,
        private_handles,
        seeds,
        case_ids,
        visibility,
        round_index,
        cohort,
        published_targets,
        registered_case_ids=None,
    ) -> tuple[VPFPRuntimeCase, ...]:
        rows = []
        for anchor, handle, seed, case_id in zip(
            anchors, private_handles, seeds, case_ids
        ):
            if not isinstance(handle, _LegacyHandle):
                raise TypeError("VPFP perturbation received an unknown private handle")
            calibration = None
            rejected_directions = []
            for attempt in range(self.maximum_attempts):
                candidate = self.api.calibrate_vpfp_robust_perturbation(
                    self.base_model,
                    handle.base_latent,
                    equation_condition=handle.equation_condition,
                    seed=_retry_seed(seed, attempt),
                    perturbation=self.perturbation_config,
                )
                if candidate.perturbed_latent is not None:
                    calibration = candidate
                    break
                rejected_directions.append(
                    _physical_rejection_summary(candidate.report)
                )
            if calibration is None:
                raise RecoverableSamplingRejection(
                    f"VPFP perturbation acceptance rejected {anchor.case_id}",
                    audit={
                        "schema": "vpfp_perturbation_sampling_rejection_v1",
                        "anchor_case_id": anchor.case_id,
                        "child_case_id": case_id,
                        "request_seed": int(seed),
                        "direction_attempt_count": len(rejected_directions),
                        "directions": rejected_directions,
                    },
                )
            _require_physical_audit(calibration.report)
            expected_embedded = _latent_like(
                handle.embedded_latent, calibration.embedded_base_latent
            )
            if not self.api.torch.equal(
                calibration.embedded_base_latent, expected_embedded
            ):
                raise RuntimeError("VPFP child does not embed its clean anchor exactly")
            audit = {
                **dict(calibration.report),
                "parent_case_id": anchor.case_id,
                "request_seed": int(seed),
            }
            record = self._materialize(
                case_id, calibration.model, calibration.perturbed_latent,
                handle.equation_condition, visibility, round_index, cohort,
                anchor.case_id, "perturbed", audit,
                published_targets=published_targets,
                registered_case_ids=registered_case_ids,
            )
            rows.append(VPFPRuntimeCase(record, audit=audit))
        return tuple(rows)

    def _resident_record(
        self,
        *,
        case_id,
        latent,
        condition,
        visibility,
        round_index,
        cohort,
        parent_case_id,
        variant,
    ) -> CaseRecord:
        """Publish a small public Case while retaining continuous truth."""

        axes = self.api.make_axes(self.grid, self.time)
        t = np.asarray(axes["t"], dtype=np.float64)
        x = np.asarray(axes["x"], dtype=np.float64)
        y = np.asarray(axes["y"], dtype=np.float64)
        mesh = np.meshgrid(x, y, indexing="ij")
        initial_points = np.column_stack(
            (
                np.full(mesh[0].size, float(t[0])),
                mesh[0].reshape(-1),
                mesh[1].reshape(-1),
            )
        )
        host_condition = _host_equation_condition(condition)
        macros = self._evaluate_macro_latent(
            latent, host_condition, initial_points
        )
        nx, ny = len(x), len(y)
        equation = dict(self.expanded_model.public_equation_metadata())
        equation.update(
            {
                "collision_nu": float(host_condition[0]),
                "temperature": float(host_condition[1]),
            }
        )
        question = {
            "case_id": case_id,
            "equation": equation.get("equation_version"),
            "equation_version": equation.get("equation_version"),
            "task": "predict exact full-space VPFP macroscopic quantities",
            "factorization": (
                "f(t,x,v)=rho(t,x)*g(t,v|x), integral_R2 g dv = 1"
            ),
            "source_origin": "generated_continuous_analytic_residual",
            "grid": self.grid.to_dict(),
            "time": self.time.to_dict(),
            "time_domain": list(self.expanded_model.config.time_domain),
            **equation,
            "materialization": {
                "role": "public_initial_macro_view_only",
                "primary_truth": "controller_private_continuous_answer",
                "agent_internal_grid_may_be_finer": True,
                "initial_macro_preview_points": nx * ny,
                "preview_grid_is_required": False,
                "source_precomputed": False,
                "g0_precomputed": False,
            },
            "numerical_access": {
                "mode": "continuous_on_demand",
                "methods": list(VPFP_PUBLIC_SESSION_METHODS),
                "coordinate_policy": "agent_selected",
                "max_points_per_request": VPFP_MAX_ORACLE_POINTS_PER_CALL,
                "unrequested_cartesian_completion": False,
                "preserves_query_order_and_duplicates": True,
            },
            "solver_output_contract_version": self.api.MACRO_OUTPUT_CONTRACT,
            "macroscopic_quantities": {
                "rho": "integral_R2 f dv",
                "momentum": "integral_R2 v f dv",
                "kinetic_energy": "0.5*integral_R2 |v|^2 f dv",
            },
        }
        arrays = {
            "t": t.copy(),
            "x": x.copy(),
            "y": y.copy(),
            "vx": np.asarray(axes["vx"], dtype=np.float64).copy(),
            "vy": np.asarray(axes["vy"], dtype=np.float64).copy(),
            "velocity_weights": np.asarray(
                self.api.trapezoidal_velocity_weights(self.grid),
                dtype=np.float64,
            ),
            "rho0": np.asarray(macros["rho"], dtype=np.float64).reshape(
                nx, ny
            ),
            "momentum0": np.asarray(
                macros["momentum"], dtype=np.float64
            ).reshape(nx, ny, 2),
            "kinetic_energy0": np.asarray(
                macros["kinetic_energy"], dtype=np.float64
            ).reshape(nx, ny),
        }
        case = Case(
            case_id=case_id,
            source=(
                CaseSource.GENERATED
                if variant == "base"
                else CaseSource.PERTURBED
            ),
            visibility=visibility,
            public_data={
                "equation": "2D2V Vlasov--Poisson--Fokker--Planck",
                "question": question,
                "arrays": arrays,
            },
            round_index=round_index,
            parent_case_id=parent_case_id,
            tags=("vpfp", "unified-paired", variant, f"cohort:{cohort}"),
        )
        record = CaseRecord(
            case,
            VPFPContinuousMacroTruth(
                case_id=case_id,
                t_axis=tuple(float(value) for value in t),
                x_axis=tuple(float(value) for value in x),
                y_axis=tuple(float(value) for value in y),
                evaluate_points=self.evaluate_macros,
                evaluate_transport_points=self.evaluate_transport_macros,
            ),
        )
        self._register_resident(record, latent, host_condition)
        return record

    def _materialize(
        self,
        case_id,
        model,
        latent,
        condition,
        visibility,
        round_index,
        cohort,
        parent_case_id,
        variant,
        audit,
        *,
        published_targets=None,
        registered_case_ids=None,
    ) -> CaseRecord:
        if not getattr(self, "materialize_cases", True):
            record = self._resident_record(
                case_id=case_id,
                latent=latent,
                condition=condition,
                visibility=visibility,
                round_index=round_index,
                cohort=cohort,
                parent_case_id=parent_case_id,
                variant=variant,
            )
            if registered_case_ids is not None:
                registered_case_ids.append(case_id)
            return record

        final_dir = _case_target(self.output_root, case_id)
        if final_dir.exists() or final_dir.is_symlink():
            raise FileExistsError(final_dir)
        staging_dir = Path(tempfile.mkdtemp(
            prefix=".vpfp-case-", dir=self.output_root
        ))
        published = False
        try:
            self._materialize_in_directory(
                staging_dir, case_id, model, latent, condition, visibility,
                round_index, cohort, parent_case_id, variant, audit,
            )
            if final_dir.exists() or final_dir.is_symlink():
                raise FileExistsError(final_dir)
            staging_dir.rename(final_dir)
            published = True
            if published_targets is not None:
                published_targets.append(final_dir)
        except BaseException:
            if published and (
                published_targets is None
                or final_dir not in published_targets
            ):
                _rollback_published_targets(self.output_root, (final_dir,))
            _remove_staging_dir(staging_dir, self.output_root)
            raise
        try:
            return load_vpfp_case_record(
                final_dir,
                visibility=visibility,
                source=(
                    CaseSource.GENERATED
                    if variant == "base"
                    else CaseSource.PERTURBED
                ),
                round_index=round_index,
                parent_case_id=parent_case_id,
                tags=("vpfp", "unified-paired", variant, f"cohort:{cohort}"),
            )
        except BaseException:
            if published_targets is None:
                _rollback_published_targets(self.output_root, (final_dir,))
            raise

    def _materialize_in_directory(
        self,
        case_dir,
        case_id,
        model,
        latent,
        condition,
        visibility,
        round_index,
        cohort,
        parent_case_id,
        variant,
        audit,
    ) -> CaseRecord:
        self.api.materialize_continuous_case(
            case_dir,
            case_id=case_id,
            model=model,
            decoder_checkpoint=self.checkpoint,
            latent=latent,
            grid=self.grid,
            time=self.time,
            equation_condition=condition,
            source_origin="generated_continuous_autodiff_residual",
            source_derivative_backend="analytic",
            solver_output_contract=self.api.MACRO_OUTPUT_CONTRACT,
            generator_metadata={
                "generator_family": "coarse_projection_answer_first",
                "unified_backend_schema": VPFP_UNIFIED_PERTURBATION_SCHEMA,
                "robust_variant": variant,
                "base_generator_parameters_modified": False,
            },
            case_spec={
                "cohort": cohort,
                "truth_primary": "continuous_hidden_answer",
                "low_fidelity_equation_solve_is_truth": False,
                "answer_first_perturbation_applied": variant == "perturbed",
                "parent_case_id": parent_case_id,
            },
            evaluation_chunk_size=self.chunk_size,
        )
        verification = self.api.verify_continuous_case_dir(
            case_dir,
            device=self.device,
            truth_sample_count=self.verification_samples,
            source_sample_count=self.verification_samples,
        )
        _require_answer_first_materialization_verification(verification)
        private_audit = {
            "schema": VPFP_UNIFIED_PERTURBATION_SCHEMA,
            "case_id": case_id,
            "variant": variant,
            "parent_case_id": parent_case_id,
            "generation": dict(audit),
            "materialization_verification": verification,
        }
        (case_dir / VPFP_UNIFIED_GENERATION_AUDIT).write_text(
            json.dumps(private_audit, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return load_vpfp_case_record(
            case_dir,
            visibility=visibility,
            source=CaseSource.GENERATED if variant == "base" else CaseSource.PERTURBED,
            round_index=round_index,
            parent_case_id=parent_case_id,
            tags=("vpfp", "unified-paired", variant, f"cohort:{cohort}"),
        )


def _host_equation_condition(condition):
    """Detach one small condition row at the Generator/runtime boundary."""

    host_condition = condition
    if hasattr(host_condition, "detach"):
        host_condition = host_condition.detach()
    if hasattr(host_condition, "cpu"):
        host_condition = host_condition.cpu()
    if hasattr(host_condition, "numpy"):
        host_condition = host_condition.numpy()
    return np.asarray(host_condition, dtype=float).copy()


def _host_latent(latent):
    """Keep hidden answers off accelerator memory between Oracle calls."""

    if latent is None or isinstance(
        latent, (str, bytes, bool, int, float, complex)
    ):
        return latent
    value = latent.detach() if hasattr(latent, "detach") else latent
    value = value.cpu() if hasattr(value, "cpu") else value
    if hasattr(value, "clone"):
        return value.clone()
    return np.asarray(value).copy()


def _latent_for_model(latent, model, torch_module):
    parameter = next(model.parameters())
    if hasattr(latent, "to"):
        return latent.to(dtype=parameter.dtype, device=parameter.device)
    return torch_module.as_tensor(
        latent, dtype=parameter.dtype, device=parameter.device
    )


def _latent_like(latent, reference):
    if hasattr(latent, "to") and hasattr(reference, "dtype"):
        return latent.to(dtype=reference.dtype, device=reference.device)
    return latent


def _validated_release_case_ids(case_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(case_ids, (str, bytes)) or not isinstance(
        case_ids, Sequence
    ):
        raise TypeError("VPFP release requires a sequence of case ids")
    identifiers = tuple(case_ids)
    if len(set(identifiers)) != len(identifiers) or any(
        not isinstance(case_id, str) or not case_id.strip()
        for case_id in identifiers
    ):
        raise ValueError(
            "VPFP release case ids must be unique nonempty strings"
        )
    return identifiers


def _snapshot_vpfp_public_case(case: Case) -> Case:
    """Detach one immutable public snapshot without exposing hidden state."""

    if not isinstance(case, Case):
        raise TypeError("VPFP public snapshot requires a Case")
    return Case(
        case_id=case.case_id,
        source=case.source,
        visibility=case.visibility,
        public_data=_freeze_vpfp_public_value(case.public_data),
        round_index=case.round_index,
        parent_case_id=case.parent_case_id,
        tags=tuple(case.tags),
    )


def _freeze_vpfp_public_value(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("VPFP public snapshot rejects object arrays")
        copied = np.array(value, copy=True, order="K", subok=False)
        copied.setflags(write=False)
        return copied
    if isinstance(value, np.generic):
        return value.copy()
    if isinstance(value, Mapping):
        keys = tuple(value)
        if any(not isinstance(key, str) for key in keys):
            raise TypeError(
                "VPFP public snapshot mapping keys must be strings"
            )
        return MappingProxyType(
            {
                key: _freeze_vpfp_public_value(value[key])
                for key in keys
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_vpfp_public_value(item) for item in value)
    if value is None or isinstance(
        value, (str, bytes, bool, int, float, complex)
    ):
        return value
    raise TypeError(
        f"unsupported VPFP public snapshot value: {type(value).__name__}"
    )


def _equivalent_vpfp_public_snapshot(trusted: Case, presented: Case) -> bool:
    if not isinstance(presented, Case):
        return False
    if (
        trusted.case_id != presented.case_id
        or trusted.source is not presented.source
        or trusted.visibility is not presented.visibility
        or trusted.round_index != presented.round_index
        or trusted.parent_case_id != presented.parent_case_id
        or tuple(trusted.tags) != tuple(presented.tags)
    ):
        return False
    return _equivalent_vpfp_public_value(
        trusted.public_data, presented.public_data
    )


def _equivalent_vpfp_public_value(trusted, presented) -> bool:
    if isinstance(trusted, np.ndarray):
        return (
            isinstance(presented, np.ndarray)
            and trusted.dtype == presented.dtype
            and trusted.shape == presented.shape
            and np.array_equal(trusted, presented)
        )
    if isinstance(trusted, np.generic):
        return (
            isinstance(presented, np.generic)
            and trusted.dtype == presented.dtype
            and bool(np.array_equal(trusted, presented))
        )
    if isinstance(trusted, Mapping):
        if not isinstance(presented, Mapping):
            return False
        trusted_keys = tuple(trusted)
        presented_keys = tuple(presented)
        if (
            any(not isinstance(key, str) for key in presented_keys)
            or set(trusted_keys) != set(presented_keys)
        ):
            return False
        return all(
            _equivalent_vpfp_public_value(trusted[key], presented[key])
            for key in trusted_keys
        )
    if isinstance(trusted, tuple):
        return (
            isinstance(presented, (list, tuple))
            and len(trusted) == len(presented)
            and all(
                _equivalent_vpfp_public_value(left, right)
                for left, right in zip(trusted, presented)
            )
        )
    return type(trusted) is type(presented) and trusted == presented


def _validated_oracle_points(points, *, width: int, name: str) -> np.ndarray:
    try:
        query = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"VPFP {name} points must be numeric") from exc
    if query.ndim != 2 or query.shape[1] != width:
        raise ValueError(f"VPFP {name} points must have shape [N,{width}]")
    if not 1 <= len(query) <= VPFP_MAX_ORACLE_POINTS_PER_CALL:
        raise ValueError(
            f"VPFP {name} request must contain "
            f"1..{VPFP_MAX_ORACLE_POINTS_PER_CALL} points"
        )
    if not np.all(np.isfinite(query)):
        raise ValueError(f"VPFP {name} points must be finite")
    return np.ascontiguousarray(query)


def _oracle_chunks(points: np.ndarray, chunk_size: int):
    for start in range(0, len(points), chunk_size):
        yield points[start : start + chunk_size]


def _trusted_oracle_array(values, shape, name: str) -> np.ndarray:
    """Convert numerical output to the public CPU ABI without a result audit.

    The optimized path is ordinary trusted arithmetic. Shape is an interface
    invariant; numerical equivalence/finite-result checks belong to independent
    regression tests, not every source batch. No resident buffer is returned.
    """
    value = values.detach() if hasattr(values, "detach") else values
    value = value.cpu() if hasattr(value, "cpu") else value
    value = value.numpy() if hasattr(value, "numpy") else value
    array = np.asarray(value, dtype=np.float64)
    if array.shape != tuple(shape):
        raise ValueError(f"VPFP {name} Oracle returned shape {array.shape}, expected {shape}")
    return array


def _finite_oracle_array(values, shape, name: str) -> np.ndarray:
    value = values.detach() if hasattr(values, "detach") else values
    value = value.cpu() if hasattr(value, "cpu") else value
    value = value.numpy() if hasattr(value, "numpy") else value
    array = np.asarray(value, dtype=np.float64)
    if array.shape != tuple(shape):
        raise ValueError(
            f"VPFP {name} Oracle returned shape {array.shape}, expected {shape}"
        )
    if not np.all(np.isfinite(array)):
        raise FloatingPointError(
            f"VPFP {name} Oracle returned non-finite values"
        )
    return array.copy()


def _finite_macro_oracle(values, count: int, *, transport=False) -> dict[str, np.ndarray]:
    expected = {
        "rho": (count,),
        "momentum": (count, 2),
        "kinetic_energy": (count,),
        "raw_second_moment": (count, 3),
    }
    if transport:
        expected.update(electric=(count, 2), pressure=(count, 3), heat_flux=(count, 2))
    if not isinstance(values, Mapping) or set(values) != set(expected):
        raise ValueError(
            "VPFP macro Oracle returned an incomplete field mapping"
        )
    return {
        name: _finite_oracle_array(values[name], shape, name)
        for name, shape in expected.items()
    }


@lru_cache(maxsize=1)
def _legacy_api():
    """Prefer an installed legacy package; otherwise use the vendored copy."""

    try:
        robust = importlib.import_module("agent_verifier.vpfp.robust_perturbation")
    except ModuleNotFoundError as exc:
        if exc.name not in ("agent_verifier", "agent_verifier.vpfp"):
            raise
        legacy_src = Path(__file__).resolve().parents[4] / "legacy" / "vpfp" / "src"
        if not (legacy_src / "agent_verifier" / "vpfp").is_dir():
            raise RuntimeError("vendored VPFP physics runtime is unavailable") from exc
        sys.path.insert(0, str(legacy_src))
        # The unified checkout also contains a namespace-only agent_verifier
        # directory. If already imported, changing sys.path cannot reliably
        # replace its package path. Add only this repository's vendored subtree;
        # the exact physics-module identity check below remains mandatory.
        package = sys.modules.get("agent_verifier")
        if package is not None:
            paths = list(getattr(package, "__path__", ()))
            vendored = str(legacy_src / "agent_verifier")
            if vendored not in paths:
                package.__path__ = [*paths, vendored]
        importlib.invalidate_caches()
        robust = importlib.import_module("agent_verifier.vpfp.robust_perturbation")
    # Both vendored adapters share the historical agent_verifier namespace.
    # A previously imported namespace (including src's neural_generator
    # placeholder) does not adopt a subsequently added regular package merely
    # because another adapter prepends sys.path. Preserve existing package
    # objects and search precedence while making this checkout's NS subtree
    # visible after VPFP initialization as well as before it.
    ns_package = Path(__file__).resolve().parents[4] / "legacy/ns/src/agent_verifier"
    for name, directory in (
        ("agent_verifier", ns_package),
        ("agent_verifier.neural_generator", ns_package / "neural_generator"),
    ):
        package = sys.modules.get(name)
        if package is not None and directory.is_dir():
            paths = list(getattr(package, "__path__", ()))
            if str(directory) not in paths:
                package.__path__ = [*paths, str(directory)]
    importlib.invalidate_caches()
    analytic = importlib.import_module("agent_verifier.vpfp.analytic_source")
    cases = importlib.import_module("agent_verifier.vpfp.continuous_cases")
    macros = importlib.import_module("agent_verifier.vpfp.continuous_macros")
    operators = importlib.import_module("agent_verifier.vpfp.operators")
    _require_legacy_module_bundle_identity({
        "agent_verifier.vpfp.robust_perturbation": robust,
        "agent_verifier.vpfp.analytic_source": analytic,
        "agent_verifier.vpfp.continuous_cases": cases,
        "agent_verifier.vpfp.continuous_macros": macros,
        "agent_verifier.vpfp.operators": operators,
    })
    return SimpleNamespace(
        analytic_source=analytic,
        VPFPRobustPerturbationConfig=robust.VPFPRobustPerturbationConfig,
        build_deterministic_direct_model=robust.build_deterministic_direct_model,
        calibrate_vpfp_robust_perturbation=robust.calibrate_vpfp_robust_perturbation,
        embed_vpfp_structured_latent=robust.embed_vpfp_structured_latent,
        expanded_vpfp_perturbation_config=robust.expanded_vpfp_perturbation_config,
        materialize_continuous_case=cases.materialize_continuous_case,
        verify_continuous_case_dir=cases.verify_continuous_case_dir,
        MACRO_OUTPUT_CONTRACT=macros.MACRO_OUTPUT_CONTRACT,
        trapezoidal_velocity_weights=cases.trapezoidal_velocity_weights,
        make_axes=operators.make_axes,
        evaluate_continuous_hermite_macro_points=(
            macros.evaluate_continuous_hermite_macro_points
        ),
        manufacture_standard_vpfp_source_analytic=(
            analytic.manufacture_standard_vpfp_source_analytic
        ),
        ieee_float32_coordinate_matmul=(
            analytic.ieee_float32_coordinate_matmul
        ),
        torch=importlib.import_module("torch"),
    )


def _require_legacy_module_bundle_identity(modules: Mapping[str, object]) -> None:
    if set(modules) != set(_EXPECTED_LEGACY_MODULE_SHA256):
        raise RuntimeError("VPFP physical module bundle is incomplete")
    package_directories = set()
    for name, expected_sha256 in _EXPECTED_LEGACY_MODULE_SHA256.items():
        source_value = getattr(modules[name], "__file__", None)
        if not isinstance(source_value, str) or not source_value:
            raise RuntimeError(f"VPFP physical module has no source identity: {name}")
        source = Path(source_value).expanduser()
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(
                f"VPFP physical module is not a regular source file: {name}"
            )
        resolved = source.resolve(strict=True)
        package_directories.add(resolved.parent)
    if len(package_directories) != 1:
        raise RuntimeError("VPFP physical modules do not share one package directory")


def _require_answer_first_materialization_verification(verification) -> None:
    if not isinstance(verification, Mapping):
        raise RuntimeError("VPFP case failed physical verification: invalid report")
    if verification.get("accepted") is not True:
        raise RuntimeError(
            f"VPFP case failed physical verification: {verification.get('failures')}"
        )
    metrics = verification.get("metrics")
    missing = tuple(
        name
        for name in _ANSWER_FIRST_VERIFICATION_METRICS
        if not isinstance(metrics, Mapping) or metrics.get(name) is not True
    )
    if missing:
        raise RuntimeError(
            "VPFP case failed answer-first initial-data verification: "
            + ", ".join(missing)
        )


def _validate_request(request: GenerationRequest) -> None:
    if not isinstance(request, GenerationRequest) or request.count < 1:
        raise ValueError("VPFP generation needs a positive GenerationRequest")
    if request.kind is GenerationKind.BASE and request.anchors:
        raise ValueError("VPFP BASE requests cannot have anchors")
    if request.kind is GenerationKind.PERTURBED:
        if len(request.anchors) != request.count:
            raise ValueError("VPFP PERTURBED requests need one anchor per child")
        if len({anchor.case_id for anchor in request.anchors}) != request.count:
            raise ValueError("VPFP perturbation anchors must be unique")
    elif request.kind is not GenerationKind.BASE:
        raise ValueError("VPFP unified generation supports only BASE and PERTURBED")
    if (
        isinstance(request.seed, bool)
        or not isinstance(request.seed, int)
        or request.seed < 0
    ):
        raise ValueError("VPFP request seed must be a nonnegative integer")
    _generation_cohort(request)


def _generation_cohort(request: GenerationRequest) -> str:
    try:
        return _PURPOSE_COHORTS[request.purpose]
    except (KeyError, TypeError) as exc:
        raise ValueError("VPFP request purpose is unsupported") from exc


def _provenance(request: GenerationRequest) -> tuple[str, str, str]:
    generator_id = request.settings.get("fixed_generator_id")
    generator_sha256 = request.settings.get("fixed_generator_sha256")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (generator_id, generator_sha256)
    ):
        raise ValueError("VPFP request lacks fixed Generator provenance")
    if not _SHA256.fullmatch(generator_sha256):
        raise ValueError("VPFP Generator artifact sha256 is invalid")
    return generator_id, generator_sha256, _generation_cohort(request)


def _seeds(request: GenerationRequest) -> tuple[int, ...]:
    labels = (
        tuple(anchor.case_id for anchor in request.anchors)
        if request.anchors
        else tuple(str(index) for index in range(request.count))
    )
    return tuple(
        int.from_bytes(sha256(
            f"{VPFP_UNIFIED_PERTURBATION_SCHEMA}\0{request.seed}\0"
            f"{request.purpose.value}\0{request.round_index}\0"
            f"{request.kind.value}\0{index}\0{label}".encode()
        ).digest()[:8], "big") & ((1 << 63) - 1)
        for index, label in enumerate(labels)
    )


def _case_ids(request, seeds) -> tuple[str, ...]:
    return tuple(
        f"vpfp-r{request.round_index}-{request.kind.value}-"
        f"{sha256(str(seed).encode()).hexdigest()[:20]}"
        for seed in seeds
    )


def _validate_rows(rows, request, case_ids) -> None:
    if len(rows) != request.count:
        raise ValueError("VPFP physical runtime returned the wrong case count")
    parents = (
        tuple(anchor.case_id for anchor in request.anchors)
        if request.anchors else (None,) * len(rows)
    )
    anchors = request.anchors or (None,) * len(rows)
    source = CaseSource.PERTURBED if request.anchors else CaseSource.GENERATED
    for row, case_id, parent, anchor in zip(rows, case_ids, parents, anchors):
        if not isinstance(row, VPFPRuntimeCase) or not isinstance(row.record, CaseRecord):
            raise TypeError("VPFP physical runtime returned an invalid row")
        case = row.record.case
        if (
            case.case_id != case_id
            or case.source is not source
            or case.parent_case_id != parent
            or case.visibility is not request.visibility
            or case.round_index != request.round_index
        ):
            raise ValueError("VPFP physical runtime changed requested envelope")
        if anchor is not None:
            _validate_pair_public_contract(anchor, case)


def _validate_pair_public_contract(parent: Case, child: Case) -> None:
    """Freeze the PDE/domain/query contract while allowing perturbed fields."""

    parent_public = parent.public_data
    child_public = child.public_data
    if not isinstance(parent_public, Mapping) or not isinstance(child_public, Mapping):
        raise TypeError("VPFP perturbation needs public mapping contracts")
    if set(parent_public) != set(child_public):
        raise ValueError("VPFP perturbation changed its public PDE contract")

    for name in parent_public:
        if name == "arrays":
            continue
        if not _equivalent_vpfp_public_value(
            _vpfp_identity_blind_contract(parent_public[name]),
            _vpfp_identity_blind_contract(child_public[name]),
        ):
            raise ValueError("VPFP perturbation changed its public PDE contract")

    parent_arrays = parent_public.get("arrays")
    child_arrays = child_public.get("arrays")
    if not isinstance(parent_arrays, Mapping) or not isinstance(child_arrays, Mapping):
        raise TypeError("VPFP perturbation needs public array contracts")
    if set(parent_arrays) != set(child_arrays):
        raise ValueError("VPFP perturbation changed its public array contract")
    fixed_geometry = frozenset({"t", "x", "y", "vx", "vy", "velocity_weights"})
    for name in parent_arrays:
        left = np.asarray(parent_arrays[name])
        right = np.asarray(child_arrays[name])
        if (
            left.dtype.hasobject
            or right.dtype.hasobject
            or left.shape != right.shape
            or left.dtype != right.dtype
            or (name in fixed_geometry and not np.array_equal(left, right))
        ):
            raise ValueError(
                "VPFP perturbation changed its public domain or array shape"
            )


def _vpfp_identity_blind_contract(value):
    """Remove only embedded Case ids before comparing public contracts."""

    if isinstance(value, Mapping):
        if any(not isinstance(name, str) for name in value):
            raise TypeError("VPFP public contract keys must be strings")
        return {
            name: _vpfp_identity_blind_contract(item)
            for name, item in value.items()
            if name != "case_id"
        }
    if isinstance(value, (list, tuple)):
        return tuple(_vpfp_identity_blind_contract(item) for item in value)
    if isinstance(value, np.ndarray):
        copied = np.array(value, copy=True, order="K", subok=False)
        copied.setflags(write=False)
        return copied
    if isinstance(value, np.generic):
        return value.copy()
    return value


def _case_target(output_root: Path, case_id: str) -> Path:
    """Resolve exactly one controller-generated child below ``output_root``."""

    if (
        not isinstance(case_id, str)
        or not case_id
        or "\x00" in case_id
        or case_id in {".", ".."}
    ):
        raise ValueError("VPFP case id is not a safe path component")
    relative = Path(case_id)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name != case_id:
        raise ValueError("VPFP case id is not a safe path component")
    target = output_root / relative
    if target.parent != output_root:
        raise ValueError("VPFP case target escapes output_root")
    return target


def _remove_staging_dir(staging_dir: Path, output_root: Path) -> None:
    """Remove only the exact private sibling allocated by ``mkdtemp``."""

    if (
        staging_dir.parent != output_root
        or not staging_dir.name.startswith(".vpfp-case-")
    ):
        raise RuntimeError("refusing to clean an unsafe VPFP staging path")
    if staging_dir.is_symlink() or staging_dir.is_file():
        staging_dir.unlink()
    elif staging_dir.exists():
        shutil.rmtree(staging_dir)


def _preflight_batch_targets(
    output_root: Path, case_ids, *, companion_lengths=()
) -> tuple[Path, ...]:
    count = len(case_ids)
    if count < 1 or any(length != count for length in companion_lengths):
        raise ValueError("VPFP physical runtime received inconsistent batch sizes")
    targets = tuple(_case_target(output_root, case_id) for case_id in case_ids)
    if len(set(targets)) != count:
        raise ValueError("VPFP materialization batch has duplicate case ids")
    for target in targets:
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
    return targets


def _rollback_published_targets(output_root: Path, targets) -> None:
    for target in reversed(tuple(targets)):
        if not isinstance(target, Path) or target.parent != output_root:
            raise RuntimeError("refusing to roll back an unsafe VPFP case path")
        if _case_target(output_root, target.name) != target:
            raise RuntimeError("refusing to roll back an unsafe VPFP case path")
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.exists():
            shutil.rmtree(target)


class _PublishedBatchTransaction:
    def __init__(
        self,
        output_root: Path,
        targets,
        *,
        release_cases=None,
        case_ids=(),
    ) -> None:
        self.output_root = output_root
        self._targets = targets
        self._release_cases = release_cases
        self._case_ids = tuple(case_ids)
        self._finished = False

    def commit(self) -> None:
        if self._finished:
            return
        self._targets.clear()
        self._finished = True

    def rollback(self) -> None:
        if self._finished:
            return
        try:
            if self._release_cases is not None:
                self._release_cases(self._case_ids)
        finally:
            _rollback_published_targets(self.output_root, self._targets)
            self._targets.clear()
            self._finished = True


def _runtime_transaction(rows):
    transactions = tuple(getattr(row, "transaction", None) for row in rows)
    present = tuple(transaction for transaction in transactions if transaction is not None)
    if not present:
        return None
    unique = {id(transaction): transaction for transaction in present}
    transaction = present[0]
    valid = (
        len(present) == len(rows)
        and len(unique) == 1
        and callable(getattr(transaction, "commit", None))
        and callable(getattr(transaction, "rollback", None))
    )
    if not valid:
        for candidate in unique.values():
            rollback = getattr(candidate, "rollback", None)
            if callable(rollback):
                rollback()
        raise ValueError("VPFP runtime returned an inconsistent batch transaction")
    return transaction


def _json_safe_physical_metric(value):
    if isinstance(value, Mapping):
        normalized = {}
        for name, item in value.items():
            if not isinstance(name, str):
                raise ValueError("VPFP physical metric names must be strings")
            normalized[name] = _json_safe_physical_metric(item)
        return normalized
    if isinstance(value, np.ndarray):
        return _json_safe_physical_metric(value.tolist())
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_json_safe_physical_metric(item) for item in value]
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isfinite(number):
            return number
        if np.isnan(number):
            label = "nan"
        elif number > 0.0:
            label = "positive_infinity"
        else:
            label = "negative_infinity"
        return {"nonfinite": label}
    raise ValueError(
        "VPFP physical metric audit contains an unsupported value"
    )


def _physical_rejection_summary(report) -> dict[str, object]:
    if (
        not isinstance(report, Mapping)
        or report.get("schema") != _PHYSICAL_REPORT_SCHEMA
        or report.get("accepted") is not False
        or not isinstance(report.get("attempts"), Sequence)
        or not report["attempts"]
    ):
        raise ValueError("VPFP rejected perturbation has no valid physical audit")
    attempts = []
    for raw_attempt in report["attempts"]:
        gates = (
            raw_attempt.get("gates")
            if isinstance(raw_attempt, Mapping)
            else None
        )
        metrics = (
            raw_attempt.get("metrics")
            if isinstance(raw_attempt, Mapping)
            else None
        )
        if (
            not isinstance(raw_attempt, Mapping)
            or raw_attempt.get("schema") != _PHYSICAL_AUDIT_SCHEMA
            or raw_attempt.get("accepted") is not False
            or not isinstance(gates, Mapping)
            or set(gates) != _PHYSICAL_GATES
            or not all(isinstance(value, bool) for value in gates.values())
            or all(gates.values())
            or not isinstance(metrics, Mapping)
        ):
            raise ValueError(
                "VPFP rejected perturbation has a malformed physical attempt"
            )
        attempts.append({
            "backtrack_index": int(raw_attempt["backtrack_index"]),
            "scale": float(raw_attempt["scale"]),
            "gates": dict(gates),
            "failed_gates": sorted(
                name for name, passed in gates.items() if not passed
            ),
            "metrics": _json_safe_physical_metric(metrics),
        })
    return {
        "seed": int(report["seed"]),
        "attempt_count": len(attempts),
        "attempts": attempts,
    }


def _require_physical_audit(audit) -> None:
    selected = audit.get("selected_attempt") if isinstance(audit, Mapping) else None
    gates = selected.get("gates") if isinstance(selected, Mapping) else None
    if (
        not isinstance(audit, Mapping)
        or audit.get("schema") != _PHYSICAL_REPORT_SCHEMA
        or audit.get("accepted") is not True
        or not isinstance(selected, Mapping)
        or selected.get("schema") != _PHYSICAL_AUDIT_SCHEMA
        or selected.get("accepted") is not True
        or not isinstance(gates, Mapping)
        or set(gates) != _PHYSICAL_GATES
        or not all(value is True for value in gates.values())
        or audit.get("base_answer_embedded_exactly") is not True
        or audit.get("initial_condition_preserved") is not True
        or audit.get("generator_parameters_modified") is not False
        or selected.get("source_recomputed_after_perturbation") is not True
        or selected.get("velocity_normalization_exact_not_quadrature_repaired") is not True
    ):
        raise ValueError("VPFP perturbation lacks a passing physical audit")


def _retry_seed(seed, attempt) -> int:
    digest = sha256(f"vpfp-perturbation-retry-v1\0{seed}\0{attempt}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _positive(value, label) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"VPFP {label} must be a positive integer")
    return value


__all__ = [
    "VPFPAnchoredPerturbationBackend",
    "VPFPLegacyStructuredRuntime",
    "VPFPPhysicalRuntime",
    "VPFPRuntimeCase",
    "VPFP_UNIFIED_GENERATION_AUDIT",
    "VPFP_UNIFIED_PERTURBATION_SCHEMA",
]
