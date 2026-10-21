"""Thin answer-first backend for continuous manufactured NS questions.

This module adapts the existing ``ContinuousStructuredNSMMS`` scientific
object to the shared MacroClosedLoop surface.  It deliberately does not own
search, ranking, promotion, or Guidance policy.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import hashlib
import hmac
import json
from math import isfinite
from random import Random
import secrets
from threading import Condition, Lock, RLock, get_ident, local
from time import perf_counter
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Mapping, Protocol, Sequence

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    AnswerView,
    Case,
    CaseRecord,
    CaseSource,
    CaseVisibility,
    GenerationKind,
    QueryKind,
)

if TYPE_CHECKING:
    from pde_agent_verifier_solver.sdk.macro_loop import CohortKind, SampleRequest

from .macro_matching import NS_VELOCITY_FIELDS
from .public_problem import (
    NS_DEVELOPMENT_SOLUTION_SESSION_METHODS,
    NS_OPERATOR_ID,
    NS_PUBLIC_SESSION_METHODS,
)


class _NSCUDAWorkGuard:
    """Queries share GPU access; graph warmup/capture excludes all queries.

    Preserve the existing reentrant lock interface for capture and unknown
    adapters. Readers are only the audited, fixed-coefficient structured fields.
    A waiting capture takes priority over newly arriving readers.
    """

    def __init__(self):
        self._condition = Condition(RLock())
        self._readers = {}
        self._writer = None
        self._writer_depth = 0
        self._waiting_writers = 0

    def acquire(self, blocking=True, timeout=-1):
        identity = get_ident()
        if not blocking and timeout != -1:
            raise ValueError("can't specify a timeout for a non-blocking call")
        if timeout < 0 and timeout != -1:
            raise ValueError("timeout value must be positive")
        with self._condition:
            if self._writer == identity:
                self._writer_depth += 1
                return True
            if self._readers.get(identity, 0):
                if not blocking:
                    return False
                raise RuntimeError("NS query cannot start nested CUDA capture")
            self._waiting_writers += 1
            self._condition.notify_all()
            try:
                ready = lambda: self._writer is None and not self._readers
                if not blocking:
                    if not ready():
                        return False
                elif not ready():
                    from pde_agent_verifier_solver.sdk._candidate_timing import controller_queue_wait

                    deadline = None if timeout == -1 else perf_counter() + timeout
                    with controller_queue_wait() as check_cancelled:
                        while not ready():
                            check_cancelled()
                            remaining = None if deadline is None else deadline - perf_counter()
                            if remaining is not None and remaining <= 0:
                                return False
                            self._condition.wait(.1 if remaining is None else min(.1, remaining))
                        check_cancelled()
                self._writer = identity
                self._writer_depth = 1
                return True
            finally:
                self._waiting_writers -= 1
                self._condition.notify_all()

    def release(self):
        with self._condition:
            if self._writer != get_ident():
                raise RuntimeError("cannot release un-acquired NS CUDA capture guard")
            self._writer_depth -= 1
            if not self._writer_depth:
                self._writer = None
                self._condition.notify_all()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_):
        self.release()

    @contextmanager
    def shared_query(self):
        from pde_agent_verifier_solver.sdk._candidate_timing import controller_queue_wait

        identity = get_ident()
        with self._condition:
            def ready():
                return self._writer == identity or (
                    self._writer is None
                    and (not self._waiting_writers or self._readers.get(identity, 0))
                )
            if not ready():
                # Capture contention is Controller waiting, not solver compute.
                with controller_queue_wait() as check_cancelled:
                    while not ready():
                        check_cancelled()
                        self._condition.wait(.1)
                    check_cancelled()
            self._readers[identity] = self._readers.get(identity, 0) + 1
        try:
            yield
        finally:
            with self._condition:
                remaining = self._readers[identity] - 1
                if remaining:
                    self._readers[identity] = remaining
                else:
                    del self._readers[identity]
                self._condition.notify_all()


# Truth materialization may run outside the Generator operation lock. The guard
# spans all kernels and host copies so no query can leak into CUDA capture.
_NS_CUDA_CAPTURE_LOCK = _NSCUDAWorkGuard()


class _NSCUDAQueryAdmission:
    """Bound and fairly schedule immutable numerical queries on each GPU."""
    def __init__(self):
        self._lock = Lock()
        self._pools = {}

    @contextmanager
    def for_device(self, device):
        from pde_agent_verifier_solver.engine.fair_execution import FairAdmission, current_oracle_policy
        owner, capacity = current_oracle_policy()
        with self._lock:
            pool = self._pools.setdefault(device, FairAdmission(capacity))
        with pool._condition:
            pool.capacity = capacity
            pool._condition.notify_all()
        with pool.enter(owner):
            yield


_NS_CUDA_QUERY_ADMISSION = _NSCUDAQueryAdmission()
_NS_QUERY_STREAMS = local()

def _parallel_oracle_queries():
    from pde_agent_verifier_solver.engine.fair_execution import current_oracle_policy
    return current_oracle_policy()[1] > 1


class ContinuousNSAnswer(Protocol):
    """Minimal continuous answer required by the equation adapter.

    An optional ``boundary_condition`` mapping may declare
    ``kind="manufactured_velocity_trace"`` and ``access="query_boundary_velocity"``.
    Such answers require an explicitly trace-enabled source. Answers without
    that declaration retain the historical zero no-slip boundary contract.
    """

    @property
    def domain_bounds(self) -> tuple[float, float, float, float]: ...

    @property
    def time_interval(self) -> tuple[float, float]: ...

    @property
    def viscosity(self) -> float: ...

    @property
    def public_geometry(self) -> Mapping[str, object]: ...

    @property
    def public_metadata(self) -> Mapping[str, object]: ...

    def evaluate_velocity(self, physical_points): ...

    def evaluate_forcing(self, physical_points): ...


@dataclass(frozen=True)
class StructuredNSPerturbationResult:
    """A perturbed raw field plus controller-side realization metadata.

    ``StructuredNSManufacturedCaseSource`` publishes only an explicit safe
    projection of ``metadata``. Realization parameters remain in its private
    resident audit and never enter ``Case.public_data``.
    """

    answer: object
    metadata: Mapping[str, object] = field(default_factory=dict)


class StructuredNSMMSAnswer:
    """Adapt the legacy continuous structured field without copying it."""

    def __init__(
        self,
        field: object,
        *,
        public_geometry: Mapping[str, object] | None = None,
        public_metadata: Mapping[str, object] | None = None,
    ) -> None:
        if not callable(getattr(field, "velocity_pressure", None)):
            raise TypeError("structured NS field must expose velocity_pressure")
        if not callable(getattr(field, "quantities", None)):
            raise TypeError("structured NS field must expose quantities")
        bounds = tuple(float(value) for value in getattr(field, "domain_bounds"))
        if len(bounds) != 4 or any(not isfinite(value) for value in bounds):
            raise ValueError("structured NS field has invalid domain bounds")
        if bounds[1] <= bounds[0] or bounds[3] <= bounds[2]:
            raise ValueError("structured NS field domain is empty")
        t0 = float(getattr(field, "t0"))
        t1 = float(getattr(field, "t1"))
        viscosity = float(getattr(field, "viscosity"))
        if not (isfinite(t0) and isfinite(t1) and t1 > t0):
            raise ValueError("structured NS field has invalid time interval")
        if not isfinite(viscosity) or viscosity < 0.0:
            raise ValueError("structured NS field has invalid viscosity")

        geometry = public_geometry
        if geometry is None:
            geometry_spec = getattr(field, "geometry_spec", None)
            to_dict = getattr(geometry_spec, "to_dict", None)
            if callable(to_dict):
                geometry = to_dict()
        if not isinstance(geometry, Mapping):
            raise ValueError(
                "continuous NS fields need a serializable public geometry; "
                "new formal cases should use the continuous polygon ABI"
            )
        self._field = field
        self._thread_safe_field = _is_read_only_structured_field(field)
        # Exact Torch jets dispatch through the frozen field's declared basis.
        # Custom methods, trainable coefficients and sampled legacy geometry
        # retain their original differentiation behavior.
        self._analytic_field = (
            self._thread_safe_field
            and field.geometry_spec is not None
            and field.continuous_mms_abi in {
                "continuous_structured_ns_mms_v2",
                "continuous_structured_ns_mms_v3",
                "continuous_structured_ns_mms_v4",
                "continuous_structured_ns_mms_v5",
                "continuous_structured_ns_mms_v6",
            }
        )
        # A hidden evaluation owns graph buffers until all its public sessions close.
        # Development answers remain eager because Generator work can overlap them.
        self._analytic_cuda_cache = None
        self._domain_bounds = bounds
        self._time_interval = (t0, t1)
        self._viscosity = viscosity
        self._public_geometry = _plain_mapping(geometry, "public geometry")
        metadata = dict(public_metadata or {})
        continuous_abi = getattr(field, "continuous_mms_abi", None)
        if continuous_abi is not None:
            metadata.setdefault("continuous_mms_abi", str(continuous_abi))
        self._public_metadata = _plain_mapping(metadata, "public metadata")

    @property
    def boundary_condition(self):
        return getattr(self._field, "boundary_condition", None)

    @property
    def raw_answer(self) -> object:
        """Controller-only object used only by the injected perturber."""

        return self._field

    @property
    def domain_bounds(self) -> tuple[float, float, float, float]:
        return self._domain_bounds

    @property
    def time_interval(self) -> tuple[float, float]:
        return self._time_interval

    @property
    def viscosity(self) -> float:
        return self._viscosity

    @property
    def public_geometry(self) -> Mapping[str, object]:
        return self._public_geometry

    @property
    def public_metadata(self) -> Mapping[str, object]:
        return self._public_metadata

    def evaluate_velocity(self, physical_points):
        coordinates = _physical_points(
            physical_points,
            width=3,
            domain_bounds=self.domain_bounds,
            time_interval=self.time_interval,
        )
        return self._evaluate_validated_velocity(coordinates)

    def _query_context(self):
        # Known CPU fields cannot issue CUDA work and need not block unrelated
        # Oracle queries or graph capture. Unknown adapters remain conservative.
        if self._thread_safe_field:
            if self._field.device.type == "cpu":
                return nullcontext()
            if self._field.device.type == "cuda":
                return self._cuda_query_context()
        return _NS_CUDA_CAPTURE_LOCK

    @contextmanager
    def _cuda_query_context(self):
        # Tensor devices have a resolved index even when the field was created
        # with the bare "cuda" alias. Waiting happens before graph construction.
        device = self._field.coefficients.psi.device.index
        with _NS_CUDA_QUERY_ADMISSION.for_device(device):
            with _NS_CUDA_CAPTURE_LOCK.shared_query():
                if not _parallel_oracle_queries():
                    yield
                    return
                torch = self._field.torch
                streams = getattr(_NS_QUERY_STREAMS, "streams", None)
                if streams is None:
                    streams = _NS_QUERY_STREAMS.streams = {}
                with torch.cuda.device(device):
                    if device not in streams:
                        streams[device] = torch.cuda.Stream(device=device)
                    stream = streams[device]
                    stream.wait_stream(torch.cuda.default_stream(device))
                    with torch.cuda.stream(stream):
                        try:
                            yield
                        finally:
                            # Publish completed host values before releasing this lane.
                            stream.synchronize()

    def _public_vectors(self, evaluate):
        from ...sdk.oracle_profiling import current_oracle_profile
        profile = current_oracle_profile()
        torch = self._field.torch
        device = self._field.device
        start_event = finish_event = None
        if profile is not None and device.type == 'cuda':
            start_event, finish_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record()
        started = perf_counter()
        components = evaluate()
        if all(torch.is_tensor(component) for component in components):
            vector = torch.stack(components, dim=-1)
        else:
            import numpy as np
            vector = np.stack(tuple(_to_numpy(component) for component in components), axis=-1)
        if finish_event is not None:
            finish_event.record()
        enqueued = perf_counter()
        result = _to_numpy(vector)
        if profile is not None:
            profile['math_enqueue_seconds'] = enqueued - started
            profile['download_and_sync_seconds'] = perf_counter() - enqueued
            if finish_event is not None:
                finish_event.synchronize()
                profile['math_device_seconds'] = start_event.elapsed_time(finish_event) / 1000.
        return result

    def _evaluate_validated_velocity(self, coordinates):
        import numpy as np

        if (self._analytic_field and self._analytic_cuda_cache is not None
                and self._field.config.is_legacy_basis):
            # The cache owns GPU admission and capture guards; never upgrade a
            # shared guard held by _query_context to an exclusive capture.
            return self._analytic_cuda_cache.evaluate(
                self._field, coordinates[:, (1, 2, 0)], "velocity"
            )

        with self._query_context():
            if self._analytic_field:
                from agent_verifier.neural_generator.structured_ns_analytic import analytic_velocity
                return self._public_vectors(lambda: analytic_velocity(
                    self._field, coordinates[:, (1, 2, 0)]))
            return self._public_vectors(lambda: self._field.velocity_pressure(
                coordinates[:, (1, 2, 0)])[:2])

    def evaluate_forcing(self, physical_points):
        coordinates = _physical_points(
            physical_points,
            width=3,
            domain_bounds=self.domain_bounds,
            time_interval=self.time_interval,
        )
        return self._evaluate_validated_forcing(coordinates)

    def _evaluate_validated_forcing(self, coordinates):
        import numpy as np

        if (self._analytic_field and self._analytic_cuda_cache is not None
                and self._field.config.is_legacy_basis):
            # The cache owns GPU admission and capture guards; never upgrade a
            # shared guard held by _query_context to an exclusive capture.
            return self._analytic_cuda_cache.evaluate(
                self._field, coordinates[:, (1, 2, 0)], "forcing"
            )

        with self._query_context():
            if self._analytic_field:
                from agent_verifier.neural_generator.structured_ns_analytic import analytic_forcing
                return self._public_vectors(lambda: analytic_forcing(
                    self._field, coordinates[:, (1, 2, 0)]))
            def evaluate():
                quantities = self._field.quantities(coordinates[:, (1, 2, 0)])
                return quantities['forcing_u'], quantities['forcing_v']
            return self._public_vectors(evaluate)


@dataclass(frozen=True)
class LegacyStructuredNSSmoothPerturber:
    """Use the existing structured robustness backend with fresh randomness.

    ``spectral_complement`` is available without a PCA artifact and leaves the
    trained low-band representation support while preserving differentiability.
    PCA tangent/normal families may be selected when the matching PCA artifact
    is explicitly supplied.  ``amplitude`` must come from an offline physical
    calibration; this online adapter performs no audit or calibration work.
    ``spline_refinement`` is a distinct midpoint-refined coefficient complement;
    its unit coefficient-L2 amplitude is not interchangeable with cosine epsilon.
    """

    amplitude: float
    family: str = "spectral_complement"
    spectral_max_mode: int = 20
    pca: object | None = None
    make_perturbation: Callable[..., object] | None = None
    field_at_amplitude: Callable[..., object] | None = None

    def __post_init__(self) -> None:
        if not isfinite(float(self.amplitude)) or float(self.amplitude) <= 0.0:
            raise ValueError("NS perturbation amplitude must be positive")
        if self.family not in {
            "spline_refinement",
            "spectral_complement",
            "pca_tangent",
            "pca_normal",
        }:
            raise ValueError("unknown structured NS perturbation family")
        if self.family.startswith("pca_") and self.pca is None:
            raise ValueError(f"{self.family} needs its matching PCA artifact")

    def __call__(self, raw_parent: object, seed: int):
        make, materialize = self._symbols()
        kwargs = {
            "family": self.family,
            "seed": int(seed),
            "pca": self.pca,
            "spectral_max_mode": int(self.spectral_max_mode),
        }
        perturbation = make(raw_parent, **kwargs)
        sign = -1.0 if Random(int(seed) ^ 0x4E5350).getrandbits(1) else 1.0
        signed_amplitude = sign * float(self.amplitude)
        child = materialize(
            raw_parent,
            perturbation,
            amplitude=signed_amplitude,
        )
        return StructuredNSPerturbationResult(
            answer=child,
            metadata={
                "family": self.family,
                "seed": int(seed),
                "amplitude_magnitude": float(self.amplitude),
                "signed_calibrated_amplitude": signed_amplitude,
                "online_calibration_or_audit": False,
                "smooth_continuous_answer": True,
                "spectral_support_complement": (
                    self.family == "spectral_complement"
                ),
                **({
                    "coefficient_support_complement": True,
                    "complement_inner_product": "euclidean_fine_spline_coefficients",
                    "amplitude_convention": "l2_normalized_joint_streamfunction_coefficients",
                    "amplitude_equivalent_to_cosine": False,
                    "refinement_details": dict(perturbation.metadata),
                } if self.family == "spline_refinement" else {}),
                **({
                    "amplitude_convention": "l2_normalized_joint_streamfunction_coefficients",
                    "amplitude_equivalent_to_cosine": False,
                    "spectral_details": dict(perturbation.metadata),
                } if self.family == "spectral_complement" and
                    getattr(getattr(raw_parent, "config", None), "temporal_basis", "chebyshev") == "bspline" else {}),
            },
        )

    def _symbols(self):
        make = self.make_perturbation
        materialize = self.field_at_amplitude
        if make is None or materialize is None:
            try:
                from agent_verifier.neural_generator.structured_ns_robustness import (
                    field_at_amplitude,
                    make_perturbation,
                )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "the legacy structured NS implementation is not importable; "
                    "add legacy/ns/src to PYTHONPATH or inject both functions"
                ) from exc
            make = make or make_perturbation
            materialize = materialize or field_at_amplitude
        return make, materialize


@dataclass(frozen=True)
class _ResidentAnswer:
    case: Case
    raw_answer: object
    answer: ContinuousNSAnswer
    cohort: CohortKind
    perturbation_audit: Mapping[str, object] = field(default_factory=dict)


class StructuredNSManufacturedCaseSource:
    """Frozen clean sampler plus answer-space perturbation sampler.

    The clean sampler may wrap the trained structured coefficient Rectified
    Flow or a fixed release for tests.  The source never trains or updates it.
    """

    public_session_methods = NS_PUBLIC_SESSION_METHODS
    development_solution_session_methods = (
        NS_DEVELOPMENT_SOLUTION_SESSION_METHODS
    )

    def __init__(
        self,
        *,
        generator_id: str,
        artifact_sha256: str,
        clean_sampler: Callable[[int], object],
        perturbation_sampler: Callable[[object, int], object] | None,
        answer_adapter: Callable[[object], ContinuousNSAnswer] | None = None,
        max_points_per_request: int = 4096,
        boundary_velocity_trace: bool = False,
    ) -> None:
        if not isinstance(generator_id, str) or not generator_id.strip():
            raise ValueError("NS Generator id cannot be empty")
        if (
            not isinstance(artifact_sha256, str)
            or len(artifact_sha256) != 64
            or any(character not in "0123456789abcdef" for character in artifact_sha256)
        ):
            raise ValueError("NS Generator artifact needs a lowercase sha256")
        if not callable(clean_sampler):
            raise TypeError("NS clean sampler must be callable")
        if perturbation_sampler is not None and not callable(perturbation_sampler):
            raise TypeError("NS perturbation sampler must be callable")
        if isinstance(max_points_per_request, bool) or int(max_points_per_request) < 1:
            raise ValueError("NS per-request point limit must be positive")
        if not isinstance(boundary_velocity_trace, bool):
            raise TypeError("NS boundary_velocity_trace must be a boolean")
        self.generator_id = generator_id.strip()
        self.artifact_sha256 = artifact_sha256
        self._clean_sampler = clean_sampler
        self._perturbation_sampler = perturbation_sampler
        self._answer_adapter = answer_adapter or _default_answer_adapter
        self._max_points = int(max_points_per_request)
        self._boundary_velocity_trace = boundary_velocity_trace
        self.public_session_methods = NS_PUBLIC_SESSION_METHODS + (
            ("query_boundary_velocity",) if boundary_velocity_trace else ()
        )
        self._case_id_key = secrets.token_bytes(32)
        self._issued: dict[str, _ResidentAnswer] = {}
        self._lock = Lock()

    @property
    def supports_perturbations(self) -> bool:
        return self._perturbation_sampler is not None

    def sample(self, request: SampleRequest) -> tuple[CaseRecord, ...]:
        if request.kind is GenerationKind.PERTURBED:
            return self._sample_perturbed(request)
        if request.kind is not GenerationKind.BASE:
            raise ValueError("NS fixed source accepts only BASE or PERTURBED")
        seeds = tuple(
            _child_seed(request.seed, "clean", index)
            for index in range(request.count)
        )
        sample_many = getattr(self._clean_sampler, "sample_many", None)
        if callable(sample_many):
            sampled_answers = tuple(sample_many(seeds))
            if len(sampled_answers) != request.count:
                raise ValueError("NS clean batch sampler returned the wrong count")
        else:
            sampled_answers = tuple(self._clean_sampler(seed) for seed in seeds)
        staged = []
        for index, (seed, sampled) in enumerate(
            zip(seeds, sampled_answers, strict=True)
        ):
            raw_answer, answer, metadata = self._adapt_sample(sampled)
            case_id = _case_id(request, seed, index, key=self._case_id_key)
            case = self._build_case(
                case_id=case_id,
                request=request,
                answer=answer,
                source=CaseSource.GENERATED,
                parent_case_id=None,
                perturbation_metadata=metadata,
            )
            staged.append(self._resident(case, raw_answer, answer, request.cohort))
        return self._commit(staged)

    def issue_answer_first_records(
        self,
        raw_answers: Sequence[object],
        *,
        round_index: int,
        visibility: CaseVisibility,
        case_id_prefix: str,
        controller_seed: int,
        distribution_lineage: Mapping[str, object],
        case_source: CaseSource = CaseSource.GENERATED,
    ) -> tuple[CaseRecord, ...]:
        """Register externally sampled answers behind the same trusted ABI.

        A concrete frozen Guidance backend may use this controller-private
        hook after it samples continuous fields.  This method does not train,
        open, or claim a Guidance model by itself. ``case_source=INITIAL``
        registers experimental-derived originals outside any particular round.
        """

        from pde_agent_verifier_solver.sdk.macro_loop import CohortKind, SampleRequest

        answers = tuple(raw_answers)
        if not answers:
            raise ValueError("external NS answer-first batch cannot be empty")
        if (
            isinstance(round_index, bool)
            or not isinstance(round_index, int)
            or round_index < 0
        ):
            raise ValueError("external NS answer-first round must be nonnegative")
        if not isinstance(visibility, CaseVisibility):
            raise TypeError("external NS answer-first visibility is invalid")
        if not isinstance(case_source, CaseSource) or case_source not in (
            CaseSource.INITIAL, CaseSource.GENERATED,
        ):
            raise ValueError("external NS answer-first source must be INITIAL or GENERATED")
        prefix = str(case_id_prefix).strip()
        if not prefix or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
            for character in prefix
        ):
            raise ValueError("external NS answer-first case prefix is invalid")
        if (
            isinstance(controller_seed, bool)
            or not isinstance(controller_seed, int)
            or controller_seed < 0
        ):
            raise ValueError("external NS controller seed must be nonnegative")
        lineage = dict(
            _plain_mapping(distribution_lineage, "distribution lineage")
        )
        request = SampleRequest(
            round_index=round_index,
            cohort=(
                CohortKind.DEVELOPMENT
                if visibility is CaseVisibility.DEVELOPMENT
                else CohortKind.SELECTION
            ),
            seed=controller_seed,
            count=len(answers),
            kind=GenerationKind.BASE,
        )
        staged = []
        for index, sampled in enumerate(answers):
            raw_answer, answer, _metadata = self._adapt_sample(sampled)
            seed = _child_seed(controller_seed, prefix, index)
            case = self._build_case(
                case_id=_case_id(request, seed, index, key=self._case_id_key, namespace=prefix),
                request=request,
                answer=answer,
                source=case_source,
                parent_case_id=None,
                perturbation_metadata={},
            )
            case = Case(
                case_id=case.case_id,
                source=case.source,
                visibility=case.visibility,
                public_data={**case.public_data, "distribution_lineage": lineage},
                round_index=(None if case_source is CaseSource.INITIAL else case.round_index),
                tags=(*case.tags, "externally-issued"),
            )
            staged.append(self._resident(case, raw_answer, answer, request.cohort))
        return self._commit(staged)

    def register_existing_answer(
        self, case: Case, raw_answer: object, *, cohort: CohortKind,
    ) -> CaseRecord:
        """Restore a prepared Case and answer without changing its id or parent relation."""

        from pde_agent_verifier_solver.sdk.macro_loop import CohortKind, SampleRequest

        if not isinstance(case, Case) or case.source not in (
            CaseSource.INITIAL, CaseSource.GENERATED, CaseSource.PERTURBED,
        ):
            raise ValueError("NS existing answer needs an original, generated or perturbed Case")
        if (case.source is CaseSource.PERTURBED) != (case.parent_case_id is not None):
            raise ValueError("NS existing answer has an inconsistent parent relation")
        if not isinstance(cohort, CohortKind):
            raise TypeError("NS existing answer cohort must be a CohortKind")
        raw, answer, metadata = self._adapt_sample(raw_answer)
        if case.source is CaseSource.PERTURBED:
            with self._lock:
                parent = self._issued.get(case.parent_case_id)
            if parent is None or parent.case.source is CaseSource.PERTURBED:
                raise ValueError("NS restored perturbation needs its active root parent")
            if parent.case.visibility is not case.visibility:
                raise ValueError("NS restored perturbation changed its parent visibility")
            _require_same_problem(parent.answer, answer)
            metadata = _plain_mapping(case.public_data.get("perturbation"), "restored perturbation")
            permitted = _SAFE_PERTURBATION_PUBLIC_FIELDS | {
                "randomized_realization", "realization_seed_exposed", "signed_amplitude_exposed",
            }
            if set(metadata) - permitted:
                raise ValueError("NS restored perturbation contains undeclared public fields")
        expected = self._build_case(
            case_id=case.case_id,
            request=SampleRequest(case.round_index or 0, cohort, seed=0, count=1),
            answer=answer,
            source=case.source,
            parent_case_id=case.parent_case_id,
            perturbation_metadata=metadata,
        )
        public_data = dict(case.public_data)
        lineage = public_data.pop("distribution_lineage", None)
        if lineage is not None:
            _plain_mapping(lineage, "existing NS distribution lineage")
        if case.visibility != expected.visibility or public_data != dict(expected.public_data):
            raise ValueError("NS existing Case differs from its manufactured answer contract")
        with self._lock:
            resident = self._issued.get(case.case_id)
            if resident is not None:
                if resident.case != case or resident.raw_answer is not raw or resident.cohort is not cohort:
                    raise ValueError("NS existing answer attempted to change an active Case")
                return CaseRecord(resident.case, _NSVelocityTruth(resident.answer))
        return self._commit((self._resident(case, raw, answer, cohort),))[0]

    def _sample_perturbed(
        self, request: SampleRequest
    ) -> tuple[CaseRecord, ...]:
        if self._perturbation_sampler is None:
            raise ValueError("NS source has no perturbation backend")
        staged = []
        with self._lock:
            parents = tuple(self._issued.get(anchor.case_id) for anchor in request.anchors)
        if any(parent is None for parent in parents):
            raise ValueError("NS perturbation anchor is not an active issued case")
        for index, (anchor, parent) in enumerate(zip(request.anchors, parents)):
            assert parent is not None
            if parent.case != anchor:
                raise ValueError("NS perturbation anchor snapshot changed")
            seed = _child_seed(request.seed, anchor.case_id, index)
            sampled = self._perturbation_sampler(parent.raw_answer, seed)
            raw_answer, answer, metadata = self._adapt_sample(sampled)
            _require_same_problem(parent.answer, answer)
            case_id = _case_id(request, seed, index, key=self._case_id_key)
            public_metadata, private_audit = _split_perturbation_metadata(
                metadata,
                realization_seed=seed,
            )
            case = self._build_case(
                case_id=case_id,
                request=request,
                answer=answer,
                source=CaseSource.PERTURBED,
                parent_case_id=anchor.case_id,
                perturbation_metadata=public_metadata,
            )
            staged.append(
                self._resident(
                    case,
                    raw_answer,
                    answer,
                    request.cohort,
                    perturbation_audit=private_audit,
                )
            )
        return self._commit(staged)

    def _adapt_sample(
        self, sampled: object
    ) -> tuple[object, ContinuousNSAnswer, Mapping[str, object]]:
        metadata: Mapping[str, object] = {}
        raw_answer = sampled
        if isinstance(sampled, StructuredNSPerturbationResult):
            raw_answer = sampled.answer
            metadata = sampled.metadata
        answer = self._answer_adapter(raw_answer)
        _validate_continuous_answer(answer)
        return raw_answer, answer, _plain_mapping(metadata, "perturbation metadata")

    def _build_case(
        self,
        *,
        case_id: str,
        request: SampleRequest,
        answer: ContinuousNSAnswer,
        source: CaseSource,
        parent_case_id: str | None,
        perturbation_metadata: Mapping[str, object],
    ) -> Case:
        from pde_agent_verifier_solver.sdk.macro_loop import CohortKind

        x_min, x_max, y_min, y_max = answer.domain_bounds
        t0, t1 = answer.time_interval
        boundary = _answer_boundary_condition(
            answer, allow_trace=self._boundary_velocity_trace,
        )
        if self._boundary_velocity_trace:
            _boundary_polygons(answer.public_geometry, answer.domain_bounds)
        public_data = {
            "equation": "2D incompressible Navier--Stokes",
            "operator": {
                "operator_id": NS_OPERATOR_ID,
                "form": (
                    "d_t velocity + velocity dot grad velocity = "
                    "-grad pressure + viscosity laplacian velocity + forcing"
                ),
                "constraint": "divergence(velocity) = 0",
            },
            "parameters": {"viscosity": float(answer.viscosity)},
            "domain": {
                "coordinate_order": ["t", "x", "y"],
                "time_interval": [float(t0), float(t1)],
                "bounds": {
                    "x": [float(x_min), float(x_max)],
                    "y": [float(y_min), float(y_max)],
                },
                "geometry": dict(
                    _plain_mapping(answer.public_geometry, "public geometry")
                ),
            },
            "boundary_condition": boundary,
            "initial_condition": {
                "field": ["velocity_x", "velocity_y"],
                "access": "query_initial_velocity",
                "derivation": "hidden_answer_evaluated_at_initial_time",
            },
            "forcing": {
                "field": ["forcing_x", "forcing_y"],
                "access": "query_forcing",
                "derivation": "public_NS_operator_applied_to_hidden_answer",
            },
            "oracle_protocol": {
                "coordinate_policy": "agent_selected",
                "limits": {
                    "max_points_per_request": self._max_points,
                    "session_total_point_quota": None,
                },
            },
            "answer_first": {
                "contract": "continuous_ns_answer_first_v1",
                "solution_chosen_before_question": True,
                "forcing_rederived_after_perturbation": True,
                "initial_data_rederived_after_perturbation": True,
                "boundary_data_rederived_after_perturbation": True,
            },
            "continuous_answer_metadata": dict(
                _plain_mapping(
                    answer.public_metadata, "continuous answer metadata"
                )
            ),
            "perturbation": (
                None
                if source is not CaseSource.PERTURBED
                else dict(
                    _plain_mapping(
                        perturbation_metadata, "perturbation metadata"
                    )
                )
            ),
        }
        visibility = (
            CaseVisibility.DEVELOPMENT
            if request.cohort is CohortKind.DEVELOPMENT
            else CaseVisibility.HIDDEN
        )
        return Case(
            case_id=case_id,
            source=source,
            visibility=visibility,
            public_data=public_data,
            round_index=request.round_index,
            parent_case_id=parent_case_id,
            tags=("ns", "answer-first", source.value),
        )

    def _resident(
        self,
        case: Case,
        raw_answer: object,
        answer: ContinuousNSAnswer,
        cohort: CohortKind,
        *,
        perturbation_audit: Mapping[str, object] | None = None,
    ) -> _ResidentAnswer:
        return _ResidentAnswer(
            case=case,
            raw_answer=raw_answer,
            answer=answer,
            cohort=cohort,
            perturbation_audit=_plain_mapping(
                perturbation_audit or {}, "private perturbation audit"
            ),
        )

    def _commit(
        self, staged: Sequence[_ResidentAnswer]
    ) -> tuple[CaseRecord, ...]:
        case_ids = tuple(row.case.case_id for row in staged)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("NS sampler returned duplicate case ids")
        with self._lock:
            if any(case_id in self._issued for case_id in case_ids):
                raise ValueError("NS sampler attempted to reuse an active case id")
            self._issued.update({row.case.case_id: row for row in staged})
        return tuple(
            CaseRecord(row.case, _NSVelocityTruth(row.answer)) for row in staged
        )

    def resolve_active_records(
        self, case_ids: Sequence[str]
    ) -> tuple[CaseRecord, ...]:
        """Resolve issued identities for a controller-private Guidance adapter."""

        identifiers = tuple(case_ids)
        if not identifiers or any(
            not isinstance(case_id, str) or not case_id
            for case_id in identifiers
        ):
            raise ValueError("resolved NS case ids must be nonempty strings")
        with self._lock:
            missing = [
                case_id for case_id in identifiers if case_id not in self._issued
            ]
            if missing:
                raise KeyError(f"NS issued answers are no longer active: {missing}")
            rows = tuple(self._issued[case_id] for case_id in identifiers)
        return tuple(
            CaseRecord(row.case, _NSVelocityTruth(row.answer)) for row in rows
        )

    def resolve_guidance_feedback_answers(
        self, case_ids: Sequence[str]
    ) -> tuple[object, ...]:
        """Read active feedback answers only inside controller sealing."""

        from pde_agent_verifier_solver.sdk.macro_loop import CohortKind

        identifiers = tuple(case_ids)
        if not identifiers or any(
            not isinstance(case_id, str) or not case_id for case_id in identifiers
        ):
            raise ValueError("NS Guidance feedback ids must be nonempty strings")
        with self._lock:
            missing = [case_id for case_id in identifiers if case_id not in self._issued]
            if missing:
                raise KeyError(f"NS Guidance feedback answers unavailable: {missing}")
            rows = tuple(self._issued[case_id] for case_id in identifiers)
            allowed = {
                CohortKind.GUIDANCE_FEEDBACK,
                CohortKind.PROMOTION,
            }
            if any(row.cohort not in allowed for row in rows):
                raise ValueError(
                    "NS Guidance sealing accepts only feedback or Official "
                    "promotion cases"
                )
        return tuple(row.raw_answer for row in rows)

    def controller_perturbation_audit(self, case_id: str) -> Mapping[str, object]:
        """Return private realization data to trusted controller code only."""

        if not isinstance(case_id, str) or not case_id:
            raise ValueError("NS perturbation audit case id must be nonempty")
        with self._lock:
            resident = self._issued.get(case_id)
            if resident is None:
                raise KeyError(f"NS issued answer is no longer active: {case_id}")
            if resident.case.source is not CaseSource.PERTURBED:
                raise ValueError("NS perturbation audit requires a perturbed case")
            audit = dict(resident.perturbation_audit)
        return MappingProxyType(audit)

    def release_cases(self, case_ids: tuple[str, ...]) -> None:
        identifiers = tuple(case_ids)
        if any(not isinstance(case_id, str) or not case_id for case_id in identifiers):
            raise ValueError("released NS case ids must be nonempty strings")
        with self._lock:
            for case_id in dict.fromkeys(identifiers):
                self._issued.pop(case_id, None)

    def public_session_factory(self, case: Case) -> object:
        resident = self._active_resident(case)
        return _NSPublicSession(
            case,
            resident.answer.evaluate_forcing,
            resident.answer.evaluate_velocity,
            resident.answer.time_interval[0],
            resident.answer.domain_bounds,
            resident.answer.time_interval,
            self._max_points,
            boundary_geometry=(
                resident.answer.public_geometry if self._boundary_velocity_trace else None
            ),
        )

    def development_solution_session_factory(self, case: Case) -> object:
        if case.visibility is not CaseVisibility.DEVELOPMENT:
            raise PermissionError(
                "NS completed-solution access is DEVELOPMENT-only"
            )
        resident = self._active_resident(case)
        return _NSDevelopmentSolutionSession(
            case,
            resident.answer.evaluate_velocity,
            resident.answer.domain_bounds,
            resident.answer.time_interval,
            self._max_points,
        )

    def _active_resident(self, case: Case) -> _ResidentAnswer:
        if not isinstance(case, Case):
            raise TypeError("NS numerical session requires a Case")
        with self._lock:
            resident = self._issued.get(case.case_id)
        if resident is None:
            raise ValueError("NS numerical session requires an active issued case")
        if resident.case != case:
            raise ValueError("NS numerical session public case snapshot changed")
        return resident


class _NSVelocityTruth:
    def __init__(self, answer: ContinuousNSAnswer) -> None:
        self._answer = answer

    def answer(self, query: AnswerQuery) -> AnswerView:
        import numpy as np

        if set(query.fields) != set(NS_VELOCITY_FIELDS):
            raise ValueError("NS truth provides both velocity components")
        if query.kind is QueryKind.POINTS:
            points = np.asarray(
                [
                    [float(point[name]) for name in ("t", "x", "y")]
                    for point in query.points
                ],
                dtype=np.float64,
            )
            velocity = _finite_vector(
                self._answer.evaluate_velocity(points), len(points), "velocity"
            )
            return AnswerView(
                kind=query.kind,
                fields={
                    "velocity_x": tuple(velocity[:, 0].tolist()),
                    "velocity_y": tuple(velocity[:, 1].tolist()),
                },
                points=query.points,
                shape=(len(points),),
            )

        resolution = {
            name: int(query.resolution[name]) for name in ("t", "x", "y")
        }
        x_min, x_max, y_min, y_max = self._answer.domain_bounds
        t0, t1 = self._answer.time_interval
        axes = {
            "t": np.linspace(t0, t1, resolution["t"]),
            "x": np.linspace(x_min, x_max, resolution["x"]),
            "y": np.linspace(y_min, y_max, resolution["y"]),
        }
        tt, xx, yy = np.meshgrid(
            axes["t"], axes["x"], axes["y"], indexing="ij"
        )
        points = np.stack((tt, xx, yy), axis=-1).reshape(-1, 3)
        velocity = _finite_vector(
            self._answer.evaluate_velocity(points), len(points), "velocity"
        )
        shape = tuple(resolution[name] for name in ("t", "x", "y"))
        return AnswerView(
            kind=query.kind,
            fields={
                "velocity_x": tuple(velocity[:, 0].tolist()),
                "velocity_y": tuple(velocity[:, 1].tolist()),
            },
            axes={name: tuple(values.tolist()) for name, values in axes.items()},
            shape=shape,
        )


class _NSPointSession:
    def __init__(
        self,
        case: Case,
        domain_bounds,
        time_interval,
        max_points: int,
    ) -> None:
        self.case_id = case.case_id
        self._domain_bounds = tuple(domain_bounds)
        self._time_interval = tuple(time_interval)
        self._max_points = int(max_points)
        self._closed = False
        self._query_calls = 0
        self._queried_points = 0
        self._wall_seconds = 0.0
        self._lock = Lock()

    def _run(self, points, *, width: int, evaluator):
        with self._lock:
            if self._closed:
                raise RuntimeError("NS numerical session is closed")
            physical = _physical_points(
                points,
                width=width,
                domain_bounds=self._domain_bounds,
                time_interval=self._time_interval,
            )
            if len(physical) > self._max_points:
                raise ValueError("NS oracle request point count exceeds its limit")
            self._query_calls += 1
            self._queried_points += len(physical)
            started = perf_counter()
            try:
                return _finite_vector(
                    evaluator(physical), len(physical), "NS oracle response"
                )
            finally:
                self._wall_seconds += perf_counter() - started

    @property
    def usage_snapshot(self) -> Mapping[str, object]:
        with self._lock:
            return MappingProxyType(
                {
                    "query_calls": self._query_calls,
                    "queried_points": self._queried_points,
                    "backend_wall_seconds": self._wall_seconds,
                    "cost_unit": "evaluated_physical_point",
                }
            )

    def close(self) -> None:
        with self._lock:
            self._closed = True


class _NSPublicSession(_NSPointSession):
    def __init__(
        self,
        case,
        forcing_evaluator,
        velocity_evaluator,
        t0,
        domain_bounds,
        time_interval,
        max_points,
        *,
        boundary_geometry=None,
    ) -> None:
        super().__init__(case, domain_bounds, time_interval, max_points)
        self._boundary_polygons = (
            _boundary_polygons(boundary_geometry, domain_bounds)
            if boundary_geometry is not None else None
        )
        self._forcing_evaluator = forcing_evaluator
        self._velocity_evaluator = velocity_evaluator
        owner = getattr(forcing_evaluator, "__self__", None)
        built_in = (
            type(owner) is StructuredNSMMSAnswer
            and getattr(velocity_evaluator, "__self__", None) is owner
            and getattr(forcing_evaluator, "__func__", None) is StructuredNSMMSAnswer.evaluate_forcing
            and getattr(velocity_evaluator, "__func__", None) is StructuredNSMMSAnswer.evaluate_velocity
        )
        evaluation_resource = None
        if built_in:
            from .evaluation_devices import public_evaluation_answer

            owner, evaluation_resource = public_evaluation_answer(case.case_id, owner)
        self.public_calls_thread_safe = bool(
            built_in and owner._thread_safe_field and owner.raw_answer.device.type in {"cpu", "cuda"}
        )
        if built_in:
            # _run validates exactly once at the public session boundary.
            self._forcing_evaluator = owner._evaluate_validated_forcing
            self._velocity_evaluator = owner._evaluate_validated_velocity
        self._t0 = float(t0)
        self._evaluation_resource = evaluation_resource
        if evaluation_resource is not None:
            evaluation_resource.attach(self)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._evaluation_resource is not None:
                self._forcing_evaluator = None
                self._velocity_evaluator = None
                self._evaluation_resource = None

    def query_forcing(self, points):
        return self._run(
            points, width=3, evaluator=self._forcing_evaluator
        )

    def query_initial_velocity(self, points):
        import numpy as np

        def evaluate_initial(spatial):
            physical = np.column_stack(
                (np.full(len(spatial), self._t0), spatial)
            )
            return self._velocity_evaluator(physical)

        return self._run(points, width=2, evaluator=evaluate_initial)

    def query_boundary_velocity(self, points):
        if self._boundary_polygons is None:
            raise PermissionError("NS boundary trace oracle is not enabled for this source")

        def evaluate_boundary(physical):
            boundary = _boundary_points(
                physical, self._domain_bounds, self._boundary_polygons,
            )
            return self._velocity_evaluator(boundary)

        return self._run(points, width=3, evaluator=evaluate_boundary)


class _NSDevelopmentSolutionSession(_NSPointSession):
    def __init__(
        self,
        case,
        velocity_evaluator,
        domain_bounds,
        time_interval,
        max_points,
    ) -> None:
        super().__init__(case, domain_bounds, time_interval, max_points)
        self._velocity_evaluator = velocity_evaluator
        owner = getattr(velocity_evaluator, "__self__", None)
        built_in = (
            type(owner) is StructuredNSMMSAnswer
            and getattr(velocity_evaluator, "__func__", None) is StructuredNSMMSAnswer.evaluate_velocity
        )
        self.public_calls_thread_safe = bool(
            built_in and owner._thread_safe_field and owner.raw_answer.device.type in {"cpu", "cuda"}
        )
        if built_in:
            self._velocity_evaluator = owner._evaluate_validated_velocity

    def query_completed_velocity(self, points):
        return self._run(
            points, width=3, evaluator=self._velocity_evaluator
        )


def _is_read_only_structured_field(field: object) -> bool:
    """Opt in only the audited built-in field with fixed, read-only coefficients."""
    if type(field).__module__ != "agent_verifier.neural_generator.structured_ns_answer_first":
        return False
    from agent_verifier.neural_generator.structured_ns_answer_first import ContinuousStructuredNSMMS

    if type(field) is not ContinuousStructuredNSMMS:
        return False
    if any(name in vars(field) for name in (
        "velocity_pressure", "quantities", "psi_pressure", "_psi_pressure_with_masks",
        "geometry_masks", "_normalize_coordinates",
    )):
        return False
    tensors = (field.coefficients.psi, field.coefficients.pressure)
    if field.canonical_geometry is not None:
        tensors += (field.canonical_geometry,)
    # Trained parameter graphs could be shared across backward calls. Issued
    # manufactured fields use fixed coefficients and independent query graphs.
    return all(not value.requires_grad and value.grad_fn is None for value in tensors)


def _answer_boundary_condition(answer, *, allow_trace: bool) -> dict[str, object]:
    """Read the optional answer-owned boundary declaration; preserve legacy zero walls."""

    declaration = getattr(answer, "boundary_condition", None)
    if declaration is None:
        declaration = {"kind": "no_slip_velocity", "velocity": [0.0, 0.0]}
    if not isinstance(declaration, Mapping):
        raise TypeError("NS answer boundary_condition must be a mapping")
    kind = declaration.get("kind")
    if kind == "manufactured_velocity_trace":
        if not allow_trace:
            raise ValueError("NS manufactured boundary trace needs boundary_velocity_trace=True")
        if declaration.get("access", "query_boundary_velocity") != "query_boundary_velocity":
            raise ValueError("NS manufactured boundary trace has an invalid access method")
        return {
            "kind": "manufactured_velocity_trace",
            "access": "query_boundary_velocity",
            "derivation": "trace_of_hidden_continuous_answer",
        }
    if kind == "no_slip_velocity":
        if tuple(declaration.get("velocity", (0.0, 0.0))) != (0.0, 0.0):
            raise ValueError("NS no-slip boundary must have zero velocity")
        return {
            "kind": "no_slip_velocity",
            "velocity": [0.0, 0.0],
            "derivation": "trace_of_hidden_continuous_answer",
        }
    raise ValueError(f"unsupported NS answer boundary kind: {kind!r}")


def _boundary_polygons(geometry, domain_bounds):
    """Validate the public polygon edges used by the optional trace oracle."""

    import numpy as np

    if not isinstance(geometry, Mapping):
        raise TypeError("NS boundary geometry must be a mapping")
    polygons = geometry.get("obstacle_polygons", geometry.get("obstacles", ()))
    if not isinstance(polygons, (tuple, list)):
        raise ValueError("NS boundary obstacle_polygons must be a sequence")
    x0, x1, y0, y1 = domain_bounds
    validated = []
    for polygon in polygons:
        values = np.asarray(polygon, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] != 2:
            raise ValueError("NS boundary polygons need at least three (x,y) vertices")
        if not np.all(np.isfinite(values)):
            raise ValueError("NS boundary polygon vertices must be finite")
        if np.any(values < (x0, y0)) or np.any(values > (x1, y1)):
            raise ValueError("NS boundary polygon leaves the declared domain")
        next_vertices = np.roll(values, -1, axis=0)
        if np.any(np.sum((next_vertices - values) ** 2, axis=1) == 0.0):
            raise ValueError("NS boundary polygon has a zero-length edge")
        twice_area = np.sum(values[:, 0] * next_vertices[:, 1] - values[:, 1] * next_vertices[:, 0])
        if twice_area == 0.0:
            raise ValueError("NS boundary polygon must enclose nonzero area")
        validated.append(values)
    return tuple(validated)


def _boundary_points(physical, domain_bounds, polygons):
    """Accept only declared edges and snap floating-point edge roundoff to the edge."""

    import numpy as np

    x0, x1, y0, y1 = domain_bounds
    rectangle = np.asarray(((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
    positions = physical[:, 1:]
    snapped = physical.copy()
    on_boundary = np.zeros(len(physical), dtype=bool)
    solid_interior = np.zeros(len(physical), dtype=bool)
    # This is floating-point roundoff, not an exposed strip of interior solution values.
    tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, *(abs(v) for v in domain_bounds))
    for index, polygon in enumerate((rectangle, *polygons)):
        on_polygon = np.zeros(len(physical), dtype=bool)
        inside = np.zeros(len(physical), dtype=bool)
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
            delta = end - start
            parameter = ((positions - start) @ delta) / float(delta @ delta)
            projection = start + np.clip(parameter, 0.0, 1.0)[:, None] * delta
            on_edge = np.linalg.norm(positions - projection, axis=1) <= tolerance
            snapped[on_edge, 1:] = projection[on_edge]
            on_polygon |= on_edge
            if delta[1] != 0.0:
                crosses = (start[1] > positions[:, 1]) != (end[1] > positions[:, 1])
                intersection = start[0] + (positions[:, 1] - start[1]) * delta[0] / delta[1]
                inside ^= crosses & (positions[:, 0] < intersection)
        on_boundary |= on_polygon
        if index:
            solid_interior |= inside & ~on_polygon
    if np.any(~on_boundary | solid_interior):
        raise ValueError("NS boundary trace requires points on declared fluid boundary edges")
    return snapped


def _default_answer_adapter(raw_answer: object) -> ContinuousNSAnswer:
    required = (
        "domain_bounds",
        "time_interval",
        "viscosity",
        "public_geometry",
        "public_metadata",
        "evaluate_velocity",
        "evaluate_forcing",
    )
    if all(hasattr(raw_answer, name) for name in required):
        return raw_answer  # type: ignore[return-value]
    return StructuredNSMMSAnswer(raw_answer)


def _validate_continuous_answer(answer: ContinuousNSAnswer) -> None:
    bounds = tuple(float(value) for value in answer.domain_bounds)
    interval = tuple(float(value) for value in answer.time_interval)
    if len(bounds) != 4 or len(interval) != 2:
        raise ValueError("continuous NS answer has invalid public domain")
    if bounds[1] <= bounds[0] or bounds[3] <= bounds[2] or interval[1] <= interval[0]:
        raise ValueError("continuous NS answer has an empty public domain")
    if not isfinite(float(answer.viscosity)) or float(answer.viscosity) < 0.0:
        raise ValueError("continuous NS answer has invalid viscosity")
    if not callable(answer.evaluate_velocity) or not callable(answer.evaluate_forcing):
        raise TypeError("continuous NS answer needs velocity and forcing evaluators")
    _plain_mapping(answer.public_geometry, "public geometry")
    _plain_mapping(answer.public_metadata, "public metadata")


def _require_same_problem(
    parent: ContinuousNSAnswer, child: ContinuousNSAnswer
) -> None:
    if (
        tuple(child.domain_bounds) != tuple(parent.domain_bounds)
        or tuple(child.time_interval) != tuple(parent.time_interval)
        or float(child.viscosity) != float(parent.viscosity)
        or _plain_mapping(child.public_geometry, "child public geometry")
        != _plain_mapping(parent.public_geometry, "parent public geometry")
    ):
        raise ValueError("NS perturbation changed the physical problem domain")


def _physical_points(
    points,
    *,
    width: int,
    domain_bounds,
    time_interval,
):
    import numpy as np

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != width:
        raise ValueError(f"NS physical points must have shape [N,{width}]")
    if not np.isfinite(values).all():
        raise ValueError("NS physical points must be finite")
    x_min, x_max, y_min, y_max = (float(value) for value in domain_bounds)
    if width == 3:
        t0, t1 = (float(value) for value in time_interval)
        t, x, y = values.T
        if np.any(t < t0) or np.any(t > t1):
            raise ValueError("NS physical time leaves the public interval")
    else:
        x, y = values.T
    if np.any(x < x_min) or np.any(x > x_max):
        raise ValueError("NS physical x leaves the public domain")
    if np.any(y < y_min) or np.any(y > y_max):
        raise ValueError("NS physical y leaves the public domain")
    return values


def _finite_vector(values, count: int, label: str):
    import numpy as np

    result = np.asarray(values, dtype=np.float64)
    if result.shape != (int(count), 2) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite [N,2] array")
    return result


def _to_numpy(value):
    tensor = value
    detach = getattr(tensor, "detach", None)
    if callable(detach):
        tensor = detach()
    cpu = getattr(tensor, "cpu", None)
    if callable(cpu):
        tensor = cpu()
    numpy = getattr(tensor, "numpy", None)
    return numpy() if callable(numpy) else tensor


def _plain_mapping(value: Mapping[str, object], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    try:
        plain = json.loads(
            json.dumps(dict(value), sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain finite JSON data") from exc
    return MappingProxyType(plain)


_SAFE_PERTURBATION_PUBLIC_FIELDS = frozenset(
    {
        "family",
        "amplitude_magnitude",
        "online_calibration_or_audit",
        "smooth_continuous_answer",
        "spectral_support_complement",
        "coefficient_support_complement",
        "complement_inner_product",
        "amplitude_convention",
        "amplitude_equivalent_to_cosine",
    }
)


def _split_perturbation_metadata(
    metadata: Mapping[str, object],
    *,
    realization_seed: int,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    private = dict(_plain_mapping(metadata, "private perturbation metadata"))
    private["seed"] = int(realization_seed)
    public = {
        key: private[key]
        for key in _SAFE_PERTURBATION_PUBLIC_FIELDS
        if key in private
    }
    public.update(
        {
            "randomized_realization": True,
            "realization_seed_exposed": False,
            "signed_amplitude_exposed": False,
        }
    )
    return (
        _plain_mapping(public, "public perturbation metadata"),
        _plain_mapping(private, "private perturbation metadata"),
    )


def _child_seed(seed: int, label: str, index: int) -> int:
    payload = f"ns:{int(seed)}:{label}:{int(index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _case_id(request: SampleRequest, seed: int, index: int, *, key: bytes, namespace: str = "sample") -> str:
    payload = json.dumps(
        {
            "version": 1,
            "namespace": str(namespace),
            "round_index": request.round_index,
            "cohort": request.cohort.value,
            "kind": request.kind.value,
            "request_seed": int(request.seed),
            "realization_seed": int(seed),
            "index": int(index),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"ns-case-{hmac.new(key, payload, hashlib.sha256).hexdigest()}"


__all__ = [
    "ContinuousNSAnswer",
    "LegacyStructuredNSSmoothPerturber",
    "StructuredNSMMSAnswer",
    "StructuredNSManufacturedCaseSource",
    "StructuredNSPerturbationResult",
]
