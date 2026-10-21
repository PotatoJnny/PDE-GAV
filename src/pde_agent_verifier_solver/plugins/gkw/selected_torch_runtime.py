"""Concrete bridge to the selected Torch GKW generator and physical oracle.

The external physics package owns sampling, decoding, smooth perturbations,
the nonlinear operator, and admissibility.  This adapter keeps only bounded
CPU resident answers plus contracted macro truth between calls.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hmac
from pathlib import Path
import re
import secrets
import shutil
import tempfile
from threading import Lock, RLock
from time import perf_counter
from types import FunctionType, MappingProxyType, MethodType, ModuleType
from typing import Any, Mapping, Sequence

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    Case,
    CaseSource,
    CaseVisibility,
    GenerationRequest,
    QueryKind,
    RecoverableSamplingRejection,
)

from .contract import (
    GKW_MACRO_FIELDS,
    gkw_answer_view_from_pairs,
    gkw_point_query_coordinates,
)
from .manufactured_backend import (
    GKWManufacturedCase,
    GKWManufacturingAudit,
    GKW_QUALIFIED_PERTURBATION_SEED_MODULUS,
    _generation_cohort,
)
from .public_contract import (
    canonical_gkw_public_metadata,
    gkw_json_values_equal_exact,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DECODER_HARD_CHECKS = frozenset(
    {
        "finite",
        "decode_log_std_clipping",
        "pathological_case_correction",
        "exact_moment_projection",
    }
)
_DEVELOPMENT_PANEL_PROFILE = "development_sampled_include_t0_v1"
_FORMAL_PANEL_PROFILE = "old_strength_full_field_exclude_t0_v1"
_SELECTED_GENERATOR_SCHEMA = (
    "fusion_gkw_selected_generator_full_rank_continuous_condition_codes_v3"
)
_PUBLIC_CONDITION_NAMES = ("rlt_i", "rln_i", "shat", "q")
GKW_SOLUTION_SAMPLING_MODES = ("local_edit", "direct_noise")
_SOLUTION_SAMPLING_MODES = frozenset(GKW_SOLUTION_SAMPLING_MODES)
_COHORT_PANEL_PROFILES = MappingProxyType(
    {
        "development": _DEVELOPMENT_PANEL_PROFILE,
        "development_screen": _DEVELOPMENT_PANEL_PROFILE,
        "selection": _FORMAL_PANEL_PROFILE,
        "promotion": _FORMAL_PANEL_PROFILE,
        "terminal": _FORMAL_PANEL_PROFILE,
        "guidance_feedback": _FORMAL_PANEL_PROFILE,
    }
)


def _empty_manufacturing_telemetry() -> dict[str, int | float]:
    """Create controller-private online-path counters without public exposure."""

    return {
        "clean_generation_calls": 0,
        "clean_generation_cases": 0,
        "clean_generation_wall_seconds": 0.0,
        "perturbed_generation_calls": 0,
        "perturbed_generation_cases": 0,
        "perturbed_generation_wall_seconds": 0.0,
        "perturbed_sample_wall_seconds": 0.0,
        "perturbed_operator_precompute_wall_seconds": 0.0,
        "perturbed_macro_truth_host_dispatch_wall_seconds": 0.0,
        "perturbed_macro_truth_to_public_pairs_wall_seconds": 0.0,
        "clean_audited_observations": 0,
        "clean_strict_nonlinear_audits_derived": 0,
        "clean_operator_frame_evaluations_derived": 0,
        "clean_decoder_rejections_before_physical_audit": 0,
        "perturbed_online_numerical_audits": 0,
        "perturbed_online_parameter_hashes": 0,
        "perturbed_failed_seed_replacements": 0,
        "unified_serialization_calls": 0,
        "unified_state_hash_calls": 0,
        "unified_replay_calls": 0,
        "resident_release_calls": 0,
        "resident_answers_released": 0,
        "resident_backend_releases": 0,
    }


class _DecoderGateRejection(RuntimeError):
    """A well-formed stochastic sample failed selected-decoder quality gates."""



def _assert_cpu_tensor_graph(value: object, *, label: str) -> None:
    """Fail closed if retained controller state reaches a non-CPU tensor.

    Manufacturing may use CUDA, but completed truth and resident handles must
    not keep transient device allocations alive. This walks object metadata
    only; it never scans tensor values or serializes an answer.
    """

    import numpy as np
    import torch

    atomic = (str, bytes, bytearray, int, float, complex, bool, type(None), Path)
    seen: set[int] = set()
    pending: list[tuple[str, object]] = [(label, value)]
    while pending:
        path, current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(current, torch.Tensor):
            if current.device.type != "cpu":
                raise RuntimeError(
                    f"{label} retains a non-CPU tensor at {path}: "
                    f"device={current.device}"
                )
            if (
                current.requires_grad
                or not current.is_contiguous()
                or current.storage_offset() != 0
            ):
                raise RuntimeError(
                    f"{label} retains a non-owning tensor at {path}: "
                    "autograd, contiguity, or storage offset is unsafe"
                )
            logical_bytes = current.numel() * current.element_size()
            storage_bytes = current.untyped_storage().nbytes()
            if storage_bytes > logical_bytes:
                raise RuntimeError(
                    f"{label} retains an oversized tensor view at {path}: "
                    f"logical_bytes={logical_bytes}, storage_bytes={storage_bytes}"
                )
            continue
        if isinstance(current, atomic) or isinstance(current, ModuleType):
            continue
        if isinstance(current, np.ndarray):
            if current.dtype.hasobject:
                pending.extend(
                    (f"{path}[{index}]", item)
                    for index, item in enumerate(current.flat)
                )
            continue
        if isinstance(current, Mapping):
            pending.extend(
                (f"{path}[{key!r}]", item) for key, item in current.items()
            )
            continue
        if isinstance(current, (tuple, list, set, frozenset)):
            pending.extend(
                (f"{path}[{index}]", item)
                for index, item in enumerate(current)
            )
            continue
        if isinstance(current, MethodType):
            pending.append((f"{path}.__self__", current.__self__))
            pending.append((f"{path}.__func__", current.__func__))
            continue
        if isinstance(current, FunctionType):
            pending.extend(
                (f"{path}.__defaults__", item)
                for item in (current.__defaults__ or ())
            )
            pending.extend(
                (f"{path}.__kwdefaults__[{key!r}]", item)
                for key, item in (current.__kwdefaults__ or {}).items()
            )
            pending.extend(
                (f"{path}.__closure__[{index}]", cell.cell_contents)
                for index, cell in enumerate(current.__closure__ or ())
            )
            continue
        attributes = getattr(current, "__dict__", None)
        if isinstance(attributes, Mapping):
            pending.extend(
                (f"{path}.{name}", item) for name, item in attributes.items()
            )
        slots = getattr(type(current), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name not in {"__dict__", "__weakref__"} and hasattr(current, name):
                pending.append((f"{path}.{name}", getattr(current, name)))


@dataclass(frozen=True)
class _SelectedSymbols:
    prepare_cases: Any
    streaming_bridge: Any
    selected_generator_schema: str
    validated_production_qualification: Any
    qualification_clean_maximum_attempts: int
    qualification_clean_retry_seed: Any
    retention_policy: Any
    build_macro_truth: Any
    build_perturbation_macro_truth: Any
    oracle_limits: Any
    oracle_protocol: Any
    capture_resident_answer: Any
    build_qualified_perturbed_observation: Any
    build_oracle_backend_from_resident: Any
    validated_direct_noise_compatibility: Any | None = None
    build_direct_noise_perturbed_observation: Any | None = None


def _load_selected_symbols() -> _SelectedSymbols:
    """Import the optional physical package only when this runtime is built."""

    try:
        from agent_verifier.fusion_simulation_first.gkw_generated_pde_bridge import (
            build_gkw_qualified_perturbed_observation_from_resident,
            build_gkw_numerical_oracle_backend_from_resident_answer,
            capture_gkw_resident_answer,
            prepare_generated_bridge_cases,
        )
        from agent_verifier.fusion_simulation_first.gkw_macro_contract import (
            build_structured_gkw_macro_truth,
        )
        from agent_verifier.fusion_simulation_first.gkw_perturbed_macro_truth import (
            build_gkw_qualified_perturbation_macro_truth,
        )
        from agent_verifier.fusion_simulation_first.gkw_perturbation_qualification import (
            GKW_QUALIFICATION_CLEAN_MAXIMUM_ATTEMPTS,
            GKWValidatedProductionQualification,
            gkw_qualification_clean_retry_seed,
        )
        from agent_verifier.fusion_simulation_first.gkw_numerical_oracle import (
            GKWOracleLimits,
            gkw_numerical_oracle_protocol,
        )
        from agent_verifier.fusion_simulation_first.gkw_online_bridge import (
            GKWOnlineRetentionPolicy,
            GKWStreamingProjectedWindowBridge,
        )
        from agent_verifier.fusion_simulation_first.gkw_selected_generator import (
            GKW_SELECTED_GENERATOR_SCHEMA,
        )
    except ImportError as exc:
        raise RuntimeError(
            "GKWSelectedTorchRuntime requires the installed selected-GKW "
            "physical runtime"
        ) from exc
    try:
        from agent_verifier.fusion_simulation_first.gkw_direct_noise_compatibility import (
            GKWValidatedDirectNoiseCompatibility,
        )
        from agent_verifier.fusion_simulation_first.gkw_generated_pde_bridge import (
            build_gkw_experimental_direct_noise_perturbed_observation_from_resident,
        )
    except ImportError:
        # Older physical installs remain valid for the default local-edit mode.
        GKWValidatedDirectNoiseCompatibility = None
        build_gkw_experimental_direct_noise_perturbed_observation_from_resident = (
            None
        )
    return _SelectedSymbols(
        prepare_cases=prepare_generated_bridge_cases,
        streaming_bridge=GKWStreamingProjectedWindowBridge,
        selected_generator_schema=GKW_SELECTED_GENERATOR_SCHEMA,
        validated_production_qualification=GKWValidatedProductionQualification,
        qualification_clean_maximum_attempts=(
            GKW_QUALIFICATION_CLEAN_MAXIMUM_ATTEMPTS
        ),
        qualification_clean_retry_seed=gkw_qualification_clean_retry_seed,
        retention_policy=GKWOnlineRetentionPolicy,
        build_macro_truth=build_structured_gkw_macro_truth,
        build_perturbation_macro_truth=(
            build_gkw_qualified_perturbation_macro_truth
        ),
        oracle_limits=GKWOracleLimits,
        oracle_protocol=gkw_numerical_oracle_protocol,
        build_qualified_perturbed_observation=(
            build_gkw_qualified_perturbed_observation_from_resident
        ),
        capture_resident_answer=capture_gkw_resident_answer,
        build_oracle_backend_from_resident=(
            build_gkw_numerical_oracle_backend_from_resident_answer
        ),
        validated_direct_noise_compatibility=(
            GKWValidatedDirectNoiseCompatibility
        ),
        build_direct_noise_perturbed_observation=(
            build_gkw_experimental_direct_noise_perturbed_observation_from_resident
        ),
    )



@dataclass(frozen=True)
class _ResidentHandle:
    """Controller-private CPU answer retained only for its owning cohort."""

    case_id: str
    resident: object
    anchor_keys: tuple[Mapping[str, int], ...]
    sampled_public_condition: tuple[float, float, float, float]
    case_prefix: str
    perturbed: bool
    answer_id: str
    parent_answer_id: str | None = None
    arrays: Mapping[str, object] | None = None
    scalars: Mapping[str, object] | None = None
    truth: object | None = None
    completed_macro_evaluator: object | None = None
    solution_sampling_mode: str = "local_edit"
    sampling_provenance: Mapping[str, object] | None = None


def _require_clean_sampling_provenance(
    handle: _ResidentHandle,
    *,
    expected: str,
) -> Mapping[str, object]:
    """Validate the private clean-root mode before anchored generation."""

    provenance = handle.sampling_provenance
    if (
        handle.perturbed
        or handle.solution_sampling_mode != expected
        or not isinstance(provenance, Mapping)
        or provenance.get("schema")
        != "gkw_unified_solution_sampling_provenance_v1"
        or provenance.get("solution_sampling_mode") != expected
        or provenance.get("inherited_from_clean_parent") is not False
        or provenance.get("parent_answer_id") is not None
    ):
        raise RuntimeError(
            "GKW perturbation parent changed clean sampling provenance"
        )
    return provenance


def _sampled_public_condition(value: object) -> tuple[float, float, float, float]:
    """Copy one public condition from a generated sample into CPU scalars."""

    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        raw = value.detach().to(device="cpu").numpy()
    else:
        raw = np.asarray(value)
    array = np.asarray(raw, dtype=np.float64)
    if array.shape == (1, len(_PUBLIC_CONDITION_NAMES)):
        array = array[0]
    if (
        array.shape != (len(_PUBLIC_CONDITION_NAMES),)
        or not bool(np.isfinite(array).all())
    ):
        raise RuntimeError("GKW generated public condition is invalid")
    return tuple(float(item) for item in array)


def _require_solution_sampling_mode(
    value: object,
    *,
    expected: str,
    label: str,
) -> str:
    """Fail before decoding if generated codes came from another sampler."""

    if expected not in _SOLUTION_SAMPLING_MODES:
        raise ValueError("GKW expected solution sampling mode is invalid")
    mode = getattr(value, "solution_sampling_mode", None)
    if mode != expected:
        raise RuntimeError(
            f"{label} solution sampling mode is {mode!r}, expected {expected!r}"
        )
    return str(mode)


def _require_inherited_public_geometry(
    observation: object,
    *,
    expected_case: object,
    parent_handle: _ResidentHandle,
) -> None:
    """Require a perturbed answer to remain on its clean parent's PDE grid."""

    try:
        observed_case = observation.case
        observed_packet = observed_case.public_packet
        expected_packet = expected_case.public_packet
        observed_times = tuple(float(value) for value in observed_case.time_nodes)
        expected_times = tuple(float(value) for value in expected_case.time_nodes)
        observed_shape = tuple(
            int(value) for value in observation.state.contract.grid.state_shape
        )
        parent_shape = tuple(
            int(value)
            for value in parent_handle.resident.state.contract.grid.state_shape
        )
        parent_public = parent_handle.truth.public_data
        parent_times = tuple(
            float(value) for value in parent_public["metadata"]["time_nodes"]
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "GKW perturbation changed the parent public PDE/domain/grid"
        ) from exc
    if (
        observed_case.case_id != expected_case.case_id
        or observed_packet.arrays is not expected_packet.arrays
        or observed_packet.scalars is not expected_packet.scalars
        or not gkw_json_values_equal_exact(
            observed_packet.metadata, expected_packet.metadata
        )
        or observed_times != expected_times
        or observed_times != parent_times
        or observed_shape != parent_shape
    ):
        raise RuntimeError(
            "GKW perturbation changed the parent public PDE/domain/grid"
        )


def _packet_public_condition(
    packet: object,
) -> tuple[float, float, float, float]:
    """Read the four continuous operator coefficients from one public packet."""

    import numpy as np

    scalars = packet if isinstance(packet, Mapping) else getattr(
        packet, "scalars", None
    )
    if not isinstance(scalars, Mapping):
        raise RuntimeError("GKW public packet lacks scalar coefficients")
    values = []
    for name in _PUBLIC_CONDITION_NAMES:
        if name not in scalars:
            raise RuntimeError(
                f"GKW public packet lacks continuous coefficient {name}"
            )
        value = np.asarray(scalars[name], dtype=np.float64)
        if value.shape != () or not bool(np.isfinite(value)):
            raise RuntimeError(
                f"GKW public packet coefficient {name} is invalid"
            )
        values.append(float(value))
    return tuple(values)


def _require_packet_public_condition(
    packet: object,
    expected: tuple[float, float, float, float],
) -> None:
    """Fail closed unless the rebuilt operator uses the sampled coefficients."""

    actual = _packet_public_condition(packet)
    if actual != expected:
        raise RuntimeError(
            "GKW public packet changed sampled continuous coefficients: "
            f"expected={expected!r}, actual={actual!r}"
        )


def _owned_cpu_tensor(value: object, *, label: str):
    """Return a compact owning CPU copy without numerically re-auditing it."""

    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} is not a tensor")
    return value.detach().to(device="cpu").contiguous().clone()


@dataclass(frozen=True)
class _CPUResidentStructuredMacroTruth:
    """CPU-only macro cache bound to the qualified resident clean state."""

    state: object
    control_fields: Mapping[str, object]
    construction_audit: Mapping[str, object]

    @classmethod
    def capture(cls, transient: object, resident_state: object):
        fields = getattr(transient, "control_fields", None)
        audit = getattr(transient, "construction_audit", None)
        if not isinstance(fields, Mapping) or set(fields) != set(GKW_MACRO_FIELDS):
            raise RuntimeError("GKW transient structured macro truth is incomplete")
        if not isinstance(audit, Mapping):
            raise RuntimeError("GKW structured macro truth lacks its audit")
        result = cls(
            state=resident_state,
            control_fields=MappingProxyType(
                {
                    name: _owned_cpu_tensor(
                        fields[name], label=f"GKW clean macro field {name}"
                    )
                    for name in GKW_MACRO_FIELDS
                }
            ),
            construction_audit=MappingProxyType(dict(audit)),
        )
        return result

    def evaluate_pairs(self, times, field_indices) -> Mapping[str, object]:
        import numpy as np
        import torch

        query = np.asarray(times, dtype=np.float64).reshape(-1)
        indices = np.asarray(field_indices, dtype=np.int64)
        if indices.ndim != 2 or indices.shape[1] != 3:
            raise ValueError("GKW macro field indices must have shape [N,3]")
        first = next(iter(self.control_fields.values()))
        device_query = torch.as_tensor(
            query.copy(), device="cpu", dtype=first.real.dtype
        )
        interpolation = self.state.axis_interpolation_matrix("t", device_query)
        device_indices = torch.as_tensor(
            indices.copy(), device="cpu", dtype=torch.int64
        )
        columns = tuple(device_indices[:, axis] for axis in range(3))
        return MappingProxyType(
            {
                name: torch.view_as_real(
                    torch.einsum(
                        "qt,tsxy->qsxy",
                        interpolation.to(value.dtype),
                        value,
                    )[:, columns[0], columns[1], columns[2]]
                )
                .to(dtype=torch.float64)
                .numpy()
                .copy()
                for name, value in self.control_fields.items()
            }
        )


@dataclass(frozen=True)
class _CPUResidentPerturbationMacroTruth:
    """CPU-only additive cache bound to the qualified resident perturbation."""

    perturbation: object
    rank_fields: Mapping[str, object]
    construction_audit: Mapping[str, object]

    @classmethod
    def capture(cls, transient: object, resident_perturbation: object):
        fields = getattr(transient, "rank_fields", None)
        audit = getattr(transient, "construction_audit", None)
        if not isinstance(fields, Mapping) or set(fields) != set(GKW_MACRO_FIELDS):
            raise RuntimeError("GKW transient perturbation macro truth is incomplete")
        if not isinstance(audit, Mapping):
            raise RuntimeError("GKW perturbation macro truth lacks its audit")
        result = cls(
            perturbation=resident_perturbation,
            rank_fields=MappingProxyType(
                {
                    name: _owned_cpu_tensor(
                        fields[name], label=f"GKW perturbation macro field {name}"
                    )
                    for name in GKW_MACRO_FIELDS
                }
            ),
            construction_audit=MappingProxyType(dict(audit)),
        )
        return result

    def evaluate(self, times, field_indices) -> Mapping[str, object]:
        import numpy as np
        import torch

        indices = np.asarray(field_indices, dtype=np.int64)
        if indices.ndim != 2 or indices.shape[1] != 3:
            raise ValueError("GKW macro field indices must have shape [N,3]")
        first = next(iter(self.rank_fields.values()))
        time_values = self.perturbation.time_values(times).to(
            device="cpu", dtype=first.dtype
        )
        device_indices = torch.as_tensor(
            indices.copy(), device="cpu", dtype=torch.int64
        )
        columns = tuple(device_indices[:, axis] for axis in range(3))
        return MappingProxyType(
            {
                name: torch.einsum("tr,rsxy->tsxy", time_values, value)[
                    :, columns[0], columns[1], columns[2]
                ]
                for name, value in self.rank_fields.items()
            }
        )


class _StructuredCompletedMacroEvaluator:
    """Private arbitrary-coordinate view of one retained clean answer."""

    def __init__(self, structured: object) -> None:
        evaluate = getattr(structured, "evaluate_pairs", None)
        if not callable(evaluate):
            raise TypeError("GKW structured macro truth needs evaluate_pairs")
        self._structured = structured
        self._lock = Lock()

    def evaluate_pairs(self, times, field_indices) -> Mapping[str, object]:
        with self._lock:
            evaluated = self._structured.evaluate_pairs(times, field_indices)
        return _normalized_macro_pairs(evaluated, times, field_indices)


class _PerturbedCompletedMacroEvaluator:
    """Compose a clean evaluator with one qualified smooth perturbation."""

    def __init__(self, parent: object, additive: object) -> None:
        if not callable(getattr(parent, "evaluate_pairs", None)):
            raise TypeError("GKW perturbation parent needs completed macro access")
        if not callable(getattr(additive, "evaluate", None)):
            raise TypeError("GKW perturbation macro truth needs evaluate")
        self._parent = parent
        self._additive = additive
        self._lock = Lock()

    def evaluate_pairs(self, times, field_indices) -> Mapping[str, object]:
        parent = self._parent.evaluate_pairs(times, field_indices)
        with self._lock:
            additive = self._additive.evaluate(times, field_indices)
        added = _normalized_macro_pairs(additive, times, field_indices)
        import numpy as np

        return MappingProxyType(
            {
                name: _readonly_array(
                    np.asarray(parent[name], dtype=np.float64)
                    + np.asarray(added[name], dtype=np.float64)
                )
                for name in GKW_MACRO_FIELDS
            }
        )


@dataclass(frozen=True)
class _CompletedMacroTruth:
    """Controller-private arbitrary-coordinate view of one completed answer."""

    case_id: str
    public_data: Mapping[str, object]
    evaluator: object

    def __post_init__(self) -> None:
        if not callable(getattr(self.evaluator, "evaluate_pairs", None)):
            raise TypeError(
                "GKW completed macro truth needs an arbitrary-coordinate evaluator"
            )

    def answer(self, query: AnswerQuery):
        import numpy as np

        case = Case(
            case_id=self.case_id,
            source=CaseSource.GENERATED,
            visibility=CaseVisibility.HIDDEN,
            public_data=self.public_data,
            tags=("gkw", "controller-private-truth"),
        )
        if query.kind is not QueryKind.POINTS:
            raise ValueError("generated GKW truth requires explicit POINTS")
        answer_times, answer_indices = gkw_point_query_coordinates(
            case,
            query,
            require_native_panel=False,
        )
        selected = {
            name: np.empty((len(answer_times), 2), dtype=np.float64)
            for name in GKW_MACRO_FIELDS
        }
        positions_by_time: dict[float, list[int]] = {}
        for position, time_value in enumerate(answer_times):
            positions_by_time.setdefault(float(time_value), []).append(position)

        for time_value, positions in positions_by_time.items():
            position_array = np.asarray(positions, dtype=np.int64)
            evaluated = self.evaluator.evaluate_pairs(
                np.asarray([time_value], dtype=np.float64),
                answer_indices[position_array],
            )
            for name in GKW_MACRO_FIELDS:
                selected[name][position_array] = np.asarray(
                    evaluated[name], dtype=np.float64
                )[0]
        return gkw_answer_view_from_pairs(
            query,
            selected,
            times=answer_times,
            field_indices=answer_indices,
        )


class _ResidentPublicSession:
    """Public equation session backed by one metered resident oracle."""

    def __init__(self, runtime: "GKWSelectedTorchRuntime", case: Case) -> None:
        self._runtime = runtime
        self._case = case
        self.case_id = case.case_id
        self.arrays = case.public_data["arrays"]
        self.scalars = case.public_data["scalars"]
        self.metadata = case.public_data["metadata"]
        self.oracle_protocol = case.public_data["oracle_protocol"]
        interval = _public_oracle_interval(self.oracle_protocol)
        self._time_min, self._time_max = interval
        self._phase_shape = _public_oracle_grid_shape(
            self.oracle_protocol,
            key="phase_grid_shape",
            width=5,
            label="phase",
        )
        self._field_shape = _public_oracle_grid_shape(
            self.oracle_protocol,
            key="field_grid_shape",
            width=3,
            label="field",
        )
        limits = self.oracle_protocol["limits"]
        self._max_rows = int(limits["max_rows_per_request"])
        # The physical oracle still has legacy per-session safety counters.
        # Roll over that cheap session transparently instead of treating them
        # as Agent-visible campaign quotas.
        self._rollover_rows = int(limits["max_total_rows"])
        self._rollover_source_times = int(
            limits["max_unique_source_times"]
        )
        self._underlying_rows = 0
        self._underlying_source_times: set[float] = set()
        self._underlying_session_rollovers = 0
        self._total_rows = 0
        self._query_calls = 0
        self._query_wall_seconds = 0.0
        self._oracle = None
        self._closed = False
        self._oracle_lock = Lock()
        self._usage_lock = Lock()
        self._query_lock = Lock()

    def _reserve(self, row_count: int, *, source_times=None) -> None:
        import numpy as np

        self._require_open()
        if row_count <= 0 or row_count > self._max_rows:
            raise ValueError("GKW oracle request row count exceeds its limit")
        requested_source_times: set[float] = set()
        if source_times is not None:
            values = np.asarray(source_times, dtype=np.float64)
            if values.ndim == 0:
                values = np.full(row_count, float(values), dtype=np.float64)
            elif values.shape != (row_count,):
                raise ValueError("GKW source times must be scalar or shape [N]")
            if not np.isfinite(values).all():
                raise ValueError("GKW source times must be finite")
            requested_source_times = {
                float(value) for value in np.unique(values)
            }
            if len(requested_source_times) > self._rollover_source_times:
                raise ValueError(
                    "GKW source request has too many unique times for one "
                    "backend-safe batch"
                )
        prospective_times = (
            self._underlying_source_times | requested_source_times
        )
        if (
            self._underlying_rows + row_count > self._rollover_rows
            or len(prospective_times) > self._rollover_source_times
        ):
            self._rollover_underlying_session()
            prospective_times = requested_source_times
        self._underlying_rows += row_count
        self._underlying_source_times = prospective_times
        with self._usage_lock:
            self._total_rows += row_count
            self._query_calls += 1

    def _rollover_underlying_session(self) -> None:
        with self._runtime._oracle_backend_lock:
            with self._oracle_lock:
                oracle = self._oracle
                self._oracle = None
            self._underlying_rows = 0
            self._underlying_source_times = set()
            if oracle is None:
                return
            close = getattr(oracle, "close", None)
            if callable(close):
                close()
            with self._usage_lock:
                self._underlying_session_rollovers += 1

    def _oracle_instance(self):
        with self._oracle_lock:
            self._require_open()
            if self._oracle is None:
                self._oracle = self._runtime._open_oracle(self._case)
            return self._oracle

    def close(self) -> None:
        with self._query_lock:
            with self._runtime._oracle_backend_lock:
                with self._oracle_lock:
                    if self._closed:
                        return
                    self._closed = True
                    oracle = self._oracle
                    self._oracle = None
                close = getattr(oracle, "close", None) if oracle is not None else None
                if callable(close):
                    close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("GKW public oracle session is closed")

    @property
    def usage_snapshot(self) -> Mapping[str, object]:
        """Return observational usage; it is not a rejection budget."""

        with self._usage_lock:
            return MappingProxyType(
                {
                    "query_calls": self._query_calls,
                    "queried_rows": self._total_rows,
                    "backend_wall_seconds": self._query_wall_seconds,
                    "underlying_session_rollovers": (
                        self._underlying_session_rollovers
                    ),
                    "cost_unit": "queried_operator_row",
                }
            )

    def _record_query_wall(self, started: float) -> None:
        with self._usage_lock:
            self._query_wall_seconds += perf_counter() - started

    def query_source(self, times, phase_indices):
        indices = _public_oracle_indices(
            phase_indices,
            shape=self._phase_shape,
            label="phase_indices",
        )
        rows = len(indices)
        _public_oracle_times(
            times,
            rows=rows,
            lower=self._time_min,
            upper=self._time_max,
        )
        with self._query_lock:
            with self._runtime._oracle_backend_lock:
                started = perf_counter()
                try:
                    self._reserve(rows, source_times=times)
                    result = self._oracle_instance().query_source(
                        times, indices
                    )
                    return _normalized_public_oracle_rows(
                        result, rows=rows, label="source"
                    )
                finally:
                    self._record_query_wall(started)

    def query_initial_h(self, phase_indices):
        indices = _public_oracle_indices(
            phase_indices,
            shape=self._phase_shape,
            label="phase_indices",
        )
        rows = len(indices)
        with self._query_lock:
            with self._runtime._oracle_backend_lock:
                started = perf_counter()
                try:
                    self._reserve(rows)
                    result = self._oracle_instance().query_initial_h(
                        indices
                    )
                    return _normalized_public_oracle_rows(
                        result, rows=rows, label="initial_h"
                    )
                finally:
                    self._record_query_wall(started)

    def query_initial_phi(self, field_indices):
        indices = _public_oracle_indices(
            field_indices,
            shape=self._field_shape,
            label="field_indices",
        )
        rows = len(indices)
        with self._query_lock:
            with self._runtime._oracle_backend_lock:
                started = perf_counter()
                try:
                    self._reserve(rows)
                    result = self._oracle_instance().query_initial_phi(
                        indices
                    )
                    return _normalized_public_oracle_rows(
                        result, rows=rows, label="initial_phi"
                    )
                finally:
                    self._record_query_wall(started)


class _ResidentDevelopmentSolutionSession:
    """Metered DEVELOPMENT-only access to a retained completed answer."""

    def __init__(self, case: Case, evaluator: object) -> None:
        import numpy as np

        if not callable(getattr(evaluator, "evaluate_pairs", None)):
            raise TypeError("GKW completed-solution evaluator is unavailable")
        protocol = case.public_data["oracle_protocol"]
        limits = protocol["limits"]
        interval = np.asarray(
            protocol["time_interval_closed"], dtype=np.float64
        )
        field_shape = np.asarray(protocol["field_grid_shape"], dtype=np.int64)
        if interval.shape != (2,) or field_shape.shape != (3,):
            raise ValueError("GKW completed-solution public domain is invalid")
        self.case_id = case.case_id
        self._evaluator = evaluator
        self._time_min = float(interval[0])
        self._time_max = float(interval[1])
        self._field_shape = field_shape.copy()
        self._max_points = int(limits["max_rows_per_request"])
        self._total_points = 0
        self._query_calls = 0
        self._query_wall_seconds = 0.0
        self._closed = False
        self._lock = Lock()

    def query_completed_macros(self, times, field_indices):
        import numpy as np

        requested_times = np.asarray(times, dtype=np.float64)
        raw_indices = np.asarray(field_indices)
        if requested_times.ndim != 1 or requested_times.size < 1:
            raise ValueError("GKW completed-solution times must have shape [T]")
        if not np.isfinite(requested_times).all():
            raise ValueError("GKW completed-solution times must be finite")
        if (
            np.any(requested_times < self._time_min)
            or np.any(requested_times > self._time_max)
        ):
            raise ValueError("GKW completed-solution times leave the public domain")
        if (
            raw_indices.ndim != 2
            or raw_indices.shape[0] < 1
            or raw_indices.shape[1] != 3
            or raw_indices.dtype.kind not in "iu"
        ):
            raise ValueError(
                "GKW completed-solution field_indices must be int64 [N,3]"
            )
        indices = np.asarray(raw_indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= self._field_shape[None, :]):
            raise ValueError(
                "GKW completed-solution field_indices leave the public grid"
            )
        point_count = int(requested_times.size) * int(indices.shape[0])
        with self._lock:
            if self._closed:
                raise RuntimeError("GKW completed-solution session is closed")
            if point_count > self._max_points:
                raise ValueError(
                    "GKW completed-solution request point count exceeds its limit"
                )
            self._total_points += point_count
            self._query_calls += 1
            evaluator = self._evaluator
        started = perf_counter()
        try:
            return evaluator.evaluate_pairs(requested_times, indices)
        finally:
            elapsed = perf_counter() - started
            with self._lock:
                self._query_wall_seconds += elapsed

    @property
    def usage_snapshot(self) -> Mapping[str, object]:
        """Return observational usage; it is not a rejection budget."""

        with self._lock:
            return MappingProxyType(
                {
                    "query_calls": self._query_calls,
                    "evaluated_points": self._total_points,
                    "backend_wall_seconds": self._query_wall_seconds,
                    "cost_unit": "evaluated_time_field_point",
                }
            )

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._evaluator = None


class GKWSelectedTorchRuntime:
    """Use the selected GKW Generator through the unified two-operation ABI."""

    def __init__(
        self,
        service: object,
        *,
        implementation_sha256: str,
        production_qualification: object | None = None,
        experimental_direct_noise_compatibility: object | None = None,
        expected_solution_sampling_mode: str = "local_edit",
        private_root: str | Path | None = None,
        private_storage_root: str | Path = "/scratch",
        transient_storage_root: str | Path = "/dev/shm",
        macro_time_count: int = 6,
        macro_field_rows: int = 2048,
        formal_macro_time_count: int = 4,
        maximum_sampling_attempts: int = 8,
        oracle_max_total_rows: int = 20_000_000,
        enforce_distribution_calibration: bool = True,
    ) -> None:
        if not isinstance(implementation_sha256, str) or not _SHA256.fullmatch(
            implementation_sha256
        ):
            raise ValueError("GKW selected runtime needs an artifact/config digest")
        required = (
            "sampler",
            "decoder",
            "config",
            "calibrated",
            "archived_cases",
            "time_registry",
            "selected_positions",
            "window_length",
        )
        if any(not hasattr(service, name) for name in required):
            raise TypeError("GKW selected service is missing physical components")
        if not callable(getattr(service.sampler, "sample", None)) or not callable(
            getattr(service.decoder, "decode_and_project", None)
        ):
            raise TypeError("GKW selected sampler or decoder is not callable")
        decoder_config = getattr(service.decoder, "config", None)
        decoder_precision = getattr(
            decoder_config, "neural_inference_precision", None
        )
        if (
            not isinstance(decoder_precision, str)
            or not decoder_precision.strip()
        ):
            raise TypeError("GKW selected decoder precision is unavailable")
        symbols = _load_selected_symbols()
        # Only the exact, already-qualified old deployment needs this explicit
        # metadata ABI adaptation. Modern/direct-noise services remain untouched.
        from .legacy_local_edit import _ServiceView, bind_qualified_80632f_symbols
        if type(service) is _ServiceView:
            symbols = bind_qualified_80632f_symbols(
                service, symbols, production_qualification=production_qualification,
            )
        if symbols.selected_generator_schema != _SELECTED_GENERATOR_SCHEMA:
            raise RuntimeError(
                "GKW selected generator lacks continuous public conditions"
            )
        if expected_solution_sampling_mode not in _SOLUTION_SAMPLING_MODES:
            raise ValueError("GKW expected solution sampling mode is invalid")
        sampler_config = getattr(service.sampler, "config", None)
        sampler_mode = getattr(sampler_config, "sampling_mode", None)
        if sampler_mode != expected_solution_sampling_mode:
            raise RuntimeError(
                "GKW selected service sampling mode changed: "
                f"{sampler_mode!r} != {expected_solution_sampling_mode!r}"
            )
        local_edit = expected_solution_sampling_mode == "local_edit"
        if local_edit and not isinstance(
            production_qualification,
            symbols.validated_production_qualification,
        ):
            raise TypeError(
                "GKW production perturbations require validated qualification"
            )
        if local_edit and experimental_direct_noise_compatibility is not None:
            raise TypeError(
                "GKW local-edit runtime rejects direct-noise authority"
            )
        direct_type = symbols.validated_direct_noise_compatibility
        if not local_edit and (
            production_qualification is not None
            or direct_type is None
            or not isinstance(
                experimental_direct_noise_compatibility, direct_type
            )
            or not callable(symbols.build_direct_noise_perturbed_observation)
        ):
            raise TypeError(
                "GKW direct-noise perturbations require experimental "
                "compatibility, not production qualification"
            )
        if symbols.qualification_clean_maximum_attempts != 8:
            raise RuntimeError(
                "GKW physical qualification clean retry policy is not the "
                "certified eight-attempt policy"
            )
        distribution_calibration_required = local_edit
        if (
            decoder_precision != "bfloat16"
            or enforce_distribution_calibration
            is not distribution_calibration_required
        ):
            raise ValueError(
                "GKW selected runtime requires its decoder admissibility policy"
            )
        default_root = Path(service.config.private_root) / "unified_runtime"
        root = Path(private_root or default_root).expanduser().resolve()
        storage_root = Path(private_storage_root).expanduser().resolve()
        if root == storage_root or not root.is_relative_to(storage_root):
            raise ValueError(
                "GKW unified physical runtime must write below its storage root"
            )
        transient_root = Path(transient_storage_root).expanduser().resolve()
        system_tmpfs = Path("/dev/shm").resolve()
        if (
            transient_root != system_tmpfs
            and not transient_root.is_relative_to(system_tmpfs)
        ):
            raise ValueError("GKW transient storage must remain below /dev/shm")
        if not transient_root.is_dir() or transient_root.is_symlink():
            raise ValueError("GKW transient storage root must be a real tmpfs directory")
        if (
            any(
                type(value) is not int
                for value in (
                    macro_time_count,
                    macro_field_rows,
                    formal_macro_time_count,
                    maximum_sampling_attempts,
                )
            )
            or macro_time_count < 2
            or macro_field_rows < 1
            or formal_macro_time_count < 1
            or maximum_sampling_attempts < 1
        ):
            raise ValueError("GKW unified runtime counts must be positive integers")
        if (
            maximum_sampling_attempts
            != symbols.qualification_clean_maximum_attempts
        ):
            raise ValueError(
                "GKW production qualification requires "
                "maximum_sampling_attempts=8"
            )
        if (
            isinstance(oracle_max_total_rows, bool)
            or not isinstance(oracle_max_total_rows, int)
            or not 65_536 <= oracle_max_total_rows <= 20_000_000
        ):
            raise ValueError("GKW oracle total rows must be in [65536, 20000000]")
        root.mkdir(parents=True, exist_ok=True)
        self.service = service
        self.implementation_sha256 = implementation_sha256
        self.production_qualification = production_qualification
        self.experimental_direct_noise_compatibility = (
            experimental_direct_noise_compatibility
        )
        self.expected_solution_sampling_mode = expected_solution_sampling_mode
        self.decoder_precision = decoder_precision
        self.private_root = root
        self.private_storage_root = storage_root
        self.transient_storage_root = transient_root
        self._transaction_parent = (
            transient_root / "pde_agent_verifier_gkw_transactions"
        ).resolve()
        self._transaction_parent.mkdir(parents=True, exist_ok=True)
        self.macro_time_count = int(macro_time_count)
        self.macro_field_rows = int(macro_field_rows)
        self.formal_macro_time_count = int(formal_macro_time_count)
        self._oracle_backend_cache_capacity = max(
            8,
            self.macro_time_count,
            self.formal_macro_time_count,
        )
        self.maximum_sampling_attempts = int(maximum_sampling_attempts)
        self.oracle_max_total_rows = oracle_max_total_rows
        self.enforce_distribution_calibration = bool(
            enforce_distribution_calibration
        )
        self._symbols = symbols
        self._oracle_limits = self._symbols.oracle_limits(
            max_total_rows=self.oracle_max_total_rows
        )
        self._handles: dict[str, _ResidentHandle] = {}
        self._pending_resident_handles: dict[str, _ResidentHandle] = {}
        # Capacity is one backend, so every open/query/close/switch/release
        # operation shares one re-entrant lock.  A query holds this lock until
        # the physical call returns; another Case therefore cannot evict the
        # backend from underneath an in-flight session.
        self._oracle_backend_lock = RLock()
        self._oracle_backend = None
        self._oracle_backend_case_id: str | None = None
        self._oracle_backend_resident = None
        self._oracle_backend_loads = 0
        self._oracle_backend_hits = 0
        self._oracle_backend_evictions = 0
        self._peak_resident_answers = 0
        self._answer_counter = 0
        self._case_id_salt = secrets.token_bytes(32)
        self._manufacturing_telemetry = _empty_manufacturing_telemetry()
        self._lock = Lock()

    @property
    def supports_perturbations(self) -> bool:
        return True

    @property
    def private_oracle_telemetry(self) -> dict[str, int]:
        """Return controller-private capacity-one backend counters."""

        with self._oracle_backend_lock:
            performance = getattr(
                self._oracle_backend,
                "private_performance_snapshot",
                {},
            )
            if not isinstance(performance, Mapping):
                performance = {}
            with self._lock:
                resident_answers = sum(
                    isinstance(handle, _ResidentHandle)
                    for handle in self._handles.values()
                )

            def counter(name: str) -> int:
                value = performance.get(name, 0)
                return value if type(value) is int and value >= 0 else 0

            return {
                "backend_loads": self._oracle_backend_loads,
                "backend_hits": self._oracle_backend_hits,
                "backend_evictions": self._oracle_backend_evictions,
                "backend_cache_capacity": 1,
                "shared_frame_cache_capacity": (
                    self._oracle_backend_cache_capacity
                ),
                "backend_resident": int(self._oracle_backend is not None),
                "resident_answers": resident_answers,
                "peak_resident_answers": self._peak_resident_answers,
                "resident_operators": 0,
                "frame_evaluations": counter("source_frame_evaluations"),
                "frame_hits": counter("source_frame_cache_hits"),
            }

    @property
    def private_manufacturing_telemetry(self) -> dict[str, int | float]:
        """Return controller-private online manufacturing and release counters."""

        with self._lock:
            return dict(self._manufacturing_telemetry)

    def _record_generation(
        self,
        kind: str,
        *,
        wall_seconds: float,
        accepted_cases: int,
    ) -> None:
        if kind not in {"clean", "perturbed"}:
            raise ValueError("unknown GKW manufacturing telemetry kind")
        elapsed = max(0.0, float(wall_seconds))
        if type(accepted_cases) is not int or accepted_cases < 0:
            raise ValueError("accepted_cases must be a nonnegative integer")
        with self._lock:
            self._manufacturing_telemetry[f"{kind}_generation_calls"] += 1
            self._manufacturing_telemetry[f"{kind}_generation_cases"] += (
                accepted_cases
            )
            self._manufacturing_telemetry[
                f"{kind}_generation_wall_seconds"
            ] += elapsed

    def _record_perturbed_stage_timings(
        self, timings: Mapping[str, float]
    ) -> None:
        """Accumulate optional controller-private host stage timings."""

        fields = {
            "perturbation_sample_wall_seconds": (
                "perturbed_sample_wall_seconds"
            ),
            "operator_precompute_wall_seconds": (
                "perturbed_operator_precompute_wall_seconds"
            ),
            "macro_truth_wall_seconds": (
                "perturbed_macro_truth_host_dispatch_wall_seconds"
            ),
            "macro_truth_to_public_pairs_wall_seconds": (
                "perturbed_macro_truth_to_public_pairs_wall_seconds"
            ),
        }
        values = {}
        for source, target in fields.items():
            value = timings.get(source)
            if type(value) is float and value >= 0.0:
                values[target] = value
        if not values:
            return
        with self._lock:
            for name, value in values.items():
                self._manufacturing_telemetry[name] += value

    def _record_clean_audit(self, observation: object) -> None:
        """Record the audited clean callback and its derivable frame count."""

        case = getattr(observation, "case", None)
        time_nodes = getattr(case, "time_nodes", ())
        try:
            frame_count = len(time_nodes)
        except TypeError:
            frame_count = 0
        with self._lock:
            self._manufacturing_telemetry["clean_audited_observations"] += 1
            self._manufacturing_telemetry[
                "clean_strict_nonlinear_audits_derived"
            ] += 1
            self._manufacturing_telemetry[
                "clean_operator_frame_evaluations_derived"
            ] += frame_count

    def _record_clean_decoder_rejection_before_physical_audit(self) -> None:
        """Record one decoder-rejected attempt that skipped physical audit."""

        with self._lock:
            self._manufacturing_telemetry[
                "clean_decoder_rejections_before_physical_audit"
            ] += 1

    def manufacture_clean(
        self,
        base_state: object,
        request: GenerationRequest,
        seeds: tuple[int, ...],
        progress=None,
    ) -> tuple[GKWManufacturedCase, ...]:
        del base_state, progress
        started = perf_counter()
        accepted_cases = 0
        rows = []
        try:
            self._validate_request_artifact(request)
            for index, seed in enumerate(seeds):
                rows.append(
                    self._manufacture_clean_case(request, seed, index)
                )
            accepted_cases = len(rows)
            return tuple(rows)
        except BaseException:
            if rows:
                self.abort_manufactured(tuple(rows))
            raise
        finally:
            self._record_generation(
                "clean",
                wall_seconds=perf_counter() - started,
                accepted_cases=accepted_cases,
            )

    def manufacture_perturbed(
        self,
        base_state: object,
        parent: GKWManufacturedCase,
        request: GenerationRequest,
        seed: int,
        progress=None,
    ) -> GKWManufacturedCase:
        del base_state, progress
        started = perf_counter()
        accepted_cases = 0
        try:
            self._validate_request_artifact(request)
            cohort, panel_profile = _request_panel_profile(request)
            handle = parent.hidden_handle
            if not isinstance(handle, _ResidentHandle) or handle.perturbed:
                raise ValueError(
                    "GKW perturbation parent lacks a clean resident handle"
                )
            if self._handles.get(parent.case_id) is not handle:
                raise ValueError("GKW perturbation parent is not registered")
            # The offline qualification covers the first fixed-rule draw. A
            # retry here would replace a failed seed and change that certified
            # distribution, so qualified production is exactly once.
            prefix = self._opaque_case_prefix(
                kind="pert",
                seed=seed,
                discriminator=handle.answer_id,
            )
            result = self._manufacture_perturbed_case(
                handle=handle,
                prefix=prefix,
                seed=seed,
                cohort=cohort,
                panel_profile=panel_profile,
            )
            accepted_cases = 1
            return result
        finally:
            self._record_generation(
                "perturbed",
                wall_seconds=perf_counter() - started,
                accepted_cases=accepted_cases,
            )

    def _manufacture_perturbed_case(
        self,
        *,
        handle: _ResidentHandle,
        prefix: str,
        seed: int,
        cohort: str,
        panel_profile: str,
    ) -> GKWManufacturedCase:
        _require_clean_sampling_provenance(
            handle,
            expected=self.expected_solution_sampling_mode,
        )
        parent_case = getattr(handle.resident, "case", None)
        parent_packet = getattr(parent_case, "public_packet", None)
        if (
            parent_case is None
            or parent_packet is None
            or not isinstance(getattr(parent_packet, "metadata", None), Mapping)
        ):
            raise RuntimeError("GKW perturbation parent lacks its public PDE")
        case_id = f"{prefix}-0000"
        child_metadata = dict(parent_packet.metadata)
        child_metadata["case_id"] = case_id
        bridge_case = replace(
            parent_case,
            case_id=case_id,
            public_packet=replace(
                parent_packet,
                metadata=child_metadata,
            ),
        )
        private_stage_timings: dict[str, float] = {}
        try:
            if self.expected_solution_sampling_mode == "local_edit":
                perturbation_builder = (
                    self._symbols.build_qualified_perturbed_observation
                )
                authority = {"qualification": self.production_qualification}
            else:
                if handle.solution_sampling_mode != "direct_noise":
                    raise RuntimeError(
                        "GKW direct-noise perturb parent changed sampling mode"
                    )
                perturbation_builder = (
                    self._symbols.build_direct_noise_perturbed_observation
                )
                authority = {
                    "compatibility": (
                        self.experimental_direct_noise_compatibility
                    )
                }
            observation = (
                perturbation_builder(
                    handle.resident,
                    case=bridge_case,
                    **authority,
                    seed=(
                        seed % GKW_QUALIFIED_PERTURBATION_SEED_MODULUS
                    ),
                    device=self.service.sampler.device,
                    private_stage_timings=private_stage_timings,
                )
            )
            _require_inherited_public_geometry(
                observation,
                expected_case=bridge_case,
                parent_handle=handle,
            )
            result = self._from_observation(
                observation,
                case_prefix=prefix,
                expected_case_id=str(bridge_case.case_id),
                anchor_keys=handle.anchor_keys,
                sampled_public_condition=handle.sampled_public_condition,
                seed=seed,
                perturbed=True,
                parent_handle=handle,
                cohort=cohort,
                panel=None,
                panel_profile=panel_profile,
                private_stage_timings=private_stage_timings,
                solution_sampling_mode=handle.solution_sampling_mode,
            )
            try:
                self._track_pending(result)
            except BaseException as exc:
                try:
                    self._rollback_pending(result)
                except BaseException as cleanup_exc:
                    exc.add_note(
                        "GKW pending-resident rollback also failed: "
                        f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                    )
                raise
            return result
        finally:
            self._record_perturbed_stage_timings(
                private_stage_timings
            )

    def public_session_factory(self, case: Case) -> object:
        handle = self._handles.get(case.case_id)
        if handle is None:
            raise KeyError("GKW Case has no registered resident answer")
        return _ResidentPublicSession(self, case)

    def development_solution_session_factory(self, case: Case) -> object:
        """Open arbitrary-coordinate truth access only for DEVELOPMENT cases."""

        if not isinstance(case, Case):
            raise TypeError("GKW completed-solution session requires a Case")
        if case.visibility is not CaseVisibility.DEVELOPMENT:
            raise PermissionError(
                "GKW completed-solution access is DEVELOPMENT-only"
            )
        with self._lock:
            handle = self._handles.get(case.case_id)
        if not isinstance(handle, _ResidentHandle):
            raise KeyError("GKW Case has no registered resident answer")
        if (
            not isinstance(handle.truth, _CompletedMacroTruth)
            or case.public_data is not handle.truth.public_data
            or not callable(
                getattr(handle.completed_macro_evaluator, "evaluate_pairs", None)
            )
        ):
            raise ValueError(
                "GKW completed-solution case changed its resident binding"
            )
        return _ResidentDevelopmentSolutionSession(
            case, handle.completed_macro_evaluator
        )

    def _validate_request_artifact(self, request: GenerationRequest) -> None:
        if request.settings.get("fixed_generator_sha256") != self.implementation_sha256:
            raise ValueError("GKW request artifact does not match the selected runtime")
        requested_mode = request.settings.get(
            "generator_sampling_mode", "local_edit"
        )
        if requested_mode != self.expected_solution_sampling_mode:
            raise ValueError(
                "GKW request sampling mode does not match the selected runtime"
            )

    def _manufacture_clean_case(
        self,
        request: GenerationRequest,
        seed: int,
        index: int,
    ) -> GKWManufacturedCase:
        cohort, panel_profile = _request_panel_profile(request)
        last_rejection = None
        for attempt in range(self.maximum_sampling_attempts):
            sampling_seed = _retry_seed(
                seed,
                attempt,
                physical_retry_seed=(
                    self._symbols.qualification_clean_retry_seed
                ),
            )
            samples = self.service.sampler.sample(
                1,
                seed=sampling_seed,
                generator_mode="base",
            )
            _require_solution_sampling_mode(
                samples,
                expected=self.expected_solution_sampling_mode,
                label="GKW clean generated code",
            )
            prefix = self._opaque_case_prefix(
                kind="base",
                seed=seed,
                discriminator=f"{request.round_index}:{index}:{attempt}",
            )
            try:
                result = self._manufacture_clean_attempt(
                    samples=samples,
                    prefix=prefix,
                    seed=seed,
                    cohort=cohort,
                    panel_profile=panel_profile,
                )
            except _DecoderGateRejection as exc:
                last_rejection = exc
                continue
            if result is not None:
                return result
            last_rejection = None
        raise RecoverableSamplingRejection(
            "GKW clean generation exhausted physical retries",
            audit={
                "reason": "gkw_clean_admissibility_exhausted",
                "attempt_count": self.maximum_sampling_attempts,
            },
        ) from last_rejection

    def _manufacture_clean_attempt(
        self,
        *,
        samples: object,
        prefix: str,
        seed: int,
        cohort: str,
        panel_profile: str,
    ) -> GKWManufacturedCase | None:
        _require_solution_sampling_mode(
            samples,
            expected=self.expected_solution_sampling_mode,
            label="GKW clean decode input",
        )
        transaction_root = self._new_transaction_root(prefix)
        try:
            bridge_case = self._prepare_case(samples, prefix)
            result = self._decode_case(
                samples,
                bridge_case,
                case_prefix=prefix,
                seed=seed,
                cohort=cohort,
                panel_profile=panel_profile,
                run_root=transaction_root / "bridge",
            )
        except BaseException as exc:
            try:
                self._discard_transaction_root(transaction_root)
            except BaseException as cleanup_exc:
                exc.add_note(
                    "GKW transaction cleanup also failed: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
            raise
        if result is None:
            self._discard_transaction_root(transaction_root)
            return None
        try:
            self._track_pending(result)
        except BaseException as exc:
            self._rollback_pending(result)
            try:
                self._discard_transaction_root(transaction_root)
            except BaseException as cleanup_exc:
                exc.add_note(
                    "GKW transaction cleanup also failed: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
            raise
        try:
            self._discard_transaction_root(transaction_root)
        except BaseException:
            self._rollback_pending(result)
            raise
        return result

    def _prepare_case(self, samples: object, prefix: str) -> object:
        return self._prepare_case_from_anchor_keys(
            tuple(samples.anchor_keys),
            prefix,
            getattr(samples, "sampled_public_conditions", None),
        )

    def _prepare_case_from_anchor_keys(
        self,
        anchor_keys: tuple[Mapping[str, int], ...],
        prefix: str,
        sampled_public_conditions: object,
    ) -> object:
        sampled_condition = _sampled_public_condition(
            sampled_public_conditions
        )
        cases = self._symbols.prepare_cases(
            anchor_keys,
            archived_cases=self.service.archived_cases,
            time_registry=self.service.time_registry,
            sampled_public_conditions=sampled_public_conditions,
            selected_positions=tuple(self.service.selected_positions),
            window_length=int(self.service.window_length),
            case_id_prefix=prefix,
        )
        if len(cases) != 1:
            raise RuntimeError("GKW unified runtime expects one decoded case")
        _require_packet_public_condition(
            cases[0].public_packet,
            sampled_condition,
        )
        return cases[0]

    def _decode_case(
        self,
        samples: object,
        bridge_case: object,
        *,
        case_prefix: str,
        seed: int,
        cohort: str,
        panel_profile: str,
        run_root: Path,
    ) -> GKWManufacturedCase | None:
        solution_sampling_mode = _require_solution_sampling_mode(
            samples,
            expected=self.expected_solution_sampling_mode,
            label="GKW decoder input",
        )
        prepared_case_id = getattr(bridge_case, "case_id", None)
        if (
            not isinstance(case_prefix, str)
            or not case_prefix.strip()
            or not isinstance(prepared_case_id, str)
            or not prepared_case_id.strip()
        ):
            raise RuntimeError("GKW prepared case lacks a stable public identity")
        sampled_condition = _sampled_public_condition(
            getattr(samples, "sampled_public_conditions", None)
        )
        _require_packet_public_condition(
            bridge_case.public_packet,
            sampled_condition,
        )
        box: list[GKWManufacturedCase] = []

        def consume(observation):
            self._record_clean_audit(observation)
            if observation.passed:
                box.append(
                    self._from_observation(
                        observation,
                        case_prefix=case_prefix,
                        expected_case_id=prepared_case_id,
                        anchor_keys=tuple(samples.anchor_keys),
                        sampled_public_condition=sampled_condition,
                        seed=seed,
                        perturbed=False,
                        parent_handle=None,
                        panel=None,
                        cohort=cohort,
                        panel_profile=panel_profile,
                        solution_sampling_mode=solution_sampling_mode,
                    )
                )
            # The physical bridge used to JSON-round-trip this callback's
            # metadata into a transient report that is deleted immediately.
            # The resident case in ``box`` is the actual controller handoff.
            return None

        report = self._run_bridge(
            samples,
            bridge_case,
            run_root=run_root,
            callback=consume,
        )
        if report.get("passed") is not True or len(box) != 1:
            return None
        return box[0]

    def _run_bridge(
        self,
        samples: object,
        bridge_case: object,
        *,
        run_root: Path,
        callback,
    ) -> Mapping[str, object]:
        bridge = self._symbols.streaming_bridge(
            (bridge_case,),
            run_root=run_root,
            storage_root=self.transient_storage_root,
            source_config=self.service.calibrated.source,
            device=self.service.sampler.device,
            retention_policy=self._symbols.retention_policy.audit_only(),
            max_representation_relative_l2=(
                self.service.config.maximum_representation_relative_l2
            ),
            perturbation_config=None,
            hidden_support_config=self.service.calibrated.hidden_support,
            off_grid_config=self.service.calibrated.off_grid,
            require_calibrated_audits=True,
            transient_case_callback=callback,
        )
        terminal_index = int(self.service.window_length) - 1
        if terminal_index < 0:
            raise RuntimeError("GKW selected decoder has an empty window")
        missing_terminal = object()
        terminal_frame = missing_terminal
        next_frame_index = 0

        def consume_projected_frame(
            sample_index: int,
            frame_index: int,
            learning_frame: object,
        ) -> None:
            """Delay only the frame that triggers the expensive physical gate."""

            nonlocal next_frame_index, terminal_frame
            if sample_index != 0:
                raise RuntimeError(
                    "streaming GKW bridge requires sample-major decode order"
                )
            if frame_index != next_frame_index or frame_index > terminal_index:
                raise RuntimeError(
                    "streaming GKW bridge requires increasing frame order"
                )
            next_frame_index += 1
            if frame_index != terminal_index:
                bridge.consume_projected_frame(
                    sample_index,
                    frame_index,
                    learning_frame,
                )
                return
            terminal_frame = learning_frame

        decoder_report = self.service.decoder.decode_and_project(
            samples,
            projected_frame_callback=consume_projected_frame,
            public_equation_packets=(bridge_case.public_packet,),
            projected_sample_audit_callback=bridge.register_decoder_case_audit,
            enforce_distribution_calibration=(
                self.enforce_distribution_calibration
            ),
        )
        try:
            _require_decoder_report(
                decoder_report,
                enforce_distribution_calibration=(
                    self.enforce_distribution_calibration
                ),
                neural_inference_precision=self.decoder_precision,
            )
        except _DecoderGateRejection:
            self._record_clean_decoder_rejection_before_physical_audit()
            raise
        if terminal_frame is not missing_terminal:
            bridge.consume_projected_frame(
                0,
                terminal_index,
                terminal_frame,
            )
        return bridge.finalize()

    def _from_observation(
        self,
        observation: object,
        *,
        case_prefix: str,
        expected_case_id: str,
        anchor_keys: tuple[Mapping[str, int], ...],
        sampled_public_condition: tuple[float, float, float, float],
        seed: int,
        perturbed: bool,
        parent_handle: _ResidentHandle | None,
        panel: Mapping[str, object] | None,
        cohort: str,
        panel_profile: str,
        private_stage_timings: dict[str, float] | None = None,
        solution_sampling_mode: str = "local_edit",
    ) -> GKWManufacturedCase:
        import numpy as np
        del panel, cohort, panel_profile

        expected_mode = getattr(
            self, "expected_solution_sampling_mode", "local_edit"
        )
        if solution_sampling_mode != expected_mode:
            raise RuntimeError("GKW observation changed solution sampling mode")
        inherited = False
        if parent_handle is not None:
            _require_clean_sampling_provenance(
                parent_handle,
                expected=solution_sampling_mode,
            )
            inherited = True
        sampling_provenance = MappingProxyType(
            {
                "schema": "gkw_unified_solution_sampling_provenance_v1",
                "solution_sampling_mode": solution_sampling_mode,
                "inherited_from_clean_parent": inherited,
                "parent_answer_id": (
                    None if parent_handle is None else parent_handle.answer_id
                ),
            }
        )

        if getattr(observation, "passed", None) is not True:
            raise RuntimeError("GKW resident capture requires a fully audited observation")
        case_id = str(observation.case.case_id)
        if case_id != expected_case_id:
            raise RuntimeError("GKW bridge changed the prepared case identity")
        packet = observation.case.public_packet
        sampled_public_condition = _sampled_public_condition(
            sampled_public_condition
        )
        _require_packet_public_condition(packet, sampled_public_condition)
        if parent_handle is None:
            arrays = _readonly_public_arrays(packet.arrays)
            scalars = MappingProxyType(
                {
                    name: np.float64(
                        np.asarray(value, dtype=np.float64).reshape(()).item()
                    )
                    for name, value in packet.scalars.items()
                }
            )
            resident_anchor_keys = tuple(
                MappingProxyType(dict(key)) for key in anchor_keys
            )
        else:
            if (
                parent_handle.perturbed
                or parent_handle.arrays is None
                or parent_handle.scalars is None
                or not isinstance(parent_handle.truth, _CompletedMacroTruth)
                or not callable(
                    getattr(
                        parent_handle.completed_macro_evaluator,
                        "evaluate_pairs",
                        None,
                    )
                )
            ):
                raise RuntimeError("GKW perturbation parent resident is incomplete")
            arrays = parent_handle.arrays
            scalars = parent_handle.scalars
            resident_anchor_keys = parent_handle.anchor_keys

        time_nodes = tuple(float(value) for value in observation.case.time_nodes)
        metadata = MappingProxyType(
            canonical_gkw_public_metadata(
                case_id=case_id,
                time_nodes=time_nodes,
                byte_count=sum(value.nbytes for value in arrays.values()),
            )
        )
        oracle = MappingProxyType(
            dict(
                self._symbols.oracle_protocol(
                    case_id=case_id,
                    time_interval=(time_nodes[0], time_nodes[-1]),
                    grid_shape=observation.state.contract.grid.state_shape,
                    limits=self._oracle_limits,
                )
            )
        )
        public_data = MappingProxyType(
            {
                "equation": "GKW electrostatic gyrokinetics",
                "arrays": arrays,
                "scalars": scalars,
                "metadata": metadata,
                "oracle_protocol": oracle,
            }
        )

        capture_mode = (
            {"solution_sampling_mode": solution_sampling_mode}
            if parent_handle is None
            else {}
        )
        resident = self._symbols.capture_resident_answer(
            observation,
            parent=None if parent_handle is None else parent_handle.resident,
            **capture_mode,
        )
        provenance = getattr(resident, "qualification_provenance", None)
        if not isinstance(provenance, Mapping) or provenance.get("passed") is not True:
            raise RuntimeError("GKW resident answer lacks offline qualification")
        resident_state = getattr(resident, "state", None)
        resident_base_state = getattr(resident, "base_state", None)
        if resident_state is None or resident_base_state is None:
            raise RuntimeError("GKW resident answer lacks its CPU state")
        if parent_handle is None and resident_state is not resident_base_state:
            raise RuntimeError(
                "GKW clean resident state is not its qualified base state"
            )

        answer_id = self._next_answer_id(case_id)
        parent_answer_id = (
            None if parent_handle is None else parent_handle.answer_id
        )
        perturbation_id = None
        perturbation_qualification_id = None
        if parent_handle is not None:
            perturbation_evidence = provenance.get("perturbation")
            if (
                not isinstance(perturbation_evidence, Mapping)
                or perturbation_evidence.get("passed") is not True
                or not isinstance(perturbation_evidence.get("schema"), str)
            ):
                raise RuntimeError("GKW resident perturbation lacks qualification")
            perturbation_id = f"{answer_id}:perturbation"
            if expected_mode == "local_edit":
                authority_id = perturbation_evidence.get("qualification_id")
            else:
                authority_id = perturbation_evidence.get("compatibility_id")
                if (
                    perturbation_evidence.get("eligibility_basis")
                    != "experimental_direct_noise_compatibility_only"
                    or perturbation_evidence.get(
                        "clean_distribution_qualified"
                    )
                    is not False
                    or perturbation_evidence.get(
                        "production_certificate_eligible"
                    )
                    is not False
                    or perturbation_evidence.get(
                        "old_local_edit_authority_inherited"
                    )
                    is not False
                ):
                    raise RuntimeError(
                        "GKW direct-noise perturbation overstates its authority"
                    )
            if not isinstance(authority_id, str) or not authority_id.strip():
                raise RuntimeError("GKW perturbation lacks its authority id")
            perturbation_qualification_id = authority_id

        if parent_handle is None:
            transient_structured = self._symbols.build_macro_truth(
                observation.state,
                observation.operator,
                arrays,
            )
            if (
                getattr(transient_structured, "state", None)
                is not observation.state
            ):
                raise RuntimeError(
                    "GKW macro truth was not derived from its final hidden answer"
                )
            structured = _CPUResidentStructuredMacroTruth.capture(
                transient_structured, resident_base_state
            )
            del transient_structured
            completed_macro_evaluator = _StructuredCompletedMacroEvaluator(
                structured
            )
        else:
            perturbation = getattr(observation, "macro_perturbation", None)
            if perturbation is None:
                raise RuntimeError(
                    "GKW perturbation observation lacks its macro perturbation"
                )
            resident_perturbation = getattr(
                resident_state, "perturbation", None
            )
            if resident_perturbation is None:
                raise RuntimeError(
                    "GKW resident perturbation state lacks additive parameters"
                )
            transient_additive = self._symbols.build_perturbation_macro_truth(
                perturbation,
                observation.operator,
                arrays,
                private_stage_timings=private_stage_timings,
            )
            additive = _CPUResidentPerturbationMacroTruth.capture(
                transient_additive, resident_perturbation
            )
            del transient_additive
            completed_macro_evaluator = _PerturbedCompletedMacroEvaluator(
                parent_handle.completed_macro_evaluator,
                additive,
            )
            if private_stage_timings is not None:
                private_stage_timings[
                    "macro_truth_to_public_pairs_wall_seconds"
                ] = 0.0
        truth = _CompletedMacroTruth(
            case_id=case_id,
            public_data=public_data,
            evaluator=completed_macro_evaluator,
        )

        handle = _ResidentHandle(
            case_id=case_id,
            resident=resident,
            anchor_keys=resident_anchor_keys,
            sampled_public_condition=sampled_public_condition,
            case_prefix=case_prefix,
            perturbed=perturbed,
            answer_id=answer_id,
            parent_answer_id=parent_answer_id,
            arrays=arrays,
            scalars=scalars,
            truth=truth,
            completed_macro_evaluator=completed_macro_evaluator,
            solution_sampling_mode=solution_sampling_mode,
            sampling_provenance=sampling_provenance,
        )
        _assert_cpu_tensor_graph(handle, label="GKW manufactured resident handle")
        return GKWManufacturedCase(
            case_id=case_id,
            public_data=public_data,
            truth=truth,
            hidden_handle=handle,
            audit=GKWManufacturingAudit(
                seed=seed,
                answer_id=answer_id,
                source_answer_id=answer_id,
                initial_answer_id=answer_id,
                boundary_answer_id=answer_id,
                truth_answer_id=answer_id,
                answer_first=True,
                passed=True,
                parent_answer_id=parent_answer_id,
                perturbation_id=perturbation_id,
                perturbation_qualification_id=perturbation_qualification_id,
                perturbation_qualification_seed=(
                    None
                    if parent_handle is None
                    else seed % GKW_QUALIFIED_PERTURBATION_SEED_MODULUS
                ),
            ),
        )

    def _opaque_case_prefix(
        self,
        *,
        kind: str,
        seed: int,
        discriminator: str,
    ) -> str:
        """Derive a public id without exposing a realization seed."""

        if kind not in {"base", "pert"}:
            raise ValueError("GKW opaque case kind is invalid")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("GKW opaque case seed is invalid")
        if not isinstance(discriminator, str) or not discriminator:
            raise ValueError("GKW opaque case discriminator is invalid")
        salt = self._case_id_salt
        if not isinstance(salt, bytes) or len(salt) < 32:
            raise RuntimeError("GKW opaque case-id salt is unavailable")
        message = f"{kind}\0{seed}\0{discriminator}".encode("utf-8")
        token = hmac.new(salt, message, digestmod="sha256").hexdigest()[:24]
        return f"gkw-u-{kind}-{token}"

    def _next_answer_id(self, case_id: str) -> str:
        with self._lock:
            index = self._answer_counter
            self._answer_counter += 1
        return f"gkw-answer:{_safe_id(case_id)}:{index:08d}"

    def _new_transaction_root(self, label: str) -> Path:
        parent = self._transaction_parent
        if parent.is_symlink() or not parent.is_dir():
            raise RuntimeError("GKW transaction parent is unavailable")
        prefix = f"{_safe_id(label)[:48] or 'gkw'}-"
        root = Path(tempfile.mkdtemp(prefix=prefix, dir=parent)).resolve()
        self._assert_transaction_root(root)
        return root

    def _assert_transaction_root(self, root: object) -> Path:
        if not isinstance(root, Path):
            raise TypeError("GKW manufacturing transaction lacks a private root")
        parent = self._transaction_parent
        if (
            parent.is_symlink()
            or not parent.is_dir()
            or root != root.resolve()
            or root.parent != parent
            or root.is_symlink()
            or not root.is_dir()
        ):
            raise RuntimeError("GKW transaction root is unsafe or unavailable")
        return root

    @staticmethod
    def _bound_resident_handle(item: object) -> _ResidentHandle:
        if not isinstance(item, GKWManufacturedCase):
            raise TypeError("GKW runtime can only manage manufactured cases")
        handle = item.hidden_handle
        if not isinstance(handle, _ResidentHandle):
            raise TypeError("GKW runtime produced an invalid resident handle")
        provenance = handle.sampling_provenance
        if (
            handle.case_id != item.case_id
            or handle.answer_id != item.audit.answer_id
            or not isinstance(handle.case_prefix, str)
            or not handle.case_prefix.strip()
            or item.audit.answer_first is not True
            or item.audit.passed is not True
            or item.public_data.get("arrays") is not handle.arrays
            or item.public_data.get("scalars") is not handle.scalars
            or _packet_public_condition(handle.scalars)
            != handle.sampled_public_condition
            or item.truth is not handle.truth
            or handle.solution_sampling_mode not in _SOLUTION_SAMPLING_MODES
            or not isinstance(provenance, Mapping)
            or provenance.get("schema")
            != "gkw_unified_solution_sampling_provenance_v1"
            or provenance.get("solution_sampling_mode")
            != handle.solution_sampling_mode
            or provenance.get("inherited_from_clean_parent")
            is not handle.perturbed
            or provenance.get("parent_answer_id")
            != handle.parent_answer_id
            or not callable(
                getattr(handle.completed_macro_evaluator, "evaluate_pairs", None)
            )
        ):
            raise ValueError("GKW resident handle is not bound to its case")
        return handle

    def _discard_transaction_root(self, transaction_root: Path) -> None:
        root = self._assert_transaction_root(transaction_root)
        shutil.rmtree(root)

    def _track_pending(self, item: GKWManufacturedCase) -> None:
        handle = self._bound_resident_handle(item)
        with self._lock:
            if (
                item.case_id in self._pending_resident_handles
                or item.case_id in self._handles
            ):
                raise RuntimeError("GKW resident answer was already tracked")
            self._pending_resident_handles[item.case_id] = handle

    def _rollback_pending(self, item: GKWManufacturedCase) -> None:
        """Idempotently undo a partially completed private registration."""

        handle = self._bound_resident_handle(item)
        with self._lock:
            if self._pending_resident_handles.get(item.case_id) is handle:
                del self._pending_resident_handles[item.case_id]

    @staticmethod
    def _close_backend(backend: object) -> None:
        close = getattr(backend, "close", None)
        if callable(close):
            close()

    def _evict_oracle_backend(self) -> None:
        """Close then clear the capacity-one backend; caller holds its lock."""

        backend = self._oracle_backend
        if backend is None:
            return
        self._close_backend(backend)
        self._oracle_backend = None
        self._oracle_backend_case_id = None
        self._oracle_backend_resident = None
        self._oracle_backend_evictions += 1

    def commit_manufactured(
        self,
        items: tuple[GKWManufacturedCase, ...],
    ) -> None:
        """Atomically publish one backend-validated resident batch."""

        if not isinstance(items, tuple) or not items:
            raise TypeError("GKW runtime commit needs a nonempty case tuple")
        pending = {
            item.case_id: self._bound_resident_handle(item) for item in items
        }
        if len(pending) != len(items):
            raise RuntimeError("GKW runtime commit has duplicate case ids")
        for case_id, handle in pending.items():
            _assert_cpu_tensor_graph(
                handle, label=f"GKW commit resident handle {case_id}"
            )
        with self._lock:
            for case_id, handle in pending.items():
                if self._pending_resident_handles.get(case_id) is not handle:
                    raise RuntimeError("GKW resident handle is not pending")
            if set(pending) & set(self._handles):
                raise RuntimeError("GKW runtime reused a registered case id")
            next_handles = dict(self._handles)
            next_handles.update(pending)
            next_pending = dict(self._pending_resident_handles)
            for case_id in pending:
                del next_pending[case_id]
            self._handles = next_handles
            self._pending_resident_handles = next_pending
            self._peak_resident_answers = max(
                self._peak_resident_answers, len(next_handles)
            )

    def abort_manufactured(
        self,
        items: tuple[GKWManufacturedCase, ...],
    ) -> None:
        """Drop exact resident answers that have not been committed."""

        if not isinstance(items, tuple):
            raise TypeError("GKW runtime abort needs a case tuple")
        if not items:
            return
        pending = {
            item.case_id: self._bound_resident_handle(item) for item in items
        }
        if len(pending) != len(items):
            raise RuntimeError("GKW runtime abort has duplicate case ids")
        with self._lock:
            for case_id, handle in pending.items():
                if self._pending_resident_handles.get(case_id) is not handle:
                    if self._handles.get(case_id) is handle:
                        raise RuntimeError(
                            "GKW cannot abort a committed resident answer"
                        )
                    raise RuntimeError("GKW resident handle is not pending")
            for case_id in pending:
                del self._pending_resident_handles[case_id]

    def release_manufactured(self, handles: Sequence[object]) -> None:
        """Release exact active resident answers and their capacity-one backend."""

        if isinstance(handles, (str, bytes)) or not isinstance(handles, Sequence):
            raise TypeError("GKW resident release requires a handle sequence")
        requested = tuple(handles)
        if len({id(handle) for handle in requested}) != len(requested):
            raise ValueError("GKW resident release cannot repeat a handle")
        if any(not isinstance(handle, _ResidentHandle) for handle in requested):
            raise TypeError("GKW resident release received an invalid handle")

        with self._oracle_backend_lock:
            with self._lock:
                active: list[_ResidentHandle] = []
                for handle in requested:
                    registered = self._handles.get(handle.case_id)
                    if registered is None:
                        continue
                    if registered is not handle:
                        raise ValueError(
                            "GKW resident release is not bound to the active case"
                        )
                    active.append(handle)
                if not active:
                    return
                active_resident_ids = {id(handle.resident) for handle in active}
                backend_released = (
                    self._oracle_backend is not None
                    and id(self._oracle_backend_resident) in active_resident_ids
                )
                if backend_released:
                    self._evict_oracle_backend()
                next_handles = dict(self._handles)
                for handle in active:
                    del next_handles[handle.case_id]
                self._handles = next_handles
                self._manufacturing_telemetry["resident_release_calls"] += 1
                self._manufacturing_telemetry[
                    "resident_answers_released"
                ] += len(active)
                self._manufacturing_telemetry[
                    "resident_backend_releases"
                ] += int(backend_released)

    def _open_oracle(self, case: Case):
        with self._oracle_backend_lock:
            with self._lock:
                handle = self._handles.get(case.case_id)
            if not isinstance(handle, _ResidentHandle):
                raise KeyError("GKW public oracle case has no active resident answer")
            if (
                case.public_data.get("arrays") is not handle.arrays
                or case.public_data.get("scalars") is not handle.scalars
            ):
                raise ValueError("GKW issued public case changed its resident binding")
            resident = handle.resident
            cache_hit = (
                self._oracle_backend is not None
                and self._oracle_backend_case_id == case.case_id
                and self._oracle_backend_resident is resident
            )
            if cache_hit:
                backend = self._oracle_backend
                self._oracle_backend_hits += 1
            else:
                # Drop the old large backend before promoting the next case.
                self._evict_oracle_backend()
                backend = self._symbols.build_oracle_backend_from_resident(
                    resident,
                    device=self.service.sampler.device,
                    limits=self._oracle_limits,
                    source_frame_cache_entries=(
                        self._oracle_backend_cache_capacity
                    ),
                )
                self._oracle_backend_loads += 1

            try:
                oracle = backend.open_session(limits=self._oracle_limits)
            except BaseException:
                if cache_hit:
                    self._evict_oracle_backend()
                else:
                    self._close_backend(backend)
                raise
            if not gkw_json_values_equal_exact(
                oracle.protocol,
                _plain_json(case.public_data["oracle_protocol"]),
            ):
                try:
                    close = getattr(oracle, "close", None)
                    if callable(close):
                        close()
                finally:
                    if cache_hit:
                        self._evict_oracle_backend()
                    else:
                        self._close_backend(backend)
                raise ValueError("GKW resident oracle protocol changed")
            if not cache_hit:
                self._oracle_backend = backend
                self._oracle_backend_case_id = case.case_id
                self._oracle_backend_resident = resident
            return oracle


def _normalized_macro_pairs(
    evaluated: object,
    times: object,
    field_indices: object,
) -> Mapping[str, object]:
    """Normalize trusted structured output without auditing the hidden answer."""

    import numpy as np

    if (
        not isinstance(evaluated, Mapping)
        or set(evaluated) != set(GKW_MACRO_FIELDS)
    ):
        raise RuntimeError("GKW completed macro truth is incomplete")
    expected_complex_shape = (len(times), len(field_indices))
    expected_pair_shape = (*expected_complex_shape, 2)
    result = {}
    for name in GKW_MACRO_FIELDS:
        value = evaluated[name]
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
            to = getattr(value, "to", None)
            if callable(to):
                value = to(device="cpu")
            numpy = getattr(value, "numpy", None)
            if callable(numpy):
                value = numpy()
        array = np.asarray(value)
        if np.iscomplexobj(array):
            if array.shape != expected_complex_shape:
                raise RuntimeError(
                    "GKW completed macro truth has an invalid complex shape"
                )
            array = np.stack((array.real, array.imag), axis=-1)
        if array.shape != expected_pair_shape:
            raise RuntimeError(
                "GKW completed macro truth has an invalid pair shape"
            )
        pairs = np.array(array, dtype=np.float64, copy=True)
        pairs.setflags(write=False)
        result[name] = pairs
    return MappingProxyType(result)


def _public_oracle_interval(protocol: Mapping[str, object]) -> tuple[float, float]:
    import numpy as np

    interval = np.asarray(protocol.get("time_interval_closed"), dtype=np.float64)
    if (
        interval.shape != (2,)
        or not np.isfinite(interval).all()
        or float(interval[1]) <= float(interval[0])
    ):
        raise ValueError("GKW public oracle time interval is invalid")
    return float(interval[0]), float(interval[1])


def _public_oracle_grid_shape(
    protocol: Mapping[str, object],
    *,
    key: str,
    width: int,
    label: str,
):
    import numpy as np

    raw = np.asarray(protocol.get(key))
    if (
        raw.shape != (width,)
        or raw.dtype.kind not in "iu"
        or np.any(raw < 1)
    ):
        raise ValueError(f"GKW public oracle {label} grid shape is invalid")
    return np.asarray(raw, dtype=np.int64)


def _public_oracle_indices(values: object, *, shape, label: str):
    import numpy as np

    raw = np.asarray(values)
    width = len(shape)
    if (
        raw.ndim != 2
        or raw.shape[1] != width
        or raw.dtype.kind not in "iu"
    ):
        raise ValueError(
            f"GKW {label} must be integer shape [N,{width}]"
        )
    indices = np.asarray(raw, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= shape[None, :]):
        raise ValueError(f"GKW {label} leave the public grid")
    return indices


def _public_oracle_times(
    values: object,
    *,
    rows: int,
    lower: float,
    upper: float,
) -> None:
    import numpy as np

    times = np.asarray(values, dtype=np.float64)
    if times.ndim == 0:
        times = np.full(rows, float(times), dtype=np.float64)
    elif times.shape != (rows,):
        raise ValueError("GKW source times must be scalar or shape [N]")
    if not np.isfinite(times).all():
        raise ValueError("GKW source times must be finite")
    if np.any(times < lower) or np.any(times > upper):
        raise ValueError("GKW source times leave the public domain")


def _normalized_public_oracle_rows(
    value: object,
    *,
    rows: int,
    label: str,
):
    """Return a detached immutable float64 ``[N, 2]`` public value."""

    import numpy as np

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        to = getattr(value, "to", None)
        if callable(to):
            value = to(device="cpu")
        numpy = getattr(value, "numpy", None)
        if callable(numpy):
            value = numpy()
    array = np.asarray(value)
    if np.iscomplexobj(array):
        if array.shape != (rows,):
            raise RuntimeError(
                f"GKW public oracle {label} has an invalid complex shape"
            )
        array = np.stack((array.real, array.imag), axis=-1)
    elif array.shape != (rows, 2):
        raise RuntimeError(
            f"GKW public oracle {label} must return shape [N,2]"
        )
    if array.dtype.kind not in "iufc":
        raise TypeError(
            f"GKW public oracle {label} must return numeric values"
        )
    result = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(
            f"GKW public oracle {label} must return finite values"
        )
    result.setflags(write=False)
    return result


def _decoder_hard_checks_valid(values: object) -> bool:
    return bool(
        isinstance(values, Mapping)
        and set(values) == _DECODER_HARD_CHECKS
        and all(type(values[name]) is bool for name in _DECODER_HARD_CHECKS)
    )


def _decoder_hard_checks_pass(values: object) -> bool:
    return bool(
        _decoder_hard_checks_valid(values)
        and all(values[name] is True for name in _DECODER_HARD_CHECKS)
    )


def _require_decoder_report(
    report: object,
    *,
    enforce_distribution_calibration: bool,
    neural_inference_precision: str,
) -> None:
    """Validate the report, then classify a well-formed quality rejection."""

    if not isinstance(report, Mapping):
        raise RuntimeError("GKW selected decoder returned no valid report")
    cases = report.get("cases")
    distribution = report.get("distribution_calibration")
    case = cases[0] if isinstance(cases, list) and len(cases) == 1 else None
    case_checks = case.get("hard_checks") if isinstance(case, Mapping) else None
    report_checks = report.get("hard_checks")
    enforced = bool(enforce_distribution_calibration)
    failure = "GKW selected decoder failed distribution or physical gates"
    checks_match = bool(
        _decoder_hard_checks_valid(case_checks)
        and _decoder_hard_checks_valid(report_checks)
        and all(
            case_checks[name] is report_checks[name]
            for name in _DECODER_HARD_CHECKS
        )
    )
    if (
        report.get("schema") != "fusion_gkw_selected_decoder_report_v1"
        or type(report.get("sample_count")) is not int
        or report.get("sample_count") != 1
        or report.get("dense_state_persisted") is not False
        or not isinstance(case, Mapping)
        or type(case.get("sample_index")) is not int
        or case.get("sample_index") != 0
        or type(case.get("passed")) is not bool
        or not checks_match
        or not isinstance(distribution, Mapping)
        or distribution.get("enforced") is not enforced
        or type(distribution.get("passed")) is not bool
        or type(report.get("passed")) is not bool
        or report.get("neural_inference_precision")
        != neural_inference_precision
        or report.get("decode_order") != "sample_major_strict"
    ):
        raise RuntimeError(failure)

    case_passed = _decoder_hard_checks_pass(case_checks)
    report_passed = bool(
        _decoder_hard_checks_pass(report_checks)
        and (not enforced or distribution["passed"] is True)
    )
    if (
        case["passed"] is not case_passed
        or report["passed"] is not report_passed
    ):
        raise RuntimeError(failure)
    if not case_passed or not report_passed:
        raise _DecoderGateRejection(failure)


def _macro_panel(
    time_nodes: tuple[float, ...],
    grid: object,
    *,
    count: int,
    time_count: int,
    seed: int,
    profile: str,
):
    import numpy as np

    shape = (int(grid.ns), int(grid.nkx), int(grid.nky))
    total = int(np.prod(shape))
    if profile == _FORMAL_PANEL_PROFILE:
        times = np.linspace(
            time_nodes[0],
            time_nodes[-1],
            num=time_count + 1,
            dtype=np.float64,
        )[1:]
        return times, np.indices(shape, dtype=np.int64).reshape(3, -1).T
    if profile != _DEVELOPMENT_PANEL_PROFILE:
        raise ValueError(f"unsupported GKW macro panel profile: {profile!r}")
    times = np.linspace(
        time_nodes[0], time_nodes[-1], num=time_count, dtype=np.float64
    )
    selected_count = min(int(count), total)
    rng = np.random.default_rng(int(seed) & ((1 << 63) - 1))
    flat = rng.choice(total, size=selected_count, replace=False)
    indices = np.column_stack(np.unravel_index(flat, shape)).astype(
        np.int64, copy=False
    )
    return times, indices


def _request_panel_profile(request: GenerationRequest) -> tuple[str, str]:
    cohort = _generation_cohort(request)
    return cohort, _COHORT_PANEL_PROFILES[cohort]


def _macro_panel_audit(
    *,
    cohort: str,
    profile: str,
    times: object,
    field_indices: object,
    time_nodes: tuple[float, ...],
    grid: object,
    development_time_count: int,
    development_field_rows: int,
    formal_time_count: int,
    inherited_from_clean_parent: bool,
) -> Mapping[str, object]:
    import numpy as np

    expected_profile = _COHORT_PANEL_PROFILES.get(cohort)
    if expected_profile != profile:
        raise ValueError("GKW request cohort and macro panel profile disagree")
    times = np.asarray(times, dtype=np.float64)
    field_indices = np.asarray(field_indices, dtype=np.int64)
    shape = (int(grid.ns), int(grid.nkx), int(grid.nky))
    total = int(np.prod(shape))
    if profile == _FORMAL_PANEL_PROFILE:
        expected_times = np.linspace(
            time_nodes[0],
            time_nodes[-1],
            num=formal_time_count + 1,
            dtype=np.float64,
        )[1:]
        expected_indices = np.indices(shape, dtype=np.int64).reshape(3, -1).T
        if not np.array_equal(times, expected_times) or not np.array_equal(
            field_indices, expected_indices
        ):
            raise ValueError(
                "GKW formal macro panel must cover every field row and exclude t0"
            )
    else:
        expected_times = np.linspace(
            time_nodes[0],
            time_nodes[-1],
            num=development_time_count,
            dtype=np.float64,
        )
        expected_rows = min(int(development_field_rows), total)
        if (
            not np.array_equal(times, expected_times)
            or field_indices.shape != (expected_rows, 3)
            or len(np.unique(field_indices, axis=0)) != expected_rows
        ):
            raise ValueError(
                "GKW development macro panel must use its sampled profile"
            )
        for axis, extent in enumerate(shape):
            if np.any(field_indices[:, axis] < 0) or np.any(
                field_indices[:, axis] >= extent
            ):
                raise ValueError("GKW development macro panel leaves the field grid")
    includes_initial = bool(
        len(times) > 0
        and np.isclose(times[0], time_nodes[0], rtol=0.0, atol=0.0)
    )
    return MappingProxyType(
        {
            "schema": "gkw_unified_macro_panel_audit_v1",
            "cohort": cohort,
            "profile": profile,
            "time_count": int(len(times)),
            "field_row_count": int(len(field_indices)),
            "total_field_rows": total,
            "includes_initial_time": includes_initial,
            "full_field_coverage": bool(len(field_indices) == total),
            "inherited_from_clean_parent": inherited_from_clean_parent,
        }
    )


def _readonly_array(value: object):
    import numpy as np

    array = np.asarray(value)
    if array.flags.writeable:
        array.setflags(write=False)
    return array


def _readonly_public_arrays(
    values: Mapping[str, object],
) -> Mapping[str, object]:
    return MappingProxyType(
        {name: _readonly_array(value) for name, value in values.items()}
    )


def _retry_seed(
    seed: int,
    attempt: int,
    *,
    physical_retry_seed: Any,
) -> int:
    """Delegate the production retry schedule to physical qualification."""

    return physical_retry_seed(seed=seed, attempt=attempt)



def _safe_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    if not normalized:
        raise ValueError("GKW case id has no safe path component")
    return normalized[:160]


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(name): _plain_json(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _plain_json(item())
    raise TypeError("GKW public metadata is not JSON-compatible")
