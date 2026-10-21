"""Fail-closed reader for the pinned stable GKW macro release package."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from math import isfinite, prod
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from zipfile import BadZipFile, ZipFile

from .contract import (
    GKW_LEGACY_MACRO_OUTPUT_CONTRACT,
    GKW_MACRO_FIELDS,
)
from .public_contract import (
    GKW_FIELD_SHAPE,
    GKW_OPERATOR_ID,
    GKW_PUBLIC_ARRAY_NAMES,
    GKW_PUBLIC_SCALAR_NAMES,
    GKW_STATE_SHAPE,
    gkw_json_values_equal_exact,
    validate_gkw_public_components,
)


GKW_RELEASED_MACRO_ANSWER_SCHEMA = (
    "fusion_gkw_released_completed_macro_answer_v1"
)
GKW_NUMERICAL_ORACLE_SCHEMA = "fusion_gkw_numerical_oracle_v1"
GKW_STRUCTURED_MACRO_TRUTH_SCHEMA = "fusion_gkw_structured_macro_truth_v1"
_ANSWER_METADATA_KEYS = frozenset(
    {
        "schema",
        "case_id",
        "fields",
        "complex_value_encoding",
        "field_axis_order",
        "sample_seed_public",
        "continuous_hidden_parameters_included",
        "macro_truth_construction",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "built_from_transient_observation",
        "case_id",
        "completed_before_release",
        "continuous_hidden_parameters_included",
        "decoder_included",
        "field_row_count",
        "files",
        "full_hidden_state_included",
        "generator_latent_included",
        "generator_parameters_included",
        "hidden_state_artifact_written",
        "macro_fields",
        "numeric_representation",
        "operator_id",
        "phase_row_count",
        "schema",
        "structure_aware_truth_contraction",
        "time_count",
        "total_bytes",
        "truth_precision",
    }
)
_ORACLE_KEYS = frozenset(
    {
        "arbitrary_phase_subsets_supported",
        "arbitrary_time_queries_supported",
        "case_id",
        "complex_value_encoding",
        "equation_view",
        "field_axis_order",
        "field_grid_shape",
        "fixed_phase_coordinates_are_in_public_packet",
        "generator_parameters_public",
        "hidden_parameterization_public",
        "limits",
        "off_grid_phase_values_are_not_part_of_the_semidiscrete_equation",
        "phase_axis_order",
        "phase_grid_shape",
        "queries",
        "schema",
        "time_interval_closed",
        "truth_precision",
    }
)
_PAYLOAD_NAMES = frozenset(
    {"case_public.npz", "oracle_protocol.json", "released_answer.npz", "README.md"}
)
_PAYLOAD_MAX_BYTES = MappingProxyType({
    "case_public.npz": 32 * 1024 * 1024,
    "oracle_protocol.json": 256 * 1024,
    "released_answer.npz": 256 * 1024 * 1024,
    "README.md": 64 * 1024,
})
_PUBLIC_NPZ_MAX_UNCOMPRESSED_BYTES = 16 * 1024 * 1024
_ANSWER_NPZ_MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
_JSON_MAX_BYTES = 256 * 1024


@dataclass(frozen=True)
class GKWReleasePackage:
    """One integrity-checked package with private macro values kept separate."""

    case_dir: Path
    case_id: str
    manifest: Mapping[str, object]
    arrays: Mapping[str, object]
    scalars: Mapping[str, float]
    metadata: Mapping[str, object]
    oracle_protocol: Mapping[str, object]
    times: object
    field_indices: object
    values: Mapping[str, object]


def load_gkw_release(
    case_dir: str | Path,
    *,
    expected_case_id: str | None = None,
) -> GKWReleasePackage:
    """Read one pinned release while binding every public and private part."""

    import numpy as np

    root = Path(case_dir).expanduser().resolve()
    manifest = _read_json_path(root / "manifest.json")
    _validate_manifest(manifest)
    case_id = _nonempty_text(manifest.get("case_id"), "manifest case_id")
    if expected_case_id is not None and case_id != expected_case_id:
        raise ValueError("GKW release case id differs from the requested case")
    blobs = _read_verified_payloads(root, manifest)

    arrays, scalars, metadata = _read_public_packet(
        blobs["case_public.npz"], expected_case_id=case_id
    )
    if manifest.get("operator_id") != metadata.get("operator_id"):
        raise ValueError("GKW release operator id differs from public packet")

    oracle = _read_json_bytes(blobs["oracle_protocol.json"], "oracle protocol")
    validate_gkw_oracle_protocol(
        oracle,
        case_id=case_id,
        time_nodes=metadata["time_nodes"],
    )

    times, indices, values, _answer_metadata = _read_answer(
        blobs["released_answer.npz"],
        expected_time_count=int(manifest["time_count"]),
        expected_field_count=int(manifest["field_row_count"]),
        expected_case_id=case_id,
    )

    expected_shape = (len(times), len(indices), 2)
    if int(manifest["time_count"]) != len(times):
        raise ValueError("GKW release time count differs from manifest")
    if int(manifest["field_row_count"]) != len(indices):
        raise ValueError("GKW release field-row count differs from manifest")
    if any(value.shape != expected_shape for value in values.values()):
        raise ValueError("GKW released macro array shape changed")
    expected_times = np.linspace(
        float(metadata["time_nodes"][0]),
        float(metadata["time_nodes"][-1]),
        num=len(times),
        dtype=np.float64,
    )
    if not np.array_equal(times, expected_times):
        raise ValueError("GKW released times differ from the stable panel rule")

    grid_shape = tuple(len(arrays[name]) for name in ("s", "kx", "ky"))
    for axis, extent in enumerate(grid_shape):
        if np.any(indices[:, axis] < 0) or np.any(indices[:, axis] >= extent):
            raise IndexError("GKW released field index leaves the public grid")
    interval = np.asarray(oracle["time_interval_closed"], dtype=np.float64)
    if np.any(times < interval[0]) or np.any(times > interval[1]):
        raise ValueError("GKW released times leave the oracle interval")

    return GKWReleasePackage(
        case_dir=root,
        case_id=case_id,
        manifest=_freeze_json(manifest),
        arrays=MappingProxyType(
            {name: _readonly_array(value) for name, value in arrays.items()}
        ),
        scalars=MappingProxyType(dict(scalars)),
        metadata=_freeze_json(metadata),
        oracle_protocol=_freeze_json(oracle),
        times=_readonly_array(times),
        field_indices=_readonly_array(indices),
        values=MappingProxyType(
            {name: _readonly_array(value) for name, value in values.items()}
        ),
    )


def freeze_gkw_public_data(package: GKWReleasePackage) -> Mapping[str, object]:
    """Return an immutable Agent-visible view with no truth arrays or paths."""

    return MappingProxyType(
        {
            "equation": "GKW electrostatic gyrokinetics",
            "arrays": package.arrays,
            "scalars": package.scalars,
            "metadata": package.metadata,
            "oracle_protocol": package.oracle_protocol,
            "macro_query": MappingProxyType(
                {
                    "times": package.times,
                    "field_indices": package.field_indices,
                    "output_contract": GKW_LEGACY_MACRO_OUTPUT_CONTRACT,
                }
            ),
        }
    )


def _validate_manifest(manifest: Mapping[str, object]) -> None:
    if set(manifest) != _MANIFEST_KEYS:
        raise ValueError("GKW stable release manifest keys changed")
    if manifest.get("schema") != GKW_RELEASED_MACRO_ANSWER_SCHEMA:
        raise ValueError("GKW case is not a stable completed macro release")
    exact = {
        "completed_before_release": True,
        "truth_precision": "float64",
        "numeric_representation": "sampled_macroscopic_linear_functionals",
        "phase_row_count": 0,
        "macro_fields": list(GKW_MACRO_FIELDS),
        "full_hidden_state_included": False,
        "continuous_hidden_parameters_included": False,
        "generator_parameters_included": False,
        "generator_latent_included": False,
        "decoder_included": False,
        "built_from_transient_observation": True,
        "hidden_state_artifact_written": False,
        "structure_aware_truth_contraction": True,
    }
    if not gkw_json_values_equal_exact(
        {name: manifest.get(name) for name in exact},
        exact,
    ):
        raise ValueError("GKW release violates stable completion/privacy policy")
    for name in ("time_count", "field_row_count", "total_bytes"):
        value = manifest.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"GKW manifest {name} must be a positive integer")
    if not 2 <= int(manifest["time_count"]) <= 256:
        raise ValueError("GKW manifest time count exceeds the release limit")
    if int(manifest["field_row_count"]) > prod(GKW_FIELD_SHAPE):
        raise ValueError("GKW manifest field-row count exceeds the public grid")
    if int(manifest["total_bytes"]) > sum(_PAYLOAD_MAX_BYTES.values()):
        raise ValueError("GKW manifest total bytes exceed the release limit")
    if manifest.get("operator_id") != GKW_OPERATOR_ID:
        raise ValueError("GKW manifest operator id differs from the pin")


def _read_verified_payloads(
    root: Path,
    manifest: Mapping[str, object],
) -> dict[str, bytes]:
    rows = manifest.get("files")
    if not isinstance(rows, list) or len(rows) != len(_PAYLOAD_NAMES):
        raise ValueError("GKW manifest files declaration changed")
    result: dict[str, bytes] = {}
    total = 0
    for row in rows:
        if not isinstance(row, Mapping) or set(row) - {"sha256"} != {"path", "bytes"}:
            raise ValueError("GKW manifest file row is invalid")
        name = row.get("path")
        if not isinstance(name, str) or name not in _PAYLOAD_NAMES or name in result:
            raise ValueError("GKW manifest contains a noncanonical payload path")
        path = root / name
        size = row.get("bytes")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or size > _PAYLOAD_MAX_BYTES[name]
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != size
        ):
            raise ValueError(f"GKW release payload integrity failed: {name}")
        data = path.read_bytes()
        if len(data) != size:
            raise ValueError(f"GKW release payload integrity failed: {name}")
        result[name] = data
        total += size
    if set(result) != _PAYLOAD_NAMES or total != manifest.get("total_bytes"):
        raise ValueError("GKW release payload inventory differs from manifest")
    return result


def _read_public_packet(
    data: bytes,
    *,
    expected_case_id: str,
) -> tuple[dict[str, object], dict[str, float], dict[str, object]]:
    import numpy as np

    _validate_npz_archive(
        data,
        label="public packet",
        max_uncompressed_bytes=_PUBLIC_NPZ_MAX_UNCOMPRESSED_BYTES,
    )
    expected_keys = {
        *(f"array__{name}" for name in GKW_PUBLIC_ARRAY_NAMES),
        *(f"scalar__{name}" for name in GKW_PUBLIC_SCALAR_NAMES),
        "metadata_json",
    }
    with np.load(BytesIO(data), allow_pickle=False) as payload:
        if set(payload.files) != expected_keys:
            raise ValueError("serialized GKW public packet keys are invalid")
        metadata = _npz_json(payload, "metadata_json", "public metadata")
        arrays = {
            name: np.asarray(payload[f"array__{name}"]).copy()
            for name in GKW_PUBLIC_ARRAY_NAMES
        }
        scalars = {
            name: _finite_scalar(payload[f"scalar__{name}"], name)
            for name in GKW_PUBLIC_SCALAR_NAMES
        }
    validate_gkw_public_components(
        arrays,
        scalars,
        metadata,
        expected_case_id=expected_case_id,
    )
    return arrays, scalars, metadata


def validate_gkw_oracle_protocol(
    oracle: Mapping[str, object],
    *,
    case_id: str,
    time_nodes: object,
) -> None:
    """Validate the exact public numerical-oracle contract for one case."""

    import numpy as np

    if set(oracle) != _ORACLE_KEYS:
        raise ValueError("GKW oracle protocol keys changed")
    exact = {
        "schema": GKW_NUMERICAL_ORACLE_SCHEMA,
        "case_id": case_id,
        "complex_value_encoding": ["real", "imag"],
        "truth_precision": "float64",
        "hidden_parameterization_public": False,
        "generator_parameters_public": False,
        "arbitrary_phase_subsets_supported": True,
        "arbitrary_time_queries_supported": True,
        "equation_view": "continuous_time_fixed_phase_grid_method_of_lines",
        "field_axis_order": ["s", "kx", "ky"],
        "field_grid_shape": list(GKW_FIELD_SHAPE),
        "fixed_phase_coordinates_are_in_public_packet": True,
        "off_grid_phase_values_are_not_part_of_the_semidiscrete_equation": True,
        "phase_axis_order": ["vpar", "mu", "s", "kx", "ky"],
        "phase_grid_shape": list(GKW_STATE_SHAPE),
    }
    if not gkw_json_values_equal_exact(
        {name: oracle.get(name) for name in exact},
        exact,
    ):
        raise ValueError("GKW oracle protocol violates the pinned policy")
    raw_interval = oracle.get("time_interval_closed")
    if (
        not isinstance(raw_interval, (list, tuple))
        or len(raw_interval) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in raw_interval
        )
    ):
        raise ValueError("GKW oracle time interval is invalid")
    interval = np.asarray(raw_interval, dtype=np.float64)
    nodes = np.asarray(time_nodes, dtype=np.float64)
    limits = oracle.get("limits")
    queries = oracle.get("queries")
    expected_queries = {
        "initial_h": {
            "inputs": {"phase_indices": "int64 [N,5]"},
            "output": "float64 [N,2]",
        },
        "initial_phi": {
            "inputs": {"field_indices": "int64 [N,3]"},
            "output": "float64 [N,2]",
        },
        "source": {
            "definition": "S_balance = d_t h_hidden - R_GKW[h_hidden]",
            "inputs": {
                "phase_indices": "int64 [N,5]",
                "times": "float64 scalar or [N]",
            },
            "output": "float64 [N,2]",
        },
    }
    if (
        interval.shape != (2,)
        or not np.isfinite(interval).all()
        or interval[1] <= interval[0]
        or nodes.shape != (4,)
        or not np.array_equal(interval, nodes[[0, -1]])
        or not isinstance(limits, Mapping)
        or set(limits)
        != {
            "max_rows_per_request",
            "max_total_rows",
            "max_unique_source_times",
            "source_frame_cache_entries",
        }
        or not gkw_json_values_equal_exact(queries, expected_queries)
    ):
        raise ValueError("GKW oracle protocol recursively changed")
    limit_caps = {
        "max_rows_per_request": 65536,
        "max_total_rows": 20000000,
        "max_unique_source_times": 256,
        "source_frame_cache_entries": 64,
    }
    for name, cap in limit_caps.items():
        value = limits[name]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= cap:
            raise ValueError(f"GKW oracle limit {name} is invalid")
    if limits["max_total_rows"] < limits["max_rows_per_request"]:
        raise ValueError("GKW oracle total-row limit is incoherent")


def _read_answer(
    data: bytes,
    *,
    expected_time_count: int,
    expected_field_count: int,
    expected_case_id: str,
) -> tuple[object, object, dict[str, object], dict[str, object]]:
    import numpy as np

    _validate_npz_archive(
        data,
        label="released answer",
        max_uncompressed_bytes=_ANSWER_NPZ_MAX_UNCOMPRESSED_BYTES,
    )
    expected_keys = {
        "times",
        "field_indices",
        "metadata_json",
        *(f"{name}_values" for name in GKW_MACRO_FIELDS),
    }
    with np.load(BytesIO(data), allow_pickle=False) as payload:
        if set(payload.files) != expected_keys:
            raise ValueError("serialized GKW released-answer keys are invalid")
        raw_times = np.asarray(payload["times"])
        raw_indices = np.asarray(payload["field_indices"])
        if (
            raw_times.dtype != np.dtype(np.float64)
            or raw_times.shape != (expected_time_count,)
        ):
            raise TypeError("GKW released times must use float64")
        if (
            raw_indices.dtype != np.dtype(np.int64)
            or raw_indices.shape != (expected_field_count, 3)
        ):
            raise TypeError("GKW released field indices must be int64 [N,3]")
        times = raw_times.copy()
        indices = raw_indices.copy()
        values = {}
        expected_shape = (expected_time_count, expected_field_count, 2)
        for name in GKW_MACRO_FIELDS:
            value = np.asarray(payload[f"{name}_values"])
            if value.dtype != np.dtype(np.float64) or value.shape != expected_shape:
                raise TypeError(
                    f"GKW released macro {name} must be float64 {expected_shape}"
                )
            values[name] = value.copy()
        metadata = _npz_json(payload, "metadata_json", "answer metadata")
    if (
        times.ndim != 1
        or len(times) <= 0
        or not np.isfinite(times).all()
        or not np.all(np.diff(times) > 0.0)
        or indices.ndim != 2
        or indices.shape != (len(indices), 3)
        or len(indices) <= 0
        or len(np.unique(indices, axis=0)) != len(indices)
    ):
        raise ValueError("GKW released answer coordinates are invalid")
    for name, value in values.items():
        if value.dtype != np.dtype(np.float64) or not np.isfinite(value).all():
            raise TypeError(f"GKW released macro {name} must be finite float64")
    _validate_answer_metadata(metadata, expected_case_id=expected_case_id)
    return times, indices, values, metadata


def _validate_answer_metadata(
    metadata: Mapping[str, object], *, expected_case_id: str
) -> None:
    if set(metadata) != _ANSWER_METADATA_KEYS:
        raise ValueError("GKW released answer metadata keys changed")
    construction = metadata.get("macro_truth_construction")
    if (
        metadata.get("schema") != GKW_RELEASED_MACRO_ANSWER_SCHEMA
        or metadata.get("fields") != list(GKW_MACRO_FIELDS)
        or metadata.get("complex_value_encoding") != ["real", "imag"]
        or metadata.get("field_axis_order")
        != ["s", "kx", "ky_nonnegative_half_spectrum"]
        or metadata.get("continuous_hidden_parameters_included") is not False
        or not isinstance(metadata.get("sample_seed_public"), int)
        or isinstance(metadata.get("sample_seed_public"), bool)
        or not isinstance(construction, Mapping)
        or not _valid_truth_construction(construction)
        or metadata.get("case_id") != expected_case_id
    ):
        raise ValueError("GKW released answer metadata violates the pinned contract")
    _nonempty_text(metadata.get("case_id"), "answer case_id")


def _valid_truth_construction(value: Mapping[str, object]) -> bool:
    if value.get("hidden_state_remains_primary") is not True:
        return False
    schema = value.get("schema")
    if schema == GKW_STRUCTURED_MACRO_TRUTH_SCHEMA:
        required = {
            "schema",
            "method",
            "time_control_count",
            "fused_velocity_contraction_calls_per_case",
            "velocity_control_slices_contracted",
            "velocity_functional",
            "dense_velocity_reconstruction_per_query",
            "velocity_requadrature_per_query",
            "quasineutrality_applied_to_time_controls_once",
            "quasineutrality_method",
            "full_five_dimensional_control_padding_avoided",
            "arbitrary_time_evaluation",
            "linearity_identity_used",
            "dense_state_elements_per_time_avoided",
            "macro_cache_complex_elements",
            "hidden_state_remains_primary",
        }
        controls = value.get("time_control_count")
        method = value.get("quasineutrality_method")
        return (
            set(value) == required
            and _is_exact_integer(controls, 4)
            and value.get("method")
            == "linear_functional_contraction_of_spline_time_controls"
            and _is_exact_integer(
                value.get("fused_velocity_contraction_calls_per_case"), 1
            )
            and _is_exact_integer(
                value.get("velocity_control_slices_contracted"), controls
            )
            and value.get("velocity_functional")
            == "declared_gkw_semidiscrete_intvp_intmu_weights"
            and value.get("dense_velocity_reconstruction_per_query") is False
            and value.get("velocity_requadrature_per_query") is False
            and value.get("quasineutrality_applied_to_time_controls_once") is True
            and method
            in {
                "direct_phi_weight_contraction_on_truncated_controls",
                "full_control_padding_compatibility_fallback",
            }
            and (
                (
                    method
                    == "direct_phi_weight_contraction_on_truncated_controls"
                    and value.get(
                        "full_five_dimensional_control_padding_avoided"
                    )
                    is True
                )
                or (
                    method == "full_control_padding_compatibility_fallback"
                    and value.get(
                        "full_five_dimensional_control_padding_avoided"
                    )
                    is False
                )
            )
            and value.get("arbitrary_time_evaluation")
            == "natural_cubic_control_interpolation"
            and value.get("linearity_identity_used")
            == "L(sum_i c_i h_i)=sum_i c_i L(h_i)"
            and _is_exact_integer(
                value.get("dense_state_elements_per_time_avoided"),
                prod(GKW_STATE_SHAPE),
            )
            and _is_exact_integer(
                value.get("macro_cache_complex_elements"),
                controls * len(GKW_MACRO_FIELDS) * prod(GKW_FIELD_SHAPE),
            )
        )
    if schema != "fusion_gkw_composite_macro_truth_v1" or set(value) != {
        "base",
        "dense_velocity_reconstruction_per_query",
        "generator_parameters_modified",
        "hidden_state_remains_primary",
        "method",
        "perturbation",
        "schema",
    }:
        return False
    base = value.get("base")
    perturbation = value.get("perturbation")
    perturbation_keys = {
        "additive_outside_generator_representation",
        "dense_velocity_reconstruction_per_query",
        "hidden_state_remains_primary",
        "method",
        "quasineutrality_calls_per_case",
        "rank",
        "schema",
        "velocity_requadrature_per_query",
    }
    rank = perturbation.get("rank") if isinstance(perturbation, Mapping) else None
    return (
        value.get("generator_parameters_modified") is False
        and value.get("dense_velocity_reconstruction_per_query") is False
        and value.get("method")
        == "analytic_sum_of_spline_and_low_rank_additive_truth"
        and isinstance(base, Mapping)
        and _valid_truth_construction(base)
        and isinstance(perturbation, Mapping)
        and set(perturbation) == perturbation_keys
        and perturbation.get("schema")
        == "fusion_gkw_low_rank_perturbation_macro_truth_v1"
        and perturbation.get("hidden_state_remains_primary") is True
        and perturbation.get("additive_outside_generator_representation") is True
        and perturbation.get("dense_velocity_reconstruction_per_query") is False
        and perturbation.get("velocity_requadrature_per_query") is False
        and perturbation.get("method") == "analytic_separable_velocity_contraction"
        and isinstance(rank, int)
        and not isinstance(rank, bool)
        and 1 <= rank <= 64
        and _is_exact_integer(
            perturbation.get("quasineutrality_calls_per_case"), rank
        )
    )


def _is_exact_integer(value: object, expected: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _finite_scalar(value: Any, name: str) -> float:
    import numpy as np

    array = np.asarray(value)
    if (
        array.shape != ()
        or array.dtype != np.dtype(np.float64)
        or not np.isfinite(array).all()
    ):
        raise TypeError(f"GKW public scalar {name} must be scalar float64")
    return float(array)


def _npz_json(payload: Any, key: str, label: str) -> dict[str, object]:
    import numpy as np

    value = np.asarray(payload[key])
    if value.size != 1 or value.dtype.kind not in "SU":
        raise ValueError(f"serialized GKW {label} must be scalar JSON text")
    encoded = str(value.reshape(-1)[0]).encode("utf-8")
    return _read_json_bytes(encoded, label)


def _read_json_path(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    if path.stat().st_size > _JSON_MAX_BYTES:
        raise ValueError(f"GKW JSON exceeds its size limit: {path}")
    return _read_json_bytes(path.read_bytes(), str(path))


def _read_json_bytes(data: bytes, label: str) -> dict[str, object]:
    if len(data) > _JSON_MAX_BYTES:
        raise ValueError(f"GKW JSON exceeds its size limit: {label}")
    value = json.loads(
        data.decode("utf-8"),
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {label}")
    return value


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate GKW JSON key: {name}")
        result[name] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite GKW JSON value: {value}")


def _validate_npz_archive(
    data: bytes, *, label: str, max_uncompressed_bytes: int
) -> None:
    try:
        with ZipFile(BytesIO(data)) as archive:
            rows = archive.infolist()
            names = [row.filename for row in rows]
            if (
                not rows
                or len(names) != len(set(names))
                or any(
                    row.is_dir()
                    or "/" in row.filename
                    or not row.filename.endswith(".npy")
                    for row in rows
                )
                or sum(row.file_size for row in rows) > max_uncompressed_bytes
            ):
                raise ValueError(f"GKW {label} archive exceeds its safe shape")
    except BadZipFile as exc:
        raise ValueError(f"GKW {label} is not a valid NPZ archive") from exc


def _nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"GKW {label} must be nonempty text")
    return value


def _valid_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _readonly_array(value: Any):
    import numpy as np

    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(name, str) for name in value):
            raise TypeError("GKW public JSON keys must be text")
        return MappingProxyType(
            {name: _freeze_json(item) for name, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("GKW public JSON contains a non-finite number")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("GKW public JSON contains a noncanonical value")
