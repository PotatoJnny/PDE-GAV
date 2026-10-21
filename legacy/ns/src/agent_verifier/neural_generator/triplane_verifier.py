from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from agent_verifier.neural_generator.triplane_autoencoder import CONTINUOUS_TRIPLANE_MMS_ABI_VERSION
from agent_verifier.neural_generator.structured_ns_answer_first import (
    LEGACY_C3_POLYGON_STRUCTURED_NS_MMS_ABI_VERSION,
    LEGACY_HARD_PRESSURE_STRUCTURED_NS_MMS_ABI_VERSION,
    LEGACY_SMOOTH_PRESSURE_STRUCTURED_NS_MMS_ABI_VERSION,
    LEGACY_STRUCTURED_NS_MMS_ABI_VERSION,
    STRUCTURED_NS_MMS_ABI_VERSION,
    SUPPORTED_STRUCTURED_NS_MMS_ABI_VERSIONS,
)
from agent_verifier.neural_generator.triplane_dataset import (
    GeometryContract,
    case_metrics,
    load_case,
    mms_forcing_from_velocity_pressure,
    repeat_initial_difficulty,
    rms,
)

CONTINUOUS_MMS_TRUTH_CONTRACT = CONTINUOUS_TRIPLANE_MMS_ABI_VERSION
SUPPORTED_CONTINUOUS_MMS_TRUTH_CONTRACTS = {
    CONTINUOUS_TRIPLANE_MMS_ABI_VERSION,
    *SUPPORTED_STRUCTURED_NS_MMS_ABI_VERSIONS,
}


@dataclass(frozen=True)
class HardGateConfig:
    divergence_rel_max: float = 5.0e-2
    outer_boundary_velocity_rel_max: float = 1.0e-6
    obstacle_velocity_rel_max: float = 1.0e-6
    forcing_inside_obstacle_rel_max: float = 1.0e-7
    mms_residual_rel_max: float = 1.0e-8
    require_mms_residual: bool = False
    require_continuous_mms: bool = False
    audit_continuous_mms: bool = True
    continuous_residual_rel_max: float = 1.0e-10
    continuous_divergence_rel_max: float = 1.0e-10
    continuous_outer_boundary_velocity_abs_max: float = 1.0e-9
    continuous_materialized_view_rel_max: float = 5.0e-5


@dataclass(frozen=True)
class DistributionGateConfig:
    enabled: bool = True
    metrics: tuple[str, ...] = (
        "velocity_rms",
        "forcing_rms",
        "time_derivative_rms",
        "vorticity_rms",
        "velocity_laplacian_rms",
        "pressure_gradient_rms",
        "repeat_initial_difficulty",
    )
    lower_ratio: float = 0.8
    upper_ratio: float = 1.2


@dataclass
class VerificationResult:
    accepted: bool
    hard_passed: bool
    distribution_passed: bool
    metrics: dict[str, float]
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "hard_passed": self.hard_passed,
            "distribution_passed": self.distribution_passed,
            "metrics": self.metrics,
            "failures": self.failures,
            "warnings": self.warnings,
        }


def verify_case_dir(
    case_dir: str | Path,
    *,
    geometry: GeometryContract,
    hard_config: HardGateConfig | None = None,
    distribution_bands: dict[str, dict[str, float]] | None = None,
    distribution_config: DistributionGateConfig | None = None,
) -> VerificationResult:
    hard = hard_config or HardGateConfig()
    dist_cfg = distribution_config or DistributionGateConfig(enabled=distribution_bands is not None)
    failures: list[str] = []
    warnings: list[str] = []
    case = load_case(case_dir)
    if hard.require_continuous_mms:
        failures.extend(continuous_mms_case_contract_failures(case_dir, require_generator_metadata=False))
    public = case.public_arrays
    hidden = case.hidden_arrays
    required_public = {
        "x",
        "y",
        "t",
        "initial_velocity",
        "forcing",
        "fluid_mask",
        "obstacle_mask",
        "outer_boundary_mask",
        "obstacle_signed_distance",
        "brinkman_indicator",
    }
    required_hidden = {"u_true", "pressure"}
    missing_public = sorted(required_public - set(public))
    missing_hidden = sorted(required_hidden - set(hidden))
    if missing_public:
        failures.append(f"missing_public_arrays:{','.join(missing_public)}")
    if missing_hidden:
        failures.append(f"missing_hidden_arrays:{','.join(missing_hidden)}")
    if failures:
        return VerificationResult(False, False, False, {}, failures, warnings)
    if str(case.public_question.get("geometry", {}).get("contract_hash")) != geometry.contract_hash:
        failures.append("geometry_contract_hash_mismatch")
    for name, expected in geometry.public_arrays().items():
        got = np.asarray(public[name])
        if got.shape != expected.shape or not np.array_equal(got, expected):
            failures.append(f"geometry_array_mismatch:{name}")
    velocity = np.asarray(hidden["u_true"], dtype=np.float64)
    pressure = np.asarray(hidden["pressure"], dtype=np.float64)
    forcing = np.asarray(public["forcing"], dtype=np.float64)
    t = np.asarray(public["t"], dtype=np.float64)
    if velocity.ndim != 4 or velocity.shape[1] != 2:
        failures.append("u_true_shape_invalid")
    if pressure.shape != (velocity.shape[0], velocity.shape[2], velocity.shape[3]):
        failures.append("pressure_shape_invalid")
    if forcing.shape != velocity.shape:
        failures.append("forcing_shape_invalid")
    if np.asarray(public["initial_velocity"]).shape != velocity[0].shape:
        failures.append("initial_velocity_shape_invalid")
    finite_arrays = [velocity, pressure, forcing, np.asarray(public["initial_velocity"])]
    if not all(np.all(np.isfinite(arr)) for arr in finite_arrays):
        failures.append("non_finite_arrays")
    if failures:
        return VerificationResult(False, False, False, {}, failures, warnings)
    metrics = case_metrics(velocity=velocity, pressure=pressure, forcing=forcing, t=t, geometry=geometry)
    diagnostics = dict(case.diagnostics or {})
    if hard.require_continuous_mms:
        if "mms_residual_rel" in diagnostics:
            metrics["generator_declared_mms_residual_rel"] = float(diagnostics["mms_residual_rel"])
        warnings.append("generator_declared_mms_residual_not_used_as_hard_evidence")
    elif "mms_residual_rel" in diagnostics:
        metrics["mms_residual_rel"] = float(diagnostics["mms_residual_rel"])
    elif hard.require_mms_residual:
        failures.append("missing_mms_residual_rel")
    else:
        warnings.append("mms_residual_rel_not_checked")
    if hard.require_continuous_mms and hard.audit_continuous_mms:
        try:
            metrics.update(_continuous_mms_metrics(case_dir, public, hidden, geometry))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"continuous_mms_evaluation_failed:{type(exc).__name__}")
    elif hard.require_continuous_mms:
        warnings.append("continuous_mms_numerical_audit_skipped")
    if hard.require_continuous_mms and hard.audit_continuous_mms:
        _check_metric_max(failures, metrics, "continuous_residual_rel", hard.continuous_residual_rel_max)
        _check_metric_max(failures, metrics, "continuous_divergence_rel_max", hard.continuous_divergence_rel_max)
        _check_metric_max(
            failures,
            metrics,
            "continuous_outer_boundary_velocity_abs_max",
            hard.continuous_outer_boundary_velocity_abs_max,
        )
        for name in (
            "continuous_velocity_view_rel",
            "continuous_pressure_view_rel",
            "continuous_forcing_view_rel",
        ):
            _check_metric_max(failures, metrics, name, hard.continuous_materialized_view_rel_max)
    if not hard.require_continuous_mms:
        _check_metric_max(failures, metrics, "divergence_rel", hard.divergence_rel_max)
        _check_metric_max(failures, metrics, "outer_boundary_velocity_rel", hard.outer_boundary_velocity_rel_max)
    _check_metric_max(failures, metrics, "obstacle_velocity_rel", hard.obstacle_velocity_rel_max)
    _check_metric_max(failures, metrics, "forcing_inside_obstacle_rel", hard.forcing_inside_obstacle_rel_max)
    if "mms_residual_rel" in metrics and not hard.require_continuous_mms:
        _check_metric_max(failures, metrics, "mms_residual_rel", hard.mms_residual_rel_max)
    hard_passed = not failures
    distribution_failures = []
    if dist_cfg.enabled:
        if not distribution_bands:
            distribution_failures.append("missing_distribution_bands")
        else:
            for name in dist_cfg.metrics:
                band = distribution_bands.get(name)
                if band is None:
                    distribution_failures.append(f"missing_distribution_band:{name}")
                    continue
                value = float(metrics.get(name, np.nan))
                low = float(band["center"]) * float(dist_cfg.lower_ratio)
                high = float(band["center"]) * float(dist_cfg.upper_ratio)
                if not np.isfinite(value) or value < low or value > high:
                    distribution_failures.append(f"{name}_outside_d0_ratio[{low:.6g},{high:.6g}]")
    failures.extend(distribution_failures)
    distribution_passed = not distribution_failures
    return VerificationResult(
        accepted=hard_passed and distribution_passed,
        hard_passed=hard_passed,
        distribution_passed=distribution_passed,
        metrics=metrics,
        failures=failures,
        warnings=warnings,
    )


def build_distribution_bands(case_dirs: Iterable[str | Path], *, center: str = "median") -> dict[str, dict[str, float]]:
    rows = []
    for case_dir in case_dirs:
        diagnostics_path = Path(case_dir) / "diagnostics.json"
        public_question_path = Path(case_dir) / "public_question.json"
        if not diagnostics_path.exists() or not public_question_path.exists():
            continue
        public_question = json.loads(public_question_path.read_text())
        if public_question.get("geometry", {}).get("contract_hash") is None:
            continue
        rows.append(json.loads(diagnostics_path.read_text()))
    metric_names = (
        "velocity_rms",
        "forcing_rms",
        "time_derivative_rms",
        "vorticity_rms",
        "velocity_laplacian_rms",
        "pressure_gradient_rms",
        "repeat_initial_difficulty",
    )
    bands = {}
    for name in metric_names:
        values = np.asarray([float(row[name]) for row in rows if name in row and np.isfinite(float(row[name]))], dtype=float)
        if values.size == 0:
            continue
        center_value = float(np.mean(values) if center == "mean" else np.median(values))
        bands[name] = {
            "center": center_value,
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q95": float(np.quantile(values, 0.95)),
            "count": int(values.size),
        }
    return bands


def near_duplicate_report(case_dirs: Iterable[str | Path], *, max_cases: int = 512, threshold: float = 1.0e-3) -> dict[str, Any]:
    dirs = [Path(p) for p in case_dirs][: int(max_cases)]
    fingerprints = []
    for path in dirs:
        case = load_case(path)
        u = np.asarray(case.hidden_arrays["u_true"], dtype=np.float64)
        fluid = np.asarray(case.public_arrays["fluid_mask"], dtype=bool)
        sample = u[:, :, fluid].reshape(u.shape[0], 2, -1)
        vec = np.concatenate([sample.mean(axis=2).ravel(), sample.std(axis=2).ravel()])
        norm = np.linalg.norm(vec) + 1.0e-12
        fingerprints.append((path.name, vec / norm))
    pairs = []
    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            dist = float(np.linalg.norm(fingerprints[i][1] - fingerprints[j][1]))
            if dist < float(threshold):
                pairs.append({"case_a": fingerprints[i][0], "case_b": fingerprints[j][0], "fingerprint_l2": dist})
    return {"checked_cases": len(fingerprints), "threshold": float(threshold), "near_duplicate_pairs": pairs}


def write_verification_report(path: str | Path, results: list[VerificationResult]) -> dict[str, Any]:
    payload = {
        "n": len(results),
        "accepted": int(sum(result.accepted for result in results)),
        "hard_passed": int(sum(result.hard_passed for result in results)),
        "distribution_passed": int(sum(result.distribution_passed for result in results)),
        "failure_counts": _failure_counts(results),
        "results": [result.to_dict() for result in results],
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _continuous_mms_metrics(
    case_dir: str | Path,
    public: dict[str, np.ndarray],
    hidden: dict[str, np.ndarray],
    geometry: GeometryContract,
) -> dict[str, float]:
    from agent_verifier.neural_generator.triplane_mms import continuous_divergence_relative_max, load_continuous_case

    x_public = np.asarray(public["x"], dtype=np.float64)
    y_public = np.asarray(public["y"], dtype=np.float64)
    t_public = np.asarray(public["t"], dtype=np.float64)
    x_indices = _audit_indices(x_public.size, 4)
    y_indices = _audit_indices(y_public.size, 4)
    t_indices = _audit_indices(t_public.size, 3)
    x = x_public[x_indices]
    y = y_public[y_indices]
    t = t_public[t_indices]
    tt, xx, yy = np.meshgrid(t, x, y, indexing="ij")
    coords = np.stack([xx, yy, tt], axis=-1).reshape(-1, 3)
    continuous = load_continuous_case(case_dir, device="auto")
    quantities = continuous.quantities(coords)
    velocity = np.stack([quantities["u"].detach().cpu().numpy(), quantities["v"].detach().cpu().numpy()], axis=1)
    forcing = np.stack([quantities["forcing_u"].detach().cpu().numpy(), quantities["forcing_v"].detach().cpu().numpy()], axis=1)
    residual = np.stack([quantities["residual_u"].detach().cpu().numpy(), quantities["residual_v"].detach().cpu().numpy()], axis=1)
    divergence = quantities["divergence"].detach().cpu().numpy()
    d_x_u = quantities["d_x_u"].detach().cpu().numpy()
    d_y_v = quantities["d_y_v"].detach().cpu().numpy()
    pressure = quantities["pressure"].detach().cpu().numpy()
    cached_velocity = _gather_vector_view(hidden["u_true"], t_indices, x_indices, y_indices)
    cached_pressure = _gather_scalar_view(hidden["pressure"], t_indices, x_indices, y_indices)
    cached_forcing = _gather_vector_view(public["forcing"], t_indices, x_indices, y_indices)
    boundary_quantities = continuous.quantities(_actual_boundary_coords(continuous.domain_bounds, t))
    boundary_velocity = np.stack(
        [
            boundary_quantities["u"].detach().cpu().numpy(),
            boundary_quantities["v"].detach().cpu().numpy(),
        ],
        axis=1,
    )
    public_operator_forcing = mms_forcing_from_velocity_pressure(
        velocity=np.asarray(hidden["u_true"], dtype=np.float64),
        pressure=np.asarray(hidden["pressure"], dtype=np.float64),
        t=t_public,
        viscosity=float(continuous.viscosity),
        geometry=geometry,
    )
    public_operator_residual = public_operator_forcing - np.asarray(public["forcing"], dtype=np.float64)
    return {
        "continuous_residual_rel": rms(residual) / (rms(forcing) + rms(velocity) + 1.0e-12),
        "continuous_divergence_abs_max": float(np.max(np.abs(divergence))),
        "continuous_divergence_rel_max": continuous_divergence_relative_max(divergence, d_x_u, d_y_v),
        "continuous_velocity_view_rel": rms(cached_velocity - velocity) / (rms(velocity) + 1.0e-12),
        "continuous_pressure_view_rel": rms(cached_pressure - pressure) / (rms(pressure) + 1.0e-12),
        "continuous_forcing_view_rel": rms(cached_forcing - forcing) / (rms(forcing) + 1.0e-12),
        "continuous_outer_boundary_velocity_abs_max": float(np.max(np.abs(boundary_velocity))),
        "diagnostic_finite_difference_materialized_view_residual_rel": rms(public_operator_residual)
        / (rms(public["forcing"]) + rms(hidden["u_true"]) + 1.0e-12),
    }


def continuous_mms_case_contract_failures(
    case_dir: str | Path,
    *,
    require_generator_metadata: bool = True,
) -> list[str]:
    path = Path(case_dir)
    failures: list[str] = []
    if not (path / "hidden_continuous_case.pt").exists():
        failures.append("missing_hidden_continuous_case")
    public_question = _contract_json(path / "public_question.json", failures, "public_question")
    hidden_manifest = _contract_json(path / "hidden_truth_manifest.json", failures, "hidden_truth_manifest")
    continuous_manifest = _contract_json(path / "hidden_continuous_manifest.json", failures, "hidden_continuous_manifest")
    if public_question:
        if public_question.get("case_source") != "fixed_geometry_incompressible_flow":
            failures.append("public_case_source_invalid")
        if public_question.get("forcing_derivation") != "provided_public_arrays_forcing":
            failures.append("public_forcing_contract_invalid")
        if "truth_representation" in public_question:
            failures.append("public_question_leaks_truth_representation")
    if hidden_manifest:
        hidden_abi = hidden_manifest.get("continuous_mms_abi")
        expected_truth_type = {
            CONTINUOUS_TRIPLANE_MMS_ABI_VERSION: "continuous_triplane_streamfunction",
            LEGACY_STRUCTURED_NS_MMS_ABI_VERSION: "continuous_structured_streamfunction",
            LEGACY_C3_POLYGON_STRUCTURED_NS_MMS_ABI_VERSION: "continuous_structured_streamfunction",
            LEGACY_HARD_PRESSURE_STRUCTURED_NS_MMS_ABI_VERSION: "continuous_structured_streamfunction",
            LEGACY_SMOOTH_PRESSURE_STRUCTURED_NS_MMS_ABI_VERSION: "continuous_structured_streamfunction",
            STRUCTURED_NS_MMS_ABI_VERSION: "continuous_structured_streamfunction",
        }.get(hidden_abi)
        if hidden_manifest.get("truth_type") != expected_truth_type:
            failures.append("hidden_truth_type_invalid")
        if hidden_abi not in SUPPORTED_CONTINUOUS_MMS_TRUTH_CONTRACTS:
            failures.append("hidden_truth_continuous_mms_abi_invalid")
        failures.extend(_continuous_provenance_failures(hidden_manifest))
    if continuous_manifest:
        if continuous_manifest.get("continuous_mms_abi") not in SUPPORTED_CONTINUOUS_MMS_TRUTH_CONTRACTS:
            failures.append("continuous_mms_abi_invalid")
        failures.extend(_continuous_provenance_failures(continuous_manifest))
    if require_generator_metadata:
        metadata = _contract_json(path / "generator_metadata.json", failures, "generator_metadata")
        if metadata:
            if metadata.get("continuous_mms_abi") not in SUPPORTED_CONTINUOUS_MMS_TRUTH_CONTRACTS:
                failures.append("generator_continuous_mms_abi_invalid")
            if "continuous_mms_v" not in str(
                metadata.get("generator_backend_and_version") or ""
            ):
                failures.append("generator_backend_not_continuous_mms")
            if metadata.get("diagnostic_field_stack_allowed") is not False:
                failures.append("diagnostic_field_stack_enabled")
            if metadata.get("velocity_projection_enabled") is not False:
                failures.append("post_hoc_velocity_projection_enabled")
            if metadata.get("D0_distribution_gate_enabled") is not False:
                failures.append("d0_distribution_gate_enabled")
            failures.extend(_continuous_provenance_failures(metadata))
    return failures


def require_continuous_mms_case_contract(
    case_dir: str | Path,
    *,
    require_generator_metadata: bool = True,
) -> None:
    failures = continuous_mms_case_contract_failures(
        case_dir,
        require_generator_metadata=require_generator_metadata,
    )
    if failures:
        raise ValueError(f"continuous MMS contract failed for {Path(case_dir)}: {', '.join(failures)}")


def _contract_json(path: Path, failures: list[str], label: str) -> dict[str, Any]:
    if not path.exists():
        failures.append(f"missing_{label}")
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        failures.append(f"invalid_{label}")
        return {}
    if not isinstance(payload, dict):
        failures.append(f"invalid_{label}")
        return {}
    return payload


def _actual_boundary_coords(domain_bounds, times: np.ndarray, *, edge_points: int = 8) -> np.ndarray:
    x_min, x_max, y_min, y_max = (float(value) for value in domain_bounds)
    x = np.linspace(x_min, x_max, int(edge_points), dtype=np.float64)
    y = np.linspace(y_min, y_max, int(edge_points), dtype=np.float64)
    rows = []
    for time_value in np.asarray(times, dtype=np.float64):
        rows.extend((x_min, value, time_value) for value in y)
        rows.extend((x_max, value, time_value) for value in y)
        rows.extend((value, y_min, time_value) for value in x)
        rows.extend((value, y_max, time_value) for value in x)
    return np.asarray(rows, dtype=np.float64)


def _continuous_provenance_failures(payload: dict[str, Any]) -> list[str]:
    from agent_verifier.neural_generator.triplane_mms import (
        DEFAULT_SOURCE_DERIVATIVE_BACKEND,
        LEGACY_SOURCE_DERIVATION_SCHEMA_VERSION,
        REFERENCE_AUTOGRAD_BACKEND_ID,
        SOURCE_DERIVATION_SCHEMA_VERSION,
        continuous_mms_provenance,
    )

    failures = []
    continuous_mms_abi = payload.get("continuous_mms_abi")
    if continuous_mms_abi in SUPPORTED_STRUCTURED_NS_MMS_ABI_VERSIONS:
        from agent_verifier.neural_generator.structured_ns_answer_first import (
            StructuredNSProjectionConfig,
            structured_ns_mms_provenance,
        )

        try:
            structured_config = StructuredNSProjectionConfig.from_dict(
                payload["structured_projection_config"]
            )
            structured_config.validate()
        except (KeyError, TypeError, ValueError):
            return ["continuous_provenance_invalid:structured_projection_config"]
        expected_provenance = structured_ns_mms_provenance(
            structured_config,
            continuous_mms_abi=str(continuous_mms_abi),
        )
    else:
        expected_provenance = continuous_mms_provenance()
    allowed_values = {
        "source_derivation_schema_version": {
            SOURCE_DERIVATION_SCHEMA_VERSION,
            LEGACY_SOURCE_DERIVATION_SCHEMA_VERSION,
        },
        "source_derivative_backend": {
            DEFAULT_SOURCE_DERIVATIVE_BACKEND,
            REFERENCE_AUTOGRAD_BACKEND_ID,
        },
    }
    for name, expected in expected_provenance.items():
        actual = payload.get(name)
        if name in allowed_values:
            valid = actual in allowed_values[name]
        else:
            valid = actual == expected
        if not valid:
            failures.append(f"continuous_provenance_invalid:{name}")
    return failures


def _audit_indices(size: int, count: int) -> np.ndarray:
    if size <= 0:
        raise ValueError("continuous MMS materialized view has an empty coordinate axis")
    return np.unique(np.linspace(0, size - 1, min(size, count), dtype=np.int64))


def _gather_vector_view(array, t_indices, x_indices, y_indices) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)[np.ix_(t_indices, np.arange(2), x_indices, y_indices)]
    return np.moveaxis(values, 1, -1).reshape(-1, 2)


def _gather_scalar_view(array, t_indices, x_indices, y_indices) -> np.ndarray:
    return np.asarray(array, dtype=np.float64)[np.ix_(t_indices, x_indices, y_indices)].reshape(-1)


def generated_fields_mms_residual_rel(fields) -> float:
    residual = np.asarray(fields.residual, dtype=np.float64)
    forcing = np.asarray(fields.forcing, dtype=np.float64)
    velocity = np.asarray(fields.velocity, dtype=np.float64)
    return rms(residual) / (rms(forcing) + rms(velocity) + 1.0e-12)


def _check_metric_max(failures: list[str], metrics: dict[str, float], name: str, maximum: float) -> None:
    value = float(metrics.get(name, np.inf))
    if not np.isfinite(value) or value > float(maximum):
        failures.append(f"{name}>{float(maximum):.6g}")


def _failure_counts(results: list[VerificationResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for failure in result.failures:
            counts[failure] = counts.get(failure, 0) + 1
    return counts
