"""Shared, versioned Plasma macro-transport scoring and reference calibration.

The five physical errors are dimensionless all-point RMS errors.  No candidate
data enter calibration, and this module deliberately supplies no default floors.
The one frozen calibration is shared by strong VPFP, weak VPFP and Artemis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from math import isfinite, prod, sqrt
from pathlib import Path
from typing import ClassVar, Iterable, Mapping

import numpy as np

from pde_agent_verifier_solver.sdk import (
    AnswerQuery, AnswerView, Case, MacroMatchResult, MacroTarget,
    PanelCard, ProgramInfo, QueryKind,
)


TRANSPORT_PROTOCOL_ID = "plasma_macro_transport_v2"
TRANSPORT_FIELDS = ("density", "electric_field", "particle_flux", "pressure", "heat_flux")
TRANSPORT_WEIGHTS = dict(zip(TRANSPORT_FIELDS, (0.125, 0.125, 0.25, 0.25, 0.25)))
VPFP_TRANSPORT_FIELDS = (
    "rho", "electric_x", "electric_y", "momentum_x", "momentum_y",
    "pressure_xx", "pressure_xy", "pressure_yy", "heat_flux_x", "heat_flux_y",
)
VP1D_TRANSPORT_FIELDS = ("M0", "M1", "M2", "M3", "E")
CALIBRATION_METHOD = "equal_family_stratum_reference_rms_v1"
CALIBRATION_FAMILIES = ("strong", "weak", "artemis")
FAILURE_POLICY = "failed_case_disqualifies_candidate_no_success_subset_official_mean_v1"
TRANSPORT_DEFINITIONS = {
    "density": "n=integral(f dv); n>0 in both prediction and reference",
    "electric_field": "self-consistent E in the equation's acceleration convention",
    "particle_flux": "j=integral(v f dv); unit particle mass",
    "pressure": "P=integral((v-u)(v-u)^T f dv); u=j/n from each own answer",
    "heat_flux": "qh=0.5*integral(|v-u|^2*(v-u)*f dv); u=j/n from each own answer",
    "norm": "scalar absolute, vector Euclidean, full pressure Frobenius",
    "average": "uniform over all evaluation points before normalization",
    "error": "sqrt(mean(norm(pred-ref)^2)/(mean(norm(ref)^2)+physical_floor^2))",
    "units": "n0, E0, n0*v0, n0*v0^2, n0*v0^3; floors dimensionless",
    "aggregation": "each case scored first; campaign uses its frozen case mixture",
}


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _unit_scales(units: Mapping[str, object]) -> dict[str, float]:
    if set(units) != {"density", "velocity", "electric_field"}:
        raise ValueError("transport units require density, velocity, and electric_field")
    n0, v0, e0 = (float(units[name]) for name in ("density", "velocity", "electric_field"))
    if not all(isfinite(value) and value > 0.0 for value in (n0, v0, e0)):
        raise ValueError("transport unit scales must be finite and positive")
    scales = dict(zip(TRANSPORT_FIELDS, (n0, e0, n0 * v0, n0 * v0**2, n0 * v0**3)))
    if not all(isfinite(value) and value > 0.0 for value in scales.values()):
        raise ValueError("transport derived unit scales overflow or underflow")
    return scales


@dataclass(frozen=True)
class TransportCalibration:
    """Immutable checked calibration asset, including its scientific identity.

    Construct with :meth:`from_dict` or :func:`fit_calibration`.  The serialized
    payload is immutable; returned dictionaries are copies, not mutable state.
    """

    _payload_json: str
    _objective_digest: str = field(init=False, repr=False)
    _physical_floor_items: tuple[tuple[str, float], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        payload = json.loads(self._payload_json)
        if not isinstance(payload, dict):
            raise ValueError("transport calibration must be a JSON object")
        if payload.get("schema_version") != 1 or payload.get("protocol_id") != TRANSPORT_PROTOCOL_ID:
            raise ValueError("transport calibration uses an unsupported protocol version")
        if payload.get("status") != "frozen" or payload.get("scope") != "shared_plasma":
            raise ValueError("transport calibration must be frozen and shared across all Plasma families")
        scales = _unit_scales(payload.get("units", {}))
        floors = payload.get("floors", {})
        if set(floors) != set(TRANSPORT_FIELDS) or not all(
            isfinite(float(value)) and float(value) > 0.0 for value in floors.values()
        ):
            raise ValueError("all five calibrated transport floors must be finite and positive")
        calibration = payload.get("calibration", {})
        if calibration.get("method") != CALIBRATION_METHOD:
            raise ValueError("transport calibration method is not the frozen reference-RMS rule")
        if not calibration.get("provenance") or calibration.get("source_role") != "independent_reference_calibration":
            raise ValueError("transport calibration requires independent reference provenance")
        families = calibration.get("families", {})
        if set(families) != set(CALIBRATION_FAMILIES):
            raise ValueError("shared transport calibration requires strong, weak, and Artemis references")
        roots: set[str] = set()
        checked_family_means = []
        for name, family in families.items():
            if int(family.get("case_count", 0)) <= 0 or not family.get("strata"):
                raise ValueError(f"transport calibration family {name} has no reference cases")
            case_count = 0
            checked_stratum_means = []
            for stratum in family["strata"].values():
                records = stratum.get("records", ())
                if not records or int(stratum.get("case_count", 0)) != len(records):
                    raise ValueError("transport calibration stratum case counts are inconsistent")
                case_count += len(records)
                for record in records:
                    root = record.get("root_id")
                    if not isinstance(root, str) or not root or root in roots:
                        raise ValueError("transport calibration root samples must be unique")
                    roots.add(root)
                    moments = record.get("mean_squares", {})
                    if int(record.get("sample_count", 0)) <= 0 or set(moments) != set(TRANSPORT_FIELDS) or any(
                        not isfinite(float(value)) or float(value) < 0.0 for value in moments.values()
                    ):
                        raise ValueError("transport reference mean-square diagnostics are invalid")
                means = {key: float(np.mean([record["mean_squares"][key] for record in records]))
                         for key in TRANSPORT_FIELDS}
                self._check_mean_squares(means, stratum.get("mean_squares", {}))
                checked_stratum_means.append(means)
            if case_count != int(family["case_count"]):
                raise ValueError("transport calibration family case count is inconsistent")
            means = {key: float(np.mean([item[key] for item in checked_stratum_means])) for key in TRANSPORT_FIELDS}
            self._check_mean_squares(means, family.get("mean_squares", {}))
            checked_family_means.append(means)
        expected_floors = {key: sqrt(float(np.mean([item[key] for item in checked_family_means])))
                           for key in TRANSPORT_FIELDS}
        self._check_mean_squares(expected_floors, floors)
        physical_floors = {name: float(floors[name]) * scales[name] for name in TRANSPORT_FIELDS}
        if any(not isfinite(value) or value <= 0.0 for value in physical_floors.values()):
            raise ValueError("transport physical floors overflow or underflow")
        supplied_hash = payload.pop("objective_hash", None)
        digest = self._hash_payload(payload)
        if supplied_hash is not None and supplied_hash != digest:
            raise ValueError("transport calibration objective hash mismatch")
        object.__setattr__(self, "_payload_json", _json(payload))
        object.__setattr__(self, "_objective_digest", digest)
        object.__setattr__(self, "_physical_floor_items", tuple(physical_floors.items()))

    @staticmethod
    def _check_mean_squares(expected: Mapping[str, float], actual: Mapping[str, object]) -> None:
        if set(actual) != set(TRANSPORT_FIELDS) or any(
            not isfinite(float(actual[name]))
            or not np.isclose(float(actual[name]), value, rtol=1e-12, atol=0.0)
            for name, value in expected.items()
        ):
            raise ValueError("transport calibration values do not match the frozen reference-RMS rule")

    @staticmethod
    def _hash_payload(payload: Mapping[str, object]) -> str:
        semantic = {
            "asset": payload, "definitions": TRANSPORT_DEFINITIONS,
            "weights": TRANSPORT_WEIGHTS, "failure_policy": FAILURE_POLICY,
        }
        return hashlib.sha256(_json(semantic).encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TransportCalibration":
        return cls(_json(dict(payload)))

    def to_dict(self) -> dict[str, object]:
        payload = json.loads(self._payload_json)
        payload["objective_hash"] = self.objective_hash
        return payload

    @property
    def objective_hash(self) -> str:
        return self._objective_digest

    @property
    def contract_id(self) -> str:
        return f"{TRANSPORT_PROTOCOL_ID}:{self.objective_hash}"

    @property
    def objective_identity(self) -> dict[str, object]:
        payload = self.to_dict()
        return {
            "protocol_id": TRANSPORT_PROTOCOL_ID, "objective_hash": self.objective_hash,
            "contract_id": self.contract_id, "weights": dict(TRANSPORT_WEIGHTS),
            "floors": payload["floors"], "units": payload["units"],
            "definitions": dict(TRANSPORT_DEFINITIONS), "failure_policy": FAILURE_POLICY,
        }

    @property
    def physical_floors(self) -> dict[str, float]:
        return dict(self._physical_floor_items)


Calibration = TransportCalibration


def load_transport_calibration(
    value: TransportCalibration | Mapping[str, object] | str | Path,
) -> TransportCalibration:
    if isinstance(value, TransportCalibration):
        return value
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("transport v2 requires an explicit frozen calibration mapping or asset path")
    return TransportCalibration.from_dict(value)


def _raw_fields(values: Mapping[str, object] | AnswerView, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    fields = values.fields if isinstance(values, AnswerView) else values
    if set(fields) != set(names):
        raise ValueError("transport answer fields do not match the physical output contract")
    arrays = {name: np.asarray(fields[name], dtype=np.float64) for name in names}
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1 or any(array.ndim != 1 or array.size == 0 for array in arrays.values()):
        raise ValueError("transport fields require a common nonempty flattened sample axis")
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("transport answer contains non-finite values")
    return arrays


def canonicalize_1d(values: Mapping[str, object] | AnswerView) -> dict[str, np.ndarray]:
    """Convert raw M0..M3,E to own-centered n,E,j,P,qh, in physical units."""
    fields = _raw_fields(values, VP1D_TRANSPORT_FIELDS)
    n, j, m2, m3 = (fields[name] for name in ("M0", "M1", "M2", "M3"))
    if np.any(n <= 0.0):
        raise ValueError("transport density must be strictly positive; candidates are not clamped")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        u = j / n
        pressure = m2 - j * u
        heat = 0.5 * (m3 - 3.0 * u * m2 + 2.0 * n * u**3)
    canonical = {
        "density": n[:, None], "electric_field": fields["E"][:, None],
        "particle_flux": j[:, None], "pressure": pressure[:, None], "heat_flux": heat[:, None],
    }
    return _validate_canonical(canonical)


def canonicalize_2d(values: Mapping[str, object] | AnswerView) -> dict[str, np.ndarray]:
    """Pack centered 2D fields; repeating Pxy implements the full Frobenius norm."""
    fields = _raw_fields(values, VPFP_TRANSPORT_FIELDS)
    canonical = {
        "density": fields["rho"][:, None],
        "electric_field": np.column_stack((fields["electric_x"], fields["electric_y"])),
        "particle_flux": np.column_stack((fields["momentum_x"], fields["momentum_y"])),
        "pressure": np.column_stack((fields["pressure_xx"], fields["pressure_xy"], fields["pressure_xy"], fields["pressure_yy"])),
        "heat_flux": np.column_stack((fields["heat_flux_x"], fields["heat_flux_y"])),
    }
    return _validate_canonical(canonical)


def _validate_canonical(values: Mapping[str, object]) -> dict[str, np.ndarray]:
    if set(values) != set(TRANSPORT_FIELDS):
        raise ValueError("canonical transport data require all five physical quantities")
    arrays = {name: np.asarray(values[name], dtype=np.float64) for name in TRANSPORT_FIELDS}
    if any(array.ndim != 2 or array.shape[0] == 0 for array in arrays.values()):
        raise ValueError("canonical transport arrays must have (sample, component) shape")
    if len({array.shape[0] for array in arrays.values()}) != 1:
        raise ValueError("canonical transport sample counts differ")
    dimension = arrays["particle_flux"].shape[1]
    expected = {"density": 1, "electric_field": dimension, "particle_flux": dimension,
                "pressure": dimension**2, "heat_flux": dimension}
    if dimension not in (1, 2) or any(arrays[name].shape[1] != count for name, count in expected.items()):
        raise ValueError("canonical transport components do not describe 1D1V or 2D2V")
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("canonical transport values are non-finite")
    if np.any(arrays["density"] <= 0.0):
        raise ValueError("transport density must be strictly positive; candidates are not clamped")
    if dimension == 2:
        # Canonicalization duplicates packed Pxy exactly, but an independently
        # rotated full tensor may differ by ordinary FP64 contraction error.
        tensor_scale = np.max(np.abs(arrays["pressure"]), axis=1)
        discrepancy = np.abs(arrays["pressure"][:, 1] - arrays["pressure"][:, 2])
        if np.any(discrepancy > 64.0 * np.finfo(np.float64).eps * tensor_scale):
            raise ValueError("canonical pressure tensor must be symmetric within FP64 roundoff")
    return arrays


def fit_calibration(
    rows: Iterable[Mapping[str, object]], *, units: Mapping[str, object],
    provenance: Mapping[str, object],
) -> TransportCalibration:
    """Fit a reference-only, equal-family/equal-stratum RMS floor asset.

    Each row has ``family`` (strong/weak/artemis), ``stratum``, globally unique
    ``root_id``, and ``values`` from canonicalize_1d/2d.  Every case contributes
    its own all-point mean-square norm, regardless of its number of points.
    Rows must be root-isolated from both training and candidate evaluation;
    their provenance is recorded, not inferred from the numerical values.
    """
    scales = _unit_scales(units)
    if not provenance:
        raise ValueError("reference calibration requires provenance")
    groups: dict[str, dict[str, list[dict[str, object]]]] = {}
    roots: set[str] = set()
    for row in rows:
        family, stratum, root = (str(row.get(key, "")) for key in ("family", "stratum", "root_id"))
        if family not in CALIBRATION_FAMILIES or not stratum or not root:
            raise ValueError("each calibration row needs a valid family, stratum, and root_id")
        if root in roots:
            raise ValueError("calibration root samples must be unique")
        roots.add(root)
        values = _validate_canonical(row["values"])
        with np.errstate(over="ignore", invalid="ignore"):
            moments = {name: float(np.mean(np.sum((values[name] / scales[name]) ** 2, axis=1)))
                       for name in TRANSPORT_FIELDS}
        if any(not isfinite(value) for value in moments.values()):
            raise ValueError("reference calibration mean-square norm is non-finite")
        groups.setdefault(family, {}).setdefault(stratum, []).append({
            "root_id": root, "sample_count": int(values["density"].shape[0]), "mean_squares": moments,
        })
    if set(groups) != set(CALIBRATION_FAMILIES):
        raise ValueError("calibration must include independent strong, weak, and Artemis references")
    families: dict[str, object] = {}
    family_means = []
    for family in CALIBRATION_FAMILIES:
        strata = {}
        for stratum, records in sorted(groups[family].items()):
            strata[stratum] = {
                "case_count": len(records), "records": records,
                "mean_squares": {name: float(np.mean([record["mean_squares"][name] for record in records]))
                                 for name in TRANSPORT_FIELDS},
            }
        means = {name: float(np.mean([item["mean_squares"][name] for item in strata.values()]))
                 for name in TRANSPORT_FIELDS}
        family_means.append(means)
        families[family] = {"case_count": sum(item["case_count"] for item in strata.values()),
                            "strata": strata, "mean_squares": means}
    floors = {name: sqrt(float(np.mean([item[name] for item in family_means]))) for name in TRANSPORT_FIELDS}
    if any(not isfinite(value) or value <= 0.0 for value in floors.values()):
        raise ValueError("zero or non-finite reference RMS requires an independently justified physical scale; no epsilon fallback")
    return TransportCalibration.from_dict({
        "schema_version": 1, "protocol_id": TRANSPORT_PROTOCOL_ID, "status": "frozen", "scope": "shared_plasma",
        "units": {name: float(value) for name, value in units.items()}, "floors": floors,
        "calibration": {"method": CALIBRATION_METHOD, "source_role": "independent_reference_calibration",
                        "provenance": dict(provenance), "families": families},
    })


def score_transport(
    prediction: Mapping[str, object], reference: Mapping[str, object],
    calibration: TransportCalibration | Mapping[str, object] | str | Path,
) -> dict[str, float]:
    """Return five physical errors, four block errors and the fixed total S."""
    asset = load_transport_calibration(calibration)
    observed, expected = _validate_canonical(prediction), _validate_canonical(reference)
    floors = asset.physical_floors
    errors = {}
    diagnostics = {}
    for name in TRANSPORT_FIELDS:
        if observed[name].shape != expected[name].shape:
            raise ValueError("prediction and reference transport shapes differ")
        with np.errstate(over="ignore", invalid="ignore"):
            error_sq = float(np.mean(np.sum((observed[name] - expected[name]) ** 2, axis=1)))
            truth_sq = float(np.mean(np.sum(expected[name] ** 2, axis=1)))
            denominator_sq = truth_sq + floors[name] ** 2
        if not all(isfinite(value) for value in (error_sq, truth_sq, denominator_sq)) or denominator_sq <= 0.0:
            raise ValueError("transport error overflowed; candidate cannot receive a finite official score")
        errors[name] = sqrt(error_sq / denominator_sq)
        diagnostics[f"{name}_rms_error"] = sqrt(error_sq)
        diagnostics[f"{name}_reference_rms"] = sqrt(truth_sq)
        diagnostics[f"{name}_physical_floor"] = floors[name]
        diagnostics[f"{name}_floor_fraction"] = floors[name] ** 2 / denominator_sq
    total = sum(TRANSPORT_WEIGHTS[name] * errors[name] for name in TRANSPORT_FIELDS)
    return {
        "macro_match_relative_l2": total, "macro_relative_l2": total,
        TRANSPORT_PROTOCOL_ID: total,
        **{f"{name}_normalized_rms": value for name, value in errors.items()},
        "charge_field_block_error": 0.5 * (errors["density"] + errors["electric_field"]),
        "particle_transport_block_error": errors["particle_flux"],
        "thermal_stress_block_error": errors["pressure"],
        "heat_transport_block_error": errors["heat_flux"],
        **diagnostics,
    }


@dataclass(frozen=True)
class PlasmaTransportMacroMatcher:
    """SDK-compatible shared scorer; subclasses supply only dimensional ABI."""

    calibration: TransportCalibration | Mapping[str, object] | str | Path
    maximum_relative_l2: float | None = None
    dimension: ClassVar[int]
    fields: ClassVar[tuple[str, ...]]
    info: ClassVar[ProgramInfo]

    def __post_init__(self) -> None:
        object.__setattr__(self, "calibration", load_transport_calibration(self.calibration))
        if self.maximum_relative_l2 is not None and (
            not isfinite(float(self.maximum_relative_l2)) or float(self.maximum_relative_l2) < 0.0
        ):
            raise ValueError("transport tolerance must be finite and nonnegative")

    @property
    def contract_id(self) -> str:
        return self.calibration.contract_id

    @property
    def objective_hash(self) -> str:
        return self.calibration.objective_hash

    @property
    def objective_identity(self) -> dict[str, object]:
        return self.calibration.objective_identity

    def _canonicalize(self, view: AnswerView) -> dict[str, np.ndarray]:
        return canonicalize_1d(view) if self.dimension == 1 else canonicalize_2d(view)

    def prepare_target(self, case: Case, truth: AnswerView, query: AnswerQuery) -> MacroTarget:
        if set(query.fields) != set(self.fields):
            raise ValueError("transport target query fields differ from the contract")
        target = MacroTarget(case.case_id, self.contract_id, query, truth)
        self._validate_coordinates(truth, query)
        self._canonicalize(truth)
        return target

    def match(self, case: Case, prediction: AnswerView, target: MacroTarget) -> MacroMatchResult:
        if case.case_id != target.case_id:
            raise ValueError("transport target belongs to a different case")
        if target.contract_id != self.contract_id:
            raise ValueError("transport target uses a different objective version or calibration")
        if target.sample_weights and len(set(target.sample_weights)) != 1:
            raise ValueError("transport v2 uses uniform all-point RMS, not nonuniform sample weights")
        self._validate_coordinates(target.values, target.query)
        self._validate_coordinates(prediction, target.query)
        if prediction.shape != target.values.shape or prediction.axes != target.values.axes or prediction.points != target.values.points:
            raise ValueError("transport prediction coordinates differ from the target")
        metrics = score_transport(self._canonicalize(prediction), self._canonicalize(target.values), self.calibration)
        primary = metrics[TRANSPORT_PROTOCOL_ID]
        passed = None if self.maximum_relative_l2 is None else primary <= float(self.maximum_relative_l2)
        if self.maximum_relative_l2 is not None:
            metrics["macro_match_tolerance"] = float(self.maximum_relative_l2)
        return MacroMatchResult(case.case_id, self.contract_id, primary, metrics, passed,
                                () if passed is not False else ("macro transport error exceeds configured tolerance",))

    def result_card(self, result: MacroMatchResult) -> PanelCard:
        if result.contract_id != self.contract_id:
            raise ValueError("cannot render another transport objective or calibration")
        names = [f"{name}_normalized_rms" for name in TRANSPORT_FIELDS] + [
            "charge_field_block_error", "particle_transport_block_error", "thermal_stress_block_error", "heat_transport_block_error",
        ]
        return PanelCard(
            "Plasma macro transport v2",
            "Five own-centered physical quantities, four equally weighted blocks; lower is better.",
            {TRANSPORT_PROTOCOL_ID: result.primary_error, **{name: result.metrics[name] for name in names},
             "objective_hash": self.objective_hash, "gate": "not configured" if result.passed is None else result.passed},
        )

    def _validate_coordinates(self, view: AnswerView, query: AnswerQuery) -> None:
        if view.kind is not query.kind or set(view.fields) != set(self.fields):
            raise ValueError("transport answer query kind or fields differ from the contract")
        lengths = {len(values) for values in view.fields.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0 or prod(view.shape) != next(iter(lengths)):
            raise ValueError("transport answer field sample counts or shape differ")
        if query.kind is QueryKind.POINTS:
            if view.points != query.points or view.shape != (len(query.points),):
                raise ValueError("transport point coordinates differ from the query")
            return
        if set(view.axes) != set(query.resolution):
            raise ValueError("transport grid axes differ from the query")
        for name, count in query.resolution.items():
            axis = np.asarray(view.axes[name], dtype=np.float64)
            if axis.ndim != 1 or axis.size != int(count) or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0.0):
                raise ValueError("transport grid axis size, ordering or finiteness is invalid")
        if view.shape != tuple(len(values) for values in view.axes.values()):
            raise ValueError("transport grid shape differs from declared axes")
