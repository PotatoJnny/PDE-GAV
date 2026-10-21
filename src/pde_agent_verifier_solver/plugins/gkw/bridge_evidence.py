"""Audit-bound additive evidence for accepted stable GKW macro calls."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

from pde_agent_verifier_solver.sdk import (
    AnswerQuery,
    Case,
    CaseSource,
    CaseVisibility,
)

from .contract import (
    GKW_LEGACY_MACRO_OUTPUT_CONTRACT,
    GKW_MACRO_FIELDS,
    GKW_MACRO_MATCH_CONTRACT,
    GKW_STABLE_SOURCE_COMMIT,
    coerce_gkw_macro_output,
    gkw_answer_view_from_pairs,
    gkw_field_quadrature_weights,
    validate_gkw_compute_budget,
)
from .macro_matching import GKWMacroMatcher
from .public_contract import (
    gkw_json_values_equal_exact,
    validate_gkw_public_components,
)


GKW_LEGACY_MACRO_BRIDGE_SCHEMA = "gkw_legacy_macro_bridge_evidence_v2"

_EVIDENCE_KEYS = frozenset(
    {
        "schema",
        "case_id",
        "legacy_source_commit",
        "legacy_output_contract",
        "unified_contract",
        "used_for_winner_selection",
        "private_truth_persisted",
        "binding_authority",
        "accepted_call_id",
        "solver_bundle_sha256",
        "legacy_evaluation_sha256",
        "budget",
        "query",
        "commitments",
        "primary_error",
        "metrics",
        "passed",
        "failures",
    }
)
_QUERY_KEYS = frozenset(
    {
        "time_count",
        "field_row_count",
        "query_times_sha256",
        "field_indices_sha256",
    }
)
_COMMITMENT_KEYS = frozenset(
    {
        "budget_sha256",
        "public_packet_sha256",
        "field_weights_sha256",
        "prediction_sha256",
        "truth_sha256",
        "accepted_call_sha256",
    }
)
_METRIC_KEYS = frozenset(
    {
        "macro_match_relative_l2",
        "macro_relative_l2",
        "gkw_macro_match_v1_relative_l2",
        "legacy_fusion_gkw_aggregate_relative_l2",
        *(f"{name}_relative_l2" for name in GKW_MACRO_FIELDS),
    }
)


def evaluate_gkw_legacy_macro_match(
    *,
    case_id: str,
    prediction: Mapping[str, Any],
    truth: Mapping[str, Any],
    query_times: Any,
    field_indices: Any,
    public_arrays: Mapping[str, Any],
    public_scalars: Mapping[str, Any],
    public_metadata: Mapping[str, Any],
    budget: Mapping[str, object],
    solver_bundle_sha256: str,
    legacy_evaluation_sha256: str,
    accepted_call_id: str,
    legacy_aggregate_relative_l2: float,
) -> dict[str, object]:
    """Dual-score one controller-accepted call without storing private truth.

    The solver/evaluation digests and call id must come from controller-owned
    artifacts. This record commits to those identities; it is not a signature
    or an external authenticity anchor.
    """

    import numpy as np

    case_id = _nonempty_text(case_id, "case_id")
    accepted_call_id = _nonempty_text(accepted_call_id, "accepted_call_id")
    _require_digest(solver_bundle_sha256, "solver bundle")
    _require_digest(legacy_evaluation_sha256, "legacy evaluation")
    structured_budget = validate_gkw_compute_budget(budget)
    validate_gkw_public_components(
        public_arrays,
        public_scalars,
        public_metadata,
        expected_case_id=case_id,
    )

    raw_times = np.asarray(query_times)
    raw_indices = np.asarray(field_indices)
    if raw_times.dtype.kind not in "iuf" or not np.isfinite(raw_times).all():
        raise TypeError("GKW bridge query times must be finite and real")
    if not np.issubdtype(raw_indices.dtype, np.integer):
        raise TypeError("GKW bridge field indices must be integer")
    times = raw_times.astype(np.float64, copy=True)
    indices = raw_indices.astype(np.int64, copy=True)
    if (
        times.ndim != 1
        or len(times) <= 0
        or indices.ndim != 2
        or indices.shape != (len(indices), 3)
        or len(indices) <= 0
    ):
        raise ValueError("GKW bridge coordinates have invalid shapes")

    time_nodes = np.asarray(public_metadata["time_nodes"], dtype=np.float64)
    if np.any(times < time_nodes[0]) or np.any(times > time_nodes[-1]):
        raise ValueError(
            "GKW bridge query times leave the public interval"
        )

    prediction_values = coerce_gkw_macro_output(
        prediction, query_count=len(times), field_count=len(indices)
    )
    truth_values = coerce_gkw_macro_output(
        truth, query_count=len(times), field_count=len(indices)
    )
    query = AnswerQuery.grid(
        fields=GKW_MACRO_FIELDS,
        t=len(times),
        field_row=len(indices),
    )
    case = Case(
        case_id=case_id,
        source=CaseSource.GENERATED,
        visibility=CaseVisibility.HIDDEN,
        public_data={
            "equation": "GKW electrostatic gyrokinetics",
            "arrays": {
                name: np.asarray(value).copy()
                for name, value in public_arrays.items()
            },
            "scalars": dict(public_scalars),
            "metadata": dict(public_metadata),
            "macro_query": {
                "times": times.copy(),
                "field_indices": indices.copy(),
                "output_contract": GKW_LEGACY_MACRO_OUTPUT_CONTRACT,
            },
        },
    )
    prediction_view = gkw_answer_view_from_pairs(
        query,
        prediction_values,
        times=times,
        field_indices=indices,
    )
    truth_view = gkw_answer_view_from_pairs(
        query,
        truth_values,
        times=times,
        field_indices=indices,
    )
    matcher = GKWMacroMatcher()
    target = matcher.prepare_target(case, truth_view, query)
    result = matcher.match(case, prediction_view, target)
    legacy = float(legacy_aggregate_relative_l2)
    if not math.isfinite(legacy) or legacy < 0.0:
        raise ValueError("legacy GKW aggregate score must be finite")
    if not math.isclose(
        legacy, result.primary_error, rel_tol=1.0e-12, abs_tol=1.0e-12
    ):
        raise ValueError("legacy and unified GKW macro scores differ")

    query_payload = {
        "time_count": len(times),
        "field_row_count": len(indices),
        "query_times_sha256": _array_digest(times),
        "field_indices_sha256": _array_digest(indices),
    }
    commitments = {
        "budget_sha256": _json_digest(structured_budget),
        "public_packet_sha256": _public_packet_digest(
            public_arrays, public_scalars, public_metadata
        ),
        "field_weights_sha256": _array_digest(
            gkw_field_quadrature_weights(case, indices)
        ),
        "prediction_sha256": _field_mapping_digest(prediction_values),
        "truth_sha256": _field_mapping_digest(truth_values),
    }
    result_payload = {
        "primary_error": result.primary_error,
        "metrics": dict(result.metrics),
        "passed": result.passed,
        "failures": list(result.failures),
    }
    commitments["accepted_call_sha256"] = _accepted_call_digest(
        case_id=case_id,
        accepted_call_id=accepted_call_id,
        solver_bundle_sha256=solver_bundle_sha256,
        legacy_evaluation_sha256=legacy_evaluation_sha256,
        query=query_payload,
        commitments=commitments,
        result=result_payload,
    )
    evidence = {
        "schema": GKW_LEGACY_MACRO_BRIDGE_SCHEMA,
        "case_id": case_id,
        "legacy_source_commit": GKW_STABLE_SOURCE_COMMIT,
        "legacy_output_contract": GKW_LEGACY_MACRO_OUTPUT_CONTRACT,
        "unified_contract": matcher.contract_id,
        "used_for_winner_selection": False,
        "private_truth_persisted": False,
        "binding_authority": "controller_verified_artifacts_required",
        "accepted_call_id": accepted_call_id,
        "solver_bundle_sha256": solver_bundle_sha256,
        "legacy_evaluation_sha256": legacy_evaluation_sha256,
        "budget": structured_budget,
        "query": query_payload,
        "commitments": commitments,
        **result_payload,
    }
    validate_gkw_legacy_macro_evidence(evidence)
    return evidence


def validate_gkw_legacy_macro_evidence(
    evidence: Mapping[str, object],
) -> None:
    """Validate a v2 record without requiring the private prediction/truth."""

    if set(evidence) != _EVIDENCE_KEYS:
        raise ValueError("GKW bridge evidence keys changed")
    exact = {
        "schema": GKW_LEGACY_MACRO_BRIDGE_SCHEMA,
        "legacy_source_commit": GKW_STABLE_SOURCE_COMMIT,
        "legacy_output_contract": GKW_LEGACY_MACRO_OUTPUT_CONTRACT,
        "unified_contract": GKW_MACRO_MATCH_CONTRACT,
        "used_for_winner_selection": False,
        "private_truth_persisted": False,
        "binding_authority": "controller_verified_artifacts_required",
    }
    if not gkw_json_values_equal_exact(
        {name: evidence.get(name) for name in exact},
        exact,
    ):
        raise ValueError("GKW bridge evidence contract changed")
    case_id = _nonempty_text(evidence.get("case_id"), "case_id")
    accepted_call_id = _nonempty_text(
        evidence.get("accepted_call_id"), "accepted_call_id"
    )
    solver_digest = evidence.get("solver_bundle_sha256")
    evaluation_digest = evidence.get("legacy_evaluation_sha256")
    _require_digest(solver_digest, "solver bundle")
    _require_digest(evaluation_digest, "legacy evaluation")

    query = evidence.get("query")
    commitments = evidence.get("commitments")
    metrics = evidence.get("metrics")
    budget = evidence.get("budget")
    if (
        not isinstance(query, Mapping)
        or set(query) != _QUERY_KEYS
        or not isinstance(commitments, Mapping)
        or set(commitments) != _COMMITMENT_KEYS
        or not isinstance(metrics, Mapping)
        or set(metrics) != _METRIC_KEYS
        or not isinstance(budget, Mapping)
    ):
        raise ValueError("GKW bridge evidence is incomplete")
    for name in ("time_count", "field_row_count"):
        count = query.get(name)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("GKW bridge query count is invalid")
    for name in ("query_times_sha256", "field_indices_sha256"):
        _require_digest(query.get(name), f"query {name}")
    for name, digest in commitments.items():
        _require_digest(digest, f"commitment {name}")

    structured_budget = validate_gkw_compute_budget(budget)
    if commitments["budget_sha256"] != _json_digest(structured_budget):
        raise ValueError("GKW bridge budget commitment differs")
    primary = _finite_nonnegative(evidence.get("primary_error"), "primary error")
    fields = [
        _finite_nonnegative(
            metrics[f"{name}_relative_l2"], f"{name} relative L2"
        )
        for name in GKW_MACRO_FIELDS
    ]
    if (
        not math.isclose(
            primary, sum(fields) / len(fields), rel_tol=1.0e-12, abs_tol=1.0e-12
        )
    ):
        raise ValueError("GKW bridge primary metric is incoherent")
    for alias in (
        "macro_match_relative_l2",
        "macro_relative_l2",
        "gkw_macro_match_v1_relative_l2",
        "legacy_fusion_gkw_aggregate_relative_l2",
    ):
        if not math.isclose(
            _finite_nonnegative(metrics[alias], alias),
            primary,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("GKW bridge primary aliases disagree")
    passed = evidence.get("passed")
    failures = evidence.get("failures")
    if passed is not None or failures != []:
        raise ValueError("GKW ungated bridge outcome must be unset and empty")

    expected_call = _accepted_call_digest(
        case_id=case_id,
        accepted_call_id=accepted_call_id,
        solver_bundle_sha256=solver_digest,
        legacy_evaluation_sha256=evaluation_digest,
        query=query,
        commitments={
            name: commitments[name]
            for name in _COMMITMENT_KEYS
            if name != "accepted_call_sha256"
        },
        result={
            "primary_error": primary,
            "metrics": dict(metrics),
            "passed": passed,
            "failures": list(failures),
        },
    )
    if commitments["accepted_call_sha256"] != expected_call:
        raise ValueError("GKW bridge accepted-call commitment differs")


def _accepted_call_digest(
    *,
    case_id: str,
    accepted_call_id: str,
    solver_bundle_sha256: str,
    legacy_evaluation_sha256: str,
    query: Mapping[str, object],
    commitments: Mapping[str, object],
    result: Mapping[str, object],
) -> str:
    return _json_digest(
        {
            "case_id": case_id,
            "accepted_call_id": accepted_call_id,
            "solver_bundle_sha256": solver_bundle_sha256,
            "legacy_evaluation_sha256": legacy_evaluation_sha256,
            "query": dict(query),
            "commitments": dict(commitments),
            "result": dict(result),
        }
    )


def _public_packet_digest(
    arrays: Mapping[str, Any],
    scalars: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str:
    return _json_digest(
        {
            "arrays": {
                str(name): _array_digest(value)
                for name, value in sorted(arrays.items())
            },
            "scalars": _canonical_json(scalars),
            "metadata": _canonical_json(metadata),
        }
    )


def _field_mapping_digest(values: Mapping[str, Any]) -> str:
    return _json_digest(
        {name: _array_digest(values[name]) for name in GKW_MACRO_FIELDS}
    )


def _array_digest(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    if array.dtype.kind == "O":
        raise TypeError("GKW bridge cannot digest object arrays")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _json_digest(value: Any) -> str:
    encoded = json.dumps(
        _canonical_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(name): _canonical_json(item)
            for name, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _canonical_json(item())
    raise TypeError("GKW bridge value is not JSON-compatible")


def _nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"GKW bridge {label} must be nonempty text")
    return value


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"GKW bridge {label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"GKW bridge {label} must be finite and nonnegative")
    return result


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"GKW bridge {label} digest is invalid")
    return value
