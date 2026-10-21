"""Build the phase-blind ``Case`` value passed across the candidate boundary.

The controller keeps the issued Case, including pairing and provenance, for
truth lookup and statistical accounting. Candidate programs need only the
physical question. In particular they must not be able to select a different
solver because the controller labelled a draw clean, perturbed, development,
selection, or terminal.
"""

from __future__ import annotations

import hashlib
import hmac
from copy import deepcopy
from typing import Mapping

from pde_agent_verifier_solver.sdk import Case, CaseSource, CaseVisibility


CANDIDATE_CASE_ID_PREFIX = "candidate-case-"
_CANDIDATE_CASE_ID_DOMAIN = b"pde-agent-verifier/candidate-case/v1"

# These are controller/distribution annotations, not coefficients of the
# public PDE. The answer-first backend has already rederived forcing and
# initial/boundary data before this projection is made.
_HIDDEN_CONTROL_KEYS = frozenset(
    {
        "answer_first_perturbation_applied",
        "anchor_case_id",
        "case_source",
        "cohort",
        "cohort_id",
        "cohort_kind",
        "controller_cohort",
        "dataset_split",
        "development_cohort",
        "distribution_lineage",
        "draw_index",
        "draw_kind",
        "evaluation_reference",
        "generation_kind",
        "generation_round",
        "generation_variant",
        "generator",
        "generator_mode",
        "is_perturbed",
        "logical_cohort_id",
        "logical_root_count",
        "logical_root_index",
        "logical_root_offset",
        "observation_kind",
        "parent_case_id",
        "perturbation",
        "perturbation_applied",
        "perturbation_metadata",
        "policy_variant",
        "robust_variant",
        "reference_kind",
        "round",
        "round_index",
        "sample_kind",
        "source_round_index",
        "tags",
        "target_round_index",
        "visibility",
    }
)
_CONTROL_VARIANT_VALUES = frozenset(
    {"base", "clean", "generated", "perturbed", "perturbation"}
)


def candidate_case_id(case_id: str, *, seed: int = 0) -> str:
    """Return a replay-stable, seed-scoped id with no phase/kind encoding."""

    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("candidate Case id source must be nonempty text")
    if case_id.startswith(CANDIDATE_CASE_ID_PREFIX):
        return case_id
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("candidate Case id seed must be a non-negative integer")
    key = hmac.new(
        _CANDIDATE_CASE_ID_DOMAIN,
        str(seed).encode("ascii"),
        hashlib.sha256,
    ).digest()
    digest = hmac.new(
        key,
        case_id.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{CANDIDATE_CASE_ID_PREFIX}{digest}"


def require_controller_case_id(case_id: str) -> None:
    """Reject ids from the candidate namespace at a controller boundary."""

    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("controller Case id must be nonempty text")
    if case_id.startswith(CANDIDATE_CASE_ID_PREFIX):
        raise ValueError(
            "controller Case id uses the reserved candidate-case prefix"
        )


def candidate_facing_case(case: Case, *, id_seed: int = 0) -> Case:
    """Project an issued controller Case onto the phase-blind public ABI."""

    if not isinstance(case, Case):
        raise TypeError("candidate Case projection requires a Case")
    opaque_id = candidate_case_id(case.case_id, seed=id_seed)
    return Case(
        case_id=opaque_id,
        # The enum remains for backwards-compatible candidate ABI typing, but
        # its value is deliberately constant for every manufactured question.
        source=CaseSource.GENERATED,
        visibility=CaseVisibility.DEVELOPMENT,
        public_data=_sanitize_public_value(case.public_data, opaque_id),
        round_index=None,
        parent_case_id=None,
        tags=(),
    )


def _sanitize_public_value(value: object, opaque_id: str) -> object:
    if isinstance(value, Mapping):
        sanitized: dict[object, object] = {}
        for key, item in value.items():
            if isinstance(key, str):
                normalized_key = key.strip().lower()
                if normalized_key == "case_id":
                    sanitized[key] = opaque_id
                    continue
                if normalized_key in _HIDDEN_CONTROL_KEYS:
                    continue
                if (
                    normalized_key == "variant"
                    and str(item).strip().lower() in _CONTROL_VARIANT_VALUES
                ):
                    continue
            sanitized[deepcopy(key)] = _sanitize_public_value(item, opaque_id)
        return sanitized
    if isinstance(value, tuple):
        return tuple(_sanitize_public_value(item, opaque_id) for item in value)
    if isinstance(value, list):
        return [_sanitize_public_value(item, opaque_id) for item in value]
    # Candidate projections must never share a mutable leaf (notably a NumPy
    # array) with the controller-owned Case. ``Case`` itself is frozen, but a
    # plugin's public_data values need not be.
    return deepcopy(value)


__all__ = [
    "CANDIDATE_CASE_ID_PREFIX",
    "candidate_case_id",
    "candidate_facing_case",
    "require_controller_case_id",
]
