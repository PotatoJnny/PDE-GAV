"""Full public Case/session binding for the trusted stable GKW adapter."""

from __future__ import annotations

from typing import Any, Mapping

from pde_agent_verifier_solver.sdk import Case

from .contract import gkw_public_arrays, gkw_public_query_domain


def validate_gkw_public_session(case: Case, session: object) -> None:
    """Require a session for exactly the same public equation and oracle ABI."""

    import numpy as np

    if str(getattr(session, "case_id", "")) != case.case_id:
        raise ValueError("GKW public session case id differs from the Case")

    case_arrays = gkw_public_arrays(case)
    session_arrays = getattr(session, "arrays", None)
    if not isinstance(session_arrays, Mapping) or set(session_arrays) != set(
        case_arrays
    ):
        raise ValueError("GKW public session array keys differ from Case")
    for name in case_arrays:
        if not np.array_equal(
            np.asarray(session_arrays[name]), np.asarray(case_arrays[name])
        ):
            raise ValueError(f"GKW public session array {name} differs from Case")

    case_scalars = case.public_data.get("scalars")
    session_scalars = getattr(session, "scalars", None)
    if (
        not isinstance(case_scalars, Mapping)
        or not isinstance(session_scalars, Mapping)
        or set(session_scalars) != set(case_scalars)
    ):
        raise ValueError("GKW public session scalar keys differ from Case")
    for name in case_scalars:
        if not np.array_equal(
            np.asarray(session_scalars[name]), np.asarray(case_scalars[name])
        ):
            raise ValueError(f"GKW public session scalar {name} differs from Case")

    case_metadata = case.public_data.get("metadata")
    session_metadata = getattr(session, "metadata", None)
    if (
        not isinstance(case_metadata, Mapping)
        or not isinstance(session_metadata, Mapping)
        or _plain_json(session_metadata) != _plain_json(case_metadata)
    ):
        raise ValueError("GKW public session metadata differs from Case")

    case_oracle = case.public_data.get("oracle_protocol")
    session_oracle = getattr(session, "oracle_protocol", None)
    if (
        not isinstance(case_oracle, Mapping)
        or not isinstance(session_oracle, Mapping)
        or _plain_json(session_oracle) != _plain_json(case_oracle)
    ):
        raise ValueError("GKW public session oracle protocol differs from Case")
    gkw_public_query_domain(case)
    for name in ("query_source", "query_initial_h", "query_initial_phi"):
        if not callable(getattr(session, name, None)):
            raise TypeError(f"GKW public session must expose {name}")


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
    raise TypeError("GKW public session metadata is not JSON-compatible")
