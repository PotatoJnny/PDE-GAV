"""Controller-private plans for sealed macroscopic scoring queries.

The plan carries controller entropy, not candidate discretization policy. It
derives one stable seed per clean root, so a clean case and its perturbed child
use the same normalized draw. Exact queries are materialized into a prepared
cohort once and are only handed to a two-phase candidate after preparation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import secrets
from typing import Protocol

from pde_agent_verifier_solver.sdk.models import AnswerQuery, Case


_PROCESS_CONTROLLER_QUERY_ENTROPY = secrets.token_bytes(32)


class _MacroObjective(Protocol):
    def query(self, case: Case, profile: str) -> AnswerQuery: ...


@dataclass(frozen=True)
class ControllerMacroQueryPlan:
    """A frozen, backend-neutral query plan owned by the controller."""

    cohort_id: str
    profile: str
    controller_seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.cohort_id, str) or not self.cohort_id.strip():
            raise ValueError("query-plan cohort_id cannot be empty")
        if not isinstance(self.profile, str) or not self.profile.strip():
            raise ValueError("query-plan profile cannot be empty")
        if (
            isinstance(self.controller_seed, bool)
            or not isinstance(self.controller_seed, int)
            or self.controller_seed < 0
        ):
            raise ValueError("query-plan controller_seed must be nonnegative")

    def seed_for(self, case: Case) -> int:
        """Derive a replayable seed without making child perturbations drift."""

        if not isinstance(case, Case):
            raise TypeError("query-plan case must be a Case")
        root_id = case.parent_case_id or case.case_id
        if not isinstance(root_id, str) or not root_id.strip():
            raise ValueError("query-plan case root identity cannot be empty")
        payload = (
            "macro-controller-query-root-v1"
            f"\0{self.controller_seed}"
            f"\0{self.cohort_id}"
            f"\0{self.profile}"
            f"\0{root_id}"
        )
        return int.from_bytes(
            hashlib.sha256(payload.encode("utf-8")).digest()[:8],
            "big",
        ) & ((1 << 63) - 1)

    def build(self, objective: _MacroObjective, case: Case) -> AnswerQuery:
        """Build one query, using seeded capability when the objective has it."""

        planned = getattr(objective, "query_from_controller_seed", None)
        if callable(planned):
            query = planned(case, self.profile, self.seed_for(case))
        else:
            query = objective.query(case, self.profile)
        if not isinstance(query, AnswerQuery):
            raise TypeError("macro query plan must produce an AnswerQuery")
        return query


def controller_query_plan_seed(
    *,
    seed_base: int,
    cohort_id: str,
    profile: str,
    controller_entropy: bytes | None = None,
) -> int:
    """Derive a controller-private query stream from public replay state.

    The process-local entropy is intentionally absent from Agent requests and
    public campaign configuration. Supplying explicit entropy is reserved for
    trusted replay/audit tooling.
    """

    if isinstance(seed_base, bool) or not isinstance(seed_base, int) or seed_base < 0:
        raise ValueError("query-plan seed_base must be nonnegative")
    if not isinstance(cohort_id, str) or not cohort_id.strip():
        raise ValueError("query-plan cohort_id cannot be empty")
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError("query-plan profile cannot be empty")
    entropy = (
        _PROCESS_CONTROLLER_QUERY_ENTROPY
        if controller_entropy is None
        else controller_entropy
    )
    if not isinstance(entropy, bytes):
        raise TypeError("controller query entropy must be bytes")
    if len(entropy) < 16:
        raise ValueError("controller query entropy must contain at least 16 bytes")
    payload = (
        "macro-controller-query-plan-v2"
        f"\0{seed_base}"
        f"\0{cohort_id}"
        f"\0{profile}"
    ).encode("utf-8")
    return int.from_bytes(
        hmac.new(entropy, payload, hashlib.sha256).digest()[:8],
        "big",
    ) & ((1 << 63) - 1)


__all__ = ["ControllerMacroQueryPlan", "controller_query_plan_seed"]
