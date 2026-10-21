"""Small public contracts for choosing development Cases during one round.

The shared framework decides *which* source to ask for and enforces the
configured total Case limit, when one is set.  The PDE plugin still owns all scientific details of generation and
perturbation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Mapping, Protocol

from .models import Case, CaseSource, CaseVisibility


class SamplingStrategy(str, Enum):
    """Who chooses the development Cases shown to an Agent."""

    AGENT_ADAPTIVE = "agent_adaptive"
    FIXED_RECIPE = "fixed_recipe"


class SampleSource(str, Enum):
    """Clean Case sources exposed by the shared sampling interface."""

    ORIGINAL = "original"
    BASE = "base"
    GUIDED = "guided"


class RecoverableSamplingRejection(RuntimeError):
    """A Generator declined one random draw without invalidating the Run.

    Plugins attach opaque JSON audit data; the shared sampler decides which
    logical root to retry and never interprets equation-specific gates.
    """

    def __init__(self, message: str, *, audit: Mapping[str, object]) -> None:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("sampling rejection message cannot be empty")
        if not isinstance(audit, Mapping):
            raise TypeError("sampling rejection audit must be a mapping")
        try:
            canonical = json.dumps(
                dict(audit),
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "sampling rejection audit must be finite JSON data"
            ) from exc
        self.audit = json.loads(canonical)
        super().__init__(message)


@dataclass(frozen=True)
class SampleCasesRequest:
    """Ask for a fresh batch from one clean Case source."""

    source: SampleSource
    count: int

    def __post_init__(self) -> None:
        if not isinstance(self.source, SampleSource):
            raise TypeError("sample source must be a SampleSource")
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise TypeError("sample count must be an integer")
        if self.count < 1:
            raise ValueError("sample count must be positive")


@dataclass(frozen=True)
class PerturbCasesRequest:
    """Ask the PDE Generator to perturb one already-issued clean Case each.

    There are deliberately no amplitude, family, or other physics parameters
    here.  Those choices belong to the Generator plugin and its user settings.
    """

    parent_case_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        parent_case_ids = tuple(self.parent_case_ids)
        if not parent_case_ids:
            raise ValueError("perturbation needs at least one parent Case")
        if any(
            not isinstance(case_id, str) or not case_id.strip()
            for case_id in parent_case_ids
        ):
            raise ValueError("perturbation parent Case ids must be nonempty strings")
        if len(parent_case_ids) != len(set(parent_case_ids)):
            raise ValueError("perturbation parent Case ids must be unique")
        object.__setattr__(self, "parent_case_ids", parent_case_ids)


@dataclass(frozen=True)
class PairedSampleRequest:
    """Ask for fresh clean roots and one perturbed child for every root.

    The operation is atomic from the caller's point of view. A recoverable
    child rejection retries only its logical root slot, so accepted pairs in
    other slots are retained.
    """

    source: SampleSource
    count: int
    max_attempts_per_root: int = 16

    def __post_init__(self) -> None:
        if not isinstance(self.source, SampleSource):
            raise TypeError("paired sample source must be a SampleSource")
        if self.source is SampleSource.ORIGINAL:
            raise ValueError("paired sampling requires base or guided roots")
        for value, label in (
            (self.count, "paired sample count"),
            (self.max_attempts_per_root, "paired sampling maximum attempts"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{label} must be an integer")
            if value < 1:
                raise ValueError(f"{label} must be positive")


@dataclass(frozen=True)
class PairedSamplingAttemptAudit:
    """Controller-owned evidence for one logical paired-sampling attempt."""

    agent_id: str
    round_index: int
    purpose: str
    source: SampleSource
    operation_index: int
    root_index: int
    attempt_index: int
    max_attempts_per_root: int
    clean_seed: int | None
    perturbed_seed: int
    root_case_id: str | None
    child_case_id: str | None
    accepted: bool
    rejection_audit: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id.strip():
            raise ValueError("sampling audit agent_id cannot be empty")
        if not isinstance(self.purpose, str) or not self.purpose.strip():
            raise ValueError("sampling audit purpose cannot be empty")
        if not isinstance(self.source, SampleSource):
            raise TypeError("sampling audit source must be a SampleSource")
        if self.source is SampleSource.ORIGINAL:
            raise ValueError("paired sampling audit cannot use original roots")
        for value, label in (
            (self.round_index, "sampling audit round index"),
            (self.operation_index, "sampling audit operation index"),
            (self.root_index, "sampling audit root index"),
            (self.attempt_index, "sampling audit attempt index"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{label} must be an integer")
            if value < 0:
                raise ValueError(f"{label} cannot be negative")
        if (
            isinstance(self.max_attempts_per_root, bool)
            or not isinstance(self.max_attempts_per_root, int)
            or self.max_attempts_per_root < 1
        ):
            raise ValueError("sampling audit maximum attempts must be positive")
        for seed, label in (
            (self.clean_seed, "clean seed"),
            (self.perturbed_seed, "perturbed seed"),
        ):
            if seed is None and label == "clean seed":
                continue
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError(f"sampling audit {label} must be nonnegative")
        if self.root_case_id is not None and (
            not isinstance(self.root_case_id, str) or not self.root_case_id.strip()
        ):
            raise ValueError("sampling audit root Case id cannot be empty")
        if self.child_case_id is not None and (
            not isinstance(self.child_case_id, str) or not self.child_case_id.strip()
        ):
            raise ValueError("sampling audit child Case id cannot be empty")
        if not isinstance(self.accepted, bool):
            raise TypeError("sampling audit accepted flag must be a boolean")
        if not isinstance(self.rejection_audit, Mapping):
            raise TypeError("sampling rejection audit must be a mapping")
        try:
            canonical = json.dumps(
                dict(self.rejection_audit),
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "sampling rejection audit must be finite JSON data"
            ) from exc
        object.__setattr__(self, "rejection_audit", json.loads(canonical))


@dataclass(frozen=True)
class SampleBatch:
    """Agent-visible Cases; remaining=None explicitly means no cumulative quota."""

    cases: tuple[Case, ...]
    remaining: int | None

    def __post_init__(self) -> None:
        cases = tuple(self.cases)
        if not cases:
            raise ValueError("a sample batch cannot be empty")
        if any(not isinstance(case, Case) for case in cases):
            raise TypeError("sample batch must contain Case values")
        if len({case.case_id for case in cases}) != len(cases):
            raise ValueError("sample batch Case ids must be unique")
        if self.remaining is not None and (isinstance(self.remaining, bool) or not isinstance(self.remaining, int)):
            raise TypeError("sample batch remaining count must be an integer")
        if self.remaining is not None and self.remaining < 0:
            raise ValueError("sample batch remaining count cannot be negative")
        object.__setattr__(self, "cases", cases)

    @property
    def count(self) -> int:
        return len(self.cases)


@dataclass(frozen=True)
class PairedSampleBatch:
    """Fresh root/child pairs returned by one atomic sampling operation."""

    roots: tuple[Case, ...]
    children: tuple[Case, ...]
    remaining: int | None
    attempts: tuple[PairedSamplingAttemptAudit, ...]

    def __post_init__(self) -> None:
        roots = tuple(self.roots)
        children = tuple(self.children)
        attempts = tuple(self.attempts)
        if not roots or len(roots) != len(children):
            raise ValueError("paired sample batch needs one child for every root")
        if any(not isinstance(case, Case) for case in roots + children):
            raise TypeError("paired sample batch must contain Case values")
        identifiers = tuple(case.case_id for case in roots + children)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("paired sample batch Case ids must be unique")
        for root, child in zip(roots, children):
            if child.parent_case_id == root.case_id:
                continue
            # Candidate-facing phase-blind projections deliberately remove
            # lineage. Pairing remains explicit and unambiguous through the
            # aligned roots/children tuples. Raw/controller batches may not
            # omit or alter the parent id.
            projected_pair = (
                _is_phase_blind_candidate_case(root)
                and _is_phase_blind_candidate_case(child)
                and child.parent_case_id is None
            )
            if not projected_pair:
                raise ValueError(
                    "paired sample children must preserve root order and lineage"
                )
        if self.remaining is not None and (
            isinstance(self.remaining, bool) or not isinstance(self.remaining, int)
        ):
            raise TypeError("paired sample remaining count must be an integer or None")
        if self.remaining is not None and self.remaining < 0:
            raise ValueError("paired sample remaining count cannot be negative")
        if any(not isinstance(row, PairedSamplingAttemptAudit) for row in attempts):
            raise TypeError("paired sample attempts must contain audit values")
        object.__setattr__(self, "roots", roots)
        object.__setattr__(self, "children", children)
        object.__setattr__(self, "attempts", attempts)

    @property
    def cases(self) -> tuple[Case, ...]:
        """Return roots followed by children for registration and accounting."""

        return self.roots + self.children

    @property
    def count(self) -> int:
        return len(self.cases)


def _is_phase_blind_candidate_case(case: Case) -> bool:
    prefix = "candidate-case-"
    suffix = case.case_id.removeprefix(prefix)
    return (
        case.case_id.startswith(prefix)
        and len(suffix) == 64
        and all(character in "0123456789abcdef" for character in suffix)
        and case.source is CaseSource.GENERATED
        and case.visibility is CaseVisibility.DEVELOPMENT
        and case.round_index is None
        and case.parent_case_id is None
        and not case.tags
    )


class CaseSampler(Protocol):
    """The only Case-creation interface exposed inside an Agent round."""

    def sample_cases(self, request: SampleCasesRequest) -> SampleBatch: ...

    def sample_paired_cases(
        self,
        request: PairedSampleRequest,
    ) -> PairedSampleBatch: ...

    def perturb_cases(self, request: PerturbCasesRequest) -> SampleBatch: ...


__all__ = [
    "CaseSampler",
    "PairedSampleBatch",
    "PairedSampleRequest",
    "PairedSamplingAttemptAudit",
    "PerturbCasesRequest",
    "RecoverableSamplingRejection",
    "SampleBatch",
    "SampleCasesRequest",
    "SampleSource",
    "SamplingStrategy",
]
