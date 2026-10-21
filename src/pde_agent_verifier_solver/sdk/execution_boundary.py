"""Execution-assurance contracts retained for the legacy formal macro loop.

``MacroClosedLoopRunner`` is intentionally an in-process reference
implementation.  Python object boundaries do not protect controller-private
targets from hostile Agent or candidate code.  This module makes that limit an
explicit API property and keeps formal execution fail closed until audited
out-of-process adapters are wired by a trusted composition root.

The protocols below describe the capabilities that composition root will need.
Implementing a Protocol, constructing a record, or returning a claimed digest
does *not* authorize a formal claim.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import re
from typing import Protocol, final

from .macro_loop import CandidateRef, SampleRequest
from .models import CaseRecord


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ExecutionAssurance(str, Enum):
    """Assurance requested from an Agent/controller run."""

    REFERENCE_IN_PROCESS = "reference_in_process"
    FORMAL_ISOLATED = "formal_isolated"


class FormalExecutionUnavailable(RuntimeError):
    """Raised when this package cannot establish formal execution assurance."""


@dataclass(frozen=True)
class ReferenceExecutionBoundary:
    """Truthful assurance record for the current local runner."""

    assurance: ExecutionAssurance = field(
        default=ExecutionAssurance.REFERENCE_IN_PROCESS,
        init=False,
    )
    formal_claims_allowed: bool = field(default=False, init=False)
    limitations: tuple[str, ...] = field(
        default=(
            "Agent and candidate code may share the controller process",
            "Candidate bundle digests are declarations rather than attestations",
            "Generator artifact digests and replay are not controller-verified",
        ),
        init=False,
    )


def require_execution_assurance(
    requested: ExecutionAssurance,
) -> ReferenceExecutionBoundary:
    """Resolve an assurance request without silently weakening it.

    Formal execution is deliberately unavailable in this package version.  The
    capability interfaces below are integration contracts, not security
    markers, so merely supplying objects with those methods must never make a
    run formal.  A future trusted composition root must wire and audit concrete
    adapters before this function can return a formal authorization.
    """

    if not isinstance(requested, ExecutionAssurance):
        raise TypeError("requested assurance must be an ExecutionAssurance")
    if requested is ExecutionAssurance.FORMAL_ISOLATED:
        raise FormalExecutionUnavailable(
            "formal_isolated execution is not wired: use the in-process runner "
            "only for reference/smoke evidence"
        )
    return ReferenceExecutionBoundary()


@dataclass(frozen=True)
class AgentBundleSubmission:
    """Agent-controlled bundle locator; deliberately contains no digest."""

    candidate_id: str
    bundle_uri: str

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id cannot be empty")
        if not self.bundle_uri.strip():
            raise ValueError("bundle_uri cannot be empty")


@dataclass(frozen=True)
class CandidateBundleAttestation:
    """Controller-produced record for bytes copied into trusted storage.

    This value is an audit record, not an unforgeable token.  Formal code must
    accept it only from its trusted ``ControllerOwnedBundleAttestor`` instance.
    """

    candidate_id: str
    sealed_bundle_uri: str
    bundle_sha256: str
    byte_length: int
    round_index: int
    agent_id: str
    attestor_id: str
    parent_sha256: str | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("candidate_id", self.candidate_id),
            ("sealed_bundle_uri", self.sealed_bundle_uri),
            ("agent_id", self.agent_id),
            ("attestor_id", self.attestor_id),
        ):
            if not value.strip():
                raise ValueError(f"{label} cannot be empty")
        _require_nonnegative_int(self.round_index, "round_index")
        if isinstance(self.byte_length, bool) or not isinstance(self.byte_length, int):
            raise ValueError("byte_length must be an integer")
        if self.byte_length <= 0:
            raise ValueError("byte_length must be positive")
        _require_sha256(self.bundle_sha256, "candidate bundle")
        if self.parent_sha256 is not None:
            _require_sha256(self.parent_sha256, "candidate parent")


class ControllerOwnedBundleAttestor(ABC):
    """Trusted adapter whose non-overridable path computes the bundle digest.

    An Agent supplies only a locator.  The adapter reads those bytes, hashes
    them inside the controller trust boundary, and seals the exact bytes under
    a controller-owned URI before creating the record.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "attest" in cls.__dict__:
            raise TypeError("ControllerOwnedBundleAttestor.attest cannot be overridden")

    @property
    @abstractmethod
    def attestor_id(self) -> str:
        """Stable identity of the trusted storage/attestation adapter."""

    @abstractmethod
    def _read_untrusted_bundle(self, bundle_uri: str) -> bytes:
        """Read the complete submitted bundle without executing it."""

    @abstractmethod
    def _seal_verified_bundle(self, bundle_sha256: str, payload: bytes) -> str:
        """Persist exact bytes immutably and return a controller-owned URI."""

    @final
    def attest(
        self,
        submission: AgentBundleSubmission,
        *,
        round_index: int,
        agent_id: str,
        parent_sha256: str | None,
    ) -> CandidateBundleAttestation:
        """Compute identity from bytes rather than trusting Agent metadata."""

        _require_nonnegative_int(round_index, "round_index")
        if not agent_id.strip():
            raise ValueError("agent_id cannot be empty")
        if parent_sha256 is not None:
            _require_sha256(parent_sha256, "candidate parent")
        payload = self._read_untrusted_bundle(submission.bundle_uri)
        if not isinstance(payload, bytes):
            raise TypeError("bundle reader must return bytes")
        if not payload:
            raise ValueError("candidate bundle cannot be empty")
        digest = hashlib.sha256(payload).hexdigest()
        sealed_uri = self._seal_verified_bundle(digest, payload)
        if not isinstance(sealed_uri, str) or not sealed_uri.strip():
            raise ValueError("bundle attestor returned an empty sealed URI")
        if not self.attestor_id.strip():
            raise ValueError("bundle attestor_id cannot be empty")
        return CandidateBundleAttestation(
            candidate_id=submission.candidate_id,
            sealed_bundle_uri=sealed_uri,
            bundle_sha256=digest,
            byte_length=len(payload),
            round_index=round_index,
            agent_id=agent_id,
            attestor_id=self.attestor_id,
            parent_sha256=parent_sha256,
        )


def candidate_ref_from_attestation(
    attestation: CandidateBundleAttestation,
) -> CandidateRef:
    """Create protocol identity only after controller-owned attestation."""

    return CandidateRef(
        candidate_id=attestation.candidate_id,
        bundle_sha256=attestation.bundle_sha256,
        round_index=attestation.round_index,
        agent_id=attestation.agent_id,
        parent_sha256=attestation.parent_sha256,
        bundle_uri=attestation.sealed_bundle_uri,
    )


@dataclass(frozen=True)
class IsolatedAgentRequest:
    """Serializable references allowed to cross into an Agent sandbox."""

    run_id: str
    round_index: int
    agent_id: str
    public_context_uri: str
    evaluator_endpoint: str
    incumbent_sha256: str

    def __post_init__(self) -> None:
        for label, value in (
            ("run_id", self.run_id),
            ("agent_id", self.agent_id),
            ("public_context_uri", self.public_context_uri),
            ("evaluator_endpoint", self.evaluator_endpoint),
        ):
            if not value.strip():
                raise ValueError(f"{label} cannot be empty")
        _require_nonnegative_int(self.round_index, "round_index")
        _require_sha256(self.incumbent_sha256, "incumbent")


class IsolatedAgentExecutor(Protocol):
    """Future trusted adapter for a serialized, out-of-process Agent channel."""

    @property
    def isolation_backend_id(self) -> str: ...

    def run_isolated(self, request: IsolatedAgentRequest) -> AgentBundleSubmission: ...


class IsolatedCandidateExecutor(Protocol):
    """Future trusted adapter that evaluates only attested sealed bundles."""

    @property
    def isolation_backend_id(self) -> str: ...

    def predict_serialized(
        self,
        attestation: CandidateBundleAttestation,
        case_payload: bytes,
        query_payload: bytes,
        budget: float | None = None,
    ) -> bytes: ...


@dataclass(frozen=True)
class GeneratorSnapshot:
    """Controller-owned identity for a frozen generator and replay plan."""

    snapshot_id: str
    generator_id: str
    sealed_artifact_uri: str
    artifact_sha256: str
    cohort_plan_sha256: str

    def __post_init__(self) -> None:
        for label, value in (
            ("snapshot_id", self.snapshot_id),
            ("generator_id", self.generator_id),
            ("sealed_artifact_uri", self.sealed_artifact_uri),
        ):
            if not value.strip():
                raise ValueError(f"{label} cannot be empty")
        _require_sha256(self.artifact_sha256, "Generator artifact")
        _require_sha256(self.cohort_plan_sha256, "cohort plan")


class GeneratorSnapshotReplay(Protocol):
    """Future trusted adapter for precommitted, outcome-independent cohorts."""

    @property
    def snapshot(self) -> GeneratorSnapshot: ...

    def materialize(
        self,
        requests: tuple[SampleRequest, ...],
    ) -> tuple[tuple[CaseRecord, ...], ...]: ...


@dataclass(frozen=True)
class FormalExecutionCapabilities:
    """Required adapter set; possession of this record is not authorization."""

    bundle_attestor: ControllerOwnedBundleAttestor
    agent_executor: IsolatedAgentExecutor
    candidate_executor: IsolatedCandidateExecutor
    generator_replay: GeneratorSnapshotReplay


def _require_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} sha256 must be 64 lowercase hex characters")


def _require_nonnegative_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
