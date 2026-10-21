"""Thin equation-neutral lifecycle for frozen generator guidance.

The controller owns cadence and sampling policy. A backend owns how a public
question plus scalar reward table is sealed, how the guided generator is
trained, and how a hidden answer-first draw is resolved to controller-private
``CaseRecord`` values. Agent code only receives ``DevelopmentGuidanceOracle``;
that surface has no training or hidden-truth resolver operation.

Candidate preparation/query clocks are deliberately absent here. Guidance
training and inference expose campaign-only timings through the SDK records.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import hmac
import math
import secrets
from time import perf_counter
from threading import Lock
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Sequence

from pde_agent_verifier_solver.sdk.guidance import (
    CandidateGuidanceSnapshot,
    CandidateGuidedSample,
    CandidateGuidedSampleBatch,
    FrozenGuidanceOnlineOracle,
    FrozenGuidanceSnapshot,
    GuidanceCapability,
    GuidanceTrainingBackend,
    GuidanceTrainingBudget,
    GuidanceTrainingRequest,
    GuidedSampleBatch,
    GuidedSampleRequest,
    GuidedSampleSpec,
    GuidedSamplingPurpose,
    OfficialCommonGuidedCohort,
    require_guidance_capability,
    sample_frozen_guidance,
)
from pde_agent_verifier_solver.sdk.models import Case, CaseRecord, CaseVisibility
from .candidate_case import candidate_facing_case


Entropy64 = Callable[[], int]
NonceFactory = Callable[[], str]


@dataclass(frozen=True)
class GuidanceProtocolConfig:
    """Campaign resources for optional guidance, never candidate resources.

    Frozen inference is expected to be the same cost order as base sampling and
    is not scarce or stage-limited. ``inference_wall_budget_seconds`` is one
    backend safety allocation per call, never a protocol query/sample quota.
    Only guidance training has an intentionally scarce lifecycle cadence.
    """

    training_budget: GuidanceTrainingBudget
    guidance_config_sha256: str
    reward_feedback_case_count: int
    official_round_case_count: int
    official_terminal_case_count: int
    inference_wall_budget_seconds: float
    inference_compute_capacity: str
    official_sample_settings: Mapping[
        str, str | int | float | bool | None
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.training_budget, GuidanceTrainingBudget):
            raise TypeError("guidance training_budget has the wrong type")
        _sha256(self.guidance_config_sha256, "guidance config")
        for name in (
            "reward_feedback_case_count",
            "official_round_case_count",
            "official_terminal_case_count",
        ):
            _positive_int(getattr(self, name), name)
        _positive_number(
            self.inference_wall_budget_seconds,
            "guidance inference wall budget",
        )
        _text(
            self.inference_compute_capacity,
            "guidance inference compute capacity",
        )
        frozen = GuidedSampleSpec(1, self.official_sample_settings).settings
        object.__setattr__(self, "official_sample_settings", frozen)


@dataclass(frozen=True)
class GuidanceRewardRow:
    """Public question plus scalar prior-solver reward; no answer or latent."""

    case: Case
    primary_error: float
    status: str
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.case, Case):
            raise TypeError("guidance reward case must be a Case")
        _nonnegative_number(self.primary_error, "guidance primary error")
        _text(self.status, "guidance reward status")
        copied: dict[str, float] = {}
        for name, value in self.metrics.items():
            _text(name, "guidance reward metric name")
            _nonnegative_number(value, f"guidance reward metric {name}")
            copied[name] = float(value)
        object.__setattr__(self, "metrics", MappingProxyType(copied))


@dataclass(frozen=True)
class GuidanceRewardFeedback:
    """Controller-retained input to one train-once boundary update."""

    source_round_index: int
    target_round_index: int
    target_solver_sha256: str
    base_generator_sha256: str
    reward_matcher_definition_sha256: str
    rows: tuple[GuidanceRewardRow, ...]

    def __post_init__(self) -> None:
        _nonnegative_int(self.source_round_index, "guidance source round")
        _positive_int(self.target_round_index, "guidance target round")
        if self.target_round_index != self.source_round_index + 1:
            raise ValueError("guidance target round must follow source round")
        _sha256(self.target_solver_sha256, "guidance target solver")
        _sha256(self.base_generator_sha256, "guidance base generator")
        _sha256(
            self.reward_matcher_definition_sha256,
            "guidance reward matcher definition",
        )
        rows = tuple(self.rows)
        if not rows:
            raise ValueError("guidance reward feedback cannot be empty")
        if any(not isinstance(row, GuidanceRewardRow) for row in rows):
            raise TypeError("guidance reward rows have the wrong type")
        ids = tuple(row.case.case_id for row in rows)
        if len(ids) != len(set(ids)):
            raise ValueError("guidance reward case ids must be unique")
        object.__setattr__(self, "rows", rows)


@dataclass(frozen=True)
class GuidanceRewardArtifact:
    """Backend-sealed reward table consumed by guidance training."""

    uri: str
    sha256: str

    def __post_init__(self) -> None:
        _text(self.uri, "guidance reward artifact uri")
        _sha256(self.sha256, "guidance reward artifact")


class MacroGuidanceBackend(GuidanceTrainingBackend, Protocol):
    """Generator-specific operations behind the shared lifecycle."""

    @property
    def guidance_capability(self) -> GuidanceCapability: ...

    def seal_reward_feedback(
        self,
        feedback: GuidanceRewardFeedback,
    ) -> GuidanceRewardArtifact: ...


class ControllerResolvedGuidanceOracle(FrozenGuidanceOnlineOracle, Protocol):
    """Private extension never placed in an Agent context."""

    def resolve_case_records(
        self,
        batch: GuidedSampleBatch,
    ) -> Sequence[CaseRecord]: ...

    @property
    def development_solution_methods(self) -> tuple[str, ...]: ...

    def open_development_solution_session(
        self,
        sample_id: str,
    ) -> object: ...


class DevelopmentGuidanceOracle:
    """Unlimited fresh question/solution access to one frozen guide.

    ``query_completed_solution`` is an allowlisted numerical Oracle. It can
    accept Agent-selected coordinates but never returns a ``TruthProgram`` or
    makes the opaque answer locator into a controller filesystem path.
    """

    protocol_request_limit = None
    protocol_sample_limit = None

    def __init__(
        self,
        oracle: FrozenGuidanceOnlineOracle,
        *,
        inference_wall_budget_seconds: float,
        compute_capacity: str,
        request_namespace: str,
        entropy64: Entropy64,
        nonce_factory: NonceFactory,
    ) -> None:
        controller_snapshot = getattr(oracle, "snapshot", None)
        if not isinstance(controller_snapshot, FrozenGuidanceSnapshot):
            raise TypeError("guided oracle needs a FrozenGuidanceSnapshot")
        self._oracle = oracle
        self._controller_snapshot = controller_snapshot
        self._public_snapshot = CandidateGuidanceSnapshot(
            snapshot_sha256=controller_snapshot.snapshot_sha256,
            campaign_guidance_training_wall_seconds=(
                controller_snapshot.campaign_guidance_training_wall_seconds
            ),
        )
        self._sample_id_key = secrets.token_bytes(32)
        self._wall = float(inference_wall_budget_seconds)
        self._capacity = compute_capacity
        self._namespace = request_namespace
        self._entropy64 = entropy64
        self._nonce_factory = nonce_factory
        self._lock = Lock()
        self._operation_lock = Lock()
        self._closed = False
        raw_methods = getattr(oracle, "development_solution_methods", None)
        if not isinstance(raw_methods, tuple) or not raw_methods:
            raise TypeError("guided oracle needs development_solution_methods")
        if any(
            not isinstance(method, str)
            or not method.isidentifier()
            or method.startswith("_")
            for method in raw_methods
        ) or len(raw_methods) != len(set(raw_methods)):
            raise ValueError(
                "guided completed-solution methods need a unique public allowlist"
            )
        if not callable(
            getattr(oracle, "open_development_solution_session", None)
        ):
            raise TypeError(
                "guided oracle needs open_development_solution_session"
            )
        self._solution_methods = raw_methods
        self._samples_by_id = {}
        self._call_count = 0
        self._generation_wall_seconds = 0.0
        self._solution_query_wall_seconds = 0.0

    @property
    def snapshot(self) -> CandidateGuidanceSnapshot:
        return self._public_snapshot

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count

    @property
    def campaign_guided_generation_wall_seconds(self) -> float:
        with self._lock:
            return self._generation_wall_seconds

    @property
    def available_solution_methods(self) -> tuple[str, ...]:
        return self._solution_methods

    @property
    def campaign_solution_query_wall_seconds(self) -> float:
        with self._lock:
            return self._solution_query_wall_seconds

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def sample(self, spec: GuidedSampleSpec) -> CandidateGuidedSampleBatch:
        with self._operation_lock:
            return self._sample_open(spec)

    def _sample_open(
        self,
        spec: GuidedSampleSpec,
    ) -> CandidateGuidedSampleBatch:
        if not isinstance(spec, GuidedSampleSpec):
            raise TypeError("guided development sample needs GuidedSampleSpec")
        with self._lock:
            if self._closed:
                raise RuntimeError("guided development oracle is closed")
            index = self._call_count
            self._call_count += 1
        nonce = self._nonce_factory()
        seed = self._entropy64()
        _text(nonce, "guidance controller nonce")
        _nonnegative_int(seed, "guidance controller seed")
        request = GuidedSampleRequest(
            spec=spec,
            request_id=f"{self._namespace}-fresh-{index}-{nonce}",
            purpose=GuidedSamplingPurpose.DEVELOPMENT_ONLINE,
            snapshot_sha256=self._controller_snapshot.snapshot_sha256,
            controller_seed=seed,
            controller_nonce=nonce,
            inference_wall_budget_seconds=self._wall,
            compute_capacity=self._capacity,
        )
        batch = sample_frozen_guidance(self._oracle, request)
        if any(
            sample.case.visibility is not CaseVisibility.DEVELOPMENT
            or sample.answer_access_locator is None
            for sample in batch.samples
        ):
            raise ValueError("guided development oracle returned hidden truth")

        public_rows = tuple(
            (self._opaque_sample_id(sample.sample_id), sample)
            for sample in batch.samples
        )
        with self._lock:
            raw_ids = {sample.sample_id for _, sample in public_rows}
            if raw_ids & {
                raw.sample_id for raw in self._samples_by_id.values()
            }:
                raise ValueError(
                    "guided development oracle reused sample identities"
                )
            opaque_ids = {sample_id for sample_id, _ in public_rows}
            if len(opaque_ids) != len(public_rows) or (
                opaque_ids & set(self._samples_by_id)
            ):
                raise ValueError("guided public sample identity collision")
            self._samples_by_id.update(dict(public_rows))
            self._generation_wall_seconds += (
                batch.campaign_guided_generation_wall_seconds
            )

        return CandidateGuidedSampleBatch(
            snapshot_sha256=self._public_snapshot.snapshot_sha256,
            sample_settings_commitment_sha256=spec.settings_sha256,
            samples=tuple(
                CandidateGuidedSample(
                    sample_id=opaque_id,
                    case=deepcopy(candidate_facing_case(raw.case)),
                    answer_to_case_derivation_commitment_sha256=(
                        raw.answer_to_case_derivation_sha256
                    ),
                )
                for opaque_id, raw in public_rows
            ),
            campaign_guided_generation_wall_seconds=(
                batch.campaign_guided_generation_wall_seconds
            ),
        )

    def _opaque_sample_id(self, raw_sample_id: str) -> str:
        digest = hmac.new(
            self._sample_id_key,
            raw_sample_id.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"guided-sample-{digest}"

    def query_completed_solution(
        self,
        sample_id: str,
        method: str,
        /,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Call one PDE-owned completed-solution method at Agent coordinates."""

        with self._operation_lock:
            return self._query_completed_solution_open(
                sample_id,
                method,
                *args,
                **kwargs,
            )

    def _query_completed_solution_open(
        self,
        sample_id: str,
        method: str,
        /,
        *args: object,
        **kwargs: object,
    ) -> object:
        _text(sample_id, "guided sample id")
        if method not in self._solution_methods:
            raise PermissionError(
                f"guided solution method is not allowlisted: {method!r}"
            )
        with self._lock:
            if self._closed:
                raise RuntimeError("guided development oracle is closed")
            raw_sample = self._samples_by_id.get(sample_id)
            if raw_sample is None:
                raise KeyError("guided solution sample was not issued by this oracle")
        opener = self._oracle.open_development_solution_session
        started = perf_counter()
        session = opener(raw_sample.sample_id)
        try:
            target = getattr(session, method, None)
            if not callable(target):
                raise TypeError(
                    "guided solution session lacks an allowlisted method"
                )
            return target(*args, **kwargs)
        finally:
            close = getattr(session, "close", None)
            try:
                if callable(close):
                    close()
            finally:
                elapsed = perf_counter() - started
                with self._lock:
                    self._solution_query_wall_seconds += elapsed

    def close(self) -> None:
        """Revoke this Agent-scoped oracle and release resident samples."""

        with self._operation_lock:
            with self._lock:
                if self._closed:
                    return
                self._closed = True
                self._samples_by_id.clear()
            hook = getattr(self._oracle, "close", None)
            if hook is None:
                hook = getattr(self._oracle, "release", None)
            if hook is not None:
                if not callable(hook):
                    raise TypeError("guided oracle release hook must be callable")
                hook()


@dataclass(frozen=True)
class ResolvedOfficialGuidedCohort:
    """One common hidden guided draw plus controller-only answer records."""

    public: OfficialCommonGuidedCohort
    records: tuple[CaseRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.public, OfficialCommonGuidedCohort):
            raise TypeError("official guided cohort has the wrong public type")
        records = tuple(self.records)
        samples = self.public.batch.samples
        if len(records) != len(samples):
            raise ValueError("guided records and public samples differ in size")
        for sample, record in zip(samples, records):
            if not isinstance(record, CaseRecord):
                raise TypeError("guided private records must be CaseRecord values")
            if record.case.case_id != sample.case.case_id:
                raise ValueError("guided private record changed sample identity")
            if record.case.visibility is not CaseVisibility.HIDDEN:
                raise ValueError("official guided records must stay hidden")
            if sample.answer_access_locator is not None:
                raise ValueError("official guided sample exposed answer access")
        object.__setattr__(self, "records", records)


class FrozenGuidanceBoundary:
    """Frozen guide shared by all Agents and official candidates in one round."""

    def __init__(
        self,
        *,
        backend: MacroGuidanceBackend,
        snapshot: FrozenGuidanceSnapshot,
        config: GuidanceProtocolConfig,
        entropy64: Entropy64,
        nonce_factory: NonceFactory,
    ) -> None:
        self.backend = backend
        self.snapshot = snapshot
        self.config = config
        self._entropy64 = entropy64
        self._nonce_factory = nonce_factory
        self._official: dict[str, tuple[str, ResolvedOfficialGuidedCohort]] = {}
        self._lock = Lock()

    def development_oracle(self, agent_id: str) -> DevelopmentGuidanceOracle:
        _text(agent_id, "guidance Agent id")
        oracle = self.backend.open_guidance(self.snapshot)
        _validate_open_oracle(oracle, self.snapshot)
        return DevelopmentGuidanceOracle(
            oracle,
            inference_wall_budget_seconds=(
                self.config.inference_wall_budget_seconds
            ),
            compute_capacity=self.config.inference_compute_capacity,
            request_namespace=(
                f"round-{self.snapshot.training_request.target_round_index}"
                f"-agent-{_short_digest(agent_id)}"
            ),
            entropy64=self._entropy64,
            nonce_factory=self._nonce_factory,
        )

    def official_common_cohort(
        self,
        cohort_id: str,
        *,
        sample_count: int,
        settings: Mapping[str, str | int | float | bool | None] | None = None,
    ) -> ResolvedOfficialGuidedCohort:
        """Generate at most one immutable hidden draw for ``cohort_id``."""

        _text(cohort_id, "official guided cohort id")
        spec = GuidedSampleSpec(
            sample_count,
            self.config.official_sample_settings if settings is None else settings,
        )
        spec_identity = f"{sample_count}:{spec.settings_sha256}"
        with self._lock:
            cached = self._official.get(cohort_id)
            if cached is not None:
                if cached[0] != spec_identity:
                    raise RuntimeError(
                        "official guided cohort cannot be resampled with new settings"
                    )
                return cached[1]
            oracle = self.backend.open_guidance(self.snapshot)
            _validate_open_oracle(oracle, self.snapshot)
            resolver = getattr(oracle, "resolve_case_records", None)
            if not callable(resolver):
                raise TypeError(
                    "guided backend lacks controller-private resolve_case_records"
                )
            nonce = self._nonce_factory()
            seed = self._entropy64()
            _text(nonce, "official guidance controller nonce")
            _nonnegative_int(seed, "official guidance controller seed")
            request = GuidedSampleRequest(
                spec=spec,
                request_id=f"official-{cohort_id}-{nonce}",
                purpose=GuidedSamplingPurpose.OFFICIAL_COMMON_HIDDEN,
                snapshot_sha256=self.snapshot.snapshot_sha256,
                controller_seed=seed,
                controller_nonce=nonce,
                inference_wall_budget_seconds=(
                    self.config.inference_wall_budget_seconds
                ),
                compute_capacity=self.config.inference_compute_capacity,
            )
            batch = sample_frozen_guidance(oracle, request)
            public = OfficialCommonGuidedCohort(cohort_id, batch)
            resolved = ResolvedOfficialGuidedCohort(
                public=public,
                records=tuple(resolver(batch)),
            )
            self._official[cohort_id] = (spec_identity, resolved)
            return resolved


class GuidanceLifecycleController:
    """Train exactly once at each boundary, then freeze for online use."""

    def __init__(
        self,
        *,
        config: GuidanceProtocolConfig,
        backend: MacroGuidanceBackend,
        entropy64: Entropy64 | None = None,
        nonce_factory: NonceFactory | None = None,
    ) -> None:
        if not isinstance(config, GuidanceProtocolConfig):
            raise TypeError("guidance config has the wrong type")
        capability = require_guidance_capability(
            getattr(backend, "guidance_capability", None)
        )
        self.config = config
        self.backend = backend
        self.capability = capability
        self._entropy64 = entropy64 or (lambda: secrets.randbits(63))
        self._nonce_factory = nonce_factory or (lambda: secrets.token_hex(12))
        self._attempted: set[tuple[int, int]] = set()

    def train_boundary(
        self,
        feedback: GuidanceRewardFeedback,
        *,
        controller_training_seed: int,
    ) -> FrozenGuidanceBoundary:
        if not isinstance(feedback, GuidanceRewardFeedback):
            raise TypeError("guidance feedback has the wrong type")
        if len(feedback.rows) != self.config.reward_feedback_case_count:
            raise ValueError("guidance feedback case count differs from config")
        _nonnegative_int(controller_training_seed, "guidance training seed")
        key = (feedback.source_round_index, feedback.target_round_index)
        if key in self._attempted:
            raise RuntimeError("guidance boundary training was already attempted")
        self._attempted.add(key)
        artifact = self.backend.seal_reward_feedback(feedback)
        if not isinstance(artifact, GuidanceRewardArtifact):
            raise TypeError(
                "guidance backend must return GuidanceRewardArtifact"
            )
        request = GuidanceTrainingRequest(
            source_round_index=feedback.source_round_index,
            target_round_index=feedback.target_round_index,
            base_generator_sha256=feedback.base_generator_sha256,
            target_solver_sha256=feedback.target_solver_sha256,
            reward_matcher_definition_sha256=(
                feedback.reward_matcher_definition_sha256
            ),
            reward_feedback_uri=artifact.uri,
            reward_feedback_sha256=artifact.sha256,
            guidance_config_sha256=self.config.guidance_config_sha256,
            controller_training_seed=controller_training_seed,
            training_budget=self.config.training_budget,
        )
        snapshot = self.backend.train_guidance(request)
        if not isinstance(snapshot, FrozenGuidanceSnapshot):
            raise TypeError("guidance backend returned the wrong snapshot type")
        if snapshot.training_request != request:
            raise ValueError("frozen guidance snapshot changed its training request")
        if snapshot.backend_id != self.capability.backend_id:
            raise ValueError("frozen guidance snapshot changed backend identity")
        return FrozenGuidanceBoundary(
            backend=self.backend,
            snapshot=snapshot,
            config=self.config,
            entropy64=self._entropy64,
            nonce_factory=self._nonce_factory,
        )


def _validate_open_oracle(
    oracle: object,
    snapshot: FrozenGuidanceSnapshot,
) -> None:
    capability = require_guidance_capability(
        getattr(oracle, "guidance_capability", None)
    )
    if capability.backend_id != snapshot.backend_id:
        raise ValueError("guided oracle changed backend identity")
    if getattr(oracle, "snapshot", None) != snapshot:
        raise ValueError("guided oracle changed frozen snapshot")


def _short_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _sha256(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} sha256 must be 64 lowercase hex characters")


def _text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} cannot be empty")


def _positive_int(value: object, label: str) -> None:
    _nonnegative_int(value, label)
    if value <= 0:
        raise ValueError(f"{label} must be positive")


def _nonnegative_int(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")


def _positive_number(value: object, label: str) -> None:
    _nonnegative_number(value, label)
    if float(value) <= 0.0:
        raise ValueError(f"{label} must be positive")


def _nonnegative_number(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")


__all__ = [
    "ControllerResolvedGuidanceOracle",
    "DevelopmentGuidanceOracle",
    "FrozenGuidanceBoundary",
    "GuidanceLifecycleController",
    "GuidanceProtocolConfig",
    "GuidanceRewardArtifact",
    "GuidanceRewardFeedback",
    "GuidanceRewardRow",
    "MacroGuidanceBackend",
    "ResolvedOfficialGuidedCohort",
]
